#!/usr/bin/env python3
"""Write SLM-313's fail-closed locked-evidence preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from slm_training.autoresearch.schemas import CampaignSpec
from slm_training.autoresearch.storage import CampaignStore
from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harnesses.experiments.abstract_plan_functional_evidence import (
    ARMS,
    PATHS,
    PlanEvidenceRow,
    build_campaign,
    locked_shard_ids,
    load_locked_protocol,
)
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
COMPONENT = "harness.experiments.slm313_abstract_plan_functional_evidence"
DEFAULT_ROOT = Path("outputs/autoresearch/slm313-abstract-plan-functional-evidence")
DEFAULT_JSON = ROOT / "docs/design/iter-slm313-abstract-plan-functional-evidence-20260726.json"
DEFAULT_MARKDOWN = ROOT / "docs/design/abstract-plan-functional-evidence.md"
DEFAULT_AGENTV = ROOT / "docs/design/iter-slm313-abstract-plan-functional-evidence-agentv-20260726"
LOCKED_MANIFEST = ROOT / "src/slm_training/resources/data/eval/manifests/abstract_planning_locked_v1.jsonl"
LOCAL_TRAIN_DIR = ROOT / "src/slm_training/resources/data/train/e1281_text_form_input_required_direct_natural_v1"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _checkpoint_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable(value: Any, root: Path) -> Any:
    prefix = str(root.resolve())
    if isinstance(value, str) and value.startswith(prefix):
        return "agentv-dir://" + value[len(prefix):].lstrip("/")
    if isinstance(value, list):
        return [_portable(item, root) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item, root) for key, item in value.items()}
    return value


def _rewrite_agentv_paths(root: Path) -> None:
    replacements = {
        str(root.resolve()): "agentv-dir://",
        quote(str(root.resolve()), safe=""): quote("agentv-dir://", safe=""),
        str(ROOT.resolve()): "repo://",
        quote(str(ROOT.resolve()), safe=""): quote("repo://", safe=""),
    }
    for path in (root / "agentv").rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md"}:
            text = path.read_text(encoding="utf-8")
            for source, replacement in replacements.items():
                text = text.replace(source, replacement)
            path.write_text(text, encoding="utf-8")


def _locked_records() -> dict[str, Any]:
    """Read the immutable held-out rows; the manifest remains authoritative."""
    from slm_training.data.contract import canonicalize_example_template_markers
    from slm_training.dsl.schema import ExampleRecord

    payload = json.loads(LOCKED_MANIFEST.read_text(encoding="utf-8"))
    return {
        str(row["record"]["id"]): canonicalize_example_template_markers(
            ExampleRecord.from_dict(row["record"])
        )
        for row in payload["rows"]
        if row.get("partition") == "locked_test"
    }


def _pair_map(record_ids: tuple[str, ...], protocol_sha256: str) -> tuple[dict[str, str], str]:
    """Freeze a fixed-point-free shuffle pairing before any decode."""
    if len(record_ids) < 2:
        raise ValueError("shuffle-between-example requires at least two locked records")
    ordered = sorted(
        record_ids,
        key=lambda record_id: hashlib.sha256(
            f"{protocol_sha256}:{record_id}".encode("utf-8")
        ).hexdigest(),
    )
    pairs = {record_id: ordered[(index + 1) % len(ordered)] for index, record_id in enumerate(ordered)}
    if any(record_id == paired for record_id, paired in pairs.items()):
        raise AssertionError("shuffle pairing unexpectedly has a fixed point")
    return pairs, _sha(pairs)


@contextmanager
def _temporary_config(model: Any, **overrides: Any):
    saved = {name: getattr(model.config, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(model.config, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(model.config, name, value)


class _PlanArmModel:
    """Thin arm adapter: shared evaluator and decoder, only plan content varies."""

    def __init__(
        self,
        model: Any,
        *,
        arm: str,
        records_by_prompt: dict[str, Any],
        paired_id_by_id: dict[str, str],
        records_by_id: dict[str, Any],
        seed: int,
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"unknown plan arm {arm!r}")
        self._model = model
        self.arm = arm
        self.config = model.config
        self.codec = getattr(model, "codec", None)
        self.speculative_stats = getattr(model, "speculative_stats", None)
        self._records_by_prompt = records_by_prompt
        self._paired_id_by_id = paired_id_by_id
        self._records_by_id = records_by_id
        self._seed = seed
        self.evidence: dict[str, dict[str, Any]] = {}

    def _generator(self, record_id: str):
        import torch

        digest = hashlib.sha256(f"{self._seed}:{self.arm}:{record_id}".encode("utf-8")).digest()
        return torch.Generator(device=self._model.device_name).manual_seed(int.from_bytes(digest[:8], "big"))

    def _sample_trace(self, prompt: str, record_id: str):
        with _temporary_config(self._model, abstract_plan_mode="sampled"):
            return self._model.abstract_plan_trace([prompt], generator=self._generator(record_id))

    def _vector(self, record: Any):
        import torch

        connector = self._model.abstract_plan_connector
        if connector is None:
            raise ValueError("SLM-313 execution requires the trained plan connector")
        generator = self._generator(str(record.id))
        trace = self._sample_trace(record.prompt, str(record.id))
        assert trace is not None
        vector = connector.plan_vector_from_trace(trace)
        tokens = trace.plan_tokens
        metadata: dict[str, Any] = {"plan_tokens": tuple(int(x) for x in tokens[0].tolist())}
        if self.arm == "empty":
            vector = torch.zeros_like(vector)
        elif self.arm == "random_norm_matched":
            random = torch.randn(vector.shape, generator=generator, device=vector.device, dtype=vector.dtype)
            vector = random * (vector.norm(dim=1, keepdim=True) / random.norm(dim=1, keepdim=True).clamp_min(1e-12))
        elif self.arm == "shuffle_between_examples":
            paired = self._records_by_id[self._paired_id_by_id[str(record.id)]]
            paired_trace = self._sample_trace(paired.prompt, str(paired.id))
            assert paired_trace is not None
            vector = connector.plan_vector_from_trace(paired_trace)
            tokens = paired_trace.plan_tokens
            metadata["plan_tokens"] = tuple(int(x) for x in tokens[0].tolist())
            metadata["paired_record_id"] = str(paired.id)
        elif self.arm == "within_plan_permutation":
            order = torch.randperm(tokens.size(1), generator=generator, device=tokens.device)
            trace = replace(trace, plan_tokens=tokens.index_select(1, order))
            vector = connector.plan_vector_from_trace(trace)
            metadata["plan_tokens"] = tuple(int(x) for x in trace.plan_tokens[0].tolist())
        elif self.arm == "prefix_truncation":
            keep = max(1, tokens.size(1) // 2)
            vector = connector.plan_embed(tokens[:, :keep]).mean(dim=1)
            metadata["plan_tokens"] = tuple(int(x) for x in tokens[0, :keep].tolist())
            metadata["prefix_length"] = keep
        elif self.arm == "length_matched_verbal":
            # A deliberately non-authoritative lexical control.  It only
            # matches plan length; it never reads target program structure.
            verbal_prompt = record.prompt + "\n[abstract-plan-length " + ("plan " * tokens.size(1)).strip() + "]"
            verbal_trace = self._sample_trace(verbal_prompt, str(record.id))
            assert verbal_trace is not None
            vector = connector.plan_vector_from_trace(verbal_trace)
            metadata["plan_tokens"] = tuple(int(x) for x in verbal_trace.plan_tokens[0].tolist())
        elif self.arm == "gold_semantic_plan":
            target = self._model._abstract_plan_targets([record])
            with _temporary_config(self._model, abstract_plan_mode="oracle"):
                trace = self._model.abstract_plan_trace([record.prompt], target_plan_ids=target, generator=generator)
            assert trace is not None
            vector = connector.plan_vector_from_trace(trace)
            metadata["plan_tokens"] = tuple(int(x) for x in trace.plan_tokens[0].tolist())
            metadata["oracle_diagnostic_only"] = True
        return vector, metadata

    def generate(self, prompt: str, *, max_len: int | None = None, grammar_constrained: bool | None = None, **_kwargs: Any) -> str:
        record = self._records_by_prompt[prompt]
        started = time.perf_counter()
        if self.arm == "no_plan":
            text = self._model.generate(prompt, gold=record, max_len=max_len, grammar_constrained=grammar_constrained)
            self.evidence[str(record.id)] = {"plan_tokens": None, "adapter_latency_ms": (time.perf_counter() - started) * 1000}
            return text
        vector, metadata = self._vector(record)
        self._model.denoiser.set_plan_vector(vector)
        try:
            text = self._model.generate(prompt, gold=record, max_len=max_len, grammar_constrained=grammar_constrained)
        finally:
            self._model.denoiser.set_plan_vector(None)
        self.evidence[str(record.id)] = {**metadata, "adapter_latency_ms": (time.perf_counter() - started) * 1000}
        return text


def _local_eval_config(*, root: Path, suite: str):
    from slm_training.harnesses.model_build import ModelBuildConfig

    return ModelBuildConfig(
        train_dir=LOCAL_TRAIN_DIR,
        test_dir=root,
        suite=suite,
        run_root=root / "runs",
        run_id="slm313-local-evidence",
        seed=313,
        device="cpu",
        model_name="twotower",
        output_tokenizer="choice",
        d_model=32,
        n_heads=4,
        context_layers=1,
        denoiser_layers=1,
        context_backend="scratch",
        local_files_only=True,
        optimizer_name="adamw",
        batch_size=4,
        grammar_ltr_max_tokens=8,
        grammar_ltr_primary=True,
        gen_steps=1,
        abstract_plan_mode="sampled",
        abstract_plan_connector_arm="learned",
        abstract_plan_loss_weight=0.5,
        abstract_plan_train_conditioning=True,
        decode_timeout_seconds=5.0,
        run_class="scratch_matrix",
    )


def _path_overrides(path: str) -> dict[str, Any]:
    if path == "raw":
        # The decoder is fail-closed: raw logits are retained as shadow
        # telemetry, never emitted as an unconstrained OpenUI program.
        return {"grammar_constrained": True, "grammar_ltr_repair": False, "raw_shadow": True}
    if path == "constrained":
        return {"grammar_constrained": True, "grammar_ltr_repair": False}
    if path == "repaired":
        return {"grammar_constrained": True, "grammar_ltr_repair": True}
    raise ValueError(f"unknown decode path {path!r}")


def _record_complexity(record: Any) -> tuple[int, int]:
    """Derive outcome-strata from the program, not variable plan-head width."""
    from slm_training.dsl.parser import parse

    binder_count = 0
    reference_diameter = 0

    def walk(node: object, depth: int) -> None:
        nonlocal binder_count, reference_diameter
        if not isinstance(node, dict):
            return
        props = node.get("props")
        if isinstance(props, dict):
            binder_count += sum(
                isinstance(value, str) and value.startswith(":slot_")
                for key, value in props.items()
                if key != "children"
            )
            children = props.get("children")
            if isinstance(children, list):
                for child in children:
                    walk(child, depth + 1)
        reference_diameter = max(reference_diameter, depth - 1)

    walk(parse(record.openui).root, 1)
    return binder_count, reference_diameter


def execute_local_shard(
    *,
    root: Path,
    checkpoint: Path,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    """Evaluate one bounded, pre-partitioned locked shard through shared scoring."""
    from slm_training.data.verify import verify_record
    from slm_training.harnesses.model_build import build_model
    from slm_training.harnesses.model_build.eval_runner import evaluate

    protocol = load_locked_protocol(LOCKED_MANIFEST)
    records_by_id = _locked_records()
    shard_ids = locked_shard_ids(protocol, shard_index=shard_index, shard_count=shard_count)
    records = [records_by_id[record_id] for record_id in shard_ids]
    pairs, pair_sha256 = _pair_map(protocol.record_ids, protocol.to_dict()["protocol_sha256"])
    suite = f"slm313_locked_{shard_index}_of_{shard_count}"
    suite_dir = root / "suites" / suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    records_path = suite_dir / "records.jsonl"
    records_path.write_text("".join(_canonical(record.to_dict()) + "\n" for record in records), encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps({"suites": {suite: str(records_path)}}, indent=2) + "\n", encoding="utf-8"
    )
    config = _local_eval_config(root=root, suite=suite)
    model = build_model(config, [], checkpoint=checkpoint)
    rows: list[PlanEvidenceRow] = []
    by_prompt = {record.prompt: record for record in records}
    for arm in ARMS:
        for path in PATHS:
            overrides = _path_overrides(path)
            adapter = _PlanArmModel(
                model,
                arm=arm,
                records_by_prompt=by_prompt,
                paired_id_by_id=pairs,
                records_by_id=records_by_id,
                seed=protocol.seed,
            )
            with _temporary_config(model, **{key: value for key, value in overrides.items() if hasattr(model.config, key)}):
                result = evaluate(
                    replace(config, run_id=f"slm313/{arm}/{path}/shard{shard_index}"),
                    model=adapter,
                    publish_agentv=False,
                    generation_overrides={"grammar_constrained": overrides["grammar_constrained"]},
                )
            if int(result.get("decode_timeout_count") or 0):
                raise TimeoutError(f"SLM-313 {arm}/{path} shard {shard_index} timed out; numeric evidence is invalid")
            for detail in result.get("details") or ():
                record = records_by_id[str(detail["id"])]
                prediction = str(detail.get("prediction") or "")
                gate_report = verify_record(replace(record, openui=prediction))
                gates = {item.gate.value: item.status.value for item in gate_report.results}
                arm_evidence = adapter.evidence[str(record.id)]
                plan_tokens = arm_evidence["plan_tokens"]
                prompt_tokens, _ = model.count_batch_tokens([record])
                output_tokens = len(model.tokenizer.encode(prediction))
                binder_count, reference_diameter = _record_complexity(record)
                rows.append(
                    PlanEvidenceRow(
                        record_id=str(record.id),
                        group_id=str((record.meta or {}).get("split_group_id") or record.id),
                        arm=arm,
                        path=path,
                        plan_tokens=plan_tokens,
                        total_tokens=int(prompt_tokens + output_tokens + (len(plan_tokens) if plan_tokens else 0)),
                        latency_ms=float(detail.get("latency_ms") or arm_evidence["adapter_latency_ms"]),
                        metrics={
                            "binding_aware_meaningful_v2": float(bool(detail.get("binding_aware_meaningful_v2"))),
                            "binder_reference_f1": float(detail.get("binder_reference_f1") or 0.0),
                            "canonical_ast_match": float(detail.get("exact_match") or 0.0),
                        },
                        gates=gates,
                        result_ref=str(result.get("output") or ""),
                        plan_length=len(plan_tokens) if plan_tokens else 0,
                        binder_count=binder_count,
                        reference_diameter=reference_diameter,
                        factor=str((record.meta or {}).get("source_family") or record.source),
                        pair_manifest_sha256=pair_sha256 if arm == "shuffle_between_examples" else None,
                    )
                )
    return {
        "schema": "slm313_abstract_plan_functional_shard/v1",
        "protocol_sha256": protocol.to_dict()["protocol_sha256"],
        "locked_eval_manifest_sha256": protocol.manifest_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "record_ids": list(shard_ids),
        "pair_manifest_sha256": pair_sha256,
        "rows": [row.to_dict() for row in rows],
        "promotion_eligible": False,
        "raw_diagnostic_only": True,
    }


def run_preflight(
    *,
    root: Path,
    checkpoint: Path | None,
    agentv_dir: Path,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    protocol = load_locked_protocol(
        ROOT / "src/slm_training/resources/data/eval/manifests/abstract_planning_locked_v1.jsonl"
    )
    campaign = build_campaign(protocol)
    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="Measure locked AbstractPlan function without proxy controls.",
            primary_metric="binding_aware_meaningful_v2",
            track="twotower",
            budget=campaign.budget,
            created_at=campaign.created_at,
        )
    )
    lock = store.lock_experiment_campaign(campaign)
    available = checkpoint is not None and checkpoint.is_file()
    stamp = build_version_stamp("harness.experiments", COMPONENT)
    if available:
        try:
            result = execute_local_shard(
                root=root,
                checkpoint=checkpoint,
                shard_index=shard_index,
                shard_count=shard_count,
            )
            result["verdict"] = "partial_locked_shard"
            result["reason"] = "A complete shard is measured; final disposition requires exact merge coverage across every locked shard."
            result["meaningful_parse"] = "measured_per_row"
        except Exception as exc:  # noqa: BLE001 - durable invalid-run evidence is required
            result = {
                "verdict": "invalid_local_run",
                "reason": f"Local shard did not complete: {type(exc).__name__}: {exc}",
                "promotion_eligible": False,
                "meaningful_parse": "not_measured",
                "numeric_evidence_valid": False,
            }
    else:
        result = {
            "verdict": "unavailable",
            "reason": "No trained learned-plan checkpoint was supplied; AP-023's side-channel head and AP-024's zero-gate connector are not evidence of learned plan function.",
            "promotion_eligible": False,
            "meaningful_parse": "not_measured",
        }
    report: dict[str, Any] = {
        "schema": "slm313_abstract_plan_functional_evidence/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_class": "diagnostic",
        "protocol": protocol.to_dict(),
        "campaign_manifest_sha256": lock.manifest_sha256,
        "recipe": {
            "device": "not_run",
            "backend": "not_run",
            "locked_test_used_for_selection": False,
            "human_rating_gate": "not_required",
            "checkpoint": str(checkpoint) if checkpoint else None,
            "shard_index": shard_index,
            "shard_count": shard_count,
        },
        "result": result,
        "version_stamp": stamp,
    }
    report["agentv"] = _portable(
        publish_agentv_evaluation(
            agentv_dir,
            name="slm313-abstract-plan-functional-evidence",
            claim="abstract_plan_preflight_not_ship",
            version_stamp=stamp,
            cases=[{
                "id": "unavailable-does-not-promote",
                "criteria": "No proxy plan result may be promoted.",
                "assertions": [{
                    "id": "no_proxy_promotion",
                    "actual": result["promotion_eligible"] is False,
                    "operator": "eq",
                    "expected": True,
                }],
                "result": result,
            }],
        ),
        agentv_dir,
    )
    _rewrite_agentv_paths(agentv_dir)
    return report


def _markdown(report: dict[str, Any]) -> str:
    result = report["result"]
    return (
        "# SLM-313 AbstractPlan functional evidence\n\n"
        "This locked functional-evidence record makes no model or promotion claim.\n\n"
        f"- Verdict: {result['verdict']}; promotion eligible: false.\n"
        f"- Reason: {result['reason']}\n"
        f"- Locked manifest: {report['protocol']['locked_eval_manifest_sha256']}.\n"
        f"- Meaningful-parse status: {result['meaningful_parse']}.\n"
        "- AgentEvals/AgentV records the fail-closed non-promotion assertion.\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--agentv-dir", type=Path, default=DEFAULT_AGENTV)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=226)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"claim": "no execution", "root": str(args.root)}, sort_keys=True))
        return 0
    report = run_preflight(
        root=args.root,
        checkpoint=args.checkpoint,
        agentv_dir=args.agentv_dir,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    if report["result"].get("verdict") == "partial_locked_shard":
        shard_path = args.root / "shards" / f"shard-{args.shard_index:03d}-of-{args.shard_count:03d}.json"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        shard_path.write_text(json.dumps(report["result"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["result"]["shard_artifact"] = str(shard_path)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_out.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(args.json_out), "verdict": report["result"]["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
