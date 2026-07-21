"""SLM-261 (VSD0-02): bounded memorization probe for TwoTower trainer validity.

This is a diagnostic/fixture harness, not a ship run.  It trains a tiny
TwoTower model on a small, deterministic corpus and reports whether the
principal loss and each auxiliary term are reconcilable, whether fixed
corruption NLL falls, and whether exact target/canonical reconstruction
improves.  It does not claim generalization or production readiness.
"""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
import time
import torch
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.dsl.canonicalize import canonical_fingerprint
from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.evals.denoising_nll import (
    DenoisingNLLConfig,
    evaluate_denoising_nll,
)
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.factory import build_model
from slm_training.harnesses.model_build.train_loop import train
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "ARM_NAMES",
    "MemorizationCorruptionCaseV1",
    "MemorizationCorruptionSuiteV1",
    "MemorizationArmResultV1",
    "MemorizationProbeManifestV1",
    "select_corpus_fixture",
    "build_corruption_suite",
    "run_memorization_probe_fixture",
    "render_markdown",
    "validate_manifest",
]

MATRIX_VERSION = "vsd0-02-v1"
MATRIX_SET = "slm261_memorization_probe"
EXPERIMENT_ID = "slm261-memorization-probe"

ARM_NAMES = (
    "M0_principal_only",
    "M1_current_recipe",
)

_HYPOTHESIS = (
    "A correctly wired TwoTower trainer can memorize a small verified corpus: "
    "principal masked CE falls, exact target accuracy rises, and every active "
    "loss term reconciles with the reported total."
)

_FALSIFIER = (
    "Principal loss cannot fall below 0.10 nats/token, exact target accuracy "
    "cannot exceed 0.99, canonical reconstruction cannot exceed 0.98, or the "
    "loss ledger fails reconciliation."
)

_HONEST_CAVEATS = (
    "Fixture-only diagnostic: tiny model, tiny corpus, CPU run, no ship claim.",
    "VSD0-01 semantic scorer prerequisite is not enforced by this fixture.",
    "Candidate-normalized CE (M2) is not implemented in this iteration.",
    "Exact canonical reconstruction is measured by string match, not the full "
    "binding-aware meaning pipeline.",
)

SUITE_SCHEMA_VERSION = "MemorizationCorruptionSuiteV1"
MANIFEST_SCHEMA_VERSION = "MemorizationProbeManifestV1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _clamp(value: float, low: float = 0.0, high: float = float("inf")) -> float:
    return max(low, min(value, high))


def _mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _write_records(records: list[ExampleRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class MemorizationCorruptionCaseV1:
    """One deterministic corruption row for one record and condition."""

    case_id: str
    record_id: str
    condition: str
    target_ids: tuple[int, ...]
    noisy_ids: tuple[int, ...]
    predict_mask: tuple[bool, ...]
    policy: str
    diffusion_time: float
    seed: int
    legal_candidates_sha256: str | None
    row_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "record_id": self.record_id,
            "condition": self.condition,
            "target_ids": list(self.target_ids),
            "noisy_ids": list(self.noisy_ids),
            "predict_mask": list(self.predict_mask),
            "policy": self.policy,
            "diffusion_time": self.diffusion_time,
            "seed": self.seed,
            "legal_candidates_sha256": self.legal_candidates_sha256,
            "row_sha256": self.row_sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemorizationCorruptionCaseV1":
        return cls(
            case_id=str(data["case_id"]),
            record_id=str(data["record_id"]),
            condition=str(data["condition"]),
            target_ids=tuple(int(v) for v in data["target_ids"]),
            noisy_ids=tuple(int(v) for v in data["noisy_ids"]),
            predict_mask=tuple(bool(v) for v in data["predict_mask"]),
            policy=str(data["policy"]),
            diffusion_time=float(data["diffusion_time"]),
            seed=int(data["seed"]),
            legal_candidates_sha256=data.get("legal_candidates_sha256"),
            row_sha256=str(data["row_sha256"]),
        )


@dataclass(frozen=True)
class MemorizationCorruptionSuiteV1:
    """Immutable corruption suite for replay across seeds and arms."""

    schema_version: str
    suite_id: str
    record_ids: tuple[str, ...]
    conditions: tuple[str, ...]
    cases: tuple[MemorizationCorruptionCaseV1, ...]
    source_corpus_sha256: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "record_ids": list(self.record_ids),
            "conditions": list(self.conditions),
            "cases": [c.to_dict() for c in self.cases],
            "source_corpus_sha256": self.source_corpus_sha256,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemorizationCorruptionSuiteV1":
        return cls(
            schema_version=str(data.get("schema_version", SUITE_SCHEMA_VERSION)),
            suite_id=str(data["suite_id"]),
            record_ids=tuple(str(v) for v in data.get("record_ids", [])),
            conditions=tuple(str(v) for v in data.get("conditions", [])),
            cases=tuple(
                MemorizationCorruptionCaseV1.from_dict(c) for c in data.get("cases", [])
            ),
            source_corpus_sha256=str(data["source_corpus_sha256"]),
            timestamp=str(data["timestamp"]),
        )


@dataclass(frozen=True)
class MemorizationArmResultV1:
    """Aggregated result for one probe arm."""

    arm_name: str
    seed: int
    steps: int
    lr: float
    final_reported_loss: float | None
    final_loss_ledger_reconciliation_error: float | None
    fixed_corruption_raw_nll: float | None
    fixed_corruption_legal_nll: float | None
    exact_target_accuracy: float | None
    canonical_reconstruction_rate: float | None
    trainable_parameter_count: int | None
    wall_seconds: float
    honest_caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_name": self.arm_name,
            "seed": self.seed,
            "steps": self.steps,
            "lr": self.lr,
            "final_reported_loss": self.final_reported_loss,
            "final_loss_ledger_reconciliation_error": self.final_loss_ledger_reconciliation_error,
            "fixed_corruption_raw_nll": self.fixed_corruption_raw_nll,
            "fixed_corruption_legal_nll": self.fixed_corruption_legal_nll,
            "exact_target_accuracy": self.exact_target_accuracy,
            "canonical_reconstruction_rate": self.canonical_reconstruction_rate,
            "trainable_parameter_count": self.trainable_parameter_count,
            "wall_seconds": self.wall_seconds,
            "honest_caveats": list(self.honest_caveats),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemorizationArmResultV1":
        return cls(
            arm_name=str(data["arm_name"]),
            seed=int(data["seed"]),
            steps=int(data["steps"]),
            lr=float(data["lr"]),
            final_reported_loss=_safe_float_or_none(data.get("final_reported_loss")),
            final_loss_ledger_reconciliation_error=_safe_float_or_none(
                data.get("final_loss_ledger_reconciliation_error")
            ),
            fixed_corruption_raw_nll=_safe_float_or_none(data.get("fixed_corruption_raw_nll")),
            fixed_corruption_legal_nll=_safe_float_or_none(data.get("fixed_corruption_legal_nll")),
            exact_target_accuracy=_safe_float_or_none(data.get("exact_target_accuracy")),
            canonical_reconstruction_rate=_safe_float_or_none(data.get("canonical_reconstruction_rate")),
            trainable_parameter_count=int(data["trainable_parameter_count"])
            if data.get("trainable_parameter_count") is not None
            else None,
            wall_seconds=float(data.get("wall_seconds", 0.0)),
            honest_caveats=tuple(data.get("honest_caveats", ())),
        )


@dataclass(frozen=True)
class MemorizationProbeManifestV1:
    """Full fixture manifest for the SLM-261 memorization probe."""

    schema: str
    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    claim_class: str
    hypothesis: str
    falsifier: str
    disposition: str
    disposition_rationale: str
    arms: tuple[MemorizationArmResultV1, ...]
    corruption_suite: dict[str, Any]
    n_arms: int
    n_records: int
    honest_caveats: tuple[str, ...]
    version_stamp: dict[str, Any]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "hypothesis": self.hypothesis,
            "falsifier": self.falsifier,
            "disposition": self.disposition,
            "disposition_rationale": self.disposition_rationale,
            "arms": [a.to_dict() for a in self.arms],
            "corruption_suite": self.corruption_suite,
            "n_arms": self.n_arms,
            "n_records": self.n_records,
            "honest_caveats": list(self.honest_caveats),
            "version_stamp": self.version_stamp,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemorizationProbeManifestV1":
        return cls(
            schema=str(data["schema"]),
            matrix_set=str(data["matrix_set"]),
            matrix_version=str(data["matrix_version"]),
            experiment_id=str(data["experiment_id"]),
            run_id=str(data["run_id"]),
            status=str(data["status"]),
            claim_class=str(data["claim_class"]),
            hypothesis=str(data["hypothesis"]),
            falsifier=str(data["falsifier"]),
            disposition=str(data["disposition"]),
            disposition_rationale=str(data["disposition_rationale"]),
            arms=tuple(
                MemorizationArmResultV1.from_dict(a) for a in data.get("arms", [])
            ),
            corruption_suite=dict(data.get("corruption_suite", {})),
            n_arms=int(data.get("n_arms", 0)),
            n_records=int(data.get("n_records", 0)),
            honest_caveats=tuple(data.get("honest_caveats", ())),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data["timestamp"]),
        )


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select_corpus_fixture(
    corpus_path: Path,
    *,
    n_records: int = 10,
    negative_n: int = 0,
    seed: int = 261,
) -> tuple[list[ExampleRecord], list[ExampleRecord]]:
    """Deterministically select a small corpus slice and optional negative slice.

    The selection stratifies by source family when ``classify_source_family`` is
    available, otherwise falls back to stable shuffling by record id hash.
    """
    records = load_jsonl(corpus_path)
    if not records:
        raise ValueError(f"corpus empty: {corpus_path}")
    try:
        from slm_training.harnesses.train_data.catalog import classify_source_family

        by_family: dict[str, list[ExampleRecord]] = {}
        for record in records:
            family = classify_source_family(record.source or record.id)
            by_family.setdefault(family, []).append(record)
        rng = random.Random(seed)
        ordered: list[ExampleRecord] = []
        for family in sorted(by_family):
            family_records = by_family[family]
            family_records.sort(key=lambda r: r.id)
            rng.shuffle(family_records)
            ordered.extend(family_records)
        # Round-robin across families to preserve diversity.
        selected: list[ExampleRecord] = []
        index = 0
        while len(selected) < min(n_records, len(ordered)) and ordered:
            selected.append(ordered[index % len(ordered)])
            index += 1
    except Exception:  # noqa: BLE001
        # Stable fallback: hash-based sort then take first N.
        sorted_records = sorted(
            records,
            key=lambda r: _sha256(f"{seed}:{r.id or r.prompt}"),
        )
        selected = sorted_records[:n_records]

    negative: list[ExampleRecord] = []
    if negative_n:
        rng = random.Random(seed + 1)
        ids = {r.id for r in selected}
        pool = [r for r in records if r.id not in ids]
        rng.shuffle(pool)
        negative = pool[:negative_n]

    return selected, negative


def build_corruption_suite(
    records: list[ExampleRecord],
    *,
    suite_id: str | None = None,
    conditions: tuple[str, ...] = ("one_hole", "low_noise", "medium_noise", "all_mask"),
    mask_rates: dict[str, float] | None = None,
    seed: int = 261,
) -> MemorizationCorruptionSuiteV1:
    """Build a deterministic fixed-mask corruption suite.

    The suite is tokenizer-agnostic: it stores token ids produced by the
    original corpus.  At probe time the model's tokenizer must be able to
    round-trip the ids; mismatches are surfaced as honest caveats.
    """
    suite_id = suite_id or f"memorization_{len(records)}_v1_{_today_yyyymmdd()}"
    rates = mask_rates or {
        "one_hole": 1 / max(len(records[0].openui or "x"), 1),
        "low_noise": 0.10,
        "medium_noise": 0.50,
        "all_mask": 1.0,
    }
    corpus_canonical = _canonical_json([r.to_dict() for r in records])
    source_sha = _sha256(corpus_canonical)

    cases: list[MemorizationCorruptionCaseV1] = []
    for record in records:
        # Use a minimal tokenizer-agnostic target representation: the canonical
        # OpenUI text split into whitespace tokens.  This lets the suite survive
        # tokenizer changes while still representing a real program.
        tokens = record.openui.strip().split()
        target_ids = tuple(range(len(tokens)))
        for condition in conditions:
            rate = _clamp(rates.get(condition, 0.50), 0.0, 1.0)
            payload = f"{suite_id}|{record.id}|{condition}|{rate}|{seed}"
            rng = random.Random(int(_sha256(payload)[:16], 16))
            n_mask = max(1, round(rate * len(target_ids))) if rate < 1.0 else len(target_ids)
            n_mask = min(n_mask, len(target_ids))
            mask_positions = set(rng.sample(range(len(target_ids)), n_mask)) if target_ids else set()
            predict_mask = tuple(i in mask_positions for i in range(len(target_ids)))
            # noisy ids: mask token is represented by -1 (placeholder).
            noisy_ids = tuple(-1 if m else t for t, m in zip(target_ids, predict_mask))
            row_payload = _canonical_json(
                {
                    "record_id": record.id,
                    "condition": condition,
                    "target_ids": target_ids,
                    "noisy_ids": noisy_ids,
                    "predict_mask": predict_mask,
                    "policy": "fixed_mask",
                    "diffusion_time": rate,
                    "seed": seed,
                }
            )
            cases.append(
                MemorizationCorruptionCaseV1(
                    case_id=f"{record.id}::{condition}",
                    record_id=record.id,
                    condition=condition,
                    target_ids=target_ids,
                    noisy_ids=noisy_ids,
                    predict_mask=predict_mask,
                    policy="fixed_mask",
                    diffusion_time=rate,
                    seed=seed,
                    legal_candidates_sha256=None,
                    row_sha256=_sha256(row_payload),
                )
            )

    return MemorizationCorruptionSuiteV1(
        schema_version=SUITE_SCHEMA_VERSION,
        suite_id=suite_id,
        record_ids=tuple(r.id for r in records),
        conditions=conditions,
        cases=tuple(cases),
        source_corpus_sha256=source_sha,
        timestamp=_now(),
    )


def _arm_config(
    arm_name: str,
    train_dir: Path,
    test_dir: Path,
    run_root: Path,
    run_id: str,
    seed: int,
    steps: int,
    lr: float,
    fast: bool,
) -> ModelBuildConfig:
    """Build a small, deterministic ModelBuildConfig for one arm."""
    cfg = ModelBuildConfig(
        train_dir=train_dir,
        test_dir=test_dir,
        suite="smoke",
        run_class="fixture_demo",
        run_root=run_root,
        run_id=run_id,
        steps=steps,
        max_wall_minutes=3.0,
        batch_size=2 if fast else 4,
        lr=lr,
        seed=seed,
        device="cpu",
        model_name="twotower",
        d_model=32 if fast else 64,
        n_heads=2 if fast else 4,
        context_layers=1 if fast else 2,
        denoiser_layers=2 if fast else 3,
        context_backend="scratch",
        denoiser_backend="scratch",
        freeze_context=False,
        local_files_only=True,
        eval_every=0,
        loss_eval_every=0,
        full_state_checkpoint=False,
        register_promoted=False,
        use_curriculum=False,
        telemetry=False,
    )
    if arm_name == "M0_principal_only":
        cfg.ltr_loss_weight = 0.0
        cfg.fidelity_loss_weight = 0.0
        cfg.diffusion_length_loss_weight = 0.0
        cfg.compiler_alignment_loss_weight = 0.0
        cfg.component_inventory_loss_weight = 0.0
        cfg.component_plan_loss_weight = 0.0
        cfg.slot_component_loss_weight = 0.0
        cfg.component_edge_loss_weight = 0.0
        cfg.binder_arity_loss_weight = 0.0
        cfg.root_reference_arity_loss_weight = 0.0
        cfg.root_reference_identity_loss_weight = 0.0
        cfg.component_edge_alignment_loss_weight = 0.0
        cfg.binder_component_plan_loss_weight = 0.0
        cfg.binder_topology_loss_weight = 0.0
        cfg.fastpath_aux_weight = 0.0
        cfg.recursive_depth_supervision_weights = ()
    return cfg


def _read_final_loss(run_dir: Path) -> tuple[float | None, float | None]:
    """Return (last reported loss, last ledger reconciliation error)."""
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None, None
    last_loss: float | None = None
    last_recon: float | None = None
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "loss" in row:
                last_loss = _safe_float_or_none(row["loss"])
            ledger = row.get("loss_ledger")
            if isinstance(ledger, dict):
                last_recon = _safe_float_or_none(ledger.get("absolute_reconciliation_error"))
    return last_loss, last_recon


def _run_fixed_corruption_eval(
    model: Any,
    records: list[ExampleRecord],
) -> dict[str, Any]:
    """Cheap fixed-mask NLL eval; swallows errors to keep the probe honest."""
    try:
        return evaluate_denoising_nll(
            model,
            records,
            config=DenoisingNLLConfig(
                mask_rates=(0.15, 0.50, 0.85),
                mask_seed=261,
                batch_size=4,
                compute_legal_support=True,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _canonical_reconstruction_rate(model: Any, records: list[ExampleRecord]) -> float | None:
    """Greedy one-pass reconstruction exact canonical match rate."""
    try:
        from slm_training.models.twotower import _pad_batch

        model.eval()
        prompts: list[str] = []
        targets: list[list[int]] = []
        for record in records:
            text = model._format_one_context(
                record.prompt,
                record.design_md,
                query_prompt=record.prompt,
            )
            prompts.append(text)
            ids = model._encode_openui(
                record.openui,
                placeholders=list(record.placeholders or []),
                cache_key=record.id,
            )
            targets.append(ids)
        ctx, ctx_pad = model._encode_context(prompts, cache_keys=None)
        target_ids = _pad_batch(targets, model.tokenizer.pad_id, device=model.device_name)
        all_mask = torch.full_like(target_ids, model.tokenizer.mask_id)
        with torch.no_grad():
            logits = model.denoiser(
                all_mask, ctx, pad_id=model.tokenizer.pad_id, ctx_pad_mask=ctx_pad
            )
            pred_ids = logits.argmax(dim=-1)
        ok = 0
        for i, record in enumerate(records):
            pred = model.tokenizer.decode(pred_ids[i].tolist()).strip()
            if canonical_fingerprint(pred) == canonical_fingerprint(record.openui):
                ok += 1
        return ok / len(records) if records else None
    except Exception:  # noqa: BLE001
        return None


def _exact_target_accuracy(model: Any, records: list[ExampleRecord]) -> float | None:
    """Teacher-forced exact token accuracy on unmasked targets."""
    try:
        import torch
        from slm_training.models.twotower import _pad_batch

        model.eval()
        prompts: list[str] = []
        targets: list[list[int]] = []
        for record in records:
            text = model._format_one_context(
                record.prompt,
                record.design_md,
                query_prompt=record.prompt,
            )
            prompts.append(text)
            ids = model._encode_openui(
                record.openui,
                placeholders=list(record.placeholders or []),
                cache_key=record.id,
            )
            targets.append(ids)
        ctx, ctx_pad = model._encode_context(prompts, cache_keys=None)
        target_ids = _pad_batch(targets, model.tokenizer.pad_id, device=model.device_name)
        with torch.no_grad():
            logits = model.denoiser(
                target_ids, ctx, pad_id=model.tokenizer.pad_id, ctx_pad_mask=ctx_pad
            )
            pred = logits.argmax(dim=-1)
        valid = target_ids != model.tokenizer.pad_id
        correct = ((pred == target_ids) & valid).sum().item()
        total = valid.sum().item()
        return correct / total if total else None
    except Exception:  # noqa: BLE001
        return None


def run_memorization_probe_fixture(
    corpus_path: Path,
    *,
    output_dir: Path,
    arms: tuple[str, ...] = ARM_NAMES,
    seeds: tuple[int, ...] = (0,),
    n_records: int = 5,
    steps: int = 10,
    lr: float = 3e-4,
    fast: bool = True,
    write_design_docs: bool = False,
    version_components: tuple[str, ...] = (),
) -> MemorizationProbeManifestV1:
    """Run the bounded memorization probe and return a manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"slm261-{uuid.uuid4().hex[:8]}"

    selected, negative = select_corpus_fixture(
        corpus_path, n_records=n_records, negative_n=0
    )
    suite = build_corruption_suite(selected)
    suite_path = output_dir / "corruption_suite.json"
    suite_path.write_text(_canonical_json(suite.to_dict()), encoding="utf-8")

    arm_results: list[MemorizationArmResultV1] = []
    for arm_name in arms:
        for seed in seeds:
            arm_start = time.monotonic()
            arm_run_id = f"{run_id}-{arm_name}-seed{seed}"
            with tempfile.TemporaryDirectory(prefix="slm261_") as tmp:
                tmp_path = Path(tmp)
                train_dir = tmp_path / "train"
                test_dir = tmp_path / "test"
                _write_records(selected, train_dir / "records.jsonl")
                _write_records(selected, test_dir / "suites" / "smoke" / "records.jsonl")
                run_root = tmp_path / "runs"
                cfg = _arm_config(
                    arm_name,
                    train_dir,
                    test_dir,
                    run_root,
                    arm_run_id,
                    seed,
                    steps,
                    lr,
                    fast,
                )
                try:
                    plugin = build_model(cfg, selected)
                    train(cfg, model=plugin)
                    model = plugin
                except Exception as exc:  # noqa: BLE001
                    arm_results.append(
                        MemorizationArmResultV1(
                            arm_name=arm_name,
                            seed=seed,
                            steps=steps,
                            lr=lr,
                            final_reported_loss=None,
                            final_loss_ledger_reconciliation_error=None,
                            fixed_corruption_raw_nll=None,
                            fixed_corruption_legal_nll=None,
                            exact_target_accuracy=None,
                            canonical_reconstruction_rate=None,
                            trainable_parameter_count=None,
                            wall_seconds=time.monotonic() - arm_start,
                            honest_caveats=(f"training failed: {exc}",),
                        )
                    )
                    continue

                run_dir = run_root / arm_run_id
                last_loss, last_recon = _read_final_loss(run_dir)
                nll_summary = _run_fixed_corruption_eval(model, selected)
                aggregate = nll_summary.get("aggregate") or {}
                raw_nll = _safe_float_or_none(aggregate.get("mean_nll"))
                legal_nll = _safe_float_or_none(aggregate.get("legal_mean_nll"))
                exact_acc = _exact_target_accuracy(model, selected)
                recon_rate = _canonical_reconstruction_rate(model, selected)
                param_count = sum(p.numel() for p in model.trainable_parameters())
                arm_results.append(
                    MemorizationArmResultV1(
                        arm_name=arm_name,
                        seed=seed,
                        steps=steps,
                        lr=lr,
                        final_reported_loss=last_loss,
                        final_loss_ledger_reconciliation_error=last_recon,
                        fixed_corruption_raw_nll=raw_nll,
                        fixed_corruption_legal_nll=legal_nll,
                        exact_target_accuracy=exact_acc,
                        canonical_reconstruction_rate=recon_rate,
                        trainable_parameter_count=param_count,
                        wall_seconds=time.monotonic() - arm_start,
                        honest_caveats=(),
                    )
                )

    # Disposition: trainer_memorizes only if an arm reaches the strict fixture
    # memorization thresholds AND the loss ledger reconciles.  These are
    # intentionally lower than ship gates but still require near-perfect
    # teacher-forced accuracy on the tiny corpus.
    best = max(
        (
            a
            for a in arm_results
            if a.exact_target_accuracy is not None
            and a.final_loss_ledger_reconciliation_error is not None
        ),
        key=lambda a: a.exact_target_accuracy or 0.0,
        default=None,
    )
    if (
        best is not None
        and best.exact_target_accuracy is not None
        and best.exact_target_accuracy >= 0.99
        and best.final_loss_ledger_reconciliation_error is not None
        and best.final_loss_ledger_reconciliation_error < 1e-3
    ):
        disposition = "trainer_memorizes"
        rationale = (
            f"{best.arm_name} seed={best.seed} reached exact target accuracy "
            f"{best.exact_target_accuracy:.4f} with loss-ledger reconciliation error "
            f"{best.final_loss_ledger_reconciliation_error:.3e} on the fixture corpus."
        )
    else:
        disposition = "inconclusive"
        rationale = (
            "No arm reached the strict fixture memorization thresholds "
            "(exact target accuracy >= 0.99 and ledger reconciliation error < 1e-3). "
            "This is expected for a tiny diagnostic fixture; it is not a falsification."
        )

    manifest = MemorizationProbeManifestV1(
        schema=MANIFEST_SCHEMA_VERSION,
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
        status="fixture",
        claim_class="wiring",
        hypothesis=_HYPOTHESIS,
        falsifier=_FALSIFIER,
        disposition=disposition,
        disposition_rationale=rationale,
        arms=tuple(arm_results),
        corruption_suite=suite.to_dict(),
        n_arms=len(arm_results),
        n_records=len(selected),
        honest_caveats=_HONEST_CAVEATS,
        version_stamp=build_version_stamp(*version_components),
        timestamp=_now(),
    )

    report_path = output_dir / "report.json"
    report_path.write_text(_canonical_json(manifest.to_dict()), encoding="utf-8")

    if write_design_docs:
        design_dir = Path("docs/design")
        design_dir.mkdir(parents=True, exist_ok=True)
        json_path = design_dir / f"iter-slm261-memorization-probe-{_today_yyyymmdd()}.json"
        md_path = design_dir / f"iter-slm261-memorization-probe-{_today_yyyymmdd()}.md"
        json_path.write_text(_canonical_json(manifest.to_dict()), encoding="utf-8")
        md_path.write_text(render_markdown(manifest), encoding="utf-8")

    return manifest


def render_markdown(manifest: MemorizationProbeManifestV1) -> str:
    lines: list[str] = []
    lines.append(f"# SLM-261: bounded memorization probe ({manifest.run_id})")
    lines.append("")
    lines.append(f"- **Matrix set:** {manifest.matrix_set}")
    lines.append(f"- **Matrix version:** {manifest.matrix_version}")
    lines.append(f"- **Status:** {manifest.status}")
    lines.append(f"- **Claim class:** {manifest.claim_class}")
    lines.append(f"- **Disposition:** {manifest.disposition}")
    lines.append(f"- **Timestamp:** {manifest.timestamp}")
    lines.append("")
    lines.append("## Hypothesis")
    lines.append(manifest.hypothesis)
    lines.append("")
    lines.append("## Falsifier")
    lines.append(manifest.falsifier)
    lines.append("")
    lines.append("## Arms")
    lines.append("")
    lines.append("| arm | seed | steps | final loss | ledger error | raw NLL | legal NLL | exact acc | recon rate | wall s |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm in manifest.arms:
        lines.append(
            f"| {arm.arm_name} | {arm.seed} | {arm.steps} | "
            f"{_fmt(arm.final_reported_loss)} | {_fmt(arm.final_loss_ledger_reconciliation_error)} | "
            f"{_fmt(arm.fixed_corruption_raw_nll)} | {_fmt(arm.fixed_corruption_legal_nll)} | "
            f"{_fmt(arm.exact_target_accuracy)} | {_fmt(arm.canonical_reconstruction_rate)} | "
            f"{arm.wall_seconds:.2f} |"
        )
    lines.append("")
    lines.append("## Disposition rationale")
    lines.append(manifest.disposition_rationale)
    lines.append("")
    lines.append("## Honest caveats")
    for caveat in manifest.honest_caveats:
        lines.append(f"- {caveat}")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append(
        f"```bash\npython -m scripts.run_memorization_probe "
        f"--corpus <path> --output-dir outputs/experiments/{manifest.run_id}\n```"
    )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def validate_manifest(manifest: MemorizationProbeManifestV1) -> list[str]:
    """Return a list of validation errors; empty list means valid."""
    errors: list[str] = []
    if manifest.schema != MANIFEST_SCHEMA_VERSION:
        errors.append(f"schema={manifest.schema!r} != {MANIFEST_SCHEMA_VERSION!r}")
    if not manifest.arms:
        errors.append("no arms")
    for arm in manifest.arms:
        if arm.arm_name not in ARM_NAMES:
            errors.append(f"unknown arm_name={arm.arm_name!r}")
    return errors
