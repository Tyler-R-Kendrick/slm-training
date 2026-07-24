#!/usr/bin/env python3
"""Run SLM-230's bounded recurrence observability and anytime audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

import torch

from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harnesses.experiments.slm230_recurrence_observability import (
    REPORT_SCHEMA,
    ExitMode,
    RecurrenceExitPolicyV1,
    RecurrenceObservabilityV1,
    classify_recurrence,
    distribution_metrics,
    histogram_matched_control,
    select_exit_depth,
    stable_hash,
    validate_report,
)
from slm_training.harnesses.model_build.data import load_suite_records
from slm_training.harnesses.model_build.eval_runner import (
    _is_meaningful_program,
    _reward_for_prediction,
    structural_similarity,
)
from slm_training.models.recursive_denoiser import (
    RecursiveDepthDiagnosticsV1,
    SharedRecursiveDenoiserTower,
)
from slm_training.models.rng_contract import seed_training_corruption
from slm_training.models.twotower import TwoTowerModel
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "docs/design/iter-slm230-recurrence-observability-20260724.json"
DEFAULT_MARKDOWN = (
    ROOT / "docs/design/iter-slm230-recurrence-observability-20260724.md"
)
DEFAULT_AGENTV = (
    ROOT / "docs/design/iter-slm230-recurrence-observability-agentv-20260724"
)
DEFAULT_CHECKPOINT = (
    ROOT / "outputs/runs/slm230_bounded_recursive_r4_r2/checkpoints/last.pt"
)
DEFAULT_TEST_DIR = (
    ROOT
    / "src/slm_training/resources/data/eval"
    / "e763_symbol_only_eval_r2_20260722"
)
MAX_RECORDS_PER_SPLIT = 2
EVAL_MAX_LEN = 32


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        parts = resolved.parts
        if "outputs" in parts:
            return str(Path(*parts[parts.index("outputs") :]))
        return f"external://{resolved.name}"


def _portable(value: Any, output_dir: Path) -> Any:
    prefix = str(output_dir.resolve())
    if isinstance(value, str) and value.startswith(prefix):
        return "agentv-dir://" + value[len(prefix) :].lstrip("/")
    if isinstance(value, list):
        return [_portable(item, output_dir) for item in value]
    if isinstance(value, dict):
        return {key: _portable(item, output_dir) for key, item in value.items()}
    return value


def _rewrite_agentv_paths(output_dir: Path) -> None:
    replacements = {
        str(output_dir.resolve()): "agentv-dir://",
        quote(str(output_dir.resolve()), safe=""): quote("agentv-dir://", safe=""),
    }
    for path in (output_dir / "agentv").rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        for source, replacement in replacements.items():
            text = text.replace(source, replacement)
        path.write_text(text, encoding="utf-8")


def _git_diff_hash() -> str | None:
    completed = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode or not completed.stdout:
        return None
    return hashlib.sha256(completed.stdout.encode()).hexdigest()


def _manifest(records: list[ExampleRecord], suite: str) -> dict[str, Any]:
    rows = [
        {
            "id": record.id,
            "source_sha256": stable_hash(record.to_dict()),
            "split": record.split,
        }
        for record in records
    ]
    return {
        "schema": "RecurrenceSplitManifestV1",
        "suite": suite,
        "record_ids": [row["id"] for row in rows],
        "records": rows,
        "manifest_hash": stable_hash(rows),
    }


@contextmanager
def _capture_record(
    model: TwoTowerModel,
) -> Iterator[dict[str, Any]]:
    tower = model.denoiser
    if not isinstance(tower, SharedRecursiveDenoiserTower):
        raise TypeError("SLM-230 requires a shared-recursive checkpoint")
    original_mask = model._mask_targets
    original_outputs = tower.recursive_outputs
    captured: dict[str, Any] = {}

    def capture_mask(target_ids: torch.Tensor) -> Any:
        noisy, mask, weights = original_mask(target_ids)
        captured["targets"] = target_ids.detach().clone()
        captured["noisy"] = noisy.detach().clone()
        captured["mask"] = mask.detach().clone()
        return noisy, mask, weights

    def capture_outputs(
        noisy_ids: torch.Tensor,
        context: torch.Tensor,
        pad_id: int,
        ctx_pad_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        features = tower._runtime_symbol_features
        captured["runtime_symbol_hash"] = (
            None
            if features is None
            else hashlib.sha256(
                features.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
        )
        kwargs.update(
            diagnostics=True,
            diagnostic_targets=captured["targets"],
            diagnostic_mask=captured["mask"],
        )
        output = original_outputs(
            noisy_ids,
            context,
            pad_id,
            ctx_pad_mask,
            **kwargs,
        )
        captured["output"] = output
        return output

    model._mask_targets = capture_mask  # type: ignore[method-assign]
    tower.recursive_outputs = capture_outputs  # type: ignore[method-assign]
    try:
        yield captured
    finally:
        model._mask_targets = original_mask  # type: ignore[method-assign]
        tower.recursive_outputs = original_outputs  # type: ignore[method-assign]


@contextmanager
def _evaluation_depth(model: TwoTowerModel, depth: int) -> Iterator[None]:
    tower = model.denoiser
    if not isinstance(tower, SharedRecursiveDenoiserTower):
        raise TypeError("evaluation depth requires SharedRecursiveDenoiserTower")
    trained_depth = int(tower.recursive_steps)
    if depth < 1 or depth > trained_depth:
        raise ValueError("SLM-230 does not authorize test-R extrapolation")
    config_depth = int(model.config.recursive_steps)
    tower.recursive_steps = depth
    model.config.recursive_steps = depth
    try:
        yield
    finally:
        tower.recursive_steps = trained_depth
        model.config.recursive_steps = config_depth


def _decode_at_depth(
    model: TwoTowerModel,
    record: ExampleRecord,
    *,
    depth: int,
) -> dict[str, Any]:
    tower = model.denoiser
    assert isinstance(tower, SharedRecursiveDenoiserTower)
    original_outputs = tower.recursive_outputs
    forward_calls = 0

    def counted_outputs(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal forward_calls
        forward_calls += 1
        return original_outputs(*args, **kwargs)

    tower.recursive_outputs = counted_outputs  # type: ignore[method-assign]
    try:
        with _evaluation_depth(model, depth):
            prediction, stats = model.generate_with_stats(
                record.prompt,
                gold=record,
                max_len=min(EVAL_MAX_LEN, int(model.config.max_target_len)),
                grammar_constrained=False,
                design_md=record.design_md,
            )
    finally:
        tower.recursive_outputs = original_outputs  # type: ignore[method-assign]
    if forward_calls < 1:
        raise RuntimeError("free-running depth evaluation used no denoiser forward")
    meaningful, _, serialized = _is_meaningful_program(prediction, gold=record)
    scored = serialized or prediction
    try:
        reward = float(_reward_for_prediction(scored, record))
    except Exception:  # noqa: BLE001 - persisted as unavailable, never inferred
        reward = None
    return {
        "prediction": prediction,
        "grammar_valid": bool(meaningful),
        "structural_similarity": float(
            structural_similarity(scored, record.openui)
        ),
        "reward_score": reward,
        "latency_ms": float(stats.total_ms),
        "forwards": forward_calls,
    }


def _record_observations(
    model: TwoTowerModel,
    record: ExampleRecord,
    *,
    split: str,
    suite: str,
    checkpoint_sha256: str,
    config_hash: str,
    tokenizer_hash: str,
    decode_hash: str,
    corruption_seed: int,
) -> list[dict[str, Any]]:
    tower = model.denoiser
    assert isinstance(tower, SharedRecursiveDenoiserTower)
    trained_depth = int(tower.recursive_steps)
    seed_training_corruption(corruption_seed, model, override_seed=corruption_seed)
    with _capture_record(model) as captured:
        with torch.no_grad():
            model.training_loss([record])
    model.eval()
    output = captured["output"]
    diagnostics = output["diagnostics"]
    depth_logits = output["depth_logits"]
    if not (
        len(diagnostics) == len(depth_logits) == trained_depth
        and all(isinstance(row, RecursiveDepthDiagnosticsV1) for row in diagnostics)
    ):
        raise RuntimeError("recursive output did not expose every trained depth")
    mask = captured["mask"][0]
    candidate_set_hash = stable_hash(
        {
            "scope": "unavailable_exact_compiler_legal_set",
            "record_id": record.id,
            "reason": "surface fixture has no frozen DecisionEvent candidate artifact",
        }
    )
    generated = {
        depth: _decode_at_depth(model, record, depth=depth)
        for depth in range(1, trained_depth + 1)
    }
    rows = []
    previous_logits: torch.Tensor | None = None
    block_layers = len(tower._f_layers) + len(tower._g_layers)
    for index, (logits, diagnostic) in enumerate(
        zip(depth_logits, diagnostics, strict=True),
        start=1,
    ):
        metrics = distribution_metrics(
            logits[0],
            mask=mask,
            previous_logits=previous_logits,
            legal_candidate_ids=None,
        )
        decoded = generated[index]
        numeric = [
            value
            for value in (
                metrics["entropy"],
                metrics["full_kl"],
                metrics["full_js"],
                metrics["logit_cosine"],
                metrics["logit_l2"],
                float(diagnostic.cross_entropy[0]),
                float(diagnostic.accuracy[0]),
                float(diagnostic.y_update_norm[0]),
                None
                if diagnostic.z_update_norm is None
                else float(diagnostic.z_update_norm[0]),
            )
            if value is not None
        ]
        row = RecurrenceObservabilityV1(
            checkpoint_sha256=checkpoint_sha256,
            model_config_hash=config_hash,
            tokenizer_hash=tokenizer_hash,
            decode_config_hash=decode_hash,
            trained_recurrence_depth=trained_depth,
            evaluated_depth=index,
            test_time_extrapolation=False,
            record_id=record.id,
            split=split,
            suite=suite,
            request_fingerprint=stable_hash(
                {
                    "prompt": record.prompt,
                    "design_md": record.design_md,
                    "target_kind": record.target_kind,
                }
            ),
            candidate_set_hash=candidate_set_hash,
            candidate_set_scope="unavailable_exact_compiler_legal_set",
            target_count=int(diagnostic.target_count[0]),
            full_entropy=float(metrics["entropy"]),
            full_top_ids=tuple(metrics["top_ids"]),
            full_top1_stable=metrics["top1_stable"],
            full_kl_from_previous=metrics["full_kl"],
            full_js_from_previous=metrics["full_js"],
            logit_cosine_from_previous=metrics["logit_cosine"],
            logit_l2_from_previous=metrics["logit_l2"],
            legal_kl_from_previous=None,
            legal_js_from_previous=None,
            target_cross_entropy=float(diagnostic.cross_entropy[0]),
            target_accuracy=float(diagnostic.accuracy[0]),
            y_residual_norm=float(diagnostic.y_update_norm[0]),
            z_residual_norm=(
                None
                if diagnostic.z_update_norm is None
                else float(diagnostic.z_update_norm[0])
            ),
            grammar_valid=bool(decoded["grammar_valid"]),
            decoded_output_sha256=hashlib.sha256(
                decoded["prediction"].encode()
            ).hexdigest(),
            decoded_output=decoded["prediction"],
            structural_similarity=float(decoded["structural_similarity"]),
            reward_score=decoded["reward_score"],
            latency_ms=float(decoded["latency_ms"]),
            forwards=int(decoded["forwards"]),
            block_evaluations=int(decoded["forwards"]) * index * block_layers,
            numerical_status=(
                "finite"
                if all(math.isfinite(float(value)) for value in numeric)
                else "nonfinite"
            ),
            good_bad_status="censored_no_exact_decision_labels",
            runtime_symbol_hash=captured["runtime_symbol_hash"],
        ).to_dict()
        rows.append(row)
        previous_logits = logits[0]
    if int(tower.recursive_steps) != trained_depth:
        raise RuntimeError("evaluation-only depth override leaked into model defaults")
    return rows


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else 0.0


def _policy_result(
    rows: list[dict[str, Any]],
    record_ids: list[str],
    policy: RecurrenceExitPolicyV1,
    *,
    forced_depths: dict[str, int] | None = None,
) -> dict[str, Any]:
    selected = []
    for record_id in record_ids:
        record_rows = [row for row in rows if row["record_id"] == record_id]
        depth = (
            forced_depths[record_id]
            if forced_depths is not None
            else int(select_exit_depth(record_rows, policy))
        )
        selected.append(
            next(row for row in record_rows if int(row["evaluated_depth"]) == depth)
        )
    return {
        "policy": policy.mode.value,
        "selected_depths": {
            row["record_id"]: int(row["evaluated_depth"]) for row in selected
        },
        "mean_depth": _mean(selected, "evaluated_depth"),
        "mean_block_evaluations": _mean(selected, "block_evaluations"),
        "mean_latency_ms": _mean(selected, "latency_ms"),
        "parse_rate": statistics.fmean(
            1.0 if row["grammar_valid"] else 0.0 for row in selected
        ),
        "mean_structural_similarity": _mean(selected, "structural_similarity"),
        "mean_reward_score": _mean(selected, "reward_score"),
    }


def _policies(
    calibration_rows: list[dict[str, Any]],
    calibration_manifest_hash: str,
    trained_depth: int,
) -> list[RecurrenceExitPolicyV1]:
    kl_values = [
        float(row["full_kl_from_previous"])
        for row in calibration_rows
        if row["full_kl_from_previous"] is not None
    ]
    threshold = statistics.median(kl_values) if kl_values else 0.0
    return [
        RecurrenceExitPolicyV1(
            mode=ExitMode.FIXED,
            minimum_depth=1,
            maximum_depth=depth,
            fallback_depth=depth,
        )
        for depth in range(1, trained_depth + 1)
    ] + [
        RecurrenceExitPolicyV1(
            mode=ExitMode.KL_PLATEAU,
            minimum_depth=1,
            maximum_depth=trained_depth,
            kl_threshold=threshold,
            fallback_depth=trained_depth,
            allowed_signals=("full_kl_from_previous",),
            calibration_split_hash=calibration_manifest_hash,
        ),
        RecurrenceExitPolicyV1(
            mode=ExitMode.TOPK_STABLE,
            minimum_depth=1,
            maximum_depth=trained_depth,
            topk_stability_k=5,
            fallback_depth=trained_depth,
            allowed_signals=("topk_stable",),
            calibration_split_hash=calibration_manifest_hash,
        ),
        RecurrenceExitPolicyV1(
            mode=ExitMode.ORACLE,
            minimum_depth=1,
            maximum_depth=trained_depth,
            fallback_depth=trained_depth,
            allowed_signals=("reward_score", "structural_similarity"),
        ),
    ]


def _run(
    *,
    checkpoint: Path,
    test_dir: Path,
    agentv_dir: Path,
    allow_dirty: bool,
    pinned_version_stamp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint_sha = _sha256_file(checkpoint)
    model = TwoTowerModel.from_checkpoint(checkpoint, device="cpu")
    model.eval()
    tower = model.denoiser
    if not isinstance(tower, SharedRecursiveDenoiserTower):
        raise TypeError("checkpoint denoiser is not shared_recursive")
    trained_depth = int(tower.recursive_steps)
    if trained_depth < 2:
        raise ValueError("SLM-230 requires a non-vacuous trained recurrence depth")
    config_payload = asdict(model.config)
    config_hash = stable_hash(config_payload)
    tokenizer_hash = stable_hash(model.tokenizer.token_to_id)
    decode_hash = stable_hash(
        {
            key: value
            for key, value in config_payload.items()
            if "decode" in key
            or key
            in {
                "grammar_constrained",
                "gen_steps",
                "parallel_unmask",
                "compiler_decode_mode",
            }
        }
    )
    calibration_records = load_suite_records(test_dir, "smoke")[
        :MAX_RECORDS_PER_SPLIT
    ]
    heldout_records = load_suite_records(test_dir, "held_out")[
        :MAX_RECORDS_PER_SPLIT
    ]
    calibration_manifest = _manifest(calibration_records, "smoke")
    heldout_manifest = _manifest(heldout_records, "held_out")
    if set(calibration_manifest["record_ids"]) & set(heldout_manifest["record_ids"]):
        raise RuntimeError("calibration and heldout records overlap")

    observations = []
    for split_index, (split, suite, records) in enumerate(
        (
            ("calibration", "smoke", calibration_records),
            ("heldout", "held_out", heldout_records),
        )
    ):
        for record_index, record in enumerate(records):
            observations.extend(
                _record_observations(
                    model,
                    record,
                    split=split,
                    suite=suite,
                    checkpoint_sha256=checkpoint_sha,
                    config_hash=config_hash,
                    tokenizer_hash=tokenizer_hash,
                    decode_hash=decode_hash,
                    corruption_seed=230_000 + split_index * 1_000 + record_index,
                )
            )

    calibration_rows = [
        row for row in observations if row["split"] == "calibration"
    ]
    heldout_rows = [row for row in observations if row["split"] == "heldout"]
    policies = _policies(
        calibration_rows,
        calibration_manifest["manifest_hash"],
        trained_depth,
    )
    for policy in policies:
        policy.validate()
    heldout_ids = heldout_manifest["record_ids"]
    policy_results = {
        f"{policy.mode.value}:{policy.maximum_depth}": _policy_result(
            heldout_rows,
            heldout_ids,
            policy,
        )
        for policy in policies
    }
    kl_policy = next(policy for policy in policies if policy.mode is ExitMode.KL_PLATEAU)
    kl_result = policy_results[f"{kl_policy.mode.value}:{kl_policy.maximum_depth}"]
    fixed_max = policy_results[f"fixed:{trained_depth}"]
    average_fixed_depth = max(
        1, min(trained_depth, int(round(kl_result["mean_depth"])))
    )
    fixed_average = policy_results[f"fixed:{average_fixed_depth}"]
    selected_depths = [
        int(kl_result["selected_depths"][record_id]) for record_id in heldout_ids
    ]
    shuffled_depths = histogram_matched_control(
        selected_depths,
        record_ids=heldout_ids,
    )
    shuffled = _policy_result(
        heldout_rows,
        heldout_ids,
        kl_policy,
        forced_depths=shuffled_depths,
    )
    early_exit_qualified = bool(
        kl_result["mean_depth"] < trained_depth
        and fixed_max["parse_rate"] > 0.0
        and kl_result["parse_rate"] >= fixed_max["parse_rate"]
        and kl_result["mean_reward_score"] >= fixed_average["mean_reward_score"]
        and kl_result["mean_reward_score"] > shuffled["mean_reward_score"]
    )
    verdict = classify_recurrence(
        observations,
        heldout_record_ids=heldout_ids,
        early_exit_qualified=early_exit_qualified,
    )
    stamp = pinned_version_stamp or build_version_stamp(
        "harness.experiments.slm230_recurrence_observability",
        "model.recursive_denoiser",
        "evals.scoring",
    )
    dirty = bool(stamp.get("code_dirty"))
    if dirty and not allow_dirty:
        raise RuntimeError("SLM-230 evidence requires a clean implementation commit")
    agentv = publish_agentv_evaluation(
        agentv_dir,
        name="slm230-recurrence-observability",
        claim="bounded_recurrence_observability_not_ship",
        version_stamp=stamp,
        cases=[
            {
                "id": "depth-completeness",
                "criteria": "Every calibration and heldout record has one finite row for every trained recurrence depth.",
                "pass": len(observations)
                == (len(calibration_records) + len(heldout_records)) * trained_depth
                and all(row["numerical_status"] == "finite" for row in observations),
                "result": {
                    "rows": len(observations),
                    "trained_depth": trained_depth,
                    "records": len(calibration_records) + len(heldout_records),
                },
            },
            {
                "id": "split-and-policy-boundary",
                "criteria": "Adaptive policies are frozen on disjoint calibration records and consume inference-available signals only.",
                "pass": not (
                    set(calibration_manifest["record_ids"])
                    & set(heldout_manifest["record_ids"])
                )
                and all(
                    not (
                        set(policy.allowed_signals)
                        & {
                            "reward_score",
                            "structural_similarity",
                            "target_cross_entropy",
                            "d_good",
                        }
                    )
                    for policy in policies
                    if policy.mode is not ExitMode.ORACLE
                ),
                "result": {
                    "calibration": calibration_manifest,
                    "heldout": heldout_manifest,
                },
            },
            {
                "id": "default-depth-restored",
                "criteria": "Evaluation-only depth overrides restore the checkpoint's trained recurrence depth.",
                "pass": int(tower.recursive_steps) == trained_depth
                and int(model.config.recursive_steps) == trained_depth,
                "result": {
                    "tower_depth": int(tower.recursive_steps),
                    "config_depth": int(model.config.recursive_steps),
                },
            },
            {
                "id": "honest-exit-verdict",
                "criteria": "An early-exit mechanism is not qualified unless it survives fixed-average and histogram-matched controls with non-vacuous quality.",
                "pass": early_exit_qualified
                or verdict.value
                in {
                    "stagnant",
                    "oscillatory",
                    "weight_sharing_only",
                    "unstable",
                    "inconclusive",
                },
                "result": {
                    "qualified": early_exit_qualified,
                    "verdict": verdict.value,
                    "kl_policy": kl_result,
                    "fixed_average": fixed_average,
                    "shuffled": shuffled,
                },
            },
        ],
    )
    _rewrite_agentv_paths(agentv_dir)
    agentv = _portable(agentv, agentv_dir)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "matrix_set": "slm230-recurrence-observability",
        "matrix_version": "rsc1-01-v1",
        "issue": "SLM-230",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "bounded_checkpoint_diagnostic",
        "claim_class": "scratch_checkpoint_not_ship",
        "verdict": verdict.value,
        "checkpoint": {
            "path": _portable_path(checkpoint),
            "sha256": checkpoint_sha,
            "trained_recurrence_depth": trained_depth,
            "sync": "local_only_explicit_no_sync",
            "promotable": False,
        },
        "recipe": {
            "device": "cpu",
            "backend": "scratch",
            "train_steps": 4,
            "train_suite_n": 97,
            "calibration_suite": "smoke",
            "calibration_n": len(calibration_records),
            "heldout_suite": "held_out",
            "heldout_n": len(heldout_records),
            "evaluated_depths": list(range(1, trained_depth + 1)),
            "test_time_extrapolation": False,
            "decode_policy": "unconstrained_fixed_32_token_canvas",
            "honesty_mode": "scratch_checkpoint_not_ship",
            "max_wall_minutes": 3.0,
        },
        "split_manifests": {
            "calibration": calibration_manifest,
            "heldout": heldout_manifest,
        },
        "candidate_scope": {
            "exact_state_legal_candidates_available": False,
            "good_bad_labels_available": False,
            "handling": "censored, never coerced to negatives",
            "resolving_evidence": (
                "a provenance-bound DecisionEvent artifact with frozen legal, "
                "good, and bad candidate partitions"
            ),
        },
        "observations": observations,
        "exit_policies": [policy.to_dict() for policy in policies],
        "policy_results": policy_results,
        "matched_controls": {
            "fixed_max": fixed_max,
            "fixed_average": fixed_average,
            "histogram_matched_time_shuffle": shuffled,
            "oracle": policy_results[f"oracle:{trained_depth}"],
        },
        "early_exit_qualified": early_exit_qualified,
        "semantic_floor": {
            "path": "docs/design/semantic-floor-gate-v1.json",
            "hash": "7839ef6b6e37710d487757da9170017d7b76a9d12ca1fb314bdb0fa23a4dd83d",
            "verdict": "inconclusive",
            "semantic_claim_authorized": False,
        },
        "prior_recurrence_evidence": {
            "path": "docs/design/iter-slm282-recurrence-health-20260723.json",
            "disposition": "recursive_core_negative",
            "scope": "two-record fixture only",
        },
        "agentv": agentv,
        "evidence_gate": {
            "code_dirty": dirty,
            "diff_hash": _git_diff_hash() if dirty else None,
            "allow_dirty": allow_dirty,
        },
        "version_stamp": stamp,
        "production_default_changed": False,
        "training_default_changed": False,
        "ship_gate_claim": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    hash_payload = copy.deepcopy(report)
    hash_payload.pop("generated_at", None)
    hash_payload.pop("elapsed_seconds", None)
    stamp_payload = dict(hash_payload["version_stamp"])
    stamp_payload.pop("stamped_at", None)
    hash_payload["version_stamp"] = stamp_payload
    hash_payload["agentv"]["summary"].pop("durationMs", None)
    report["report_hash"] = stable_hash(hash_payload)
    errors = validate_report(report)
    if errors:
        raise RuntimeError("invalid SLM-230 report: " + "; ".join(errors))
    return report


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SLM-230 recurrence observability and anytime audit",
        "",
        f"Verdict: **{report['verdict']}**",
        "",
        f"Report hash: `{report['report_hash']}`",
        "",
        "This is a bounded scratch-checkpoint diagnostic, not a ship, semantic, "
        "training-default, or serving-default claim.",
        "",
        "## Recipe and evidence boundary",
        "",
        f"- Checkpoint: `{report['checkpoint']['path']}` (`{report['checkpoint']['sha256']}`)",
        f"- Train recipe: CPU scratch, 4 optimizer steps, 97 fixture-source records, trained R={report['checkpoint']['trained_recurrence_depth']}",
        f"- Calibration/final: smoke n={report['recipe']['calibration_n']} / held_out n={report['recipe']['heldout_n']}",
        f"- AgentV: `{report['agentv']['summary']}`",
        f"- Clean evidence: `{not report['evidence_gate']['code_dirty']}`",
        "- No test-R extrapolation; no checkpoint sync or promotion.",
        "",
        "## Depth-wise heldout observations",
        "",
        "| record | depth | CE | accuracy | KL prev | JS prev | top1 stable | parse | structure | reward | block evals |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in report["observations"]:
        if row["split"] != "heldout":
            continue
        lines.append(
            "| {record_id} | {evaluated_depth} | {ce} | {acc} | {kl} | {js} | "
            "{stable} | {parse} | {structure} | {reward} | {blocks} |".format(
                record_id=row["record_id"],
                evaluated_depth=row["evaluated_depth"],
                ce=_fmt(row["target_cross_entropy"]),
                acc=_fmt(row["target_accuracy"]),
                kl=_fmt(row["full_kl_from_previous"]),
                js=_fmt(row["full_js_from_previous"]),
                stable=_fmt(row["full_top1_stable"]),
                parse=row["grammar_valid"],
                structure=_fmt(row["structural_similarity"]),
                reward=_fmt(row["reward_score"]),
                blocks=row["block_evaluations"],
            )
        )
    lines.extend(
        [
            "",
            "## Anytime policies and matched controls",
            "",
            "| policy | mean depth | block evals | latency ms | parse | structure | reward |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, result in report["policy_results"].items():
        lines.append(
            f"| `{key}` | {_fmt(result['mean_depth'])} | "
            f"{_fmt(result['mean_block_evaluations'])} | "
            f"{_fmt(result['mean_latency_ms'])} | {_fmt(result['parse_rate'])} | "
            f"{_fmt(result['mean_structural_similarity'])} | "
            f"{_fmt(result['mean_reward_score'])} |"
        )
    shuffled = report["matched_controls"]["histogram_matched_time_shuffle"]
    lines.extend(
        [
            f"| `kl_histogram_time_shuffle` | {_fmt(shuffled['mean_depth'])} | "
            f"{_fmt(shuffled['mean_block_evaluations'])} | "
            f"{_fmt(shuffled['mean_latency_ms'])} | {_fmt(shuffled['parse_rate'])} | "
            f"{_fmt(shuffled['mean_structural_similarity'])} | "
            f"{_fmt(shuffled['mean_reward_score'])} |",
            "",
            f"Early exit qualified: **{report['early_exit_qualified']}**.",
            "",
            "The policy cannot qualify from zero/invalid quality, and must beat both "
            "the closest fixed-average-depth control and the identical-histogram "
            "time-shuffled control.",
            "",
            "## Exact-state and semantic limits",
            "",
            "This bounded source has no provenance-bound DecisionEvent candidate "
            "artifact. Legal-renormalized KL, D_good/D_bad, and protected exact-state "
            "claims are therefore censored rather than filled with full-vocabulary "
            "surrogates. The SemanticFloorGateV1 verdict remains inconclusive, so "
            "strict semantic improvement is not authorized.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "timeout 170s env PYTHONPATH=src .venv/bin/python -m "
            "scripts.run_slm230_recurrence_observability "
            f"--checkpoint {report['checkpoint']['path']} "
            "--test-dir "
            "src/slm_training/resources/data/eval/e763_symbol_only_eval_r2_20260722 "
            "--check",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--agentv-dir", type=Path, default=DEFAULT_AGENTV)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    committed = (
        json.loads(args.json_out.read_text(encoding="utf-8"))
        if args.check
        else None
    )
    report = _run(
        checkpoint=args.checkpoint,
        test_dir=args.test_dir,
        agentv_dir=args.agentv_dir,
        allow_dirty=args.allow_dirty or args.check,
        pinned_version_stamp=(
            committed["version_stamp"] if committed is not None else None
        ),
    )
    markdown = _markdown(report)
    if args.check:
        assert committed is not None
        if committed["report_hash"] != report["report_hash"]:
            raise SystemExit(
                "SLM-230 report hash mismatch: "
                f"{committed['report_hash']} != {report['report_hash']}"
            )
        if args.markdown_out.read_text(encoding="utf-8") != _markdown(committed):
            raise SystemExit("SLM-230 Markdown is inconsistent with committed JSON")
    else:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args.markdown_out.write_text(markdown, encoding="utf-8")
    print(f"{REPORT_SCHEMA} {report['report_hash']} {report['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
