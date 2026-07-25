#!/usr/bin/env python3
"""Run SLM-232's bounded latent-state representation and causal-use audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from urllib.parse import quote

import torch

from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harnesses.experiments.slm230_recurrence_observability import (
    stable_hash,
)
from slm_training.harnesses.experiments.slm231_recurrence_dynamics import linear_cka
from slm_training.harnesses.experiments.slm232_latent_state_use import (
    INITIAL_ABLATIONS,
    NON_APPLICABLE_ABLATIONS,
    PATH_ABLATIONS,
    LatentAblationResultV1,
    LatentStateUseGateV1,
    RecursiveStateAblationV1,
    apply_initial_ablation,
    classify_latent_state_use,
    compose_z0,
    representation_summary,
    within_group_permutation,
)
from slm_training.harnesses.model_build.data import load_suite_records
from slm_training.harnesses.model_build.eval_runner import (
    _is_meaningful_program,
    _reward_for_prediction,
    structural_similarity,
)
from slm_training.models.recursive_denoiser import SharedRecursiveDenoiserTower
from slm_training.models.rng_contract import seed_training_corruption
from slm_training.models.twotower import TwoTowerModel
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "docs/design/iter-slm232-latent-state-use-20260724.json"
DEFAULT_MARKDOWN = ROOT / "docs/design/iter-slm232-latent-state-use-20260724.md"
DEFAULT_AGENTV = ROOT / "docs/design/iter-slm232-latent-state-use-agentv-20260724"
DEFAULT_CHECKPOINT = (
    ROOT / "outputs/runs/slm230_bounded_recursive_r4_r2/checkpoints/last.pt"
)
DEFAULT_TEST_DIR = (
    ROOT / "src/slm_training/resources/data/eval/e763_symbol_only_eval_r2_20260722"
)
DEFAULT_SLM230 = ROOT / "docs/design/iter-slm230-recurrence-observability-20260724.json"
DEFAULT_SLM231 = ROOT / "docs/design/iter-slm231-recurrence-dynamics-20260724.json"
COMPONENT = "harness.experiments.slm232_latent_state_use"
SEED = 232
PRIMARY_RECORDS = {
    "smoke": ("smoke_hero_01", "smoke_button_01"),
    "held_out": ("held_out_form_01", "held_out_dual_card_01"),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        if "outputs" in resolved.parts:
            return str(Path(*resolved.parts[resolved.parts.index("outputs") :]))
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


def _scientific_hash(report: dict[str, Any]) -> str:
    payload = json.loads(json.dumps(report))
    for key in ("report_hash", "generated_at", "elapsed_seconds"):
        payload.pop(key, None)
    stamp = dict(payload["version_stamp"])
    stamp.pop("stamped_at", None)
    payload["version_stamp"] = stamp
    summary = dict(payload["agentv"]["summary"])
    summary.pop("durationMs", None)
    payload["agentv"]["summary"] = summary
    for cell in payload.get("free_running", {}).values():
        cell.pop("latency_seconds", None)
        for row in cell.get("rows", []):
            row.pop("latency_seconds", None)
            stats = row.get("decode_stats", {})
            for key in list(stats):
                if key.endswith("_ms") or key == "total_ms":
                    stats.pop(key, None)
    return stable_hash(payload)


def _select(records: list[Any], ids: tuple[str, ...]) -> list[Any]:
    by_id = {record.id: record for record in records}
    missing = [record_id for record_id in ids if record_id not in by_id]
    if missing:
        raise ValueError(f"missing preregistered records: {missing}")
    return [by_id[record_id] for record_id in ids]


def _manifest(records: list[Any], *, suite: str) -> dict[str, Any]:
    rows = [
        {
            "id": record.id,
            "split": record.split,
            "source_sha256": stable_hash(record.to_dict()),
        }
        for record in records
    ]
    return {
        "schema": "LatentStateGroupManifestV1",
        "suite": suite,
        "records": rows,
        "manifest_sha256": stable_hash(rows),
    }


def _capture_batch(model: TwoTowerModel, records: list[Any]) -> dict[str, Any]:
    tower = model.denoiser
    if not isinstance(tower, SharedRecursiveDenoiserTower):
        raise TypeError("SLM-232 requires a shared-recursive checkpoint")
    original_mask = model._mask_targets
    original_outputs = tower.recursive_outputs
    captured: dict[str, Any] = {}

    def capture_mask(targets: torch.Tensor) -> Any:
        noisy, mask, weights = original_mask(targets)
        captured.update(
            targets=targets.detach().clone(),
            noisy=noisy.detach().clone(),
            mask=mask.detach().clone(),
        )
        return noisy, mask, weights

    def capture_outputs(
        noisy: torch.Tensor,
        context: torch.Tensor,
        pad_id: int,
        ctx_pad_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        captured.update(
            context=context.detach().clone(),
            ctx_pad_mask=None
            if ctx_pad_mask is None
            else ctx_pad_mask.detach().clone(),
            pad_id=pad_id,
            components=tower.initial_transition_components(
                noisy, context, pad_id, ctx_pad_mask
            ),
        )
        return original_outputs(noisy, context, pad_id, ctx_pad_mask, **kwargs)

    model._mask_targets = capture_mask  # type: ignore[method-assign]
    tower.recursive_outputs = capture_outputs  # type: ignore[method-assign]
    try:
        torch.manual_seed(SEED)
        seed_training_corruption(SEED, model, override_seed=SEED)
        with torch.no_grad():
            model.training_loss(records)
    finally:
        model._mask_targets = original_mask  # type: ignore[method-assign]
        tower.recursive_outputs = original_outputs  # type: ignore[method-assign]
        model.eval()
    required = {"targets", "noisy", "mask", "context", "pad_id", "components"}
    if required.difference(captured):
        raise RuntimeError(f"incomplete state capture: {required.difference(captured)}")
    return captured


def _manual_outputs(
    tower: SharedRecursiveDenoiserTower,
    captured: dict[str, Any],
    *,
    mode: str,
    calibration_mean: torch.Tensor,
    permutation: torch.Tensor,
    pair_manifest_sha256: str,
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor | None]], float | None]:
    components = captured["components"]
    y = components["y"]
    assert isinstance(y, torch.Tensor)
    baseline_z = compose_z0(components)
    path = mode if mode in PATH_ABLATIONS else "none"
    if mode == "y_only_repeated_control":
        z = None
    else:
        matched = (
            baseline_z[permutation]
            if mode == "swap_z_matched" and baseline_z is not None
            else None
        )
        z = apply_initial_ablation(
            components,
            RecursiveStateAblationV1(mode),
            mean_z0=calibration_mean,
            matched_z0=matched,
            permutation=permutation,
            pair_manifest_sha256=pair_manifest_sha256,
        )
    rows: list[dict[str, torch.Tensor | None]] = []
    for _ in range(tower.recursive_steps):
        step = tower.transition_step(
            y,
            z,
            captured["context"],
            components["self_pad_mask"],
            captured["ctx_pad_mask"],
            runtime_symbol_features=components["runtime_symbol_features"],
            state_path_ablation=path,
        )
        rows.append(step)
        y = step["y"]
        z = step["z"]
        assert isinstance(y, torch.Tensor)
    logits = rows[-1]["logits"]
    assert isinstance(logits, torch.Tensor)
    norm_delta = None
    if baseline_z is not None and z is not None and mode == "random_norm_matched":
        norm_delta = float(
            (
                torch.linalg.vector_norm(
                    apply_initial_ablation(
                        components,
                        RecursiveStateAblationV1(mode),
                    ),
                    dim=-1,
                )
                - torch.linalg.vector_norm(baseline_z, dim=-1)
            )
            .abs()
            .max()
        )
    return logits, rows, norm_delta


def _teacher_metrics(
    logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    captured: dict[str, Any],
    record_ids: tuple[str, ...],
) -> dict[str, Any]:
    mask = captured["mask"].bool()
    targets = captured["targets"]
    base_logp = torch.log_softmax(baseline_logits.double(), dim=-1)
    logp = torch.log_softmax(logits.double(), dim=-1)
    base_p = base_logp.exp()
    token_kl = (base_p * (base_logp - logp)).sum(dim=-1)
    rows = []
    for index, record_id in enumerate(record_ids):
        selected = mask[index]
        count = int(selected.sum())
        if count:
            predicted = logits[index].argmax(dim=-1)
            rows.append(
                {
                    "record_id": record_id,
                    "support_tokens": count,
                    "full_vocab_kl": float(token_kl[index][selected].mean()),
                    "top1_change_rate": float(
                        (
                            predicted[selected]
                            != baseline_logits[index].argmax(dim=-1)[selected]
                        )
                        .double()
                        .mean()
                    ),
                    "target_accuracy": float(
                        (predicted[selected] == targets[index][selected])
                        .double()
                        .mean()
                    ),
                    "target_cross_entropy": float(
                        torch.nn.functional.cross_entropy(
                            logits[index][selected].double(),
                            targets[index][selected],
                        )
                    ),
                }
            )
    return {
        "per_record": rows,
        "full_vocab_kl": statistics.mean(row["full_vocab_kl"] for row in rows),
        "top1_change_rate": statistics.mean(
            row["top1_change_rate"] for row in rows
        ),
        "target_accuracy": statistics.mean(row["target_accuracy"] for row in rows),
        "target_cross_entropy": statistics.mean(
            row["target_cross_entropy"] for row in rows
        ),
        "worst_record_full_vocab_kl": max(
            row["full_vocab_kl"] for row in rows
        ),
        "empirical_cvar50_full_vocab_kl": max(
            row["full_vocab_kl"] for row in rows
        ),
    }


def _representation(
    tower: SharedRecursiveDenoiserTower,
    captures: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    by_depth: list[dict[str, Any]] = []
    trajectories: list[list[dict[str, torch.Tensor | None]]] = []
    initial_rows: list[torch.Tensor] = []
    initial_residual_rows: list[torch.Tensor] = []
    y_rows: list[torch.Tensor] = []
    position_rows: list[torch.Tensor] = []
    component_norms: list[dict[str, float]] = []
    common_sequence_length = min(
        int(captured["components"]["y"].shape[1]) for captured in captures
    )
    for captured in captures:
        components = captured["components"]
        y = components["y"]
        z = compose_z0(components)
        assert isinstance(y, torch.Tensor)
        assert isinstance(z, torch.Tensor)
        for index in range(y.shape[0]):
            bounded_z = z[index, :common_sequence_length]
            initial_rows.append(bounded_z.mean(dim=0))
            position_rows.append(bounded_z)
            y_rows.append(y[index, :common_sequence_length].mean(dim=0))
            learned = components["z_latent_component"]
            assert isinstance(learned, torch.Tensor)
            initial_residual_rows.append(
                learned[index, :common_sequence_length].mean(dim=0)
            )
            component_norms.append(
                {
                    key: float(torch.linalg.vector_norm(components[key][index]))
                    for key in (
                        "z_latent_component",
                        "z_context_component",
                        "z_position_component",
                    )
                    if isinstance(components[key], torch.Tensor)
                }
            )
        steps = []
        current_y, current_z = y, z
        for _ in range(tower.recursive_steps):
            step = tower.transition_step(
                current_y,
                current_z,
                captured["context"],
                components["self_pad_mask"],
                captured["ctx_pad_mask"],
                runtime_symbol_features=components["runtime_symbol_features"],
            )
            steps.append(step)
            current_y, current_z = step["y"], step["z"]
            assert isinstance(current_y, torch.Tensor)
            assert isinstance(current_z, torch.Tensor)
        trajectories.append(steps)

    initial_matrix = torch.stack(initial_rows)
    residual_matrix = torch.stack(initial_residual_rows)
    by_depth.append(
        {
            "depth": 0,
            "z": representation_summary(initial_matrix),
            "z_after_context_and_position_removal": representation_summary(
                residual_matrix
            ),
            "z_y_linear_cka": linear_cka(initial_matrix, torch.stack(y_rows)),
        }
    )
    previous = initial_matrix
    for depth in range(1, tower.recursive_steps + 1):
        z_rows, y_depth_rows, update_rows = [], [], []
        for steps in trajectories:
            step = steps[depth - 1]
            z, y, update = step["z"], step["y"], step["z_update"]
            assert isinstance(z, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            assert isinstance(update, torch.Tensor)
            z_rows.extend(
                z[index, :common_sequence_length].mean(dim=0)
                for index in range(z.shape[0])
            )
            y_depth_rows.extend(
                y[index, :common_sequence_length].mean(dim=0)
                for index in range(y.shape[0])
            )
            update_rows.extend(
                update[index, :common_sequence_length].mean(dim=0)
                for index in range(update.shape[0])
            )
        matrix = torch.stack(z_rows)
        update_matrix = torch.stack(update_rows)
        by_depth.append(
            {
                "depth": depth,
                "z": representation_summary(matrix),
                "z_y_linear_cka": linear_cka(matrix, torch.stack(y_depth_rows)),
                "z_update_linear_cka": linear_cka(matrix, update_matrix),
                "normalized_change": float(
                    torch.linalg.vector_norm(matrix - previous)
                    / torch.linalg.vector_norm(previous).clamp_min(1e-12)
                ),
                "cosine_previous": float(
                    torch.nn.functional.cosine_similarity(
                        matrix.flatten().unsqueeze(0),
                        previous.flatten().unsqueeze(0),
                    )
                ),
            }
        )
        previous = matrix
    context_total = sum(row["z_context_component"] for row in component_norms)
    all_total = sum(sum(row.values()) for row in component_norms)
    position_matrix = torch.stack(position_rows).double()
    per_position_variance = position_matrix.var(dim=0, unbiased=False).mean(dim=1)
    position_vectors = position_matrix.permute(1, 0, 2).reshape(
        common_sequence_length, -1
    )
    position_vectors = position_vectors - position_vectors.mean(dim=1, keepdim=True)
    position_correlation = torch.nn.functional.cosine_similarity(
        position_vectors[:, None, :],
        position_vectors[None, :, :],
        dim=-1,
    )
    off_diagonal = position_correlation[
        ~torch.eye(common_sequence_length, dtype=torch.bool)
    ]
    context_component = [
        captured["components"]["z_context_component"][
            :, :common_sequence_length
        ]
        for captured in captures
    ]
    assert all(isinstance(value, torch.Tensor) for value in context_component)
    context_matrix = torch.cat(context_component)
    context_swapped = context_matrix.flip(0)
    context_delta = context_swapped - context_matrix
    return {
        "projection": (
            f"per_record_mean_over_fixed_first_{common_sequence_length}_positions"
        ),
        "independent_record_support": len(initial_rows),
        "common_sequence_length": common_sequence_length,
        "by_depth": by_depth,
        "initial_component_norms": component_norms,
        "initial_context_norm_fraction": context_total / all_total,
        "position_correlation": {
            "status": "descriptive_only",
            "per_position_variance": [
                float(value) for value in per_position_variance
            ],
            "mean_off_diagonal_correlation": float(off_diagonal.mean()),
            "independent_group_n": len(initial_rows),
            "reason": "fixed common positions; token rows are not independent groups",
        },
        "context_sensitivity": {
            "pair_type": "ctx_proj_init_shuffle",
            "group_n": len(initial_rows),
            "normalized_l2": float(
                torch.linalg.vector_norm(context_delta)
                / torch.linalg.vector_norm(context_matrix).clamp_min(1e-12)
            ),
            "same_canvas_different_context": "not_constructible_from_retained_records",
            "same_prompt_altered_legal_canvas": "unavailable_legal_canvas_not_verified",
            "runtime_symbol_alpha_pair": "unavailable_runtime_symbol_hash_null",
        },
        "decision_kind_role_rank": {
            "status": "unavailable",
            "reason": "checkpoint has no provenance-compatible DecisionEvent groups",
        },
    }


@contextmanager
def _generation_ablation(
    tower: SharedRecursiveDenoiserTower,
    *,
    mode: str,
    calibration_mean: torch.Tensor,
) -> Iterator[None]:
    original_initial = tower.initial_transition_state
    original_step = tower.transition_step

    def initial(
        noisy: torch.Tensor,
        context: torch.Tensor,
        pad_id: int,
        ctx_pad_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        components = tower.initial_transition_components(
            noisy, context, pad_id, ctx_pad_mask
        )
        y = components["y"]
        assert isinstance(y, torch.Tensor)
        z = (
            None
            if mode == "y_only_repeated_control"
            else apply_initial_ablation(
                components,
                RecursiveStateAblationV1(
                    mode if mode in INITIAL_ABLATIONS else "none"
                ),
                mean_z0=calibration_mean,
            )
        )
        return {
            "y": y,
            "z": z,
            "self_pad_mask": components["self_pad_mask"],
            "runtime_symbol_features": components["runtime_symbol_features"],
        }

    def step(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor | None]:
        if mode in PATH_ABLATIONS:
            kwargs["state_path_ablation"] = mode
        return original_step(*args, **kwargs)

    tower.initial_transition_state = initial  # type: ignore[method-assign]
    tower.transition_step = step  # type: ignore[method-assign]
    try:
        yield
    finally:
        tower.initial_transition_state = original_initial  # type: ignore[method-assign]
        tower.transition_step = original_step  # type: ignore[method-assign]


def _free_running(
    model: TwoTowerModel,
    records: list[Any],
    *,
    modes: list[str],
    calibration_mean: torch.Tensor,
) -> dict[str, Any]:
    tower = model.denoiser
    assert isinstance(tower, SharedRecursiveDenoiserTower)
    result: dict[str, Any] = {}
    for mode in modes:
        rows = []
        with _generation_ablation(
            tower, mode=mode, calibration_mean=calibration_mean
        ):
            for record in records:
                started = time.perf_counter()
                torch.manual_seed(SEED)
                prediction, stats = model.generate_with_stats(
                    record.prompt,
                    gold=record,
                    max_len=min(24, int(model.config.max_target_len)),
                    grammar_constrained=False,
                    design_md=record.design_md,
                )
                meaningful, _, serialized = _is_meaningful_program(
                    prediction, gold=record
                )
                scored = serialized or prediction
                try:
                    reward = float(_reward_for_prediction(scored, record))
                except Exception:  # noqa: BLE001 - unavailable, never inferred
                    reward = None
                rows.append(
                    {
                        "record_id": record.id,
                        "prediction_sha256": hashlib.sha256(
                            prediction.encode()
                        ).hexdigest(),
                        "meaningful_parse": bool(meaningful),
                        "structural_similarity": structural_similarity(
                            scored, record.openui
                        ),
                        "reward": reward,
                        "latency_seconds": time.perf_counter() - started,
                        "block_evaluations": tower.recursive_steps * 2,
                        "decode_stats": stats.as_dict(),
                    }
                )
        result[mode] = {
            "support": len(rows),
            "rows": rows,
            "meaningful_parse_rate": statistics.mean(
                float(row["meaningful_parse"]) for row in rows
            ),
            "structural_similarity": statistics.mean(
                row["structural_similarity"] for row in rows
            ),
            "reward": (
                statistics.mean(
                    row["reward"] for row in rows if row["reward"] is not None
                )
                if any(row["reward"] is not None for row in rows)
                else None
            ),
            "latency_seconds": sum(row["latency_seconds"] for row in rows),
        }
    return result


def _run(
    *,
    checkpoint: Path,
    test_dir: Path,
    slm230_path: Path,
    slm231_path: Path,
    agentv_dir: Path,
    pinned_version_stamp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    stamp = pinned_version_stamp or build_version_stamp(COMPONENT)
    model = TwoTowerModel.from_checkpoint(checkpoint, device="cpu")
    model.eval()
    tower = model.denoiser
    if not isinstance(tower, SharedRecursiveDenoiserTower):
        raise TypeError("checkpoint denoiser must be shared_recursive")
    smoke = _select(load_suite_records(test_dir, "smoke"), PRIMARY_RECORDS["smoke"])
    heldout = _select(
        load_suite_records(test_dir, "held_out"), PRIMARY_RECORDS["held_out"]
    )
    smoke_capture = _capture_batch(model, smoke)
    heldout_capture = _capture_batch(model, heldout)
    checkpoint_sha256 = _sha256_file(checkpoint)
    config_sha256 = stable_hash(asdict(model.config))
    source_state_hash = _state_dict_hash(model)

    smoke_z = compose_z0(smoke_capture["components"])
    assert isinstance(smoke_z, torch.Tensor)
    calibration_mean = smoke_z.mean(dim=(0, 1), keepdim=True)
    permutation, permutation_hash = within_group_permutation(
        ["heldout_same_root", "heldout_same_root"], seed=SEED
    )
    pair_manifest = {
        "schema": "LatentMatchedPairManifestV1",
        "pair_type": "same_root_different_inventory",
        "record_ids": [record.id for record in heldout],
        "permutation": permutation.tolist(),
        "source": "preregistered_bounded_heldout_pair",
    }
    pair_manifest_sha256 = stable_hash(pair_manifest)
    representation = _representation(tower, (smoke_capture, heldout_capture))

    modes = (
        "none",
        "zero_z0",
        "mean_z0",
        "shuffle_z_across_examples",
        "swap_z_matched",
        "zero_ctx_proj",
        "zero_z_latent",
        "remove_z_position",
        "detach_z_to_y",
        "detach_y_to_z",
        "y_only_repeated_control",
        "random_norm_matched",
    )
    teacher_rows: dict[str, Any] = {}
    ablation_results: list[LatentAblationResultV1] = []
    baseline_logits, _, _ = _manual_outputs(
        tower,
        heldout_capture,
        mode="none",
        calibration_mean=calibration_mean,
        permutation=permutation,
        pair_manifest_sha256=pair_manifest_sha256,
    )
    for mode in modes:
        logits, _, norm_delta = _manual_outputs(
            tower,
            heldout_capture,
            mode=mode,
            calibration_mean=calibration_mean,
            permutation=permutation,
            pair_manifest_sha256=pair_manifest_sha256,
        )
        metrics = _teacher_metrics(
            logits, baseline_logits, heldout_capture, PRIMARY_RECORDS["held_out"]
        )
        teacher_rows[mode] = metrics
        ablation_results.append(
            LatentAblationResultV1(
                ablation_id=mode,
                applicability="applicable",
                support=len(heldout),
                seed=SEED,
                pair_manifest_sha256=(
                    pair_manifest_sha256
                    if mode == "swap_z_matched"
                    else permutation_hash
                    if mode == "shuffle_z_across_examples"
                    else None
                ),
                achieved_norm_max_abs_delta=norm_delta,
                teacher_forced_full_vocab_kl=metrics["full_vocab_kl"],
                teacher_forced_top1_change_rate=metrics["top1_change_rate"],
                exact_candidate_status="unavailable_no_provenance_bound_candidates",
                protected_outcome_status="unavailable_no_provenance_bound_decisions",
                free_running_status="pending_primary_selection",
                numerical_status="finite"
                if math.isfinite(metrics["full_vocab_kl"])
                else "nonfinite",
            )
        )
    ablation_results.append(
        LatentAblationResultV1(
            ablation_id="gold_oracle_z",
            applicability="not_applicable",
            support=0,
            seed=None,
            pair_manifest_sha256=None,
            achieved_norm_max_abs_delta=None,
            teacher_forced_full_vocab_kl=None,
            teacher_forced_top1_change_rate=None,
            exact_candidate_status="not_applicable_no_exact_oracle",
            protected_outcome_status="not_applicable_no_exact_oracle",
            free_running_status="not_applicable",
            numerical_status="not_applicable",
            reason="current unsupervised z has no exact compiler-latent oracle",
        )
    )
    targeted = [
        mode
        for mode in modes
        if mode
        not in {"none", "random_norm_matched", "shuffle_z_across_examples"}
    ]
    strongest = max(
        targeted, key=lambda mode: teacher_rows[mode]["full_vocab_kl"]
    )
    free_modes = list(
        dict.fromkeys(["none", strongest, "random_norm_matched", "y_only_repeated_control"])
    )
    free_running = _free_running(
        model, heldout, modes=free_modes, calibration_mean=calibration_mean
    )
    ablation_results = [
        LatentAblationResultV1(
            **{
                **asdict(row),
                "free_running_status": (
                    "bounded_heldout_n2"
                    if row.ablation_id in free_running
                    else row.free_running_status.replace(
                        "pending_primary_selection", "not_run_by_preregistered_gate"
                    )
                ),
            }
        )
        for row in ablation_results
    ]
    post_state_hash = _state_dict_hash(model)
    if post_state_hash != source_state_hash:
        raise RuntimeError("SLM-232 intervention mutated source checkpoint state")

    slm230 = json.loads(slm230_path.read_text(encoding="utf-8"))
    slm231 = json.loads(slm231_path.read_text(encoding="utf-8"))
    nuisance = max(
        teacher_rows["random_norm_matched"]["full_vocab_kl"],
        teacher_rows["shuffle_z_across_examples"]["full_vocab_kl"],
    )
    targeted_effect = teacher_rows[strongest]["full_vocab_kl"]
    verdict = classify_latent_state_use(
        rank_qualified=representation["by_depth"][0][
            "z_after_context_and_position_removal"
        ]["effective_rank"]
        > 1.25,
        context_only=None,
        targeted_effect_reproduced=None,
        targeted_exceeds_nuisance=targeted_effect > nuisance,
        matched_y_only_equivalent=None,
        powered_no_effect=None,
        actual_legal_effect=None,
        protected_outcome_effect=None,
        uncertainty_excludes_zero=None,
        unstable_dynamics=slm231["verdict"] == "expansive_unstable",
        nonvacuous_outcome=any(
            row["meaningful_parse_rate"] > 0.0 for row in free_running.values()
        ),
    )
    gate = LatentStateUseGateV1(
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_config_sha256=config_sha256,
        state_projection=representation["projection"],
        trained_depth=tower.recursive_steps,
        evaluated_depths=tuple(range(tower.recursive_steps + 1)),
        support=4,
        group_support=4,
        representation=representation,
        ablations=tuple(ablation_results),
        exact_state_evidence={
            "status": "censored",
            "reproducible_actual_legal_effect": None,
            "reason": "SLM-230 exact-state legal candidates were unavailable",
        },
        protected_outcome_evidence={
            "status": "censored",
            "reproducible_protected_effect": None,
            "reason": "no provenance-compatible DecisionEvent artifact",
        },
        free_running_evidence={
            "status": "bounded_primary_cells",
            "nonvacuous": any(
                row["meaningful_parse_rate"] > 0.0 for row in free_running.values()
            ),
            "cells": free_modes,
        },
        control_comparison={
            "strongest_targeted_cell": strongest,
            "strongest_targeted_full_vocab_kl": targeted_effect,
            "maximum_nuisance_full_vocab_kl": nuisance,
            "targeted_exceeds_nuisance": targeted_effect > nuisance,
            "matched_y_only_available": True,
            "y_only_block_evaluations": tower.recursive_steps * 2,
            "full_z_block_evaluations": tower.recursive_steps * 2,
            "z_specific_parameter_count_ignored_by_y_only": sum(
                value.numel()
                for name, value in tower.named_parameters()
                if name.startswith(("z_latent", "ctx_proj"))
            ),
        },
        uncertainty={
            "status": "unavailable_tiny_group_support",
            "group_n": 2,
            "positive_effect_excludes_zero": None,
        },
        slm230_join={
            "source": _portable_path(slm230_path),
            "report_hash": slm230["report_hash"],
            "verdict": slm230["verdict"],
        },
        slm231_join={
            "source": _portable_path(slm231_path),
            "report_hash": slm231["report_hash"],
            "verdict": slm231["verdict"],
            "request_scope": ["held_out_form_01"],
            "held_out_dual_card_01": "unavailable_not_copied",
        },
        floor_gate_scope="diagnostic_only_semantic_floor_inconclusive",
        verdict=verdict.value,
        allowed_downstream_work=(
            "diagnostics",
            "control_replication",
            "architecture_repair_without_workspace_claim",
        ),
        blocking_evidence=(
            "SLM-231 expansive_unstable recurrence dynamics",
            "SLM-230 stagnant outcomes with zero meaningful parse/structure/reward",
            "exact legal/protected DecisionEvent evidence unavailable",
            "bounded heldout support n=2 cannot establish uncertainty",
        ),
        version_stamp=stamp,
    )
    gate.validate()

    agentv = publish_agentv_evaluation(
        agentv_dir,
        name="slm232-latent-state-use",
        claim="bounded_latent_state_diagnostic_not_ship",
        version_stamp=stamp,
        cases=[
            {
                "id": "schema-and-scientific-hash",
                "criteria": "The authoritative gate validates and is scientific-hash ready.",
                "pass": gate.to_dict()["schema"] == "LatentStateUseGateV1",
                "result": {"verdict": gate.verdict, "schema": gate.schema},
            },
            {
                "id": "checkpoint-and-state-integrity",
                "criteria": "All interventions are in-memory and restore the exact source state.",
                "pass": source_state_hash == post_state_hash,
                "result": {
                    "checkpoint_sha256": checkpoint_sha256,
                    "pre_state_sha256": source_state_hash,
                    "post_state_sha256": post_state_hash,
                },
            },
            {
                "id": "ablation-coverage-and-selectivity",
                "criteria": "Every declared cell is present and unavailable oracle evidence is explicit.",
                "pass": {row.ablation_id for row in ablation_results}
                == set(modes).union(NON_APPLICABLE_ABLATIONS)
                and next(
                    row
                    for row in ablation_results
                    if row.ablation_id == "gold_oracle_z"
                ).applicability
                == "not_applicable",
                "result": [row.to_dict() for row in ablation_results],
            },
            {
                "id": "pair-manifest-and-support",
                "criteria": "Pairs, groups, source hashes, and deterministic permutations are persisted.",
                "pass": len(permutation_hash) == 64
                and len(pair_manifest_sha256) == 64
                and gate.group_support == 4,
                "result": {
                    "smoke": _manifest(smoke, suite="smoke"),
                    "heldout": _manifest(heldout, suite="held_out"),
                    "shuffle_manifest_sha256": permutation_hash,
                    "swap": pair_manifest,
                    "swap_manifest_sha256": pair_manifest_sha256,
                },
            },
            {
                "id": "honest-censoring-and-prior-join",
                "criteria": "Unavailable exact/protected evidence stays censored and unstable prior dynamics blocks workspace claims.",
                "pass": gate.verdict == "unstable"
                and gate.exact_state_evidence["status"] == "censored"
                and gate.protected_outcome_evidence["status"] == "censored",
                "result": {
                    "verdict": gate.verdict,
                    "slm230": gate.slm230_join,
                    "slm231": gate.slm231_join,
                    "blocking_evidence": gate.blocking_evidence,
                },
            },
        ],
    )
    _rewrite_agentv_paths(agentv_dir)
    report: dict[str, Any] = {
        "schema": "LatentStateUseReportV1",
        "matrix_set": "slm232-latent-state-use",
        "matrix_version": "v1",
        "issue": "SLM-232",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "bounded_checkpoint_diagnostic",
        "claim_class": "scratch_checkpoint_not_ship",
        "verdict": gate.verdict,
        "checkpoint": {
            "path": _portable_path(checkpoint),
            "sha256": checkpoint_sha256,
            "config_sha256": config_sha256,
            "created": False,
            "promotable": False,
        },
        "recipe": {
            "device": "cpu",
            "backend": "scratch",
            "trained_depth": tower.recursive_steps,
            "evaluated_depths": list(gate.evaluated_depths),
            "smoke_calibration_n": 2,
            "heldout_ablation_n": 2,
            "representation_group_n": 4,
            "free_running_primary_cells": free_modes,
            "max_generation_len": 24,
            "max_wall_minutes": 3.0,
            "honesty_mode": "bounded_diagnostic_not_ship",
        },
        "manifests": {
            "smoke": _manifest(smoke, suite="smoke"),
            "heldout": _manifest(heldout, suite="held_out"),
            "shuffle_manifest_sha256": permutation_hash,
            "matched_swap": pair_manifest,
            "matched_swap_manifest_sha256": pair_manifest_sha256,
        },
        "representation": representation,
        "teacher_forced_full_vocab": teacher_rows,
        "free_running": free_running,
        "gate": gate.to_dict(),
        "censored": {
            "exact_legal_candidates": "unavailable_not_zero",
            "D_legal_D_good_D_bad": "unavailable_not_zero",
            "protected_mass_margin_debt": "unavailable_not_zero",
            "decision_kind_role_strata": "unavailable_not_zero",
            "different_root_similar_length_swap": "unavailable_not_zero",
            "runtime_symbol_alpha_pair": "unavailable_not_zero",
            "bootstrap_confidence_interval": "unavailable_n_equals_2",
            "full_frozen_suites": "not_run_by_gate_prior_unstable_and_vacuous",
        },
        "agentv": _portable(agentv, agentv_dir),
        "version_stamp": stamp,
        "training_default_changed": False,
        "generation_default_changed": False,
        "checkpoint_created": False,
        "ship_gate_claim": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["report_hash"] = _scientific_hash(report)
    return report


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _markdown(report: dict[str, Any]) -> str:
    representation = report["representation"]
    gate = report["gate"]
    strongest = gate["control_comparison"]["strongest_targeted_cell"]
    lines = [
        "# SLM-232 latent-state rank and causal-use audit",
        "",
        f"Verdict: **{report['verdict']}**",
        "",
        f"Report hash: `{report['report_hash']}`",
        "",
        "This bounded CPU audit reuses the rejected SLM-230 scratch checkpoint. "
        "It changes no source weights, training default, generation default, or "
        "promotion state.",
        "",
        "## Pathway and intervention points",
        "",
        "```text",
        "z_latent[position] ----[zero_z_latent]---\\",
        "ctx_proj(pool(context))-[zero_ctx_proj]----+--> z0 --> F(norm(y+z)) --> z'",
        "position[position] ----[remove_z_position]-/      |                  |",
        "                                                       detach_y_to_z    |",
        "y0 -----------------------------------------------> G(norm(y+z')) --> y'",
        "                                                       detach_z_to_y",
        "```",
        "",
        "All cells are evaluation-only functional overrides. The checkpoint state "
        "hash is identical before and after the matrix.",
        "",
        "## Representation",
        "",
        "| depth | effective rank | participation | centered energy | z/y CKA |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in representation["by_depth"]:
        lines.append(
            f"| {row['depth']} | {_fmt(row['z']['effective_rank'])} | "
            f"{_fmt(row['z']['participation_ratio'])} | "
            f"{_fmt(row['z']['total_centered_energy'])} | "
            f"{_fmt(row['z_y_linear_cka'])} |"
        )
    lines.extend(
        [
            "",
            "The z0 rank after removing the pooled-context and position terms is "
            f"`{_fmt(representation['by_depth'][0]['z_after_context_and_position_removal']['effective_rank'])}`. "
            "This is a four-record descriptive estimate; token positions are not "
            "treated as independent groups.",
            "",
            "## Causal cells",
            "",
            "| cell | full-vocab KL | top-1 change | target accuracy |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    cell_order = (
        "none",
        "zero_z0",
        "mean_z0",
        "shuffle_z_across_examples",
        "swap_z_matched",
        "zero_ctx_proj",
        "zero_z_latent",
        "remove_z_position",
        "detach_z_to_y",
        "detach_y_to_z",
        "y_only_repeated_control",
        "random_norm_matched",
    )
    for mode in cell_order:
        row = report["teacher_forced_full_vocab"][mode]
        lines.append(
            f"| `{mode}` | {_fmt(row['full_vocab_kl'])} | "
            f"{_fmt(row['top1_change_rate'])} | {_fmt(row['target_accuracy'])} |"
        )
    lines.extend(
        [
            "",
            f"The largest targeted full-vocabulary effect is `{strongest}`. "
            "This is sensitivity, not evidence of useful or legal reasoning: exact "
            "legal/protected candidate sets are unavailable and remain censored.",
            "",
            "## Outcome join and disposition",
            "",
            f"- SLM-230: `{gate['slm230_join']['verdict']}` "
            f"(`{gate['slm230_join']['report_hash']}`).",
            f"- SLM-231: `{gate['slm231_join']['verdict']}` "
            f"(`{gate['slm231_join']['report_hash']}`).",
            f"- AgentV: `{json.dumps(report['agentv']['summary'], sort_keys=True)}`.",
            "",
            "The current z is measurably variable and its removal can alter full-"
            "vocabulary logits, but the authoritative disposition is **unstable**: "
            "joined recurrence dynamics are expansive, bounded outputs remain "
            "vacuous, and no provenance-compatible legal/protected outcome artifact "
            "exists. RSC2/RSC3 must not treat this checkpoint as evidence for a "
            "causally useful reasoning workspace. Diagnostic replication and "
            "architecture repair remain allowed.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "timeout 170s env PYTHONPATH=src .venv/bin/python -m "
            "scripts.run_slm232_latent_state_use --check",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--slm230", type=Path, default=DEFAULT_SLM230)
    parser.add_argument("--slm231", type=Path, default=DEFAULT_SLM231)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--agentv-dir", type=Path, default=DEFAULT_AGENTV)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    committed = (
        json.loads(args.json_out.read_text(encoding="utf-8")) if args.check else None
    )
    agentv_context = (
        TemporaryDirectory(prefix="slm232-agentv-check-")
        if args.check
        else nullcontext(str(args.agentv_dir))
    )
    with agentv_context as agentv_dir:
        report = _run(
            checkpoint=args.checkpoint,
            test_dir=args.test_dir,
            slm230_path=args.slm230,
            slm231_path=args.slm231,
            agentv_dir=Path(agentv_dir),
            pinned_version_stamp=(
                committed["version_stamp"] if committed is not None else None
            ),
        )
    if args.check:
        assert committed is not None
        if committed["report_hash"] != report["report_hash"]:
            raise SystemExit(
                f"SLM-232 report hash mismatch: {committed['report_hash']} "
                f"!= {report['report_hash']}"
            )
        if args.markdown_out.read_text(encoding="utf-8") != _markdown(committed):
            raise SystemExit("SLM-232 Markdown is inconsistent with committed JSON")
    else:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        args.markdown_out.write_text(_markdown(report), encoding="utf-8")
    print(
        f"LatentStateUseGateV1 {report['report_hash']} {report['verdict']} "
        f"{report['elapsed_seconds']:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
