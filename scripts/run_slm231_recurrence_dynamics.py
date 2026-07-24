#!/usr/bin/env python3
"""Run SLM-231's bounded residual recurrence-dynamics audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import quote

import torch

from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harnesses.experiments.slm231_recurrence_dynamics import (
    DYNAMICS_SCHEMA,
    ESTIMATOR_VERSION,
    RecurrenceDynamicsSnapshotV1,
    block_singular_estimate,
    classify_dynamics,
    exact_product,
    finite_time_lyapunov,
    flatten_projected_state,
    linear_cka,
    matrix_summary,
    principal_angles,
    projected_transition,
    residual_jacobians,
    stable_hash,
    state_projection,
    trajectory_product_estimate,
)
from slm_training.harnesses.model_build.data import load_suite_records
from slm_training.models.recursive_denoiser import SharedRecursiveDenoiserTower
from slm_training.models.rng_contract import seed_training_corruption
from slm_training.models.twotower import TwoTowerModel
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "docs/design/iter-slm231-recurrence-dynamics-20260724.json"
DEFAULT_MARKDOWN = ROOT / "docs/design/iter-slm231-recurrence-dynamics-20260724.md"
DEFAULT_AGENTV = ROOT / "docs/design/iter-slm231-recurrence-dynamics-agentv-20260724"
DEFAULT_CHECKPOINT = (
    ROOT / "outputs/runs/slm230_bounded_recursive_r4_r2/checkpoints/last.pt"
)
DEFAULT_TEST_DIR = (
    ROOT / "src/slm_training/resources/data/eval" / "e763_symbol_only_eval_r2_20260722"
)
DEFAULT_SLM230 = ROOT / "docs/design/iter-slm230-recurrence-observability-20260724.json"
COMPONENT = "harness.experiments.slm231_recurrence_dynamics"


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
    payload.pop("report_hash", None)
    payload.pop("generated_at", None)
    payload.pop("elapsed_seconds", None)
    stamp = dict(payload.get("version_stamp") or {})
    stamp.pop("stamped_at", None)
    payload["version_stamp"] = stamp
    agentv = dict(payload.get("agentv") or {})
    summary = dict(agentv.get("summary") or {})
    summary.pop("durationMs", None)
    agentv["summary"] = summary
    payload["agentv"] = agentv
    return stable_hash(payload)


def _linear_controls() -> dict[str, Any]:
    state = torch.zeros(4, dtype=torch.float64)
    controls = {
        "identity": torch.eye(4, dtype=torch.float64),
        "stable_diagonal": torch.diag(
            torch.tensor([1.02, 0.97, 0.8, 0.6], dtype=torch.float64)
        ),
        "rotating": torch.tensor(
            [
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.1, 0.0],
                [0.0, 0.0, 0.0, 0.9],
            ],
            dtype=torch.float64,
        ),
    }
    rows = {}
    for index, (name, matrix) in enumerate(controls.items()):
        increment, composite = residual_jacobians(
            lambda value, matrix=matrix: matrix @ value, state
        )
        estimate = block_singular_estimate(
            lambda value, matrix=matrix: matrix @ value,
            state,
            k=2,
            iterations=16,
            seed=231_000 + index,
        )
        exact = torch.linalg.svdvals(composite)
        rows[name] = {
            "increment": matrix_summary(increment),
            "composite": matrix_summary(composite),
            "estimated_top": list(estimate.singular_values),
            "exact_top": [float(value) for value in exact[:2]],
            "maximum_absolute_error": max(
                abs(left - float(right))
                for left, right in zip(estimate.singular_values, exact[:2])
            ),
            "estimator_residual": estimate.residual,
        }
    product = exact_product((controls["stable_diagonal"], controls["rotating"]))
    rows["known_product"] = {
        "order": "rotating @ stable_diagonal",
        "singular_values": [float(value) for value in torch.linalg.svdvals(product)],
        "ftle": list(
            finite_time_lyapunov(
                tuple(float(value) for value in torch.linalg.svdvals(product)),
                depth=2,
            )
        ),
    }
    return rows


def _fixture_profile() -> dict[str, Any]:
    """Exercise exact and approximate estimators on the SLM-138 architecture."""
    torch.manual_seed(138)
    tower = (
        SharedRecursiveDenoiserTower(
            vocab_size=13,
            d_model=4,
            n_layers=2,
            n_heads=1,
            max_len=4,
            recursive_steps=2,
            recursive_transition_layers=2,
        )
        .double()
        .eval()
    )
    noisy = torch.tensor([[1, 2]])
    context = torch.randn(1, 2, 4, dtype=torch.float64)
    initial = tower.initial_transition_state(noisy, context, 0)
    y, z = initial["y"], initial["z"]
    assert isinstance(y, torch.Tensor)
    assert isinstance(z, torch.Tensor)
    self_pad = initial["self_pad_mask"]
    assert isinstance(self_pad, torch.Tensor)
    projection = state_projection(y, z, (0,))
    target = projected_transition(
        tower,
        base_y=y,
        base_z=z,
        context=context,
        self_pad_mask=self_pad,
        ctx_pad_mask=None,
        runtime_symbol_features=None,
        projection=projection,
    )
    state = flatten_projected_state(y, z, projection)
    increment, composite = residual_jacobians(target, state)
    estimate = block_singular_estimate(target, state, k=2, iterations=16, seed=138_231)
    exact = torch.linalg.svdvals(composite)
    with torch.no_grad():
        normal = tower.recursive_outputs(noisy, context, 0)
        manual = tower.transition_step(y, z, context, self_pad)
    logits = manual["logits"]
    assert isinstance(logits, torch.Tensor)
    parity_error = float(
        (logits - normal["depth_logits"][0]).abs().max().detach().cpu()
    )
    return {
        "source": "SLM-138 canonical SharedRecursiveDenoiserTower fixture configuration",
        "claim_scope": "api_and_estimator_wiring_only",
        "projection": asdict(projection),
        "transition_parity_max_abs_error": parity_error,
        "increment": matrix_summary(increment),
        "composite": matrix_summary(composite),
        "estimated_top": list(estimate.singular_values),
        "exact_top": [float(value) for value in exact[:2]],
        "maximum_absolute_error": max(
            abs(left - float(right))
            for left, right in zip(estimate.singular_values, exact[:2])
        ),
        "estimator_residual": estimate.residual,
    }


def _capture_checkpoint_state(
    model: TwoTowerModel,
    record: Any,
) -> dict[str, Any]:
    tower = model.denoiser
    assert isinstance(tower, SharedRecursiveDenoiserTower)
    original_outputs = tower.recursive_outputs
    original_mask = model._mask_targets
    captured: dict[str, Any] = {}

    def capture_mask(target_ids: torch.Tensor) -> Any:
        noisy, mask, weights = original_mask(target_ids)
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
        initial = tower.initial_transition_state(
            noisy_ids, context, pad_id, ctx_pad_mask
        )
        captured.update(
            {
                "context": context.detach().clone(),
                "ctx_pad_mask": (
                    None if ctx_pad_mask is None else ctx_pad_mask.detach().clone()
                ),
                "pad_id": pad_id,
                "initial": {
                    key: None if value is None else value.detach().clone()
                    for key, value in initial.items()
                },
            }
        )
        return original_outputs(noisy_ids, context, pad_id, ctx_pad_mask, **kwargs)

    model._mask_targets = capture_mask  # type: ignore[method-assign]
    tower.recursive_outputs = capture_outputs  # type: ignore[method-assign]
    try:
        seed_training_corruption(231_230, model, override_seed=231_230)
        with torch.no_grad():
            model.training_loss([record])
    finally:
        model._mask_targets = original_mask  # type: ignore[method-assign]
        tower.recursive_outputs = original_outputs  # type: ignore[method-assign]
    return captured


def _alignment_row(
    y_before: torch.Tensor,
    z_before: torch.Tensor | None,
    y_after: torch.Tensor,
    z_after: torch.Tensor | None,
    previous_update: torch.Tensor | None,
) -> tuple[dict[str, Any], torch.Tensor]:
    y_left = y_before[0]
    y_right = y_after[0]
    update = y_right - y_left
    cosine = None
    if previous_update is not None:
        cosine = float(
            torch.nn.functional.cosine_similarity(
                update.reshape(1, -1), previous_update.reshape(1, -1)
            )
        )
    row: dict[str, Any] = {
        "y_principal_angles": list(principal_angles(y_left, y_right, rank=2)),
        "y_linear_cka": linear_cka(y_left, y_right),
        "y_update_norm": float(torch.linalg.vector_norm(update)),
        "y_update_cosine_previous": cosine,
        "z_principal_angles": None,
        "z_linear_cka": None,
    }
    if z_before is not None and z_after is not None:
        row["z_principal_angles"] = list(
            principal_angles(z_before[0], z_after[0], rank=2)
        )
        row["z_linear_cka"] = linear_cka(z_before[0], z_after[0])
    return row, update


def _checkpoint_profile(
    *,
    checkpoint: Path,
    test_dir: Path,
    slm230_path: Path,
    stamp: dict[str, Any],
) -> RecurrenceDynamicsSnapshotV1:
    model = TwoTowerModel.from_checkpoint(checkpoint, device="cpu")
    model.eval()
    tower = model.denoiser
    if not isinstance(tower, SharedRecursiveDenoiserTower):
        raise TypeError("checkpoint denoiser must be shared_recursive")
    record = load_suite_records(test_dir, "held_out")[0]
    captured = _capture_checkpoint_state(model, record)
    initial = captured["initial"]
    y, z = initial["y"], initial["z"]
    assert isinstance(y, torch.Tensor)
    assert isinstance(z, torch.Tensor)
    context = captured["context"]
    self_pad = initial["self_pad_mask"]
    runtime_features = initial["runtime_symbol_features"]
    assert isinstance(context, torch.Tensor)
    assert isinstance(self_pad, torch.Tensor)
    assert runtime_features is None or isinstance(runtime_features, torch.Tensor)
    ctx_pad = captured["ctx_pad_mask"]
    active = torch.nonzero(captured["mask"][0], as_tuple=False).flatten()
    projection = state_projection(y, z, (int(active[0]),))

    transitions = []
    states = []
    increment_rows = []
    composite_rows = []
    alignment_rows = []
    exact_composites = []
    previous_update = None
    for depth in range(1, tower.recursive_steps + 1):
        target = projected_transition(
            tower,
            base_y=y,
            base_z=z,
            context=context,
            self_pad_mask=self_pad,
            ctx_pad_mask=ctx_pad,
            runtime_symbol_features=runtime_features,
            projection=projection,
        )
        state = flatten_projected_state(y, z, projection)
        transitions.append(target)
        states.append(state)

        def increment_target(
            value: torch.Tensor,
            target: Any = target,
        ) -> torch.Tensor:
            return target(value) - value

        increment_estimate = block_singular_estimate(
            increment_target,
            state,
            k=2,
            iterations=6,
            seed=231_000 + depth,
        )
        composite_estimate = block_singular_estimate(
            target,
            state,
            k=2,
            iterations=6,
            seed=231_100 + depth,
        )
        increment_rows.append(
            {
                "depth": depth,
                "top_singular_values": list(increment_estimate.singular_values),
                "residual": increment_estimate.residual,
                "method": increment_estimate.method,
            }
        )
        composite_rows.append(
            {
                "depth": depth,
                "top_singular_values": list(composite_estimate.singular_values),
                "residual": composite_estimate.residual,
                "identity_control_top": [1.0, 1.0],
                "method": composite_estimate.method,
            }
        )
        # Exact products are bounded to a single active token (2D state tuple).
        _, exact_composite = residual_jacobians(target, state)
        exact_composites.append(exact_composite)
        with torch.no_grad():
            step = tower.transition_step(
                y,
                z,
                context,
                self_pad,
                ctx_pad,
                runtime_symbol_features=runtime_features,
            )
        next_y, next_z = step["y"], step["z"]
        assert isinstance(next_y, torch.Tensor)
        assert isinstance(next_z, torch.Tensor)
        alignment, previous_update = _alignment_row(
            y, z, next_y, next_z, previous_update
        )
        alignment["depth"] = depth
        alignment_rows.append(alignment)
        y, z = next_y, next_z

    product = exact_product(exact_composites)
    exact_product_singular = tuple(
        float(value) for value in torch.linalg.svdvals(product)
    )
    product_estimate = trajectory_product_estimate(
        transitions,
        states,
        k=2,
        iterations=8,
        seed=231_200,
    )
    ftle = finite_time_lyapunov(exact_product_singular, depth=tower.recursive_steps)
    slm230 = json.loads(slm230_path.read_text(encoding="utf-8"))
    joined = [
        row
        for row in slm230["observations"]
        if row["record_id"] == record.id and row["split"] == "heldout"
    ]
    update_cosines = [
        row["y_update_cosine_previous"]
        for row in alignment_rows
        if row["y_update_cosine_previous"] is not None
    ]
    verdict = classify_dynamics(
        increment_spectral_norm=max(
            row["top_singular_values"][0] for row in increment_rows
        ),
        product_spectral_norm=exact_product_singular[0],
        maximum_ftle=max(ftle),
        update_alignment_cosine=(
            statistics.mean(update_cosines) if update_cosines else 0.0
        ),
        outcome_verdict=slm230["verdict"],
    )
    snapshot = RecurrenceDynamicsSnapshotV1(
        checkpoint_sha256=_sha256_file(checkpoint),
        transition_hash=stable_hash(
            {
                "model_config": asdict(model.config),
                "projection": asdict(projection),
                "checkpoint": _sha256_file(checkpoint),
            }
        ),
        request_id=record.id,
        group_id=record.id,
        split="heldout",
        suite="e763_symbol_only_eval_r2_20260722/held_out",
        trained_depth=tower.recursive_steps,
        evaluated_depth=tower.recursive_steps,
        projection=asdict(projection),
        increment_by_depth=tuple(increment_rows),
        composite_by_depth=tuple(composite_rows),
        product_singular_values=exact_product_singular,
        finite_time_lyapunov=ftle,
        alignment_by_depth=tuple(alignment_rows),
        outcome_join={
            "source": _portable_path(slm230_path),
            "source_report_hash": slm230["report_hash"],
            "source_verdict": slm230["verdict"],
            "observations": joined,
            "slm220_verifier_subspace": {
                "status": "unavailable_for_checkpoint_state",
                "handling": "censored_not_zero",
            },
        },
        nulls={
            "identity_increment_top": [0.0, 0.0],
            "identity_composite_top": [1.0, 1.0],
            "depth_order_shuffle": "not_interpretable_for_shared_identical_transition",
            "group_permutation": "unavailable_n_equals_1",
            "random_matched_increment": "synthetic_control_only",
        },
        estimator={
            "version": ESTIMATOR_VERSION,
            "increment_iterations": 6,
            "product_iterations": 8,
            "top_k": 2,
            "product_estimated_top": list(product_estimate.singular_values),
            "product_exact_top": list(exact_product_singular[:2]),
            "product_residual": product_estimate.residual,
            "product_max_absolute_error": max(
                abs(left - right)
                for left, right in zip(
                    product_estimate.singular_values, exact_product_singular[:2]
                )
            ),
            "group_bootstrap": {
                "status": "unavailable",
                "reason": "bounded initial study has one independent request group",
            },
        },
        numerical_flags=(
            ("product_estimator_not_converged",)
            if product_estimate.residual > 0.05
            or max(
                abs(left - right)
                for left, right in zip(
                    product_estimate.singular_values, exact_product_singular[:2]
                )
            )
            > 0.05
            else ()
        ),
        floor_gate_claim_scope="diagnostic_only_semantic_floor_inconclusive",
        verdict=verdict.value,
        version_stamp=stamp,
    )
    snapshot.validate()
    return snapshot


def _run(
    *,
    checkpoint: Path,
    test_dir: Path,
    slm230_path: Path,
    agentv_dir: Path,
    pinned_version_stamp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    stamp = pinned_version_stamp or build_version_stamp(COMPONENT)
    synthetic = _linear_controls()
    fixture = _fixture_profile()
    snapshot = _checkpoint_profile(
        checkpoint=checkpoint,
        test_dir=test_dir,
        slm230_path=slm230_path,
        stamp=stamp,
    )
    estimator_error = snapshot.estimator["product_max_absolute_error"]
    agentv = publish_agentv_evaluation(
        agentv_dir,
        name="slm231-recurrence-dynamics",
        claim="bounded_recurrence_dynamics_diagnostic_not_ship",
        version_stamp=stamp,
        cases=[
            {
                "id": "residual-identity-separation",
                "criteria": "Every real profile reports learned increment and composite spectra separately beside an identity control.",
                "pass": len(snapshot.increment_by_depth)
                == len(snapshot.composite_by_depth)
                == snapshot.evaluated_depth
                and all(
                    row["identity_control_top"] == [1.0, 1.0]
                    for row in snapshot.composite_by_depth
                ),
                "result": {
                    "increment": snapshot.increment_by_depth,
                    "composite": snapshot.composite_by_depth,
                },
            },
            {
                "id": "analytic-estimator-agreement",
                "criteria": "Exact and JVP/VJP top singular values agree within 2e-4 on all analytic controls and the SLM-138 wiring fixture.",
                "pass": max(
                    [
                        row["maximum_absolute_error"]
                        for row in synthetic.values()
                        if "maximum_absolute_error" in row
                    ]
                    + [fixture["maximum_absolute_error"]]
                )
                <= 2e-4,
                "result": {"synthetic": synthetic, "fixture": fixture},
            },
            {
                "id": "trajectory-product-agreement",
                "criteria": "The bounded product has an exact reference; approximate error and convergence status are explicit and do not silently own the verdict.",
                "pass": math.isfinite(estimator_error)
                and (
                    estimator_error <= 0.05
                    or "product_estimator_not_converged" in snapshot.numerical_flags
                ),
                "result": {
                    **snapshot.estimator,
                    "numerical_flags": snapshot.numerical_flags,
                    "verdict_owner": "exact_bounded_product",
                },
            },
            {
                "id": "outcome-joined-verdict",
                "criteria": "The dynamics verdict is joined to SLM-230 heldout depth outcomes and does not authorize a semantic or promotion claim.",
                "pass": bool(snapshot.outcome_join["observations"])
                and snapshot.floor_gate_claim_scope
                == "diagnostic_only_semantic_floor_inconclusive",
                "result": {
                    "verdict": snapshot.verdict,
                    "outcome_join": snapshot.outcome_join,
                },
            },
        ],
    )
    _rewrite_agentv_paths(agentv_dir)
    report: dict[str, Any] = {
        "schema": "RecurrenceDynamicsReportV1",
        "matrix_set": "slm231-recurrence-dynamics",
        "matrix_version": ESTIMATOR_VERSION,
        "issue": "SLM-231",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "bounded_checkpoint_diagnostic",
        "claim_class": "scratch_checkpoint_not_ship",
        "verdict": snapshot.verdict,
        "checkpoint": {
            "path": _portable_path(checkpoint),
            "sha256": snapshot.checkpoint_sha256,
            "trained_recurrence_depth": snapshot.trained_depth,
            "created": False,
            "promotable": False,
        },
        "recipe": {
            "device": "cpu",
            "backend": "scratch",
            "synthetic_controls": len(synthetic),
            "fixture_n": 1,
            "heldout_n": 1,
            "state_projection": snapshot.projection,
            "honesty_mode": "bounded_diagnostic_not_ship",
            "max_wall_minutes": 3.0,
        },
        "thresholds": {
            "dead_increment_spectral_norm_max": 1e-5,
            "expansive_product_spectral_norm_min": 4.0,
            "expansive_ftle_min": 0.35,
            "overcontractive_product_spectral_norm_max": 0.1,
            "oscillatory_update_cosine_max": -0.25,
            "identity_only_increment_spectral_norm_max": 0.05,
            "identity_only_product_band": [0.95, 1.05],
        },
        "synthetic_validation": synthetic,
        "slm138_fixture": fixture,
        "snapshots": [snapshot.to_dict()],
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
    snapshot = report["snapshots"][0]
    lines = [
        "# SLM-231 residual recurrence dynamics",
        "",
        f"Verdict: **{report['verdict']}**",
        "",
        f"Report hash: `{report['report_hash']}`",
        "",
        "This is a bounded CPU diagnostic on the non-promotable SLM-230 scratch "
        "checkpoint. It changes no training, generation, promotion, or ship default.",
        "",
        "## Evidence boundary",
        "",
        f"- Checkpoint: `{report['checkpoint']['path']}` (`{report['checkpoint']['sha256']}`)",
        f"- State projection: `{json.dumps(report['recipe']['state_projection'], sort_keys=True)}`",
        f"- AgentV: `{json.dumps(report['agentv']['summary'], sort_keys=True)}`",
        "- SLM-138 is API/estimator wiring evidence only.",
        "- SLM-220 verifier subspaces were unavailable for this exact checkpoint state and remain censored.",
        "",
        "## Residual-correct spectra",
        "",
        "| depth | increment top | composite top | identity control |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for increment, composite in zip(
        snapshot["increment_by_depth"],
        snapshot["composite_by_depth"],
        strict=True,
    ):
        lines.append(
            f"| {increment['depth']} | {_fmt(increment['top_singular_values'][0])} | "
            f"{_fmt(composite['top_singular_values'][0])} | 1.000000 |"
        )
    lines.extend(
        [
            "",
            "## Trajectory product",
            "",
            f"- Exact product top singular value: **{_fmt(snapshot['product_singular_values'][0])}**",
            f"- Maximum FTLE: **{_fmt(max(snapshot['finite_time_lyapunov']))}**",
            f"- JVP/VJP vs exact maximum error: **{_fmt(snapshot['estimator']['product_max_absolute_error'])}**",
            "",
            "## Outcome join and disposition",
            "",
            f"The exact-state profile joins `{snapshot['request_id']}` to SLM-230's "
            f"**{snapshot['outcome_join']['source_verdict']}** depth verdict. Dynamics "
            "are diagnostic rather than independent evidence of useful reasoning. "
            f"The resulting gate verdict is **{report['verdict']}**.",
            "",
            "RSC2/RSC3 must remain blocked unless later independent groups supply "
            "non-vacuous semantic outcomes and uncertainty-qualified dynamics.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "timeout 170s env PYTHONPATH=src .venv/bin/python -m "
            "scripts.run_slm231_recurrence_dynamics --check",
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
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--agentv-dir", type=Path, default=DEFAULT_AGENTV)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    committed = (
        json.loads(args.json_out.read_text(encoding="utf-8")) if args.check else None
    )
    agentv_context = (
        TemporaryDirectory(prefix="slm231-agentv-check-")
        if args.check
        else nullcontext(str(args.agentv_dir))
    )
    with agentv_context as agentv_dir:
        report = _run(
            checkpoint=args.checkpoint,
            test_dir=args.test_dir,
            slm230_path=args.slm230,
            agentv_dir=Path(agentv_dir),
            pinned_version_stamp=(
                committed["version_stamp"] if committed is not None else None
            ),
        )
    if args.check:
        assert committed is not None
        if committed["report_hash"] != report["report_hash"]:
            raise SystemExit(
                f"SLM-231 report hash mismatch: {committed['report_hash']} "
                f"!= {report['report_hash']}"
            )
        if args.markdown_out.read_text(encoding="utf-8") != _markdown(committed):
            raise SystemExit("SLM-231 Markdown is inconsistent with committed JSON")
    else:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        args.markdown_out.write_text(_markdown(report), encoding="utf-8")
    print(f"{DYNAMICS_SCHEMA} {report['report_hash']} {report['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
