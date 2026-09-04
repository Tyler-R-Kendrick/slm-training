"""Screening-lane policy: how many candidates, on what device, for how long.

One responsibility: the decisions that shape a screening round before anything
runs -- candidate count, GPU-hour ceiling, thrash step budget, the regime the
round is in, and the argv used to rebuild its data locally.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.autotrain_arms import (
    size_match_skip_reason,
)
from scripts.autotrain_budget import (
    fit_screening_steps,
    screening_thrash_steps_max,
    steps_per_sec_from_train_payload,
    thrash_timing_block,
)
from scripts.autotrain_levers import (
    lever_knobs,
    matrix_experiment_knobs,
    matrix_treatment_signature,
)
from slm_training.autoresearch.hillclimb import (
    climb_champion_checkpoint_path,
    load_climb_champion,
)
from slm_training.autoresearch.thrash_regime import (
    ThrashRegimeDecision,
    decide_screening_regime,
    select_climb_baseline_entry,
)
from slm_training.levers import (
    HARNESS_FINALIZATION_RESERVE_SECONDS,
    MAX_RUN_SECONDS,
)


def local_rebuild_screening_eval_argv(
    *, eval_version: str, train_manifest: Path
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.build_test_data",
        "--source",
        "fixture",
        "--version",
        eval_version,
        "--train-manifest",
        str(train_manifest),
    ]


def screening_train_device() -> str:
    """CUDA at train launch when present; CPU fallback. Never raises."""
    try:
        import torch

        return "cuda" if bool(torch.cuda.is_available()) else "cpu"
    except Exception:  # noqa: BLE001 — missing torch / driver is CPU
        return "cpu"


def screening_max_gpu_hours(*, role: str, device: str | None = None) -> float:
    """Engine routes ``--device auto`` iff max_gpu_hours > 0."""
    if role != "screening":
        return 0.0
    chosen = device if device is not None else screening_train_device()
    if chosen != "cuda":
        return 0.0
    return float(MAX_RUN_SECONDS) / 3600.0


def latest_train_telemetry_payload(root: Path | None) -> dict[str, Any] | None:
    if root is None or not root.is_dir():
        return None
    newest: Path | None = None
    newest_mtime = -1.0
    try:
        summaries = root.glob("*/runs/*/train_summary.json")
    except OSError:
        return None
    for path in summaries:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > newest_mtime:
            newest_mtime = mtime
            newest = path
    if newest is None:
        return None
    try:
        payload = json.loads(newest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    payload["_telemetry_path"] = str(newest)
    tel_path = newest.parent / "train_telemetry.json"
    if tel_path.is_file() and payload.get("elapsed_wall_seconds") is None:
        try:
            tel = json.loads(tel_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            tel = None
        if isinstance(tel, dict) and tel.get("total_ms") is not None:
            payload["total_ms"] = tel.get("total_ms")
    return payload


def screening_thrash_steps(
    policy: Any,
    requested_steps: int,
    *,
    floor_seconds: float | None = None,
    measured_steps_per_sec: float | None = None,
    telemetry_root: Path | None = None,
) -> int:
    """Fit screening train steps to the (grown) train floor.

    ``steps = clamp(floor * measured_sps * 0.9, 1, screening_thrash_steps_max)``.
    Cold-start uses ``_COLD_START_STEPS_PER_SEC`` when no telemetry exists.
    """
    thrash = thrash_timing_block(policy)
    steps_max = screening_thrash_steps_max(thrash)
    floor = (
        float(floor_seconds)
        if floor_seconds is not None
        else float(thrash.get("min_train_floor_seconds") or 20.0)
    )
    sps = measured_steps_per_sec
    if sps is None:
        payload = latest_train_telemetry_payload(telemetry_root)
        if payload is not None:
            sps = steps_per_sec_from_train_payload(payload)
    fitted, _evidence = fit_screening_steps(
        floor_seconds=floor,
        measured_steps_per_sec=sps,
        steps_max=steps_max,
    )
    del requested_steps
    return int(fitted)


def policy_default_decode_floor_seconds(policy: Any) -> float:
    """``measurement.screening_sample_size.default_decode_floor_seconds`` (2 s)."""

    measurement = getattr(policy, "measurement", None) or {}
    block = (
        measurement.get("screening_sample_size")
        if isinstance(measurement, dict)
        else None
    )
    raw = block.get("default_decode_floor_seconds") if isinstance(block, dict) else None
    try:
        return max(0.0, float(raw)) if raw is not None else 2.0
    except (TypeError, ValueError):
        return 2.0


def fit_screening_candidate_count(
    *,
    max_candidates: int,
    arm_wall_seconds: float,
    stage_remaining_seconds: float,
    finalization_reserve: float = HARNESS_FINALIZATION_RESERVE_SECONDS,
) -> tuple[int, str | None]:
    """Fit k candidates beside one control; never shrink a running arm wall."""

    requested = max(1, int(max_candidates))
    wall = float(arm_wall_seconds)
    usable = float(stage_remaining_seconds) - float(finalization_reserve)
    if wall <= 0:
        return 1, "invalid_arm_wall"
    fit = int(usable // wall) - 1
    if fit < 1:
        return 1, "stage_wall_fits_one_candidate"
    if fit < requested:
        return fit, "stage_wall_fitted_candidate_count"
    return requested, None


def screening_multi_arm_ids(
    *,
    matrix: Mapping[str, Any],
    control_id: str,
    recommended_id: str,
    fitted_candidates: int,
    by_id: Mapping[str, Path],
) -> tuple[list[str], list[dict[str, str]]]:
    skipped: list[dict[str, str]] = []
    control_knobs = matrix_experiment_knobs(matrix, control_id)
    ordered: list[str] = []
    if recommended_id and recommended_id != control_id:
        ordered.append(str(recommended_id))
    for row in matrix.get("hypotheses") or []:
        if not isinstance(row, dict):
            continue
        eid = str((row.get("experiment") or {}).get("experiment_id") or "")
        if not eid or eid == control_id or eid in ordered or eid not in by_id:
            continue
        ordered.append(eid)
    picked: list[str] = []
    for eid in ordered:
        if len(picked) >= fitted_candidates:
            break
        reason = size_match_skip_reason(
            control_knobs, matrix_experiment_knobs(matrix, eid)
        )
        if reason:
            skipped.append({"arm_id": eid, "reason": reason})
            continue
        picked.append(eid)
    if not picked and recommended_id and recommended_id != control_id:
        picked = [str(recommended_id)]
    return picked, skipped


def exhaust_screening_losers(
    *,
    root: Path,
    loop_id: str,
    policy: Any,
    matrix: Mapping[str, Any],
    control_id: str,
    loser_ids: Sequence[str],
    claim_class: str,
    primary_metric: str,
    direction: str,
    train_version: str,
    eval_version: str,
) -> None:
    from slm_training.autoresearch.climb_policy import (
        load_loop_exhausted_ledger,
        loop_data_eval_identity,
        save_loop_exhausted_ledger,
    )

    if not loser_ids:
        return
    ledger = load_loop_exhausted_ledger(root, loop_id, policy)
    for eid in loser_ids:
        knobs = matrix_experiment_knobs(matrix, eid)
        signature = matrix_treatment_signature(matrix, eid, control_id)
        if not signature:
            continue
        identity = loop_data_eval_identity(
            policy,
            claim_class=claim_class,
            train_version=str(knobs.get("train_version") or train_version),
            eval_version=str(knobs.get("eval_version") or eval_version),
            primary_metric=primary_metric,
            direction=direction,
        )
        ledger.record_null(
            knob_signature_sha256=signature,
            data_eval_identity=identity,
            claim_class=claim_class,
            reason="multi_arm_screening_loser",
            note=f"arm={eid}",
        )
    save_loop_exhausted_ledger(ledger, root, loop_id, policy)


LOCAL_I10_ROOT_CAP = 8


def local_i10_train_version(loop_id: str, cycle_index: int) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", str(loop_id).lower()).strip("_")
    return f"continuous_i10_{safe}_c{int(cycle_index)}"


def local_rebuild_data_argv(
    *, train_version: str, adequacy: dict | None = None
) -> list[str]:
    """Compile a local, data-only build_train_data command from climb policy.

    When the cycle's sample-adequacy verdict is ``generate_more``, the
    rebuild is shaped by ``sample_adequacy_intervention`` (until_coverage
    targeting). The local wall-capped heal keeps the plan's own coverage
    minima — the raised component minimum applies at promotion scale, where
    the fail-closed build has the attempt budget to satisfy it.
    """
    from slm_training.autoresearch.climb_policy import (
        data_intervention_action,
        load_climb_policy,
        sample_adequacy_intervention,
    )
    from slm_training.autoresearch.engine import _data_generation_flags
    from slm_training.autoresearch.schemas import DataGenerationKnobs

    policy = load_climb_policy()
    spec = data_intervention_action(policy)
    if adequacy is not None:
        shaped = sample_adequacy_intervention(policy, adequacy)
        if shaped is not None and shaped.get("kind") == "rebuild_data":
            spec = shaped
    raw = dict(spec.get("data_generation") or {})
    # Use the policy plan's legal surface. cap0_tiny is CAP0_GRAMMAR and
    # rejects simplified_nl; I10 NL waits on a CAP1 plan, not a forced flag.
    raw["data_only"] = True
    target = int(raw.get("unique_root_target") or spec.get("min_unique_roots") or 8)
    raw["unique_root_target"] = max(1, min(target, LOCAL_I10_ROOT_CAP))
    # Local CPU heal stays reliable: the plan's own minima apply here; the
    # raised component_coverage_minimum is a promotion-scale instruction.
    raw.pop("component_coverage_minimum", None)
    generation = DataGenerationKnobs.model_validate(raw)
    return [
        sys.executable,
        "-m",
        "scripts.build_train_data",
        "--source",
        "programspec",
        "--version",
        train_version,
        "--immutable",
        # The heal snapshot lives under outputs/; publishing into tracked
        # resources/ mid-cycle parks the loop as foreign_dirty_tree.
        "--no-publish",
        *_data_generation_flags(generation),
    ]


def screening_regime_decision(
    *,
    queue_entries: Sequence[Mapping[str, Any]] | None,
    compiler_ms_timeout: bool,
    root: Path | None = None,
    loop_id: str | None = None,
) -> ThrashRegimeDecision:
    """Decide isolate / climb / timeout residual for the next screening thrash."""

    baseline_entry = select_climb_baseline_entry(queue_entries)
    climb_knobs = None
    if baseline_entry is not None:
        raw = baseline_entry.get("knobs")
        if isinstance(raw, dict) and raw:
            climb_knobs = lever_knobs(raw)
    champion_ok = False
    if root is not None and loop_id:
        loop_dir = root / "loops" / loop_id
        sidecar = load_climb_champion(loop_dir)
        champion_ok = climb_champion_checkpoint_path(loop_dir).is_file()
        if climb_knobs is None and sidecar is not None and sidecar.knobs:
            climb_knobs = lever_knobs(sidecar.knobs)
    return decide_screening_regime(
        climb_baseline_knobs=climb_knobs,
        compiler_ms_timeout=compiler_ms_timeout,
        climb_champion_available=champion_ok,
    )


def screening_enqueue_allowed(
    *, cycle_intent: str, replay: dict[str, Any] | None
) -> bool:
    """Preserve screening queue semantics across an exact frozen retry."""

    if cycle_intent in {"screening", "promotion"}:
        return True
    return bool(
        cycle_intent == "retry_measurement"
        and replay is not None
        and replay["handoff"].cycle_role == "screening"
        and getattr(replay["handoff"], "cycle_intent", None) != "confirm"
    )
