"""Wall-clock and step budget arithmetic for the continuous autotrain runner.

Extracted from ``scripts/run_autotrain_continuous.py``, which had grown to
18,665 lines and 321 top-level definitions. This module owns one thing: turning
a policy and a deadline into the seconds, steps, and deadlines an arm may spend.

Nothing here runs a subprocess or touches the filesystem, so the budget rules
can be read and tested without standing up a training cycle. They are not all
*pure*: the deadline helpers read the monotonic clock, and every function reads
the wall-clock levers imported below. Those are the only two impurities.

The names are re-exported into ``run_autotrain_continuous`` under their original
private aliases, so existing call sites and test monkeypatches are unaffected.

See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import math
import subprocess
import time
from collections.abc import Mapping, Sequence
from typing import Any

from slm_training.levers import (
    HARNESS_FINALIZATION_RESERVE_SECONDS,
    MAX_HARNESS_WALL_SECONDS,
    MAX_RUN_SECONDS,
)

COLD_START_STEPS_PER_SEC = 2.9

COLD_START_STEPS_PER_SEC_EVIDENCE: dict[str, Any] = {'doc': 'docs/design/p8-screening-cold-start-steps-prior-20260902.md',
 'e53_scratch_arch': '--d-model 192 --n-heads 6 --context-layers 3 '
                     '--denoiser-layers 6 --mask-pattern mixed (E53 recipe shape; '
                     'hf context tower not installable here, so scratch context)',
 'expected_cold_steps_at_100s_floor': 261,
 'expected_measured_steps_at_100s_floor': 400,
 'host': '4 CPU, no GPU; sibling agents shared the CPUs (1-min load average at '
         'launch recorded per run)',
 'launch_shape': '--context-backend scratch --device cpu --lr 3e-4 --seed 0 '
                 '--train-version wf_smoke_v2 --local-files-only '
                 '--no-sync-checkpoints --no-full-state-checkpoint (engine.py '
                 'screening train command)',
 'max_steps_per_sec': 19.633,
 'measured_at': '2026-09-02',
 'median_steps_per_sec': 16.091,
 'min_steps_per_sec': 2.906,
 'prior_steps_per_sec': 2.9,
 'record_count': 101,
 'rule': 'prior = min(measured steps/s) rounded down to 0.1; train telemetry '
         'replaces it after the first arm',
 'runs': [{'arch': 'trainer_default_arch',
           'batch_size': 2,
           'elapsed_wall_seconds': 68.82,
           'load_avg_1m_at_launch': 4.17,
           'process_wall_seconds': 79.02,
           'run_id': 'initial_b2_s200_r1',
           'steps': 200,
           'steps_per_sec': 2.906,
           'stopped_on': 'steps',
           'trainable_params': 1608962},
          {'arch': 'trainer_default_arch',
           'batch_size': 2,
           'elapsed_wall_seconds': 10.47,
           'load_avg_1m_at_launch': 10.83,
           'process_wall_seconds': 16.06,
           'run_id': 'initial_b2_s200_r2',
           'steps': 200,
           'steps_per_sec': 19.107,
           'stopped_on': 'steps',
           'trainable_params': 1608962},
          {'arch': 'trainer_default_arch',
           'batch_size': 2,
           'elapsed_wall_seconds': 11.15,
           'load_avg_1m_at_launch': 8.25,
           'process_wall_seconds': 16.75,
           'run_id': 'initial_b2_s200_r3',
           'steps': 200,
           'steps_per_sec': 17.939,
           'stopped_on': 'steps',
           'trainable_params': 1608962},
          {'arch': 'trainer_default_arch',
           'batch_size': 3,
           'elapsed_wall_seconds': 12.68,
           'load_avg_1m_at_launch': 7.18,
           'process_wall_seconds': 18.38,
           'run_id': 'initial_b3_s200_r1',
           'steps': 200,
           'steps_per_sec': 15.767,
           'stopped_on': 'steps',
           'trainable_params': 1608962},
          {'arch': 'trainer_default_arch',
           'batch_size': 3,
           'elapsed_wall_seconds': 12.82,
           'load_avg_1m_at_launch': 5.87,
           'process_wall_seconds': 18.55,
           'run_id': 'initial_b3_s200_r2',
           'steps': 200,
           'steps_per_sec': 15.606,
           'stopped_on': 'steps',
           'trainable_params': 1608962},
          {'arch': 'trainer_default_arch',
           'batch_size': 3,
           'elapsed_wall_seconds': 12.18,
           'load_avg_1m_at_launch': 5.09,
           'process_wall_seconds': 17.93,
           'run_id': 'initial_b3_s200_r3',
           'steps': 200,
           'steps_per_sec': 16.416,
           'stopped_on': 'steps',
           'trainable_params': 1608962},
          {'arch': 'e53_scratch_arch',
           'batch_size': 2,
           'elapsed_wall_seconds': 19.92,
           'load_avg_1m_at_launch': 3.24,
           'process_wall_seconds': 25.33,
           'run_id': 'e53_b2_s200_r1',
           'steps': 200,
           'steps_per_sec': 10.04,
           'stopped_on': 'steps',
           'trainable_params': 5125058},
          {'arch': 'e53_scratch_arch',
           'batch_size': 2,
           'elapsed_wall_seconds': 19.93,
           'load_avg_1m_at_launch': 3.42,
           'process_wall_seconds': 25.95,
           'run_id': 'e53_b2_s200_r2',
           'steps': 200,
           'steps_per_sec': 10.037,
           'stopped_on': 'steps',
           'trainable_params': 5125058},
          {'arch': 'e53_scratch_arch',
           'batch_size': 2,
           'elapsed_wall_seconds': 18.52,
           'load_avg_1m_at_launch': 3.41,
           'process_wall_seconds': 24.55,
           'run_id': 'e53_b2_s200_r3',
           'steps': 200,
           'steps_per_sec': 10.798,
           'stopped_on': 'steps',
           'trainable_params': 5125058},
          {'arch': 'trainer_default_arch',
           'batch_size': 2,
           'elapsed_wall_seconds': 10.19,
           'load_avg_1m_at_launch': 2.32,
           'process_wall_seconds': 15.81,
           'run_id': 'final_b2_s200_r1',
           'steps': 200,
           'steps_per_sec': 19.633,
           'stopped_on': 'steps',
           'trainable_params': 1608962},
          {'arch': 'trainer_default_arch',
           'batch_size': 2,
           'elapsed_wall_seconds': 10.64,
           'load_avg_1m_at_launch': 2.33,
           'process_wall_seconds': 16.18,
           'run_id': 'final_b2_s200_r2',
           'steps': 200,
           'steps_per_sec': 18.801,
           'stopped_on': 'steps',
           'trainable_params': 1608962},
          {'arch': 'trainer_default_arch',
           'batch_size': 2,
           'elapsed_wall_seconds': 10.87,
           'load_avg_1m_at_launch': 2.49,
           'process_wall_seconds': 16.64,
           'run_id': 'final_b2_s200_r3',
           'steps': 200,
           'steps_per_sec': 18.398,
           'stopped_on': 'steps',
           'trainable_params': 1608962}],
 'train_dir': 'src/slm_training/resources/data/train/wf_smoke_v2',
 'trainer': 'python -m scripts.train_model'}

STEPS_PER_SEC_SAFETY = 0.9

SCREENING_THRASH_STEPS_MAX_DEFAULT = 400

ARM_BUDGET_SCHEDULE_MARGIN_SECONDS = 0.25

def remaining_timeout(deadline: float | None = None) -> float:
    if deadline is None:
        return float(MAX_RUN_SECONDS)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired("autotrain bounded cycle", MAX_RUN_SECONDS)
    return min(float(MAX_RUN_SECONDS), remaining)

def arm_wall_minutes(policy_minutes: float, *, formal_required: bool) -> float:
    """Give required stages equal room while retaining finalization."""

    stage_count = 3 if formal_required else 2
    arm_seconds = (
        float(MAX_HARNESS_WALL_SECONDS) - HARNESS_FINALIZATION_RESERVE_SECONDS
    ) / stage_count
    symmetric_minutes = arm_seconds / 60
    return min(float(policy_minutes), symmetric_minutes)

def arm_wall_seconds(*, policy_minutes: float, formal_required: bool) -> float:
    return (
        float(arm_wall_minutes(policy_minutes, formal_required=formal_required)) * 60.0
    )

def thrash_timing_block(policy: Any) -> dict[str, Any]:
    measurement = getattr(policy, "measurement", None) or {}
    if not isinstance(measurement, dict):
        return {}
    thrash = measurement.get("thrash_timing") or {}
    return dict(thrash) if isinstance(thrash, dict) else {}

def screening_thrash_steps_max(thrash: Mapping[str, Any] | None) -> int:
    raw = (thrash or {}).get("screening_thrash_steps_max")
    if raw is None:
        return int(SCREENING_THRASH_STEPS_MAX_DEFAULT)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return int(SCREENING_THRASH_STEPS_MAX_DEFAULT)

def steps_per_sec_from_train_payload(payload: Mapping[str, Any]) -> float | None:
    try:
        steps = int(payload.get("steps") or 0)
    except (TypeError, ValueError):
        return None
    wall = payload.get("elapsed_wall_seconds")
    if wall is None:
        total_ms = payload.get("total_ms")
        if total_ms is not None:
            try:
                wall = float(total_ms) / 1000.0
            except (TypeError, ValueError):
                wall = None
    try:
        wall_s = float(wall or 0.0)
    except (TypeError, ValueError):
        return None
    if steps <= 0 or wall_s <= 0.0:
        return None
    return float(steps) / wall_s

def nearest_rank_p95(values: Sequence[float]) -> float | None:
    """Nearest-rank p95 (no interpolation; censored walls count as observed)."""

    ordered = sorted(float(v) for v in values if isinstance(v, (int, float)))
    if not ordered:
        return None
    rank = max(1, min(len(ordered), math.ceil(0.95 * len(ordered))))
    return float(ordered[rank - 1])

def require_symmetric_arm_budget(
    *, deadline: float, arm_count: int, arm_wall_minutes: float
) -> None:
    required = arm_count * arm_wall_minutes * 60 + HARNESS_FINALIZATION_RESERVE_SECONDS
    remaining = remaining_timeout(deadline)
    if remaining < required:
        raise subprocess.TimeoutExpired("symmetric decision-arm budget", required)

def fit_symmetric_arm_budget(
    *, deadline: float, arm_count: int, requested_arm_wall_minutes: float
) -> float:
    """Share the post-planning budget equally while preserving finalization."""

    if arm_count <= 0:
        raise ValueError("arm_count must be positive")
    remaining = remaining_timeout(deadline)
    usable = (
        remaining
        - HARNESS_FINALIZATION_RESERVE_SECONDS
        - ARM_BUDGET_SCHEDULE_MARGIN_SECONDS
    )
    if usable <= 0:
        raise subprocess.TimeoutExpired("symmetric decision-arm budget", remaining)
    return min(float(requested_arm_wall_minutes), usable / arm_count / 60)

def arm_execution_deadline(*, cycle_deadline: float, arm_wall_minutes: float) -> float:
    """Cap one arm without spending the cycle's finalization reserve."""

    return min(
        cycle_deadline - HARNESS_FINALIZATION_RESERVE_SECONDS,
        time.monotonic() + arm_wall_minutes * 60,
    )

def fit_screening_steps(
    *,
    floor_seconds: float,
    measured_steps_per_sec: float | None,
    steps_max: int,
) -> tuple[int, dict[str, Any]]:
    """Fit train steps to the (grown) train floor handed over by the decode fit.

    ``steps = clamp(floor_seconds * sps * safety, 1, steps_max)`` where ``sps``
    is measured telemetry when available and the measured cold-start prior
    otherwise. ``steps_max`` (policy ``screening_thrash_steps_max``, 400) is
    the only cap: a larger floor buys more steps, never a larger model.
    """

    cold = measured_steps_per_sec is None or measured_steps_per_sec <= 0.0
    sps = COLD_START_STEPS_PER_SEC if cold else float(measured_steps_per_sec)
    raw = float(floor_seconds) * float(sps) * float(STEPS_PER_SEC_SAFETY)
    fitted = max(1, min(int(raw), int(steps_max)))
    evidence = {
        "floor_seconds": float(floor_seconds),
        "measured_steps_per_sec": None if cold else float(measured_steps_per_sec),
        "steps_per_sec_used": float(sps),
        "steps_per_sec_source": "cold_start_prior" if cold else "train_telemetry",
        "cold_start_prior_steps_per_sec": float(COLD_START_STEPS_PER_SEC),
        "cold_start_prior_evidence": dict(COLD_START_STEPS_PER_SEC_EVIDENCE),
        "safety": float(STEPS_PER_SEC_SAFETY),
        "raw_steps": float(raw),
        "fitted_steps": int(fitted),
        "steps_max": int(steps_max),
        "cold_start": bool(cold),
    }
    return int(fitted), evidence
