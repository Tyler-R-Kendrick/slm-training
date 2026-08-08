"""Tests for the PCT-003 honest cold/warm benchmark harness."""

from __future__ import annotations

import dataclasses
import sys

import pytest

from slm_training.harnesses.model_build.cold_warm_bench import (
    BENCH_TRIAL_SCHEMA,
    BenchTrialV1,
    HardwareIdentityV1,
    phase_marker,
    run_cold_trial,
    run_warm_trial,
    summarize_trials,
    trial_integrity_ok,
)

_COLD_SCRIPT = (
    "import time;"
    "from slm_training.harnesses.model_build.cold_warm_bench import phase_marker;"
    "print(phase_marker('process_launch'));"
    "time.sleep(0.01);"
    "print(phase_marker('bootstrap'));"
    "time.sleep(0.01);"
    "print(phase_marker('first_candidate'))"
)

_SLOW_SCRIPT = "import time; time.sleep(5)"

_ABORT_SCRIPT = "import sys; sys.exit(3)"


def test_cold_trial_spawns_a_real_fresh_subprocess() -> None:
    trial = run_cold_trial(
        [sys.executable, "-c", _COLD_SCRIPT], config={"arm": "control"}
    )
    assert trial.schema_version == BENCH_TRIAL_SCHEMA
    assert trial.trial_kind == "cold"
    assert trial.measurement_stage == "process_cold"
    assert trial.completeness == "complete"
    assert trial.exit_code == 0
    assert [p.phase for p in trial.phases] == [
        "process_launch",
        "bootstrap",
        "first_candidate",
    ]
    # First phase has no predecessor marker, so its delta is unmeasured (None);
    # later phases measure real elapsed time between markers.
    assert trial.phases[0].duration_ms is None
    assert trial.phases[1].duration_ms is not None
    assert trial.phases[1].duration_ms >= 0
    assert trial.total_ms is not None and trial.total_ms > 0


def test_cold_trial_timeout_marks_partial_not_complete() -> None:
    trial = run_cold_trial(
        [sys.executable, "-c", _SLOW_SCRIPT], config={"arm": "control"}, timeout_s=0.2
    )
    assert trial.completeness == "partial_timeout"
    assert trial.completeness != "complete"


def test_cold_trial_nonzero_exit_marks_aborted() -> None:
    trial = run_cold_trial([sys.executable, "-c", _ABORT_SCRIPT], config={})
    assert trial.completeness == "aborted"
    assert trial.exit_code == 3


def test_warm_trial_is_labeled_separately_from_cold() -> None:
    def _fn() -> tuple[dict, ...]:
        return (
            {"phase": "request_start", "t": 0.0},
            {"phase": "response_ready", "t": 0.002},
        )

    warm = run_warm_trial(_fn, config={"arm": "control"})
    assert warm.trial_kind == "warm"
    assert warm.measurement_stage == "steady_state"
    assert warm.measurement_stage != "process_cold"
    assert warm.completeness == "complete"


def test_warm_trial_raising_callable_is_aborted_not_fabricated() -> None:
    def _boom() -> tuple[dict, ...]:
        raise RuntimeError("warm callable failed")

    warm = run_warm_trial(_boom, config={})
    assert warm.completeness == "aborted"
    assert warm.phases == ()
    assert warm.total_ms is None


def test_hardware_identity_is_persisted() -> None:
    identity = HardwareIdentityV1.capture()
    assert identity.python_version
    assert identity.platform
    assert identity.to_dict()["python_version"] == identity.python_version

    trial = run_warm_trial(lambda: (), config={})
    assert trial.hardware_identity["python_version"]


def test_matched_config_hash_separates_arms_in_summary() -> None:
    control = run_warm_trial(lambda: (), config={"artifact_enabled": False})
    treated = run_warm_trial(lambda: (), config={"artifact_enabled": True})
    assert control.config_hash != treated.config_hash

    summary = summarize_trials([control, treated])
    assert len(summary["arms"]) == 2
    hashes = {arm["config_hash"] for arm in summary["arms"].values()}
    assert hashes == {control.config_hash, treated.config_hash}


def test_summary_excludes_incomplete_trials() -> None:
    complete = run_warm_trial(lambda: (), config={"arm": "x"})
    aborted = run_warm_trial(
        lambda: (_ for _ in ()).throw(RuntimeError("x")), config={"arm": "x"}
    )
    summary = summarize_trials([complete, aborted])
    assert summary["excluded_incomplete"] == 1
    (arm,) = summary["arms"].values()
    assert arm["n"] == 1


def test_trial_is_frozen_and_deterministic_schema_roundtrip() -> None:
    trial = run_warm_trial(lambda: (), config={"arm": "x"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        trial.completeness = "complete"  # type: ignore[misc]

    assert trial_integrity_ok(trial)
    assert BenchTrialV1.from_dict(trial.to_dict()) == trial


def test_trial_tamper_and_schema_rejection() -> None:
    trial = run_warm_trial(lambda: (), config={})

    tampered = trial.to_dict()
    tampered["completeness"] = "aborted" if trial.completeness != "aborted" else "complete"
    with pytest.raises(ValueError, match="hash mismatch"):
        BenchTrialV1.from_dict(tampered)

    wrong_schema = trial.to_dict()
    wrong_schema["schema_version"] = "cold_warm_bench_trial/v0"
    with pytest.raises(ValueError, match="unsupported cold/warm bench trial schema"):
        BenchTrialV1.from_dict(wrong_schema)


def test_phase_marker_round_trips_through_json() -> None:
    import json

    line = phase_marker("model_load")
    payload = json.loads(line.split(" ", 1)[1])
    assert payload["phase"] == "model_load"
    assert isinstance(payload["t"], float)
