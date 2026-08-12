"""RESEARCH-13 harness tests."""

from __future__ import annotations

from slm_training.harnesses.experiments.research_13_dropped_assumption_counterexample import (
    lock_campaign,
    run_experiment,
)


def test_default_off_skips() -> None:
    result = run_experiment(enabled=False)
    assert result["executed"] is False


def test_lock_is_stable() -> None:
    assert lock_campaign().manifest_sha256 == lock_campaign().manifest_sha256
