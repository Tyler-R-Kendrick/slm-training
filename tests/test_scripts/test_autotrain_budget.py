"""Budget arithmetic, tested without standing up a training cycle.

These rules previously lived inside the 18,665-line continuous-autotrain
runner and could only be exercised through it. Extracted into
``scripts/autotrain_budget.py`` they are pure functions, so the arithmetic that
decides how long an arm may run is now checked directly.

Contract: ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import subprocess
import time

import pytest

from tests.casefiles import case_values

from scripts import autotrain_budget as budget


def test_nearest_rank_p95_takes_an_observed_value_not_an_interpolation() -> None:
    assert budget.nearest_rank_p95([float(n) for n in range(1, 11)]) == 10.0
    assert budget.nearest_rank_p95([5.0]) == 5.0


def test_nearest_rank_p95_of_nothing_is_undefined() -> None:
    assert budget.nearest_rank_p95([]) is None


def test_nearest_rank_p95_ignores_non_numeric_entries() -> None:
    assert budget.nearest_rank_p95([1.0, "x", None, 2.0]) == 2.0  # type: ignore[list-item]


def test_formal_lane_splits_the_wall_three_ways_not_two() -> None:
    """A required formal stage is a third claimant on the same wall."""

    two = budget.arm_wall_minutes(1e9, formal_required=False)
    three = budget.arm_wall_minutes(1e9, formal_required=True)
    assert three < two
    assert three == pytest.approx(two * 2 / 3)


def test_policy_minutes_cap_the_symmetric_share() -> None:
    assert budget.arm_wall_minutes(0.1, formal_required=False) == 0.1


def test_arm_wall_seconds_is_minutes_scaled() -> None:
    minutes = budget.arm_wall_minutes(0.5, formal_required=True)
    seconds = budget.arm_wall_seconds(policy_minutes=0.5, formal_required=True)
    assert seconds == pytest.approx(minutes * 60.0)


def test_remaining_timeout_without_a_deadline_is_the_run_cap() -> None:
    assert budget.remaining_timeout(None) == float(budget.MAX_RUN_SECONDS)


def test_remaining_timeout_never_exceeds_the_run_cap() -> None:
    far = time.monotonic() + 10 * float(budget.MAX_RUN_SECONDS)
    assert budget.remaining_timeout(far) == float(budget.MAX_RUN_SECONDS)


def test_a_passed_deadline_raises_rather_than_returning_negative_budget() -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        budget.remaining_timeout(time.monotonic() - 1.0)


def test_symmetric_budget_rejects_a_non_positive_arm_count() -> None:
    with pytest.raises(ValueError, match="arm_count must be positive"):
        budget.fit_symmetric_arm_budget(
            deadline=time.monotonic() + 600,
            arm_count=0,
            requested_arm_wall_minutes=1.0,
        )


def test_symmetric_budget_shrinks_the_request_to_what_remains() -> None:
    fitted = budget.fit_symmetric_arm_budget(
        deadline=time.monotonic() + 120,
        arm_count=4,
        requested_arm_wall_minutes=60.0,
    )
    assert 0 < fitted < 60.0


def test_arm_deadline_preserves_the_finalization_reserve() -> None:
    cycle_deadline = time.monotonic() + 10_000
    fitted = budget.arm_execution_deadline(
        cycle_deadline=cycle_deadline, arm_wall_minutes=10_000
    )
    assert fitted == pytest.approx(
        cycle_deadline - budget.HARNESS_FINALIZATION_RESERVE_SECONDS
    )


def test_cold_start_uses_the_measured_prior_and_matches_its_evidence() -> None:
    """The recorded prior must still produce the step count it documents."""

    evidence = budget.COLD_START_STEPS_PER_SEC_EVIDENCE
    steps, detail = budget.fit_screening_steps(
        floor_seconds=100.0, measured_steps_per_sec=None, steps_max=400
    )
    assert steps == evidence["expected_cold_steps_at_100s_floor"]
    assert detail["steps_per_sec_source"] == "cold_start_prior"
    assert detail["cold_start"] is True


def test_measured_telemetry_replaces_the_cold_start_prior() -> None:
    steps, detail = budget.fit_screening_steps(
        floor_seconds=10.0, measured_steps_per_sec=10.0, steps_max=400
    )
    assert detail["steps_per_sec_source"] == "train_telemetry"
    assert detail["cold_start"] is False
    # Well under steps_max, so the fit is the rate, not the cap.
    assert steps == int(10.0 * 10.0 * budget.STEPS_PER_SEC_SAFETY)


def test_a_larger_floor_buys_steps_only_up_to_the_cap() -> None:
    """Policy invariant: a larger floor buys more steps, never a larger model."""

    steps, _ = budget.fit_screening_steps(
        floor_seconds=1e9, measured_steps_per_sec=100.0, steps_max=400
    )
    assert steps == 400


def test_fitted_steps_never_fall_below_one() -> None:
    steps, _ = budget.fit_screening_steps(
        floor_seconds=0.0, measured_steps_per_sec=1.0, steps_max=400
    )
    assert steps == 1


def test_steps_per_sec_prefers_elapsed_wall_then_total_ms() -> None:
    assert budget.steps_per_sec_from_train_payload(
        {"steps": 100, "elapsed_wall_seconds": 10.0}
    ) == 10.0
    assert budget.steps_per_sec_from_train_payload(
        {"steps": 100, "total_ms": 10_000}
    ) == 10.0


@pytest.mark.parametrize(
    "payload",
    case_values(__file__, "test_unusable_telemetry_yields_no_rate"),
)
def test_unusable_telemetry_yields_no_rate(payload: dict) -> None:
    assert budget.steps_per_sec_from_train_payload(payload) is None


def test_thrash_steps_max_falls_back_to_the_default() -> None:
    default = budget.SCREENING_THRASH_STEPS_MAX_DEFAULT
    assert budget.screening_thrash_steps_max(None) == default
    assert budget.screening_thrash_steps_max({}) == default
    assert budget.screening_thrash_steps_max({"screening_thrash_steps_max": "x"}) == default


def test_thrash_steps_max_honours_a_policy_value() -> None:
    assert budget.screening_thrash_steps_max({"screening_thrash_steps_max": 25}) == 25


def test_thrash_timing_block_tolerates_a_malformed_policy() -> None:
    class Policy:
        measurement = {"thrash_timing": {"screening_thrash_steps_max": 7}}

    assert budget.thrash_timing_block(Policy()) == {"screening_thrash_steps_max": 7}
    assert budget.thrash_timing_block(object()) == {}
