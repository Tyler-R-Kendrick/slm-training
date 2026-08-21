"""screening_sample_size/v1: certified screening-n range for the climb loop."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from slm_training.autoresearch.power import min_attainable_n, required_n_for_effect
from slm_training.autoresearch.schemas import DEFAULT_ALLOWED_KNOBS
from slm_training.autoresearch.power import min_detectable_effect
from slm_training.autoresearch.screening_sample_size import (
    FINDING_RANGE_EMPTY,
    FROZEN_EVAL_SNAPSHOTS,
    FrozenEvalSnapshotError,
    SCREENING_SAMPLE_SIZE_SCHEMA,
    SCREENING_SMOKE6_EVAL_VERSION,
    ScreeningSampleSizeObservation,
    TARGET_SMOKE_N,
    assert_eval_publish_target_writable,
    compute_screening_sample_size,
    extra_smoke_fixtures_for_deficit,
)


def _obs(**overrides: object) -> ScreeningSampleSizeObservation:
    base = {
        "suite_records": 24,
        "arm_wall_seconds": 70,
        "min_train_floor_seconds": 20,
        "suite_overhead_seconds": 8,
        "per_record_decode_floor_seconds": 2,
    }
    base.update(overrides)
    return ScreeningSampleSizeObservation(**base)


def test_extra_smoke_fixtures_cover_deficit_without_duplicates() -> None:
    extras = extra_smoke_fixtures_for_deficit(existing_ids=set(), need=3)
    assert len(extras) == 3
    assert {row["id"] for row in extras} == {
        "smoke_tabs_01",
        "smoke_form_01",
        "smoke_switch_01",
    }
    used = {row["id"] for row in extras}
    more = extra_smoke_fixtures_for_deficit(existing_ids=set(used), need=3)
    assert more
    assert used.isdisjoint({row["id"] for row in more})
    exhausted = extra_smoke_fixtures_for_deficit(
        existing_ids={row["id"] for row in extras + more}, need=0
    )
    assert exhausted == []


def test_extra_smoke_fixtures_can_fill_target_24() -> None:
    seeded = {
        "smoke_hero_01",
        "smoke_button_01",
        "smoke_callout_01",
        "smoke_tabs_01",
        "smoke_form_01",
        "smoke_switch_01",
    }
    extras = extra_smoke_fixtures_for_deficit(
        existing_ids=set(seeded), need=TARGET_SMOKE_N - len(seeded)
    )
    assert len(seeded) + len(extras) >= TARGET_SMOKE_N
    assert seeded.isdisjoint({row["id"] for row in extras})


def test_latency_probe_knobs_are_allowed() -> None:
    assert "latency_probe_records" in DEFAULT_ALLOWED_KNOBS
    assert "latency_probe_planned_n" in DEFAULT_ALLOWED_KNOBS


def test_decidability_floor_matches_power_module_exact_search() -> None:
    for alpha, expected in (("1/4", 3), ("1/10", 5), ("1/20", 6)):
        report = compute_screening_sample_size(_obs(alpha=alpha))
        assert report.decidability_floor_n == expected
        assert report.decidability_floor_n == min_attainable_n(
            float(Fraction(alpha)), paired=True
        )


def test_budget_ceiling_arithmetic() -> None:
    assert (
        compute_screening_sample_size(_obs(per_record_decode_floor_seconds=14))
        .budget_ceiling_n
        == 3
    )
    assert compute_screening_sample_size(_obs()).budget_ceiling_n == 21
    assert (
        compute_screening_sample_size(_obs(arm_wall_seconds=10)).budget_ceiling_n == 0
    )


def test_feasible_range_climbs_at_smallest_sufficient_n() -> None:
    report = compute_screening_sample_size(_obs())
    assert report.verdict == "feasible"
    assert report.n_min == 6
    assert report.chosen_n == 6
    assert report.n_max == 21
    assert report.binding_constraints == ()
    assert report.findings == ()


def test_today_fixture_ceiling_is_suite_bound() -> None:
    """The committed 3-record smoke suite makes the certified range empty."""

    report = compute_screening_sample_size(_obs(suite_records=3))
    assert report.verdict == "infeasible_range_empty"
    assert report.n_min == 6
    assert report.chosen_n is None
    assert "suite_volume" in report.binding_constraints
    finding = next(f for f in report.findings if f["code"] == FINDING_RANGE_EMPTY)
    assert finding["authority"] == "climb_signal_not_gate"
    assert report.must_generate is True
    assert "do not screen" in finding["suggestion"]


def test_wall_budget_binding_when_decode_is_expensive() -> None:
    report = compute_screening_sample_size(
        _obs(suite_records=24, per_record_decode_floor_seconds=14)
    )
    assert report.verdict == "infeasible_range_empty"
    assert report.budget_ceiling_n == 3
    assert report.binding_constraints == ("wall_budget",)
    assert report.must_generate is False


def test_both_axes_binding() -> None:
    report = compute_screening_sample_size(
        _obs(suite_records=3, per_record_decode_floor_seconds=14)
    )
    assert report.verdict == "infeasible_range_empty"
    assert set(report.binding_constraints) == {"wall_budget", "suite_volume"}


def test_insufficient_evidence_without_decode_observation() -> None:
    report = compute_screening_sample_size(
        _obs(per_record_decode_floor_seconds=None)
    )
    assert report.verdict == "insufficient_evidence"
    assert report.budget_ceiling_n is None
    assert report.chosen_n is None
    assert report.n_min == 6  # the exact floor is still computed


def test_power_floor_mde_is_monotone_decreasing_in_n() -> None:
    sd = 0.1741
    mdes = [
        min_detectable_effect(n, sd, 0.05, 0.8, paired=True)
        for n in (6, 12, 24, 48)
    ]
    assert mdes == sorted(mdes, reverse=True)
    floors = [
        compute_screening_sample_size(
            _obs(minimum_effect=str(mdes[0]), observed_sd=str(sd))
        ).power_floor_n
    ]
    # Larger declared effect cannot raise the power floor.
    smaller = compute_screening_sample_size(
        _obs(minimum_effect=str(mdes[-1]), observed_sd=str(sd))
    )
    larger = compute_screening_sample_size(
        _obs(minimum_effect=str(mdes[0]), observed_sd=str(sd))
    )
    assert smaller.power_floor_n is not None
    assert larger.power_floor_n is not None
    assert smaller.power_floor_n >= larger.power_floor_n
    assert floors[0] == larger.power_floor_n


def test_chosen_n_is_max_of_decidability_and_power_floors() -> None:
    report = compute_screening_sample_size(
        _obs(minimum_effect="1/100", observed_sd="1/10", suite_records=64)
    )
    assert report.decidability_floor_n == 6
    assert report.power_floor_n is not None
    assert report.n_min == max(report.decidability_floor_n, report.power_floor_n)


def test_publish_refuses_frozen_eval_dirs() -> None:
    with pytest.raises(FrozenEvalSnapshotError):
        assert_eval_publish_target_writable(SCREENING_SMOKE6_EVAL_VERSION)
    for frozen in FROZEN_EVAL_SNAPSHOTS:
        with pytest.raises(FrozenEvalSnapshotError):
            assert_eval_publish_target_writable(frozen)
    assert_eval_publish_target_writable("e938_role_safe_all_targets_smoke24_v1")


def test_power_floor_is_assumption_backed_and_dominates() -> None:
    report = compute_screening_sample_size(
        _obs(minimum_effect="1/100", observed_sd="1/10")
    )
    expected = required_n_for_effect(0.01, 0.1, 0.05, paired=True)
    assert report.power_floor_n == expected
    assert report.n_min == max(6, expected)
    bound = next(b for b in report.bounds if b.bound_ast_id.startswith("power."))
    assert bound.authority == "assumption_backed"
    exact = next(
        b for b in report.bounds if b.bound_ast_id.startswith("bound.screening_n")
    )
    assert exact.authority == "theorem_backed_exact"


def test_floor_beyond_search_cap_fails_closed() -> None:
    report = compute_screening_sample_size(_obs(max_candidate_n=4))
    assert report.decidability_floor_n is None
    assert report.verdict == "infeasible_range_empty"
    assert report.chosen_n is None
    assert any(
        f["code"] == "screening_n_floor_beyond_search_cap" for f in report.findings
    )


def test_strict_validation() -> None:
    with pytest.raises(ValidationError):
        _obs(alpha="0")
    with pytest.raises(ValidationError):
        _obs(alpha="1")
    with pytest.raises(ValidationError):
        _obs(minimum_effect="1/100")  # effect without sd
    with pytest.raises(ValidationError):
        _obs(observed_sd="1/10")  # sd without effect
    with pytest.raises(ValidationError):
        _obs(unknown_field=1)


def test_report_envelope() -> None:
    report = compute_screening_sample_size(_obs())
    assert report.schema_version == SCREENING_SAMPLE_SIZE_SCHEMA
    assert report.promotion_authority is False
    digest = report.certificate_sha256()
    assert len(digest) == 64
    payload = report.model_dump(mode="json")
    assert payload["alpha"] == "1/20"
