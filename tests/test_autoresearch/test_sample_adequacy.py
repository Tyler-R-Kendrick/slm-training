"""Sample-size adequacy bounds: certified band + climb-signal verdicts."""

from __future__ import annotations

from fractions import Fraction

import pytest

from slm_training.autoresearch.climb_policy import (
    ClimbPolicyError,
    data_intervention_indicated,
    load_climb_policy,
    sample_adequacy_intervention,
)
from slm_training.autoresearch.sample_adequacy import (
    FINDING_ABOVE_CEILING,
    FINDING_BELOW_FLOOR,
    FINDING_COVERAGE_GAP,
    SampleAdequacyObservation,
    compute_sample_adequacy,
    mean_description_bits,
)
from slm_training.formal.bound_ast import (
    BOUND_SAMPLE_SIZE_CAPACITY_UPPER,
    BOUND_SAMPLE_SIZE_COVERAGE_LOWER,
    evaluate_bound,
)
from slm_training.harnesses.train_data.feedback import (
    _EXECUTABLE_KNOBS,
    FINDING_CODES,
)


def _obs(**overrides):
    base = dict(
        observed_records=101,
        component_witnesses={"container": 50, "rare_slot": 2},
        trainable_params=1_000_000,
        mean_example_description_bits=3200,
    )
    base.update(overrides)
    return SampleAdequacyObservation(**base)


def test_registry_bounds_match_lean_guard_anchors() -> None:
    cover = evaluate_bound(
        BOUND_SAMPLE_SIZE_COVERAGE_LOWER,
        {"witnesses_target": 4, "observed_records": 101, "min_witness_count": 2},
    )
    assert cover.value == "202"
    cap = evaluate_bound(
        BOUND_SAMPLE_SIZE_CAPACITY_UPPER,
        {"trainable_params": 1000, "bits_per_param": 6, "bits_per_example": 60},
    )
    assert cap.value == "100"


def test_generate_more_below_certified_floor() -> None:
    report = compute_sample_adequacy(_obs())
    assert report.verdict == "generate_more"
    assert report.coverage_floor_records == 202
    assert report.recommended_records == 202
    assert report.min_witness_count == 2
    codes = [f["code"] for f in report.findings]
    assert codes == [FINDING_BELOW_FLOOR]
    assert all(f["authority"] == "climb_signal_not_gate" for f in report.findings)
    assert report.promotion_authority is False


def test_sufficient_inside_band() -> None:
    report = compute_sample_adequacy(
        _obs(component_witnesses={"container": 50, "rare_slot": 8})
    )
    assert report.coverage_floor_records == 51
    assert report.verdict == "sufficient"
    assert report.findings == ()


def test_saturated_above_capacity_ceiling() -> None:
    report = compute_sample_adequacy(
        _obs(
            component_witnesses={"container": 50, "rare_slot": 8},
            trainable_params=1000,
            mean_example_description_bits=60,
        )
    )
    assert report.capacity_ceiling_records_high_prior == 100
    assert report.verdict == "saturated_change_trajectory"
    assert [f["code"] for f in report.findings] == [FINDING_ABOVE_CEILING]


def test_conflict_floor_exceeds_ceiling() -> None:
    report = compute_sample_adequacy(
        _obs(trainable_params=1000, mean_example_description_bits=60)
    )
    assert report.coverage_floor_records == 202
    assert report.capacity_ceiling_records_high_prior == 100
    assert report.verdict == "conflict_change_trajectory"


def test_zero_witness_component_blocks_volume_answer() -> None:
    report = compute_sample_adequacy(
        _obs(component_witnesses={"container": 50, "never_seen": 0})
    )
    assert report.verdict == "coverage_gap_blocked"
    assert report.zero_witness_components == ("never_seen",)
    assert [f["code"] for f in report.findings] == [FINDING_COVERAGE_GAP]


def test_insufficient_evidence_without_records_or_witnesses() -> None:
    empty = compute_sample_adequacy(
        _obs(observed_records=0, component_witnesses={})
    )
    assert empty.verdict == "insufficient_evidence"
    assert empty.coverage_floor_records is None


def test_bounds_recalibrate_with_new_evidence() -> None:
    scarce = compute_sample_adequacy(_obs())
    richer_rate = compute_sample_adequacy(
        _obs(component_witnesses={"container": 50, "rare_slot": 8})
    )
    assert richer_rate.coverage_floor_records < scarce.coverage_floor_records
    small = compute_sample_adequacy(_obs(trainable_params=1000))
    big = compute_sample_adequacy(_obs(trainable_params=2000))
    assert (
        big.capacity_ceiling_records_high_prior
        > small.capacity_ceiling_records_high_prior
    )


def test_generator_ceiling_caps_recommendation() -> None:
    report = compute_sample_adequacy(_obs(reachable_unique_roots=150))
    assert report.verdict == "generate_more"
    assert report.generator_ceiling_limits_floor is True
    assert report.recommended_records == 150
    finding = report.findings[0]
    assert finding["evidence"]["generator_ceiling_limits_floor"] is True


def test_certificate_is_deterministic() -> None:
    a = compute_sample_adequacy(_obs())
    b = compute_sample_adequacy(_obs())
    assert a.certificate_sha256() == b.certificate_sha256()


def test_mean_description_bits_exact() -> None:
    assert mean_description_bits(1010, 101) == Fraction(80)
    with pytest.raises(ValueError):
        mean_description_bits(0, 101)
    with pytest.raises(ValueError):
        mean_description_bits(1010, 0)


def test_intervention_generate_more_compiles_rebuild_data() -> None:
    policy = load_climb_policy()
    report = compute_sample_adequacy(_obs()).model_dump(mode="json")
    action = sample_adequacy_intervention(policy, report)
    assert action is not None
    assert action["kind"] == "rebuild_data"
    assert action["owner"] == "sample-adequacy"
    assert action["data_generation"]["unique_root_target"] == 202
    assert action["promotion_authorized"] is False
    assert action["sample_adequacy"]["verdict"] == "generate_more"
    assert action["sample_adequacy"]["bounds"]


def test_intervention_trajectory_and_none_paths() -> None:
    policy = load_climb_policy()
    saturated = compute_sample_adequacy(
        _obs(
            component_witnesses={"container": 50, "rare_slot": 8},
            trainable_params=1000,
            mean_example_description_bits=60,
        )
    ).model_dump(mode="json")
    action = sample_adequacy_intervention(policy, saturated)
    assert action["kind"] == "change_trajectory"
    assert action["close_approach"] == "data_volume_at_current_trajectory"
    assert "charged_capacity_growth_EG_params" in action["successor_axes"]

    gap = compute_sample_adequacy(
        _obs(component_witnesses={"container": 50, "never_seen": 0})
    ).model_dump(mode="json")
    gap_action = sample_adequacy_intervention(policy, gap)
    assert gap_action["kind"] == "change_trajectory"
    assert tuple(gap_action["successor_axes"]) == ("generator_coverage_widening",)

    sufficient = compute_sample_adequacy(
        _obs(component_witnesses={"container": 50, "rare_slot": 8})
    ).model_dump(mode="json")
    assert sample_adequacy_intervention(policy, sufficient) is None

    with pytest.raises(ClimbPolicyError):
        sample_adequacy_intervention(policy, {"schema_version": "other/v1"})


def test_data_intervention_indicated_consumes_adequacy_verdict() -> None:
    policy = load_climb_policy()
    common = dict(feedback=None, unique_roots=None, blocking_findings=())
    assert data_intervention_indicated(
        policy, **common, sample_adequacy={"verdict": "generate_more"}
    )
    assert not data_intervention_indicated(
        policy, **common, sample_adequacy={"verdict": "sufficient"}
    )
    assert not data_intervention_indicated(policy, **common)


def test_adequacy_finding_codes_are_registered_vocabulary() -> None:
    assert FINDING_BELOW_FLOOR in FINDING_CODES
    assert FINDING_ABOVE_CEILING in FINDING_CODES
    assert FINDING_COVERAGE_GAP in FINDING_CODES
    assert "data_generation.unique_root_target" in _EXECUTABLE_KNOBS[
        FINDING_BELOW_FLOOR
    ]
    assert "max_records_per_parent" in _EXECUTABLE_KNOBS[FINDING_ABOVE_CEILING]
