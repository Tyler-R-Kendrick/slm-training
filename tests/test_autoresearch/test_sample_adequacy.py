"""Sample-size adequacy v2: per-component coverage + measured saturation."""

from __future__ import annotations

import pytest

from slm_training.autoresearch.climb_policy import (
    ClimbPolicyError,
    load_climb_policy,
    sample_adequacy_intervention,
)
from slm_training.autoresearch.engine import _data_generation_flags
from slm_training.autoresearch.sample_adequacy import (
    FINDING_ABOVE_CEILING,
    FINDING_BELOW_FLOOR,
    SampleAdequacyObservation,
    compute_sample_adequacy,
    observation_from_train_stats,
)
from slm_training.autoresearch.schemas import DataGenerationKnobs
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


def test_generate_more_reports_per_component_deficits() -> None:
    report = compute_sample_adequacy(_obs())
    assert report.verdict == "generate_more"
    assert report.coverage_deficits == {"rare_slot": 2}
    assert report.coverage_floor_records == 202
    assert report.recommended_records == 202
    finding = report.findings[0]
    assert finding["code"] == FINDING_BELOW_FLOOR
    assert finding["authority"] == "climb_signal_not_gate"
    assert "until_coverage" in finding["suggestion"]
    assert report.promotion_authority is False


def test_zero_witness_components_are_targeted_not_blocked() -> None:
    report = compute_sample_adequacy(
        _obs(component_witnesses={"container": 50, "never_seen": 0})
    )
    assert report.verdict == "generate_more"
    assert report.zero_witness_components == ("never_seen",)
    assert report.coverage_deficits["never_seen"] == 4


def test_sufficient_when_every_component_meets_witness_target() -> None:
    report = compute_sample_adequacy(
        _obs(component_witnesses={"container": 50, "rare_slot": 8})
    )
    assert report.verdict == "sufficient"
    assert report.coverage_deficits == {}
    assert report.coverage_floor_records is None
    assert report.findings == ()


def test_saturation_requires_measured_flat_marginal_gain() -> None:
    covered = dict(component_witnesses={"container": 50, "rare_slot": 8})
    unmeasured = compute_sample_adequacy(_obs(**covered))
    assert unmeasured.verdict == "sufficient"

    rising = compute_sample_adequacy(
        _obs(
            **covered,
            marginal_gain_flat=False,
            marginal_gain_source="outputs/ladders/x/data_adequacy_ladder.json",
        )
    )
    assert rising.verdict == "sufficient"

    flat = compute_sample_adequacy(
        _obs(
            **covered,
            marginal_gain_flat=True,
            marginal_gain_source="outputs/ladders/x/data_adequacy_ladder.json",
        )
    )
    assert flat.verdict == "saturated_change_trajectory"
    assert [f["code"] for f in flat.findings] == [FINDING_ABOVE_CEILING]


def test_coverage_outranks_flat_marginal_gain() -> None:
    report = compute_sample_adequacy(
        _obs(
            marginal_gain_flat=True,
            marginal_gain_source="outputs/ladders/x/data_adequacy_ladder.json",
        )
    )
    assert report.verdict == "generate_more"


def test_marginal_gain_requires_source() -> None:
    with pytest.raises(ValueError, match="marginal_gain_source"):
        _obs(marginal_gain_flat=True)


def test_capacity_prior_is_diagnostic_never_a_verdict() -> None:
    report = compute_sample_adequacy(
        _obs(
            component_witnesses={"container": 50, "rare_slot": 8},
            trainable_params=1000,
            mean_example_description_bits=60,
        )
    )
    # 101 records exceed the generous-prior ceiling of 100, yet the verdict
    # stays sufficient: exceeding memorization capacity is not a stop signal.
    assert report.capacity_ceiling_records_high_prior == 100
    assert report.memorization_regime_expected is True
    assert report.verdict == "sufficient"
    assert report.findings == ()


def test_capacity_inputs_come_paired() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        _obs(trainable_params=1000)


def test_bounds_recalibrate_with_new_evidence() -> None:
    scarce = compute_sample_adequacy(_obs())
    richer = compute_sample_adequacy(
        _obs(component_witnesses={"container": 50, "rare_slot": 3})
    )
    assert richer.coverage_floor_records < scarce.coverage_floor_records


def test_generator_ceiling_caps_recommendation() -> None:
    report = compute_sample_adequacy(_obs(reachable_unique_roots=150))
    assert report.verdict == "generate_more"
    assert report.generator_ceiling_limits_floor is True
    assert report.recommended_records == 150


def test_certificate_is_deterministic() -> None:
    assert (
        compute_sample_adequacy(_obs()).certificate_sha256()
        == compute_sample_adequacy(_obs()).certificate_sha256()
    )


def test_observation_from_train_stats_reads_histogram() -> None:
    stats = {
        "record_count": 101,
        "component_histogram": {"Button": 60, "SwitchGroup": 2},
    }
    observation = observation_from_train_stats(stats)
    assert observation.observed_records == 101
    assert observation.component_witnesses == {"Button": 60, "SwitchGroup": 2}
    assert observation.trainable_params is None
    report = compute_sample_adequacy(observation)
    assert report.verdict == "generate_more"
    assert report.coverage_deficits == {"SwitchGroup": 2}


def test_intervention_compiles_targeted_until_coverage_rebuild() -> None:
    policy = load_climb_policy()
    report = compute_sample_adequacy(_obs()).model_dump(mode="json")
    action = sample_adequacy_intervention(policy, report)
    assert action is not None
    assert action["kind"] == "rebuild_data"
    generation = action["data_generation"]
    assert generation["generation_mode"] == "until_coverage"
    assert generation["component_coverage_minimum"] == 4
    assert generation["unique_root_target"] == 202
    assert action["promotion_authorized"] is False
    assert action["sample_adequacy"]["coverage_deficits"] == {"rare_slot": 2}
    # The compiled knobs stay schema-legal and CLI-compilable.
    knobs = DataGenerationKnobs.model_validate(generation)
    flags = _data_generation_flags(knobs)
    assert "--generation-mode" in flags
    assert "--component-coverage-minimum" in flags


def test_intervention_saturated_and_none_paths() -> None:
    policy = load_climb_policy()
    covered = dict(component_witnesses={"container": 50, "rare_slot": 8})
    flat = compute_sample_adequacy(
        _obs(
            **covered,
            marginal_gain_flat=True,
            marginal_gain_source="outputs/ladders/x/data_adequacy_ladder.json",
        )
    ).model_dump(mode="json")
    action = sample_adequacy_intervention(policy, flat)
    assert action["kind"] == "change_trajectory"
    assert action["close_approach"] == "data_volume_at_current_trajectory"
    assert "charged_capacity_growth_EG_params" in action["successor_axes"]
    assert action["sample_adequacy"]["marginal_gain_source"]

    sufficient = compute_sample_adequacy(_obs(**covered)).model_dump(mode="json")
    assert sample_adequacy_intervention(policy, sufficient) is None

    with pytest.raises(ClimbPolicyError):
        sample_adequacy_intervention(policy, {"schema_version": "other/v1"})


def test_adequacy_finding_codes_are_registered_vocabulary() -> None:
    assert FINDING_BELOW_FLOOR in FINDING_CODES
    assert FINDING_ABOVE_CEILING in FINDING_CODES
    below = _EXECUTABLE_KNOBS[FINDING_BELOW_FLOOR]
    assert "data_generation.component_coverage_minimum" in below
    assert "data_generation.generation_mode" in below
