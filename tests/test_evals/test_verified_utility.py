"""Tests for the SLM-186 verified-utility ladder helpers."""

from __future__ import annotations

import math

import pytest

from slm_training.evals.verified_utility import (
    FACTOR_NAMES,
    UtilityWeightManifestV1,
    VerifiedUtilityV1,
    abstention_economics,
    canonical_equivalent_utility,
    cvar_tail,
    lexicographic_score,
    pareto_dominance,
    pareto_front,
    safe_ratio,
    scalarized_score,
    sensitivity_rank_reversals,
)


def _util(**kwargs: float | bool | str | tuple[str, ...]) -> VerifiedUtilityV1:
    """Build a utility with availability inferred from kwargs."""
    availability = {name: "available" if name in kwargs else "unavailable" for name in FACTOR_NAMES}
    # hard_valid and support_status are not numeric factors; still mark them.
    if "hard_valid" in kwargs:
        availability["hard_valid"] = "available"
    if "support_status" in kwargs:
        availability["support_status"] = "available"
    return VerifiedUtilityV1(availability=availability, **kwargs)


def test_schema_factor_names_match_dataclass() -> None:
    util = VerifiedUtilityV1()
    for name in FACTOR_NAMES:
        assert hasattr(util, name)
    assert set(util.availability) == set(FACTOR_NAMES)


def test_to_dict_round_trip() -> None:
    util = _util(
        hard_valid=True,
        support_status="supported",
        contract_coverage=0.9,
        binding_aware_meaningful_v2=0.85,
    )
    data = util.to_dict()
    restored = VerifiedUtilityV1.from_dict(data)
    assert restored.hard_valid is util.hard_valid
    assert restored.support_status == util.support_status
    assert restored.contract_coverage == pytest.approx(util.contract_coverage)
    assert restored.availability == util.availability


def test_lexicographic_score_orders_by_policy() -> None:
    high = _util(binding_aware_meaningful_v2=0.9, component_role_recall=0.5)
    low = _util(binding_aware_meaningful_v2=0.5, component_role_recall=0.9)
    policy = ["binding_aware_meaningful_v2", "component_role_recall"]
    high_score = lexicographic_score(high, policy)
    low_score = lexicographic_score(low, policy)
    assert high_score["rank_vector"] > low_score["rank_vector"]


def test_lexicographic_score_penalizes_not_hard_valid() -> None:
    valid = _util(hard_valid=True, binding_aware_meaningful_v2=0.9)
    invalid = _util(hard_valid=False, binding_aware_meaningful_v2=0.9)
    policy = ["binding_aware_meaningful_v2"]
    assert lexicographic_score(valid, policy)["rank_vector"] > lexicographic_score(invalid, policy)["rank_vector"]


def test_scalarized_score_weights_benefits_and_costs() -> None:
    manifest = UtilityWeightManifestV1(
        weights={
            "binding_aware_meaningful_v2": 1.0,
            "complexity_cost": -1.0,
        },
        normalization="unit",
    )
    good = _util(binding_aware_meaningful_v2=1.0, complexity_cost=0.1)
    bad = _util(binding_aware_meaningful_v2=0.1, complexity_cost=0.9)
    assert scalarized_score(good, manifest)["score"] > scalarized_score(bad, manifest)["score"]


def test_scalarized_score_ignores_unavailable_factors() -> None:
    manifest = UtilityWeightManifestV1(
        weights={"independent_judge_score": 1.0},
        normalization="unit",
    )
    no_judge = _util(binding_aware_meaningful_v2=0.9)
    with_judge = _util(
        binding_aware_meaningful_v2=0.9,
        independent_judge_score=0.5,
    )
    assert scalarized_score(no_judge, manifest)["score"] == 0.0
    assert scalarized_score(with_judge, manifest)["score"] > 0.0


def test_pareto_dominance_detects_domination() -> None:
    left = _util(
        binding_aware_meaningful_v2=0.9,
        component_role_recall=0.9,
        complexity_cost=0.1,
    )
    right = _util(
        binding_aware_meaningful_v2=0.5,
        component_role_recall=0.5,
        complexity_cost=0.5,
    )
    dom = pareto_dominance(left, right)
    assert dom["left_dominates"] is True
    assert dom["right_dominates"] is False
    assert dom["incomparable"] is False


def test_pareto_front_returns_non_dominated_points() -> None:
    dominant = ("dominant", _util(binding_aware_meaningful_v2=0.9, component_role_recall=0.9))
    dominated = ("dominated", _util(binding_aware_meaningful_v2=0.5, component_role_recall=0.5))
    incomparable = ("incomparable", _util(binding_aware_meaningful_v2=0.6, component_role_recall=0.95))
    front = pareto_front([dominant, dominated, incomparable])
    labels = {label for label, _ in front}
    assert "dominant" in labels
    assert "dominated" not in labels
    assert "incomparable" in labels


def test_cvar_tail_computes_worst_quantile_mean() -> None:
    values = list(range(100))
    result = cvar_tail(values, alpha=0.05)
    assert result["n"] == 100
    # Worst 5% is {0..4}; CVaR is their mean.
    assert result["cvar"] <= 5.0
    assert min(result["tail"]) <= result["cvar"] <= max(result["tail"])


def test_cvar_tail_degrades_on_empty() -> None:
    result = cvar_tail([])
    assert math.isnan(result["cvar"])
    assert result["n"] == 0


def test_abstention_economics_values_abstention() -> None:
    good = _util(binding_aware_meaningful_v2=0.9)
    bad = _util(binding_aware_meaningful_v2=0.1)
    abstained = _util(binding_aware_meaningful_v2=0.0, abstained=True)
    result = abstention_economics([good, bad, abstained], risk_threshold=0.3)
    assert result["accepted"] == 1
    assert result["abstained"] == 2
    assert result["mean_utility_accepted"] > result["mean_utility_if_forced"]
    assert result["value_of_abstention"] > 0.0


def test_sensitivity_rank_reversals_runs_and_reports_shape() -> None:
    candidates = [
        ("a", _util(binding_aware_meaningful_v2=0.9)),
        ("b", _util(binding_aware_meaningful_v2=0.5)),
        ("c", _util(binding_aware_meaningful_v2=0.7)),
    ]
    manifest = UtilityWeightManifestV1(
        weights={"binding_aware_meaningful_v2": 1.0},
        permitted_ranges={"binding_aware_meaningful_v2": (0.5, 1.5)},
    )
    result = sensitivity_rank_reversals(candidates, [manifest], perturbations_per_manifest=10, seed=0)
    assert result["n_candidates"] == 3
    assert result["n_manifests"] == 1
    assert result["total_perturbations"] == 10
    assert 0 <= result["reversal_count"] <= 10


def test_canonical_equivalent_utility_true_within_margin() -> None:
    a = _util(binding_aware_meaningful_v2=0.9, component_role_recall=0.8)
    b = _util(binding_aware_meaningful_v2=0.905, component_role_recall=0.805)
    assert canonical_equivalent_utility(a, b, margin=0.01) is True


def test_canonical_equivalent_utility_false_outside_margin() -> None:
    a = _util(binding_aware_meaningful_v2=0.9)
    b = _util(binding_aware_meaningful_v2=0.5)
    assert canonical_equivalent_utility(a, b, margin=0.01) is False


def test_canonical_equivalent_utility_false_on_hard_valid_mismatch() -> None:
    a = _util(hard_valid=True, binding_aware_meaningful_v2=0.9)
    b = _util(hard_valid=False, binding_aware_meaningful_v2=0.9)
    assert canonical_equivalent_utility(a, b) is False


def test_safe_ratio_returns_none_on_zero_denominator() -> None:
    result = safe_ratio(1.0, 0.0, "test")
    assert result["ratio"] is None
    assert result["numerator"] == 1.0
    assert result["denominator"] == 0.0


def test_safe_ratio_computes_ratio() -> None:
    result = safe_ratio(4.0, 2.0, "test")
    assert result["ratio"] == pytest.approx(2.0)


def test_weight_manifest_validate_catches_unknown_factor() -> None:
    manifest = UtilityWeightManifestV1(weights={"unknown_factor": 1.0})
    errors = manifest.validate()
    assert any("unknown weight factor" in e for e in errors)


def test_weight_manifest_validate_catches_weight_outside_range() -> None:
    manifest = UtilityWeightManifestV1(
        weights={"binding_aware_meaningful_v2": 2.0},
        permitted_ranges={"binding_aware_meaningful_v2": (0.0, 1.0)},
    )
    errors = manifest.validate()
    assert any("outside" in e for e in errors)
