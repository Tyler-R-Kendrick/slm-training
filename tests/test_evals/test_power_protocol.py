"""Tests for SLM-183 statistical utilities."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from slm_training.evals.power_protocol import (
    benjamini_hochberg,
    binomial_rate_evidence,
    bootstrap_paired_ci,
    classify_power,
    cluster_bootstrap_ci,
    exact_binomial_interval,
    intraclass_correlation,
    mde_simulation,
    plan_binomial_rate_test,
    wilson_interval,
)


def test_wilson_interval_basic() -> None:
    result = wilson_interval(50, 100)
    assert result["n"] == 100
    assert math.isclose(result["estimate"], 0.5, abs_tol=1e-9)
    assert 0.0 < result["low"] < result["estimate"] < result["high"] < 1.0


def test_wilson_interval_degrades_on_zero_n() -> None:
    result = wilson_interval(0, 0)
    assert result == {
        "n": 0,
        "estimate": None,
        "low": None,
        "high": None,
        "confidence_level": 0.95,
    }


@pytest.mark.parametrize("successes,n", [(-5, 10), (15, 10)])
def test_wilson_interval_rejects_invalid_successes(successes: int, n: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, n)


@pytest.mark.parametrize(
    "successes,n,low,high",
    [
        (0, 5, 0.0, 0.43448246478317465),
        (1, 4, 0.0455872608097006, 0.699358157417598),
        (5, 5, 0.5655175352168254, 1.0),
        (20, 20, 0.8388748419471808, 1.0),
    ],
)
def test_wilson_interval_fixed_examples(
    successes: int, n: int, low: float, high: float
) -> None:
    result = wilson_interval(successes, n)
    assert result["low"] == pytest.approx(low)
    assert result["high"] == pytest.approx(high)


def test_wilson_interval_supports_configurable_confidence() -> None:
    narrow = wilson_interval(5, 10, confidence_level=0.80)
    wide = wilson_interval(5, 10, confidence_level=0.99)
    assert float(narrow["high"]) - float(narrow["low"]) < (
        float(wide["high"]) - float(wide["low"])
    )


def test_binomial_rate_evidence_discloses_counts_and_seed_class() -> None:
    evidence = binomial_rate_evidence(
        1, 4, seed_count=1, evidence_class="fixture_under_minimum_n"
    )
    assert evidence["numerator"] == 1
    assert evidence["denominator"] == 4
    assert evidence["seed_count"] == 1
    assert evidence["interval"]["method"] == "wilson_score"
    assert evidence["evidence_class"] == "fixture_under_minimum_n"


def test_plan_binomial_rate_test_is_prospective_and_seed_separate() -> None:
    plan = plan_binomial_rate_test(
        null_rate=0.5,
        target_delta=0.1,
        alpha=0.05,
        target_power=0.8,
        seeds=(0, 1, 2),
    )
    assert plan["required_n"] == plan["planned_sample_size_per_seed"]
    assert plan["seed_count"] == 3
    assert plan["seed_aggregation"] == "report_separately_no_pooling"
    assert "observed" not in json.dumps(plan)
    assert plan["use"] == "preregistration_only_not_post_hoc_success_evidence"


@pytest.mark.parametrize("seeds", [(0, 0), (True,), (1.5,)])
def test_plan_binomial_rate_test_rejects_invalid_seeds(seeds: tuple[object, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        plan_binomial_rate_test(null_rate=0.5, target_delta=0.1, seeds=seeds)


@pytest.mark.parametrize("sides", [True, False, 1.0, 2.0])
def test_plan_binomial_rate_test_rejects_non_integer_sides(sides: object) -> None:
    with pytest.raises(ValueError, match="sides must be 1 or 2"):
        plan_binomial_rate_test(null_rate=0.5, target_delta=0.1, sides=sides)


def test_exact_binomial_interval_basic() -> None:
    result = exact_binomial_interval(50, 100)
    assert result["n"] == 100
    assert math.isclose(result["estimate"], 0.5, abs_tol=1e-9)
    assert 0.0 < result["low"] < result["estimate"] < result["high"] < 1.0


def test_exact_binomial_interval_boundary_cases() -> None:
    zero = exact_binomial_interval(0, 20)
    assert zero["low"] == 0.0
    assert zero["high"] > 0.0
    full = exact_binomial_interval(20, 20)
    assert full["low"] < 1.0
    assert full["high"] == 1.0


def test_exact_binomial_interval_degrades_on_zero_n() -> None:
    result = exact_binomial_interval(0, 0)
    assert result == {"n": 0, "estimate": 0.0, "low": 0.0, "high": 0.0}


def test_exact_binomial_covers_wilson() -> None:
    """Exact interval should be at least as wide as Wilson for the same data."""
    wilson = wilson_interval(30, 100)
    exact = exact_binomial_interval(30, 100)
    assert exact["low"] <= wilson["low"]
    assert exact["high"] >= wilson["high"]


def test_bootstrap_paired_ci_mean_difference() -> None:
    left = [1.0, 2.0, 3.0, 4.0, 5.0]
    right = [1.1, 2.1, 3.1, 4.1, 5.1]
    result = bootstrap_paired_ci(
        left, right, lambda a, b: float(np.mean(a) - np.mean(b)), seed=42
    )
    assert math.isclose(result["estimate"], -0.1, abs_tol=1e-9)
    assert result["low"] <= result["estimate"] <= result["high"]


def test_bootstrap_paired_ci_degrades_on_empty() -> None:
    result = bootstrap_paired_ci([], [], lambda a, b: 0.0)
    assert math.isnan(result["estimate"])


def test_cluster_bootstrap_ci_basic() -> None:
    values = [1.0, 1.1, 2.0, 2.1, 3.0, 3.1]
    cluster_ids = ["a", "a", "b", "b", "c", "c"]
    result = cluster_bootstrap_ci(values, cluster_ids, np.mean, seed=7)
    assert math.isclose(result["estimate"], 2.05, abs_tol=1e-9)
    assert result["low"] <= result["estimate"] <= result["high"]


def test_cluster_bootstrap_ci_degrades_on_empty() -> None:
    result = cluster_bootstrap_ci([], [], np.mean)
    assert math.isnan(result["estimate"])


def test_intraclass_correlation_perfect_clusters() -> None:
    # Values within each cluster are identical; ICC should be high.
    values = [1.0, 1.0, 5.0, 5.0, 9.0, 9.0]
    cluster_ids = ["a", "a", "b", "b", "c", "c"]
    result = intraclass_correlation(values, cluster_ids)
    assert result["icc"] > 0.9
    assert result["n_clusters"] == 3


def test_intraclass_correlation_degrades_single_cluster() -> None:
    result = intraclass_correlation([1.0, 2.0, 3.0], ["a", "a", "a"])
    assert result["icc"] == 0.0
    assert result["n_clusters"] == 1


def test_mde_simulation_returns_curve() -> None:
    result = mde_simulation(
        base_rate=0.7,
        sigma_seed=0.1,
        sigma_target=0.2,
        n_targets=20,
        paths_per_target=2,
        n_seeds=2,
        n_simulations=40,
        effect_sizes=[0.0, 0.1, 0.2],
        seed=1,
    )
    assert "curve" in result
    assert len(result["curve"]) == 3
    # Power at effect_size 0 should be near alpha.
    null_power = result["curve"][0]["power"]
    assert 0.0 <= null_power <= 0.35
    # Largest effect should tend to have the highest power.
    powers = [pt["power"] for pt in result["curve"]]
    assert max(powers) == powers[-1]


def test_mde_simulation_degrades_with_zero_variance() -> None:
    result = mde_simulation(
        base_rate=0.7,
        sigma_seed=0.0,
        sigma_target=0.0,
        n_targets=10,
        paths_per_target=2,
        n_seeds=2,
        n_simulations=10,
        effect_sizes=[0.0, 0.1],
        seed=2,
    )
    assert result["curve"][0]["power"] <= 0.3


def test_benjamini_hochberg_rejects_expected() -> None:
    p_values = [0.001, 0.02, 0.04, 0.06, 0.5]
    result = benjamini_hochberg(p_values, alpha=0.05)
    rejected = [e["rejected"] for e in result]
    # BH thresholds: [0.01, 0.02, 0.03, 0.04, 0.05]; first two pass.
    assert rejected[:2] == [True, True]
    assert rejected[2:] == [False, False, False]


def test_benjamini_hochberg_empty() -> None:
    assert benjamini_hochberg([]) == []


def test_benjamini_hochberg_preserves_order() -> None:
    p_values = [0.5, 0.001]
    result = benjamini_hochberg(p_values)
    assert result[0]["p_value"] == 0.5
    assert result[1]["p_value"] == 0.001


@pytest.mark.parametrize(
    "conclusion,mde,effect_size,expected",
    [
        (True, 0.08, 0.10, "decidable"),
        ("power_met", 0.08, 0.10, "decidable"),
        (0.85, 0.08, 0.10, "decidable"),
        (False, 0.08, 0.05, "large_effect_only"),
        (False, 0.08, 0.01, "underpowered"),
    ],
)
def test_classify_power(conclusion, mde, effect_size, expected) -> None:
    assert classify_power(conclusion, mde, effect_size) == expected
