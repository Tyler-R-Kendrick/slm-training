"""Goal invariant VI: capability may not be bought with parameters.

Each test pins one way the harness previously let a scaled-up model be
credited with a capability win.
"""

from __future__ import annotations

import pytest

import math

from slm_training.harness_core.efficiency_gain import (
    efficiency_gain,
    efficiency_gain_lcb,
)
from slm_training.harness_core.promotion_engine import (
    PromotionCriteria,
    check_parameter_efficiency,
    check_rank_stability,
    evaluate_promotion,
    select_smallest_sufficient,
)
from slm_training.harness_core.scaling_fit import (
    ScalingObservation,
    fit_power_law,
    observation_cost,
)
from slm_training.levers import (
    CAPACITY_SCALING_LEVERS,
    require_size_matched_arms,
    size_matched_violations,
)


def _obs(params: int, loss: float, *, seed: int = 0) -> ScalingObservation:
    return ScalingObservation(
        track="scratch",
        candidate_id=f"d{params}",
        point_id=f"d{params}",
        seed=seed,
        loss=loss,
        cost_time_s=1.0,
        trainable_params=params,
    )


def test_params_is_a_chargeable_cost_axis() -> None:
    # Recorded-but-never-charged was the defect: trainable_params existed on
    # every observation and no cost accessor would return it.
    obs = _obs(1000, 2.0)
    assert observation_cost(obs, "params") == 1000.0
    assert observation_cost(obs, "time") == 1.0


def test_unmeasured_params_are_not_charged_as_zero() -> None:
    obs = ScalingObservation(
        track="t", candidate_id="c", point_id="p", seed=0, loss=1.0, cost_time_s=1.0
    )
    assert observation_cost(obs, "params") is None


def test_power_law_and_efficiency_gain_accept_params() -> None:
    observations = [_obs(1000, 4.0), _obs(4000, 2.0)]
    fit = fit_power_law(observations, cost_key="params")
    assert fit["n"] == 2.0
    # The cost switch is shared, so a new axis reaches efficiency_gain too.
    assert efficiency_gain(fit, observations[1], cost_key="params") is not None


def test_growth_without_measured_efficiency_fails_closed() -> None:
    result = check_parameter_efficiency(
        baseline_params=1_000,
        candidate_params=2_000,
        eg_params_by_seed=None,
    )
    assert result["pass"] is False
    assert result["reason"] == "capacity_increased_without_parameter_efficiency_evidence"
    assert result["growth_ratio"] == pytest.approx(2.0)


def test_holding_or_shrinking_capacity_always_passes() -> None:
    for candidate in (1_000, 400):
        result = check_parameter_efficiency(
            baseline_params=1_000,
            candidate_params=candidate,
            eg_params_by_seed=None,
        )
        assert result["pass"] is True
        assert result["reason"] == "capacity_not_increased"


def test_growth_passes_only_with_a_size_normalized_gain() -> None:
    weak = check_parameter_efficiency(
        baseline_params=1_000,
        candidate_params=2_000,
        eg_params_by_seed=[0.5, 0.4, 0.6],
    )
    assert weak["pass"] is False
    strong = check_parameter_efficiency(
        baseline_params=1_000,
        candidate_params=2_000,
        eg_params_by_seed=[2.0, 2.1, 1.9],
    )
    assert strong["pass"] is True


def test_promotion_rejects_a_bigger_better_model_that_only_won_on_time() -> None:
    # The pre-fix hole: strong wall-time efficiency and improved loss, bought
    # by doubling parameters, promoted cleanly.
    result = evaluate_promotion(
        eg_time_by_seed=[3.0, 3.1, 2.9],
        baseline_trainable_params=1_000,
        candidate_trainable_params=2_000,
        hard_categories=(),
        gate_evaluator=lambda *_: {"pass": True},
    )
    assert result["promotable"] is False
    assert "eg_params" in result["failures"]
    assert result["checks"]["eg_time"]["pass"] is True


def test_promotion_allows_growth_that_pays_for_itself() -> None:
    result = evaluate_promotion(
        eg_time_by_seed=[3.0, 3.1, 2.9],
        eg_params_by_seed=[2.0, 2.1, 1.9],
        baseline_trainable_params=1_000,
        candidate_trainable_params=2_000,
        hard_categories=(),
        gate_evaluator=lambda *_: {"pass": True},
    )
    assert result["promotable"] is True


def test_criteria_expose_a_parameter_efficiency_bar() -> None:
    assert PromotionCriteria().eg_params_lcb_min == 1.0


def test_select_smallest_sufficient_prefers_the_smaller_model_in_the_band() -> None:
    summaries = [
        {"id": "small", "params": 1_000, "loss": 2.01},
        {"id": "huge", "params": 8_000, "loss": 2.00},
    ]
    best = select_smallest_sufficient(
        summaries,
        params_of=lambda s: s["params"],
        loss_of=lambda s: s["loss"],
        tolerance=0.02,
    )
    assert best is not None and best["id"] == "small"


def test_select_smallest_sufficient_still_takes_a_real_win() -> None:
    summaries = [
        {"id": "small", "params": 1_000, "loss": 4.00},
        {"id": "huge", "params": 8_000, "loss": 2.00},
    ]
    best = select_smallest_sufficient(
        summaries,
        params_of=lambda s: s["params"],
        loss_of=lambda s: s["loss"],
        tolerance=0.02,
    )
    assert best is not None and best["id"] == "huge"


def test_rank_stability_orders_ladder_points_numerically() -> None:
    # A lexical sort ranks "d96" > "d64" > "d192" and compares the wrong rungs.
    rankings = {"d64": ["a"], "d96": ["a"], "d192": ["b"]}
    result = check_rank_stability(rankings)
    assert result["keys"] == ["d192", "d96"]


class _Arm:
    def __init__(self, d_model: int, denoiser_layers: int = 4) -> None:
        self.d_model = d_model
        self.denoiser_layers = denoiser_layers


def test_size_matched_arms_reject_a_capacity_difference() -> None:
    assert size_matched_violations(_Arm(128), _Arm(128)) == {}
    violations = size_matched_violations(_Arm(128), _Arm(192))
    assert "d_model" in violations
    with pytest.raises(ValueError, match="different capacity"):
        require_size_matched_arms(_Arm(128), _Arm(192), context="matrix")


def test_capacity_registry_declares_baseline_and_axis() -> None:
    assert CAPACITY_SCALING_LEVERS
    for name, spec in CAPACITY_SCALING_LEVERS.items():
        assert "baseline_value" in spec, name
        assert spec.get("axis") in {"width", "depth", "backbone"}, name


def test_efficiency_gain_lcb_empty_is_all_nan() -> None:
    mean, lcb, ucb = efficiency_gain_lcb([])
    assert math.isnan(mean) and math.isnan(lcb) and math.isnan(ucb)


def test_efficiency_gain_lcb_single_seed_has_no_dispersion_estimate() -> None:
    # A single observation must not present a zero-width interval that
    # trivially clears any lcb>=threshold gate — the mean is honest, the
    # bounds are unknown.
    mean, lcb, ucb = efficiency_gain_lcb([1.5])
    assert mean == pytest.approx(1.5)
    assert math.isnan(lcb) and math.isnan(ucb)


def test_efficiency_gain_lcb_matches_student_t_at_small_n() -> None:
    # df=1 (n=2), 95% two-sided critical value is 12.706 (textbook), not the
    # normal-approx 1.96 the old implementation used.
    vals = [1.0, 1.2]
    mean, lcb, ucb = efficiency_gain_lcb(vals, confidence=0.95)
    n = len(vals)
    sample_mean = sum(vals) / n
    var = sum((v - sample_mean) ** 2 for v in vals) / (n - 1)
    se = math.sqrt(var) / math.sqrt(n)
    expected_lcb = sample_mean - 12.706 * se
    expected_ucb = sample_mean + 12.706 * se
    assert lcb == pytest.approx(expected_lcb, rel=1e-3)
    assert ucb == pytest.approx(expected_ucb, rel=1e-3)


def test_efficiency_gain_lcb_is_wider_than_the_old_normal_approx_at_small_n() -> None:
    vals = [1.0, 1.1, 0.95]
    _, lcb, _ = efficiency_gain_lcb(vals, confidence=0.95)
    n = len(vals)
    sample_mean = sum(vals) / n
    var = sum((v - sample_mean) ** 2 for v in vals) / (n - 1)
    se = math.sqrt(var) / math.sqrt(n)
    stale_normal_lcb = sample_mean - 1.96 * se
    assert lcb < stale_normal_lcb


def test_efficiency_gain_lcb_clamps_beyond_the_tabulated_degrees_of_freedom() -> None:
    # 35 seeds -> df=34, beyond the table's df=30 ceiling; the largest
    # tabulated (most conservative) critical value is reused rather than
    # extrapolating.
    vals = [1.0 + 0.01 * i for i in range(35)]
    _, lcb, ucb = efficiency_gain_lcb(vals, confidence=0.95)
    n = len(vals)
    sample_mean = sum(vals) / n
    var = sum((v - sample_mean) ** 2 for v in vals) / (n - 1)
    se = math.sqrt(var) / math.sqrt(n)
    expected_lcb = sample_mean - 2.042 * se
    assert lcb == pytest.approx(expected_lcb, rel=1e-3)
    assert ucb > sample_mean > lcb
