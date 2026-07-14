"""P1c ladder / scaling-fit / promotion tests."""

from __future__ import annotations

from pathlib import Path

from slm_training.experiments.efficiency_gain import efficiency_gain, efficiency_gain_lcb
from slm_training.experiments.ladder import proportional_depths, scratch_ladder_default
from slm_training.experiments.promotion import (
    check_category_regression,
    check_rank_stability,
    evaluate_promotion,
    register_promoted_checkpoint,
)
from slm_training.experiments.scaling_fit import (
    ScalingObservation,
    fit_power_law,
    invert_loss,
    predict_loss,
)


def test_proportional_depths_and_scratch_ladder() -> None:
    h, c, d = proportional_depths(128)
    assert h >= 2 and d >= 2 and c >= 1
    ladder = scratch_ladder_default(base_token_budget=1000, widths=(64, 128), horizons=(1.0,))
    assert ladder.track == "scratch"
    assert len(ladder.points) == 2


def test_power_law_fit_and_eg() -> None:
    obs = [
        ScalingObservation("scratch", "c", "p1", 0, loss=2.0, cost_time_s=10.0),
        ScalingObservation("scratch", "c", "p2", 0, loss=1.2, cost_time_s=40.0),
        ScalingObservation("scratch", "c", "p3", 0, loss=0.9, cost_time_s=160.0),
    ]
    fit = fit_power_law(obs, cost_key="time")
    assert fit["A"] > 0 and fit["alpha"] > 0
    pred = predict_loss(fit, 40.0)
    assert abs(pred - 1.2) < 0.8
    inv = invert_loss(fit, 1.2)
    assert inv > 0
    eg = efficiency_gain(fit, obs[1], cost_key="time")
    assert eg is not None and eg > 0
    mean, lcb, ucb = efficiency_gain_lcb([1.1, 1.2, 1.15])
    assert lcb <= mean <= ucb


def test_promotion_checks(tmp_path: Path) -> None:
    baseline = {
        "categories": {
            "binding": {"aggregate": {"mean_nll": 1.0}},
            "structural": {"aggregate": {"mean_nll": 1.0}},
            "repair": {"aggregate": {"mean_nll": 1.0}},
        },
        "aggregate": {"weighted_nll": 1.0},
    }
    better = {
        "categories": {
            "binding": {"aggregate": {"mean_nll": 0.9}},
            "structural": {"aggregate": {"mean_nll": 0.9}},
            "repair": {"aggregate": {"mean_nll": 0.9}},
        },
        "aggregate": {"weighted_nll": 0.9},
    }
    assert check_category_regression(baseline, better)["pass"]
    assert check_rank_stability({"z": ["a"], "y": ["a"]})["pass"]
    result = evaluate_promotion(
        integrity={"pass": True, "failures": []},
        baseline_loss_report=baseline,
        candidate_loss_report=better,
        rankings={"z": ["cand"], "y": ["cand"]},
        eg_time_by_seed=[1.2, 1.3, 1.1],
    )
    assert result["promotable"]

    src = tmp_path / "last.pt"
    src.write_bytes(b"ckpt")
    dest = register_promoted_checkpoint(tmp_path / "ckpts", source=src, meta={"ok": True})
    assert dest.exists()
    assert (tmp_path / "ckpts" / "promoted.json").exists()
