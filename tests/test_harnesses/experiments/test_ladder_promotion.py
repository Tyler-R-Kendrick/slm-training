"""P1c ladder / scaling-fit / promotion tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.autoresearch.experiment_campaign import (
    CampaignResultV1,
    campaign_manifest_sha256,
)
from slm_training.harnesses.experiments.efficiency_gain import efficiency_gain, efficiency_gain_lcb
from slm_training.harnesses.experiments.ladder import (
    capacity_ladder_pair,
    model_build_config_for_point,
    proportional_depths,
    scratch_ladder_default,
)
from slm_training.harnesses.experiments.promotion import (
    check_category_regression,
    check_rank_stability,
    evaluate_promotion,
    register_promoted_checkpoint,
)
from slm_training.harnesses.experiments.scaling_fit import (
    ScalingObservation,
    fit_power_law,
    invert_loss,
    predict_loss,
)
from slm_training.harnesses.experiments.slm183_power_protocol import (
    build_experiment_campaign,
)
from slm_training.harnesses.model_build import ModelBuildConfig, train


def test_proportional_depths_and_scratch_ladder() -> None:
    h, c, d = proportional_depths(128)
    assert h >= 2 and d >= 2 and c >= 1
    ladder = scratch_ladder_default(base_token_budget=1000, widths=(64, 128), horizons=(1.0,))
    assert ladder.track == "scratch"
    assert len(ladder.points) == 2


# --- B3 (SLM-23): representation axis ---------------------------------------


def test_default_representation_keeps_legacy_point_ids() -> None:
    ladder = scratch_ladder_default(base_token_budget=1000, widths=(64,), horizons=(1.0,))
    point = ladder.points[0]
    assert point.representation == "compositional"
    assert "_r" not in point.point_id


def test_capacity_ladder_pair_matched_arms() -> None:
    lexer, choice = capacity_ladder_pair(
        base_token_budget=1000, widths=(64, 128), horizons=(1.0,)
    )
    assert lexer.ladder_id == "capacity_lexer_v1"
    assert choice.ladder_id == "capacity_choice_v1"
    assert len(lexer.points) == len(choice.points)
    for a, b in zip(lexer.points, choice.points):
        # Same capacity/budget point, differing only in representation.
        assert (a.d_model, a.target_token_budget) == (b.d_model, b.target_token_budget)
        assert a.representation == "lexer" and b.representation == "choice"
        assert a.point_id.endswith("_rlexer")
        assert b.point_id.endswith("_rchoice")


def test_lexer_representation_threads_into_config(tmp_path: Path) -> None:
    (lexer_ladder,) = capacity_ladder_pair(
        base_token_budget=1000, widths=(64,), horizons=(1.0,), representations=("lexer",)
    )
    cfg = model_build_config_for_point(
        lexer_ladder.points[0],
        lexer_ladder,
        train_dir=tmp_path,
        test_dir=None,
        run_root=tmp_path / "runs",
        seed=0,
    )
    assert cfg.output_tokenizer == "lexer"


def test_choice_representation_threads_into_config(tmp_path: Path) -> None:
    (choice_ladder,) = capacity_ladder_pair(
        base_token_budget=1000, widths=(64,), horizons=(1.0,), representations=("choice",)
    )
    cfg = model_build_config_for_point(
        choice_ladder.points[0],
        choice_ladder,
        train_dir=tmp_path,
        test_dir=None,
        run_root=tmp_path / "runs",
        seed=0,
    )
    assert cfg.output_tokenizer == "choice"


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
    assert result["promotable"] is False
    assert "campaign_governance_missing" in result["failures"]

    src = tmp_path / "last.pt"
    src.write_bytes(b"ckpt")
    with pytest.raises(ValueError, match="campaign-governed"):
        register_promoted_checkpoint(tmp_path / "ckpts", source=src, meta={"ok": True})


def test_integrity_alone_cannot_promote() -> None:
    result = evaluate_promotion(
        integrity={"pass": True, "failures": [], "leakage_hits": 0}
    )

    assert result["promotable"] is False
    assert result["failures"] == [
        "sufficient_evidence",
        "campaign_governance_missing",
    ]


def test_registration_rejects_forged_governance_result(tmp_path: Path) -> None:
    forged = {
        "promotable": True,
        "checks": {
            "campaign_governance": {
                "pass": True,
                "manifest_sha256": "a" * 64,
            }
        },
    }
    with pytest.raises(ValueError, match="campaign-governed"):
        register_promoted_checkpoint(
            tmp_path / "ckpts",
            promotion_result=forged,
        )


def test_training_promotion_requires_governance_before_data_load(
    tmp_path: Path,
) -> None:
    config = ModelBuildConfig(
        train_dir=tmp_path / "missing",
        run_root=tmp_path,
        run_id="governance_preflight",
        register_promoted=True,
    )

    with pytest.raises(ValueError, match="campaign manifest"):
        train(config)


def test_wiring_claim_cannot_satisfy_promotion_governance() -> None:
    manifest = build_experiment_campaign(seeds=(0,))
    result = CampaignResultV1(
        campaign_id=manifest.campaign_id,
        experiment_id=manifest.experiment_id,
        manifest_sha256=campaign_manifest_sha256(manifest),
        claim_class="wiring",
    )
    promotion = evaluate_promotion(
        integrity={"pass": True, "failures": []},
        campaign_manifest=manifest,
        campaign_result=result,
    )
    assert "claim_class_not_promotable" in promotion["failures"]
