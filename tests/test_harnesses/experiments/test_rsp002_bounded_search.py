"""RSP-002 (SLM-489): EXP-SR-6 bounded search campaign tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.experiments.exp_sr_catalogue import exp_sr_campaign
from slm_training.harnesses.experiments.rsp002_bounded_search import (
    ARM_IDS,
    CATALOGUE_ID,
    PRIMARY_METRIC,
    Rsp002CampaignV1,
    build_search_fixtures,
    plan_only_preview,
    prove_reachability,
    run_campaign,
    run_search_arm,
    score_bounded_search_hit_rate,
)


def test_campaign_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="max_decisions"):
        Rsp002CampaignV1(max_decisions=0)
    with pytest.raises(ValueError, match="fixture"):
        Rsp002CampaignV1(claim_class="diagnostic")
    with pytest.raises(ValueError, match="MAX_RUN_MINUTES"):
        Rsp002CampaignV1(max_wall_minutes=1000.0)


def test_manifest_binds_catalogue_identity() -> None:
    campaign = Rsp002CampaignV1()
    manifest = campaign.manifest()
    catalogue = exp_sr_campaign(CATALOGUE_ID)
    assert isinstance(manifest, ExperimentCampaignV1)
    assert manifest.experiment_id == CATALOGUE_ID
    assert manifest.claim_class == "fixture"
    assert manifest.claim_class not in {"promotion_candidate", "ship_gate"}
    assert {arm.arm_id for arm in manifest.arms} == set(ARM_IDS)
    assert manifest.endpoints[0].metric == PRIMARY_METRIC
    assert manifest.endpoints[0].metric == catalogue.endpoints[0].metric
    assert manifest.rollback_gates[0].threshold == catalogue.rollback_gates[0].threshold


def test_plan_only_preview_exposes_catalogue_and_arms() -> None:
    preview = plan_only_preview()
    assert preview["status"] == "plan_only"
    assert preview["manifest"]["experiment_id"] == CATALOGUE_ID
    assert preview["claim_class_execution"] == "fixture"
    assert preview["promotion"] is False
    assert preview["default_lever_state"] == "OFF"
    assert set(preview["fixture_arms"]) == set(ARM_IDS)


def test_reachability_holds_for_fixtures() -> None:
    reach = prove_reachability(build_search_fixtures())
    assert reach["reachable"] is True
    for problem in build_search_fixtures():
        assert reach["problems"][problem.problem_id]["reachable"] is True


def test_neural_prior_beats_left_to_right_on_fixtures() -> None:
    for problem_id in ("ordering_trap_v1", "ranking_trap_v1"):
        problem = next(p for p in build_search_fixtures() if p.problem_id == problem_id)
        ltr = run_search_arm("left_to_right", problem, max_decisions=4)
        prior = run_search_arm("neural_prior", problem, max_decisions=4)
        assert ltr["hit"] is False, problem_id
        assert prior["hit"] is True, problem_id


def test_planned_hole_order_reachability_only() -> None:
    trap = next(p for p in build_search_fixtures() if p.problem_id == "ordering_trap_v1")
    planned = run_search_arm("planned_hole_order", trap, generous=True, max_decisions=32)
    assert planned["hit"] is True


def test_score_bounded_search_hit_rate() -> None:
    scored = score_bounded_search_hit_rate(
        successes={"left_to_right": 0, "neural_prior": 3},
        attempts={"left_to_right": 3, "neural_prior": 3},
    )
    assert scored[PRIMARY_METRIC] == 1.0
    assert scored["control_left_to_right"] == 0.0
    assert scored["delta_vs_left_to_right"] == 1.0


def test_run_campaign_end_to_end(tmp_path: Path) -> None:
    campaign = Rsp002CampaignV1(seeds=(0,))
    result = run_campaign(campaign, root=tmp_path)

    assert result["claim_class"] == "fixture"
    assert result["promotion"] is False
    assert result["reachability"]["reachable"] is True
    assert result["bounded_search_hit_rate"] >= result["aggregate_hit_rates"]["control_left_to_right"]
    assert "falsifier" in result
    assert "recommendation" in result

    store = CampaignStore(campaign.campaign_id, tmp_path)
    locked = store.load_experiment_campaign(CATALOGUE_ID)
    assert locked.manifest_sha256 == result["manifest_sha256"]
