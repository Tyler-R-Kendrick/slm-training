"""RSP-008 (SLM-487): EXP-SR-12 second-pack portability campaign tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.experiments.exp_sr_catalogue import exp_sr_campaign
from slm_training.harnesses.experiments.rsp008_second_pack_portability import (
    ARM_IDS,
    CATALOGUE_ID,
    PRIMARY_METRIC,
    SEAM_CHECK_IDS,
    Rsp008CampaignV1,
    exercise_openui_control_arm,
    plan_only_preview,
    run_campaign,
    score_second_pack_portability,
)
from slm_training.harnesses.experiments.srp010_pack_portability import (
    assess_symbolic_pack_portability,
)


def test_campaign_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="diagnostic"):
        Rsp008CampaignV1(claim_class="promotion_candidate")
    with pytest.raises(ValueError, match="diagnostic"):
        Rsp008CampaignV1(claim_class="ship_gate")
    with pytest.raises(ValueError, match="diagnostic"):
        Rsp008CampaignV1(claim_class="fixture")
    with pytest.raises(ValueError, match="MAX_RUN_MINUTES"):
        Rsp008CampaignV1(max_wall_minutes=1000.0)
    with pytest.raises(ValueError, match="seeds"):
        Rsp008CampaignV1(seeds=())


def test_manifest_binds_catalogue_identity_and_matched_arms() -> None:
    campaign = Rsp008CampaignV1()
    manifest = campaign.manifest()
    catalogue = exp_sr_campaign(CATALOGUE_ID)
    assert isinstance(manifest, ExperimentCampaignV1)
    assert manifest.experiment_id == CATALOGUE_ID
    assert manifest.claim_class == "diagnostic"
    assert manifest.claim_class not in {"promotion_candidate", "ship_gate"}
    assert {arm.arm_id for arm in manifest.arms} == set(ARM_IDS)
    assert manifest.endpoints[0].metric == PRIMARY_METRIC
    assert manifest.endpoints[0].metric == catalogue.endpoints[0].metric
    assert manifest.endpoints[0].minimum_effect == catalogue.endpoints[0].minimum_effect
    assert manifest.rollback_gates[0].operator == "lt"
    assert manifest.rollback_gates[0].threshold == catalogue.rollback_gates[0].threshold
    assert manifest.seeds == (0, 1, 2)


def test_plan_only_preview_exposes_catalogue_and_arms() -> None:
    preview = plan_only_preview()
    assert preview["status"] == "plan_only"
    assert preview["manifest"]["experiment_id"] == CATALOGUE_ID
    assert preview["fixture_arms"] == list(ARM_IDS)
    assert preview["seam_checks"] == list(SEAM_CHECK_IDS)
    assert preview["claim_class_execution"] == "diagnostic"
    assert preview["promotion"] is False


def test_score_reports_forks_and_fail_closed_mismatch() -> None:
    openui = exercise_openui_control_arm()
    sr = assess_symbolic_pack_portability(run_id="rsp008_score_test")
    score = score_second_pack_portability(openui_arm=openui, sr_report=sr)
    assert set(score["checks"]) == set(SEAM_CHECK_IDS)
    assert score["checks"]["cross_pack_mismatch_fails_closed"] is True
    assert score["checks"]["no_learned_legality"] is True
    assert score["checks"]["default_isolation_openui"] is True
    # SRP-010 inventorizes honest shared-seam forks → falsifier holds.
    assert score["fork_ids"]
    assert score["checks"]["zero_required_shared_seam_forks"] is False
    assert score["falsifier_holds"] is True
    assert score["certified"] is False
    assert 0.0 <= score["second_pack_portability_parity_rate"] < 1.0


def test_run_campaign_end_to_end_locks_store(tmp_path: Path) -> None:
    campaign = Rsp008CampaignV1(seeds=(0, 1))
    result = run_campaign(campaign, root=tmp_path)

    assert result["claim_class"] == "diagnostic"
    assert result["promotion"] is False
    assert result["catalogue_id"] == CATALOGUE_ID
    assert result["rate_stable_across_seeds"] is True
    assert 0.0 <= result["second_pack_portability_parity_rate"] < 1.0
    assert result["kill_gate_triggered"] is True
    assert result["certified"] is False
    assert result["analysis"]["falsifier_holds"] is True
    assert result["openui_control"]["pack_available"] is True
    assert result["symbolic_regression_candidate"]["forks"]
    assert len(result["seed_runs"]) == 2

    store = CampaignStore(campaign.campaign_id, tmp_path)
    locked = store.load_experiment_campaign(CATALOGUE_ID)
    assert locked.manifest_sha256 == result["manifest_sha256"]
    assert locked.manifest.claim_class == "diagnostic"
