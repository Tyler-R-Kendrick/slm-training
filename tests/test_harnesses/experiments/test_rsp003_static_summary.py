"""RSP-003 (SLM-485): EXP-SR-7 static summary campaign tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.experiments.exp_sr_catalogue import exp_sr_campaign
from slm_training.harnesses.experiments.rsp003_static_summary import (
    COLD_ARMS,
    CATALOGUE_ID,
    PARITY_CHECK_IDS,
    PRIMARY_METRIC,
    WARM_ARM,
    Rsp003CampaignV1,
    check_differential_parity,
    plan_only_preview,
    run_campaign,
    score_static_summary_domain_parity,
)


def test_campaign_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="cold_trials_per_arm"):
        Rsp003CampaignV1(cold_trials_per_arm=0)
    with pytest.raises(ValueError, match="warm_trials"):
        Rsp003CampaignV1(warm_trials=0)
    with pytest.raises(ValueError, match="fixture"):
        Rsp003CampaignV1(claim_class="diagnostic")
    with pytest.raises(ValueError, match="MAX_RUN_MINUTES"):
        Rsp003CampaignV1(max_wall_minutes=1000.0)


def test_manifest_binds_catalogue_identity() -> None:
    campaign = Rsp003CampaignV1()
    manifest = campaign.manifest()
    catalogue = exp_sr_campaign(CATALOGUE_ID)
    assert isinstance(manifest, ExperimentCampaignV1)
    assert manifest.experiment_id == CATALOGUE_ID
    assert manifest.claim_class == "fixture"
    assert manifest.claim_class not in {"promotion_candidate", "ship_gate"}
    assert {arm.arm_id for arm in manifest.arms} == {*COLD_ARMS, WARM_ARM}
    assert manifest.endpoints[0].metric == PRIMARY_METRIC
    assert manifest.endpoints[0].metric == catalogue.endpoints[0].metric
    assert manifest.rollback_gates[0].threshold == catalogue.rollback_gates[0].threshold


def test_plan_only_preview_exposes_catalogue_and_arms() -> None:
    preview = plan_only_preview()
    assert preview["status"] == "plan_only"
    assert preview["manifest"]["experiment_id"] == CATALOGUE_ID
    assert preview["claim_class_execution"] == "fixture"
    assert preview["promotion"] is False
    assert set(preview["fixture_arms"]) == {*COLD_ARMS, WARM_ARM}
    assert set(preview["parity_checks"]) == set(PARITY_CHECK_IDS)


def test_differential_parity_holds() -> None:
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    parity = check_differential_parity(tok)
    assert parity["equal"] is True
    assert parity["mismatch_count"] == 0
    assert parity["prefixes_checked"] > 0


def test_static_summary_domain_parity_rate_is_one() -> None:
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    score = score_static_summary_domain_parity(tok)
    assert score[PRIMARY_METRIC] == 1.0
    assert score["certified"] is True
    assert score["kill_gate_triggered"] is False
    assert score["checks_passed"] == score["checks_total"]


def test_run_campaign_end_to_end(tmp_path: Path) -> None:
    campaign = Rsp003CampaignV1(cold_trials_per_arm=1, warm_trials=2)
    result = run_campaign(campaign, root=tmp_path, timeout_s=20.0)

    assert result["static_summary_domain_parity_rate"] == 1.0
    assert result["parity_analysis"]["certified"] is True
    assert result["claim_class"] == "fixture"
    assert result["promotion"] is False
    assert len(result["trials"]) == len(COLD_ARMS) + 1
    assert "cold_path_improvement" in result
    assert "recommendation" in result

    store = CampaignStore(campaign.campaign_id, tmp_path)
    locked = store.load_experiment_campaign(CATALOGUE_ID)
    assert locked.manifest_sha256 == result["manifest_sha256"]


def test_run_campaign_aborts_on_parity_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import slm_training.harnesses.experiments.rsp003_static_summary as mod

    def _fake_fail(_tokenizer):
        return {
            PRIMARY_METRIC: 0.5,
            "certified": False,
            "kill_gate_triggered": True,
        }

    monkeypatch.setattr(mod, "score_static_summary_domain_parity", _fake_fail)

    campaign = Rsp003CampaignV1(cold_trials_per_arm=1, warm_trials=1)
    with pytest.raises(RuntimeError, match="aborted before any timing trial"):
        run_campaign(campaign, root=tmp_path, timeout_s=20.0)
