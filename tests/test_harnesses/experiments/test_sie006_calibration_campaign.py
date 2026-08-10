"""SIE-006 (SLM-483) regression tests: EXP-SR-3 evaluator construct-validity
and human calibration campaign driver."""

from __future__ import annotations

import pytest

from slm_training.harnesses.experiments.sie006_calibration_campaign import (
    CATALOGUE_ID,
    PRIMARY_METRIC,
    Sie006CampaignV1,
    run_campaign,
)


def test_campaign_rejects_non_diagnostic_claim_class():
    with pytest.raises(ValueError, match="diagnostic"):
        Sie006CampaignV1(claim_class="fixture")


def test_campaign_rejects_fewer_than_two_raters():
    with pytest.raises(ValueError, match="min_raters"):
        Sie006CampaignV1(min_raters=1)


def test_campaign_manifest_matches_the_locked_exp_sr_3_catalogue_entry():
    campaign = Sie006CampaignV1()
    manifest = campaign.manifest()
    assert manifest.claim_class == "diagnostic"
    assert manifest.experiment_id == CATALOGUE_ID
    assert manifest.endpoints[0].metric == PRIMARY_METRIC
    assert manifest.endpoints[0].minimum_effect == campaign.minimum_effect()
    assert campaign.minimum_effect() == pytest.approx(0.1)


def test_run_campaign_produces_diagnostic_evidence_and_never_promotes(tmp_path):
    campaign = Sie006CampaignV1(campaign_id="test-sie006-exp-sr-3-calibration")
    result = run_campaign(campaign, root=tmp_path)

    assert result["claim_class"] == "diagnostic"
    assert result["promotion"] is False
    assert result["catalogue_id"] == CATALOGUE_ID
    assert result["primary_metric"] == PRIMARY_METRIC
    assert result["pair_n"] > 0
    # independence_ok must be reported, never silently dropped -- reused
    # directly from VCE-010, not recomputed here.
    assert result["independence_ok"] is not None
    # authorized is only ever True when the catalogue's own kill gate
    # (kappa <= 0) does not fire AND the preregistered minimum_effect is met.
    if result["authorized"]:
        assert not result["killed_by_catalogue_gate"]
        assert result["evaluator_human_agreement_kappa"] >= result["minimum_effect"]
    assert "mock" in result["scope_disclaimer"]
    assert "no downstream experiment" in result["scope_disclaimer"].lower()


def test_run_campaign_is_deterministic_for_a_fixed_campaign_id_and_seed(tmp_path):
    # protocol_id is derived from campaign_id and feeds VCE-010's own
    # content-hash-seeded sample selection, so determinism must be checked
    # for identical campaign params against independent storage roots --
    # not across differing campaign_ids, which legitimately vary the sample.
    campaign_a = Sie006CampaignV1(campaign_id="det-fixed")
    campaign_b = Sie006CampaignV1(campaign_id="det-fixed")
    result_a = run_campaign(campaign_a, root=tmp_path / "a")
    result_b = run_campaign(campaign_b, root=tmp_path / "b")
    assert result_a["evaluator_human_agreement_kappa"] == result_b["evaluator_human_agreement_kappa"]
    assert result_a["pair_n"] == result_b["pair_n"]


def test_killed_by_catalogue_gate_is_consistent_with_kappa_sign(tmp_path):
    campaign = Sie006CampaignV1(campaign_id="test-sie006-gate-consistency")
    result = run_campaign(campaign, root=tmp_path)
    kappa = result["evaluator_human_agreement_kappa"]
    if kappa is None:
        assert result["killed_by_catalogue_gate"] is True
    else:
        assert result["killed_by_catalogue_gate"] == (kappa <= 0.0)
