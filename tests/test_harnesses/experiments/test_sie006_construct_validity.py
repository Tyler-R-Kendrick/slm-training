"""SIE-006 (SLM-483): construct-validity campaign harness tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.harnesses.experiments.exp_sr_catalogue import exp_sr_campaign
from slm_training.harnesses.experiments.sie006_construct_validity import (
    ARM_IDS,
    BLOCKED_CLAIMS,
    CATALOGUE_ID,
    PRIMARY_METRIC,
    SIE005_FROZEN_DOC,
    Sie006CampaignV1,
    load_frozen_sie005_pin,
    materialize_sie005_packet,
    plan_only_preview,
    run_external_blocked_campaign,
    run_fixture_campaign,
)


def test_campaign_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="evidence classes"):
        Sie006CampaignV1(claim_class="promotion_candidate")
    with pytest.raises(ValueError, match="MAX_RUN_MINUTES"):
        Sie006CampaignV1(max_wall_minutes=1000.0)


def test_manifest_binds_exp_sr_3_catalogue() -> None:
    campaign = Sie006CampaignV1()
    manifest = campaign.manifest()
    catalogue = exp_sr_campaign(CATALOGUE_ID)
    assert isinstance(manifest, ExperimentCampaignV1)
    assert manifest.experiment_id == CATALOGUE_ID
    assert manifest.endpoints[0].metric == PRIMARY_METRIC
    assert manifest.endpoints[0].minimum_effect == catalogue.endpoints[0].minimum_effect
    assert {arm.arm_id for arm in manifest.arms} == set(ARM_IDS)
    assert manifest.claim_class == "fixture"


def test_plan_only_preview_references_frozen_sie005_pin() -> None:
    preview = plan_only_preview()
    assert preview["status"] == "plan_only"
    assert preview["catalogue_id"] == CATALOGUE_ID
    assert preview["primary_metric"] == PRIMARY_METRIC
    assert preview["fixture_arms"] == list(ARM_IDS)
    assert preview["sie005_pin"]["pair_n"] == 8
    assert preview["promotion"] is False
    assert set(preview["blocked_without_real_evidence"]) == set(BLOCKED_CLAIMS)


def test_materialize_sie005_packet_matches_frozen_doc(tmp_path: Path) -> None:
    pin = load_frozen_sie005_pin()
    packet = materialize_sie005_packet(root=tmp_path, pin=pin)
    assert packet["manifest"]["training_use"] is False
    assert packet["manifest"]["audit_holdout"] is True
    assert len(packet["public_pairs"]) == int(pin["pair_n"])
    assert (packet["public_dir"] / "blind_pairs.jsonl").is_file()


def test_fixture_campaign_exercises_plumbing_without_real_human(tmp_path: Path) -> None:
    campaign = Sie006CampaignV1()
    result = run_fixture_campaign(campaign, root=tmp_path / "fixture")
    assert result["status"] == "fixture"
    assert result["external_blocked"] is False
    assert result["promotion"] is False
    assert result["training_excluded"] is True
    assert result["evaluator_human_agreement_kappa"] is not None
    assert result["qualified_claims"]["blinded_human_adjudication"] == "synthetic_labels_only"
    assert result["qualified_claims"]["external_judge_calibration"] == "mock_transport_only"
    assert result["blocked_claims"] == []
    full_arm = next(arm for arm in result["arms"] if arm["arm"] == "full_construct_validity")
    assert full_arm["evidence_class"] == "fixture_plumbing"


def test_external_blocked_campaign_is_honest(tmp_path: Path) -> None:
    campaign = Sie006CampaignV1()
    result = run_external_blocked_campaign(campaign, root=tmp_path / "blocked")
    assert result["status"] == "external_blocked"
    assert result["disposition"] == "external_blocked"
    assert result["external_blocked"] is True
    assert result["incomplete"] is True
    assert result["evaluator_human_agreement_kappa"] is None
    assert result["external_human_calibration_error"] is None
    assert set(result["blocked_claims"]) == set(BLOCKED_CLAIMS)
    assert all(v == "external_blocked" for v in result["qualified_claims"].values())
    assert result["training_excluded"] is True


def test_frozen_sie005_doc_exists() -> None:
    assert SIE005_FROZEN_DOC.is_file()
    payload = json.loads(SIE005_FROZEN_DOC.read_text(encoding="utf-8"))
    assert payload["related_catalogue_id"] == CATALOGUE_ID
