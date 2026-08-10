"""SIE-008 (SLM-484): fixture-scale EXP-SR-5 value-of-compute controller tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.experiments.exp_sr_catalogue import exp_sr_campaign
from slm_training.harnesses.experiments.sie008_voc_controller import (
    ARM_IDS,
    CATALOGUE_ID,
    PRIMARY_METRIC,
    SYMBOLIC_RULE,
    Sie008CampaignV1,
    compute_recommendation,
    fixture_decode_records,
    hand_threshold_recommend,
    legal_support_parity,
    plan_only_preview,
    replay_arm,
    run_campaign,
    singleton_bypass_audit,
    split_discovery_eval,
)
from slm_training.harnesses.experiments.compute_state_controller import (
    features_from_decode_record,
)


def test_campaign_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="n_records"):
        Sie008CampaignV1(n_records=10)
    with pytest.raises(ValueError, match="discovery_fraction"):
        Sie008CampaignV1(discovery_fraction=0.95)
    with pytest.raises(ValueError, match="fixture"):
        Sie008CampaignV1(claim_class="promotion_candidate")
    with pytest.raises(ValueError, match="MAX_RUN_MINUTES"):
        Sie008CampaignV1(max_wall_minutes=1000.0)
    with pytest.raises(ValueError, match="seeds"):
        Sie008CampaignV1(seeds=())


def test_manifest_binds_catalogue_identity_and_arms() -> None:
    campaign = Sie008CampaignV1()
    manifest = campaign.manifest()
    catalogue = exp_sr_campaign(CATALOGUE_ID)
    assert isinstance(manifest, ExperimentCampaignV1)
    assert manifest.experiment_id == CATALOGUE_ID
    assert manifest.claim_class == "fixture"
    assert {arm.arm_id for arm in manifest.arms} == set(ARM_IDS)
    assert manifest.endpoints[0].metric == PRIMARY_METRIC
    assert manifest.endpoints[0].direction == "decrease"
    assert manifest.endpoints[0].minimum_effect == catalogue.endpoints[0].minimum_effect
    roles = {arm.arm_id: arm.role for arm in manifest.arms}
    assert roles["symbolic_rule"] == "candidate"
    assert roles["hand_threshold"] == "control"
    assert roles["gold_oracle"] == "control"


def test_plan_only_preview_exposes_catalogue_and_fixture_arms() -> None:
    preview = plan_only_preview()
    assert preview["status"] == "plan_only"
    assert preview["manifest"]["experiment_id"] == CATALOGUE_ID
    assert preview["fixture_arms"] == list(ARM_IDS)
    assert preview["claim_class_execution"] == "fixture"
    assert preview["promotion"] is False


def test_discovery_eval_split_is_disjoint() -> None:
    records = fixture_decode_records(seed=0, n=40)
    discovery, eval_records = split_discovery_eval(records, seed=0)
    assert discovery
    assert eval_records
    assert {r.record_digest for r in discovery} & {r.record_digest for r in eval_records} == set()


def test_singleton_bypass_holds_on_fixture() -> None:
    records = fixture_decode_records(seed=1, n=60)
    audit = singleton_bypass_audit(records)
    assert audit["singleton_bypass_holds"] is True


def test_hand_threshold_respects_singleton() -> None:
    records = fixture_decode_records(seed=2, n=30)
    singleton = next(r for r in records if features_from_decode_record(r).is_exact_singleton)
    rec = hand_threshold_recommend(features_from_decode_record(singleton))
    assert rec.expressions_evaluated == 0
    assert rec.action == "deterministic_ranker"


def test_legal_support_parity_exact() -> None:
    parity = legal_support_parity()
    assert parity["legal_support_parity_exact"] is True


def test_gold_oracle_zero_regret_on_eval() -> None:
    records = fixture_decode_records(seed=3, n=40)
    _, eval_records = split_discovery_eval(records, seed=3)
    metrics = replay_arm("gold_oracle", eval_records)
    assert metrics["compute_value_regret"] == 0.0


def test_symbolic_rule_is_preregistered_ir() -> None:
    assert SYMBOLIC_RULE.rule_id == "sie008-symbolic-v1"
    assert "neural_forward" in SYMBOLIC_RULE.expressions


def test_compute_recommendation_dispositions() -> None:
    adopt = compute_recommendation(
        symbolic_regret=0.1,
        control_regret=0.2,
        minimum_effect=0.05,
        falsifier_holds=False,
        safety_ok=True,
    )
    assert adopt["disposition"] == "adopt_optional"

    inconclusive = compute_recommendation(
        symbolic_regret=0.15,
        control_regret=0.17,
        minimum_effect=0.05,
        falsifier_holds=False,
        safety_ok=True,
    )
    assert inconclusive["disposition"] == "inconclusive"

    reject = compute_recommendation(
        symbolic_regret=0.3,
        control_regret=0.2,
        minimum_effect=0.05,
        falsifier_holds=True,
        safety_ok=True,
    )
    assert reject["disposition"] == "reject"


def test_run_campaign_end_to_end(tmp_path: Path) -> None:
    campaign = Sie008CampaignV1(n_records=48, seeds=(0, 1))
    result = run_campaign(campaign, root=tmp_path)

    assert result["claim_class"] == "fixture"
    assert result["promotion"] is False
    assert result["catalogue_id"] == CATALOGUE_ID
    assert result["legal_support_parity"]["legal_support_parity_exact"] is True
    assert 0.0 <= result["compute_value_regret"] <= 1.0
    assert set(result["aggregate_arms"]) == set(ARM_IDS)
    assert result["aggregate_arms"]["gold_oracle"]["compute_value_regret_mean"] == 0.0
    assert result["recommendation"]["disposition"] in {
        "adopt_optional",
        "reject",
        "inconclusive",
    }

    store = CampaignStore(campaign.campaign_id, tmp_path)
    locked = store.load_experiment_campaign(CATALOGUE_ID)
    assert locked.manifest_sha256 == result["manifest_sha256"]
