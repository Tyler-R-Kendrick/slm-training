"""Governed proof-carrying goal-support fixture campaign tests."""

from __future__ import annotations

import json

import pytest

from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.experiments.goal_support_domain_adequacy import (
    ARM_IDS,
    METRIC_IDS,
    GoalSupportDomainAdequacyCampaignV1,
    run_campaign,
)


def test_manifest_preregisters_the_four_authority_separated_arms() -> None:
    manifest = GoalSupportDomainAdequacyCampaignV1().manifest()
    assert tuple(arm.arm_id for arm in manifest.arms) == ARM_IDS
    assert manifest.claim_class == "fixture"
    assert manifest.locked_eval_manifest_sha256 is None
    assert manifest.endpoints[0].endpoint_id == "false_hard_prune_count"
    assert manifest.rollback_gates[0].operator == "gt"


def test_campaign_runs_live_compiler_verifier_vss_replay_and_closure(tmp_path) -> None:
    campaign = GoalSupportDomainAdequacyCampaignV1()
    result = run_campaign(campaign, root=tmp_path)
    arms = result["arm_results"]
    assert tuple(arms) == ARM_IDS
    assert all(set(arm["metrics"]) == set(METRIC_IDS) for arm in arms.values())

    production = arms["goal_support_production_exact_diagnostic"]["metrics"]
    assert production["supported_action_rate"] > 0
    assert production["unsupported_action_rate"] > 0
    assert production["unknown_action_rate"] > 0
    assert production["unobserved_action_rate"] > 0
    assert production["selection_regret_rate"] == 0.25
    assert production["domain_inadequate_under_bounds_rate"] == 0.25
    assert production["obstruction_core_emission_rate"] == 1.0
    assert production["obstruction_core_replay_rate"] == 1.0

    evaluation = arms["goal_support_evaluation_oracle_diagnostic"]["metrics"]
    assert evaluation["supported_action_rate"] == 0.0
    assert evaluation["unknown_action_rate"] > 0
    assert evaluation["obstruction_core_emission_rate"] == 0.0

    certified = arms["goal_support_certified_fixture"]
    assert certified["goal_support_pruned"] == 3
    assert certified["certified_empty_domains"] == 1
    assert certified["certified_cases_skipped"] == 2
    assert certified["metrics"]["false_hard_prune_count"] == 0
    assert certified["metrics"]["support_certificate_replay_failure_count"] == 0

    persisted = json.dumps(result, sort_keys=True)
    assert "root =" not in persisted
    assert ":slot_" not in persisted
    assert result["checkpoint_created"] is False
    assert result["model_card_updated"] is False

    events = CampaignStore(campaign.campaign_id, tmp_path).verify_event_chain()
    event_types = [event["event_type"] for event in events]
    assert (
        event_types.index("experiment_campaign_locked")
        < event_types.index("experiment_started")
        < event_types.index("experiment_finished")
    )


def test_query_cap_exhaustion_is_unobserved_not_unsupported(tmp_path) -> None:
    result = run_campaign(
        GoalSupportDomainAdequacyCampaignV1(goal_support_query_cap=0),
        root=tmp_path,
    )
    for arm_id in ARM_IDS[1:]:
        metrics = result["arm_results"][arm_id]["metrics"]
        assert metrics["exact_action_coverage_rate"] == 0.0
        assert metrics["unsupported_action_rate"] == 0.0
        assert metrics["unobserved_action_rate"] == 1.0
        assert metrics["coverage_unknown_rate"] == 1.0
    certified = result["arm_results"]["goal_support_certified_fixture"]
    assert certified["goal_support_pruned"] == 0
    assert certified["metrics"]["false_hard_prune_count"] == 0


def test_independent_runs_have_identical_semantic_digest(tmp_path) -> None:
    campaign = GoalSupportDomainAdequacyCampaignV1()
    first = run_campaign(campaign, root=tmp_path / "a")
    second = run_campaign(campaign, root=tmp_path / "b")
    assert first["canonical_result_digest"] == second["canonical_result_digest"]


def test_campaign_enforces_declared_wall_cap(tmp_path) -> None:
    campaign = GoalSupportDomainAdequacyCampaignV1(max_wall_minutes=1e-12)
    with pytest.raises(TimeoutError, match="max_wall_minutes"):
        run_campaign(campaign, root=tmp_path)
