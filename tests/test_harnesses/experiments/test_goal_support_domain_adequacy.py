"""PGS-H01 (SLM-510): goal-support domain-adequacy campaign harness tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.storage import CampaignStore
from slm_training.dsl.solver.goal_support import exact_goal_closure
from slm_training.harnesses.experiments.goal_support_domain_adequacy import (
    ARM_IDS,
    CAMPAIGN_ID,
    EXPERIMENT_ID,
    FIXTURE_IDS,
    PRIMARY_METRIC,
    GoalSupportDomainAdequacyCampaignV1,
    _profile_for_mode,
    build_fixture_matrix,
    canonical_result_digest,
    evaluate_falsifiers,
    plan_only_preview,
    render_markdown,
    run_campaign,
    run_fixture_arm,
)


def test_campaign_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="MAX_RUN_MINUTES"):
        GoalSupportDomainAdequacyCampaignV1(max_wall_minutes=1000.0)
    with pytest.raises(ValueError, match="promotion"):
        GoalSupportDomainAdequacyCampaignV1(claim_class="promotion_candidate")  # type: ignore[arg-type]


def test_manifest_binds_arms_endpoints_and_falsifiers() -> None:
    campaign = GoalSupportDomainAdequacyCampaignV1()
    manifest = campaign.manifest()
    assert isinstance(manifest, ExperimentCampaignV1)
    assert manifest.campaign_id == CAMPAIGN_ID
    assert manifest.experiment_id == EXPERIMENT_ID
    assert manifest.claim_class == "diagnostic"
    assert manifest.claim_class not in {"promotion_candidate", "ship_gate"}
    assert {arm.arm_id for arm in manifest.arms} == set(ARM_IDS)
    assert manifest.endpoints[0].metric == PRIMARY_METRIC
    assert manifest.rollback_gates[0].endpoint_id == f"{EXPERIMENT_ID}_falsifier"
    assert manifest.rollback_gates[0].operator == "gt"


def test_plan_only_preview_exposes_fixtures_and_arms() -> None:
    preview = plan_only_preview()
    assert preview["status"] == "plan_only"
    assert preview["fixture_ids"] == list(FIXTURE_IDS)
    assert preview["arms"] == list(ARM_IDS)
    assert preview["promotion"] is False
    assert len(preview["fixture_ids"]) >= 14


def test_fixture_matrix_covers_required_scenarios() -> None:
    ids = {spec.fixture_id for spec in build_fixture_matrix()}
    required = {
        "exact_satisfying_alternative",
        "exact_violating_alternative",
        "structural_ne_goal_support",
        "bounded_domain_inadequacy",
        "unknown_mandatory_evidence",
        "partial_forest",
        "bound_exhaustion",
        "advisory_cannot_prune",
        "evaluation_oracle_cannot_prune",
        "cap_excludes_unique_support",
        "certified_singleton_zero_forward",
    }
    assert required <= ids


def test_arm_isolation_evaluation_oracle_cannot_call_certified_closure() -> None:
    profile = _profile_for_mode("evaluation_oracle")
    spec = next(s for s in build_fixture_matrix() if s.fixture_id == "bounded_domain_inadequacy")
    from slm_training.harnesses.experiments.goal_support_domain_adequacy import _build_provider

    expander, provider, _ = _build_provider(spec, profile=profile)
    assert provider is not None
    with pytest.raises(ValueError, match="evaluation_oracle"):
        exact_goal_closure(expander.root_state(), provider)


def test_advisory_fixture_never_claims_domain_inadequate() -> None:
    spec = next(s for s in build_fixture_matrix() if s.fixture_id == "advisory_cannot_prune")
    metrics = run_fixture_arm(spec, "goal_support_production_exact_diagnostic")
    assert not metrics.skipped
    assert metrics.classification == "coverage_unknown"
    assert not metrics.domain_inadequate_under_bounds


def test_certified_fixture_skipped_for_advisory_and_eval_only_fixtures() -> None:
    for fixture_id in ("advisory_cannot_prune", "evaluation_oracle_cannot_prune"):
        spec = next(s for s in build_fixture_matrix() if s.fixture_id == fixture_id)
        metrics = run_fixture_arm(spec, "goal_support_certified_fixture")
        assert metrics.skipped


def test_bounded_domain_inadequacy_under_production_exact() -> None:
    spec = next(s for s in build_fixture_matrix() if s.fixture_id == "bounded_domain_inadequacy")
    metrics = run_fixture_arm(spec, "goal_support_production_exact_diagnostic")
    assert metrics.classification == "domain_inadequate_under_bounds"
    assert metrics.obstruction_core_emitted >= 1


def test_deterministic_rerun_digest_stable(tmp_path: Path) -> None:
    campaign = GoalSupportDomainAdequacyCampaignV1()
    a = run_campaign(campaign, root=tmp_path / "a")
    b = run_campaign(campaign, root=tmp_path / "b")
    assert a["result_digest"] == b["result_digest"]
    assert canonical_result_digest(a) == canonical_result_digest(b)


def test_run_campaign_end_to_end(tmp_path: Path) -> None:
    campaign = GoalSupportDomainAdequacyCampaignV1()
    result = run_campaign(campaign, root=tmp_path)
    assert result["claim_class"] == "diagnostic"
    assert result["promotion"] is False
    assert set(result["aggregate_arms"]) == set(ARM_IDS)
    assert result["falsifier"]["false_hard_prune_count"] == 0
    assert result["falsifier"]["evaluation_oracle_called_certified_closure"] is False
    assert len(result["fixture_results"]) == len(FIXTURE_IDS)

    store = CampaignStore(campaign.campaign_id, tmp_path)
    locked = store.load_experiment_campaign(EXPERIMENT_ID)
    assert locked.manifest_sha256 == result["manifest_sha256"]


def test_result_renderer_never_uses_global_unsat_language(tmp_path: Path) -> None:
    campaign = GoalSupportDomainAdequacyCampaignV1()
    result = run_campaign(campaign, root=tmp_path)
    md = render_markdown(result)
    lowered = md.lower()
    for term in ("certified_unsat", "globally unsatisfiable"):
        assert term not in lowered


def test_evaluate_falsifiers_detects_false_hard_prune() -> None:
    from slm_training.harnesses.experiments.goal_support_domain_adequacy import (
        FixtureArmMetrics,
    )

    bad = FixtureArmMetrics(
        fixture_id="x",
        arm_id="goal_support_certified_fixture",
        false_hard_prune_count=1,
    )
    good = FixtureArmMetrics(
        fixture_id="x",
        arm_id="goal_support_production_exact_diagnostic",
        classification="coverage_unknown",
    )
    falsifier = evaluate_falsifiers({"x": {"goal_support_certified_fixture": bad, "goal_support_production_exact_diagnostic": good}})
    assert falsifier["falsifier_holds"] is True
    assert falsifier["false_hard_prune_count"] == 1


def test_result_schema_serializes_to_json(tmp_path: Path) -> None:
    campaign = GoalSupportDomainAdequacyCampaignV1()
    result = run_campaign(campaign, root=tmp_path)
    payload = json.loads(json.dumps(result))
    assert payload["schema"].endswith("/v1")
    assert "result_digest" in payload
