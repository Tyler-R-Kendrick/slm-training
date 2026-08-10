"""SIE-002 (SLM-476): fixture-scale oracle localization campaign tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.storage import CampaignStore
from slm_training.data.semantic_plan.oracle import PlanInterventionRecordV1
from slm_training.harnesses.experiments.exp_sr_catalogue import exp_sr_campaign
from slm_training.harnesses.experiments.sie002_oracle_localization import (
    ALL_FACTORS,
    ARM_IDS,
    CATALOGUE_ID,
    ONE_FACTOR_ARMS,
    Sie002CampaignV1,
    analyze_factor_localization,
    authorize_factors,
    plan_only_preview,
    run_campaign,
)
from slm_training.harnesses.experiments.vce009_oracle_contrast_campaign import (
    generate_baseline_arms,
    load_fixture_plan_pool,
)


def test_campaign_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="n_baseline_plans"):
        Sie002CampaignV1(n_baseline_plans=0)
    with pytest.raises(ValueError, match="pool_size"):
        Sie002CampaignV1(n_baseline_plans=5, pool_size=5)
    with pytest.raises(ValueError, match="fixture"):
        Sie002CampaignV1(claim_class="promotion_candidate")
    with pytest.raises(ValueError, match="MAX_RUN_MINUTES"):
        Sie002CampaignV1(max_wall_minutes=1000.0)
    with pytest.raises(ValueError, match="seeds"):
        Sie002CampaignV1(seeds=())


def test_manifest_binds_catalogue_identity_and_vce009_arms() -> None:
    campaign = Sie002CampaignV1()
    manifest = campaign.manifest()
    catalogue = exp_sr_campaign(CATALOGUE_ID)
    assert isinstance(manifest, ExperimentCampaignV1)
    assert manifest.experiment_id == CATALOGUE_ID
    assert manifest.claim_class == "fixture"
    assert manifest.claim_class not in {"promotion_candidate", "ship_gate"}
    assert {arm.arm_id for arm in manifest.arms} == set(ARM_IDS)
    assert set(ONE_FACTOR_ARMS) <= {arm.arm_id for arm in manifest.arms}
    assert manifest.endpoints[0].metric == "oracle_factor_localization_precision"
    assert manifest.endpoints[0].minimum_effect == catalogue.endpoints[0].minimum_effect
    assert manifest.seeds == (0, 1, 2)


def test_plan_only_preview_exposes_catalogue_and_fixture_arms() -> None:
    preview = plan_only_preview()
    assert preview["status"] == "plan_only"
    assert preview["manifest"]["experiment_id"] == CATALOGUE_ID
    assert preview["fixture_arms"] == list(ARM_IDS)
    assert preview["claim_class_execution"] == "fixture"
    assert preview["promotion"] is False


def test_authorize_factors_fail_closed_on_empty_ceiling() -> None:
    analysis = {
        "no_op_control_holds": True,
        "declared_factor_fidelity_rate": 1.0,
        "arm_change_rates": {"all_oracle": 0.0},
        "per_factor": {
            factor: {
                "change_rate": 1.0,
                "mechanical_one_factor_fidelity": 1.0,
                "ceiling_presence_n": 0,
                "ceiling_recovery_rate": 0.0,
            }
            for factor in ALL_FACTORS
        },
    }
    auth = authorize_factors(analysis, leakage_passed=True, minimum_effect=0.03)
    assert auth["authorized_factors"] == []
    assert auth["falsifier_holds"] is True


def test_run_campaign_end_to_end_composes_vce009_generators(tmp_path: Path) -> None:
    pool = load_fixture_plan_pool(limit=8)
    assert len(pool) >= 2
    _ = generate_baseline_arms(
        pool[0][0], pool[0][1], [p for _, p in pool[1:]], rng_seed=0
    )

    campaign = Sie002CampaignV1(n_baseline_plans=2, pool_size=12, seeds=(0, 1))
    result = run_campaign(campaign, root=tmp_path)

    assert result["claim_class"] == "fixture"
    assert result["promotion"] is False
    assert result["catalogue_id"] == CATALOGUE_ID
    assert result["leakage_audit"]["passed"] is True
    assert result["analysis"]["no_op_control_holds"] is True
    assert result["analysis"]["declared_factor_fidelity_rate"] == 1.0
    assert 0.0 <= result["oracle_factor_localization_precision"] <= 1.0
    assert set(result["authorized_factors"]) <= set(ALL_FACTORS)
    assert len(result["seed_runs"]) == 2

    store = CampaignStore(campaign.campaign_id, tmp_path)
    # Store locks are keyed by experiment_id (catalogue id), not campaign_id.
    locked = store.load_experiment_campaign(CATALOGUE_ID)
    assert locked.manifest_sha256 == result["manifest_sha256"]


def test_analyze_localization_scores_ceiling_recovery(tmp_path: Path) -> None:
    campaign = Sie002CampaignV1(n_baseline_plans=2, pool_size=12, seeds=(0,))
    result = run_campaign(campaign, root=tmp_path)
    reports = []
    for report in result["seed_runs"][0]["baseline_reports"]:
        if report["skipped"]:
            reports.append({"skipped": True})
            continue
        reports.append(
            {
                "skipped": False,
                "arms": {
                    arm: PlanInterventionRecordV1.from_dict(payload)
                    for arm, payload in report["arms"].items()
                },
            }
        )
    live = analyze_factor_localization(reports)
    assert live["oracle_factor_localization_precision"] == result["seed_runs"][0][
        "analysis"
    ]["oracle_factor_localization_precision"]
