"""RSP-006 (SLM-481): EXP-SR-10 quality-diversity corpus generation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.experiments.exp_sr_catalogue import exp_sr_campaign
from slm_training.harnesses.experiments.rsp006_quality_diversity import (
    ARM_IDS,
    CATALOGUE_ID,
    PRIMARY_METRIC,
    QD_DESCRIPTOR_AXES,
    QD_DESCRIPTOR_VERSION,
    Rsp006CampaignV1,
    compute_corpus_topology_coverage,
    enumerate_descriptor_grid,
    plan_only_preview,
    run_arm,
    run_campaign,
)


def test_campaign_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="generation_budget"):
        Rsp006CampaignV1(generation_budget=0)
    with pytest.raises(ValueError, match="shards"):
        Rsp006CampaignV1(shards=0)
    with pytest.raises(ValueError, match="fixture"):
        Rsp006CampaignV1(claim_class="diagnostic")
    with pytest.raises(ValueError, match="MAX_RUN_MINUTES"):
        Rsp006CampaignV1(max_wall_minutes=1000.0)


def test_manifest_binds_catalogue_identity() -> None:
    campaign = Rsp006CampaignV1()
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
    assert preview["automatic_adoption"] is False
    assert preview["eval_target_access"] is False
    assert set(preview["fixture_arms"]) == set(ARM_IDS)
    assert preview["qd_descriptor_version"] == QD_DESCRIPTOR_VERSION
    assert preview["qd_descriptor_axes"] == list(QD_DESCRIPTOR_AXES)
    assert preview["descriptor_grid_size"] > 0


def test_descriptor_grid_is_non_empty_and_frozen_version() -> None:
    grid = enumerate_descriptor_grid()
    assert len(grid) >= 8
    assert all(len(key) == len(QD_DESCRIPTOR_AXES) for key in grid)


def test_run_arm_is_deterministic() -> None:
    first = run_arm("random_matched", generation_budget=24, seed=3)
    second = run_arm("random_matched", generation_budget=24, seed=3)
    assert first == second


def test_run_arm_rejects_unknown_arm() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        run_arm("not_an_arm", generation_budget=1, seed=0)


def test_quality_diversity_beats_or_matches_random_at_fixture_budget() -> None:
    random_arm = run_arm("random_matched", generation_budget=48, seed=0)
    qd_arm = run_arm("quality_diversity", generation_budget=48, seed=0)
    assert qd_arm.corpus_topology_coverage >= random_arm.corpus_topology_coverage


def test_equivalent_roots_do_not_inflate_accepted_count() -> None:
    qd = run_arm("quality_diversity", generation_budget=48, seed=1)
    assert qd.duplicate_roots_skipped >= 0
    assert qd.unique_canonical_roots >= qd.accepted


def test_compute_corpus_topology_coverage_bounds() -> None:
    grid = frozenset({("1", "star", "TextContent", "1", "plain")})
    assert compute_corpus_topology_coverage(covered_keys=set(), grid_keys=grid) == 0.0
    assert (
        compute_corpus_topology_coverage(covered_keys=grid, grid_keys=grid) == 1.0
    )


def test_run_campaign_end_to_end(tmp_path: Path) -> None:
    campaign = Rsp006CampaignV1(generation_budget=24, seed=0)
    result = run_campaign(campaign, root=tmp_path)

    assert result["catalogue_id"] == CATALOGUE_ID
    assert result["claim_class"] == "fixture"
    assert result["promotion"] is False
    assert result["automatic_adoption"] is False
    assert PRIMARY_METRIC in result
    assert set(result["arm_results"]) == set(ARM_IDS)
    assert "comparison" in result
    assert "recommendation" in result

    store = CampaignStore(campaign.campaign_id, tmp_path)
    locked = store.load_experiment_campaign(CATALOGUE_ID)
    assert locked.manifest_sha256 == result["manifest_sha256"]
