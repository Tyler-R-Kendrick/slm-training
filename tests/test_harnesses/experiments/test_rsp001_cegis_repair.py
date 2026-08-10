"""RSP-001 (SLM-488): fixture-scale EXP-SR-4 CEGIS repair campaign tests."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.experiments.exp_sr_catalogue import exp_sr_campaign
from slm_training.harnesses.experiments.rsp001_cegis_repair import (
    ARM_IDS,
    CATALOGUE_ID,
    PRIMARY_METRIC,
    SEMANTIC_ARM_IDS,
    TOPOLOGY_ARM_IDS,
    Rsp001CampaignV1,
    build_topology_fixture,
    load_semantic_corpus,
    plan_only_preview,
    run_campaign,
    run_semantic_repair_arm,
    run_topology_repair_arm,
    run_witness_cegis_repair,
    score_verified_repair_yield,
    witness_from_record,
)


def test_campaign_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="max_records"):
        Rsp001CampaignV1(max_records=0)
    with pytest.raises(ValueError, match="max_cycles"):
        Rsp001CampaignV1(max_cycles=0)
    with pytest.raises(ValueError, match="fixture"):
        Rsp001CampaignV1(claim_class="promotion_candidate")
    with pytest.raises(ValueError, match="MAX_RUN_MINUTES"):
        Rsp001CampaignV1(max_wall_minutes=1000.0)
    with pytest.raises(ValueError, match="seeds"):
        Rsp001CampaignV1(seeds=())


def test_manifest_binds_catalogue_identity_and_arms() -> None:
    campaign = Rsp001CampaignV1()
    manifest = campaign.manifest()
    catalogue = exp_sr_campaign(CATALOGUE_ID)
    assert isinstance(manifest, ExperimentCampaignV1)
    assert manifest.experiment_id == CATALOGUE_ID
    assert manifest.claim_class == "fixture"
    assert manifest.claim_class not in {"promotion_candidate", "ship_gate"}
    assert {arm.arm_id for arm in manifest.arms} == set(ARM_IDS)
    assert manifest.endpoints[0].metric == PRIMARY_METRIC
    assert manifest.endpoints[0].minimum_effect == catalogue.endpoints[0].minimum_effect
    roles = {arm.arm_id: arm.role for arm in manifest.arms}
    assert roles["witness_cegis"] == "candidate"
    assert roles["regenerate"] == "control"


def test_plan_only_preview_exposes_catalogue_and_arms() -> None:
    preview = plan_only_preview()
    assert preview["status"] == "plan_only"
    assert preview["manifest"]["experiment_id"] == CATALOGUE_ID
    assert preview["fixture_arms"] == list(ARM_IDS)
    assert preview["semantic_arms"] == list(SEMANTIC_ARM_IDS)
    assert preview["topology_arms"] == list(TOPOLOGY_ARM_IDS)
    assert preview["claim_class_execution"] == "fixture"
    assert preview["promotion"] is False
    assert preview["default_lever_state"] == "OFF"


def test_witness_cegis_respects_legal_domain() -> None:
    records = load_semantic_corpus(max_records=5)
    assert records
    record = records[0]
    witness = witness_from_record(record)
    ok, cycles = run_witness_cegis_repair(
        record, witness=witness, max_cycles=3, rng=random.Random(0)
    )
    assert isinstance(ok, bool)
    assert cycles
    for cycle in cycles:
        if cycle.get("authorized") is False:
            assert cycle.get("verified") is False


def test_no_repair_never_succeeds() -> None:
    records = load_semantic_corpus(max_records=3)
    rng = random.Random(0)
    for record in records:
        ok, detail = run_semantic_repair_arm(
            "no_repair", record, max_cycles=3, rng=rng
        )
        assert ok is False
        assert detail["verified"] is False


def test_topology_arms_recover_on_exact_slice() -> None:
    tree, slice_ = build_topology_fixture()
    for arm in TOPOLOGY_ARM_IDS:
        ok, detail = run_topology_repair_arm(
            arm,
            tree,
            slice_,
            seed=0,
            budget_forwards=64,
            budget_verifier_calls=16,
        )
        if arm in {"conflict_slice", "conflict_slice_expanded"}:
            assert ok is True
            assert detail["protected_mutations"] == 0
        assert "remasked_node_ids" in detail


def test_score_verified_repair_yield_composites() -> None:
    yields = score_verified_repair_yield(
        semantic_successes={"witness_cegis": 2, "regenerate": 1},
        semantic_total={"witness_cegis": 4, "regenerate": 4},
        topology_successes={"conflict_slice": 3},
        topology_total={"conflict_slice": 3},
    )
    assert yields["witness_cegis"] == pytest.approx(0.5)
    assert yields["regenerate"] == pytest.approx(0.25)
    assert yields["conflict_slice"] == pytest.approx(1.0)


def test_run_campaign_end_to_end(tmp_path: Path) -> None:
    records = load_semantic_corpus(max_records=8)
    assert records

    campaign = Rsp001CampaignV1(max_records=8, seeds=(0, 1))
    result = run_campaign(campaign, root=tmp_path)

    assert result["claim_class"] == "fixture"
    assert result["promotion"] is False
    assert result["catalogue_id"] == CATALOGUE_ID
    assert result["default_lever_state"] == "OFF"
    assert result["invariants"]["default_off"] is True
    assert 0.0 <= result["verified_repair_yield"] <= 1.0
    assert set(result["aggregate_yields"]) == set(ARM_IDS)
    assert result["recommendation"] in {"adopt-optional", "reject", "inconclusive"}

    store = CampaignStore(campaign.campaign_id, tmp_path)
    locked = store.load_experiment_campaign(CATALOGUE_ID)
    assert locked.manifest_sha256 == result["manifest_sha256"]
