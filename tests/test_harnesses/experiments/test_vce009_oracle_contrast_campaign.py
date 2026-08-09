"""VCE-009 (SLM-468): fixture oracle/contrast evidence campaign tests.

Drives the shipped ``vce009_oracle_contrast_campaign`` entry points -- and,
through them, the real VCE-004/005 oracle-intervention machinery and VCE-008
leakage audit -- rather than reimplementing any of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.experiments.vce009_oracle_contrast_campaign import (
    ALL_FACTORS,
    ARM_IDS,
    ONE_FACTOR_ARMS,
    Vce009CampaignV1,
    run_campaign,
)


def test_campaign_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="n_baseline_plans"):
        Vce009CampaignV1(n_baseline_plans=0)
    with pytest.raises(ValueError, match="pool_size"):
        Vce009CampaignV1(n_baseline_plans=5, pool_size=5)
    with pytest.raises(ValueError, match="fixture"):
        Vce009CampaignV1(claim_class="promotion_candidate")
    with pytest.raises(ValueError, match="MAX_RUN_MINUTES"):
        Vce009CampaignV1(max_wall_minutes=1000.0)


def test_manifest_is_a_real_valid_campaign_with_nine_arms() -> None:
    manifest = Vce009CampaignV1().manifest()
    assert isinstance(manifest, ExperimentCampaignV1)
    arm_ids = {arm.arm_id for arm in manifest.arms}
    assert arm_ids == set(ARM_IDS)
    assert len(ARM_IDS) == 9  # none + 4 one_factor + all_oracle + shuffled + destructive + random
    assert manifest.claim_class == "fixture"
    assert manifest.claim_class not in {"promotion_candidate", "ship_gate"}
    assert manifest.locked_eval_manifest_sha256 is None
    control_arms = {arm.arm_id for arm in manifest.arms if arm.role == "control"}
    assert control_arms == {"none", "destructive", "random"}


def test_manifest_rollback_gate_is_not_vacuous() -> None:
    manifest = Vce009CampaignV1().manifest()
    gate = manifest.rollback_gates[0]
    perfect_fidelity = 1.0
    broken_fidelity = 0.5
    fires = {
        "ge": lambda v: v >= gate.threshold,
        "gt": lambda v: v > gate.threshold,
        "le": lambda v: v <= gate.threshold,
        "lt": lambda v: v < gate.threshold,
        "eq": lambda v: v == gate.threshold,
    }[gate.operator]
    assert not fires(perfect_fidelity)
    assert fires(broken_fidelity)


def test_run_campaign_end_to_end_with_real_oracle_arms(tmp_path: Path) -> None:
    campaign = Vce009CampaignV1(n_baseline_plans=3, pool_size=20)
    result = run_campaign(campaign, root=tmp_path)

    assert result["claim_class"] == "fixture"
    assert result["baselines_attempted"] == 3
    assert result["baselines_skipped"] == 0
    assert result["records_total"] == 3 * len(ARM_IDS)
    # AC: each arm declares exactly which factor(s) changed.
    assert result["declared_factor_fidelity_rate"] == 1.0
    # AC: no-op equals baseline within deterministic equality.
    assert result["no_op_control_holds"] is True

    store = CampaignStore(campaign.campaign_id, tmp_path)
    locked = store.load_experiment_campaign(campaign.campaign_id)
    assert locked.manifest_sha256 == result["manifest_sha256"]


def test_only_gold_sourced_arms_carry_the_contamination_banner(tmp_path: Path) -> None:
    """AC: oracle contamination banner survives all exports -- and only
    where it genuinely belongs (real gold-provenance content)."""
    campaign = Vce009CampaignV1(n_baseline_plans=2, pool_size=15)
    result = run_campaign(campaign, root=tmp_path)

    expected_contaminated = {*ONE_FACTOR_ARMS, "all_oracle", "shuffled"}
    expected_clean = {"none", "destructive", "random"}
    assert set(result["contaminated_arm_ids"]) == expected_contaminated
    assert set(result["expected_contaminated_arm_ids"]) == expected_contaminated

    for report in result["baseline_reports"]:
        if report["skipped"]:
            continue
        for arm_id, record in report["arms"].items():
            banner = record["contamination_banner"]
            if arm_id in expected_contaminated:
                assert banner is not None, f"arm {arm_id} must carry a contamination banner"
                assert banner.startswith("ORACLE_DIAGNOSTIC:")
            else:
                assert arm_id in expected_clean
                assert banner is None, f"arm {arm_id} must never be contamination-bannered"


def test_filter_manifest_safe_excludes_only_contaminated_records(tmp_path: Path) -> None:
    campaign = Vce009CampaignV1(n_baseline_plans=2, pool_size=15)
    result = run_campaign(campaign, root=tmp_path)
    assert result["records_safe_for_export"] == result["records_total"] - result["records_contaminated"]
    assert result["records_contaminated"] > 0  # at least one genuine gold-sourced arm ran


def test_every_record_exposes_a_comparable_compute_field(tmp_path: Path) -> None:
    """AC: all matched arms expose comparable compute/exposure fields."""
    campaign = Vce009CampaignV1(n_baseline_plans=2, pool_size=15)
    result = run_campaign(campaign, root=tmp_path)
    for report in result["baseline_reports"]:
        if report["skipped"]:
            continue
        for arm_id, record in report["arms"].items():
            assert "apply_ms" in record["compute"], f"arm {arm_id} missing compute.apply_ms"
            assert isinstance(record["compute"]["apply_ms"], (int, float))
            assert record["compute"]["apply_ms"] >= 0.0


def test_destructive_and_random_arms_never_read_real_oracle_content(tmp_path: Path) -> None:
    """AC: destructive/random controls cannot access target identity through
    hidden fields -- they must change baseline's own content only."""
    campaign = Vce009CampaignV1(n_baseline_plans=2, pool_size=15)
    result = run_campaign(campaign, root=tmp_path)
    for report in result["baseline_reports"]:
        if report["skipped"]:
            continue
        for arm_id in ("destructive", "random"):
            record = report["arms"][arm_id]
            assert record["contamination_banner"] is None
            assert set(record["changed_factors"]) <= set(ALL_FACTORS)


def test_plan_only_manifest_matches_run_campaign_arms(tmp_path: Path) -> None:
    campaign = Vce009CampaignV1(n_baseline_plans=2, pool_size=15)
    manifest = campaign.manifest()
    result = run_campaign(campaign, root=tmp_path)
    ran_arm_ids = {
        arm_id
        for report in result["baseline_reports"]
        if not report["skipped"]
        for arm_id in report["arms"]
    }
    assert ran_arm_ids == {arm.arm_id for arm in manifest.arms}
