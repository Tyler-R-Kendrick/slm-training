"""RSP-007 (SLM-486): EXP-SR-11 matched PySR/SRBench benchmark tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.storage import CampaignStore
from slm_training.dsl.symbolic_expr_pysr_adapter import DEFAULT_PYSR_MANIFEST
from slm_training.harnesses.experiments.exp_sr_catalogue import exp_sr_campaign
from slm_training.harnesses.experiments.rsp007_pysr_srbench import (
    ARM_IDS,
    CATALOGUE_ID,
    EVIDENCE_EXTERNAL_BLOCKED,
    MATCHED_ITERATION_BUDGET,
    PRIMARY_METRIC,
    Rsp007CampaignV1,
    compute_matched_score_gap,
    dry_run_pysr_adapter,
    plan_only_preview,
    probe_pysr_environment,
    run_campaign,
    run_pack_enumerate_arm,
    run_pysr_adapter_arm,
)
from slm_training.harnesses.experiments.srp010_pack_portability import (
    build_tiny_sr_problem,
)


def test_campaign_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="diagnostic"):
        Rsp007CampaignV1(claim_class="fixture")
    with pytest.raises(ValueError, match="MAX_RUN_MINUTES"):
        Rsp007CampaignV1(max_wall_minutes=1000.0)


def test_manifest_binds_catalogue_identity() -> None:
    campaign = Rsp007CampaignV1()
    manifest = campaign.manifest()
    catalogue = exp_sr_campaign(CATALOGUE_ID)
    assert isinstance(manifest, ExperimentCampaignV1)
    assert manifest.experiment_id == CATALOGUE_ID
    assert manifest.claim_class == "diagnostic"
    assert manifest.claim_class not in {"promotion_candidate", "ship_gate"}
    assert {arm.arm_id for arm in manifest.arms} == set(ARM_IDS)
    assert manifest.endpoints[0].metric == PRIMARY_METRIC
    assert manifest.endpoints[0].metric == catalogue.endpoints[0].metric
    assert manifest.rollback_gates[0].threshold == catalogue.rollback_gates[0].threshold


def test_plan_only_preview_exposes_catalogue_and_arms() -> None:
    preview = plan_only_preview()
    assert preview["status"] == "plan_only"
    assert preview["manifest"]["experiment_id"] == CATALOGUE_ID
    assert preview["claim_class_execution"] == "diagnostic"
    assert preview["promotion"] is False
    assert preview["no_sota_claims"] is True
    assert preview["evidence_when_blocked"] == EVIDENCE_EXTERNAL_BLOCKED
    assert set(preview["fixture_arms"]) == set(ARM_IDS)
    assert preview["matched_iteration_budget"] == MATCHED_ITERATION_BUDGET


def test_probe_pysr_environment_records_manifest() -> None:
    probe = probe_pysr_environment()
    assert probe["manifest_fingerprint"] == DEFAULT_PYSR_MANIFEST.fingerprint()
    assert probe["tool_name"] == "pysr"
    assert "pysr_import_available" in probe


def test_dry_run_adapter_is_external_blocked() -> None:
    problem = build_tiny_sr_problem()
    adapter_input, result = dry_run_pysr_adapter(problem)
    assert adapter_input.problem_fingerprint == problem.fingerprint
    assert result.status == "external_blocked"
    assert result.evidence is None


def test_pack_enumerate_arm_is_complete() -> None:
    problem = build_tiny_sr_problem()
    arm = run_pack_enumerate_arm(problem)
    assert arm["status"] == "complete"
    assert arm["validation_loss"] is not None
    assert arm["matched_iteration_budget"] == MATCHED_ITERATION_BUDGET


def test_pysr_arm_blocked_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "slm_training.harnesses.experiments.rsp007_pysr_srbench.probe_pysr_environment",
        lambda manifest=DEFAULT_PYSR_MANIFEST: {
            "tool_name": manifest.tool_name,
            "tool_version": manifest.tool_version,
            "python_version": "3.11",
            "manifest_fingerprint": manifest.fingerprint(),
            "pysr_import_available": False,
            "live_execution_ready": False,
        },
    )
    problem = build_tiny_sr_problem()
    arm = run_pysr_adapter_arm(problem)
    assert arm["status"] == "incomplete"
    assert arm["evidence"] == EVIDENCE_EXTERNAL_BLOCKED
    assert arm["validation_loss"] is None
    assert arm["adapter_status"] == "external_blocked"
    assert arm["dry_run"] is True


def test_gap_not_scored_when_external_blocked() -> None:
    pack = {"validation_loss": 0.1, "status": "complete", "evidence": "pack_enumerate"}
    pysr = {
        "validation_loss": None,
        "status": "incomplete",
        "evidence": EVIDENCE_EXTERNAL_BLOCKED,
    }
    gap = compute_matched_score_gap(pack, pysr)
    assert gap[PRIMARY_METRIC] is None
    assert gap["evidence"] == EVIDENCE_EXTERNAL_BLOCKED
    assert gap["complete"] is False


def test_gap_scored_when_both_complete() -> None:
    pack = {"validation_loss": 0.5, "status": "complete", "evidence": "pack_enumerate"}
    pysr = {"validation_loss": 0.2, "status": "complete", "evidence": "pysr_adapter"}
    gap = compute_matched_score_gap(pack, pysr)
    assert gap[PRIMARY_METRIC] == pytest.approx(0.3)
    assert gap["complete"] is True


def test_run_campaign_end_to_end(tmp_path: Path) -> None:
    campaign = Rsp007CampaignV1(seed=0)
    result = run_campaign(campaign, root=tmp_path)

    assert result["catalogue_id"] == CATALOGUE_ID
    assert result["claim_class"] == "diagnostic"
    assert result["promotion"] is False
    assert result["no_sota_claims"] is True
    assert result["pack_enumerate_matched"]["status"] == "complete"
    assert result["pysr_adapter_matched"]["status"] in {"incomplete", "complete"}
    assert "env_probe" in result
    assert "gap_analysis" in result

    store = CampaignStore(campaign.campaign_id, tmp_path)
    locked = store.load_experiment_campaign(CATALOGUE_ID)
    assert locked.manifest_sha256 == result["manifest_sha256"]

    if not result["complete"]:
        assert result["evidence"] == EVIDENCE_EXTERNAL_BLOCKED
        assert result[PRIMARY_METRIC] is None
