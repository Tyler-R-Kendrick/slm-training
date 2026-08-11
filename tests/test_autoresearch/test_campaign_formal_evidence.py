"""INTEG-03 (SLM-556): formal↔empirical campaign evidence linkage."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.autoresearch.campaign_formal_evidence import (
    CampaignFormalEmpiricalSplitV1,
    build_formal_empirical_split,
    decision_support_report,
    empirical_status_from_result,
    formal_status_from_authority,
    link_four_axis_and_revmath,
    migrate_campaign_result_payload,
    validate_formal_empirical_separation,
)
from slm_training.autoresearch.experiment_campaign import (
    CampaignLockV1,
    CampaignResultV1,
    ExperimentCampaignV1,
    campaign_manifest_sha256,
)
from slm_training.lineage.records import content_sha


def _minimal_manifest(**updates: object) -> ExperimentCampaignV1:
    payload: dict[str, object] = {
        "campaign_id": "integ03-campaign",
        "experiment_id": "integ03-exp",
        "hypothesis": "Formal and empirical statuses stay independent under INTEG-03.",
        "decision": "Whether a proved preflight can falsely authorize ship success.",
        "endpoints": [
            {
                "endpoint_id": "primary_metric",
                "metric": "parse_rate",
                "role": "primary",
                "direction": "increase",
                "minimum_effect": 0.01,
            }
        ],
        "arms": [
            {
                "arm_id": "control",
                "role": "control",
                "config_sha256": "a" * 64,
            },
            {
                "arm_id": "candidate",
                "role": "candidate",
                "config_sha256": "b" * 64,
            },
        ],
        "seeds": [1],
        "budget": {
            "max_experiments": 2,
            "max_wall_minutes": 5.0,
            "max_gpu_hours": 0.0,
        },
        "stopping_rules": ["single seed fixture stop"],
        "controls": [
            {
                "control_id": "neg",
                "description": "destructive negative control arm",
                "kind": "negative",
            }
        ],
        "negative_controls": ["neg"],
        "multiplicity_families": [
            {
                "family_id": "primary_family",
                "hypothesis_ids": ["primary_metric"],
                "alpha": 0.05,
            }
        ],
        "promotion_gates": [
            {
                "gate_id": "pg",
                "endpoint_id": "primary_metric",
                "operator": "ge",
                "threshold": 0.0,
            }
        ],
        "rollback_gates": [
            {
                "gate_id": "rg",
                "endpoint_id": "primary_metric",
                "operator": "lt",
                "threshold": -1.0,
            }
        ],
        "artifact_requirements": [{"kind": "version_stamp", "minimum_count": 1}],
        "claim_class": "fixture",
        "source_commit": "0" * 40,
        "source_dirty": False,
        "author": "integ-03",
        "created_at": "1970-01-01T00:00:00Z",
    }
    payload.update(updates)
    return ExperimentCampaignV1.model_validate(payload)


def test_historical_result_without_split_still_validates() -> None:
    manifest = _minimal_manifest()
    digest = campaign_manifest_sha256(manifest)
    raw = {
        "campaign_id": manifest.campaign_id,
        "experiment_id": manifest.experiment_id,
        "manifest_sha256": digest,
        "claim_class": "fixture",
        "exploratory": True,
        "ship_gates_passed": False,
    }
    migrated = migrate_campaign_result_payload(raw)
    assert "formal_empirical" not in migrated
    result = CampaignResultV1.model_validate(migrated)
    assert result.formal_empirical is None
    report = decision_support_report(result.formal_empirical)
    assert report["present"] is False


def test_lock_digest_unchanged_when_result_carries_split() -> None:
    manifest = _minimal_manifest()
    digest = campaign_manifest_sha256(manifest)
    lock = CampaignLockV1(manifest_sha256=digest, manifest=manifest)
    assert lock.manifest_sha256 == digest
    # Re-dumping the manifest without formal_empirical fields keeps the digest.
    restored = ExperimentCampaignV1.model_validate(manifest.model_dump(mode="json"))
    assert campaign_manifest_sha256(restored) == digest


def test_proved_preflight_failed_empirical_does_not_conflate() -> None:
    split = link_four_axis_and_revmath(
        formal_status="proved",
        empirical_status="failure",
        formal_authority_digests=["c" * 64],
        four_axis_ledger_sha256="d" * 64,
        revmath_report_sha256="e" * 64,
        assumption_axis_status="proved_axis",
        refinement_axis_status="unknown_axis",
        notes="proved preflight + failed empirical",
    )
    assert split.formal_status == "proved"
    assert split.empirical_status == "failure"
    failures = validate_formal_empirical_separation(
        formal_status=split.formal_status,
        empirical_status=split.empirical_status,
        ship_gates_passed=True,  # illegal: formal must not stamp ship
        claim_class="ship_gate",
        exploratory=False,
    )
    assert "formal_cannot_authorize_ship_gates" in failures
    assert "empirical_status_blocks_ship_gates" in failures

    ok = validate_formal_empirical_separation(
        formal_status=split.formal_status,
        empirical_status=split.empirical_status,
        ship_gates_passed=False,
        claim_class="ship_gate",
        exploratory=False,
    )
    assert ok == ()

    manifest = _minimal_manifest()
    result = CampaignResultV1(
        campaign_id=manifest.campaign_id,
        experiment_id=manifest.experiment_id,
        manifest_sha256=campaign_manifest_sha256(manifest),
        claim_class="fixture",
        exploratory=True,
        ship_gates_passed=False,
        formal_empirical=split,
    )
    report = decision_support_report(result.formal_empirical)
    assert report["separation"]["formal_authorizes_empirical"] is False
    assert report["axes"]["assumption_strength"] == "proved_axis"
    assert report["revmath_report_sha256"] == "e" * 64


def test_weak_formal_successful_empirical_allowed() -> None:
    split = link_four_axis_and_revmath(
        formal_status="unknown",
        empirical_status="success",
        revmath_report_sha256="f" * 64,
        notes="weak formal + successful empirical",
    )
    failures = validate_formal_empirical_separation(
        formal_status=split.formal_status,
        empirical_status=split.empirical_status,
        ship_gates_passed=True,
        claim_class="promotion_candidate",
        exploratory=False,
    )
    assert failures == ()
    report = decision_support_report(split)
    assert report["separation"]["weak_or_unknown_formal"] is True
    assert report["separation"]["empirical_success"] is True


def test_content_bound_refs_replayable() -> None:
    split = build_formal_empirical_split(
        formal_status="unchecked",
        empirical_status="inconclusive",
        refs=[
            {
                "kind": "formal_authority_v2",
                "sha256": "1" * 64,
                "label": "formal_authority/v2",
            },
            {
                "kind": "bound_ast",
                "sha256": "2" * 64,
                "label": "bound.unit_work",
            },
            {
                "kind": "runtime_refinement_trace",
                "sha256": "3" * 64,
                "label": "kern11_slot",
                "notes": "optional until KERN-11 lands",
            },
        ],
        decision_support={
            "bound_ast_ids": ["bound.unit_work"],
            "formal_authority_digests": ["1" * 64],
            "runtime_refinement_trace_sha256": "3" * 64,
            "notes": "mixed refs",
        },
    )
    dumped = split.model_dump(mode="json")
    restored = CampaignFormalEmpiricalSplitV1.model_validate(dumped)
    assert restored.content_digest == split.content_digest
    tampered = dict(dumped)
    tampered["formal_status"] = "proved"
    with pytest.raises(ValueError, match="content_digest mismatch"):
        CampaignFormalEmpiricalSplitV1.model_validate(tampered)


def test_migrate_mixed_version_seals_missing_digest() -> None:
    body = {
        "schema_version": "campaign_formal_empirical_split/v1",
        "formal_status": "absent",
        "empirical_status": "not_run",
        "refs": [],
        "empirical_remainder_claim_ids": [],
        "decision_support": None,
    }
    raw = {
        "campaign_id": "c",
        "experiment_id": "e",
        "manifest_sha256": "a" * 64,
        "claim_class": "fixture",
        "formal_empirical": body,
    }
    migrated = migrate_campaign_result_payload(raw)
    assert "content_digest" in migrated["formal_empirical"]
    split = CampaignFormalEmpiricalSplitV1.model_validate(migrated["formal_empirical"])
    assert split.content_digest == content_sha(body)


def test_empirical_status_ignores_formal_proof() -> None:
    status = empirical_status_from_result(
        ship_gates_passed=False,
        exploratory=False,
        claim_class="promotion_candidate",
        endpoint_success=False,
        runs_failed=False,
    )
    assert status == "failure"
    assert (
        formal_status_from_authority(
            {
                "authority_class": "checked_semantic",
                "judgment": {"outcome": "witnessed"},
            }
        )
        == "proved"
    )


def test_validate_result_claim_rejects_formal_ship_stamp(tmp_path: Path) -> None:
    from slm_training.autoresearch.experiment_campaign import validate_result_claim

    manifest = _minimal_manifest()
    split = link_four_axis_and_revmath(
        formal_status="proved",
        empirical_status="failure",
        formal_authority_digests=["9" * 64],
    )
    result = CampaignResultV1(
        campaign_id=manifest.campaign_id,
        experiment_id=manifest.experiment_id,
        manifest_sha256=campaign_manifest_sha256(manifest),
        claim_class="fixture",
        exploratory=True,
        ship_gates_passed=True,
        formal_empirical=split,
    )
    failures = validate_result_claim(manifest, result, artifact_root=tmp_path)
    assert "formal_cannot_authorize_ship_gates" in failures
    assert "empirical_status_blocks_ship_gates" in failures
