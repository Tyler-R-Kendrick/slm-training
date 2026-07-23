"""Focused contract tests for preregistered autoresearch campaigns."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from slm_training.autoresearch.experiment_campaign import (
    AP001CertificationV1,
    CampaignLockV1,
    CampaignResultV1,
    ExperimentCampaignV1,
    campaign_manifest_sha256,
    load_ap001_certification,
    select_primary_endpoint,
    validate_result_claim,
)
from slm_training.lineage.records import canonical_json

HEX_40 = "a" * 40
HEX_64 = "b" * 64


def _manifest_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "campaign_id": "ap-036",
        "experiment_id": "e001",
        "hypothesis": "Candidate improves the locked primary endpoint.",
        "decision": "Promote only when every preregistered gate passes.",
        "endpoints": [
            {
                "endpoint_id": "meaning",
                "metric": "binder_reference_f1",
                "role": "primary",
                "direction": "increase",
                "minimum_effect": 0.01,
            }
        ],
        "arms": [
            {"arm_id": "control", "role": "control", "config_sha256": "c" * 64},
            {
                "arm_id": "candidate",
                "role": "candidate",
                "config_sha256": "d" * 64,
            },
        ],
        "seeds": [7, 11],
        "budget": {
            "max_experiments": 2,
            "max_gpu_hours": 0,
            "max_wall_minutes": 2,
        },
        "stopping_rules": ["Stop after the declared seeds finish."],
        "controls": [
            {
                "control_id": "unchanged-baseline",
                "description": "Unchanged baseline must reproduce.",
                "kind": "negative",
            }
        ],
        "negative_controls": ["unchanged-baseline"],
        "multiplicity_families": [
            {
                "family_id": "primary",
                "hypothesis_ids": ["meaning"],
                "alpha": 0.05,
            }
        ],
        "promotion_gates": [
            {
                "gate_id": "meaning-improves",
                "endpoint_id": "meaning",
                "operator": "ge",
                "threshold": 0.01,
            }
        ],
        "rollback_gates": [
            {
                "gate_id": "meaning-regresses",
                "endpoint_id": "meaning",
                "operator": "le",
                "threshold": -0.01,
            }
        ],
        "artifact_requirements": [
            {"kind": "version_stamp", "minimum_count": 1}
        ],
        "claim_class": "promotion_candidate",
        "source_commit": HEX_40,
        "source_dirty": False,
        "author": "test",
        "created_at": "2026-07-23T00:00:00Z",
    }
    payload.update(updates)
    return payload


def _manifest(**updates: object) -> ExperimentCampaignV1:
    return ExperimentCampaignV1.model_validate(_manifest_payload(**updates))


def _complete_result(
    manifest: ExperimentCampaignV1, **updates: object
) -> CampaignResultV1:
    payload: dict[str, object] = {
        "campaign_id": manifest.campaign_id,
        "experiment_id": manifest.experiment_id,
        "manifest_sha256": campaign_manifest_sha256(manifest),
        "claim_class": manifest.claim_class,
        "arm_seed_results": [
            [arm.arm_id, seed] for arm in manifest.arms for seed in manifest.seeds
        ],
        "paired_example_ids": {
            arm.arm_id: ["example-1", "example-2"] for arm in manifest.arms
        },
        "endpoint_ids": [endpoint.endpoint_id for endpoint in manifest.endpoints],
        "holm_hypothesis_ids": [
            hypothesis
            for family in manifest.multiplicity_families
            for hypothesis in family.hypothesis_ids
        ],
        "promotion_gate_ids_passed": [
            gate.gate_id for gate in manifest.promotion_gates
        ],
        "rollback_gate_ids_passed": [
            gate.gate_id for gate in manifest.rollback_gates
        ],
        "artifacts": [
            {"kind": "version_stamp", "uri": "docs/result.json", "sha256": HEX_64}
        ],
    }
    payload.update(updates)
    return CampaignResultV1.model_validate(payload)


def test_manifest_digest_and_lock_roundtrip() -> None:
    manifest = _manifest()
    digest = campaign_manifest_sha256(manifest)
    lock = CampaignLockV1(
        manifest_sha256=digest,
        manifest=manifest,
        locked_at="2026-07-23T00:01:00Z",
    )

    restored = CampaignLockV1.model_validate_json(lock.model_dump_json())
    assert restored == lock
    assert restored.manifest_sha256 == campaign_manifest_sha256(restored.manifest)


def test_decision_bearing_mutation_changes_digest() -> None:
    manifest = _manifest()
    mutated = ExperimentCampaignV1.model_validate(
        _manifest_payload(decision="Reject unless every preregistered gate passes.")
    )

    assert campaign_manifest_sha256(mutated) != campaign_manifest_sha256(manifest)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("seeds", [7, 7], "seeds must be unique"),
        ("seeds", [True], "seeds must contain only integer identifiers"),
        (
            "endpoints",
            [
                {
                    "endpoint_id": "meaning",
                    "metric": "binder_reference_f1",
                    "role": "primary",
                    "direction": "increase",
                    "minimum_effect": float("nan"),
                }
            ],
            "minimum_effect must be finite",
        ),
    ],
)
def test_manifest_rejects_invalid_seed_and_nonfinite_values(
    field: str, value: object, error: str
) -> None:
    with pytest.raises((TypeError, ValidationError), match=error):
        ExperimentCampaignV1.model_validate(_manifest_payload(**{field: value}))


def test_manifest_rejects_duplicate_identifiers() -> None:
    arms = _manifest_payload()["arms"]
    assert isinstance(arms, list)
    duplicate = [dict(arms[0]), {**dict(arms[1]), "arm_id": "control"}]

    with pytest.raises(ValidationError, match="arm identifiers must be unique"):
        ExperimentCampaignV1.model_validate(_manifest_payload(arms=duplicate))


def _write_certification(
    path: Path, disposition: str, *, valid_digest: bool = True
) -> None:
    signed = {
        "disposition": disposition,
        "artifact_path": "docs/design/ap-001.json",
    }
    digest = hashlib.sha256(canonical_json(signed).encode("utf-8")).hexdigest()
    path.write_text(
        json.dumps({**signed, "artifact_sha256": digest if valid_digest else HEX_64}),
        encoding="utf-8",
    )


def test_ap001_primary_endpoint_requires_verified_certification(
    tmp_path: Path,
) -> None:
    certification_path = tmp_path / "ap-001.json"
    assert load_ap001_certification(certification_path) is None
    assert select_primary_endpoint(None) == "binder_reference_f1"

    _write_certification(certification_path, "certified", valid_digest=False)
    assert load_ap001_certification(certification_path) is None

    _write_certification(certification_path, "revise")
    revise = load_ap001_certification(certification_path)
    assert isinstance(revise, AP001CertificationV1)
    assert select_primary_endpoint(revise) == "binder_reference_f1"

    _write_certification(certification_path, "certified")
    certified = load_ap001_certification(certification_path)
    assert isinstance(certified, AP001CertificationV1)
    assert select_primary_endpoint(certified) == "binding_aware_meaningful_v2"


def test_complete_promotion_candidate_passes() -> None:
    manifest = _manifest()

    assert validate_result_claim(manifest, _complete_result(manifest)) == ()


def test_promotion_candidate_fails_closed_on_incomplete_or_exploratory_result() -> None:
    manifest = _manifest()
    result = _complete_result(
        manifest,
        arm_seed_results=[["control", 7]],
        promotion_gate_ids_passed=[],
        artifacts=[],
        exploratory=True,
    )

    assert set(validate_result_claim(manifest, result)) == {
        "exploratory_result",
        "incomplete_arm_seed_results",
        "promotion_gates_not_passed",
        "missing_artifact:version_stamp",
    }


def test_ship_gate_claim_requires_ship_gates_to_pass() -> None:
    manifest = _manifest(claim_class="ship_gate")
    result = _complete_result(manifest, ship_gates_passed=False)

    assert validate_result_claim(manifest, result) == ("ship_gates_not_passed",)
