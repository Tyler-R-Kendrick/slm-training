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
            {"kind": kind, "minimum_count": 1}
            for kind in (
                "version_stamp",
                "seed_result",
                "paired_examples",
                "endpoint_result",
                "holm_family",
                "agentevals",
                "agentv",
            )
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
    manifest: ExperimentCampaignV1,
    artifact_root: Path,
    **updates: object,
) -> CampaignResultV1:
    artifacts = []
    for requirement in manifest.artifact_requirements:
        path = artifact_root / f"{requirement.kind}.json"
        path.write_text(json.dumps({"kind": requirement.kind}), encoding="utf-8")
        artifacts.append(
            {
                "kind": requirement.kind,
                "uri": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
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
        "endpoint_values": {
            endpoint.endpoint_id: 0.02 for endpoint in manifest.endpoints
        },
        "holm_results": [
            {
                "hypothesis_id": hypothesis,
                "raw_p_value": 0.01,
                "rank": rank,
                "threshold": 0.05,
                "adjusted_p_value": 0.01,
                "rejected": True,
            }
            for rank, hypothesis in enumerate(
                (
                    hypothesis
                    for family in manifest.multiplicity_families
                    for hypothesis in family.hypothesis_ids
                ),
                start=1,
            )
        ],
        "artifacts": artifacts,
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
    artifact = path.parent / "ap-001-evidence.json"
    artifact.write_text(
        json.dumps({"disposition": disposition, "metric": "meaning-v2"}),
        encoding="utf-8",
    )
    envelope = {
        "disposition": disposition,
        "artifact_path": artifact.name,
        "artifact_sha256": (
            hashlib.sha256(artifact.read_bytes()).hexdigest()
            if valid_digest
            else HEX_64
        ),
    }
    path.write_text(json.dumps(envelope), encoding="utf-8")


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


def test_complete_promotion_candidate_passes(tmp_path: Path) -> None:
    manifest = _manifest()

    assert (
        validate_result_claim(
            manifest,
            _complete_result(manifest, tmp_path),
            artifact_root=tmp_path,
        )
        == ()
    )


def test_promotion_candidate_fails_closed_on_incomplete_or_exploratory_result(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    result = _complete_result(
        manifest,
        tmp_path,
        arm_seed_results=[["control", 7]],
        endpoint_values={"meaning": -0.02},
        artifacts=[],
        exploratory=True,
    )

    assert set(
        validate_result_claim(manifest, result, artifact_root=tmp_path)
    ) == {
        "exploratory_result",
        "incomplete_arm_seed_results",
        "promotion_gates_not_passed",
        "rollback_gates_not_passed",
        "missing_artifact:version_stamp",
        "missing_artifact:seed_result",
        "missing_artifact:paired_examples",
        "missing_artifact:endpoint_result",
        "missing_artifact:holm_family",
        "missing_artifact:agentevals",
        "missing_artifact:agentv",
    }


def test_promotion_candidate_rejects_empty_pairs_and_duplicate_rows(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    result = _complete_result(
        manifest,
        tmp_path,
        arm_seed_results=[
            ["control", 7],
            ["control", 11],
            ["candidate", 7],
            ["candidate", 11],
            ["candidate", 11],
        ],
        paired_example_ids={"control": [], "candidate": []},
        holm_results=[
            {
                "hypothesis_id": "meaning",
                "raw_p_value": 0.01,
                "rank": 1,
                "threshold": 0.05,
                "adjusted_p_value": 0.01,
                "rejected": True,
            },
            {
                "hypothesis_id": "meaning",
                "raw_p_value": 0.01,
                "rank": 1,
                "threshold": 0.05,
                "adjusted_p_value": 0.01,
                "rejected": True,
            },
        ],
    )
    failures = set(
        validate_result_claim(manifest, result, artifact_root=tmp_path)
    )
    assert "incomplete_arm_seed_results" in failures
    assert "incomplete_paired_examples" in failures
    assert "incomplete_holm_family" in failures


def test_ship_gate_claim_requires_ship_gates_to_pass(tmp_path: Path) -> None:
    requirements = list(_manifest_payload()["artifact_requirements"])
    requirements.append({"kind": "ship_gates", "minimum_count": 1})
    manifest = _manifest(
        claim_class="ship_gate",
        artifact_requirements=requirements,
    )
    result = _complete_result(manifest, tmp_path, ship_gates_passed=False)

    assert validate_result_claim(
        manifest, result, artifact_root=tmp_path
    ) == ("ship_gates_not_passed",)
