"""Regression tests for CAP5-02 campaign manifest (SLM-101)."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.harnesses.experiments.cap5_02_campaign import (
    CAMPAIGN_ID,
    MANIFEST_SCHEMA,
    CampaignArm,
    Cap5CampaignManifest,
    build_cap5_campaign_manifest,
    validate_cap5_campaign_manifest,
)


def test_default_arms_include_eligible_and_omitted() -> None:
    from scripts.build_cap5_02_manifest import DEFAULT_ARMS

    eligible = [a for a in DEFAULT_ARMS if a.eligible]
    omitted = [a for a in DEFAULT_ARMS if not a.eligible]
    assert len(eligible) == 10
    assert len(omitted) == 4
    for arm in omitted:
        assert arm.omission_reason is not None


def test_build_manifest_has_version_and_hash() -> None:
    arms = [
        CampaignArm(arm_id="a1", hypothesis_id="h1", mechanism="m1", eligible=True),
    ]
    manifest = build_cap5_campaign_manifest(
        arms,
        manifest_version="v1",
        note="test",
    )
    assert isinstance(manifest, Cap5CampaignManifest)
    assert manifest.campaign_id == CAMPAIGN_ID
    assert manifest.schema_version == MANIFEST_SCHEMA
    assert manifest.manifest_version == "v1"
    assert manifest.manifest_hash is not None
    assert len(manifest.manifest_hash) == 16
    assert manifest.no_test_peeking is True


def test_manifest_hash_changes_with_version() -> None:
    arms = [
        CampaignArm(arm_id="a1", hypothesis_id="h1", mechanism="m1", eligible=True),
    ]
    m1 = build_cap5_campaign_manifest(arms, manifest_version="v1")
    m2 = build_cap5_campaign_manifest(arms, manifest_version="v2")
    assert m1.manifest_hash != m2.manifest_hash


def test_omitted_arm_requires_reason() -> None:
    errors = validate_cap5_campaign_manifest(
        build_cap5_campaign_manifest(
            [CampaignArm(arm_id="a1", hypothesis_id="h1", mechanism="m1", eligible=False)],
            manifest_version="v1",
        ).to_dict()
    )
    assert any("omission_reason" in e for e in errors)


def test_validate_manifest_catches_schema_errors() -> None:
    assert validate_cap5_campaign_manifest({}) != []
    assert validate_cap5_campaign_manifest(
        {"schema_version": "wrong", "campaign_id": CAMPAIGN_ID}
    ) != []
    assert validate_cap5_campaign_manifest(
        {
            "schema_version": MANIFEST_SCHEMA,
            "campaign_id": "wrong",
            "manifest_version": "v1",
        }
    ) != []


def test_manifest_round_trip_dict() -> None:
    arms = [
        CampaignArm(
            arm_id="a1",
            hypothesis_id="h1",
            mechanism="m1",
            eligible=True,
            selection_evidence="test",
        ),
        CampaignArm(
            arm_id="a2",
            hypothesis_id="h2",
            mechanism="m2",
            eligible=False,
            omission_reason="not ready",
        ),
    ]
    manifest = build_cap5_campaign_manifest(
        arms,
        manifest_version="v1",
        primary_metric="meaningful_rate",
        comparison_regimes=("equal_bytes",),
        seeds=(7,),
        hardware_targets=("cpu",),
        note="round trip",
    )
    data = manifest.to_dict()
    assert data["schema_version"] == MANIFEST_SCHEMA
    assert data["campaign_id"] == CAMPAIGN_ID
    assert data["primary_metric"] == "meaningful_rate"
    assert len(data["arms"]) == 2
    assert validate_cap5_campaign_manifest(data) == []


def test_cli_builds_manifest(tmp_path: Path) -> None:
    from scripts.build_cap5_02_manifest import main

    out_file = tmp_path / "manifest.json"
    rc = main(
        [
            "--manifest-version",
            "v1",
            "--primary-metric",
            "component_recall",
            "--out",
            str(out_file),
        ]
    )
    assert rc == 0
    assert out_file.is_file()
    manifest = json.loads(out_file.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == MANIFEST_SCHEMA
    assert manifest["campaign_id"] == CAMPAIGN_ID
    assert manifest["primary_metric"] == "component_recall"
    assert manifest["manifest_hash"] is not None
    assert len(manifest["arms"]) == 14
    assert validate_cap5_campaign_manifest(manifest) == []
