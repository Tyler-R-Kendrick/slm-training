"""Tests for the SLM-108 external ceiling matrix harness."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.harnesses.experiments.external_ceiling_matrix import (
    ExternalCeilingArm,
    ExternalCeilingManifest,
    build_external_ceiling_manifest,
    render_markdown,
    run_fixture_matrix,
    validate_external_ceiling_manifest,
)


def test_default_manifest_has_required_arms() -> None:
    manifest = build_external_ceiling_manifest()
    ids = {arm.arm_id for arm in manifest.arms}
    assert "A" in ids
    assert "B" in ids


def test_manifest_validation_catches_duplicate_ids() -> None:
    manifest = ExternalCeilingManifest(
        arms=[
            ExternalCeilingArm(arm_id="A", description="x", model_id="m", revision="r", decode_mode="c"),
            ExternalCeilingArm(arm_id="A", description="y", model_id="m", revision="r", decode_mode="c"),
        ]
    )
    errors = validate_external_ceiling_manifest(manifest)
    assert any("duplicate" in e for e in errors)


def test_manifest_validation_requires_checkpoint_for_frontier() -> None:
    manifest = build_external_ceiling_manifest()
    # A is frontier by default and has no checkpoint_reference_uri.
    errors = validate_external_ceiling_manifest(manifest)
    assert any("frontier" in e and "checkpoint_reference_uri" in e for e in errors)


def test_validation_passes_with_checkpoint_uri() -> None:
    manifest = build_external_ceiling_manifest(
        checkpoint_reference_uri="hf://buckets/TKendrick/OpenUI/checkpoints/x/ref.json"
    )
    errors = validate_external_ceiling_manifest(manifest)
    assert not errors


def test_fixture_run_produces_report(tmp_path: Path) -> None:
    manifest = build_external_ceiling_manifest(
        checkpoint_reference_uri="hf://buckets/TKendrick/OpenUI/checkpoints/x/ref.json"
    )
    report = run_fixture_matrix(manifest, run_id="test_slm108", output_dir=tmp_path)
    assert report.status == "fixture"
    result_ids = {r.arm_id for r in report.results}
    assert "B" in result_ids
    assert "C" not in result_ids or any(r.status == "not_run" for r in report.results if r.arm_id == "C")
    assert (tmp_path / "external_ceiling_report.json").exists()


def test_render_markdown_includes_all_arms() -> None:
    manifest = build_external_ceiling_manifest(
        checkpoint_reference_uri="hf://buckets/TKendrick/OpenUI/checkpoints/x/ref.json"
    )
    report = run_fixture_matrix(manifest, run_id="test_slm108")
    md = render_markdown(report)
    assert "External constrained-decoding semantic ceiling" in md
    for arm in manifest.arms:
        assert arm.arm_id in md


def test_report_json_round_trip(tmp_path: Path) -> None:
    manifest = build_external_ceiling_manifest(
        checkpoint_reference_uri="hf://buckets/TKendrick/OpenUI/checkpoints/x/ref.json"
    )
    run_fixture_matrix(manifest, run_id="test_slm108", output_dir=tmp_path)
    payload = json.loads((tmp_path / "external_ceiling_report.json").read_text())
    assert payload["run_id"] == "test_slm108"
    assert payload["matrix_set"] == "external-ceiling"
    assert "version_stamp" in payload
