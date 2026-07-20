"""Tests for slm_training.harnesses.experiments.corruption_curriculum (SLM-120)."""

from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.experiments.corruption_curriculum import (
    CORRUPTION_CURRICULUM_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    CorruptionCurriculumManifest,
    build_corruption_curriculum_manifest,
    render_markdown,
    run_fixture_curriculum,
    validate_manifest,
)


def test_default_manifest() -> None:
    manifest = build_corruption_curriculum_manifest()
    assert manifest.curriculum_id == CORRUPTION_CURRICULUM_ID
    assert manifest.matrix_set == MATRIX_SET
    assert manifest.matrix_version == MATRIX_VERSION
    assert 0.0 in manifest.near_solved_shares
    assert manifest.status == "not_run"
    assert manifest.parent_checkpoint_uri is None


def test_manifest_with_parent_is_pending() -> None:
    manifest = build_corruption_curriculum_manifest(
        parent_checkpoint_uri="hf://bucket/checkpoint/ref.json",
        checkpoint_bucket="hf://bucket",
    )
    assert manifest.status == "frontier_pending_gpu"


def test_validate_manifest_ok() -> None:
    manifest = build_corruption_curriculum_manifest()
    assert validate_manifest(manifest) == []


def test_validate_missing_control() -> None:
    manifest = build_corruption_curriculum_manifest(near_solved_shares=(0.05, 0.10))
    errors = validate_manifest(manifest)
    assert any("0.0 control share is required" in e for e in errors)


def test_validate_invalid_share() -> None:
    manifest = build_corruption_curriculum_manifest(
        near_solved_shares=(0.0, 0.05, 1.5)
    )
    errors = validate_manifest(manifest)
    assert any("near_solved_shares must be in [0, 1]" in e for e in errors)


def test_validate_frontier_requires_parent() -> None:
    manifest = build_corruption_curriculum_manifest(
        near_solved_shares=(0.0, 0.05),
        parent_checkpoint_uri=None,
        checkpoint_bucket=None,
    )
    manifest = CorruptionCurriculumManifest(
        near_solved_shares=(0.0, 0.05),
        claim_class="frontier",
        parent_checkpoint_uri=None,
        checkpoint_bucket=None,
    )
    errors = validate_manifest(manifest)
    assert any("parent_checkpoint_uri" in e for e in errors)
    assert any("checkpoint_bucket" in e for e in errors)


def test_run_fixture_curriculum(tmp_path: Path) -> None:
    manifest = build_corruption_curriculum_manifest(
        near_solved_shares=(0.0, 0.05),
        seeds=(0, 1),
    )
    report = run_fixture_curriculum(manifest, run_id="test", output_dir=tmp_path)
    assert report.status == "fixture"
    assert report.matrix_set == MATRIX_SET
    assert len(report.arms) == 4  # 2 shares * 2 seeds
    assert all(a.status == "fixture_planned" for a in report.arms)
    assert (tmp_path / "corruption_curriculum_report.json").exists()


def test_render_markdown_includes_hypothesis() -> None:
    manifest = build_corruption_curriculum_manifest(near_solved_shares=(0.0,))
    report = run_fixture_curriculum(manifest, run_id="md_test")
    md = render_markdown(report)
    assert "SLM-120" in md
    assert manifest.hypothesis[:20] in md
    assert "S0_clean" in md
    assert "fixture-only" in md


def test_render_markdown_arm_table() -> None:
    manifest = build_corruption_curriculum_manifest(
        near_solved_shares=(0.0, 0.05), seeds=(0,)
    )
    report = run_fixture_curriculum(manifest, run_id="arm_test")
    md = render_markdown(report)
    assert "A_control" in md
    assert "B05" in md
