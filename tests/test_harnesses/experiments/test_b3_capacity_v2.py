"""Tests for slm_training.harnesses.experiments.b3_capacity_v2 (SLM-124)."""

from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.experiments.b3_capacity_v2 import (
    B3_CAPACITY_V2_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    B3CapacityV2Manifest,
    build_b3_capacity_v2_manifest,
    render_markdown,
    run_fixture_ladder,
    validate_manifest,
)


def test_default_manifest() -> None:
    manifest = build_b3_capacity_v2_manifest()
    assert manifest.b3_id == B3_CAPACITY_V2_ID
    assert manifest.matrix_set == MATRIX_SET
    assert manifest.matrix_version == MATRIX_VERSION
    assert len(manifest.arms) == 2
    assert {a.representation for a in manifest.arms} == {"lexer", "choice"}
    assert manifest.status == "not_run"
    assert manifest.claim_class == "wiring"


def test_manifest_with_parent_is_pending() -> None:
    manifest = build_b3_capacity_v2_manifest(
        parent_checkpoint_uri="hf://bucket/checkpoint/ref.json",
        checkpoint_bucket="hf://bucket",
    )
    assert manifest.status == "frontier_pending_gpu"
    assert manifest.claim_class == "frontier"


def test_validate_manifest_ok() -> None:
    manifest = build_b3_capacity_v2_manifest()
    assert validate_manifest(manifest) == []


def test_validate_unsupported_representation() -> None:
    manifest = build_b3_capacity_v2_manifest(representations=("lexer", "bad_rep"))
    errors = validate_manifest(manifest)
    assert any("unsupported representation: bad_rep" in e for e in errors)


def test_validate_frontier_requires_parent() -> None:
    manifest = B3CapacityV2Manifest(
        arms=(
            build_b3_capacity_v2_manifest(representations=("lexer",)).arms[0],
        ),
        claim_class="frontier",
        parent_checkpoint_uri=None,
        checkpoint_bucket=None,
    )
    errors = validate_manifest(manifest)
    assert any("parent_checkpoint_uri" in e for e in errors)
    assert any("checkpoint_bucket" in e for e in errors)


def test_run_fixture_ladder(tmp_path: Path) -> None:
    manifest = build_b3_capacity_v2_manifest(widths=(64, 128), seeds=(0, 1))
    report = run_fixture_ladder(manifest, run_id="test", output_dir=tmp_path)
    assert report.status == "fixture"
    assert len(report.rows) == 8  # 2 arms * 2 widths * 2 seeds
    assert all(r.status == "fixture_planned" for r in report.rows)
    assert (tmp_path / "b3_capacity_v2_report.json").exists()


def test_render_markdown_includes_hypothesis() -> None:
    manifest = build_b3_capacity_v2_manifest(widths=(64,), seeds=(0,))
    report = run_fixture_ladder(manifest, run_id="md_test")
    md = render_markdown(report)
    assert "SLM-124" in md
    assert manifest.hypothesis[:20] in md
    assert "lexer" in md
    assert "Fixture/plan only" in md
