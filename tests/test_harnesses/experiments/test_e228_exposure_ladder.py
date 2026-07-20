"""Tests for the SLM-109 E228 exposure ladder harness."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.harnesses.experiments.e228_exposure_ladder import (
    E228ExposureLadderManifest,
    E228_TARGET_TOKENS,
    build_e228_exposure_ladder,
    build_e228_recipe_config,
    render_markdown,
    run_fixture_ladder,
    validate_manifest,
)


def test_default_manifest_has_required_multipliers() -> None:
    manifest = build_e228_exposure_ladder()
    assert 1 in manifest.multipliers
    assert max(manifest.multipliers) >= 100
    assert manifest.base_target_tokens == E228_TARGET_TOKENS


def test_validation_passes_with_frontier_uris() -> None:
    manifest = build_e228_exposure_ladder(
        parent_checkpoint_uri="hf://buckets/TKendrick/OpenUI/checkpoints/e228/ref.json",
        checkpoint_bucket="hf://buckets/TKendrick/OpenUI",
    )
    assert not validate_manifest(manifest)


def test_validation_fails_without_parent_for_frontier() -> None:
    manifest = E228ExposureLadderManifest(claim_class="frontier")
    errors = validate_manifest(manifest)
    assert any("parent_checkpoint_uri" in e for e in errors)
    assert any("checkpoint_bucket" in e for e in errors)


def test_validation_requires_100x_multiplier() -> None:
    manifest = E228ExposureLadderManifest(multipliers=(1, 4, 16))
    errors = validate_manifest(manifest)
    assert any("≥100" in e for e in errors)


def test_recipe_config_preserves_e228_flags() -> None:
    manifest = build_e228_exposure_ladder()
    cfg = build_e228_recipe_config(manifest)
    assert cfg.compiler_alignment_loss_weight == 1.0
    assert cfg.compiler_alignment_margin == 1.0
    assert cfg.compiler_alignment_stratified
    assert cfg.compiler_alignment_semantic_exhaustive
    assert cfg.compiler_decode_mode == "tree"
    assert cfg.allow_unconstrained_fallback is False


def test_fixture_run_produces_report(tmp_path: Path) -> None:
    manifest = build_e228_exposure_ladder(
        parent_checkpoint_uri="hf://buckets/TKendrick/OpenUI/checkpoints/e228/ref.json",
        checkpoint_bucket="hf://buckets/TKendrick/OpenUI",
    )
    report = run_fixture_ladder(manifest, run_id="test_slm109", output_dir=tmp_path)
    assert report.status == "fixture"
    assert len(report.points) == len(manifest.multipliers) * len(manifest.seeds)
    assert (tmp_path / "e228_exposure_report.json").exists()


def test_render_markdown_contains_all_multipliers() -> None:
    manifest = build_e228_exposure_ladder(
        parent_checkpoint_uri="hf://buckets/TKendrick/OpenUI/checkpoints/e228/ref.json",
        checkpoint_bucket="hf://buckets/TKendrick/OpenUI",
    )
    report = run_fixture_ladder(manifest, run_id="test_slm109")
    md = render_markdown(report)
    assert "SLM-109" in md
    for mult in manifest.multipliers:
        assert f"{mult}×" in md


def test_manifest_json_round_trip(tmp_path: Path) -> None:
    manifest = build_e228_exposure_ladder(
        parent_checkpoint_uri="hf://buckets/TKendrick/OpenUI/checkpoints/e228/ref.json",
        checkpoint_bucket="hf://buckets/TKendrick/OpenUI",
    )
    manifest.to_json(tmp_path / "manifest.json")
    payload = json.loads((tmp_path / "manifest.json").read_text())
    assert payload["matrix_set"] == "e228-exposure-ladder"
    assert "base_recipe_hash" in payload
