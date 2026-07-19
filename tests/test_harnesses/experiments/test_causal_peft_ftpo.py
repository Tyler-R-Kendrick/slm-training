"""Tests for slm_training.harnesses.experiments.causal_peft_ftpo (SLM-121)."""

from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.experiments.causal_peft_ftpo import (
    CAUSAL_PEFT_FTPO_ID,
    FTPO_OBJECTIVES,
    MATRIX_SET,
    MATRIX_VERSION,
    CausalPeftFtpoManifest,
    build_causal_peft_ftpo_manifest,
    render_markdown,
    run_fixture_ftpo,
    validate_manifest,
)


def test_default_manifest() -> None:
    manifest = build_causal_peft_ftpo_manifest()
    assert manifest.ftpo_id == CAUSAL_PEFT_FTPO_ID
    assert manifest.matrix_set == MATRIX_SET
    assert manifest.matrix_version == MATRIX_VERSION
    assert set(manifest.objectives) == set(FTPO_OBJECTIVES)
    assert manifest.adapter_methods == ("lora",)
    assert manifest.status == "not_run"
    assert manifest.claim_class == "wiring"


def test_manifest_with_parent_is_pending() -> None:
    manifest = build_causal_peft_ftpo_manifest(
        parent_checkpoint_uri="hf://bucket/checkpoint/ref.json",
        checkpoint_bucket="hf://bucket",
    )
    assert manifest.status == "frontier_pending_gpu"
    assert manifest.claim_class == "frontier"


def test_validate_manifest_ok() -> None:
    manifest = build_causal_peft_ftpo_manifest()
    assert validate_manifest(manifest) == []


def test_validate_unsupported_objective() -> None:
    manifest = build_causal_peft_ftpo_manifest(objectives=("ftpo_single", "bad_obj"))
    errors = validate_manifest(manifest)
    assert any("unsupported objective: bad_obj" in e for e in errors)


def test_validate_unsupported_method() -> None:
    manifest = build_causal_peft_ftpo_manifest(adapter_methods=("bad_method",))
    errors = validate_manifest(manifest)
    assert any("unsupported adapter method: bad_method" in e for e in errors)


def test_validate_frontier_requires_parent() -> None:
    manifest = CausalPeftFtpoManifest(
        objectives=("ftpo_single",),
        adapter_methods=("lora",),
        claim_class="frontier",
        parent_checkpoint_uri=None,
        checkpoint_bucket=None,
    )
    errors = validate_manifest(manifest)
    assert any("parent_checkpoint_uri" in e for e in errors)
    assert any("checkpoint_bucket" in e for e in errors)


def test_run_fixture_ftpo(tmp_path: Path) -> None:
    manifest = build_causal_peft_ftpo_manifest(
        objectives=("ftpo_single",),
        adapter_methods=("lora",),
        seeds=(0, 1),
    )
    report = run_fixture_ftpo(manifest, run_id="test", output_dir=tmp_path)
    assert report.status == "fixture"
    assert len(report.arms) == 2  # 1 objective * 1 method * 2 seeds
    assert all(a.status == "fixture_planned" for a in report.arms)
    assert (tmp_path / "causal_peft_ftpo_report.json").exists()


def test_render_markdown_includes_hypothesis() -> None:
    manifest = build_causal_peft_ftpo_manifest(
        objectives=("ftpo_single",), adapter_methods=("lora",), seeds=(0,)
    )
    report = run_fixture_ftpo(manifest, run_id="md_test")
    md = render_markdown(report)
    assert "SLM-121" in md
    assert manifest.hypothesis[:20] in md
    assert "ftpo_single" in md
    assert "fixture-only" in md
