"""Tests for slm_training.harnesses.experiments.self_context_curriculum (SLM-300)."""

from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.experiments.self_context_curriculum import (
    MATRIX_SET,
    MATRIX_VERSION,
    SELF_CONTEXT_CURRICULUM_ID,
    PolicyOrigin,
    SelfContextCurriculumManifest,
    build_self_context_manifest,
    policy_origin_mixture,
    render_markdown,
    run_fixture_self_context_curriculum,
    validate_manifest,
)


def test_default_manifest() -> None:
    manifest = build_self_context_manifest()
    assert manifest.curriculum_id == SELF_CONTEXT_CURRICULUM_ID
    assert manifest.matrix_set == MATRIX_SET
    assert manifest.matrix_version == MATRIX_VERSION
    assert 0.0 in manifest.self_context_rates
    assert manifest.status == "not_run"
    assert manifest.parent_checkpoint_uri is None


def test_manifest_with_parent_is_pending() -> None:
    manifest = build_self_context_manifest(
        parent_checkpoint_uri="hf://bucket/checkpoint/ref.json",
        checkpoint_bucket="hf://bucket",
    )
    assert manifest.status == "frontier_pending_gpu"


def test_validate_manifest_ok() -> None:
    manifest = build_self_context_manifest()
    assert validate_manifest(manifest) == []


def test_validate_missing_control() -> None:
    manifest = build_self_context_manifest(self_context_rates=(0.10, 0.25))
    errors = validate_manifest(manifest)
    assert any("0.0 control rate is required" in e for e in errors)


def test_validate_invalid_rate() -> None:
    manifest = build_self_context_manifest(self_context_rates=(0.0, 0.10, 1.5))
    errors = validate_manifest(manifest)
    assert any("self_context_rates must be in [0, 1]" in e for e in errors)


def test_validate_invalid_lagged_share() -> None:
    manifest = SelfContextCurriculumManifest(lagged_policy_share=1.5)
    errors = validate_manifest(manifest)
    assert any("lagged_policy_share must be in [0, 1]" in e for e in errors)


def test_validate_frontier_requires_parent() -> None:
    manifest = SelfContextCurriculumManifest(
        self_context_rates=(0.0, 0.10),
        claim_class="frontier",
        parent_checkpoint_uri=None,
        checkpoint_bucket=None,
    )
    errors = validate_manifest(manifest)
    assert any("parent_checkpoint_uri" in e for e in errors)
    assert any("checkpoint_bucket" in e for e in errors)


def test_policy_origin_mixture_sums_to_one() -> None:
    for rate in (0.0, 0.10, 0.25, 0.50, 1.0):
        for lagged_share in (0.0, 0.3, 0.5, 1.0):
            mixture = policy_origin_mixture(rate, lagged_share)
            assert abs(sum(mixture.values()) - 1.0) < 1e-12
            assert set(mixture) == {
                PolicyOrigin.GOLD.value,
                PolicyOrigin.MODEL_CURRENT.value,
                PolicyOrigin.MODEL_LAGGED.value,
            }


def test_policy_origin_mixture_zero_reproduces_legacy_behavior() -> None:
    """Mixture zero must reproduce legacy (gold-only) corruption exactly.

    This is the wiring-level proof of the issue's acceptance criterion:
    "Mixture zero reproduces legacy behavior." Regardless of the lagged-policy
    split, a self-context rate of 0.0 must yield 100% gold-derived corruption
    and zero model-sampled mass.
    """
    for lagged_share in (0.0, 0.25, 0.5, 0.75, 1.0):
        mixture = policy_origin_mixture(0.0, lagged_share)
        assert mixture == {
            PolicyOrigin.GOLD.value: 1.0,
            PolicyOrigin.MODEL_CURRENT.value: 0.0,
            PolicyOrigin.MODEL_LAGGED.value: 0.0,
        }


def test_policy_origin_mixture_rejects_out_of_range() -> None:
    import pytest

    with pytest.raises(ValueError):
        policy_origin_mixture(-0.1)
    with pytest.raises(ValueError):
        policy_origin_mixture(0.5, lagged_policy_share=1.1)


def test_policy_origin_mixture_splits_current_and_lagged() -> None:
    mixture = policy_origin_mixture(0.5, lagged_policy_share=0.5)
    assert mixture[PolicyOrigin.MODEL_CURRENT.value] == 0.25
    assert mixture[PolicyOrigin.MODEL_LAGGED.value] == 0.25
    assert mixture[PolicyOrigin.GOLD.value] == 0.5


def test_run_fixture_curriculum(tmp_path: Path) -> None:
    manifest = build_self_context_manifest(
        self_context_rates=(0.0, 0.10),
        seeds=(0, 1),
    )
    report = run_fixture_self_context_curriculum(
        manifest, run_id="test", output_dir=tmp_path
    )
    assert report.status == "fixture"
    assert report.matrix_set == MATRIX_SET
    assert len(report.arms) == 4  # 2 rates * 2 seeds
    assert all(a.status == "fixture_planned" for a in report.arms)
    assert (tmp_path / "self_context_curriculum_report.json").exists()


def test_control_arm_has_pure_gold_mixture(tmp_path: Path) -> None:
    manifest = build_self_context_manifest(self_context_rates=(0.0, 0.25), seeds=(0,))
    report = run_fixture_self_context_curriculum(manifest, run_id="control_test")
    control_arms = [a for a in report.arms if a.self_context_rate == 0.0]
    assert len(control_arms) == 1
    assert control_arms[0].policy_origin_mixture[PolicyOrigin.GOLD.value] == 1.0
    assert any("legacy" in note for note in control_arms[0].notes)


def test_render_markdown_includes_hypothesis() -> None:
    manifest = build_self_context_manifest(self_context_rates=(0.0,))
    report = run_fixture_self_context_curriculum(manifest, run_id="md_test")
    md = render_markdown(report)
    assert "SLM-300" in md
    assert manifest.hypothesis[:20] in md
    assert "fixture-only" in md
    assert "Policy origin mixture" in md


def test_render_markdown_arm_table() -> None:
    manifest = build_self_context_manifest(
        self_context_rates=(0.0, 0.10), seeds=(0,)
    )
    report = run_fixture_self_context_curriculum(manifest, run_id="arm_test")
    md = render_markdown(report)
    assert "A_control" in md
    assert "SC10" in md
