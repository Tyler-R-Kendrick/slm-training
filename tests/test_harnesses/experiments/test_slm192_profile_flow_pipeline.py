"""Tests for SLM-192 stage-accurate flow-pipeline cost-profile harness."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm192_profile_flow_pipeline import (
    ARM_NAMES,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    FlowPipelineManifestV1,
    render_markdown,
    run_profile_flow_pipeline,
    validate_manifest,
)


def test_constants() -> None:
    assert EXPERIMENT_ID == "slm192-profile-flow-pipeline"
    assert MATRIX_SET == "slm192_profile_flow_pipeline"
    assert MATRIX_VERSION == "ffe3-01-v1"
    assert len(ARM_NAMES) == 6
    assert "bridge_planner_canonical_greedy" in ARM_NAMES
    assert "exact_closure_toy" in ARM_NAMES


def test_run_profile_flow_pipeline_produces_manifest(tmp_path) -> None:
    manifest = run_profile_flow_pipeline(
        output_dir=tmp_path,
        n_repeats=2,
        seed=0,
        write_design_docs=False,
    )
    assert manifest.status == "fixture"
    assert manifest.claim_class == "wiring"
    assert manifest.disposition == "cost_profile_wired"
    assert manifest.n_arms == len(ARM_NAMES)
    assert manifest.n_cases == len(ARM_NAMES) * 2  # cold + warm per arm
    assert len(manifest.arms) == len(ARM_NAMES) * 2
    assert len(manifest.cases) == len(manifest.arms)
    assert manifest.cost_gate.max_on_policy_epoch_seconds == 1800.0
    assert manifest.on_policy.strategy in {"on_policy_viable", "offline_only"}
    report = tmp_path / "slm192_profile_flow_pipeline_report.json"
    assert report.exists()


def test_manifest_round_trip(tmp_path) -> None:
    manifest = run_profile_flow_pipeline(
        output_dir=tmp_path,
        n_repeats=2,
        seed=1,
        write_design_docs=False,
    )
    data = manifest.to_dict()
    restored = FlowPipelineManifestV1.from_dict(data)
    assert restored.matrix_set == manifest.matrix_set
    assert restored.matrix_version == manifest.matrix_version
    assert len(restored.arms) == len(manifest.arms)
    assert len(restored.cases) == len(manifest.cases)
    assert restored.cost_gate.allowed_strategy == manifest.cost_gate.allowed_strategy


def test_render_markdown_contains_caveats(tmp_path) -> None:
    manifest = run_profile_flow_pipeline(
        output_dir=tmp_path,
        n_repeats=2,
        seed=2,
        write_design_docs=False,
    )
    md = render_markdown(manifest)
    assert "wiring / fixture only" in md
    assert "No-go for promotion" in md
    assert "SLM-192" in md
    assert "bridge_planner_canonical_greedy" in md
    assert "cost_profile_wired" in md


def test_validate_manifest_passes(tmp_path) -> None:
    manifest = run_profile_flow_pipeline(
        output_dir=tmp_path,
        n_repeats=2,
        seed=3,
        write_design_docs=False,
    )
    errors = validate_manifest(manifest)
    assert not errors
