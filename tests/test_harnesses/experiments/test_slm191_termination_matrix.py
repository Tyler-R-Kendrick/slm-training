"""Tests for SLM-191 termination-policy fixture matrix harness."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm191_termination_matrix import (
    ARM_NAMES,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    TerminationManifestV1,
    build_default_adapters,
    render_markdown,
    run_termination_matrix,
    validate_manifest,
)


def test_constants() -> None:
    assert EXPERIMENT_ID == "slm191-termination-matrix"
    assert MATRIX_SET == "slm191_termination_matrix"
    assert MATRIX_VERSION == "ffe2-03-v1"
    assert len(ARM_NAMES) == 6


def test_build_default_adapters_torch_free() -> None:
    adapters = build_default_adapters()
    assert len(adapters) == 3
    assert {a.domain_id for a in adapters} == {
        "toy_layout",
        "choice_sequence",
        "canonical_edit_graph",
    }


def test_run_termination_matrix_produces_manifest(tmp_path) -> None:
    manifest = run_termination_matrix(
        output_dir=tmp_path,
        k_value=2,
        n_samples_per_arm=5,
        horizon=1.0,
        seed=0,
        write_design_docs=False,
    )
    assert manifest.status == "fixture"
    assert manifest.claim_class == "wiring"
    assert manifest.n_arms == 18
    assert manifest.n_cases == 3 * 6 * 5
    assert len(manifest.arms) == 18
    assert len(set(a.arm_name for a in manifest.arms)) == 6
    assert len(manifest.target_rows) == 3
    assert all(a.n_samples == 5 for a in manifest.arms)


def test_manifest_round_trip(tmp_path) -> None:
    manifest = run_termination_matrix(
        output_dir=tmp_path,
        k_value=2,
        n_samples_per_arm=4,
        horizon=1.0,
        seed=1,
        write_design_docs=False,
    )
    data = manifest.to_dict()
    restored = TerminationManifestV1.from_dict(data)
    assert restored.matrix_set == manifest.matrix_set
    assert len(restored.cases) == len(manifest.cases)
    assert len(restored.arms) == len(manifest.arms)


def test_render_markdown_contains_caveats(tmp_path) -> None:
    manifest = run_termination_matrix(
        output_dir=tmp_path,
        k_value=2,
        n_samples_per_arm=4,
        horizon=1.0,
        seed=2,
        write_design_docs=False,
    )
    md = render_markdown(manifest)
    assert "wiring / fixture only" in md
    assert "No-go for promotion" in md
    assert "SLM-191" in md
    assert "explicit_stop" in md


def test_validate_manifest_passes(tmp_path) -> None:
    manifest = run_termination_matrix(
        output_dir=tmp_path,
        k_value=2,
        n_samples_per_arm=4,
        horizon=1.0,
        seed=3,
        write_design_docs=False,
    )
    errors = validate_manifest(manifest)
    assert not errors
