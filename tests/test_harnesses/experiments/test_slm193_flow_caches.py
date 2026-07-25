"""Tests for the SLM-193 bit-exact flow-cache fixture harness."""

from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.experiments.slm193_flow_caches import (
    ARM_NAMES,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    FlowCacheManifestV1,
    run_flow_cache_fixture,
    validate_manifest,
)


def test_arm_names_cover_conditions() -> None:
    assert "exact_closure_cold" in ARM_NAMES
    assert "exact_closure_warm" in ARM_NAMES
    assert "disk_restart" in ARM_NAMES
    assert "version_invalidation" in ARM_NAMES


def test_run_flow_cache_fixture(tmp_path: Path) -> None:
    manifest = run_flow_cache_fixture(
        output_dir=tmp_path,
        write_design_docs=False,
    )
    assert isinstance(manifest, FlowCacheManifestV1)
    assert manifest.matrix_set == MATRIX_SET
    assert manifest.matrix_version == MATRIX_VERSION
    assert manifest.experiment_id == EXPERIMENT_ID
    assert manifest.status == "fixture"
    assert manifest.claim_class == "wiring"
    assert manifest.n_arms == len(ARM_NAMES)
    assert manifest.n_cases == len(ARM_NAMES)
    assert len(manifest.arms) == len(ARM_NAMES)
    assert len(manifest.cases) == len(ARM_NAMES)

    arm_names = {a.arm_name for a in manifest.arms}
    for name in ARM_NAMES:
        assert name in arm_names

    errors = validate_manifest(manifest)
    assert not errors, errors

    # The warm closure arm should have non-trivial hits.
    warm_arm = next(a for a in manifest.arms if a.arm_name == "exact_closure_warm")
    assert warm_arm.hit_rate > 0.0
    assert warm_arm.n_entries > 0

    # Version invalidation should miss (hit rate 0) on the measured run.
    inv_arm = next(a for a in manifest.arms if a.arm_name == "version_invalidation")
    assert inv_arm.hit_rate == 0.0

    # Disk restart should hit.
    disk_arm = next(a for a in manifest.arms if a.arm_name == "disk_restart")
    assert disk_arm.hit_rate == 1.0


def test_manifest_to_dict_round_trip(tmp_path: Path) -> None:
    manifest = run_flow_cache_fixture(
        output_dir=tmp_path,
        write_design_docs=False,
    )
    data = manifest.to_dict()
    restored = FlowCacheManifestV1.from_dict(data)
    assert restored.matrix_set == manifest.matrix_set
    assert restored.n_cases == manifest.n_cases
    assert len(restored.arms) == len(manifest.arms)


def test_validate_manifest_catches_errors() -> None:
    manifest = run_flow_cache_fixture(
        output_dir=Path("outputs/runs/slm193-validate"),
        write_design_docs=False,
    )
    data = manifest.to_dict()
    data["matrix_version"] = "wrong"
    data["n_cases"] = 999
    bad = FlowCacheManifestV1.from_dict(data)
    errors = validate_manifest(bad)
    assert any("matrix_version mismatch" in e for e in errors)
    assert any("n_cases does not match" in e for e in errors)


def test_design_docs_written(tmp_path: Path) -> None:
    json_path = tmp_path / "design.json"
    md_path = tmp_path / "design.md"
    run_flow_cache_fixture(
        output_dir=tmp_path / "run",
        write_design_docs=True,
        design_json=json_path,
        design_md=md_path,
    )
    assert json_path.exists()
    assert md_path.exists()
    assert "SLM-193" in md_path.read_text()
