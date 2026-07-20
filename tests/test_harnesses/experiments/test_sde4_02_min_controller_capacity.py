"""Tests for slm_training.harnesses.experiments.sde4_02_min_controller_capacity (SLM-180)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from slm_training.harnesses.experiments.sde4_02_min_controller_capacity import (
    COMPETENCE_TARGET,
    MATRIX_SET,
    MATRIX_VERSION,
    build_manifest,
    render_markdown,
    run_fixture_ladder,
)

torch = pytest.importorskip("torch")


def test_default_manifest() -> None:
    manifest = build_manifest()
    assert manifest.matrix_set == MATRIX_SET
    assert manifest.matrix_version == MATRIX_VERSION
    assert manifest.competence_target == COMPETENCE_TARGET
    assert len(manifest.rungs) == 5
    assert len(manifest.seeds) == 3
    assert manifest.seeds == (0, 1, 2)
    assert manifest.status == "not_run"
    assert manifest.claim_class == "wiring"
    dims = [r.hidden_dim for r in manifest.rungs]
    assert dims == sorted(dims)
    assert dims == [8, 16, 32, 64, 128]


def test_package_import_does_not_require_torch() -> None:
    source = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__
        def without_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ModuleNotFoundError("No module named 'torch'", name="torch")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = without_torch
        from slm_training.harnesses.experiments.sde4_02_min_controller_capacity import (
            MATRIX_SET, build_manifest
        )
        assert MATRIX_SET == "sde4-02-min-controller"
        assert build_manifest().status == "not_run"
        """
    )
    subprocess.run([sys.executable, "-c", source], check=True)


def test_build_manifest_custom_rungs() -> None:
    manifest = build_manifest(rungs=3, seeds=(0,), hidden_dims=(4, 8, 16))
    assert len(manifest.rungs) == 3
    assert manifest.seeds == (0,)
    assert [r.hidden_dim for r in manifest.rungs] == [4, 8, 16]


def test_build_manifest_not_enough_dims() -> None:
    with pytest.raises(ValueError):
        build_manifest(rungs=5, hidden_dims=(8, 16))


def test_run_fixture_ladder(tmp_path: Path) -> None:
    manifest = build_manifest(rungs=2, seeds=(0, 1), hidden_dims=(8, 16))
    report = run_fixture_ladder(
        manifest, run_id="test", output_dir=tmp_path, train_steps=50
    )
    assert report.status == "fixture"
    assert len(report.rows) == 4  # 2 rungs * 2 seeds
    assert all(r.trainable_parameters > 0 for r in report.rows)
    assert all(r.active_parameters > 0 for r in report.rows)
    assert all(isinstance(r.meets_competence_target, bool) for r in report.rows)
    assert (tmp_path / "sde4_02_min_controller_capacity_report.json").exists()


def test_run_fixture_ladder_selects_smallest_rung() -> None:
    manifest = build_manifest(rungs=2, seeds=(0,), hidden_dims=(8, 16))
    report = run_fixture_ladder(manifest, run_id="select_test", train_steps=200)
    # The 8-dim rung should reach the competence target on this trivial fixture.
    assert report.selected_rung_id == manifest.rungs[0].rung_id
    assert not report.capacity_threshold_not_identifiable


def test_render_markdown_includes_caveat_and_target() -> None:
    manifest = build_manifest(rungs=1, seeds=(0,), hidden_dims=(8,))
    report = run_fixture_ladder(manifest, run_id="md_test", train_steps=50)
    md = render_markdown(report)
    assert "SLM-180" in md
    assert str(COMPETENCE_TARGET) in md
    assert "Fixture caveat" in md
    assert "wiring-only evidence" in md
    assert report.rows[0].rung_id in md
