"""Tests for ``scripts/run_slm195_solver_semantic_ceiling.py``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.run_slm195_solver_semantic_ceiling import main
from slm_training.harnesses.experiments.slm195_solver_semantic_ceiling import (
    SolverCeilingManifestV1,
    SolverCeilingReport,
    build_default_manifest,
)


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    manifest = build_default_manifest("slm195_cli_test")
    manifest = manifest.__class__.from_dict(
        {**manifest.to_dict(), "source_commit": "a" * 40, "dirty_tree_ok": True}
    )
    path = tmp_path / "manifest.json"
    manifest.write_json(path)
    return path


def test_init_command(tmp_path: Path) -> None:
    out = tmp_path / "new.json"
    rc = main(["init", "--run-id", "slm195_init", "--output", str(out)])
    assert rc == 0
    manifest = SolverCeilingManifestV1.load_json(out)
    assert manifest.run_id == "slm195_init"


def test_describe_command(manifest_path: Path, capsys: Any) -> None:
    rc = main(["describe", "--manifest", str(manifest_path)])
    assert rc == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["run_id"] == "slm195_cli_test"
    assert "canonical_dfs" in parsed["arms"]


def test_exact_command(manifest_path: Path, tmp_path: Path, capsys: Any) -> None:
    out = tmp_path / "report.json"
    rc = main(["exact", "--manifest", str(manifest_path), "--output", str(out)])
    assert rc == 0
    report = SolverCeilingReport.load_json(out)
    arms_in_report = {r.arm_name for r in report.rows}
    assert "canonical_dfs" in arms_in_report
    assert "oracle_order" in arms_in_report
    assert "bfs_min_edits" in arms_in_report
    assert "astar_admissible" in arms_in_report
    assert "beam_symbolic" in arms_in_report
    captured = capsys.readouterr()
    parsed = json.loads(captured.out.splitlines()[0])
    assert parsed["run_id"] == "slm195_cli_test"


def test_budget_grid_command(manifest_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    rc = main(["budget-grid", "--manifest", str(manifest_path), "--output", str(out)])
    assert rc == 0
    report = SolverCeilingReport.load_json(out)
    assert report.rows
    budgets = {r.budget for r in report.rows}
    assert budgets == {10, 100, 1000}
