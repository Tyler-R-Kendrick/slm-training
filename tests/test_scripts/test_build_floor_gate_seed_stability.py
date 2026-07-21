"""Tests for SLM-226 (NCS0-06) build_floor_gate_seed_stability CLI."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_floor_gate_seed_stability import main


def test_describe_mode(capsys) -> None:
    assert main(["--describe"]) == 0
    out = capsys.readouterr().out
    assert "SemanticFloorGateV1" in out


def test_plan_only_writes_manifest(tmp_path: Path) -> None:
    assert main(["--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm226_floor_gate_seed_stability_report.json"
    assert run_json.is_file()
    data = json.loads(run_json.read_text())
    assert data["status"] == "plan_only"
    assert data["claim_class"] == "wiring"
    assert "version_stamp" in data


def test_fixture_writes_report_and_design_docs(tmp_path: Path) -> None:
    design_json = tmp_path / "design.json"
    design_md = tmp_path / "design.md"
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--output-dir",
                str(tmp_path),
                "--sweep-grid",
                "2",
                "4",
                "--seeds",
                "11",
                "3",
                "--write-design-docs",
                "--design-json",
                str(design_json),
                "--design-md",
                str(design_md),
            ]
        )
        == 0
    )
    run_json = tmp_path / "slm226_floor_gate_seed_stability_report.json"
    assert run_json.is_file()
    data = json.loads(run_json.read_text())
    assert data["status"] == "fixture"
    assert data["sweep_grid"] == [2, 4]
    assert data["seeds"] == [11, 3]
    assert (
        data["version_stamp"]["components"]["harness.experiments.slm226_floor_gate_seed_stability"]
        == "v1"
    )
    assert design_json.is_file()
    assert design_md.is_file()
    assert "Honest caveats" in design_md.read_text()
    assert "No-go for promotion" in design_md.read_text()


def test_sweep_grid_and_seeds_flags_are_plumbed(tmp_path: Path) -> None:
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--output-dir",
                str(tmp_path),
                "--sweep-grid",
                "2",
                "4",
                "--seeds",
                "11",
                "3",
                "7",
                "--runs-per-family",
                "3",
                "--no-write-design-docs",
            ]
        )
        == 0
    )
    data = json.loads((tmp_path / "slm226_floor_gate_seed_stability_report.json").read_text())
    assert data["sweep_grid"] == [2, 4]
    assert data["runs_per_family"] == 3
    assert len(data["points"]) == 2
    assert data["points"][0]["synthetic_runs"] == 2 * 3
    assert len(data["points"][0]["margins"]) == 3
