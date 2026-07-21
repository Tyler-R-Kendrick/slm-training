"""Tests for SLM-215 (NCS0-02) build_spectral_atlas CLI."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_spectral_atlas import main


def test_describe_mode(capsys) -> None:
    assert main(["--describe"]) == 0
    out = capsys.readouterr().out
    assert "SpectralAtlasV1" in out


def test_plan_only_writes_manifest(tmp_path: Path) -> None:
    assert main(["--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm215_spectral_atlas_report.json"
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
                "--synthetic-runs",
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
    run_json = tmp_path / "slm215_spectral_atlas_report.json"
    assert run_json.is_file()
    data = json.loads(run_json.read_text())
    assert data["status"] == "fixture"
    assert data["n_rows"] > 0
    assert data["version_stamp"]["components"]["harness.experiments.slm215_spectral_atlas"] == "v1"
    assert design_json.is_file()
    assert design_md.is_file()
    assert "Honest caveats" in design_md.read_text()


def test_fixture_uses_reports_dir(tmp_path: Path) -> None:
    from slm_training.harnesses.experiments.slm214_spectral_snapshot import (
        run_spectral_snapshot_fixture,
    )

    spectral_report = run_spectral_snapshot_fixture(null_draws=5)
    spectral_report.to_json(tmp_path / "slm214_spectral_report.json")
    out = tmp_path / "atlas"
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--reports-dir",
                str(tmp_path),
                "--output-dir",
                str(out),
                "--no-write-design-docs",
            ]
        )
        == 0
    )
    data = json.loads((out / "slm215_spectral_atlas_report.json").read_text())
    assert data["n_rows"] > 0
    assert any("slm214_spectral_report.json" in s for s in data["source_reports"])
