"""Tests for SLM-214 (NCS0-01) inspect_spectral CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.inspect_spectral import main

pytest.importorskip("torch")


def test_describe_mode(capsys) -> None:
    assert main(["--describe"]) == 0
    out = capsys.readouterr().out
    assert "SpectralSnapshotV1" in out
    assert "claim_class" in out


def test_plan_only_writes_manifest(tmp_path: Path) -> None:
    assert main(["spectral", "--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm214_spectral_report.json"
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
                "spectral",
                "--mode",
                "fixture",
                "--output-dir",
                str(tmp_path),
                "--null-draws",
                "5",
                "--write-design-docs",
                "--design-json",
                str(design_json),
                "--design-md",
                str(design_md),
            ]
        )
        == 0
    )
    run_json = tmp_path / "slm214_spectral_report.json"
    assert run_json.is_file()
    data = json.loads(run_json.read_text())
    assert data["status"] == "fixture"
    assert data["n_matrices"] > 0
    assert data["version_stamp"]["components"]["harness.experiments.slm214_spectral_snapshot"] == "v2"
    assert design_json.is_file()
    assert design_md.is_file()
    assert "Honest caveats" in design_md.read_text()


def test_spectral_null_subcommand(tmp_path: Path) -> None:
    assert (
        main(
            [
                "spectral-null",
                "--shape",
                "16x16",
                "--draws",
                "5",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    out = tmp_path / "slm214_spectral_null_summary.json"
    assert out.is_file()
    data = json.loads(out.read_text())
    assert data["schema"] == "SpectralNullSummaryV1"
    assert data["mean_alpha"] is not None


def test_spectral_compare_subcommand(tmp_path: Path, capsys) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    main(["spectral", "--mode", "fixture", "--output-dir", str(tmp_path / "a"), "--null-draws", "5"])
    main(["spectral", "--mode", "fixture", "--output-dir", str(tmp_path / "b"), "--null-draws", "5"])
    left.write_text((tmp_path / "a" / "slm214_spectral_report.json").read_text())
    right.write_text((tmp_path / "b" / "slm214_spectral_report.json").read_text())
    assert main(["spectral-compare", "--left", str(left), "--right", str(right)]) == 0
    out = capsys.readouterr().out
    assert "matrix_id" in out


def test_toy_model_runs_without_checkpoint(tmp_path: Path) -> None:
    assert main(["spectral", "--output-dir", str(tmp_path), "--null-draws", "5"]) == 0
    data = json.loads((tmp_path / "slm214_spectral_report.json").read_text())
    assert data["n_matrices"] > 0
