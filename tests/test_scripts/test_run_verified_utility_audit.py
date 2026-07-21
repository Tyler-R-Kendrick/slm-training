"""Tests for the SLM-186 verified-utility audit CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_verified_utility_audit import main


def test_describe_mode(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--mode", "describe"]) == 0
    captured = capsys.readouterr()
    assert "SLM-186 verified-utility audit schema" in captured.out
    assert "VerifiedUtilityV1 factors" in captured.out


def test_fixture_mode_writes_report(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    assert main(["--mode", "fixture", "--output-dir", str(output_dir), "--seed", "0"]) == 0
    report_path = output_dir / "verified_utility_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "VerifiedUtilityAuditReportV1"
    assert report["status"] == "fixture"
    assert report["claim_class"] == "wiring"
    assert report["candidates"]
    assert "version_stamp" in report


def test_fixture_mode_writes_design_docs(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    design_json = tmp_path / "design.json"
    design_md = tmp_path / "design.md"
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--output-dir",
                str(output_dir),
                "--write-design-docs",
                "--design-json",
                str(design_json),
                "--design-md",
                str(design_md),
                "--seed",
                "0",
            ]
        )
        == 0
    )
    assert design_json.is_file()
    assert design_md.is_file()
    assert "SLM-186" in design_md.read_text(encoding="utf-8")


def test_analyze_history_mode(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    assert main(["--mode", "fixture", "--output-dir", str(output_dir), "--seed", "0"]) == 0
    report_path = output_dir / "verified_utility_report.json"

    analysis_dir = tmp_path / "analysis"
    assert (
        main(
            [
                "--mode",
                "analyze-history",
                "--output-dir",
                str(analysis_dir),
                "--history",
                str(report_path),
            ]
        )
        == 0
    )
    analysis_path = analysis_dir / "verified_utility_history_analysis.json"
    assert analysis_path.is_file()
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "VerifiedUtilityHistoryAnalysisV1"
    assert payload["claim_class"] == "wiring"
    assert payload["n_candidates"] > 0
    assert "utility_table" in payload


def test_sensitivity_mode(tmp_path: Path) -> None:
    output_dir = tmp_path / "sensitivity"
    assert main(["--mode", "sensitivity", "--output-dir", str(output_dir), "--seed", "0"]) == 0
    path = output_dir / "verified_utility_sensitivity.json"
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "VerifiedUtilitySensitivityV1"
    assert "sensitivity" in payload
    assert "reversal_count" in payload["sensitivity"]


def test_analyze_history_requires_history(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--mode", "analyze-history", "--output-dir", str(tmp_path)]) != 0
    assert "--history" in capsys.readouterr().err
