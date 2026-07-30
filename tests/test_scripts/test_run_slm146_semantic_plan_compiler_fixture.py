"""Tests for scripts/run_slm146_semantic_plan_compiler_fixture.py (SLM-146)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_slm146_semantic_plan_compiler_fixture


@pytest.fixture(autouse=True)
def _isolate_design_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_slm146_semantic_plan_compiler_fixture,
        "_DESIGN_JSON",
        str(tmp_path / "design.json"),
    )
    monkeypatch.setattr(
        run_slm146_semantic_plan_compiler_fixture,
        "_DESIGN_MD",
        str(tmp_path / "design.md"),
    )


def test_plan_only_cli_writes_json(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    rc = run_slm146_semantic_plan_compiler_fixture.main(
        ["--mode", "plan-only", "--output-dir", str(out_dir)]
    )
    assert rc == 0
    assert (out_dir / "slm146_semantic_plan_compiler_report.json").exists()
    report = json.loads(
        (out_dir / "slm146_semantic_plan_compiler_report.json").read_text(encoding="utf-8")
    )
    assert report["schema"] == "Slm146PlanCompilerReportV1"
    assert report["status"] == "plan_only"
    assert report["claim_class"] == "wiring"
    assert "manifest" in report
    assert "version_stamp" in report


def test_fixture_cli_writes_json_and_design_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    rc = run_slm146_semantic_plan_compiler_fixture.main(
        ["--mode", "fixture", "--output-dir", str(out_dir)]
    )
    assert rc == 0
    report_path = out_dir / "slm146_semantic_plan_compiler_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "Slm146PlanCompilerReportV1"
    assert report["status"] == "fixture"
    assert "rows" in report
    assert "manifest" in report
    ids = {row["arm_id"] for row in report["rows"]}
    assert "A_baseline" in ids
    assert "F_unsafe_predicted_hard" in ids

    design_json = tmp_path / "design.json"
    design_md = tmp_path / "design.md"
    assert design_json.exists()
    assert design_md.exists()


def test_fixture_cli_prints_path(capsys, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    rc = run_slm146_semantic_plan_compiler_fixture.main(
        ["--mode", "plan-only", "--output-dir", str(out_dir)]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert str(out_dir / "slm146_semantic_plan_compiler_report.json") in captured.out


def test_plan_only_default_out_path_uses_date() -> None:
    # The default path is under outputs/runs/ and includes today's date. We
    # verify the CLI still exits 0 when --output-dir is omitted.
    rc = run_slm146_semantic_plan_compiler_fixture.main(["--mode", "plan-only"])
    assert rc == 0
