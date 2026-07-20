"""Tests for scripts/run_slm135_trailed_assumptions_fixture.py (SLM-135)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import run_slm135_trailed_assumptions_fixture


def test_plan_only_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    rc = run_slm135_trailed_assumptions_fixture.main(
        ["--mode", "plan-only", "--output-dir", str(out_dir), "--seeds", "0"]
    )
    assert rc == 0
    assert (out_dir / "slm135_trailed_assumptions_report.json").exists()
    assert (out_dir / "slm135_trailed_assumptions_report.md").exists()
    report = json.loads(
        (out_dir / "slm135_trailed_assumptions_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "plan_only"
    assert report["claim_class"] == "wiring"
    assert "manifest" in report
    assert "version_stamp" in report


def test_fixture_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    rc = run_slm135_trailed_assumptions_fixture.main(
        ["--mode", "fixture", "--output-dir", str(out_dir), "--seeds", "0"]
    )
    assert rc == 0
    report_path = out_dir / "slm135_trailed_assumptions_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "fixture"
    assert report["verdict"] == "trail_required"
    assert "rows" in report
    ids = {row["arm_id"] for row in report["rows"]}
    assert "trail" in ids
    assert "monotone" in ids
    assert "partial" in ids
    assert "certified_only" in ids


def test_fixture_cli_prints_path(capsys, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    rc = run_slm135_trailed_assumptions_fixture.main(
        ["--mode", "plan-only", "--output-dir", str(out_dir), "--seeds", "0"]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert str(out_dir / "slm135_trailed_assumptions_report.json") in captured.out


def test_plan_only_default_out_path_uses_date() -> None:
    # The default path is under outputs/runs/ and includes today's date. We
    # verify the CLI still exits 0 when --output-dir is omitted.
    rc = run_slm135_trailed_assumptions_fixture.main(["--mode", "plan-only"])
    assert rc == 0


def test_plan_only_writes_design_docs(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    rc = run_slm135_trailed_assumptions_fixture.main(
        ["--mode", "plan-only", "--output-dir", str(out_dir), "--seeds", "0"]
    )
    assert rc == 0
    json_files = list(
        Path("docs/design").glob("iter-slm135-trailed-assumptions-*.json")
    )
    md_files = list(Path("docs/design").glob("iter-slm135-trailed-assumptions-*.md"))
    assert json_files
    assert md_files
