"""Tests for scripts/run_sae_diagnostic_fixture.py (SLM-136)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_sae_diagnostic_fixture


def test_plan_only_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    rc = run_sae_diagnostic_fixture.main(
        ["--mode", "plan-only", "--output-dir", str(out_dir)]
    )
    assert rc == 0
    assert (out_dir / "ldi4_02_sae_diagnostic_report.json").exists()
    assert (out_dir / "ldi4_02_sae_diagnostic_report.md").exists()
    report = json.loads(
        (out_dir / "ldi4_02_sae_diagnostic_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "plan_only"
    assert report["claim_class"] == "wiring"
    assert "version_stamp" in report
    assert len(report["arms"]) == 8


@pytest.mark.training
def test_fixture_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    rc = run_sae_diagnostic_fixture.main(
        [
            "--mode",
            "fixture",
            "--output-dir",
            str(out_dir),
            "--d-in",
            "8",
            "--n",
            "16",
            "--seed",
            "0",
        ]
    )
    assert rc == 0
    report_path = out_dir / "ldi4_02_sae_diagnostic_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "wiring_only"
    assert report["claim_class"] == "wiring"
    assert "version_stamp" in report
    ids = {arm["arm_id"] for arm in report["arms"]}
    assert ids == {"S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"}


def test_fixture_cli_prints_path(capsys, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    rc = run_sae_diagnostic_fixture.main(
        ["--mode", "plan-only", "--output-dir", str(out_dir)]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert str(out_dir / "ldi4_02_sae_diagnostic_report.json") in captured.out


def test_plan_only_default_out_path_uses_date() -> None:
    rc = run_sae_diagnostic_fixture.main(["--mode", "plan-only"])
    assert rc == 0


def test_plan_only_writes_design_docs(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    rc = run_sae_diagnostic_fixture.main(
        ["--mode", "plan-only", "--output-dir", str(out_dir)]
    )
    assert rc == 0
    json_files = list(
        Path("docs/design").glob("iter-ldi4-02-sae-decision-state-diagnostic-*.json")
    )
    md_files = list(
        Path("docs/design").glob("iter-ldi4-02-sae-decision-state-diagnostic-*.md")
    )
    assert json_files
    assert md_files
