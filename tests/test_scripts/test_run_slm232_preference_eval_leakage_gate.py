"""Tests for the SLM-232 preference build-pairs eval-holdout leakage gate
stress test CLI."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_slm232_preference_eval_leakage_gate import main


def test_plan_only_mode(tmp_path: Path) -> None:
    rc = main(["--mode", "plan-only", "--output-dir", str(tmp_path)])
    assert rc == 0
    report = tmp_path / "slm232_preference_eval_leakage_gate_report.json"
    assert report.is_file()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "plan_only"
    assert payload["matrix_set"] == "slm232_preference_eval_leakage_gate"
    assert payload["results"] == []


def test_fixture_mode_writes_design_docs(tmp_path: Path) -> None:
    design_json = tmp_path / "iter.json"
    design_md = tmp_path / "iter.md"
    rc = main(
        [
            "--mode",
            "fixture",
            "--output-dir",
            str(tmp_path),
            "--design-json",
            str(design_json),
            "--design-md",
            str(design_md),
        ]
    )
    assert rc == 0
    assert design_json.is_file()
    assert design_md.is_file()
    payload = json.loads(design_json.read_text(encoding="utf-8"))
    assert payload["status"] == "fixture"
    assert len(payload["results"]) == 5
    assert payload["disposition"] == "gap_confirmed"
    md_text = design_md.read_text(encoding="utf-8")
    assert "SLM-232" in md_text
    assert "No-go for promotion" in md_text
