"""Tests for the SLM-234 (CKM0-01) merge interference recovery probe CLI."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_slm234_merge_interference_recovery import main


def test_plan_only_mode(tmp_path: Path) -> None:
    rc = main(["--mode", "plan-only", "--output-dir", str(tmp_path)])
    assert rc == 0
    report = tmp_path / "slm234_merge_interference_recovery_report.json"
    assert report.is_file()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "plan_only"
    assert payload["matrix_set"] == "slm234_merge_interference_recovery"
    assert payload["rows"] == []


def test_fixture_mode_writes_design_docs(tmp_path: Path) -> None:
    design_json = tmp_path / "iter.json"
    design_md = tmp_path / "iter.md"
    rc = main(
        [
            "--mode",
            "fixture",
            "--seeds",
            "0",
            "1",
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
    # 2 seeds x 4 default conflict_probs x 2 methods = 16 rows.
    assert len(payload["rows"]) == 16
    assert payload["disposition"] in {
        "fully_confirmed",
        "partial_confirmation_mechanism_specific",
        "no_advantage_detected",
    }
    md_text = design_md.read_text(encoding="utf-8")
    assert "SLM-234" in md_text
    assert "No-go for promotion" in md_text
