"""Tests for the SLM-231 RoleSlot cardinality dead-field consumption probe
CLI."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_slm231_role_cardinality_dead_field import main


def test_plan_only_mode(tmp_path: Path) -> None:
    rc = main(["--mode", "plan-only", "--output-dir", str(tmp_path)])
    assert rc == 0
    report = tmp_path / "slm231_role_cardinality_dead_field_report.json"
    assert report.is_file()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "plan_only"
    assert payload["matrix_set"] == "slm231_role_cardinality_dead_field"
    assert payload["rows"] == []


def test_fixture_mode_writes_design_docs(tmp_path: Path) -> None:
    design_json = tmp_path / "iter.json"
    design_md = tmp_path / "iter.md"
    rc = main(
        [
            "--mode",
            "fixture",
            "--corpus-size",
            "24",
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
    assert len(payload["rows"]) == 2
    assert payload["disposition"] == "cardinality_confirmed_unconsumed"
    md_text = design_md.read_text(encoding="utf-8")
    assert "SLM-231" in md_text
    assert "No-go for promotion" in md_text
