"""Tests for the SLM-241 D2 canonicalizer round-trip / alpha-invariance
stress probe CLI."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_slm241_grammar_round_trip_alpha_invariance import main


def test_plan_only_mode(tmp_path: Path) -> None:
    rc = main(["--mode", "plan-only", "--output-dir", str(tmp_path)])
    assert rc == 0
    report = tmp_path / "slm241_grammar_round_trip_alpha_invariance_report.json"
    assert report.is_file()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "plan_only"
    assert payload["matrix_set"] == "slm241_grammar_round_trip_alpha_invariance"
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
            "--count-per-seed",
            "8",
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
    assert len(payload["rows"]) == 8
    assert payload["disposition"] in {
        "ceiling_confirmed_at_scale",
        "gap_confirmed",
        "inconclusive",
    }
    md_text = design_md.read_text(encoding="utf-8")
    assert "SLM-241" in md_text
    assert "No-go for promotion" in md_text
