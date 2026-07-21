"""Tests for the SLM-227 Muon/AdamW convergence-direction sweep CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from scripts.run_slm227_muon_convergence import main


def test_plan_only_mode(tmp_path: Path) -> None:
    rc = main(
        [
            "--mode",
            "plan-only",
            "--output-dir",
            str(tmp_path),
            "--steps",
            "5",
            "--seeds",
            "0",
            "1",
        ]
    )
    assert rc == 0
    report = tmp_path / "slm227_muon_convergence_report.json"
    assert report.is_file()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "plan_only"
    assert payload["matrix_set"] == "slm227_muon_convergence"
    assert payload["comparisons"] == []


def test_fixture_mode_writes_design_docs(tmp_path: Path) -> None:
    design_json = tmp_path / "iter.json"
    design_md = tmp_path / "iter.md"
    rc = main(
        [
            "--mode",
            "fixture",
            "--output-dir",
            str(tmp_path),
            "--steps",
            "3",
            "--n-records",
            "2",
            "--seeds",
            "0",
            "1",
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
    assert len(payload["comparisons"]) == 2
    md_text = design_md.read_text(encoding="utf-8")
    assert "SLM-227" in md_text
    assert "No-go for promotion" in md_text
