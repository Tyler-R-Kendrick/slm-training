"""Tests for the SLM-240 learning-rate schedule gap probe CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from scripts.run_slm240_lr_schedule_gap import main


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
        ]
    )
    assert rc == 0
    report = tmp_path / "slm240_lr_schedule_gap_report.json"
    assert report.is_file()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "plan_only"
    assert payload["matrix_set"] == "slm240_lr_schedule_gap"
    assert payload["arms"] == []


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
            "2",
            "--n-records",
            "4",
            "--optimizers",
            "adamw",
            "--seeds",
            "0",
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
    assert len(payload["arms"]) == 1
    md_text = design_md.read_text(encoding="utf-8")
    assert "SLM-240" in md_text
    assert "No-go for any 'schedule already works' claim" in md_text
