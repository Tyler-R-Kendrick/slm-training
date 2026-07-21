"""Tests for the SLM-239 checkpoint-migrate output-head corruption CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from scripts.run_slm239_checkpoint_migrate_tied_head_corruption import main


def test_plan_only_mode(tmp_path: Path) -> None:
    rc = main(
        [
            "--mode",
            "plan-only",
            "--output-dir",
            str(tmp_path),
            "--seeds",
            "0",
            "1",
        ]
    )
    assert rc == 0
    report = tmp_path / "slm239_checkpoint_migrate_tied_head_corruption_report.json"
    assert report.is_file()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "plan_only"
    assert payload["matrix_set"] == "slm239_checkpoint_migrate_tied_head_corruption"
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
    assert len(payload["results"]) == 2  # 1 seed x 2 tie arms
    md_text = design_md.read_text(encoding="utf-8")
    assert "SLM-239" in md_text
    assert "No-go for trusting migrate_twotower_checkpoint" in md_text
