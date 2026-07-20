"""Tests for scripts/run_sde4_02_min_controller_capacity.py (SLM-180)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_sde4_02_min_controller_capacity

torch = pytest.importorskip("torch")


def test_plan_only_mode(tmp_path: Path) -> None:
    out = tmp_path / "plan"
    rc = run_sde4_02_min_controller_capacity.main(
        ["--mode", "plan-only", "--output-dir", str(out), "--seeds", "0"]
    )
    assert rc == 0
    assert (out / "sde4_02_min_controller_capacity_manifest.json").exists()
    assert (out / "sde4_02_min_controller_capacity_report.json").exists()
    assert (out / "sde4_02_min_controller_capacity_report.md").exists()
    report_text = (out / "sde4_02_min_controller_capacity_report.json").read_text()
    assert "plan_only" in report_text


def test_fixture_mode(tmp_path: Path) -> None:
    out = tmp_path / "fixture"
    rc = run_sde4_02_min_controller_capacity.main(
        [
            "--mode",
            "fixture",
            "--output-dir",
            str(out),
            "--seeds",
            "0,1",
            "--rungs",
            "2",
            "--hidden-dims",
            "8,16",
            "--train-steps",
            "50",
        ]
    )
    assert rc == 0
    report_path = out / "sde4_02_min_controller_capacity_report.json"
    assert report_path.exists()
    report_text = report_path.read_text()
    assert "fixture" in report_text
    assert "sde4_02_fixture" in report_text
    assert "train_accuracy" in report_text


def test_plan_only_writes_design_docs(tmp_path: Path) -> None:
    out = tmp_path / "plan"
    rc = run_sde4_02_min_controller_capacity.main(
        ["--mode", "plan-only", "--output-dir", str(out), "--seeds", "0"]
    )
    assert rc == 0
    json_files = list(Path("docs/design").glob("iter-sde4-02-min-controller-capacity-*.json"))
    md_files = list(Path("docs/design").glob("iter-sde4-02-min-controller-capacity-*.md"))
    assert json_files
    assert md_files
