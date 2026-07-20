"""Regression tests for the SLM-139 gate closeout script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_slm139_gate_closeout import main


@pytest.fixture
def args(tmp_path: Path) -> list[str]:
    return ["--output-dir", str(tmp_path / "out")]


def test_plan_only_emits_skeleton(args: list[str], tmp_path: Path) -> None:
    assert main([*args, "--mode", "plan-only"]) == 0
    report_path = tmp_path / "out" / "slm139_gate_closeout_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["matrix_set"] == "slm139-stochastic-recursive-width"
    assert report["status"] == "plan_only"
    assert report["claim_class"] == "wiring"
    assert "gates" in report


def test_closeout_records_gate_failure(args: list[str], tmp_path: Path) -> None:
    assert main([*args, "--mode", "closeout"]) == 0
    report_path = tmp_path / "out" / "slm139_gate_closeout_report.json"
    markdown_path = tmp_path / "out" / "slm139_gate_closeout_report.md"
    assert report_path.is_file()
    assert markdown_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "closeout"
    assert report["decision"] == "no_supported_probabilistic_regime"
    assert report["claim_class"] == "wiring"
    assert "SLM-138" in report["reason"]

    gates = report["gates"]
    assert gates["gate_1_recursive_base"]["passed"] is False
    assert gates["gate_2_multimodal_regime"]["issue"] == "SLM-130"
    assert gates["gate_3_selector"]["issue"] == "SLM-127"
    assert report["failed_gates"] == ["gate_1_recursive_base"]
    assert "high_trained" in report["blocked_arms"]
    assert "none" in report["allowed_arms"]

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "no_supported_probabilistic_regime" in markdown
    assert "SLM-138" in markdown
