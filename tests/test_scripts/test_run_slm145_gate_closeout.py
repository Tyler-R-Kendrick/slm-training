"""Regression tests for the SLM-145 gate closeout script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_slm145_gate_closeout import main


@pytest.fixture
def args(tmp_path: Path) -> list[str]:
    return ["--output-dir", str(tmp_path / "out")]


def test_plan_only_emits_skeleton(args: list[str], tmp_path: Path) -> None:
    assert main([*args, "--mode", "plan-only"]) == 0
    report_path = tmp_path / "out" / "slm145_gate_closeout_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["matrix_set"] == "slm145-plan-predictor-factors"
    assert report["status"] == "plan_only"
    assert report["claim_class"] == "wiring"
    assert "factors" in report


def test_closeout_records_gate_failure(args: list[str], tmp_path: Path) -> None:
    assert main([*args, "--mode", "closeout"]) == 0
    report_path = tmp_path / "out" / "slm145_gate_closeout_report.json"
    markdown_path = tmp_path / "out" / "slm145_gate_closeout_report.md"
    assert report_path.is_file()
    assert markdown_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "closeout"
    assert report["decision"] == "blocked_pending_spv0_02_ceiling_evidence"
    assert report["claim_class"] == "wiring"
    assert "SPV0-02" in report["reason"]

    factors = report["factors"]
    assert not factors["topology"]["ceiling_observed"]
    assert not factors["cardinality"]["ceiling_observed"]
    assert not factors["bindings_pointers"]["ceiling_observed"]
    assert report["blocked_heads"] == [
        "topology_head",
        "cardinality_head",
        "live_symbol_pointer_head",
    ]
    assert "Run a factor-wise oracle-substitution matrix" in report["recommended_next_step"]

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "blocked_pending_spv0_02_ceiling_evidence" in markdown
    assert "topology_head" in markdown
