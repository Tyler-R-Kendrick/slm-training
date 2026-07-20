"""Tests for the SLM-156 plan-refinement fixture CLI."""

from __future__ import annotations

import json

import pytest

from scripts.run_slm156_plan_refinement_fixture import main


def test_plan_only_mode_writes_manifest(tmp_path) -> None:
    assert main(["--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm156_plan_refinement_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "plan_only"
    assert data["claim_class"] == "wiring"
    assert data["schema"] == "Slm156PlanRefinementReportV1"
    assert "manifest" in data
    arms = data["manifest"]["arms"]
    assert any(arm["arm_id"] == "A_one_pass" for arm in arms)
    assert any(arm["arm_id"] == "H_gold_oracle" for arm in arms)


def test_fixture_mode_writes_run_json(tmp_path) -> None:
    pytest.importorskip("torch")
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--output-dir",
                str(tmp_path),
                "--n-train",
                "32",
                "--n-eval",
                "8",
                "--epochs",
                "5",
                "--seeds",
                "0",
            ]
        )
        == 0
    )
    run_json = tmp_path / "slm156_plan_refinement_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "fixture"
    assert data["claim_class"] == "wiring"
    assert data["rows"]
    assert data["version_stamp"]
