"""Tests for the SLM-157 flow / consistency / trajectory-imitation fixture CLI."""

from __future__ import annotations

import json

from scripts.run_slm157_flow_consistency_fixture import main


def test_plan_only_mode_writes_manifest(tmp_path) -> None:
    assert main(["--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm157_flow_consistency_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "plan_only"
    assert data["claim_class"] == "wiring"
    assert data["schema"] == "Slm157FlowConsistencyReportV1"
    assert "manifest" in data
    arms = data["manifest"]["arms"]
    assert any(arm["arm_id"] == "A_teacher_long_x22" for arm in arms)
    assert any(arm["arm_id"] == "H_oracle_boundary" for arm in arms)


def test_fixture_mode_writes_run_json(tmp_path) -> None:
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--output-dir",
                str(tmp_path),
                "--n-records",
                "2",
                "--steps",
                "4",
                "--seeds",
                "0",
            ]
        )
        == 0
    )
    run_json = tmp_path / "slm157_flow_consistency_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "fixture"
    assert data["claim_class"] == "wiring"
    assert data["rows"]
    assert data["version_stamp"]
    assert any(r["arm_id"] == "H_oracle_boundary" for r in data["rows"])
