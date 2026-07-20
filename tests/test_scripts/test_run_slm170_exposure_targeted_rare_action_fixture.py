"""Tests for the SLM-170 (SDE2-03) exposure-targeted rare-action sampling fixture CLI."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.run_slm170_exposure_targeted_rare_action_fixture as _runner_module
from scripts.run_slm170_exposure_targeted_rare_action_fixture import main


def test_plan_only_writes_manifest(tmp_path) -> None:
    assert main(["--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm170_exposure_targeted_rare_action_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "plan_only"
    assert data["claim_class"] == "wiring"
    assert data["schema"] == "Slm170ExposureTargetedRareActionReportV1"
    assert "cells" in data
    cells = data["cells"]
    # 5 arms × 3 default seeds.
    assert len(cells) == 15
    names = {c["arm_name"] for c in cells}
    assert "current" in names
    assert "minimum_exposure_floor" in names


def test_fixture_writes_design_docs(tmp_path) -> None:
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--output-dir",
                str(tmp_path),
                "--seeds",
                "0",
            ]
        )
        == 0
    )
    run_json = tmp_path / "slm170_exposure_targeted_rare_action_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "fixture"
    assert data["claim_class"] == "wiring"
    assert data["rows"]
    assert data["version_stamp"]

    root = Path(_runner_module.__file__).resolve().parents[1]
    design_json = root / "docs/design/iter-slm170-exposure-targeted-rare-action-20260720.json"
    design_md = root / "docs/design/iter-slm170-exposure-targeted-rare-action-20260720.md"
    assert design_json.exists()
    assert design_md.exists()
    design_data = json.loads(design_json.read_text())
    assert design_data["status"] == "fixture"
    assert design_data["experiment_id"] == "slm170-exposure-targeted-rare-action"
