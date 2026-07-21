"""Tests for the SLM-209 (SDE5-02) debt-targeted curriculum fixture CLI."""

from __future__ import annotations

import json

from scripts.run_slm209_debt_targeted_curriculum_fixture import main


def test_plan_only_writes_manifest(tmp_path) -> None:
    assert main(["--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm209_debt_targeted_curriculum_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "plan_only"
    assert data["claim_class"] == "wiring"
    assert data["schema"] == "DebtCurriculumManifestV1"
    assert "cells" in data
    cells = data["cells"]
    # 6 policies × 3 default seeds.
    assert len(cells) == 18
    names = {c["policy_name"] for c in cells}
    assert "uniform" in names
    assert "preregistered_composite" in names
    assert "version_stamp" in data
    components = data["version_stamp"].get("components", {})
    assert "harness.experiments.slm209_debt_targeted_curriculum" in components


def test_plan_only_uses_custom_budget(tmp_path) -> None:
    assert (
        main(
            [
                "--mode",
                "plan-only",
                "--output-dir",
                str(tmp_path),
                "--total-decision-budget",
                "50",
                "--per-group-cap",
                "3",
            ]
        )
        == 0
    )
    data = json.loads((tmp_path / "slm209_debt_targeted_curriculum_report.json").read_text())
    assert data["total_decision_budget"] == 50
    assert data["per_group_cap"] == 3


def test_fixture_writes_design_docs(tmp_path) -> None:
    design_json = tmp_path / "design.json"
    design_md = tmp_path / "design.md"
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--output-dir",
                str(tmp_path),
                "--seeds",
                "0",
                "--n-states",
                "60",
                "--design-json",
                str(design_json),
                "--design-md",
                str(design_md),
            ]
        )
        == 0
    )
    run_json = tmp_path / "slm209_debt_targeted_curriculum_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "fixture"
    assert data["claim_class"] == "wiring"
    assert data["cells"]
    assert data["version_stamp"]

    assert design_json.exists()
    assert design_md.exists()
    design_data = json.loads(design_json.read_text())
    assert design_data["status"] == "fixture"
    assert design_data["experiment_id"] == "slm209-debt-targeted-curriculum"
    assert "cells" in design_data
