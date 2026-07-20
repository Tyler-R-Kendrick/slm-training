"""Tests for the SLM-174 (SDE2-07) action-alias generalization fixture CLI."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.run_slm174_action_alias_generalization_fixture as _runner_module
from scripts.run_slm174_action_alias_generalization_fixture import main
from slm_training.harnesses.experiments.slm174_action_alias_generalization import (
    ARM_NAMES,
)


def test_plan_only_writes_manifest(tmp_path) -> None:
    assert main(["--mode", "plan-only", "--output-dir", str(tmp_path), "--seeds", "0"]) == 0
    run_json = tmp_path / "slm174_action_alias_generalization_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "plan_only"
    assert data["claim_class"] == "wiring"
    assert data["schema"] == "Slm174ActionAliasGeneralizationReportV1"
    assert "cells" in data
    cells = data["cells"]
    assert len(cells) == len(ARM_NAMES)
    names = {c["arm_name"] for c in cells}
    assert "canonical_name_plus_description" in names
    assert "canonical_evaluated_under_unseen_alias" in names


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
                "--d-model",
                "32",
            ]
        )
        == 0
    )
    run_json = tmp_path / "slm174_action_alias_generalization_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "fixture"
    assert data["claim_class"] == "wiring"
    assert data["rows"]
    assert data["version_stamp"]

    root = Path(_runner_module.__file__).resolve().parents[1]
    design_json = root / "docs/design/iter-slm174-action-alias-generalization-20260720.json"
    design_md = root / "docs/design/iter-slm174-action-alias-generalization-20260720.md"
    assert design_json.exists()
    assert design_md.exists()
    design_data = json.loads(design_json.read_text())
    assert design_data["status"] == "fixture"
    assert design_data["experiment_id"] == "slm174-action-alias-generalization"
