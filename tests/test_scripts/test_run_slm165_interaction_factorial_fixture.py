"""Tests for the SLM-165 (SDE1-03) interaction factorial fixture CLI."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.run_slm165_interaction_factorial_fixture as _runner_module
from scripts.run_slm165_interaction_factorial_fixture import main


def test_plan_only_writes_manifest(tmp_path) -> None:
    assert main(["--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm165_interaction_factorial_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "plan_only"
    assert data["claim_class"] == "wiring"
    assert data["schema"] == "Slm165InteractionFactorialReportV1"
    assert "cells" in data
    cells = data["cells"]
    assert len(cells) == 24
    action_inits = {c["action_init"] for c in cells}
    assert "schema_description" in action_inits
    assert "current_stub" in action_inits


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
    run_json = tmp_path / "slm165_interaction_factorial_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "fixture"
    assert data["claim_class"] == "wiring"
    assert data["rows"]
    assert data["version_stamp"]

    root = Path(_runner_module.__file__).resolve().parents[1]
    design_json = root / "docs/design/iter-slm165-interaction-factorial-20260720.json"
    design_md = root / "docs/design/iter-slm165-interaction-factorial-20260720.md"
    assert design_json.exists()
    assert design_md.exists()
    design_data = json.loads(design_json.read_text())
    assert design_data["status"] == "fixture"
    assert design_data["experiment_id"] == "slm165-interaction-factorial"
