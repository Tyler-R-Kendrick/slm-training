"""Tests for the SLM-189 bridge planner audit CLI."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_bridge_planner_audit import main


def test_plan_only_mode(tmp_path: Path) -> None:
    rc = main(
        [
            "--mode",
            "plan-only",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    report = tmp_path / "slm189_bridge_planner_report.json"
    assert report.is_file()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "plan_only"
    assert payload["matrix_set"] == "slm189_bridge_planner"
    assert len(payload["arms"]) == 7


def test_fixture_mode_writes_design_docs(tmp_path: Path) -> None:
    design_json = tmp_path / "iter.json"
    design_md = tmp_path / "iter.md"
    rc = main(
        [
            "--mode",
            "fixture",
            "--output-dir",
            str(tmp_path),
            "--arms",
            "canonical_greedy,dependency_dag",
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
    assert "canonical_greedy" in {a["arm_name"] for a in payload["arms"]}
    md_text = design_md.read_text(encoding="utf-8")
    assert "SLM-189" in md_text
    assert "No-go for promotion" in md_text


def test_fixture_mode_emits_training_manifest(tmp_path: Path) -> None:
    rc = main(
        [
            "--mode",
            "fixture",
            "--output-dir",
            str(tmp_path),
            "--arms",
            "canonical_greedy",
            "--emit-training-manifest",
        ]
    )
    assert rc == 0
    training = tmp_path / "slm189_bridge_planner_training_manifest.json"
    assert training.is_file()
    payload = json.loads(training.read_text(encoding="utf-8"))
    assert payload["schema"] == "BridgePlannerManifestV1"
    assert payload["permitted_arms"] == ["canonical_greedy"]
    assert payload["source_policy"] == "minimal"
