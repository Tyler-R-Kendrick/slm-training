"""Tests for the SLM-474 (SRP-010) pack portability fixture CLI."""

from __future__ import annotations

import json

from scripts.run_srp010_pack_portability_fixture import main


def test_plan_only_mode_writes_manifest(tmp_path) -> None:
    assert main(["--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "srp010_pack_portability_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "plan_only"
    assert data["claim_class"] == "fixture"
    assert data["schema"] == "Slm474Srp010PackPortabilityReportV1"
    assert data["related_catalogue_id"] == "exp-sr-12"
    assert data["version_stamp"]
    assert data["steps"]


def test_fixture_mode_writes_run_json(tmp_path) -> None:
    assert main(["--mode", "fixture", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "srp010_pack_portability_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] in {"fixture", "fixture_failed"}
    assert data["claim_class"] == "fixture"
    assert data["pack_id"] == "symbolic_regression"
    assert data["version_stamp"]
    assert data["forks"]
    assert data["unsupported_hooks"]
    assert data["sygus_capability"]["conformance"] == "inspired_interoperability"
    assert data["default_isolation"]["openui_is_default"] is True
