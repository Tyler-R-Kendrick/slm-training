"""Tests for the SLM-163 schema-description action-embedding fixture CLI."""

from __future__ import annotations

import json

from scripts.run_slm163_schema_action_embedding_fixture import main


def test_plan_only_mode_writes_manifest(tmp_path) -> None:
    assert main(["--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm163_schema_action_embedding_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "plan_only"
    assert data["claim_class"] == "wiring"
    assert data["schema"] == "Slm163SchemaActionEmbeddingReportV1"
    assert "arms" in data
    sources = {arm["source"] for arm in data["arms"]}
    assert "schema_description" in sources
    assert "expanded_description" in sources


def test_fixture_mode_writes_run_json(tmp_path) -> None:
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
    run_json = tmp_path / "slm163_schema_action_embedding_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "fixture"
    assert data["claim_class"] == "wiring"
    assert data["rows"]
    assert data["version_stamp"]
    assert any(r["source"] == "schema_description" for r in data["rows"])
