"""Tests for the SLM-159 cross-DSL replication fixture CLI."""

from __future__ import annotations

import json

import pytest

from scripts.run_slm159_cross_dsl_replication_fixture import main


def test_plan_only_mode_writes_manifest(tmp_path) -> None:
    assert main(["--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm159_cross_dsl_replication_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "plan_only"
    assert data["claim_class"] == "wiring"
    assert data["schema"] == "Slm159CrossDslReplicationReportV1"
    assert "manifest" in data
    arms = data["manifest"]["arms"]
    assert any(arm["arm_id"] == "G1_graphql" for arm in arms)
    assert any(arm["arm_id"] == "S1_second_pack" for arm in arms)


def test_fixture_mode_writes_run_json_without_publishing_history(tmp_path, monkeypatch) -> None:
    from scripts import run_slm159_cross_dsl_replication_fixture as cli

    historical = tmp_path / "historical"
    historical.mkdir()
    monkeypatch.setattr(cli, "_DESIGN_JSON", str(historical / "result.json"))
    monkeypatch.setattr(cli, "_DESIGN_MD", str(historical / "result.md"))
    from slm_training.dsl.grammar.backends.graphql_js import bridge_available

    pytest.importorskip("slm_training.dsl.grammar.backends.graphql_js")
    if not bridge_available():
        pytest.skip("graphql bridge unavailable")
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--output-dir",
                str(tmp_path),
                "--n-graphql-records",
                "4",
                "--graphql-depth",
                "1",
                "--seeds",
                "0",
            ]
        )
        == 0
    )
    run_json = tmp_path / "slm159_cross_dsl_replication_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "fixture"
    assert data["claim_class"] == "wiring"
    assert data["rows"]
    assert data["version_stamp"]
    assert (tmp_path / "slm159_cross_dsl_replication_report.md").is_file()
    assert not list(historical.iterdir())
    readiness = data["readiness_reports"]
    assert any(r["pack_id"] == "graphql" for r in readiness)
