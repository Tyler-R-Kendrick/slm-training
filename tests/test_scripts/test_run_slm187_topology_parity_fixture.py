"""Tests for the SLM-187 (FFE1-01) topology parity fixture CLI."""

from __future__ import annotations

import json

from scripts.run_slm187_topology_parity_fixture import main


def test_plan_only_writes_manifest(tmp_path) -> None:
    assert main(["--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm187_topology_parity_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "plan_only"
    assert data["claim_class"] == "wiring"
    assert data["schema"] == "TopologyParityReportV1"
    assert "version_stamp" in data
    components = data["version_stamp"].get("components", {})
    assert "harness.experiments.slm187_topology_parity" in components
    assert "dsl.solver.topology" in components


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
                "--design-json",
                str(design_json),
                "--design-md",
                str(design_md),
                "--seed",
                "0",
            ]
        )
        == 0
    )
    run_json = tmp_path / "slm187_topology_parity_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "fixture"
    assert data["claim_class"] == "wiring"
    assert data["experiment_id"] == "slm187-topology-parity"
    assert data["cases"]
    assert data["version_stamp"]

    assert design_json.exists()
    assert design_md.exists()
    design_data = json.loads(design_json.read_text())
    assert design_data["status"] == "fixture"
    assert "cases" in design_data


def test_describe_prints_schema() -> None:
    import io
    import sys

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        assert main(["--mode", "describe"]) == 0
    finally:
        sys.stdout = old_stdout
    output = captured.getvalue()
    assert "SLM-187" in output
    assert "TopologyStateV2" in output
