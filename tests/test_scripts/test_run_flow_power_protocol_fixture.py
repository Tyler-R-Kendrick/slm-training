"""Tests for the SLM-183 (PQR) power-protocol fixture CLI."""

from __future__ import annotations

import json

from scripts.run_flow_power_protocol import main


def test_plan_only_writes_manifest(tmp_path) -> None:
    assert (
        main(
            [
                "--mode",
                "plan-only",
                "--output-dir",
                str(tmp_path),
                "--seeds",
                "0,1",
            ]
        )
        == 0
    )
    run_json = tmp_path / "slm183_power_protocol_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "plan_only"
    assert data["claim_class"] == "wiring"
    assert data["schema"] == "Slm183PowerProtocolReportV1"
    assert "manifest" in data


def test_fixture_writes_design_docs(tmp_path) -> None:
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--output-dir",
                str(tmp_path),
                "--n-targets",
                "8",
                "--paths-per-target",
                "2",
                "--n-seeds",
                "2",
                "--seeds",
                "0,1",
            ]
        )
        == 0
    )
    run_json = tmp_path / "slm183_power_protocol_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "fixture"
    assert data["claim_class"] == "wiring"
    assert data["cells"]
    assert data["version_stamp"]

    assert data["experiment_campaign"]["seeds"] == [0, 1]
    assert {cell["seed"] for cell in data["cells"]} == {0, 1}
    assert {cell["arm_id"] for cell in data["cells"]} == {
        "synthetic_control",
        "synthetic_candidate",
    }


def test_analyze_existing_writes_report(tmp_path) -> None:
    records = [
        {"example_id": "ex1", "seed": 0, "pass": True},
        {"example_id": "ex2", "seed": 0, "pass": False},
        {"example_id": "ex3", "seed": 1, "pass": True},
    ]
    iter_path = tmp_path / "iter.json"
    iter_path.write_text(json.dumps({"records": records}))
    assert (
        main(
            [
                "--mode",
                "analyze-existing",
                "--output-dir",
                str(tmp_path),
                "--iter-json",
                str(iter_path),
            ]
        )
        == 0
    )
    run_json = tmp_path / "slm183_power_protocol_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "analysis"
    assert data["claim_class"] == "wiring"
    assert data["analysis"]["n_records"] == 3


def test_analyze_existing_requires_iter_json(tmp_path) -> None:
    assert main(["--mode", "analyze-existing", "--output-dir", str(tmp_path)]) == 2
