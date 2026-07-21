"""Tests for scripts/run_exact_flow_fixture.py."""

from __future__ import annotations

import json

from scripts.run_exact_flow_fixture import main


def test_describe_mode_prints_schema(capsys) -> None:
    rc = main(["--mode", "describe"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "SLM-190" in captured.out
    assert "ExactFlowReport" in captured.out


def test_plan_only_writes_manifest(tmp_path) -> None:
    out = tmp_path / "out"
    rc = main(["--mode", "plan-only", "--output-dir", str(out)])
    assert rc == 0
    report_json = out / "slm190_exact_flow_report.json"
    assert report_json.exists()
    data = json.loads(report_json.read_text())
    assert data["status"] == "plan_only"
    assert data["matrix_set"] == "slm190_exact_flow"
    assert data["n_domains"] == 3


def test_fixture_mode_writes_design_docs(tmp_path) -> None:
    out = tmp_path / "out"
    design_json = tmp_path / "iter-slm190-exact-flow-test.json"
    design_md = tmp_path / "iter-slm190-exact-flow-test.md"
    rc = main(
        [
            "--mode",
            "fixture",
            "--output-dir",
            str(out),
            "--rate-fn-names",
            "uniform_rate",
            "--times",
            "1.0",
            "--seed",
            "0",
            "--design-json",
            str(design_json),
            "--design-md",
            str(design_md),
        ]
    )
    assert rc == 0
    assert design_json.exists()
    assert design_md.exists()
    data = json.loads(design_json.read_text())
    assert data["status"] == "fixture"
    assert data["claim_class"] == "wiring"
    assert "disposition" in data
    md = design_md.read_text()
    assert "No-go for promotion" in md
