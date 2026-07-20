"""Tests for the SLM-164 (SDE1-02) targeted-margin fixture CLI."""

from __future__ import annotations

import json

from scripts.run_slm164_targeted_margin_fixture import main


def test_plan_only_mode_writes_manifest(tmp_path) -> None:
    assert main(["--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm164_targeted_margin_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "plan_only"
    assert data["claim_class"] == "wiring"
    assert data["schema"] == "Slm164TargetedMarginReportV1"
    assert "arms" in data
    sources = {arm["source"] for arm in data["arms"]}
    assert "targeted_weighted" in sources
    assert "shuffled" in sources


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
                "--margin",
                "1.0",
            ]
        )
        == 0
    )
    run_json = tmp_path / "slm164_targeted_margin_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "fixture"
    assert data["claim_class"] == "wiring"
    assert data["rows"]
    assert data["version_stamp"]
    assert any(r["source"] == "targeted_weighted" for r in data["rows"])
