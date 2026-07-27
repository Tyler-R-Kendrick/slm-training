"""Tests for the SLM-267 (VSD2-02) ProgramSpec coverage-scaling fixture CLI."""

from __future__ import annotations

import json

from scripts.generate_programspec_corpus import main


def test_plan_only_writes_manifest(tmp_path) -> None:
    assert main(["--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm267_programspec_coverage_scaling_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "plan_only"
    assert data["claim_class"] == "wiring"
    assert data["schema"] == "Slm267ProgramspecCoverageScalingManifestV1"
    assert data["version_stamp"]


def test_fixture_writes_design_docs(tmp_path) -> None:
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--output-dir",
                str(tmp_path),
                "--target-count",
                "80",
                "--seed",
                "0",
                "--shards",
                "2",
            ]
        )
        == 0
    )
    run_json = tmp_path / "slm267_programspec_coverage_scaling_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "inconclusive"
    assert data["claim_class"] == "wiring"
    assert data["arms"]
    assert data["manifest_hash"]
