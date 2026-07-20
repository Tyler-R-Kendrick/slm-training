"""Tests for the SLM-155 factorization comparison fixture CLI."""

from __future__ import annotations

import json

import pytest

from scripts.run_slm155_factorization_comparison_fixture import main


def test_plan_only_mode_writes_manifest(tmp_path) -> None:
    assert main(["--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm155_factorization_comparison_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "plan_only"


def test_fixture_mode_writes_run_json(tmp_path) -> None:
    pytest.importorskip("torch")
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--output-dir",
                str(tmp_path),
                "--n-records",
                "4",
                "--scorer-steps",
                "10",
            ]
        )
        == 0
    )
    run_json = tmp_path / "slm155_factorization_comparison_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "fixture"
    assert data["claim_class"] == "wiring"
    assert data["rows"]
