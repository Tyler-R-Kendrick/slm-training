"""Tests for the SLM-160 (SPV4-02) disposition audit CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.run_slm160_spv_disposition as _runner_module
from scripts.run_slm160_spv_disposition import main


@pytest.fixture(autouse=True)
def _isolate_design_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_runner_module, "_DESIGN_JSON", str(tmp_path / "design.json"))
    monkeypatch.setattr(_runner_module, "_DESIGN_MD", str(tmp_path / "design.md"))


def test_plan_only_mode_writes_json(tmp_path) -> None:
    assert main(["--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm160_spv_disposition_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "plan_only"
    assert data["schema"] == "SPVDispositionV1"
    assert data["claim_class"] == "wiring"
    assert data["mechanism_dispositions"]
    assert data["version_stamp"]


def test_fixture_mode_writes_json_and_docs(tmp_path) -> None:
    assert main(["--mode", "fixture", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm160_spv_disposition_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "fixture"
    assert data["schema"] == "SPVDispositionV1"
    assert data["claim_class"] == "wiring"
    assert data["version_stamp"]
    assert data["mechanism_dispositions"]
    assert data["cross_pack_summary"]
    assert data["canonical_architecture_recommendation"]

    design_json = tmp_path / "design.json"
    design_md = tmp_path / "design.md"
    assert design_json.exists()
    assert design_md.exists()
    design_data = json.loads(design_json.read_text())
    assert design_data["status"] == "fixture"
    assert "## Executive finding" in design_md.read_text()


def test_fixture_mode_report_is_idempotent_in_contents(tmp_path) -> None:
    assert main(["--mode", "fixture", "--output-dir", str(tmp_path)]) == 0
    data1 = json.loads((tmp_path / "slm160_spv_disposition_report.json").read_text())
    assert data1["status"] == "fixture"
