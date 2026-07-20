"""Tests for the SLM-160 (SPV4-02) disposition audit CLI."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_slm160_spv_disposition import main


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

    # Design artifacts are written relative to the repository root.

    root = Path(__file__).resolve().parents[2]
    design_json = root / "docs/design/iter-slm160-spv-disposition-20260720.json"
    design_md = root / "docs/design/iter-slm160-spv-disposition-20260720.md"
    assert design_json.exists()
    assert design_md.exists()
    design_data = json.loads(design_json.read_text())
    assert design_data["status"] == "fixture"
    assert "## Executive finding" in design_md.read_text()


def test_fixture_mode_report_is_idempotent_in_contents(tmp_path) -> None:
    assert main(["--mode", "fixture", "--output-dir", str(tmp_path)]) == 0
    data1 = json.loads((tmp_path / "slm160_spv_disposition_report.json").read_text())
    assert data1["status"] == "fixture"
