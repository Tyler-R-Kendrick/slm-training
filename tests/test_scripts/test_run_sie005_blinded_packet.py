"""Tests for the SLM-478 (SIE-005) blinded packet CLI."""

from __future__ import annotations

import json

from scripts.run_sie005_blinded_packet import main


def test_plan_only_mode_writes_manifest(tmp_path) -> None:
    assert main(["--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "sie005_blinded_packet_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "plan_only"
    assert data["claim_class"] == "fixture"
    assert data["schema"] == "Slm478Sie005BlindedPacketReportV1"
    assert data["related_catalogue_id"] == "exp-sr-3"
    assert data["version_stamp"]
    assert data["steps"]


def test_fixture_mode_writes_run_json(tmp_path) -> None:
    assert main(["--mode", "fixture", "--output-dir", str(tmp_path), "--seed", "0"]) == 0
    run_json = tmp_path / "sie005_blinded_packet_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "fixture"
    assert data["claim_class"] == "fixture"
    assert data["blinding_ok"] is True
    assert data["training_excluded"] is True
    assert data["independence_ok"] is True
    assert data["pair_n"] == 8
    assert (tmp_path / "annotation_packet" / "blind_pairs.jsonl").exists()
    assert (tmp_path / "private" / "unblinding_key.jsonl").exists()
