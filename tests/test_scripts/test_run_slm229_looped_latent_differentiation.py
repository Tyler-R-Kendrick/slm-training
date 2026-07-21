"""Tests for the SLM-229 (RSC0-01) looped-latent differentiation audit CLI."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_slm229_looped_latent_differentiation import main


def test_plan_only_mode_writes_json(tmp_path) -> None:
    assert main(["--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm229_looped_latent_differentiation_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "plan_only"
    assert data["schema"] == "LoopedLatentDifferentiationV1"
    assert data["claim_class"] == "wiring"
    assert data["mechanism_comparison"]
    assert data["differentiators"]
    assert data["version_stamp"]


def test_fixture_mode_writes_json_and_docs(tmp_path) -> None:
    assert main(["--mode", "fixture", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm229_looped_latent_differentiation_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "fixture"
    assert data["schema"] == "LoopedLatentDifferentiationV1"
    assert data["claim_class"] == "wiring"
    assert data["version_stamp"]
    assert data["mechanism_comparison"]
    assert data["target_support_audit"]
    assert data["oracle_intervention_ceiling"]
    assert data["scale_regime_audit"]
    assert data["prior_art_audit"]
    assert data["differentiators"]
    assert data["verdict"] == "blocked_by_recurrence"
    assert data["minimal_contract"] is None

    root = Path(__file__).resolve().parents[2]
    design_json = root / "docs/design/iter-slm229-looped-latent-differentiation-20260721.json"
    design_md = root / "docs/design/iter-slm229-looped-latent-differentiation-20260721.md"
    assert design_json.exists()
    assert design_md.exists()
    design_data = json.loads(design_json.read_text())
    assert design_data["status"] == "fixture"
    assert "## 1. Mechanism comparison table" in design_md.read_text()


def test_fixture_mode_report_is_idempotent_in_contents(tmp_path) -> None:
    assert main(["--mode", "fixture", "--output-dir", str(tmp_path)]) == 0
    data1 = json.loads(
        (tmp_path / "slm229_looped_latent_differentiation_report.json").read_text()
    )
    assert data1["status"] == "fixture"
    assert data1["verdict"] == "blocked_by_recurrence"
