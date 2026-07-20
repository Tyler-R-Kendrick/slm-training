"""Regression tests for the SLM-137 intervention unification fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_intervention_unification_fixture import main


@pytest.fixture
def args(tmp_path: Path) -> list[str]:
    return [
        "--output-dir",
        str(tmp_path / "out"),
    ]


def test_plan_only_emits_skeleton(args: list[str], tmp_path: Path) -> None:
    assert main([*args, "--mode", "plan-only"]) == 0
    report_path = tmp_path / "out" / "ldi4_03_intervention_unification_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["matrix_set"] == "ldi4-03-intervention-unification"
    assert report["matrix_version"] == "ldi4-03-v1"
    assert report["status"] == "plan_only"
    assert report["manifests"] == []
    assert report["promotions"] == []
    assert report["version_stamp"]["stamp_schema"] == "version_stamp/v1"


def test_fixture_exercises_registry_and_writes_design_docs(
    args: list[str], tmp_path: Path
) -> None:
    assert main([*args, "--mode", "fixture"]) == 0
    report_path = tmp_path / "out" / "ldi4_03_intervention_unification_report.json"
    markdown_path = tmp_path / "out" / "ldi4_03_intervention_unification_report.md"
    assert report_path.is_file()
    assert markdown_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "wiring_only"
    assert report["claim_class"] == "wiring"
    assert set(report["kinds"]) == {
        "causal_peft",
        "twotower_delta",
        "reft",
        "sae_diagnostic",
    }
    assert report["one_active_asserted"] is True
    assert report["lineage_cycle"] is False
    assert report["closeout_index"]["artifact_count"] == 4
    assert report["closeout_index"]["best_deployable"] == "peft-1"
    assert report["version_stamp"]["stamp_schema"] == "version_stamp/v1"
    assert report["version_stamp"]["components"]["harness_core.lineage.interventions"] == "v1"

    promotions = report["promotions"]
    peft_promotions = [p for p in promotions if p["intervention_id"] == "peft-1"]
    assert all(p["ok"] for p in peft_promotions)

    delta_promotions = [p for p in promotions if p["intervention_id"] == "delta-1"]
    assert delta_promotions and not delta_promotions[-1]["ok"]
    assert "protected ship gate failed" in delta_promotions[-1]["failures"][0]

    sae_blocked = [p for p in promotions if p["intervention_id"] == "sae-1"][-1]
    assert sae_blocked["to"] == "promoted"
    assert not sae_blocked["ok"]


@pytest.mark.parametrize("mode", ["plan-only", "fixture"])
def test_markdown_summary_contains_caveat(mode: str, args: list[str], tmp_path: Path) -> None:
    assert main([*args, "--mode", mode]) == 0
    markdown = (tmp_path / "out" / "ldi4_03_intervention_unification_report.md").read_text(
        encoding="utf-8"
    )
    assert "SLM-137" in markdown
    assert "wiring" in markdown.lower()
