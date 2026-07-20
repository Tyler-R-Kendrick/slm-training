"""Tests for slm_training.harnesses.experiments.efs4_04_causal_synthesis (SLM-140)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from slm_training.harnesses.experiments.efs4_04_causal_synthesis import (
    CampaignHypothesisSpec,
    CampaignManifestV1,
    EvidenceFirstSemanticSynthesisV1,
    build_default_campaign_manifest,
    load_manifest,
    load_result_manifests,
    render_dot,
    render_markdown,
    render_mermaid,
    save_manifest,
    save_synthesis,
    synthesize_campaign,
    validate_synthesis,
)


_REQUIRED_ISSUES = {
    "SLM-103",
    "SLM-104",
    "SLM-105",
    "SLM-106",
    "SLM-107",
    "SLM-108",
    "SLM-109",
    "SLM-110",
    "SLM-111",
    "SLM-112",
    "SLM-113",
    "SLM-115",
    "SLM-118",
    "SLM-120",
    "SLM-124",
    "SLM-127",
    "SLM-130",
    "SLM-133",
    "SLM-135",
    "SLM-138",
    "SLM-139",
}


def test_default_manifest_has_all_required_issues() -> None:
    manifest = build_default_campaign_manifest()
    issues = {h.linear_issue for h in manifest.hypotheses}
    assert _REQUIRED_ISSUES <= issues
    assert len(manifest.hypotheses) == len({h.hypothesis_id for h in manifest.hypotheses})
    assert len(manifest.hypotheses) == len(issues)


def test_manifest_rejects_duplicate_hypothesis_ids() -> None:
    base = build_default_campaign_manifest()
    first = base.hypotheses[0]
    dup = CampaignHypothesisSpec(
        hypothesis_id=first.hypothesis_id,
        linear_issue="SLM-999",
        milestone="dup",
        claim="dup",
        falsifier="dup",
    )
    with pytest.raises(ValidationError):
        CampaignManifestV1(hypotheses=base.hypotheses + (dup,))


def test_manifest_rejects_duplicate_linear_issues() -> None:
    base = build_default_campaign_manifest()
    first = base.hypotheses[0]
    dup = CampaignHypothesisSpec(
        hypothesis_id="unique-hypothesis-id",
        linear_issue=first.linear_issue,
        milestone="dup",
        claim="dup",
        falsifier="dup",
    )
    with pytest.raises(ValidationError):
        CampaignManifestV1(hypotheses=base.hypotheses + (dup,))


def test_manifest_save_and_load_roundtrip(tmp_path: Path) -> None:
    manifest = build_default_campaign_manifest()
    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)
    loaded = load_manifest(path)
    assert loaded.campaign_id == manifest.campaign_id
    assert len(loaded.hypotheses) == len(manifest.hypotheses)
    assert loaded.hypotheses[0].hypothesis_id == manifest.hypotheses[0].hypothesis_id


def test_load_result_manifests_skips_bad_files(tmp_path: Path) -> None:
    (tmp_path / "iter-good.json").write_text(json.dumps({"rows": [{"x": 1}]}), encoding="utf-8")
    (tmp_path / "iter-bad.json").write_text("not json", encoding="utf-8")
    (tmp_path / "iter-list.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    manifests = load_result_manifests(tmp_path)
    assert len(manifests) == 1
    assert manifests[0].source_path == "iter-good.json"
    assert manifests[0].rows == ({"x": 1},)


def test_synthesize_with_no_results_is_all_missing() -> None:
    manifest = build_default_campaign_manifest()
    synthesis = synthesize_campaign(manifest, (), generation_command="test")
    assert synthesis.campaign_id == manifest.campaign_id
    assert len(synthesis.hypotheses) == len(manifest.hypotheses)
    assert all(syn.state == "MISSING" for syn in synthesis.hypotheses)
    assert synthesis.causal_diagnosis is not None
    assert synthesis.causal_diagnosis.primary == "insufficient_valid_evidence"
    assert synthesis.champion_decision == "no_promotion"


def _write_result(
    directory: Path, filename: str, verdict: str | None = None, status: str = "complete"
) -> None:
    payload: dict = {"status": status, "schema_version": "test/v1"}
    if verdict is not None:
        payload["verdict"] = verdict
    (directory / filename).write_text(json.dumps(payload), encoding="utf-8")


def test_synthesize_respects_verdict_positive(tmp_path: Path) -> None:
    _write_result(tmp_path, "iter-efs0-01-checkpoint-provenance-fixture.json", verdict="POSITIVE")
    results = load_result_manifests(tmp_path)
    manifest = build_default_campaign_manifest()
    synthesis = synthesize_campaign(manifest, results, generation_command="test")
    syn = next(
        s for s in synthesis.hypotheses if s.hypothesis_id == "efs0-01-checkpoint-provenance"
    )
    assert syn.state == "POSITIVE"
    assert syn.result_refs


def test_synthesize_maps_complete_to_inconclusive(tmp_path: Path) -> None:
    _write_result(tmp_path, "iter-efs-decode-invariance-fixture.json", status="complete")
    results = load_result_manifests(tmp_path)
    manifest = build_default_campaign_manifest()
    synthesis = synthesize_campaign(manifest, results)
    syn = next(s for s in synthesis.hypotheses if s.hypothesis_id == "efs0-02-decode-invariance")
    assert syn.state == "INCONCLUSIVE"


def test_synthesize_detects_contradictory_state(tmp_path: Path) -> None:
    _write_result(tmp_path, "iter-contradictory-fixture.json", verdict="INCONCLUSIVE")
    results = load_result_manifests(tmp_path)
    hyp = CampaignHypothesisSpec(
        hypothesis_id="contradictory-hyp",
        linear_issue="SLM-999",
        milestone="test",
        claim="claim",
        falsifier="falsifier",
        expected_result_refs=("iter-contradictory-fixture.json",),
        allowed_decisions=("POSITIVE", "NEGATIVE"),
    )
    manifest = CampaignManifestV1(hypotheses=(hyp,))
    synthesis = synthesize_campaign(manifest, results)
    assert synthesis.hypotheses[0].state == "CONTRADICTORY"
    errors = validate_synthesis(synthesis)
    assert any("contradictory" in err.lower() for err in errors)


def test_architecture_dispositions_default_to_not_run_or_safety_only() -> None:
    manifest = build_default_campaign_manifest()
    synthesis = synthesize_campaign(manifest, ())
    assert len(synthesis.architecture_dispositions) > 0
    for disp in synthesis.architecture_dispositions:
        assert disp.disposition in {
            "NOT_RUN_BY_GATE",
            "INCONCLUSIVE",
            "CONDITIONAL_RESEARCH",
            "ADOPT_AS_SAFETY_ONLY",
        }
        if "exact closure" in disp.item or "trailing" in disp.item or "verifier cascade" in disp.item:
            assert disp.disposition == "ADOPT_AS_SAFETY_ONLY"


def test_validate_synthesis_reports_missing_required_issues() -> None:
    manifest = CampaignManifestV1(hypotheses=())
    synthesis = synthesize_campaign(manifest, ())
    errors = validate_synthesis(synthesis)
    assert any("missing required" in err.lower() for err in errors)


def test_render_markdown_includes_all_sections() -> None:
    synthesis = synthesize_campaign(build_default_campaign_manifest(), ())
    md = render_markdown(synthesis)
    assert "# EFS4-04" in md
    assert "## 1. Executive verdict" in md
    assert "## 6. Architecture disposition table" in md
    assert "## 10. Next three experiments" in md
    assert "insufficient valid evidence" in md
    assert "no_promotion" in md


def test_render_mermaid_and_dot_are_deterministic() -> None:
    synthesis = synthesize_campaign(build_default_campaign_manifest(), ())
    mmd1 = render_mermaid(synthesis.evidence_graph)
    mmd2 = render_mermaid(synthesis.evidence_graph)
    dot1 = render_dot(synthesis.evidence_graph)
    dot2 = render_dot(synthesis.evidence_graph)
    assert mmd1 == mmd2
    assert dot1 == dot2
    assert "flowchart TD" in mmd1
    assert "digraph evidence_graph" in dot1


def test_synthesis_save_and_load_roundtrip(tmp_path: Path) -> None:
    synthesis = synthesize_campaign(build_default_campaign_manifest(), ())
    path = tmp_path / "synthesis.json"
    save_synthesis(synthesis, path)
    loaded = EvidenceFirstSemanticSynthesisV1.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    assert loaded.campaign_id == synthesis.campaign_id
    assert loaded.manifest_hash == synthesis.manifest_hash
    assert len(loaded.hypotheses) == len(synthesis.hypotheses)
