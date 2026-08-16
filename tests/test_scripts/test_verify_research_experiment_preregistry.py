"""RESEARCH-02 experiment preregistry must fail on real regressions (SLM-533)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import verify_research_experiment_preregistry as verifier
from slm_training.research_preregistry import (
    ResearchPreregistryError,
    assert_execution_allowed,
    compute_knob_hypothesis_signature,
    experiment_by_key,
    reject_equivalent_rerun,
    validate_entry,
    validate_registry_document,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_passes_on_the_real_repository():
    result = verifier.certify()
    assert result["schema"] == "research_experiment_preregistry/v1"
    assert result["issue_key"] == "RESEARCH-02"
    assert result["linear_id"] == "SLM-533"
    assert result["experiment_count"] >= 18
    assert "RESEARCH-03" in result["experiment_keys"]
    assert "RESEARCH-20" in result["experiment_keys"]
    assert result["default_off"] is True
    assert result["research_only"] is True


def test_loader_entry_authority_and_signature():
    row = experiment_by_key("RESEARCH-03")
    assert row["default_off"] is True
    assert row["research_only"] is True
    assert row["matched_control"]
    assert row["falsifier"]
    assert row["resource_cap"]["max_run_minutes"] >= 0
    assert row["cited_lineage"]
    sig = compute_knob_hypothesis_signature(
        knobs=row["knobs"], hypothesis=row["hypothesis"]
    )
    assert sig == row["knob_hypothesis_signature"]
    validate_entry(row)


def test_missing_falsifier_fails_closed():
    row = dict(experiment_by_key("RESEARCH-03"))
    row["falsifier"] = ""
    with pytest.raises(ResearchPreregistryError, match="falsifier"):
        validate_entry(row, check_citations=False)


def test_missing_control_fails_closed():
    row = dict(experiment_by_key("RESEARCH-03"))
    row["matched_control"] = "  "
    with pytest.raises(ResearchPreregistryError, match="matched_control"):
        validate_entry(row, check_citations=False)


def test_missing_resource_cap_fails_closed():
    row = dict(experiment_by_key("RESEARCH-03"))
    row["resource_cap"] = {"max_run_minutes": 1, "notes": "x"}
    with pytest.raises(ResearchPreregistryError, match="max_gpu_hours"):
        validate_entry(row, check_citations=False)


def test_missing_citation_fails_closed():
    row = dict(experiment_by_key("RESEARCH-03"))
    row["cited_lineage"] = []
    with pytest.raises(ResearchPreregistryError, match="cited_lineage"):
        validate_entry(row, check_citations=False)


def test_missing_blocker_fails_closed():
    row = dict(experiment_by_key("RESEARCH-03"))
    row["blockers"] = []
    with pytest.raises(ResearchPreregistryError, match="blockers"):
        validate_entry(row, check_citations=False)


def test_execution_requires_campaign_lock_and_blockers():
    row = dict(experiment_by_key("RESEARCH-05"))
    row["campaign_lock_sha256"] = None
    with pytest.raises(ResearchPreregistryError, match="campaign_lock"):
        assert_execution_allowed(row)

    row["campaign_lock_sha256"] = "b" * 64
    blockers = [dict(b) for b in row["blockers"]]
    blockers[0]["status"] = "open"
    row["blockers"] = blockers
    with pytest.raises(ResearchPreregistryError, match="blockers incomplete"):
        assert_execution_allowed(row)


def test_historical_rejected_blocks_equivalent_rerun():
    base = experiment_by_key("RESEARCH-08")
    terminal = dict(base)
    terminal["disposition"] = "rejected"
    twin = dict(base)
    twin["experiment_key"] = "RESEARCH-09"
    twin["disposition"] = "registered"
    twin["knobs"] = dict(base["knobs"])
    twin["hypothesis"] = base["hypothesis"]
    twin["knob_hypothesis_signature"] = base["knob_hypothesis_signature"]
    with pytest.raises(ResearchPreregistryError, match="equivalent"):
        reject_equivalent_rerun(candidate=twin, historical=[terminal])


def test_reopen_with_evidence_allowed():
    base = experiment_by_key("RESEARCH-08")
    terminal = dict(base)
    terminal["disposition"] = "completed"
    reopened = dict(base)
    reopened["disposition"] = "reopened"
    reopened["reopen_evidence_path"] = "docs/design/iter-revmath-research-08-reopen.md"
    reopened["knob_hypothesis_signature"] = base["knob_hypothesis_signature"]
    reject_equivalent_rerun(candidate=reopened, historical=[terminal])


def test_duplicate_signature_in_document_rejected(tmp_path):
    doc = json.loads(
        (REPO_ROOT / verifier.REGISTRY_RELATIVE).read_text(encoding="utf-8")
    )
    a = dict(doc["experiments"][0])
    b = dict(doc["experiments"][1])
    b["knobs"] = dict(a["knobs"])
    b["hypothesis"] = a["hypothesis"]
    b["knob_hypothesis_signature"] = a["knob_hypothesis_signature"]
    doc["experiments"] = [a, b]
    with pytest.raises(
        ResearchPreregistryError,
        match=r"duplicate knob/hypothesis|reject equivalent|owned by terminal",
    ):
        validate_registry_document(doc, root=REPO_ROOT, check_citations=False)


def test_non_default_off_rejected():
    row = dict(experiment_by_key("RESEARCH-03"))
    row["default_off"] = False
    with pytest.raises(ResearchPreregistryError, match="default_off"):
        validate_entry(row, check_citations=False)
