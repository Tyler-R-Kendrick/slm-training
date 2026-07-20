"""Tests for SLM-160 (SPV4-02) causal architecture disposition harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.harnesses.experiments.slm160_spv_disposition import (
    Disposition,
    SPVDispositionReport,
    SPVMechanismDisposition,
    build_default_dispositions,
    load_evidence_claim_class,
    render_markdown,
    run_disposition_audit,
)


@pytest.fixture
def repo_root() -> Path:
    # tests/ lives at the repository root, one level shallower than src/.
    return Path(__file__).resolve().parents[3]


def test_disposition_enum_values() -> None:
    values = {d.value for d in Disposition}
    assert values == {
        "adopt_primary",
        "adopt_optional",
        "retain_diagnostic",
        "revise_and_retest",
        "reject",
        "blocked",
        "inconclusive",
    }


def test_default_dispositions_are_produced(repo_root: Path) -> None:
    dispositions = build_default_dispositions(repo_root=repo_root)
    ids = {d.mechanism_id for d in dispositions}
    expected = {
        "semantic_plan_v1_ir",
        "gold_oracle_factor_heads",
        "plan_seed_builder_soft_restrictions",
        "x22_seed_retrieval_conflict_repair",
        "ar_legal_action_scorer",
        "global_semantic_critic",
        "hard_valid_contrasts",
        "dense_legal_set_distillation",
        "semantic_repair",
        "plan_refinement_slm156",
        "mixer_slm158",
        "flow_consistency_slm157",
        "multi_pack_graphql",
        "multi_pack_second_pack",
        "prompt_plan_soft_scoring_e575_e576_e579",
    }
    assert expected <= ids
    assert len(dispositions) == len(ids)


def test_report_round_trip_serialization(repo_root: Path) -> None:
    report = run_disposition_audit(repo_root=repo_root)
    data = report.to_dict()
    restored = SPVDispositionReport.from_dict(data)
    assert restored.schema == report.schema
    assert restored.matrix_set == report.matrix_set
    assert restored.status == report.status
    assert {m.mechanism_id for m in restored.mechanism_dispositions} == {
        m.mechanism_id for m in report.mechanism_dispositions
    }


def test_mechanism_round_trip_serialization() -> None:
    mech = SPVMechanismDisposition(
        mechanism_id="test_mech",
        issue_ids=("SLM-999",),
        evidence_paths=("docs/design/iter-test.json",),
        hypothesis="h",
        falsifier="f",
        disposition=Disposition.RETAIN_DIAGNOSTIC,
        rationale="r",
        next_action="a",
        default_state="off",
    )
    data = mech.to_dict()
    assert data["disposition"] == "retain_diagnostic"
    assert data["issue_ids"] == ["SLM-999"]
    restored = SPVMechanismDisposition.from_dict(data)
    assert restored == mech


def test_missing_evidence_marks_blocked_or_inconclusive(repo_root: Path) -> None:
    dispositions = build_default_dispositions(repo_root=repo_root)
    flow = next(d for d in dispositions if d.mechanism_id == "flow_consistency_slm157")
    assert flow.disposition == Disposition.BLOCKED
    assert "missing" in flow.rationale.lower() or "dependency" in flow.rationale.lower()


def test_render_markdown_contains_expected_sections(repo_root: Path) -> None:
    report = run_disposition_audit(repo_root=repo_root)
    md = render_markdown(report)
    for section in (
        "# SLM-160 (SPV4-02): Causal architecture disposition report",
        "## Executive finding",
        "## Evidence chronology",
        "## Mechanism disposition table",
        "## Cross-pack summary",
        "## Canonical architecture recommendation",
        "## Rejected or blocked mechanisms",
        "## Reproducibility commands",
        "## Limitations",
    ):
        assert section in md
    assert "| semantic_plan_v1_ir |" in md
    assert "| flow_consistency_slm157 |" in md


def test_no_mechanism_receives_adopt_primary(repo_root: Path) -> None:
    dispositions = build_default_dispositions(repo_root=repo_root)
    assert all(d.disposition != Disposition.ADOPT_PRIMARY for d in dispositions)


def test_e575_e576_e579_not_promotable(repo_root: Path) -> None:
    dispositions = build_default_dispositions(repo_root=repo_root)
    mech = next(
        d
        for d in dispositions
        if d.mechanism_id == "prompt_plan_soft_scoring_e575_e576_e579"
    )
    assert mech.disposition == Disposition.RETAIN_DIAGNOSTIC
    assert mech.default_state == "off"
    assert "not promotable" in mech.rationale.lower()


def test_load_evidence_claim_class_reads_existing_doc(repo_root: Path) -> None:
    claim_class, status = load_evidence_claim_class(
        "docs/design/iter-slm159-cross-dsl-replication-20260720.json",
        repo_root=repo_root,
    )
    assert claim_class == "wiring"
    assert status is not None


def test_load_evidence_claim_class_graceful_on_missing(repo_root: Path) -> None:
    claim_class, status = load_evidence_claim_class(
        "docs/design/does-not-exist-99999999.json",
        repo_root=repo_root,
    )
    assert claim_class is None
    assert status is None


def test_fixture_audit_has_version_stamp(repo_root: Path) -> None:
    report = run_disposition_audit(repo_root=repo_root)
    assert report.version_stamp
    assert report.version_stamp.get("stamp_schema") == "version_stamp/v1"
    assert "harness.experiments" in report.version_stamp.get("components", {})
