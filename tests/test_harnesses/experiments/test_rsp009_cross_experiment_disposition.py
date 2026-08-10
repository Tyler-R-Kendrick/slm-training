"""Tests for SLM-490 (RSP-009) cross-experiment EXP-SR disposition harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.harnesses.experiments.exp_sr_catalogue import EXP_SR_FAMILY_SPECS
from slm_training.harnesses.experiments.mechanism_disposition_report import (
    Disposition,
    disposition_report_integrity_ok,
)
from slm_training.harnesses.experiments.rsp009_cross_experiment_disposition import (
    EXPERIMENT_ID,
    _EVIDENCE_BY_FAMILY,
    build_exp_sr_disposition_records,
    load_evidence_summary,
    render_markdown,
    run_rsp009_disposition_audit,
)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_every_catalogue_family_has_evidence_paths() -> None:
    for spec in EXP_SR_FAMILY_SPECS:
        assert spec.family_id in _EVIDENCE_BY_FAMILY
        assert _EVIDENCE_BY_FAMILY[spec.family_id]


def test_evidence_docs_exist(repo_root: Path) -> None:
    for paths in _EVIDENCE_BY_FAMILY.values():
        for rel in paths:
            assert (repo_root / rel).exists(), rel


def test_build_records_cover_all_families(repo_root: Path) -> None:
    records = build_exp_sr_disposition_records(repo_root=repo_root)
    ids = {r.mechanism_id for r in records}
    assert "exp-sr-1" in ids
    assert "exp-sr-12:openui" in ids
    assert "exp-sr-12:symbolic_regression" in ids
    assert len(records) == len(EXP_SR_FAMILY_SPECS) + 1  # exp-sr-12 split


def test_no_adopt_primary_or_optional(repo_root: Path) -> None:
    records = build_exp_sr_disposition_records(repo_root=repo_root)
    adopted = {
        r.mechanism_id
        for r in records
        if r.disposition in (Disposition.ADOPT_PRIMARY, Disposition.ADOPT_OPTIONAL)
    }
    assert not adopted


def test_fixture_evidence_never_default_on(repo_root: Path) -> None:
    records = build_exp_sr_disposition_records(repo_root=repo_root)
    for record in records:
        if record.evidence_class == "fixture":
            assert record.default_state != "on"


def test_exp_sr_11_is_blocked_not_reject(repo_root: Path) -> None:
    records = build_exp_sr_disposition_records(repo_root=repo_root)
    by_id = {r.mechanism_id: r for r in records}
    assert by_id["exp-sr-11"].disposition == Disposition.BLOCKED
    assert by_id["exp-sr-11"].evidence_class == "blocked"


def test_run_audit_integrity_and_empty_champions(repo_root: Path) -> None:
    report = run_rsp009_disposition_audit(repo_root=repo_root)
    assert report.experiment_id == EXPERIMENT_ID
    assert report.champion_pointer_ids == ()
    assert disposition_report_integrity_ok(report.mechanism_disposition_report)
    md = render_markdown(report)
    assert "EXP-SR-1..12" in md
    assert "exp-sr-11" in md


def test_load_evidence_summary_reads_claim_class(repo_root: Path) -> None:
    summary = load_evidence_summary(
        "docs/design/iter-slm476-sie-002-oracle-localization-20260810.json",
        repo_root,
    )
    assert summary["present"] is True
    assert summary["claim_class"] == "fixture"
