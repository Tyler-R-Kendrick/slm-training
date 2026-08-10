"""Tests for SLM-491/492 supersession wiring in RSP-009 disposition."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.harnesses.experiments.rsp009_cross_experiment_disposition import (
    build_exp_sr_disposition_records,
    build_slm491_492_supersessions,
    run_rsp009_disposition_audit,
)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_supersessions_when_closeout_docs_exist(repo_root: Path) -> None:
    slm491 = repo_root / "docs/design/iter-slm491-exp-sr-3-external-blocked-closeout-20260810.json"
    slm492 = repo_root / "docs/design/iter-slm492-exp-sr-11-external-blocked-closeout-20260810.json"
    if not slm491.is_file() or not slm492.is_file():
        return

    records = build_exp_sr_disposition_records(repo_root=repo_root)
    extended, supersessions = build_slm491_492_supersessions(records, repo_root=repo_root)
    assert len(extended) == len(records) + 2
    assert len(supersessions) == 2
    reasons = {s.reason for s in supersessions}
    assert any("SLM-491" in r for r in reasons)
    assert any("SLM-492" in r for r in reasons)


def test_audit_integrity_with_closeouts(repo_root: Path) -> None:
    slm491 = repo_root / "docs/design/iter-slm491-exp-sr-3-external-blocked-closeout-20260810.json"
    if not slm491.is_file():
        return

    report = run_rsp009_disposition_audit(repo_root=repo_root)
    supersessions = report.mechanism_disposition_report.supersessions
    assert supersessions
    assert report.champion_pointer_ids == ()
