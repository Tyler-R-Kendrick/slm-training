"""RESEARCH-17 / SLM-557: proof-mined quantitative bounds backend tests."""

from __future__ import annotations

from slm_training.formal.proof_mined_quantitative_bounds import (
    default_theorems,
    paired_mining_report,
)


def test_corpus_has_three_theorems() -> None:
    assert len(default_theorems()) == 3


def test_paired_report_runs() -> None:
    report = paired_mining_report()
    assert report["schema"] == "proof_mined_quantitative_bounds_report/v1"
    assert "decision" in report
    assert report["expected_actionable_count"] == 2
