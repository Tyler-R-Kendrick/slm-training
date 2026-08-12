"""RESEARCH-13 / SLM-568: dropped-assumption counterexample backend tests."""

from __future__ import annotations

from slm_training.formal.dropped_assumption_counterexample import (
    default_tasks,
    paired_counterexample_report,
)


def test_corpus_has_four_tasks() -> None:
    assert len(default_tasks()) == 4


def test_paired_report_runs() -> None:
    report = paired_counterexample_report()
    assert report["schema"] == "dropped_assumption_counterexample_report/v1"
    assert "decision" in report
    assert report["expected_independent_check_count"] == 1
