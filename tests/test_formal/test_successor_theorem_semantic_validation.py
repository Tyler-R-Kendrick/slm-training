"""RESEARCH-12 / SLM-567: successor-theorem semantic validation backend tests."""

from __future__ import annotations

from slm_training.formal.successor_theorem_semantic_validation import (
    default_records,
    paired_validation_report,
)


def test_corpus_has_six_records() -> None:
    assert len(default_records()) == 6


def test_paired_report_runs() -> None:
    report = paired_validation_report()
    assert report["schema"] == "successor_theorem_semantic_validation_report/v1"
    assert "decision" in report
    assert report["successor_incremental_catch_count"] >= 1
