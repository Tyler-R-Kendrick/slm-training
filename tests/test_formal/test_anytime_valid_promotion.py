"""RESEARCH-15 / SLM-569: anytime-valid promotion backend tests."""

from __future__ import annotations

from slm_training.formal.anytime_valid_promotion import (
    default_scenarios,
    paired_calibration_report,
)


def test_corpus_has_null_and_planted_scenarios() -> None:
    scenarios = default_scenarios()
    kinds = {row.kind for row in scenarios}
    assert kinds == {"null", "planted"}
    assert len(scenarios) == 250


def test_paired_report_runs() -> None:
    report = paired_calibration_report()
    assert report["schema"] == "anytime_valid_promotion_report/v1"
    assert "decision" in report
    assert report["null_scenario_count"] == 200
