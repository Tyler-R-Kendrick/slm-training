"""RESEARCH-14 / SLM-572: checker diversity backend tests."""

from __future__ import annotations

from slm_training.formal.checker_diversity_backend import (
    CONTROL_CHECKERS,
    TREATMENT_CHECKERS,
    paired_diversity_report,
)


def test_paired_diversity_report_accepts_fixture_corpus() -> None:
    report = paired_diversity_report()
    assert report["decision"] == "accept"
    assert report["unique_fault_class_gain_count"] >= 2
    assert report["false_alarm_rate"] == 0.0
    assert report["unique_defect_detection_gain_vs_single_checker"] > 0.0


def test_treatment_uses_distinct_trust_domains() -> None:
    report = paired_diversity_report()
    domains = report["treatment_arm"]["distinct_trust_domains"]
    assert len(domains) >= 3
    assert "lean4_kernel" in domains
    assert "python_encoding_bridge" in domains


def test_control_is_single_checker() -> None:
    report = paired_diversity_report()
    assert report["control_arm"]["checkers"] == list(CONTROL_CHECKERS)
    assert report["treatment_arm"]["checkers"] == list(TREATMENT_CHECKERS)
