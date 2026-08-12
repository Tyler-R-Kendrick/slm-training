"""RESEARCH-16 / SLM-570: rational float transfer backend tests."""

from __future__ import annotations

from fractions import Fraction

from slm_training.formal.rational_float_transfer import (
    GateCaseV1,
    certified_gate_decision,
    paired_transfer_report,
)


def test_paired_transfer_report_accepts_fixture_corpus() -> None:
    report = paired_transfer_report()
    assert report["decision"] == "accept"
    assert report["transfer_certificate_coverage_on_selected_metric_gates"] == 1.0
    assert report["silent_cast_incidents"] == 0


def test_near_threshold_flip_returns_indeterminate() -> None:
    case = GateCaseV1(
        case_id="test",
        metric=Fraction(9999999999999999999, 10000000000000000000),
        threshold=Fraction(1, 1),
        operator="ge",
        float_flip_expected=True,
        notes="",
    )
    cert = certified_gate_decision(case)
    assert cert["exact_decision"] == "fail"
    assert cert["float_decision"] == "pass"
    assert cert["certified_decision"] == "indeterminate"
