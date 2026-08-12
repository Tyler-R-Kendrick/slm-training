"""RESEARCH-19 / SLM-558 backend tests."""

from pathlib import Path

from slm_training.formal import lean_rademacher_bound as lrb

ROOT = Path(__file__).resolve().parents[2]


def test_paired_report_accepts_toy_certificate() -> None:
    report = lrb.paired_bound_report(root=ROOT)
    assert report["decision"] == "accept"
    assert report["lean_checked_toy_bound_certificate_validity"] == 1.0


def test_vacuous_probe_rejected() -> None:
    cases = lrb.default_cases(root=ROOT)
    vac = next(c for c in cases if c.case_id == "vacuous_class_eval")
    row = lrb.evaluate_treatment(vac)
    assert row["certificate_valid"] is False
    assert row["bound_vacuous"] is True
