"""RESEARCH-18 / SLM-551: Weihrauch multivalued classification backend tests."""

from __future__ import annotations

from pathlib import Path

from slm_training.formal import weihrauch_multivalued_classification as wmc

ROOT = Path(__file__).resolve().parents[2]


def test_corpus_loads() -> None:
    tasks = wmc.default_tasks(root=ROOT)
    assert len(tasks) >= 5


def test_paired_report_accepts_fixture_corpus() -> None:
    report = wmc.paired_classification_report(root=ROOT)
    assert report["decision"] == "accept"
    assert report["faithful_weihrauch_class_assignment_rate_on_frozen_task_set"] == 1.0
    assert report["control_misclassifies_oracle_tasks"] is True


def test_big_five_smuggling_rejected() -> None:
    tasks = wmc.default_tasks(root=ROOT)
    smuggle = next(t for t in tasks if t.rm_label_claim == "WKL0")
    row = wmc.classify_treatment(smuggle)
    assert row["faithful"] is False
    assert row["big_five_inflation"] is True
