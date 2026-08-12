"""RESEARCH-04 / SLM-550: RM Zoo benchmark backend tests."""

from __future__ import annotations

from pathlib import Path

from slm_training.formal import rm_zoo_benchmark as zoo

ROOT = Path(__file__).resolve().parents[2]


def test_corpus_loads() -> None:
    _, entries = zoo.load_corpus(root=ROOT)
    assert len(entries) >= 6
    zoo_ids = {e.principle_id for e in entries if e.arm == "treatment_zoo"}
    assert any(p.startswith("zoo:") for p in zoo_ids)


def test_paired_report_accepts_fixture_corpus() -> None:
    report = zoo.paired_benchmark_report(root=ROOT)
    assert report["decision"] == "accept"
    assert report["overclaim_rate"] == 0.0
    assert report["exact_classification_reversal_success_dependency_disjoint"] == 1.0


def test_eval_entries_are_dependency_disjoint() -> None:
    _, entries = zoo.load_corpus(root=ROOT)
    eval_families = {e.root_family_id for e in entries if e.split == "eval"}
    train_families = {e.root_family_id for e in entries if e.split == "train"}
    assert not eval_families & train_families
