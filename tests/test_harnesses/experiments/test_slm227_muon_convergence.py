"""Tests for the SLM-227 Muon/AdamW convergence-direction sweep harness."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.harnesses.experiments.slm227_muon_convergence import (
    EXPERIMENT_ID,
    MATRIX_SET,
    MuonConvergenceReport,
    render_markdown,
    run_muon_convergence_sweep,
)


def test_sweep_runs_both_arms_for_every_seed() -> None:
    report = run_muon_convergence_sweep(steps=3, n_records=2, batch_size=2, seeds=(0, 1))

    assert report.matrix_set == MATRIX_SET
    assert report.experiment_id == EXPERIMENT_ID
    assert report.claim_class == "wiring"
    assert report.status == "fixture"
    assert len(report.comparisons) == 2

    for comparison in report.comparisons:
        assert comparison.adamw.steps_completed == 3
        assert comparison.muon.steps_completed == 3
        assert comparison.adamw.optimizer_name == "adamw"
        assert comparison.muon.optimizer_name == "muon_hybrid"
        assert comparison.adamw.last_loss is not None
        assert comparison.muon.last_loss is not None
        assert comparison.winner in {"muon", "adamw", "tie", "unstable"}

    assert report.muon_wins + report.adamw_wins + report.ties + report.unstable_seeds == 2
    assert report.disposition in {
        "unstable",
        "no_signal",
        "consistent_muon_lower_loss",
        "consistent_adamw_lower_loss",
        "majority_muon_lower_loss",
        "majority_adamw_lower_loss",
        "mixed_no_signal",
    }
    assert report.version_stamp.get("stamp_schema") == "version_stamp/v1"


def test_seeds_start_from_matched_initialization() -> None:
    # Same seed used for both arms within a comparison should give identical
    # first-step loss (same init, same data, same forward pass before any
    # optimizer-specific update has been applied).
    report = run_muon_convergence_sweep(steps=2, n_records=2, batch_size=2, seeds=(0,))
    comparison = report.comparisons[0]
    assert comparison.adamw.first_loss == pytest.approx(comparison.muon.first_loss, rel=1e-4)


def test_disposition_requires_unanimous_direction_for_consistent_label() -> None:
    report = run_muon_convergence_sweep(steps=3, n_records=2, batch_size=2, seeds=(0, 1, 2))
    if report.all_finite and report.muon_wins + report.adamw_wins == len(report.comparisons):
        if report.disposition == "consistent_muon_lower_loss":
            assert report.muon_wins == len(report.comparisons)
        elif report.disposition == "consistent_adamw_lower_loss":
            assert report.adamw_wins == len(report.comparisons)


def test_report_roundtrips_through_dict() -> None:
    report = run_muon_convergence_sweep(steps=2, n_records=2, batch_size=2, seeds=(0,))
    payload = report.to_dict()
    restored = MuonConvergenceReport.from_dict(payload)
    assert restored.to_dict() == payload


def test_render_markdown_includes_disposition_and_table() -> None:
    report = run_muon_convergence_sweep(steps=2, n_records=2, batch_size=2, seeds=(0, 1))
    text = render_markdown(report)
    assert report.disposition in text
    assert "| seed | adamw last_loss | muon last_loss | muon - adamw | winner |" in text
    assert "No-go for promotion" in text
