"""Tests for the SLM-238 gradient-accumulation equivalence sweep harness."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.harnesses.experiments.slm238_gradaccum_equivalence import (
    EXPERIMENT_ID,
    MATRIX_SET,
    GradAccumEquivalenceReport,
    render_markdown,
    run_gradaccum_equivalence_sweep,
)


def test_sweep_runs_both_arms_for_every_seed() -> None:
    report = run_gradaccum_equivalence_sweep(steps=3, n_records=4, seeds=(0, 1))

    assert report.matrix_set == MATRIX_SET
    assert report.experiment_id == EXPERIMENT_ID
    assert report.claim_class == "wiring"
    assert report.status == "fixture"
    assert len(report.comparisons) == 2

    for comparison in report.comparisons:
        assert comparison.direct.steps_completed == 3
        assert comparison.accum.steps_completed == 3
        assert comparison.direct.arm == "direct"
        assert comparison.accum.arm == "accum"
        assert comparison.direct.grad_accum == 1
        assert comparison.accum.grad_accum == 2
        assert comparison.direct.effective_batch_size == comparison.accum.effective_batch_size
        assert comparison.direct.last_loss is not None
        assert comparison.accum.last_loss is not None
        assert comparison.winner in {"accum", "direct", "tie", "unstable"}

    assert (
        report.accum_wins + report.direct_wins + report.ties + report.unstable_seeds == 2
    )
    assert report.disposition in {
        "unstable",
        "metadata_gap",
        "no_signal",
        "close_approximation_confirmed",
        "consistent_direction_but_diverges",
        "inconsistent_and_diverges",
    }
    assert report.version_stamp.get("stamp_schema") == "version_stamp/v1"


def test_effective_batch_size_matches_configured_accumulation() -> None:
    # accel.effective_batch_size / accel.grad_accum are the real telemetry
    # fields the July-15 probes said made accumulation "directly auditable" --
    # confirm the contract, not just the loss trajectory.
    report = run_gradaccum_equivalence_sweep(steps=2, n_records=4, seeds=(0,))
    comparison = report.comparisons[0]
    assert comparison.direct.batch_size == 4
    assert comparison.accum.batch_size == 2
    assert comparison.accum.grad_accum == 2
    assert comparison.direct.effective_batch_size == 4
    assert comparison.accum.effective_batch_size == 4
    assert comparison.direct.metadata_ok
    assert comparison.accum.metadata_ok
    assert report.all_metadata_ok


def test_rejects_odd_record_count() -> None:
    with pytest.raises(ValueError):
        run_gradaccum_equivalence_sweep(steps=2, n_records=3, seeds=(0,))


def test_report_roundtrips_through_dict() -> None:
    report = run_gradaccum_equivalence_sweep(steps=2, n_records=4, seeds=(0,))
    payload = report.to_dict()
    restored = GradAccumEquivalenceReport.from_dict(payload)
    assert restored.to_dict() == payload


def test_render_markdown_includes_disposition_and_table() -> None:
    report = run_gradaccum_equivalence_sweep(steps=2, n_records=4, seeds=(0, 1))
    text = render_markdown(report)
    assert report.disposition in text
    assert (
        "| seed | direct last_loss | accum last_loss | delta (accum-direct) | rel diff | close? | winner |"
        in text
    )
    assert "No-go for promotion" in text
