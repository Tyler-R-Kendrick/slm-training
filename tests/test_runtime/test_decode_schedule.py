"""Decode invariant I4: symbol tables schedule compute.

The planner is pure and observational until its lever is on. These tests pin
both halves: with ``mode="off"`` nothing about the legacy forward budget moves,
and with the lever on the window is placed by the grammar, not by the canvas.
"""

from __future__ import annotations

import pytest

from slm_training.models.decode_stats import DecodeStats
from slm_training.runtime.decode_schedule import (
    PrefillPlanV1,
    next_grammar_checkpoint,
    plan_prefill,
    record_plan,
    schedule_backend,
)


def test_off_mode_reproduces_the_legacy_budget() -> None:
    plan = plan_prefill(
        position=4,
        canvas=64,
        active_rows=[0, 1, 2],
        committed_rows=[],
        mode="off",
        legacy_lookahead=16,
    )
    assert plan.rows == (0, 1, 2)
    assert plan.prefill_end == 20
    assert plan.reason == "legacy_budget"
    assert plan.tokens_saved == 0


def test_off_mode_without_a_lookahead_reads_the_whole_canvas() -> None:
    plan = plan_prefill(
        position=4, canvas=64, active_rows=[0], committed_rows=[], mode="off"
    )
    assert plan.prefill_end == 64


def test_committed_rows_are_never_forwarded_in_any_mode() -> None:
    """Dropping a bypassed row follows from I2, not from the scheduling lever."""
    for mode in ("off", "checkpoint"):
        plan = plan_prefill(
            position=3,
            canvas=32,
            active_rows=[0, 1, 2, 3],
            committed_rows=[1, 3],
            mode=mode,
            legacy_lookahead=8,
        )
        assert plan.rows == (0, 2)
        assert plan.rows_skipped == 2


def test_a_fully_committed_step_plans_no_forward() -> None:
    plan = plan_prefill(
        position=3,
        canvas=32,
        active_rows=[0, 1],
        committed_rows=[0, 1],
        mode="checkpoint",
    )
    assert not plan.needs_forward
    assert plan.reason == "fully_committed"


def test_checkpoint_mode_stops_the_window_at_a_forced_lexeme() -> None:
    plan = plan_prefill(
        position=2,
        canvas=64,
        active_rows=[0],
        committed_rows=[],
        mode="checkpoint",
        legacy_lookahead=32,
        max_lookahead=32,
        backend="cpu",
        forced_run_lengths={6: 3},
    )
    assert plan.checkpoint == 6
    assert plan.prefill_end == 6
    assert plan.reason == "grammar_checkpoint"
    assert plan.tokens_saved == 28


def test_checkpoint_mode_without_a_checkpoint_uses_the_device_window() -> None:
    plan = plan_prefill(
        position=2,
        canvas=64,
        active_rows=[0],
        committed_rows=[],
        mode="checkpoint",
        legacy_lookahead=8,
        max_lookahead=12,
        backend="cpu",
    )
    assert plan.checkpoint is None
    assert plan.prefill_end == 14
    assert plan.reason == "device_window"


def test_accelerator_window_is_never_narrower_than_the_legacy_one() -> None:
    """A launch is the expensive part on cuda/npu — do not shrink below legacy."""
    plan = plan_prefill(
        position=0,
        canvas=128,
        active_rows=[0],
        committed_rows=[],
        mode="checkpoint",
        legacy_lookahead=48,
        max_lookahead=4,
        backend="cuda",
    )
    assert plan.prefill_end >= 48


def test_prefill_end_always_advances_past_the_current_position() -> None:
    plan = plan_prefill(
        position=10,
        canvas=12,
        active_rows=[0],
        committed_rows=[],
        mode="checkpoint",
        max_lookahead=1,
        forced_run_lengths={10: 4},
    )
    assert plan.prefill_end > 10
    assert plan.prefill_end <= 12


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="prefill_schedule"):
        plan_prefill(
            position=0,
            canvas=8,
            active_rows=[0],
            committed_rows=[],
            mode="greedy",
        )


def test_next_grammar_checkpoint_picks_the_nearest_forced_position() -> None:
    assert next_grammar_checkpoint(2, 32, {9: 1, 5: 2, 1: 3}) == 5
    assert next_grammar_checkpoint(2, 4, {9: 1}) is None
    assert next_grammar_checkpoint(2, 32, {5: 0}) is None
    assert next_grammar_checkpoint(2, 32, None) is None


def test_record_plan_folds_counters_into_decode_stats() -> None:
    stats = DecodeStats()
    record_plan(
        stats,
        PrefillPlanV1(
            rows=(0,),
            prefill_end=6,
            checkpoint=6,
            backend="cpu",
            reason="grammar_checkpoint",
            rows_skipped=2,
            tokens_saved=10,
        ),
    )
    assert stats.scheduled_prefills == 1
    assert stats.scheduled_rows_skipped == 2
    assert stats.scheduled_prefill_tokens_saved == 10
    assert stats.schedule_checkpoint_hits == 1


def test_record_plan_ignores_a_step_with_no_forward() -> None:
    stats = DecodeStats()
    record_plan(
        stats,
        PrefillPlanV1(
            rows=(), prefill_end=4, checkpoint=None, backend="cpu", reason="fully_committed"
        ),
    )
    assert stats.scheduled_prefills == 0


def test_schedule_backend_never_raises() -> None:
    assert schedule_backend() in {"cpu", "cuda", "npu", "dml"}
