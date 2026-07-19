"""Regression tests for SLM-173 decision-difficulty extraction."""

from __future__ import annotations

import math

import pytest

from slm_training.dsl.analysis.arity import (
    DecisionDifficulty,
    ProgramDifficulty,
    aggregate_program_difficulties,
    decision_difficulty_from_trace,
)
from slm_training.harnesses.distill.grammar_trace import GrammarTraceRecorder


def _make_trace(
    *,
    legal_action_ids: list[str],
    logits: list[float] | None = None,
    selected: str | None = None,
    completion_support: int | None = None,
    example_id: str = "ex1",
    seed: int = 0,
) -> object:
    recorder = GrammarTraceRecorder(
        run_id="r1",
        checkpoint_id="ckpt",
        dataset_id="ds",
        example_id=example_id,
        seed=seed,
        capture_logits=logits is not None,
    )
    return recorder.record(
        state_fingerprint="fp_" + "_".join(legal_action_ids),
        legal_action_ids=legal_action_ids,
        selected_action_id=selected,
        logits_or_energies=logits,
        compiler_coverage="complete",
        completion_support_size_exact=completion_support,
    )


def test_decision_difficulty_from_trace_has_version_and_arity() -> None:
    trace = _make_trace(legal_action_ids=["a", "b", "c"], selected="b")
    assert trace is not None
    diff = decision_difficulty_from_trace(trace)
    assert isinstance(diff, DecisionDifficulty)
    assert diff.schema_version == "sde2-06.v1"
    assert diff.live_legal_action_count == 3
    assert abs(diff.log2_live_legal_action_count - math.log2(3)) < 1e-9
    assert diff.source_hash is not None
    assert len(diff.source_hash) == 16


def test_decision_difficulty_reuses_trace_entropy_and_margin() -> None:
    trace = _make_trace(
        legal_action_ids=["a", "b"],
        logits=[1.0, 3.0],
        selected="b",
        completion_support=7,
    )
    assert trace is not None
    diff = decision_difficulty_from_trace(trace)
    assert diff.posterior_entropy_bits == pytest.approx(trace.posterior_entropy_bits)
    assert diff.top1_margin == pytest.approx(trace.top1_margin)  # type: ignore[arg-type]
    assert diff.completion_support_size_exact == 7


def test_decision_difficulty_honors_missing_logits() -> None:
    trace = _make_trace(legal_action_ids=["a", "b", "c"], selected="a")
    assert trace is not None
    diff = decision_difficulty_from_trace(trace)
    assert diff.posterior_entropy_bits is None
    assert diff.top1_margin is None


def test_decision_difficulty_arity_one_has_zero_log() -> None:
    trace = _make_trace(legal_action_ids=["only"], selected="only")
    assert trace is not None
    diff = decision_difficulty_from_trace(trace)
    assert diff.live_legal_action_count == 1
    assert diff.log2_live_legal_action_count == 0.0


def test_aggregate_program_difficulty_empty() -> None:
    prog = aggregate_program_difficulties([], example_id="empty")
    assert prog.example_id == "empty"
    assert prog.decision_count == 0
    assert prog.mean_entropy_bits == 0.0
    assert prog.max_entropy_bits == 0.0
    assert prog.mean_arity == 0.0
    assert prog.max_arity == 0


def test_aggregate_program_difficulty_stable() -> None:
    traces = [
        _make_trace(
            legal_action_ids=["a", "b"],
            logits=[0.0, 0.0],
            selected="a",
            example_id="prog1",
        ),
        _make_trace(
            legal_action_ids=["a", "b", "c", "d"],
            logits=[0.0, 0.0, 0.0, 0.0],
            selected="a",
            example_id="prog1",
        ),
    ]
    difficulties = [decision_difficulty_from_trace(t) for t in traces if t is not None]
    prog = aggregate_program_difficulties(difficulties, example_id="prog1")
    assert prog.example_id == "prog1"
    assert prog.decision_count == 2
    assert prog.mean_entropy_bits == pytest.approx(1.5)
    assert prog.max_entropy_bits == pytest.approx(2.0)
    assert prog.mean_arity == pytest.approx(3.0)
    assert prog.max_arity == 4
    assert prog.schema_version == "sde2-06.v1"


def test_decision_difficulty_round_trip_dict() -> None:
    trace = _make_trace(
        legal_action_ids=["x", "y"],
        logits=[1.0, 2.0],
        selected="y",
        completion_support=3,
    )
    assert trace is not None
    diff = decision_difficulty_from_trace(trace, quotient_color=5)
    data = diff.to_dict()
    assert data["schema_version"] == "sde2-06.v1"
    assert data["live_legal_action_count"] == 2
    assert data["quotient_color"] == 5
    assert data["completion_support_size_exact"] == 3


def test_program_difficulty_round_trip_dict() -> None:
    prog = ProgramDifficulty(
        example_id="p",
        decision_count=3,
        mean_entropy_bits=1.5,
        max_entropy_bits=2.0,
        mean_arity=2.5,
        max_arity=4,
    )
    data = prog.to_dict()
    assert data["schema_version"] == "sde2-06.v1"
    assert data["example_id"] == "p"
    assert data["mean_arity"] == 2.5
