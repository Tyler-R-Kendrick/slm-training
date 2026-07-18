"""VSS1-04 (SLM-64): model-level solver trace + replay + decode-stats wiring.

Attaches a `DecodeTraceRecorder` to a solver-enabled TwoTower and drives one
`_solver_prune_forest` decision (fast — no full decode loop). Asserts the
recorder captures replayable solver-transition events + a bounded certificate
sidecar (`replay_violations` clean), the solver work-metric counters land in the
`DecodeStats` envelope (zero on the default path), and historical decode-only
traces still replay. Core validator semantics live in
tests/test_dsl/test_solver_replay.py.
"""

from __future__ import annotations

from slm_training.dsl.grammar.fastpath.compiler_draft import build_completion_forest
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.distill.trace_store import (
    DecodeTraceRecorder,
    replay_violations,
)
from slm_training.models.decode_stats import DecodeStats, aggregate_stats, collect_decode_stats
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel


def _solver_model():
    record = ExampleRecord(
        id="compiler",
        prompt="card",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")\n',
        placeholders=[":hero.title"],
        split="train",
        source="fixture",
    )
    config = TwoTowerConfig(
        context_backend="scratch",
        output_tokenizer="lexer",
        d_model=32,
        n_heads=2,
        context_layers=1,
        denoiser_layers=1,
        max_prompt_len=32,
        max_target_len=32,
        grammar_ltr_max_tokens=32,
        gen_steps=1,
        seed=0,
        verified_solver_decode=True,
        solver_max_nodes=4,
        solver_certificate_mode="full",
    )
    model = TwoTowerModel.from_records([record], config=config, device="cpu")
    model.eval()
    return model


def test_recorder_captures_replayable_solver_events():
    model = _solver_model()
    prefix = [model.tokenizer.bos_id]
    forest = build_completion_forest(model.tokenizer, prefix)
    recorder = DecodeTraceRecorder()
    model.trace_recorder = recorder

    model._solver_prune_forest(forest, prefix)

    from slm_training.dsl.solver.replay import SOLVER_EVENT_KINDS

    solver_events = [e for e in recorder.events if e.get("kind") in SOLVER_EVENT_KINDS]
    assert solver_events, "expected solver-transition events on the recorder"
    assert any(e["kind"] == "solver_state" for e in solver_events)
    assert any(e["kind"] == "solver_terminal" for e in solver_events)
    assert recorder.solver is not None
    assert recorder.solver["certificate_mode"] == "full"

    trace = recorder.finalize()
    assert trace["version"] == 3
    assert "solver" in trace
    # The captured solver stream replays with zero violations.
    assert replay_violations(trace) == []


def test_solver_counters_land_in_decode_stats_envelope():
    model = _solver_model()
    prefix = [model.tokenizer.bos_id]
    forest = build_completion_forest(model.tokenizer, prefix)

    with collect_decode_stats() as stats:
        model._solver_prune_forest(forest, prefix)

    assert stats.solver_enabled == 1
    assert stats.solver_terminal_status in {"unknown", "certified_unsat", "budget_exhausted"}
    # Solver time is tracked separately from denoiser/projection.
    assert stats.solver_ms >= 0.0
    # Counters surface (only) under metrics["decode_stats"] via aggregate_stats.
    agg = aggregate_stats([stats])
    assert "solver_enabled_sum" in agg
    assert "solver_support_queries_sum" in agg


def test_solver_counters_default_zero_when_disabled():
    stats = DecodeStats()
    assert stats.solver_enabled == 0
    assert stats.solver_ms == 0.0
    assert stats.solver_certified_removed == 0
    assert stats.solver_terminal_status == ""
    agg = aggregate_stats([stats])
    assert agg["solver_enabled_sum"] == 0.0
    assert agg["solver_certified_removed_mean"] == 0.0


def test_historical_decode_only_trace_still_replays():
    # A v2-style decode trace with no solver events / no solver block is
    # unaffected by the VSS1-04 replay extension.
    trace = {
        "version": 2,
        "meta": {},
        "steps": [
            {"step": 0, "canvas": [5, 0], "commits": [{"t": 0, "id": 5}], "remasks": []}
        ],
        "events": [],
        "final": {"canvas": [5, 2], "text": "x"},
    }
    assert replay_violations(trace) == []
