"""KERN-06 / SLM-524: event traces and named machine cost models."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.formal.bound_ast import evaluate_bound
from slm_training.formal.event_trace import (
    BOUND_AST_ID,
    DECODE_STATS_EVENT_OWNERS,
    DECODE_UNIT_WORK_MODEL,
    EVENT_TRACE_SCHEMA,
    NAMED_MODELS,
    NEURAL_FORWARD_ONLY_MODEL,
    SOLVER_QUERY_MODEL,
    TRAIN_STEP_MODEL,
    DecodeUnitWorkCounts,
    Event,
    EventKind,
    ThroughputAssumption,
    build_event_trace_evidence,
    export_four_axis_event_trace_evidence,
    frozen_fixture_rows,
    physical_latency_hypothesis_holds,
    project_decode_stats,
    resource_cost_model_id,
    trace_cost,
    trace_cost_append,
    unobserved_event_kinds_for_decode,
    wall_clock_upper_bound,
)
from slm_training.models.decode_stats import DecodeStats, build_decode_stats_record

_FIXTURES = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "slm_training"
    / "resources"
    / "formal"
    / "event_trace_fixtures.v1.json"
)


def test_named_models_registered() -> None:
    assert set(NAMED_MODELS) == {
        "DecodeUnitWorkModel",
        "SolverQueryModel",
        "NeuralForwardOnlyModel",
        "TrainStepModel",
    }
    assert DECODE_UNIT_WORK_MODEL.version == 1
    assert resource_cost_model_id(DECODE_UNIT_WORK_MODEL) == "DecodeUnitWorkModel.v1"


def test_trace_cost_composition_and_nonneg() -> None:
    left = (Event(EventKind.NEURAL_FORWARD, 2),)
    right = (Event(EventKind.SOLVER_EXPANSION, 3),)
    assert trace_cost(DECODE_UNIT_WORK_MODEL, left) == 2
    assert trace_cost(DECODE_UNIT_WORK_MODEL, right) == 3
    assert trace_cost_append(DECODE_UNIT_WORK_MODEL, left, right) == 5
    assert trace_cost(DECODE_UNIT_WORK_MODEL, left + right) == 5
    assert trace_cost(SOLVER_QUERY_MODEL, left + right) == 3
    assert trace_cost(NEURAL_FORWARD_ONLY_MODEL, left + right) == 2
    assert trace_cost(TRAIN_STEP_MODEL, left + right) == 2


def test_decode_projection_lossless_represented_unknown_unobserved() -> None:
    stats = DecodeStats(
        tokens_emitted=2,
        dfa_sync_count=3,
        solver_expanded_nodes=5,
        solver_verifier_calls=7,
        forwards_count=11,
        denoiser_ms=12.5,  # empirical wall — must not enter abstract cost
        total_ms=99.0,
    )
    projection = project_decode_stats(stats)
    assert projection.abstract_cost == 28
    assert projection.wall_clock_claim is False
    assert projection.counts == DecodeUnitWorkCounts(2, 3, 5, 7, 11)
    assert EventKind.NEURAL_BACKWARD.value in projection.unobserved_event_kinds
    assert EventKind.OPTIMIZER_STEP.value in projection.unobserved_event_kinds
    assert EventKind.KERNEL_LAUNCH.value in projection.unobserved_event_kinds
    assert EventKind.MEMORY_WRITE.value in projection.unobserved_event_kinds
    # Represented fields project exactly.
    assert projection.projected_fields[EventKind.TOKENIZE.value] == "tokens_emitted"
    assert DECODE_STATS_EVENT_OWNERS[EventKind.NEURAL_BACKWARD] is None
    record = build_decode_stats_record(stats, measurement_stage="steady_state")
    from_record = project_decode_stats(record.stats, record=record)
    assert from_record.abstract_cost == 28
    assert from_record.record_digest == record.record_digest


def test_wall_clock_requires_explicit_assumption() -> None:
    trace = (Event(EventKind.NEURAL_FORWARD, 3),)
    assumption = ThroughputAssumption(latency_per_cost_unit=10)
    assert wall_clock_upper_bound(DECODE_UNIT_WORK_MODEL, assumption, trace) == 30
    assert physical_latency_hypothesis_holds(
        DECODE_UNIT_WORK_MODEL, assumption, trace, 30
    )
    assert not physical_latency_hypothesis_holds(
        DECODE_UNIT_WORK_MODEL, assumption, trace, 31
    )
    # Without an assumption object there is no transfer API that invents latency.
    assert assumption.to_dict()["wall_clock_derived_in_kernel"] is False


def test_four_axis_export_persists_model_identity() -> None:
    evidence = build_event_trace_evidence(
        DecodeStats(tokens_emitted=1, forwards_count=1)
    )
    axes = export_four_axis_event_trace_evidence(evidence)
    assert axes.resource_bounds.status == "proved_axis"
    assert axes.resource_bounds.bound_ast_id == BOUND_AST_ID
    assert "DecodeUnitWorkModel" in (axes.resource_bounds.notes or "")
    assert axes.computability.status == "proved_axis"
    assert axes.implementation_refinement.status == "empirical_remainder"
    payload = evidence.to_dict()
    assert payload["schema"] == EVENT_TRACE_SCHEMA
    assert payload["resource_cost_model_id"] == "DecodeUnitWorkModel.v1"
    assert payload["wall_clock_claim"] is False


def test_bound_ast_matches_unit_work_sum() -> None:
    result = evaluate_bound(
        BOUND_AST_ID,
        {
            "tokenize": 2,
            "grammar_transition": 3,
            "solver_expansion": 5,
            "certificate_check": 7,
            "neural_forward": 11,
        },
    )
    assert result.is_integer
    assert result.value == "28"


def test_frozen_fixture_file() -> None:
    payload = json.loads(_FIXTURES.read_text(encoding="utf-8"))
    assert payload["schema"] == "event_trace_fixtures/v1"
    assert payload["cost_model"] == "DecodeUnitWorkModel"
    live = {row["id"]: row for row in frozen_fixture_rows()}
    for row in payload["cases"]:
        got = live[row["id"]]
        assert got["abstract_cost"] == row["expected_cost"]
        assert got["compose_left_right"] == 5
    assert set(k.value for k in unobserved_event_kinds_for_decode()) == set(
        payload["unobserved_under_decode"]
    )
