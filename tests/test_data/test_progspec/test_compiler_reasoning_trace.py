"""Schema/serialization tests for CompilerReasoningTraceV1 (LOT0-02 / SLM-249).

Acceptance-criteria test numbering below refers to the "Tests" section of
Linear issue SLM-249.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from slm_training.data.progspec.compiler_reasoning_trace import (
    CANONICAL_STAGE_ORDER,
    CompilerReasoningTraceV1,
    TraceStepV1,
    accepted_set_equal,
    make_partial_trace,
    make_shuffled_control,
    parse_visible_trace,
    serialize_visible_trace,
    validate_no_cycles,
)


def _step(index: int, kind: str, *, target: str = "t", **overrides) -> TraceStepV1:
    fields = dict(
        record_id="rec1",
        source="fixture",
        split="smoke",
        fingerprint="fp1",
        step_index=index,
        step_kind=kind,
        canonical_target=f"{target}_{index}",
        accepted_targets=(f"{target}_{index}",),
    )
    fields.update(overrides)
    return TraceStepV1(**fields)


def _linear_trace() -> CompilerReasoningTraceV1:
    steps = tuple(
        _step(
            i,
            kind,
            partial_order=(CANONICAL_STAGE_ORDER[i - 1],) if i > 0 else (),
        )
        for i, kind in enumerate(CANONICAL_STAGE_ORDER)
    )
    return CompilerReasoningTraceV1(
        record_id="rec1", source="fixture", split="smoke", fingerprint="fp1", steps=steps
    )


# Test 1: typed trace round-trips through visible serialization without loss.
def test_visible_serialization_round_trips_without_loss() -> None:
    trace = _linear_trace()
    text = serialize_visible_trace(trace)
    assert text.startswith("<trace_v1>") and text.endswith("</trace_v1>")
    rebuilt = parse_visible_trace(text)
    assert rebuilt == trace


def test_to_dict_from_dict_round_trip() -> None:
    trace = _linear_trace()
    rebuilt = CompilerReasoningTraceV1.from_dict(trace.to_dict())
    assert rebuilt == trace
    reparsed = CompilerReasoningTraceV1.from_dict(json.loads(json.dumps(trace.to_dict())))
    assert reparsed == trace


# Test 2 (extraction determinism/versioning is covered in
# test_compiler_reasoning_trace_extract.py against real ProgramSpec input).
def test_trace_version_is_pinned() -> None:
    trace = _linear_trace()
    assert trace.trace_version == "1"
    for step in trace.steps:
        assert step.trace_version == "1"


# Test 3: accepted-set/set-valued comparison is permutation invariant.
def test_accepted_set_equal_is_permutation_invariant() -> None:
    a = _step(0, "intent_contract", accepted_targets=("x", "y", "z"), canonical_target="x")
    b = _step(0, "intent_contract", accepted_targets=("z", "x", "y"), canonical_target="z")
    assert accepted_set_equal(a, b)
    c = _step(0, "intent_contract", accepted_targets=("x", "y"), canonical_target="x")
    assert not accepted_set_equal(a, c)


# Test 4: dependencies/partial order contain no cycles unless explicitly supported.
def test_cyclic_partial_order_is_rejected() -> None:
    steps = list(_linear_trace().steps)
    steps[0] = steps[0].model_copy(update={"partial_order": ("serialization_plan",)})
    with pytest.raises(ValidationError, match="cyclic"):
        CompilerReasoningTraceV1(
            record_id="rec1",
            source="fixture",
            split="smoke",
            fingerprint="fp1",
            steps=tuple(steps),
        )


def test_validate_no_cycles_allows_dependency_on_a_later_declared_stage() -> None:
    # A partial trace's step may declare a dependency on a stage_order member
    # that has not been extracted yet (e.g. a curriculum-stage-0 forward
    # reference); that alone is not a cycle.
    steps = (_step(0, "intent_contract", partial_order=("component_inventory",)),)
    validate_no_cycles(CANONICAL_STAGE_ORDER, steps)  # must not raise


# Test 5: no evaluator/model output enters target extraction -- schema-level check
# that TraceStepV1 has no field named for a score/reward/judge/model output.
def test_schema_has_no_evaluator_or_model_score_field() -> None:
    forbidden_substrings = ("score", "reward", "judge", "logit", "probability")
    for name in TraceStepV1.model_fields:
        lowered = name.lower()
        assert not any(bad in lowered for bad in forbidden_substrings), (
            f"TraceStepV1.{name} looks like it carries evaluator/model output"
        )


# Test 8: truncation/fallback is explicit and cannot silently drop a semantic
# obligation.
def test_partial_trace_is_explicitly_marked() -> None:
    trace = _linear_trace()
    assert not trace.is_partial
    partial = make_partial_trace(trace, 3)
    assert partial.is_partial
    assert partial.step_count == 3
    assert partial.stage_order == trace.stage_order  # full stage set still declared


def test_out_of_range_partial_stage_count_rejected() -> None:
    trace = _linear_trace()
    with pytest.raises(ValueError, match="stage_count"):
        make_partial_trace(trace, len(trace.stage_order) + 1)


def test_is_truncated_flag_is_explicit_not_silent() -> None:
    step = _step(0, "intent_contract", is_truncated=True, ambiguity_reason="exceeded K")
    assert step.is_truncated is True
    assert step.ambiguity_reason == "exceeded K"


# Test 9: shuffled/wrong controls preserve length/support while changing intended
# semantics.
def test_shuffled_control_preserves_length_changes_targets() -> None:
    trace = _linear_trace()
    shuffled = make_shuffled_control(trace, seed=7)
    assert shuffled.step_count == trace.step_count
    assert [s.step_kind for s in shuffled.steps] == [s.step_kind for s in trace.steps]
    for original, permuted in zip(trace.steps, shuffled.steps, strict=True):
        assert original.canonical_target != permuted.canonical_target


def test_shuffled_control_is_deterministic_for_a_fixed_seed() -> None:
    trace = _linear_trace()
    a = make_shuffled_control(trace, seed=11)
    b = make_shuffled_control(trace, seed=11)
    assert a == b


def test_shuffled_control_requires_at_least_two_steps() -> None:
    trace = make_partial_trace(_linear_trace(), 1)
    with pytest.raises(ValueError, match="at least 2 steps"):
        make_shuffled_control(trace, seed=1)


def test_step_out_of_declared_order_rejected() -> None:
    steps = (
        _step(0, "component_inventory"),
        _step(1, "intent_contract"),
    )
    with pytest.raises(ValidationError, match="out of declared stage order"):
        CompilerReasoningTraceV1(
            record_id="rec1",
            source="fixture",
            split="smoke",
            fingerprint="fp1",
            steps=steps,
        )


def test_step_kind_not_in_stage_order_rejected() -> None:
    step = _step(0, "intent_contract")
    with pytest.raises(ValidationError, match="not declared in stage_order"):
        CompilerReasoningTraceV1(
            record_id="rec1",
            source="fixture",
            split="smoke",
            fingerprint="fp1",
            stage_order=("component_inventory",),
            steps=(step,),
        )


def test_canonical_target_must_be_in_accepted_targets_when_declared() -> None:
    with pytest.raises(ValidationError, match="accepted_targets"):
        TraceStepV1(
            record_id="rec1",
            source="fixture",
            split="smoke",
            fingerprint="fp1",
            step_index=0,
            step_kind="intent_contract",
            canonical_target="a",
            accepted_targets=("b", "c"),
        )


def test_mismatched_child_record_id_or_fingerprint_rejected() -> None:
    steps = list(_linear_trace().steps)
    steps[0] = steps[0].model_copy(update={"record_id": "other"})
    with pytest.raises(ValidationError, match="record_id"):
        CompilerReasoningTraceV1(
            record_id="rec1",
            source="fixture",
            split="smoke",
            fingerprint="fp1",
            steps=tuple(steps),
        )
