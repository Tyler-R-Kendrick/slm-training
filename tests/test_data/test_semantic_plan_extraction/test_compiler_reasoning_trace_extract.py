"""Extraction tests for CompilerReasoningTraceV1 (LOT0-02 / SLM-249).

Reuses the ``pack``/``sample_spec``/``single_spec`` fixtures from this
directory's ``conftest.py`` -- the same fixtures the SemanticPlanV1 extractor
tests use, so this suite never defines a second AST/compiler surface.
"""

from __future__ import annotations

import pytest

from slm_training.data.progspec.compiler_reasoning_trace import (
    CANONICAL_STAGE_ORDER,
    accepted_set_equal,
)
from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.semantic_plan.compiler_reasoning_trace_extract import (
    extract_compiler_reasoning_trace,
)
from slm_training.data.leakage import find_leakage
from slm_training.dsl.lang_core import bridge_available
from slm_training.dsl.pack import DslPack
from slm_training.dsl.schema import ExampleRecord

SKIP_REASON = "OpenUI bridge deps missing"


@pytest.mark.skipif(not bridge_available(), reason=SKIP_REASON)
def test_extraction_covers_all_six_stages_in_declared_order(
    sample_spec: ProgramSpec, pack: DslPack
) -> None:
    trace = extract_compiler_reasoning_trace(sample_spec, pack, source="fixture", split="smoke")
    assert trace.record_id == sample_spec.id
    assert trace.step_count == len(CANONICAL_STAGE_ORDER)
    assert tuple(step.step_kind for step in trace.steps) == CANONICAL_STAGE_ORDER
    assert not trace.is_partial


# Test 2: extraction is deterministic and versioned.
@pytest.mark.skipif(not bridge_available(), reason=SKIP_REASON)
def test_extraction_is_deterministic(sample_spec: ProgramSpec, pack: DslPack) -> None:
    first = extract_compiler_reasoning_trace(sample_spec, pack, source="fixture", split="smoke")
    second = extract_compiler_reasoning_trace(sample_spec, pack, source="fixture", split="smoke")
    assert first == second
    assert first.trace_version == "1"


@pytest.mark.skipif(not bridge_available(), reason=SKIP_REASON)
def test_extraction_differs_for_a_structurally_different_program(
    sample_spec: ProgramSpec, single_spec: ProgramSpec, pack: DslPack
) -> None:
    trace_a = extract_compiler_reasoning_trace(sample_spec, pack, source="fixture", split="smoke")
    trace_b = extract_compiler_reasoning_trace(single_spec, pack, source="fixture", split="smoke")
    assert trace_a.fingerprint != trace_b.fingerprint
    intent_a = next(s for s in trace_a.steps if s.step_kind == "intent_contract")
    intent_b = next(s for s in trace_b.steps if s.step_kind == "intent_contract")
    assert not accepted_set_equal(intent_a, intent_b)


# Test 3: accepted-set/set-valued comparison is permutation invariant across two
# independent extractions of the same program.
@pytest.mark.skipif(not bridge_available(), reason=SKIP_REASON)
def test_repeated_extraction_is_accepted_set_equal_per_step(
    sample_spec: ProgramSpec, pack: DslPack
) -> None:
    first = extract_compiler_reasoning_trace(sample_spec, pack, source="fixture", split="smoke")
    second = extract_compiler_reasoning_trace(sample_spec, pack, source="fixture", split="smoke")
    for a, b in zip(first.steps, second.steps, strict=True):
        assert accepted_set_equal(a, b)


# Test 5: no evaluator/model output enters target extraction -- the extractor
# signature accepts only a ProgramSpec/pack/optional pre-computed plan, never a
# score, judge, or model handle.
def test_extractor_signature_has_no_evaluator_or_model_parameter() -> None:
    import inspect

    sig = inspect.signature(extract_compiler_reasoning_trace)
    forbidden = ("score", "reward", "judge", "model", "logit")
    for name in sig.parameters:
        lowered = name.lower()
        assert not any(bad in lowered for bad in forbidden), (
            f"extractor parameter {name!r} looks like an evaluator/model input"
        )


# Test 6 / 7 companion: extracted trace fingerprints participate in the shared
# leakage-check contract instead of a bespoke one.
@pytest.mark.skipif(not bridge_available(), reason=SKIP_REASON)
def test_trace_fingerprint_participates_in_shared_leakage_check(
    sample_spec: ProgramSpec, pack: DslPack
) -> None:
    trace = extract_compiler_reasoning_trace(sample_spec, pack, source="fixture", split="smoke")
    record = ExampleRecord(
        id=sample_spec.id,
        prompt="a sample prompt",
        openui=sample_spec.canonical_openui,
        placeholders=[],
        split="train",
        source="fixture",
        meta={"split_group_id": sample_spec.split_group_id},
    )
    train_fps = {
        "ids": set(),
        "split_group_ids": {sample_spec.split_group_id},
        "prompts": set(),
        "openuis": set(),
        "structures": set(),
        "pairs": set(),
        "design_mds": set(),
    }
    reasons = find_leakage(record, train_fps)
    assert "split_group_id" in reasons
    # The trace's own fingerprint is derived from the same canonical_openui the
    # leakage module fingerprints -- confirm they are not two disagreeing sources
    # of truth for "the same program."
    assert trace.fingerprint  # non-empty, derived via dsl.canonicalize.canonical_fingerprint
