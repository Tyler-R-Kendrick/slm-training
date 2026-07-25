from __future__ import annotations

from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.semantic_failure import (
    SemanticFailureFamily,
    SemanticFailureTaxonomyV1,
    family_for_reason,
    trace_semantic_failure,
)


def _record() -> ExampleRecord:
    return ExampleRecord(id="x", prompt="Build a Button. Placeholders: :cta.label", openui='root = Button(":cta.label")', split="smoke", source="fixture")


def test_taxonomy_is_complete_and_known_reason_maps_losslessly() -> None:
    assert len(SemanticFailureTaxonomyV1().families) == 20
    assert family_for_reason("schema_value_role_mismatch:Button") is SemanticFailureFamily.SCHEMA_OR_VALUE_ROLE
    assert family_for_reason("future_reason") is SemanticFailureFamily.UNKNOWN


def test_trace_is_stable_preserves_first_gate_and_has_no_human_gate() -> None:
    trace = trace_semantic_failure("root = Stack([])", _record())
    restored = type(trace).from_dict(trace.to_dict())
    assert restored == trace
    assert trace.first_failed_gate is None
    assert trace.first_failure_family == SemanticFailureFamily.TRIVIAL_EMPTY_OR_MINIMAL_SHELL.value
    assert len(trace.gate_outcomes) == 13
    assert trace.gate_outcomes[-1]["status"] == "skip"


def test_parse_failure_has_no_canonical_hash_and_stable_fingerprint() -> None:
    one = trace_semantic_failure("root =", _record())
    two = trace_semantic_failure("root =", _record())
    assert one.canonical_program_sha256 is None
    assert one.trace_fingerprint == two.trace_fingerprint
