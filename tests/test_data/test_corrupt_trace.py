"""Tests for slm_training.data.corrupt.trace (SLM-120)."""

from __future__ import annotations

from slm_training.data.corrupt.trace import (
    CorruptionOperation,
    CorruptionTraceV2,
    SemanticClass,
    SeverityLevel,
)


def test_severity_levels_are_strings() -> None:
    assert SeverityLevel.S0_CLEAN.value == "S0_clean"
    assert SeverityLevel.S4_HEAVY.value == "S4_heavy"


def test_semantic_classes_are_strings() -> None:
    assert SemanticClass.COMPONENT.value == "component"
    assert SemanticClass.BINDING.value == "binding"


def test_trace_to_dict_roundtrip() -> None:
    op = CorruptionOperation(
        operator="unknown_component",
        operator_family="schema",
        semantic_class=SemanticClass.COMPONENT,
        ast_path=("root", 0),
        source_span=(12, 24),
        depends_on=("schema",),
        surface_only=False,
    )
    trace = CorruptionTraceV2(
        source_program_hash="sha256:abc",
        prompt_hash="sha256:def",
        contract_hash="sha256:ghi",
        representation="topology",
        model_family="twotower",
        severity=SeverityLevel.S1_NEAR_SOLVED_1,
        operations=(op,),
        rng_seed=42,
        policy_version="corruption/v1",
        corrupted_valid=False,
        repairable=True,
        inverse_target_hash="sha256:target",
    )
    data = trace.to_dict()
    assert data["trace_schema_version"] == "corruption_trace/v2"
    assert data["severity"] == "S1_near_solved_1"
    assert data["operations"][0]["semantic_class"] == "component"
    assert data["semantic_operation_count"] == 1


def test_s0_clean_with_operations_is_invalid() -> None:
    op = CorruptionOperation(
        operator="missing_assignment",
        operator_family="grammar",
        semantic_class=SemanticClass.STRUCTURE,
    )
    trace = CorruptionTraceV2(
        source_program_hash="sha256:abc",
        severity=SeverityLevel.S0_CLEAN,
        operations=(op,),
    )
    errors = trace.validate()
    assert any("S0_clean trace must not contain operations" in e for e in errors)


def test_non_clean_without_operations_is_invalid() -> None:
    trace = CorruptionTraceV2(
        source_program_hash="sha256:abc",
        severity=SeverityLevel.S2_NEAR_SOLVED_2,
        operations=(),
    )
    errors = trace.validate()
    assert any("non-clean trace must contain at least one operation" in e for e in errors)


def test_equivalent_rewrite_must_be_s0() -> None:
    op = CorruptionOperation(
        operator="alpha_rename",
        operator_family="reference_graph",
        semantic_class=SemanticClass.BINDING,
        equivalent_rewrite=True,
    )
    trace = CorruptionTraceV2(
        source_program_hash="sha256:abc",
        severity=SeverityLevel.S1_NEAR_SOLVED_1,
        operations=(op,),
    )
    errors = trace.validate()
    assert any("equivalent rewrite must be labeled S0_clean" in e for e in errors)


def test_surface_only_does_not_count_semantic() -> None:
    op = CorruptionOperation(
        operator="extra_quote",
        operator_family="lexical",
        semantic_class=SemanticClass.SURFACE_ONLY,
        surface_only=True,
    )
    trace = CorruptionTraceV2(
        source_program_hash="sha256:abc",
        severity=SeverityLevel.S1_NEAR_SOLVED_1,
        operations=(op,),
    )
    assert trace.semantic_operation_count == 0
    assert not trace.validate()
