from __future__ import annotations

import pytest

from slm_training.data.verify.stack import Gate
from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.semantic_failure import (
    WITNESS_SCHEMA_VERSION,
    SemanticFailureFamily,
    SemanticFailureTaxonomyV1,
    VerifierWitnessV1,
    build_verifier_witness,
    family_for_reason,
    trace_semantic_failure,
    witness_integrity_ok,
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


def test_witness_is_lossless_deterministic_and_covers_every_gate() -> None:
    trace = trace_semantic_failure("root = Stack([])", _record())
    one = build_verifier_witness(trace)
    two = build_verifier_witness(trace)
    assert one == two  # same trace -> byte/canonical-identical witness payload
    assert one.to_dict() == two.to_dict()
    assert one.schema_version == WITNESS_SCHEMA_VERSION
    assert one.source_trace_fingerprint == trace.trace_fingerprint
    gate_localizations = [loc for loc in one.localizations if loc.source == "gate"]
    assert len(gate_localizations) == 13
    assert {loc.gate for loc in gate_localizations} == {gate.value for gate in Gate}
    # G12 (human audit) has no evidence by default: stays UNKNOWN, never fabricated.
    human_audit = next(loc for loc in gate_localizations if loc.gate == Gate.HUMAN_AUDIT.value)
    assert human_audit.completeness_class == "UNKNOWN"
    assert f"gate:{Gate.HUMAN_AUDIT.value}" in one.unresolved_localizations


def test_witness_round_trips_and_rejects_tampering() -> None:
    trace = trace_semantic_failure("root =", _record())
    witness = build_verifier_witness(trace)
    assert witness_integrity_ok(witness)

    payload = witness.to_dict()
    restored = VerifierWitnessV1.from_dict(payload)
    assert restored == witness

    tampered = dict(payload)
    tampered["first_failed_gate"] = "G3"
    with pytest.raises(ValueError, match="digest mismatch"):
        VerifierWitnessV1.from_dict(tampered)


def test_witness_rejects_unknown_schema_version() -> None:
    trace = trace_semantic_failure("root =", _record())
    payload = build_verifier_witness(trace).to_dict()
    payload["schema_version"] = "verifier_witness/v0"
    with pytest.raises(ValueError, match="unsupported verifier witness schema"):
        VerifierWitnessV1.from_dict(payload)


def test_witness_redacts_secret_shaped_gate_detail() -> None:
    trace = trace_semantic_failure("root =", _record())
    injected_detail = "token=sk-abcdefghijklmnop leaked in parse error"
    poisoned_outcomes = tuple(
        {**outcome, "detail": injected_detail} if outcome["gate"] == Gate.LEXICAL.value else outcome
        for outcome in trace.gate_outcomes
    )
    from dataclasses import replace

    poisoned_trace = replace(trace, gate_outcomes=poisoned_outcomes)
    witness = build_verifier_witness(poisoned_trace)
    lexical = next(loc for loc in witness.localizations if loc.gate == Gate.LEXICAL.value)
    assert "sk-abcdefghijklmnop" not in lexical.detail
    assert "[REDACTED]" in lexical.detail
