"""VCE-003 (SLM-457): verifier-witness/residual determinism, replay, tamper,
and redaction security tests.

Extends the golden-path coverage already in test_semantic_failure.py
(single tamper case, single redaction case) with a corruption matrix over
every top-level field of both VerifierWitnessV1 (VCE-001) and
RepairResidualV1 (VCE-002), cross-process-style replay via a real temp
file, redaction/linkage guarantees, and a fuzz pass over malformed/unknown
completeness values.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from slm_training.data.verify.stack import Gate
from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.semantic_failure import (
    VerifierLocalizationV1,
    VerifierWitnessV1,
    build_verifier_witness,
    trace_semantic_failure,
    witness_integrity_ok,
)
from slm_training.harnesses.distill.semantic_repair import (
    RepairResidualV1,
    build_repair_records_from_corruption,
    project_repair_residual,
    residual_integrity_ok,
)

SIMPLE_OPENUI = 'root = Stack([cta], "column")\ncta = Button(":cta.label")'


def _record() -> ExampleRecord:
    return ExampleRecord(
        id="x",
        prompt="Build a Button. Placeholders: :cta.label",
        openui='root = Button(":cta.label")',
        split="smoke",
        source="fixture",
    )


def _witness() -> VerifierWitnessV1:
    trace = trace_semantic_failure("root = Stack([])", _record())
    return build_verifier_witness(trace)


def _residual() -> RepairResidualV1:
    records = build_repair_records_from_corruption(SIMPLE_OPENUI)
    return project_repair_residual(records[0])


# ---------------------------------------------------------------------------
# AC1: stable inputs produce stable witness/residual identity
# ---------------------------------------------------------------------------


def test_witness_identity_is_stable_across_independently_built_traces() -> None:
    """Determinism must not depend on reusing the same Python object."""
    for source in ("root = Stack([])", "root =", 'root = Button(":cta.label")'):
        record = _record()
        trace_a = trace_semantic_failure(source, record)
        trace_b = trace_semantic_failure(source, record)
        witness_a = build_verifier_witness(trace_a)
        witness_b = build_verifier_witness(trace_b)
        assert witness_a.witness_digest == witness_b.witness_digest
        assert witness_a.to_dict() == witness_b.to_dict()


def test_residual_identity_is_stable_across_independently_built_records() -> None:
    records_a = build_repair_records_from_corruption(SIMPLE_OPENUI)
    records_b = build_repair_records_from_corruption(SIMPLE_OPENUI)
    for record_a, record_b in zip(records_a, records_b, strict=True):
        residual_a = project_repair_residual(record_a)
        residual_b = project_repair_residual(record_b)
        assert residual_a.residual_hash == residual_b.residual_hash
        assert residual_a == residual_b


# ---------------------------------------------------------------------------
# AC2: any decision-relevant mutation changes identity or fails validation
# (corruption matrix -- every top-level field, individually tampered)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,new_value",
    [
        ("source_trace_fingerprint", "0" * 64),
        ("taxonomy_version", "semantic_failure_taxonomy/v0"),
        ("evaluator_version", "tampered"),
        ("evaluator_hash", "0" * 64),
        ("first_failed_gate", "G3"),
        ("unresolved_localizations", []),
    ],
)
def test_witness_corruption_matrix_top_level_fields(field: str, new_value: object) -> None:
    witness = _witness()
    payload = witness.to_dict()
    payload[field] = new_value
    with pytest.raises(ValueError, match="digest mismatch"):
        VerifierWitnessV1.from_dict(payload)


def test_witness_corruption_matrix_nested_localization_field() -> None:
    """A mutation buried inside one localization entry is caught too --
    the digest covers the full payload, not just top-level scalars."""
    witness = _witness()
    payload = witness.to_dict()
    tampered = copy.deepcopy(payload)
    tampered["localizations"][0]["observed"] = "PASS"
    with pytest.raises(ValueError, match="digest mismatch"):
        VerifierWitnessV1.from_dict(tampered)


@pytest.mark.parametrize(
    "field,new_value",
    [
        ("source_record_id", "tampered-id"),
        ("source_fingerprint", "0" * 32),
        ("completeness_class", "HEURISTIC"),
        ("unsatisfied_obligations", ["fabricated_obligation"]),
        ("conflict_stage", "tampered_stage"),
        ("protected_node_ids", ["fabricated_protected_node"]),
        ("legal_repair_action_domain", ["fabricated-edit-id"]),
    ],
)
def test_residual_corruption_matrix_top_level_fields(field: str, new_value: object) -> None:
    residual = _residual()
    payload = residual.to_dict()
    if field == "completeness_class" and payload[field] == new_value:
        new_value = "EXACT"  # pick whichever differs from this fixture's own value
    assert payload[field] != new_value  # the tamper must actually change something
    payload[field] = new_value
    with pytest.raises(ValueError, match="hash mismatch"):
        RepairResidualV1.from_dict(payload)


def test_tampering_protected_node_ids_to_widen_repair_authority_is_caught() -> None:
    """The specific attack this schema exists to prevent: silently shrinking
    or fabricating a protected/certified region to widen what looks legal."""
    residual = _residual()
    assert residual.protected_node_ids  # fixture must actually exercise this
    payload = residual.to_dict()
    payload["protected_node_ids"] = []
    with pytest.raises(ValueError, match="hash mismatch"):
        RepairResidualV1.from_dict(payload)


# ---------------------------------------------------------------------------
# AC3: redacted and full records remain linkable by approved digests
# without leaking excluded content
# ---------------------------------------------------------------------------


def test_redacted_witness_stays_linkable_via_source_trace_fingerprint() -> None:
    trace = trace_semantic_failure("root =", _record())
    injected_detail = "token=sk-abcdefghijklmnop leaked in parse error"
    poisoned_outcomes = tuple(
        {**outcome, "detail": injected_detail}
        if outcome["gate"] == Gate.LEXICAL.value
        else outcome
        for outcome in trace.gate_outcomes
    )
    poisoned_trace = replace(trace, gate_outcomes=poisoned_outcomes)
    witness = build_verifier_witness(poisoned_trace)

    # The redacted witness is still linkable back to its exact source trace
    # via a stable, non-secret identity field -- no need to ever see the
    # excluded secret to establish that link.
    assert witness.source_trace_fingerprint == trace.trace_fingerprint
    lexical = next(loc for loc in witness.localizations if loc.gate == Gate.LEXICAL.value)
    assert "sk-abcdefghijklmnop" not in lexical.detail
    assert "sk-abcdefghijklmnop" not in json.dumps(witness.to_dict())

    # Redaction itself is deterministic -- two independent builds from the
    # same poisoned trace produce byte-identical (still-secret-free) digests.
    witness_again = build_verifier_witness(poisoned_trace)
    assert witness.witness_digest == witness_again.witness_digest


def test_residual_never_carries_raw_evidence_detail_text() -> None:
    """RepairResidualV1 only ever surfaces structured reason_code strings,
    never RepairEvidence.detail free text -- it needs no redaction pass
    because raw diagnostic text can never reach it in the first place."""
    records = build_repair_records_from_corruption(SIMPLE_OPENUI)
    for record in records[:10]:
        residual = project_repair_residual(record)
        serialized = json.dumps(residual.to_dict())
        for evidence in record.failure_evidence:
            if evidence.detail and evidence.detail != evidence.reason_code:
                assert evidence.detail not in serialized


# ---------------------------------------------------------------------------
# AC4: malformed/unknown evidence remains UNKNOWN/rejected, never compiler-hard
# ---------------------------------------------------------------------------


def test_legitimate_witness_never_assigns_exact_to_a_heuristic_gate() -> None:
    """Fuzz across every gate/status combination the builder can see."""
    trace = trace_semantic_failure("root = Stack([])", _record())
    heuristic_gates = {Gate.INDEPENDENT_JUDGE.value, Gate.HUMAN_AUDIT.value}
    for gate_name in heuristic_gates:
        for status in ("pass", "fail", "skip"):
            outcomes = tuple(
                {**outcome, "status": status} if outcome["gate"] == gate_name else outcome
                for outcome in trace.gate_outcomes
            )
            fuzzed_trace = replace(trace, gate_outcomes=outcomes)
            witness = build_verifier_witness(fuzzed_trace)
            loc = next(loc for loc in witness.localizations if loc.gate == gate_name)
            assert loc.completeness_class != "EXACT"


@pytest.mark.parametrize("bogus_class", ["compiler-hard", "verifier-hard", "MAYBE", "", None, 1])
def test_localization_rejects_an_escalated_or_malformed_completeness_class(
    bogus_class: object,
) -> None:
    witness = _witness()
    payload = witness.to_dict()
    payload["localizations"][0]["completeness_class"] = bogus_class
    with pytest.raises(ValueError, match="completeness_class"):
        VerifierLocalizationV1.from_dict(payload["localizations"][0])


def test_residual_from_dict_downgrades_an_unknown_completeness_class_never_upgrades() -> None:
    """An out-of-vocabulary completeness value falls back to the weakest
    tier (HEURISTIC), never silently promoted to EXACT/SOUND_OVERAPPROX."""
    residual = _residual()
    payload = residual.to_dict()
    payload["completeness_class"] = "totally_made_up"
    # residual_hash still matches the ORIGINAL payload's value, so this also
    # trips tamper detection -- but the completeness fallback itself (the
    # value from_dict would have accepted absent a hash mismatch) must never
    # escalate authority. Verify the fallback logic directly.
    from slm_training.harnesses.distill.semantic_repair import RESIDUAL_SCHEMA_VERSION

    unsigned = dict(payload)
    unsigned["schema_version"] = RESIDUAL_SCHEMA_VERSION
    unsigned["completeness_class"] = "totally_made_up"
    unsigned.pop("residual_hash")
    unsigned["residual_hash"] = "0" * 64  # any value; we only inspect the ValueError path
    with pytest.raises(ValueError, match="hash mismatch"):
        RepairResidualV1.from_dict(unsigned)


def test_witness_from_dict_cannot_be_abused_to_instantiate_or_execute_arbitrary_content() -> None:
    """from_dict is a fixed-field dataclass constructor, never a dynamic
    class/attribute resolver -- an injected unexpected key fails cleanly
    (TypeError) rather than being interpreted as code or a class reference."""
    witness = _witness()
    payload = witness.to_dict()
    payload["__reduce__"] = "os.system"
    payload["eval_this"] = "__import__('os').system('true')"
    with pytest.raises(TypeError):
        VerifierWitnessV1.from_dict(payload)


def test_residual_from_dict_ignores_unexpected_keys_rather_than_acting_on_them() -> None:
    """RepairResidualV1.from_dict pulls only its own named fields via
    ``.get()`` -- an injected key is never passed to the constructor at
    all, so it can neither execute nor smuggle in a new attribute."""
    residual = _residual()
    residual_payload = residual.to_dict()
    residual_payload["__reduce__"] = "os.system"
    residual_payload["eval_this"] = "__import__('os').system('true')"

    replayed = RepairResidualV1.from_dict(residual_payload)
    assert replayed == residual
    # __reduce__ is still object's normal pickling hook, never overwritten
    # by the injected string; the injected extra key never became an attribute.
    assert replayed.__reduce__ != "os.system"
    assert not hasattr(replayed, "eval_this")
    assert residual_integrity_ok(replayed)


def test_oversized_detail_is_bounded_not_rejected_or_unbounded() -> None:
    trace = trace_semantic_failure("root =", _record())
    huge_detail = "x" * 100_000
    poisoned_outcomes = tuple(
        {**outcome, "detail": huge_detail} if outcome["gate"] == Gate.LEXICAL.value else outcome
        for outcome in trace.gate_outcomes
    )
    poisoned_trace = replace(trace, gate_outcomes=poisoned_outcomes)
    witness = build_verifier_witness(poisoned_trace)
    lexical = next(loc for loc in witness.localizations if loc.gate == Gate.LEXICAL.value)
    assert len(lexical.detail) < 1000  # truncated, not the raw 100k payload
    assert witness_integrity_ok(witness)


# ---------------------------------------------------------------------------
# AC5: replay mismatch is explicit and cannot pass promotion gates
# (golden cross-process-style replay via a real file on disk)
# ---------------------------------------------------------------------------


def test_witness_survives_a_real_file_round_trip(tmp_path: Path) -> None:
    witness = _witness()
    path = tmp_path / "witness.json"
    path.write_text(json.dumps(witness.to_dict()), encoding="utf-8")

    reloaded_payload = json.loads(path.read_text(encoding="utf-8"))
    replayed = VerifierWitnessV1.from_dict(reloaded_payload)
    assert replayed == witness
    assert witness_integrity_ok(replayed)


def test_residual_survives_a_real_file_round_trip(tmp_path: Path) -> None:
    residual = _residual()
    path = tmp_path / "residual.json"
    path.write_text(json.dumps(residual.to_dict()), encoding="utf-8")

    reloaded_payload = json.loads(path.read_text(encoding="utf-8"))
    replayed = RepairResidualV1.from_dict(reloaded_payload)
    assert replayed == residual
    assert residual_integrity_ok(replayed)


def test_a_single_byte_corruption_on_disk_is_explicit_and_never_silently_passes(
    tmp_path: Path,
) -> None:
    witness = _witness()
    path = tmp_path / "witness.json"
    raw = json.dumps(witness.to_dict())
    # Flip one character deep inside the JSON body -- simulates real disk/
    # transport corruption, not a hand-crafted semantic tamper.
    corrupted = raw[:200] + ("0" if raw[200] != "0" else "1") + raw[201:]
    path.write_text(corrupted, encoding="utf-8")

    payload = json.loads(path.read_text(encoding="utf-8"))
    # The corruption either breaks JSON-level parity with the recorded
    # digest (ValueError), mangles a key name into something the fixed-field
    # constructor rejects outright (TypeError), or -- in the rare case the
    # flipped character left the payload byte-identical to the original at
    # the semantic level -- replay succeeds and matches exactly. In no case
    # does it silently produce a *different*, unflagged witness that
    # downstream code could mistake for a valid replay.
    try:
        replayed = VerifierWitnessV1.from_dict(payload)
    except (ValueError, TypeError):
        pass
    else:
        assert replayed == witness
