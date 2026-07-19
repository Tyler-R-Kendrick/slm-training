"""Regression tests for synthetic-data integrity gates."""

from __future__ import annotations

from slm_training.data.contract import GenerationRequest
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.train_data.integrity import (
    CheckStatus,
    IntegrityCheck,
    evaluate_integrity,
)


def _record(openui: str, **kwargs: object) -> ExampleRecord:
    return ExampleRecord(
        id="test-1",
        prompt="Build a card.",
        openui=openui,
        split="train",
        source="fixture",
        **kwargs,
    )


def test_valid_document_passes_all_checks() -> None:
    source = (
        'root = Card([header, body])\n'
        'header = CardHeader(":card.title")\n'
        'body = TextContent(":card.body")\n'
    )
    record = _record(source, placeholders=[":card.title", ":card.body"])
    report = evaluate_integrity(record)
    assert report.passed
    failed = [c.name for c in report.checks if c.status is CheckStatus.FAIL]
    assert not failed


def test_parse_invalid_record_fails() -> None:
    record = _record('root = Card(["unclosed')
    report = evaluate_integrity(record)
    assert not report.passed
    assert IntegrityCheck.PARSE_VALID.value in report.hard_fail_reasons


def test_missing_root_fails() -> None:
    record = _record('header = CardHeader(":card.title")\n')
    report = evaluate_integrity(record)
    assert not report.passed
    # Missing root is caught by the compiler/codec round-trip and reference gate.
    assert any(
        reason in report.hard_fail_reasons
        for reason in {
            IntegrityCheck.COMPILER_VALID.value,
            IntegrityCheck.PRODUCTION_CODEC_ROUNDTRIP_HASH.value,
            IntegrityCheck.REFERENCE_SCOPE_VALID.value,
        }
    )


def test_duplicate_binder_fails_reference_scope() -> None:
    source = (
        'root = Card([header])\n'
        'header = CardHeader(":card.title")\n'
        'header = TextContent(":card.body")\n'
    )
    record = _record(source)
    report = evaluate_integrity(record)
    assert not report.passed
    assert IntegrityCheck.REFERENCE_SCOPE_VALID.value in report.hard_fail_reasons


def test_unresolved_ref_fails_reference_scope() -> None:
    source = 'root = Card([missing])\n'
    record = _record(source)
    report = evaluate_integrity(record)
    assert not report.passed
    assert IntegrityCheck.REFERENCE_SCOPE_VALID.value in report.hard_fail_reasons


def test_placeholder_set_mismatch_fails() -> None:
    source = (
        'root = Card([TextContent(":card.body")])\n'
    )
    record = _record(source, placeholders=[":card.title"])
    report = evaluate_integrity(record)
    assert not report.passed
    assert IntegrityCheck.PLACEHOLDER_SET_MATCH.value in report.hard_fail_reasons


def test_request_target_contract_match() -> None:
    source = 'root = Card([TextContent(":card.title")])\n'
    record = _record(source, placeholders=[":card.title"])
    request = GenerationRequest(prompt=record.prompt, slot_contract=(":card.title",))
    report = evaluate_integrity(record, request=request)
    assert report.passed


def test_request_target_contract_mismatch() -> None:
    source = 'root = Card([TextContent(":card.title")])\n'
    record = _record(source, placeholders=[":card.title"])
    request = GenerationRequest(prompt=record.prompt, slot_contract=(":card.other",))
    report = evaluate_integrity(record, request=request)
    assert not report.passed
    assert IntegrityCheck.REQUEST_TARGET_CONTRACT_MATCH.value in report.hard_fail_reasons


def test_split_leakage_detects_exact_fingerprint() -> None:
    source = 'root = Card([TextContent(":card.title")])\n'
    record = _record(source, placeholders=[":card.title"])
    base = evaluate_integrity(record)
    held_out = {base.hashes["structure"]}
    report = evaluate_integrity(record, held_out_fingerprints=held_out)
    assert not report.passed
    assert IntegrityCheck.SPLIT_LEAKAGE_STATUS.value in report.hard_fail_reasons


def test_alpha_equivalent_records_share_ast_canonical_hash() -> None:
    a = _record(
        'root = Card([h])\nh = CardHeader(":card.title")\n',
        placeholders=[":card.title"],
    )
    b = _record(
        'root = Card([title])\ntitle = CardHeader(":card.title")\n',
        placeholders=[":card.title"],
    )
    ra = evaluate_integrity(a)
    rb = evaluate_integrity(b)
    assert ra.hashes["ast_canonical"] == rb.hashes["ast_canonical"]


def test_report_to_dict_round_trip() -> None:
    source = 'root = Card([TextContent(":card.title")])\n'
    record = _record(source, placeholders=[":card.title"])
    report = evaluate_integrity(record)
    d = report.to_dict()
    assert d["record_id"] == record.id
    assert d["schema_version"] == "synthetic_integrity/v1"
    assert isinstance(d["checks"], list)
