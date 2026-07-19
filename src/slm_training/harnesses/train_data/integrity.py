"""Synthetic-data integrity gates for train-data admission.

The gate proves that a generated or transformed training example's surface
program, typed AST, production/choice representation, slot contract, and
binding graph describe the same canonical program. It is intentionally
separable from the verifier tier stack so it can run in audit mode before
being promoted to a hard admission gate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from slm_training.data.contract import GenerationRequest, canonical_slot_contract
from slm_training.data.leakage import fingerprint_openui, fingerprint_openui_structure
from slm_training.data.verify.stack import Gate, evaluate_gate
from slm_training.dsl.canonicalize import canonical_fingerprint
from slm_training.dsl.parser import ParseError, validate
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.production_codec import (
    decode_choices,
    decode_productions,
    encode_choices,
    encode_openui,
)
from slm_training.dsl.schema import ExampleRecord

SCHEMA_VERSION = "synthetic_integrity/v1"


class IntegrityCheck(str, Enum):
    PARSE_VALID = "parse_valid"
    SCHEMA_VALID = "schema_valid"
    COMPILER_VALID = "compiler_valid"
    AST_CANONICAL_HASH = "ast_canonical_hash"
    SURFACE_REPARSE_AST_HASH = "surface_reparse_ast_hash"
    PRODUCTION_CODEC_ROUNDTRIP_HASH = "production_codec_roundtrip_hash"
    CHOICE_CODEC_ROUNDTRIP_HASH = "choice_codec_roundtrip_hash"
    SLOT_CONTRACT_HASH = "slot_contract_hash"
    BINDING_GRAPH_HASH = "binding_graph_hash"
    PLACEHOLDER_SET_MATCH = "placeholder_set_match"
    REFERENCE_SCOPE_VALID = "reference_scope_valid"
    REQUEST_TARGET_CONTRACT_MATCH = "request_target_contract_match"
    ROOT_LINEAGE_HASH = "root_lineage_hash"
    SPLIT_LEAKAGE_STATUS = "split_leakage_status"


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IntegrityCheckResult:
    name: str
    status: CheckStatus
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status.value, "detail": self.detail}


@dataclass
class SyntheticIntegrityReport:
    schema_version: str
    record_id: str
    checks: tuple[IntegrityCheckResult, ...]
    hashes: dict[str, str]
    hard_fail_reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.hard_fail_reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "passed": self.passed,
            "hard_fail_reasons": list(self.hard_fail_reasons),
            "checks": [c.to_dict() for c in self.checks],
            "hashes": self.hashes,
        }


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_parse(record: ExampleRecord) -> IntegrityCheckResult:
    try:
        validate(record.openui)
        return IntegrityCheckResult(
            IntegrityCheck.PARSE_VALID.value, CheckStatus.PASS
        )
    except ParseError as exc:
        return IntegrityCheckResult(
            IntegrityCheck.PARSE_VALID.value, CheckStatus.FAIL, str(exc)[:240]
        )
    except Exception as exc:  # noqa: BLE001
        return IntegrityCheckResult(
            IntegrityCheck.PARSE_VALID.value, CheckStatus.FAIL, f"runtime: {exc}"[:240]
        )


def _check_schema(record: ExampleRecord) -> IntegrityCheckResult:
    try:
        program = validate(record.openui)
        if program.root is None:
            return IntegrityCheckResult(
                IntegrityCheck.SCHEMA_VALID.value, CheckStatus.FAIL, "empty root"
            )
        return IntegrityCheckResult(
            IntegrityCheck.SCHEMA_VALID.value, CheckStatus.PASS
        )
    except Exception as exc:  # noqa: BLE001
        return IntegrityCheckResult(
            IntegrityCheck.SCHEMA_VALID.value, CheckStatus.FAIL, str(exc)[:240]
        )


def _check_compiler(record: ExampleRecord) -> IntegrityCheckResult:
    try:
        encode_openui(record.openui)
        return IntegrityCheckResult(
            IntegrityCheck.COMPILER_VALID.value, CheckStatus.PASS
        )
    except Exception as exc:  # noqa: BLE001
        return IntegrityCheckResult(
            IntegrityCheck.COMPILER_VALID.value, CheckStatus.FAIL, str(exc)[:240]
        )


def _canonical_hash(record: ExampleRecord) -> tuple[IntegrityCheckResult, str]:
    try:
        fp = canonical_fingerprint(record.openui)
        return (
            IntegrityCheckResult(
                IntegrityCheck.AST_CANONICAL_HASH.value, CheckStatus.PASS
            ),
            fp,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            IntegrityCheckResult(
                IntegrityCheck.AST_CANONICAL_HASH.value,
                CheckStatus.FAIL,
                str(exc)[:240],
            ),
            "",
        )


def _surface_reparse_hash(record: ExampleRecord) -> tuple[IntegrityCheckResult, str]:
    try:
        program = validate(record.openui)
        reparsed = validate(program.serialized)
        fp = canonical_fingerprint(reparsed.serialized)
        return (
            IntegrityCheckResult(
                IntegrityCheck.SURFACE_REPARSE_AST_HASH.value, CheckStatus.PASS
            ),
            fp,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            IntegrityCheckResult(
                IntegrityCheck.SURFACE_REPARSE_AST_HASH.value,
                CheckStatus.FAIL,
                str(exc)[:240],
            ),
            "",
        )


def _production_roundtrip(
    record: ExampleRecord,
) -> tuple[IntegrityCheckResult, str]:
    try:
        contract = canonical_slot_contract(
            record.openui, declared=tuple(record.placeholders)
        )
        program = encode_openui(record.openui, slot_contract=contract)
        decoded = decode_productions(program.tokens, program.slot_contract)
        reencoded = encode_openui(decoded, slot_contract=program.slot_contract)
        if reencoded.tokens != program.tokens:
            return (
                IntegrityCheckResult(
                    IntegrityCheck.PRODUCTION_CODEC_ROUNDTRIP_HASH.value,
                    CheckStatus.FAIL,
                    "production codec not idempotent",
                ),
                "",
            )
        fp = _hash_text(" ".join(program.tokens))
        return (
            IntegrityCheckResult(
                IntegrityCheck.PRODUCTION_CODEC_ROUNDTRIP_HASH.value, CheckStatus.PASS
            ),
            fp,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            IntegrityCheckResult(
                IntegrityCheck.PRODUCTION_CODEC_ROUNDTRIP_HASH.value,
                CheckStatus.FAIL,
                str(exc)[:240],
            ),
            "",
        )


def _choice_roundtrip(record: ExampleRecord) -> tuple[IntegrityCheckResult, str]:
    try:
        contract = canonical_slot_contract(
            record.openui, declared=tuple(record.placeholders)
        )
        program = encode_choices(record.openui, slot_contract=contract)
        decoded = decode_choices(program.tokens, program.slot_contract)
        reencoded = encode_choices(decoded, slot_contract=program.slot_contract)
        if reencoded.tokens != program.tokens:
            return (
                IntegrityCheckResult(
                    IntegrityCheck.CHOICE_CODEC_ROUNDTRIP_HASH.value,
                    CheckStatus.FAIL,
                    "choice codec not idempotent",
                ),
                "",
            )
        fp = _hash_text(" ".join(program.tokens))
        return (
            IntegrityCheckResult(
                IntegrityCheck.CHOICE_CODEC_ROUNDTRIP_HASH.value, CheckStatus.PASS
            ),
            fp,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            IntegrityCheckResult(
                IntegrityCheck.CHOICE_CODEC_ROUNDTRIP_HASH.value,
                CheckStatus.FAIL,
                str(exc)[:240],
            ),
            "",
        )


def _slot_contract_hash(record: ExampleRecord) -> tuple[IntegrityCheckResult, str]:
    try:
        contract = canonical_slot_contract(
            record.openui, declared=tuple(record.placeholders)
        )
        fp = _hash_text(" ".join(contract))
        return (
            IntegrityCheckResult(
                IntegrityCheck.SLOT_CONTRACT_HASH.value, CheckStatus.PASS
            ),
            fp,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            IntegrityCheckResult(
                IntegrityCheck.SLOT_CONTRACT_HASH.value,
                CheckStatus.FAIL,
                str(exc)[:240],
            ),
            "",
        )


def _binding_graph_hash(record: ExampleRecord) -> tuple[IntegrityCheckResult, str]:
    try:
        program = validate(record.openui)
        fp = _hash_text(str(program.meta.get("unresolved")) + str(program.meta.get("orphaned")))
        return (
            IntegrityCheckResult(
                IntegrityCheck.BINDING_GRAPH_HASH.value, CheckStatus.PASS
            ),
            fp,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            IntegrityCheckResult(
                IntegrityCheck.BINDING_GRAPH_HASH.value,
                CheckStatus.FAIL,
                str(exc)[:240],
            ),
            "",
        )


def _placeholder_set_match(record: ExampleRecord) -> IntegrityCheckResult:
    try:
        contract = canonical_slot_contract(
            record.openui, declared=tuple(record.placeholders)
        )
        present = set(extract_placeholders(record.openui))
        expected = set(contract)
        missing = expected - present
        unexpected = present - expected
        if missing or unexpected:
            return IntegrityCheckResult(
                IntegrityCheck.PLACEHOLDER_SET_MATCH.value,
                CheckStatus.FAIL,
                f"missing={sorted(missing)} unexpected={sorted(unexpected)}"[:240],
            )
        return IntegrityCheckResult(
            IntegrityCheck.PLACEHOLDER_SET_MATCH.value, CheckStatus.PASS
        )
    except Exception as exc:  # noqa: BLE001
        return IntegrityCheckResult(
            IntegrityCheck.PLACEHOLDER_SET_MATCH.value,
            CheckStatus.FAIL,
            str(exc)[:240],
        )


def _reference_scope(record: ExampleRecord) -> IntegrityCheckResult:
    result = evaluate_gate(Gate.REFERENCES, record)
    if result.status.value == "pass":
        return IntegrityCheckResult(
            IntegrityCheck.REFERENCE_SCOPE_VALID.value, CheckStatus.PASS
        )
    return IntegrityCheckResult(
        IntegrityCheck.REFERENCE_SCOPE_VALID.value,
        CheckStatus.FAIL,
        result.detail[:240],
    )


def _request_target_contract(
    record: ExampleRecord, request: GenerationRequest | None
) -> IntegrityCheckResult:
    if request is None:
        return IntegrityCheckResult(
            IntegrityCheck.REQUEST_TARGET_CONTRACT_MATCH.value, CheckStatus.SKIP
        )
    try:
        contract = canonical_slot_contract(
            record.openui, declared=tuple(record.placeholders)
        )
        present = set(extract_placeholders(record.openui))
        expected = set(request.slot_contract)
        missing = expected - present
        unexpected = present - expected
        if missing or unexpected:
            return IntegrityCheckResult(
                IntegrityCheck.REQUEST_TARGET_CONTRACT_MATCH.value,
                CheckStatus.FAIL,
                f"missing={sorted(missing)} unexpected={sorted(unexpected)}"[:240],
            )
        if request.slot_contract and set(contract) != expected:
            return IntegrityCheckResult(
                IntegrityCheck.REQUEST_TARGET_CONTRACT_MATCH.value,
                CheckStatus.FAIL,
                f"slot_contract mismatch: record={sorted(contract)} request={sorted(expected)}"[
                    :240
                ],
            )
        return IntegrityCheckResult(
            IntegrityCheck.REQUEST_TARGET_CONTRACT_MATCH.value, CheckStatus.PASS
        )
    except Exception as exc:  # noqa: BLE001
        return IntegrityCheckResult(
            IntegrityCheck.REQUEST_TARGET_CONTRACT_MATCH.value,
            CheckStatus.UNKNOWN,
            str(exc)[:240],
        )


def _root_lineage_hash(record: ExampleRecord) -> tuple[IntegrityCheckResult, str]:
    parent_id = record.meta.get("root_parent_id") or record.meta.get("parent_id")
    lineage = record.meta.get("transformation_lineage") or []
    fp = _hash_text(
        f"{parent_id or record.id}|{json.dumps(lineage, sort_keys=True, default=str)}"
    )
    return (
        IntegrityCheckResult(
            IntegrityCheck.ROOT_LINEAGE_HASH.value, CheckStatus.PASS
        ),
        fp,
    )


def _split_leakage(
    record: ExampleRecord,
    held_out_fingerprints: Iterable[str] | None,
) -> IntegrityCheckResult:
    if held_out_fingerprints is None:
        return IntegrityCheckResult(
            IntegrityCheck.SPLIT_LEAKAGE_STATUS.value, CheckStatus.SKIP
        )
    try:
        struct_fp = fingerprint_openui_structure(record.openui)
        exact_fp = fingerprint_openui(record.openui)
        if struct_fp in held_out_fingerprints or exact_fp in held_out_fingerprints:
            return IntegrityCheckResult(
                IntegrityCheck.SPLIT_LEAKAGE_STATUS.value,
                CheckStatus.FAIL,
                "structure or exact fingerprint matches held-out set",
            )
        return IntegrityCheckResult(
            IntegrityCheck.SPLIT_LEAKAGE_STATUS.value, CheckStatus.PASS
        )
    except Exception as exc:  # noqa: BLE001
        return IntegrityCheckResult(
            IntegrityCheck.SPLIT_LEAKAGE_STATUS.value,
            CheckStatus.UNKNOWN,
            str(exc)[:240],
        )


# Hard-fail checks: integrity failures that should reject a record in enforce mode.
_HARD_FAIL_CHECKS = frozenset(
    {
        IntegrityCheck.PARSE_VALID.value,
        IntegrityCheck.SCHEMA_VALID.value,
        IntegrityCheck.COMPILER_VALID.value,
        IntegrityCheck.PRODUCTION_CODEC_ROUNDTRIP_HASH.value,
        IntegrityCheck.CHOICE_CODEC_ROUNDTRIP_HASH.value,
        IntegrityCheck.PLACEHOLDER_SET_MATCH.value,
        IntegrityCheck.REFERENCE_SCOPE_VALID.value,
        IntegrityCheck.REQUEST_TARGET_CONTRACT_MATCH.value,
        IntegrityCheck.SPLIT_LEAKAGE_STATUS.value,
    }
)


def evaluate_integrity(
    record: ExampleRecord,
    request: GenerationRequest | None = None,
    *,
    held_out_fingerprints: Iterable[str] | None = None,
) -> SyntheticIntegrityReport:
    """Run the full synthetic-integrity gate on one record."""
    checks: list[IntegrityCheckResult] = []
    hashes: dict[str, str] = {}

    check, hashes["ast_canonical"] = _canonical_hash(record)
    checks.append(check)

    check, hashes["surface_reparse"] = _surface_reparse_hash(record)
    checks.append(check)

    check, hashes["production_roundtrip"] = _production_roundtrip(record)
    checks.append(check)

    check, hashes["choice_roundtrip"] = _choice_roundtrip(record)
    checks.append(check)

    check, hashes["slot_contract"] = _slot_contract_hash(record)
    checks.append(check)

    check, hashes["binding_graph"] = _binding_graph_hash(record)
    checks.append(check)

    hashes["exact"] = fingerprint_openui(record.openui)
    hashes["structure"] = fingerprint_openui_structure(record.openui)
    checks.append(_check_parse(record))
    checks.append(_check_schema(record))
    checks.append(_check_compiler(record))
    checks.append(_placeholder_set_match(record))
    checks.append(_reference_scope(record))
    checks.append(_request_target_contract(record, request))

    check, hashes["root_lineage"] = _root_lineage_hash(record)
    checks.append(check)

    checks.append(_split_leakage(record, held_out_fingerprints))

    hard_fails = tuple(
        c.name
        for c in checks
        if c.name in _HARD_FAIL_CHECKS and c.status is CheckStatus.FAIL
    )

    return SyntheticIntegrityReport(
        schema_version=SCHEMA_VERSION,
        record_id=record.id,
        checks=tuple(checks),
        hashes=hashes,
        hard_fail_reasons=hard_fails,
    )


__all__ = [
    "CheckStatus",
    "IntegrityCheck",
    "IntegrityCheckResult",
    "SCHEMA_VERSION",
    "SyntheticIntegrityReport",
    "evaluate_integrity",
]
