"""Bounded exact operator legal-set enumeration and reserved serialization."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Mapping

from slm_training.dsl.operators.contracts import (
    ApplicationProvenanceV1,
    BoundArgumentV1,
    IndexRef,
    NodeRef,
    OperatorRef,
    RefKind,
    RoleRef,
    SymbolRef,
    TemplateRef,
    ValueRef,
    _fingerprint,
    _require_digest,
    _require_identifier,
)
from slm_training.dsl.operators.references import ReferenceTableV1
from slm_training.dsl.operators.registry import OperatorLibraryV1, OperatorStateV1
from slm_training.dsl.pack import DslPack

_REF_TYPES = {
    RefKind.NODE: NodeRef,
    RefKind.ROLE: RoleRef,
    RefKind.INDEX: IndexRef,
    RefKind.VALUE: ValueRef,
    RefKind.SYMBOL: SymbolRef,
    RefKind.TEMPLATE: TemplateRef,
}


class LegalSetCoverage(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"


class OperatorSupportVerdict(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OperatorArgumentDomainV1:
    slot_id: str
    ref_kind: RefKind
    candidates: tuple[OperatorRef, ...]
    required: bool
    repeated: bool
    complete: bool = True
    schema: str = "operator_argument_domain/v1"

    def __post_init__(self) -> None:
        _require_identifier(self.slot_id, "slot_id")
        if any(candidate.KIND is not self.ref_kind for candidate in self.candidates):
            raise ValueError("argument-domain candidate has the wrong reference kind")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("argument-domain candidates must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "slot_id": self.slot_id,
            "ref_kind": self.ref_kind.value,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "required": self.required,
            "repeated": self.repeated,
            "complete": self.complete,
        }


@dataclass(frozen=True)
class LegalOperatorActionV1:
    operator_id: str
    operator_fingerprint: str
    arguments: tuple[BoundArgumentV1, ...]
    semantic_id: str
    application_id: str
    proof_fingerprint: str
    proof_checks: tuple[str, ...]
    serialized: str
    schema: str = "legal_operator_action/v1"

    def __post_init__(self) -> None:
        _require_identifier(self.operator_id, "operator_id")
        _require_digest(self.operator_fingerprint, "operator_fingerprint")
        _require_digest(self.semantic_id, "semantic_id")
        _require_digest(self.application_id, "application_id")
        _require_digest(self.proof_fingerprint, "proof_fingerprint")
        if not self.proof_checks:
            raise ValueError("legal operator action requires pack-authority proof checks")
        if self.serialized != serialize_operator_action(
            self.operator_id, self.arguments
        ):
            raise ValueError("legal operator serialization is not canonical")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "operator_id": self.operator_id,
            "operator_fingerprint": self.operator_fingerprint,
            "arguments": [argument.to_dict() for argument in self.arguments],
            "semantic_id": self.semantic_id,
            "application_id": self.application_id,
            "proof_fingerprint": self.proof_fingerprint,
            "proof_checks": list(self.proof_checks),
            "serialized": self.serialized,
        }


@dataclass(frozen=True)
class OperatorLegalEntryV1:
    operator_id: str
    operator_fingerprint: str
    argument_domains: tuple[OperatorArgumentDomainV1, ...]
    legal_actions: tuple[LegalOperatorActionV1, ...]
    verdict: OperatorSupportVerdict
    coverage: LegalSetCoverage
    evaluated_combinations: int
    total_combinations: int
    rejection_counts: tuple[tuple[str, int], ...]
    schema: str = "operator_legal_entry/v1"

    def __post_init__(self) -> None:
        _require_identifier(self.operator_id, "operator_id")
        _require_digest(self.operator_fingerprint, "operator_fingerprint")
        if self.evaluated_combinations < 0 or self.total_combinations < 0:
            raise ValueError("combination counts must be non-negative")
        if self.verdict is OperatorSupportVerdict.UNSUPPORTED and (
            self.coverage is not LegalSetCoverage.COMPLETE or self.legal_actions
        ):
            raise ValueError("unsupported requires complete coverage and zero actions")
        if self.verdict is OperatorSupportVerdict.SUPPORTED and not self.legal_actions:
            raise ValueError("supported requires at least one legal action")
        if self.evaluated_combinations > self.total_combinations:
            raise ValueError("evaluated combinations exceed the declared product")
        if any(action.operator_id != self.operator_id for action in self.legal_actions):
            raise ValueError("legal action belongs to another operator")
        if len({code for code, _ in self.rejection_counts}) != len(
            self.rejection_counts
        ):
            raise ValueError("rejection reason codes must be unique")
        if any(count <= 0 for _, count in self.rejection_counts):
            raise ValueError("rejection reason counts must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "operator_id": self.operator_id,
            "operator_fingerprint": self.operator_fingerprint,
            "argument_domains": [domain.to_dict() for domain in self.argument_domains],
            "legal_actions": [action.to_dict() for action in self.legal_actions],
            "verdict": self.verdict.value,
            "coverage": self.coverage.value,
            "evaluated_combinations": self.evaluated_combinations,
            "total_combinations": self.total_combinations,
            "rejection_counts": dict(self.rejection_counts),
        }


@dataclass(frozen=True)
class OperatorLegalSetV1:
    state_fingerprint: str
    reference_table_fingerprint: str
    registry_fingerprint: str
    entries: tuple[OperatorLegalEntryV1, ...]
    ordinary_nonoperator_actions: tuple[str, ...]
    coverage: LegalSetCoverage
    max_combinations_per_operator: int
    schema: str = "operator_legal_set/v1"

    def __post_init__(self) -> None:
        _require_digest(self.state_fingerprint, "state_fingerprint")
        _require_digest(
            self.reference_table_fingerprint, "reference_table_fingerprint"
        )
        _require_digest(self.registry_fingerprint, "registry_fingerprint")
        if self.max_combinations_per_operator <= 0:
            raise ValueError("max_combinations_per_operator must be positive")
        if len({entry.operator_id for entry in self.entries}) != len(self.entries):
            raise ValueError("operator legal-set entries must be unique")
        expected = (
            LegalSetCoverage.COMPLETE
            if all(
                entry.coverage is LegalSetCoverage.COMPLETE
                for entry in self.entries
            )
            else LegalSetCoverage.PARTIAL
        )
        if self.coverage is not expected:
            raise ValueError("legal-set coverage disagrees with its entries")

    @property
    def legal_operator_ids(self) -> tuple[str, ...]:
        return tuple(
            entry.operator_id
            for entry in self.entries
            if entry.verdict is OperatorSupportVerdict.SUPPORTED
        )

    @property
    def unknown_operator_ids(self) -> tuple[str, ...]:
        return tuple(
            entry.operator_id
            for entry in self.entries
            if entry.verdict is OperatorSupportVerdict.UNKNOWN
        )

    @property
    def hard_prunable_operator_ids(self) -> tuple[str, ...]:
        return tuple(
            entry.operator_id
            for entry in self.entries
            if entry.verdict is OperatorSupportVerdict.UNSUPPORTED
        )

    @property
    def retained_operator_ids(self) -> tuple[str, ...]:
        return tuple(
            entry.operator_id
            for entry in self.entries
            if entry.verdict is not OperatorSupportVerdict.UNSUPPORTED
        )

    @property
    def operator_actions(self) -> tuple[LegalOperatorActionV1, ...]:
        return tuple(
            action
            for entry in self.entries
            for action in entry.legal_actions
        )

    @property
    def all_serialized_actions(self) -> tuple[str, ...]:
        return (
            *self.ordinary_nonoperator_actions,
            *(action.serialized for action in self.operator_actions),
        )

    @property
    def forced_action(self) -> str | None:
        actions = self.all_serialized_actions
        if self.coverage is LegalSetCoverage.COMPLETE and len(actions) == 1:
            return actions[0]
        return None

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "state_fingerprint": self.state_fingerprint,
            "reference_table_fingerprint": self.reference_table_fingerprint,
            "registry_fingerprint": self.registry_fingerprint,
            "entries": [entry.to_dict() for entry in self.entries],
            "ordinary_nonoperator_actions": list(self.ordinary_nonoperator_actions),
            "coverage": self.coverage.value,
            "max_combinations_per_operator": self.max_combinations_per_operator,
        }


def _semantic_action_id(
    operator_fingerprint: str,
    arguments: tuple[BoundArgumentV1, ...],
    descriptor_by_ref: Mapping[OperatorRef, str],
) -> str:
    return _fingerprint(
        {
            "schema": "semantic_operator_action/v1",
            "operator_fingerprint": operator_fingerprint,
            "arguments": [
                {
                    "slot_id": argument.slot_id,
                    "ref_kind": argument.value.KIND.value,
                    "semantic_fingerprint": descriptor_by_ref[argument.value],
                }
                for argument in arguments
            ],
        }
    )


def serialize_operator_action(
    operator_id: str, arguments: tuple[BoundArgumentV1, ...]
) -> str:
    """Canonical reserved baseline form: ``OPERATOR <id> <typed args>``."""
    _require_identifier(operator_id, "operator_id")
    fields = ["OPERATOR", operator_id]
    for argument in arguments:
        ref = argument.value
        fields.append(
            f"{argument.slot_id}={ref.KIND.value}:{ref.request_id}:{ref.opaque_id}"
        )
    return " ".join(fields)


def deserialize_operator_action(value: str) -> tuple[str, tuple[BoundArgumentV1, ...]]:
    fields = value.split(" ")
    if len(fields) < 2 or fields[0] != "OPERATOR" or any(not field for field in fields):
        raise ValueError("invalid reserved operator serialization")
    operator_id = fields[1]
    _require_identifier(operator_id, "operator_id")
    arguments: list[BoundArgumentV1] = []
    for field in fields[2:]:
        try:
            slot_id, encoded = field.split("=", 1)
            kind_value, request_id, opaque_id = encoded.split(":", 2)
            kind = RefKind(kind_value)
            ref = _REF_TYPES[kind](request_id, opaque_id)
            arguments.append(BoundArgumentV1(slot_id, ref))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid typed operator argument serialization") from exc
    if len({argument.slot_id for argument in arguments}) != len(arguments):
        raise ValueError("serialized operator argument slots must be unique")
    return operator_id, tuple(arguments)


def _domains(
    library: OperatorLibraryV1,
    reference_table: ReferenceTableV1,
    operator_id: str,
) -> tuple[OperatorArgumentDomainV1, ...]:
    declaration = library.lookup(operator_id)
    by_kind: dict[RefKind, list[tuple[str, OperatorRef]]] = {}
    for entry in reference_table.entries:
        by_kind.setdefault(entry.descriptor.ref_kind, []).append(
            (entry.descriptor.fingerprint, entry.ref)
        )
    for values in by_kind.values():
        values.sort(key=lambda item: item[0])
    return tuple(
        OperatorArgumentDomainV1(
            slot_id=slot.slot_id,
            ref_kind=slot.ref_kind,
            candidates=tuple(ref for _, ref in by_kind.get(slot.ref_kind, ())),
            required=slot.required,
            repeated=slot.repeated,
            complete=not slot.repeated,
        )
        for slot in declaration.argument_slots
    )


def iter_operator_argument_tuples(
    domains: tuple[OperatorArgumentDomainV1, ...],
) -> Iterator[tuple[BoundArgumentV1, ...]]:
    """Lazily walk one declaration-ordered typed Cartesian product."""
    if any(domain.repeated for domain in domains):
        raise ValueError("repeated operator slots require an explicit finite arity")
    choices: list[tuple[OperatorRef | None, ...]] = []
    for domain in domains:
        values: tuple[OperatorRef | None, ...] = domain.candidates
        if not domain.required:
            values = (None, *values)
        choices.append(values)
    if any(not values for values in choices):
        return
    for combination in itertools.product(*choices):
        yield tuple(
            BoundArgumentV1(domain.slot_id, ref)
            for domain, ref in zip(domains, combination, strict=True)
            if ref is not None
        )


def _combination_count(domains: tuple[OperatorArgumentDomainV1, ...]) -> int:
    if any(domain.repeated for domain in domains):
        return 0
    return math.prod(
        len(domain.candidates) + (0 if domain.required else 1)
        for domain in domains
    )


def enumerate_operator_legal_set(
    *,
    pack: DslPack,
    library: OperatorLibraryV1,
    state: OperatorStateV1,
    reference_table: ReferenceTableV1,
    provenance: ApplicationProvenanceV1,
    ordinary_nonoperator_actions: tuple[str, ...] = (),
    max_combinations_per_operator: int = 10_000,
) -> OperatorLegalSetV1:
    """Dry-run exact tuples; budget truncation stays UNKNOWN, never unsupported."""
    if max_combinations_per_operator <= 0:
        raise ValueError("max_combinations_per_operator must be positive")
    if reference_table.state_digest != state.state_digest:
        raise ValueError("reference table is stale for operator state")
    if reference_table.request_id != provenance.request_id:
        raise ValueError("reference table and provenance request differ")
    if pack.require("operator_library") is not library:
        raise ValueError("operator library is not owned by the pack")

    descriptor_by_ref = {
        entry.ref: entry.descriptor.fingerprint
        for entry in reference_table.entries
    }
    entries: list[OperatorLegalEntryV1] = []
    for declaration in library.declarations:
        domains = _domains(library, reference_table, declaration.operator_id)
        total = _combination_count(domains)
        if any(domain.repeated for domain in domains):
            entries.append(
                OperatorLegalEntryV1(
                    operator_id=declaration.operator_id,
                    operator_fingerprint=declaration.fingerprint,
                    argument_domains=domains,
                    legal_actions=(),
                    verdict=OperatorSupportVerdict.UNKNOWN,
                    coverage=LegalSetCoverage.PARTIAL,
                    evaluated_combinations=0,
                    total_combinations=0,
                    rejection_counts=(("operator.repeated_slot_unbounded", 1),),
                )
            )
            continue
        budget = min(total, max_combinations_per_operator)
        actions: list[LegalOperatorActionV1] = []
        rejection_counts: dict[str, int] = {}
        evaluated = 0
        for arguments in itertools.islice(
            iter_operator_argument_tuples(domains), budget
        ):
            evaluated += 1
            application = library.dry_run(
                pack,
                state,
                declaration.operator_id,
                arguments,
                provenance,
            )
            if application.succeeded:
                assert application.proof is not None
                semantic_id = _semantic_action_id(
                    declaration.fingerprint, arguments, descriptor_by_ref
                )
                actions.append(
                    LegalOperatorActionV1(
                        operator_id=declaration.operator_id,
                        operator_fingerprint=declaration.fingerprint,
                        arguments=arguments,
                        semantic_id=semantic_id,
                        application_id=application.application_id,
                        proof_fingerprint=_fingerprint(
                            application.proof.to_dict()
                        ),
                        proof_checks=tuple(sorted(set(application.proof.checks))),
                        serialized=serialize_operator_action(
                            declaration.operator_id, arguments
                        ),
                    )
                )
            else:
                assert application.rejection is not None
                code = application.rejection.code
                rejection_counts[code] = rejection_counts.get(code, 0) + 1
        coverage = (
            LegalSetCoverage.COMPLETE
            if evaluated == total
            else LegalSetCoverage.PARTIAL
        )
        actions.sort(key=lambda action: action.semantic_id)
        if actions:
            verdict = OperatorSupportVerdict.SUPPORTED
        elif coverage is LegalSetCoverage.COMPLETE:
            verdict = OperatorSupportVerdict.UNSUPPORTED
        else:
            verdict = OperatorSupportVerdict.UNKNOWN
        entries.append(
            OperatorLegalEntryV1(
                operator_id=declaration.operator_id,
                operator_fingerprint=declaration.fingerprint,
                argument_domains=domains,
                legal_actions=tuple(actions),
                verdict=verdict,
                coverage=coverage,
                evaluated_combinations=evaluated,
                total_combinations=total,
                rejection_counts=tuple(sorted(rejection_counts.items())),
            )
        )
    entries.sort(key=lambda entry: entry.operator_id)
    coverage = (
        LegalSetCoverage.COMPLETE
        if all(entry.coverage is LegalSetCoverage.COMPLETE for entry in entries)
        else LegalSetCoverage.PARTIAL
    )
    return OperatorLegalSetV1(
        state_fingerprint=_fingerprint(
            {
                "schema": "operator_legal_state/v1",
                "state_digest": state.state_digest,
                "ast_digest": state.ast_digest,
            }
        ),
        reference_table_fingerprint=reference_table.fingerprint,
        registry_fingerprint=library.registry_fingerprint,
        entries=tuple(entries),
        ordinary_nonoperator_actions=ordinary_nonoperator_actions,
        coverage=coverage,
        max_combinations_per_operator=max_combinations_per_operator,
    )


__all__ = [
    "LegalOperatorActionV1",
    "LegalSetCoverage",
    "OperatorArgumentDomainV1",
    "OperatorLegalEntryV1",
    "OperatorLegalSetV1",
    "OperatorSupportVerdict",
    "deserialize_operator_action",
    "enumerate_operator_legal_set",
    "iter_operator_argument_tuples",
    "serialize_operator_action",
]
