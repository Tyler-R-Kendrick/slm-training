"""Versioned contracts for compiler-owned AST operators (DSH3-01).

The records in this module define identity, typed arguments, declared effects,
and application evidence. They deliberately do not execute operators: a DSL
pack owns legality and application in the next layer. No display name, token
ID, or learned representation participates in any fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Mapping, TypeAlias

_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.:-]*$")
_VERSION_RE = re.compile(r"^v[1-9][0-9]*$")
_OPAQUE_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_identifier(value: str, field: str) -> None:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{field} must be a stable lowercase identifier")


def _require_digest(value: str, field: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase sha256 digest")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class RefKind(str, Enum):
    NODE = "node"
    ROLE = "role"
    INDEX = "index"
    VALUE = "value"
    SYMBOL = "symbol"
    TEMPLATE = "template"


class BindingPhase(str, Enum):
    """The compiler phase at which a typed argument must resolve."""

    REQUEST = "request"
    STATE = "state"
    APPLICATION = "application"


class CompilerCoverage(str, Enum):
    EXACT = "exact"
    BOUNDED = "bounded"
    APPROXIMATE = "approximate"


class EffectDeltaKind(str, Enum):
    SCOPE = "scope"
    CARDINALITY = "cardinality"
    PROPERTY = "property"
    TOPOLOGY = "topology"


@dataclass(frozen=True)
class _OpaqueRef:
    """Opaque request-local surface resolved only by the compiler."""

    request_id: str
    opaque_id: str

    KIND: ClassVar[RefKind]

    def __post_init__(self) -> None:
        if not _OPAQUE_RE.fullmatch(self.request_id):
            raise ValueError("request_id must be opaque and contain no whitespace")
        if not _OPAQUE_RE.fullmatch(self.opaque_id):
            raise ValueError("opaque_id must be opaque and contain no whitespace")

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.KIND.value,
            "request_id": self.request_id,
            "opaque_id": self.opaque_id,
        }


@dataclass(frozen=True)
class NodeRef(_OpaqueRef):
    KIND: ClassVar[RefKind] = RefKind.NODE


@dataclass(frozen=True)
class RoleRef(_OpaqueRef):
    KIND: ClassVar[RefKind] = RefKind.ROLE


@dataclass(frozen=True)
class IndexRef(_OpaqueRef):
    KIND: ClassVar[RefKind] = RefKind.INDEX


@dataclass(frozen=True)
class ValueRef(_OpaqueRef):
    KIND: ClassVar[RefKind] = RefKind.VALUE


@dataclass(frozen=True)
class SymbolRef(_OpaqueRef):
    KIND: ClassVar[RefKind] = RefKind.SYMBOL


@dataclass(frozen=True)
class TemplateRef(_OpaqueRef):
    KIND: ClassVar[RefKind] = RefKind.TEMPLATE


OperatorRef: TypeAlias = (
    NodeRef | RoleRef | IndexRef | ValueRef | SymbolRef | TemplateRef
)


@dataclass(frozen=True)
class OperatorArgumentSlotV1:
    slot_id: str
    ref_kind: RefKind
    binding_phase: BindingPhase
    required: bool = True
    repeated: bool = False

    def __post_init__(self) -> None:
        _require_identifier(self.slot_id, "slot_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "ref_kind": self.ref_kind.value,
            "binding_phase": self.binding_phase.value,
            "required": self.required,
            "repeated": self.repeated,
        }


@dataclass(frozen=True)
class PreconditionV1:
    predicate_id: str
    argument_slots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.predicate_id, "predicate_id")
        for slot_id in self.argument_slots:
            _require_identifier(slot_id, "argument slot")
        if len(set(self.argument_slots)) != len(self.argument_slots):
            raise ValueError("precondition argument slots must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate_id": self.predicate_id,
            "argument_slots": sorted(self.argument_slots),
        }


@dataclass(frozen=True)
class AstOperatorV1:
    """A declarative operator identity independent of model representation."""

    operator_id: str
    version: str
    domain: str
    codomain: str
    argument_slots: tuple[OperatorArgumentSlotV1, ...]
    preconditions: tuple[PreconditionV1, ...]
    effect_signature: tuple[EffectDeltaKind, ...]
    locality: str
    cost: float
    inverse_operator_id: str | None = None
    commutes_with: tuple[str, ...] = ()
    idempotent: bool = False
    schema: str = "ast_operator/v1"

    def __post_init__(self) -> None:
        _require_identifier(self.operator_id, "operator_id")
        if not _VERSION_RE.fullmatch(self.version):
            raise ValueError("version must use monotonic vN form")
        _require_identifier(self.domain, "domain")
        _require_identifier(self.codomain, "codomain")
        _require_identifier(self.locality, "locality")
        if self.inverse_operator_id is not None:
            _require_identifier(self.inverse_operator_id, "inverse_operator_id")
        for operator_id in self.commutes_with:
            _require_identifier(operator_id, "commutes_with")
        if not math.isfinite(self.cost) or self.cost < 0:
            raise ValueError("cost must be finite and non-negative")
        slot_ids = tuple(slot.slot_id for slot in self.argument_slots)
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("argument slot ids must be unique")
        unknown = {
            slot_id
            for condition in self.preconditions
            for slot_id in condition.argument_slots
            if slot_id not in slot_ids
        }
        if unknown:
            raise ValueError(
                f"preconditions reference unknown slots: {sorted(unknown)}"
            )
        if len(set(self.effect_signature)) != len(self.effect_signature):
            raise ValueError("effect_signature entries must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "operator_id": self.operator_id,
            "version": self.version,
            "domain": self.domain,
            "codomain": self.codomain,
            "argument_slots": [slot.to_dict() for slot in self.argument_slots],
            "preconditions": sorted(
                (condition.to_dict() for condition in self.preconditions),
                key=_canonical_json,
            ),
            "effect_signature": sorted(kind.value for kind in self.effect_signature),
            "inverse_operator_id": self.inverse_operator_id,
            "commutes_with": sorted(set(self.commutes_with)),
            "idempotent": self.idempotent,
            "locality": self.locality,
            "cost": self.cost,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def validate_arguments(
        self, arguments: tuple[BoundArgumentV1, ...]
    ) -> tuple[BoundArgumentV1, ...]:
        """Validate and declaration-order one compiler argument binding."""
        by_slot = {argument.slot_id: argument for argument in arguments}
        if len(by_slot) != len(arguments):
            raise ValueError("bound argument slots must be unique")
        slots = {slot.slot_id: slot for slot in self.argument_slots}
        unknown = set(by_slot) - set(slots)
        if unknown:
            raise ValueError(f"unknown bound argument slots: {sorted(unknown)}")
        missing = {
            slot.slot_id
            for slot in self.argument_slots
            if slot.required and slot.slot_id not in by_slot
        }
        if missing:
            raise ValueError(f"missing required argument slots: {sorted(missing)}")
        for slot_id, argument in by_slot.items():
            expected = slots[slot_id].ref_kind
            if argument.value.KIND is not expected:
                raise ValueError(
                    f"slot {slot_id!r} requires {expected.value}, "
                    f"got {argument.value.KIND.value}"
                )
        return tuple(
            by_slot[slot.slot_id]
            for slot in self.argument_slots
            if slot.slot_id in by_slot
        )


@dataclass(frozen=True)
class EffectDeltaV1:
    """One typed before/after delta over an opaque compiler target."""

    kind: EffectDeltaKind
    target: OperatorRef
    before: Any
    after: Any

    def __post_init__(self) -> None:
        _canonical_json(self.before)
        _canonical_json(self.after)
        if self.before == self.after:
            raise ValueError("an effect delta must change its target")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "target": self.target.to_dict(),
            "before": self.before,
            "after": self.after,
        }


@dataclass(frozen=True)
class ActionEffectV1:
    consumed_roles: tuple[RoleRef, ...] = ()
    produced_roles: tuple[RoleRef, ...] = ()
    consumed_binders: tuple[SymbolRef, ...] = ()
    produced_binders: tuple[SymbolRef, ...] = ()
    scope_deltas: tuple[EffectDeltaV1, ...] = ()
    cardinality_deltas: tuple[EffectDeltaV1, ...] = ()
    property_deltas: tuple[EffectDeltaV1, ...] = ()
    topology_deltas: tuple[EffectDeltaV1, ...] = ()
    compiler_coverage: CompilerCoverage = CompilerCoverage.EXACT
    estimated_completion_cost: float = 0.0
    schema: str = "action_effect/v1"

    def __post_init__(self) -> None:
        expected = (
            (self.scope_deltas, EffectDeltaKind.SCOPE),
            (self.cardinality_deltas, EffectDeltaKind.CARDINALITY),
            (self.property_deltas, EffectDeltaKind.PROPERTY),
            (self.topology_deltas, EffectDeltaKind.TOPOLOGY),
        )
        for deltas, kind in expected:
            if any(delta.kind is not kind for delta in deltas):
                raise ValueError(f"{kind.value}_deltas contains a mismatched kind")
        if (
            not math.isfinite(self.estimated_completion_cost)
            or self.estimated_completion_cost < 0
        ):
            raise ValueError(
                "estimated_completion_cost must be finite and non-negative"
            )
        if len(self.request_ids) > 1:
            raise ValueError("all effect references must belong to one request")

    @property
    def request_ids(self) -> frozenset[str]:
        refs: tuple[_OpaqueRef, ...] = (
            *self.consumed_roles,
            *self.produced_roles,
            *self.consumed_binders,
            *self.produced_binders,
            *(delta.target for delta in self.scope_deltas),
            *(delta.target for delta in self.cardinality_deltas),
            *(delta.target for delta in self.property_deltas),
            *(delta.target for delta in self.topology_deltas),
        )
        return frozenset(ref.request_id for ref in refs)

    def to_dict(self) -> dict[str, Any]:
        def refs(values: tuple[_OpaqueRef, ...]) -> list[dict[str, str]]:
            return sorted((value.to_dict() for value in values), key=_canonical_json)

        def deltas(values: tuple[EffectDeltaV1, ...]) -> list[dict[str, Any]]:
            return sorted((value.to_dict() for value in values), key=_canonical_json)

        return {
            "schema": self.schema,
            "consumed_roles": refs(self.consumed_roles),
            "produced_roles": refs(self.produced_roles),
            "consumed_binders": refs(self.consumed_binders),
            "produced_binders": refs(self.produced_binders),
            "scope_deltas": deltas(self.scope_deltas),
            "cardinality_deltas": deltas(self.cardinality_deltas),
            "property_deltas": deltas(self.property_deltas),
            "topology_deltas": deltas(self.topology_deltas),
            "compiler_coverage": self.compiler_coverage.value,
            "estimated_completion_cost": self.estimated_completion_cost,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True)
class BoundArgumentV1:
    slot_id: str
    value: OperatorRef

    def __post_init__(self) -> None:
        _require_identifier(self.slot_id, "slot_id")

    def to_dict(self) -> dict[str, Any]:
        return {"slot_id": self.slot_id, "value": self.value.to_dict()}


@dataclass(frozen=True)
class ApplicationProvenanceV1:
    pack_id: str
    compiler_id: str
    compiler_version: str
    source_artifact_digest: str
    request_id: str

    def __post_init__(self) -> None:
        _require_identifier(self.pack_id, "pack_id")
        _require_identifier(self.compiler_id, "compiler_id")
        if not self.compiler_version:
            raise ValueError("compiler_version is required")
        _require_digest(self.source_artifact_digest, "source_artifact_digest")
        if not _OPAQUE_RE.fullmatch(self.request_id):
            raise ValueError("request_id must be opaque and contain no whitespace")

    def to_dict(self) -> dict[str, str]:
        return {
            "pack_id": self.pack_id,
            "compiler_id": self.compiler_id,
            "compiler_version": self.compiler_version,
            "source_artifact_digest": self.source_artifact_digest,
            "request_id": self.request_id,
        }


@dataclass(frozen=True)
class ApplicationProofV1:
    proof_kind: str
    checks: tuple[str, ...]
    compiler_result_digest: str
    effect_fingerprint: str

    def __post_init__(self) -> None:
        _require_identifier(self.proof_kind, "proof_kind")
        if not self.checks:
            raise ValueError("a successful proof requires at least one check")
        for check in self.checks:
            _require_identifier(check, "proof check")
        _require_digest(self.compiler_result_digest, "compiler_result_digest")
        _require_digest(self.effect_fingerprint, "effect_fingerprint")

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof_kind": self.proof_kind,
            "checks": sorted(set(self.checks)),
            "compiler_result_digest": self.compiler_result_digest,
            "effect_fingerprint": self.effect_fingerprint,
        }


@dataclass(frozen=True)
class OperatorRejectionV1:
    code: str
    failed_precondition: str | None
    compiler_result_digest: str

    def __post_init__(self) -> None:
        _require_identifier(self.code, "rejection code")
        if self.failed_precondition is not None:
            _require_identifier(self.failed_precondition, "failed_precondition")
        _require_digest(self.compiler_result_digest, "compiler_result_digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "failed_precondition": self.failed_precondition,
            "compiler_result_digest": self.compiler_result_digest,
        }


@dataclass(frozen=True)
class OperatorApplicationV1:
    """Deterministic success or rejection evidence for one compiler attempt."""

    operator_fingerprint: str
    arguments: tuple[BoundArgumentV1, ...]
    before_state_digest: str
    before_ast_digest: str
    provenance: ApplicationProvenanceV1
    effect: ActionEffectV1 | None = None
    after_state_digest: str | None = None
    after_ast_digest: str | None = None
    proof: ApplicationProofV1 | None = None
    rejection: OperatorRejectionV1 | None = None
    schema: str = "operator_application/v1"

    def __post_init__(self) -> None:
        _require_digest(self.operator_fingerprint, "operator_fingerprint")
        _require_digest(self.before_state_digest, "before_state_digest")
        _require_digest(self.before_ast_digest, "before_ast_digest")
        slot_ids = tuple(argument.slot_id for argument in self.arguments)
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("bound argument slots must be unique")
        succeeded = self.proof is not None
        if succeeded == (self.rejection is not None):
            raise ValueError("exactly one of proof or rejection is required")
        success_fields = (
            self.effect,
            self.after_state_digest,
            self.after_ast_digest,
        )
        if succeeded:
            if any(value is None for value in success_fields):
                raise ValueError(
                    "successful applications require effect and after digests"
                )
            assert self.effect is not None
            assert self.proof is not None
            if self.proof.effect_fingerprint != self.effect.fingerprint:
                raise ValueError("proof effect fingerprint does not match effect")
            _require_digest(self.after_state_digest or "", "after_state_digest")
            _require_digest(self.after_ast_digest or "", "after_ast_digest")
        elif any(value is not None for value in success_fields):
            raise ValueError(
                "rejected applications cannot claim effects or after state"
            )
        request_ids = {argument.value.request_id for argument in self.arguments} | {
            self.provenance.request_id
        }
        if self.effect is not None:
            request_ids.update(self.effect.request_ids)
        if len(request_ids) != 1:
            raise ValueError(
                "all opaque references must belong to the application request"
            )

    @property
    def succeeded(self) -> bool:
        return self.proof is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "operator_fingerprint": self.operator_fingerprint,
            "arguments": sorted(
                (argument.to_dict() for argument in self.arguments),
                key=lambda value: value["slot_id"],
            ),
            "before_state_digest": self.before_state_digest,
            "before_ast_digest": self.before_ast_digest,
            "after_state_digest": self.after_state_digest,
            "after_ast_digest": self.after_ast_digest,
            "effect": self.effect.to_dict() if self.effect is not None else None,
            "proof": self.proof.to_dict() if self.proof is not None else None,
            "rejection": (
                self.rejection.to_dict() if self.rejection is not None else None
            ),
            "provenance": self.provenance.to_dict(),
        }

    @property
    def application_id(self) -> str:
        return _fingerprint(self.to_dict())
