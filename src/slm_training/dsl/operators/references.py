"""Permutation-invariant, state-bound opaque operator references (DSH3-03)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from slm_training.dsl.operators.contracts import (
    IndexRef,
    NodeRef,
    OperatorRef,
    RefKind,
    RoleRef,
    SymbolRef,
    TemplateRef,
    ValueRef,
    _canonical_json,
    _fingerprint,
    _require_digest,
    _require_identifier,
)

_REF_CLASS = {
    RefKind.NODE: NodeRef,
    RefKind.ROLE: RoleRef,
    RefKind.INDEX: IndexRef,
    RefKind.VALUE: ValueRef,
    RefKind.SYMBOL: SymbolRef,
    RefKind.TEMPLATE: TemplateRef,
}


class ReferenceResolutionError(ValueError):
    """Stable, machine-readable opaque-reference resolution failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class CompilerFact(str, Enum):
    """Allowlisted inference-visible facts; no free-form descriptor channel."""

    NODE_VISIBLE = "node.visible"
    NODE_MUTABLE = "node.mutable"
    ROLE_AVAILABLE = "role.available"
    INDEX_ORDERED_PARENT = "index.ordered_parent"
    SYMBOL_IN_SCOPE = "symbol.in_scope"
    TEMPLATE_AVAILABLE = "template.available"
    VALUE_VISIBLE = "value.visible"


class RuntimeSymbolRole(str, Enum):
    ALPHA_BINDER = "alpha_binder"
    EXTERNAL_ENTITY = "external_entity"
    STATE = "state"
    FRESH_BINDER = "fresh_binder"


def branch_fingerprint(root_state_digest: str, branch_nonce_digest: str) -> str:
    """Derive an opaque branch identity without a branch display name."""
    _require_digest(root_state_digest, "root_state_digest")
    _require_digest(branch_nonce_digest, "branch_nonce_digest")
    return _fingerprint(
        {
            "schema": "operator_branch/v1",
            "root_state_digest": root_state_digest,
            "branch_nonce_digest": branch_nonce_digest,
        }
    )


def branch_local_disambiguator(
    branch_digest: str,
    structural_fingerprint: str,
    collision_index: int,
) -> str:
    """Hash an internal same-structure occurrence index; never expose it."""
    _require_digest(branch_digest, "branch_digest")
    _require_digest(structural_fingerprint, "structural_fingerprint")
    if collision_index < 0:
        raise ValueError("collision_index must be non-negative")
    return _fingerprint(
        {
            "schema": "branch_local_disambiguator/v1",
            "branch_digest": branch_digest,
            "structural_fingerprint": structural_fingerprint,
            "collision_index": collision_index,
        }
    )


def persistent_node_fingerprint(
    canonical_structure: Mapping[str, Any],
    *,
    parent_fingerprint: str | None,
    branch_disambiguator: str,
) -> str:
    """Hash canonical structural identity plus an opaque branch-local tie break."""
    if parent_fingerprint is not None:
        _require_digest(parent_fingerprint, "parent_fingerprint")
    _require_digest(branch_disambiguator, "branch_disambiguator")
    _canonical_json(canonical_structure)
    return _fingerprint(
        {
            "schema": "persistent_node_fingerprint/v1",
            "canonical_structure": canonical_structure,
            "parent_fingerprint": parent_fingerprint,
            "branch_disambiguator": branch_disambiguator,
        }
    )


def ordered_parent_digest(
    parent_fingerprint: str, child_fingerprints: tuple[str, ...]
) -> str:
    """Bind an index argument to one exact current parent ordering."""
    _require_digest(parent_fingerprint, "parent_fingerprint")
    for child in child_fingerprints:
        _require_digest(child, "child_fingerprint")
    return _fingerprint(
        {
            "schema": "ordered_parent/v1",
            "parent_fingerprint": parent_fingerprint,
            "child_fingerprints": child_fingerprints,
        }
    )


@dataclass(frozen=True)
class ReferenceDescriptorV1:
    """Inference-visible compiler facts for one semantic reference."""

    ref_kind: RefKind
    semantic_fingerprint: str
    value_type: str
    compiler_facts: tuple[CompilerFact, ...] = ()
    parent_fingerprint: str | None = None
    parent_order_digest: str | None = None
    position: int | None = None
    schema: str = "operator_ref_descriptor/v1"

    def __post_init__(self) -> None:
        _require_digest(self.semantic_fingerprint, "semantic_fingerprint")
        _require_identifier(self.value_type, "value_type")
        if self.parent_fingerprint is not None:
            _require_digest(self.parent_fingerprint, "parent_fingerprint")
        if self.ref_kind is RefKind.INDEX:
            if (
                self.parent_fingerprint is None
                or self.parent_order_digest is None
                or self.position is None
            ):
                raise ValueError(
                    "index descriptors require parent, order digest, and position"
                )
            _require_digest(self.parent_order_digest, "parent_order_digest")
            if self.position < 0:
                raise ValueError("index position must be non-negative")
        elif self.parent_order_digest is not None or self.position is not None:
            raise ValueError("only index descriptors may carry positional fields")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "ref_kind": self.ref_kind.value,
            "semantic_fingerprint": self.semantic_fingerprint,
            "value_type": self.value_type,
            "compiler_facts": sorted({fact.value for fact in self.compiler_facts}),
            "parent_fingerprint": self.parent_fingerprint,
            "parent_order_digest": self.parent_order_digest,
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReferenceDescriptorV1:
        return cls(
            ref_kind=RefKind(value["ref_kind"]),
            semantic_fingerprint=str(value["semantic_fingerprint"]),
            value_type=str(value["value_type"]),
            compiler_facts=tuple(
                CompilerFact(item) for item in value.get("compiler_facts", ())
            ),
            parent_fingerprint=(
                str(value["parent_fingerprint"])
                if value.get("parent_fingerprint") is not None
                else None
            ),
            parent_order_digest=(
                str(value["parent_order_digest"])
                if value.get("parent_order_digest") is not None
                else None
            ),
            position=(
                int(value["position"]) if value.get("position") is not None else None
            ),
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True)
class RuntimeSymbolDescriptorV1:
    """Inference-visible runtime-symbol facts without the original surface."""

    symbol_fingerprint: str
    ref_fingerprint: str
    symbol_role: RuntimeSymbolRole
    semantic_role: str | None = None
    schema: str = "runtime_symbol_descriptor/v1"

    def __post_init__(self) -> None:
        _require_digest(self.symbol_fingerprint, "symbol_fingerprint")
        _require_digest(self.ref_fingerprint, "ref_fingerprint")
        if self.semantic_role is not None:
            _require_identifier(self.semantic_role, "semantic_role")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "symbol_fingerprint": self.symbol_fingerprint,
            "ref_fingerprint": self.ref_fingerprint,
            "symbol_role": self.symbol_role.value,
            "semantic_role": self.semantic_role,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RuntimeSymbolDescriptorV1:
        return cls(
            symbol_fingerprint=str(value["symbol_fingerprint"]),
            ref_fingerprint=str(value["ref_fingerprint"]),
            symbol_role=RuntimeSymbolRole(value["symbol_role"]),
            semantic_role=(
                str(value["semantic_role"])
                if value.get("semantic_role") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ReferenceEntryV1:
    ref: OperatorRef
    descriptor: ReferenceDescriptorV1

    def __post_init__(self) -> None:
        if self.ref.KIND is not self.descriptor.ref_kind:
            raise ValueError("reference kind does not match its descriptor")

    def to_dict(self) -> dict[str, Any]:
        return {"ref": self.ref.to_dict(), "descriptor": self.descriptor.to_dict()}


def _ref_from_dict(value: Mapping[str, Any]) -> OperatorRef:
    kind = RefKind(value["kind"])
    return _REF_CLASS[kind](str(value["request_id"]), str(value["opaque_id"]))


@dataclass(frozen=True)
class ReferenceTableV1:
    request_id: str
    state_digest: str
    branch_digest: str
    entries: tuple[ReferenceEntryV1, ...]
    runtime_symbols: tuple[RuntimeSymbolDescriptorV1, ...] = ()
    schema: str = "operator_reference_table/v1"

    def __post_init__(self) -> None:
        ValueRef(self.request_id, "request-validation")
        _require_digest(self.state_digest, "state_digest")
        _require_digest(self.branch_digest, "branch_digest")
        keys = tuple((entry.ref.KIND, entry.ref.opaque_id) for entry in self.entries)
        if len(set(keys)) != len(keys):
            raise ReferenceResolutionError("ref.duplicate")
        if any(entry.ref.request_id != self.request_id for entry in self.entries):
            raise ReferenceResolutionError("ref.cross_request")
        descriptor_fingerprints = {
            entry.descriptor.fingerprint for entry in self.entries
        }
        if any(
            symbol.ref_fingerprint not in descriptor_fingerprints
            for symbol in self.runtime_symbols
        ):
            raise ReferenceResolutionError("ref.runtime_symbol_missing")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "request_id": self.request_id,
            "state_digest": self.state_digest,
            "branch_digest": self.branch_digest,
            "entries": [entry.to_dict() for entry in self.entries],
            "runtime_symbols": [symbol.to_dict() for symbol in self.runtime_symbols],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ReferenceTableV1:
        return cls(
            request_id=str(value["request_id"]),
            state_digest=str(value["state_digest"]),
            branch_digest=str(value["branch_digest"]),
            entries=tuple(
                ReferenceEntryV1(
                    ref=_ref_from_dict(item["ref"]),
                    descriptor=ReferenceDescriptorV1.from_dict(item["descriptor"]),
                )
                for item in value.get("entries", ())
            ),
            runtime_symbols=tuple(
                RuntimeSymbolDescriptorV1.from_dict(item)
                for item in value.get("runtime_symbols", ())
            ),
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def resolve(
        self,
        ref: OperatorRef,
        *,
        state_digest: str,
        branch_digest: str,
        expected_kind: RefKind,
        current_parent_order_digest: str | None = None,
    ) -> ReferenceDescriptorV1:
        if ref.request_id != self.request_id:
            raise ReferenceResolutionError("ref.cross_request")
        if state_digest != self.state_digest:
            raise ReferenceResolutionError("ref.stale_state")
        if branch_digest != self.branch_digest:
            raise ReferenceResolutionError("ref.cross_branch")
        if ref.KIND is not expected_kind:
            raise ReferenceResolutionError("ref.type_incompatible")
        matches = [
            entry
            for entry in self.entries
            if entry.ref.KIND is ref.KIND and entry.ref.opaque_id == ref.opaque_id
        ]
        if not matches:
            raise ReferenceResolutionError("ref.missing")
        if len(matches) != 1:
            raise ReferenceResolutionError("ref.duplicate")
        descriptor = matches[0].descriptor
        if descriptor.ref_kind is not expected_kind:
            raise ReferenceResolutionError("ref.type_incompatible")
        if expected_kind is RefKind.INDEX:
            if current_parent_order_digest is None:
                raise ReferenceResolutionError("ref.index_context_required")
            if current_parent_order_digest != descriptor.parent_order_digest:
                raise ReferenceResolutionError("ref.stale_index")
        return descriptor

    def permuted(self, seed: int) -> ReferenceTableV1:
        """Return the same descriptors under new opaque IDs and candidate order."""
        rng = random.Random(seed)
        remapped: list[ReferenceEntryV1] = []
        for entry in sorted(self.entries, key=lambda item: item.descriptor.fingerprint):
            opaque_id = _fingerprint(
                {
                    "schema": "permuted_operator_ref/v1",
                    "seed": seed,
                    "descriptor_fingerprint": entry.descriptor.fingerprint,
                }
            )[:24]
            ref = _REF_CLASS[entry.ref.KIND](self.request_id, opaque_id)
            remapped.append(ReferenceEntryV1(ref, entry.descriptor))
        rng.shuffle(remapped)
        return ReferenceTableV1(
            request_id=self.request_id,
            state_digest=self.state_digest,
            branch_digest=self.branch_digest,
            entries=tuple(remapped),
            runtime_symbols=self.runtime_symbols,
        )


def build_reference_table(
    *,
    request_id: str,
    state_digest: str,
    branch_digest: str,
    descriptors: tuple[ReferenceDescriptorV1, ...],
    seed: int,
    runtime_symbols: tuple[RuntimeSymbolDescriptorV1, ...] = (),
) -> ReferenceTableV1:
    """Allocate opaque, seed-permutable IDs without semantic ordinals."""
    entries: list[ReferenceEntryV1] = []
    for descriptor in sorted(descriptors, key=lambda item: item.fingerprint):
        opaque_id = _fingerprint(
            {
                "schema": "allocated_operator_ref/v1",
                "request_id": request_id,
                "seed": seed,
                "descriptor_fingerprint": descriptor.fingerprint,
            }
        )[:24]
        ref = _REF_CLASS[descriptor.ref_kind](request_id, opaque_id)
        entries.append(ReferenceEntryV1(ref, descriptor))
    random.Random(seed).shuffle(entries)
    return ReferenceTableV1(
        request_id=request_id,
        state_digest=state_digest,
        branch_digest=branch_digest,
        entries=tuple(entries),
        runtime_symbols=runtime_symbols,
    )
