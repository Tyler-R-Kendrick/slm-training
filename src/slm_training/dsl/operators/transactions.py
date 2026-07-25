"""One-base multi-action operator transaction contracts (DSH5-04).

SLM-412 (DSH5-04) is milestone M2's opening decision for "DSH5 — Bulk
Operators, Transactions & Control Plane": several actions can be dry-run
against the *same* base state, but naively chaining their sequential
application invites exactly the staleness DSH3-03 typed references already
guard against (G0) — action *N* would resolve its refs against an AST that
action *N-1* has already changed, which is precisely the "reusing stale
request-local references" this issue's Decision names. This module never
chains dry-runs and never applies a mutation: every :class:`PreparedOperatorActionV1`
is dry-run against one immutable base (:class:`~slm_training.dsl.operators.registry.OperatorStateV1`
+ :class:`~slm_training.dsl.operators.references.ReferenceTableV1`), and a
:class:`OperatorTransactionV1` composes a *set* of such actions whose exact,
semantic read/write footprints have already been proven disjoint (or proven
commutative and exactly composable) — never by pretending a second dry-run
against the first action's *result* stays fresh.

This is a schema/safety layer only, per the issue's own agent contract: it
declares contracts, canonicalization, conflict/dependency derivation, and
typed rejections. It never wires a commit/execute path — no function in this
module applies a mutation to pack-authorized source. A future issue owns
actual multi-action commit execution.

Conflict rule composition with :mod:`slm_training.dsl.operators.merge`: the
write-footprint extraction reuses :func:`slm_training.dsl.operators.merge._effect_targets`
directly (not a re-implementation) to map an :class:`ActionEffectV1` onto
stable semantic target fingerprints. The mutual-commutativity gate mirrors
merge.py's own ``mutually_commuting`` check (both operators must name each
other in ``commutes_with``) — the same trust boundary, applied to a flat
N-way, one-base action set instead of a two-branch fork. Pairwise overlap
classification itself is implemented locally rather than by importing
merge.py's ``_conflict_kind``, because that helper's taxonomy
(``DELETE_MODIFY`` / ``CHILD_ORDER`` / ...) is fork-merge-specific; a
transaction only needs to know *whether* an overlap is resolved, not which of
merge.py's structural-merge categories it resembles.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from slm_training.dsl.operators.contracts import (
    ActionEffectV1,
    ApplicationProofV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BoundArgumentV1,
    CompilerCoverage,
    OperatorApplicationV1,
    OperatorRef,
    RefKind,
    _fingerprint,
    _require_digest,
    _require_identifier,
)
from slm_training.dsl.operators.legal_set import (
    _semantic_action_id as _legal_semantic_action_id,
)
from slm_training.dsl.operators.merge import _effect_targets
from slm_training.dsl.operators.references import ReferenceTableV1
from slm_training.dsl.operators.registry import OperatorLibraryV1, OperatorStateV1
from slm_training.dsl.pack import DslPack

#: Independent of (and possibly tighter than) any single operator's own
#: fanout policy — the DSH5-02 precedent (``MAP_SET_PROPERTY_MAX_FANOUT``)
#: for a bounded-budget guard against an unbounded action set.
OPERATOR_TRANSACTION_MAX_ACTIONS = 8


class OperatorTransactionRejectionKind(str, Enum):
    """Typed, data-dependent reasons a transaction refuses to build.

    ``noncanonical_order`` is deliberately *not* a member here: it is a
    structural invariant of :class:`OperatorTransactionV1` itself (raised as
    a plain ``ValueError`` from ``__post_init__``, mirroring how
    ``BranchMergeConflictV1``/``BranchMergeArtifactV1`` in ``merge.py``
    enforce canonical branch/application ID order) rather than a data-
    dependent outcome the builder decides between.
    """

    STALE_BASE = "stale_base"
    DUPLICATE_ACTION = "duplicate_action"
    DEPENDENCY_CYCLE = "dependency_cycle"
    UNKNOWN_FOOTPRINT = "unknown_footprint"
    CONFLICT = "conflict"
    PARTIAL_PREPARATION = "partial_preparation"
    FANOUT_EXCEEDED = "fanout_exceeded"


@dataclass(frozen=True)
class SemanticArgumentV1:
    """One bound argument's stable semantic identity, never its opaque ref."""

    slot_id: str
    ref_kind: RefKind
    target_fingerprint: str
    schema: str = "semantic_operator_argument/v1"

    def __post_init__(self) -> None:
        _require_identifier(self.slot_id, "slot_id")
        _require_digest(self.target_fingerprint, "target_fingerprint")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "slot_id": self.slot_id,
            "ref_kind": self.ref_kind.value,
            "target_fingerprint": self.target_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticArgumentV1:
        if str(value.get("schema")) != cls.schema:
            raise ValueError("semantic_argument.schema_unsupported")
        return cls(
            slot_id=str(value["slot_id"]),
            ref_kind=RefKind(value["ref_kind"]),
            target_fingerprint=str(value["target_fingerprint"]),
        )


@dataclass(frozen=True)
class TargetFootprintV1:
    """One semantic target's stable fingerprint plus the categories that touch it."""

    target_fingerprint: str
    categories: tuple[str, ...]
    schema: str = "operator_target_footprint/v1"

    def __post_init__(self) -> None:
        _require_digest(self.target_fingerprint, "target_fingerprint")
        if not self.categories:
            raise ValueError("a target footprint requires at least one category")
        object.__setattr__(
            self, "categories", tuple(sorted(set(self.categories)))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "target_fingerprint": self.target_fingerprint,
            "categories": list(self.categories),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TargetFootprintV1:
        if str(value.get("schema")) != cls.schema:
            raise ValueError("operator_target_footprint.schema_unsupported")
        return cls(
            target_fingerprint=str(value["target_fingerprint"]),
            categories=tuple(str(item) for item in value.get("categories", ())),
        )


@dataclass(frozen=True)
class OperatorReadWriteSetV1:
    """Stable semantic read/write footprints for one prepared action.

    Derived from :class:`ActionEffectV1` (writes: consumed/produced
    roles/binders and every delta bucket, via
    :func:`slm_training.dsl.operators.merge._effect_targets`) and the
    operator's own declared :class:`PreconditionV1` argument-slot footprint
    (reads) — never from the operator's ``operator_id``/name.
    """

    reads: tuple[TargetFootprintV1, ...]
    writes: tuple[TargetFootprintV1, ...]
    schema: str = "operator_read_write_set/v1"

    def __post_init__(self) -> None:
        for label, footprints in (("reads", self.reads), ("writes", self.writes)):
            targets = tuple(item.target_fingerprint for item in footprints)
            if len(set(targets)) != len(targets):
                raise ValueError(f"{label} footprint targets must be unique")
            if tuple(sorted(targets)) != targets:
                raise ValueError(f"{label} footprint must be canonically ordered")

    @property
    def read_targets(self) -> frozenset[str]:
        return frozenset(item.target_fingerprint for item in self.reads)

    @property
    def write_targets(self) -> frozenset[str]:
        return frozenset(item.target_fingerprint for item in self.writes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "reads": [item.to_dict() for item in self.reads],
            "writes": [item.to_dict() for item in self.writes],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OperatorReadWriteSetV1:
        if str(value.get("schema")) != cls.schema:
            raise ValueError("operator_read_write_set.schema_unsupported")
        return cls(
            reads=tuple(
                TargetFootprintV1.from_dict(item) for item in value.get("reads", ())
            ),
            writes=tuple(
                TargetFootprintV1.from_dict(item) for item in value.get("writes", ())
            ),
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True)
class PreparedOperatorActionV1:
    """One action dry-run against a fixed base; opaque IDs are evidence only.

    ``dry_run`` (the ``OperatorApplicationV1`` the base's live ``dry_run``
    produced) is kept as replay evidence — the same reason every other
    ``OperatorApplicationV1`` in this package is kept — but no field on this
    dataclass, nor on :class:`OperatorTransactionV1`, is ever keyed by
    ``dry_run``'s opaque argument refs. Identity, canonical ordering, and
    read/write derivation all route through ``semantic_action_id`` and
    ``semantic_arguments`` instead, which are stable across opaque-ID reuse
    or reallocation (DSH3-03 permutation invariance) and are never reused to
    author a second, post-commit application.
    """

    base_state_digest: str
    base_ast_digest: str
    operator_fingerprint: str
    semantic_action_id: str
    semantic_arguments: tuple[SemanticArgumentV1, ...]
    effect: ActionEffectV1
    proof: ApplicationProofV1
    dry_run: OperatorApplicationV1
    schema: str = "prepared_operator_action/v1"

    def __post_init__(self) -> None:
        _require_digest(self.base_state_digest, "base_state_digest")
        _require_digest(self.base_ast_digest, "base_ast_digest")
        _require_digest(self.operator_fingerprint, "operator_fingerprint")
        _require_digest(self.semantic_action_id, "semantic_action_id")
        slot_ids = tuple(argument.slot_id for argument in self.semantic_arguments)
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("semantic arguments must have unique slot ids")
        if tuple(sorted(slot_ids)) != slot_ids:
            raise ValueError(
                "semantic arguments must be canonically ordered by slot id"
            )
        if not self.dry_run.succeeded:
            raise ValueError("a prepared action requires a successful dry run")
        if self.dry_run.operator_fingerprint != self.operator_fingerprint:
            raise ValueError(
                "prepared action operator fingerprint disagrees with its dry run"
            )
        if (
            self.dry_run.before_state_digest != self.base_state_digest
            or self.dry_run.before_ast_digest != self.base_ast_digest
        ):
            raise ValueError(
                "prepared action base digests disagree with its dry run"
            )
        if self.dry_run.effect != self.effect or self.dry_run.proof != self.proof:
            raise ValueError(
                "prepared action effect/proof must equal its own dry run's"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "base_state_digest": self.base_state_digest,
            "base_ast_digest": self.base_ast_digest,
            "operator_fingerprint": self.operator_fingerprint,
            "semantic_action_id": self.semantic_action_id,
            "semantic_arguments": [
                argument.to_dict() for argument in self.semantic_arguments
            ],
            "effect": self.effect.to_dict(),
            "proof": self.proof.to_dict(),
            "dry_run": self.dry_run.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True)
class OperatorDependencyEdgeV1:
    """One required commit-order edge between two resolved, overlapping actions."""

    predecessor_semantic_id: str
    successor_semantic_id: str
    reason: str
    target_fingerprints: tuple[str, ...]
    schema: str = "operator_dependency_edge/v1"

    def __post_init__(self) -> None:
        _require_digest(self.predecessor_semantic_id, "predecessor_semantic_id")
        _require_digest(self.successor_semantic_id, "successor_semantic_id")
        if self.predecessor_semantic_id == self.successor_semantic_id:
            raise ValueError("a dependency edge cannot self-reference one action")
        _require_identifier(self.reason, "reason")
        for value in self.target_fingerprints:
            _require_digest(value, "target_fingerprint")
        if len(set(self.target_fingerprints)) != len(self.target_fingerprints):
            raise ValueError("dependency edge targets must be unique")
        if tuple(sorted(self.target_fingerprints)) != self.target_fingerprints:
            raise ValueError("dependency edge targets must be canonically ordered")
        if not self.target_fingerprints:
            raise ValueError("a dependency edge requires at least one target")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "predecessor_semantic_id": self.predecessor_semantic_id,
            "successor_semantic_id": self.successor_semantic_id,
            "reason": self.reason,
            "target_fingerprints": list(self.target_fingerprints),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OperatorDependencyEdgeV1:
        if str(value.get("schema")) != cls.schema:
            raise ValueError("operator_dependency_edge.schema_unsupported")
        return cls(
            predecessor_semantic_id=str(value["predecessor_semantic_id"]),
            successor_semantic_id=str(value["successor_semantic_id"]),
            reason=str(value["reason"]),
            target_fingerprints=tuple(
                str(item) for item in value.get("target_fingerprints", ())
            ),
        )


@dataclass(frozen=True)
class OperatorConflictEdgeV1:
    """Evidence that two actions overlapped and were resolved, never a live risk.

    Only *resolved* overlaps (mutual ``commutes_with``) ever reach this
    dataclass — an unresolved overlap rejects the whole transaction before
    one is ever constructed (see :data:`OperatorTransactionRejectionKind.CONFLICT`),
    so ``resolved_by_commutativity`` is always ``True`` here; the field exists
    so a serialized conflict-graph edge is self-explanatory without cross-
    referencing the rejection path.
    """

    left_semantic_id: str
    right_semantic_id: str
    target_fingerprints: tuple[str, ...]
    resolved_by_commutativity: bool
    schema: str = "operator_conflict_edge/v1"

    def __post_init__(self) -> None:
        _require_digest(self.left_semantic_id, "left_semantic_id")
        _require_digest(self.right_semantic_id, "right_semantic_id")
        if tuple(sorted((self.left_semantic_id, self.right_semantic_id))) != (
            self.left_semantic_id,
            self.right_semantic_id,
        ):
            raise ValueError("conflict edge action ids must be canonical")
        for value in self.target_fingerprints:
            _require_digest(value, "target_fingerprint")
        if len(set(self.target_fingerprints)) != len(self.target_fingerprints):
            raise ValueError("conflict edge targets must be unique")
        if tuple(sorted(self.target_fingerprints)) != self.target_fingerprints:
            raise ValueError("conflict edge targets must be canonically ordered")
        if not self.target_fingerprints:
            raise ValueError("a conflict edge requires at least one overlapping target")
        if not self.resolved_by_commutativity:
            raise ValueError(
                "only a resolved overlap may be recorded as conflict-graph evidence"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "left_semantic_id": self.left_semantic_id,
            "right_semantic_id": self.right_semantic_id,
            "target_fingerprints": list(self.target_fingerprints),
            "resolved_by_commutativity": self.resolved_by_commutativity,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OperatorConflictEdgeV1:
        if str(value.get("schema")) != cls.schema:
            raise ValueError("operator_conflict_edge.schema_unsupported")
        return cls(
            left_semantic_id=str(value["left_semantic_id"]),
            right_semantic_id=str(value["right_semantic_id"]),
            target_fingerprints=tuple(
                str(item) for item in value.get("target_fingerprints", ())
            ),
            resolved_by_commutativity=bool(value["resolved_by_commutativity"]),
        )


@dataclass(frozen=True)
class OperatorTransactionProofV1:
    """One composite proof binding a transaction's identity to its effect."""

    proof_kind: str
    checks: tuple[str, ...]
    transaction_result_digest: str
    composite_effect_fingerprint: str
    schema: str = "operator_transaction_proof/v1"

    def __post_init__(self) -> None:
        _require_identifier(self.proof_kind, "proof_kind")
        if not self.checks:
            raise ValueError("a transaction proof requires at least one check")
        for check in self.checks:
            _require_identifier(check, "proof check")
        _require_digest(
            self.transaction_result_digest, "transaction_result_digest"
        )
        _require_digest(
            self.composite_effect_fingerprint, "composite_effect_fingerprint"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "proof_kind": self.proof_kind,
            "checks": sorted(set(self.checks)),
            "transaction_result_digest": self.transaction_result_digest,
            "composite_effect_fingerprint": self.composite_effect_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OperatorTransactionProofV1:
        if str(value.get("schema")) != cls.schema:
            raise ValueError("operator_transaction_proof.schema_unsupported")
        return cls(
            proof_kind=str(value["proof_kind"]),
            checks=tuple(str(item) for item in value.get("checks", ())),
            transaction_result_digest=str(value["transaction_result_digest"]),
            composite_effect_fingerprint=str(value["composite_effect_fingerprint"]),
        )


@dataclass(frozen=True)
class OperatorTransactionRejectionV1:
    """One typed, deterministic reason a transaction refused to build."""

    kind: OperatorTransactionRejectionKind
    base_state_digest: str
    action_semantic_ids: tuple[str, ...]
    target_fingerprints: tuple[str, ...] = ()
    schema: str = "operator_transaction_rejection/v1"

    def __post_init__(self) -> None:
        _require_digest(self.base_state_digest, "base_state_digest")
        for value in (*self.action_semantic_ids, *self.target_fingerprints):
            _require_digest(value, "operator transaction rejection fingerprint")
        if tuple(sorted(self.action_semantic_ids)) != self.action_semantic_ids:
            raise ValueError("action semantic ids must be canonical")
        if tuple(sorted(self.target_fingerprints)) != self.target_fingerprints:
            raise ValueError("target fingerprints must be canonical")

    @property
    def rejection_id(self) -> str:
        return _fingerprint(self.to_dict(include_rejection_id=False))

    def to_dict(self, *, include_rejection_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "kind": self.kind.value,
            "base_state_digest": self.base_state_digest,
            "action_semantic_ids": list(self.action_semantic_ids),
            "target_fingerprints": list(self.target_fingerprints),
        }
        if include_rejection_id:
            value["rejection_id"] = self.rejection_id
        return value


@dataclass(frozen=True)
class OperatorTransactionV1:
    """One base, its canonical prepared actions, and one composite proof.

    Canonicalization: ``prepared_actions``, ``dependency_edges``, and
    ``conflict_graph`` are always ordered by ``semantic_action_id`` (never by
    submission order or any opaque ID) — two callers submitting the same
    action *set* in different orders always build byte-identical instances,
    per the DSH5-04 acceptance criterion. A directly constructed instance
    that violates this raises ``ValueError`` immediately, mirroring how
    ``merge.py``'s ``BranchMergeConflictV1``/``BranchMergeArtifactV1`` enforce
    canonical branch/application ID order.
    """

    base_state_digest: str
    base_ast_digest: str
    base_reference_table_fingerprint: str
    prepared_actions: tuple[PreparedOperatorActionV1, ...]
    dependency_edges: tuple[OperatorDependencyEdgeV1, ...]
    conflict_graph: tuple[OperatorConflictEdgeV1, ...]
    composite_effect: ActionEffectV1
    expected_final_state_digest: str
    expected_final_ast_digest: str
    proof: OperatorTransactionProofV1
    provenance: ApplicationProvenanceV1
    coverage: CompilerCoverage = CompilerCoverage.EXACT
    schema: str = "operator_transaction/v1"

    def __post_init__(self) -> None:
        _require_digest(self.base_state_digest, "base_state_digest")
        _require_digest(self.base_ast_digest, "base_ast_digest")
        _require_digest(
            self.base_reference_table_fingerprint, "base_reference_table_fingerprint"
        )
        _require_digest(
            self.expected_final_state_digest, "expected_final_state_digest"
        )
        _require_digest(self.expected_final_ast_digest, "expected_final_ast_digest")
        if not self.prepared_actions:
            raise ValueError("a transaction requires at least one prepared action")
        if any(
            action.base_state_digest != self.base_state_digest
            or action.base_ast_digest != self.base_ast_digest
            for action in self.prepared_actions
        ):
            raise ValueError("every prepared action must share the transaction base")
        ids = tuple(action.semantic_action_id for action in self.prepared_actions)
        if len(set(ids)) != len(ids):
            raise ValueError(
                "prepared actions must be unique by semantic action id"
            )
        if tuple(sorted(ids)) != ids:
            raise ValueError(
                "prepared actions must be canonically ordered by semantic action id"
            )
        id_set = set(ids)
        dependency_keys = tuple(
            (edge.predecessor_semantic_id, edge.successor_semantic_id)
            for edge in self.dependency_edges
        )
        if len(set(dependency_keys)) != len(dependency_keys):
            raise ValueError("dependency edges must be unique")
        if tuple(sorted(dependency_keys)) != dependency_keys:
            raise ValueError("dependency edges must be canonically ordered")
        if any(
            predecessor not in id_set or successor not in id_set
            for predecessor, successor in dependency_keys
        ):
            raise ValueError(
                "a dependency edge names an action outside this transaction"
            )
        if _has_cycle(ids, dependency_keys):
            raise ValueError("dependency edges contain a cycle")
        conflict_keys = tuple(
            (edge.left_semantic_id, edge.right_semantic_id)
            for edge in self.conflict_graph
        )
        if len(set(conflict_keys)) != len(conflict_keys):
            raise ValueError("conflict graph must be unique")
        if tuple(sorted(conflict_keys)) != conflict_keys:
            raise ValueError("conflict graph must be canonically ordered")
        if any(
            left not in id_set or right not in id_set
            for left, right in conflict_keys
        ):
            raise ValueError(
                "a conflict-graph edge names an action outside this transaction"
            )
        if self.coverage is not CompilerCoverage.EXACT:
            raise ValueError("a committed transaction requires exact coverage")
        if self.composite_effect.compiler_coverage is not CompilerCoverage.EXACT:
            raise ValueError("composite effect must have exact coverage")
        if self.proof.composite_effect_fingerprint != self.composite_effect.fingerprint:
            raise ValueError("transaction proof does not bind its composite effect")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "base_state_digest": self.base_state_digest,
            "base_ast_digest": self.base_ast_digest,
            "base_reference_table_fingerprint": self.base_reference_table_fingerprint,
            "prepared_actions": [
                action.to_dict() for action in self.prepared_actions
            ],
            "dependency_edges": [
                edge.to_dict() for edge in self.dependency_edges
            ],
            "conflict_graph": [edge.to_dict() for edge in self.conflict_graph],
            "composite_effect": self.composite_effect.to_dict(),
            "expected_final_state_digest": self.expected_final_state_digest,
            "expected_final_ast_digest": self.expected_final_ast_digest,
            "proof": self.proof.to_dict(),
            "provenance": self.provenance.to_dict(),
            "coverage": self.coverage.value,
        }

    @property
    def transaction_id(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True)
class OperatorTransactionDecisionV1:
    """Exactly one of a canonical transaction or a typed rejection."""

    transaction: OperatorTransactionV1 | None = None
    rejection: OperatorTransactionRejectionV1 | None = None
    schema: str = "operator_transaction_decision/v1"

    def __post_init__(self) -> None:
        if (self.transaction is None) == (self.rejection is None):
            raise ValueError("exactly one transaction or rejection is required")

    @property
    def succeeded(self) -> bool:
        return self.transaction is not None

    @property
    def decision_id(self) -> str:
        return _fingerprint(self.to_dict(include_decision_id=False))

    def to_dict(self, *, include_decision_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "transaction": (
                self.transaction.to_dict() if self.transaction is not None else None
            ),
            "rejection": (
                self.rejection.to_dict() if self.rejection is not None else None
            ),
        }
        if include_decision_id:
            value["decision_id"] = self.decision_id
        return value


def _target_lineage(table: ReferenceTableV1) -> dict[OperatorRef, str]:
    """Map every base-table ref to its stable semantic target fingerprint."""
    return {entry.ref: entry.descriptor.fingerprint for entry in table.entries}


def _precondition_footprint(
    declaration: AstOperatorV1,
    arguments: tuple[BoundArgumentV1, ...],
    lineage: Mapping[OperatorRef, str],
) -> tuple[dict[str, set[str]], bool]:
    """Read footprint: targets named by a declared precondition's argument slots.

    Never derived from ``declaration.operator_id`` — only the *shape* of its
    declared ``preconditions`` decides which bound arguments are reads.
    """
    footprint_slots = {
        slot_id
        for condition in declaration.preconditions
        for slot_id in condition.argument_slots
    }
    categories: dict[str, set[str]] = {}
    for argument in arguments:
        if argument.slot_id not in footprint_slots:
            continue
        target = lineage.get(argument.value)
        if target is None:
            return {}, False
        categories.setdefault(target, set()).add(argument.value.KIND.value)
    return categories, True


def derive_read_write_set(
    declaration: AstOperatorV1,
    arguments: tuple[BoundArgumentV1, ...],
    effect: ActionEffectV1,
    lineage: Mapping[OperatorRef, str],
) -> OperatorReadWriteSetV1 | None:
    """Derive one action's stable read/write footprint, or ``None`` if unknown.

    ``None`` covers both halves of "unknown effects or approximate coverage
    reject": a non-exact ``effect.compiler_coverage``, or any target (from
    the effect or from a precondition-footprint argument) absent from
    ``lineage`` — the caller must fail closed on ``None`` rather than
    guessing a partial footprint.
    """
    if effect.compiler_coverage is not CompilerCoverage.EXACT:
        return None
    write_categories, write_ok = _effect_targets(effect, lineage)
    if not write_ok:
        return None
    read_categories, read_ok = _precondition_footprint(
        declaration, arguments, lineage
    )
    if not read_ok:
        return None
    reads = tuple(
        sorted(
            (
                TargetFootprintV1(target, tuple(categories))
                for target, categories in read_categories.items()
            ),
            key=lambda item: item.target_fingerprint,
        )
    )
    writes = tuple(
        sorted(
            (
                TargetFootprintV1(target, tuple(categories))
                for target, categories in write_categories.items()
            ),
            key=lambda item: item.target_fingerprint,
        )
    )
    return OperatorReadWriteSetV1(reads=reads, writes=writes)


def _has_cycle(
    nodes: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> bool:
    graph: dict[str, list[str]] = {node: [] for node in nodes}
    for predecessor, successor in edges:
        graph[predecessor].append(successor)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for neighbor in graph[node]:
            if visit(neighbor):
                return True
        visiting.discard(node)
        visited.add(node)
        return False

    return any(visit(node) for node in nodes if node not in visited)


def prepare_operator_action(
    *,
    pack: DslPack,
    library: OperatorLibraryV1,
    state: OperatorStateV1,
    reference_table: ReferenceTableV1,
    operator_id: str,
    arguments: tuple[BoundArgumentV1, ...],
    provenance: ApplicationProvenanceV1,
) -> PreparedOperatorActionV1:
    """Dry-run one action against ``state``/``reference_table`` (the base).

    Never applies a mutation (only ``library.dry_run``) — half of the G0-
    staleness fix is that every prepared action is dry-run against the exact
    same, unmodified base, never against another action's result. Raises
    ``ValueError`` (never a silent ``None``) on a stale reference table or a
    rejected dry run; ``prepare_operator_transaction`` turns those into typed
    ``OperatorTransactionRejectionV1`` decisions.
    """
    if reference_table.state_digest != state.state_digest:
        raise ValueError(
            "reference table is stale for the prepared action's base state"
        )
    if reference_table.request_id != provenance.request_id:
        raise ValueError("reference table and provenance request differ")
    if any(
        argument.value.request_id != reference_table.request_id
        for argument in arguments
    ):
        raise ValueError(
            "action arguments do not belong to the base reference table's request"
        )
    application = library.dry_run(
        pack, state, operator_id, arguments, provenance
    )
    if not application.succeeded:
        raise ValueError("action dry run was rejected")
    assert application.effect is not None
    assert application.proof is not None
    declaration = library.lookup(operator_id)
    lineage = _target_lineage(reference_table)
    try:
        ordered = declaration.validate_arguments(application.arguments)
        semantic_action_id = _legal_semantic_action_id(
            declaration.fingerprint, ordered, lineage
        )
        semantic_arguments = tuple(
            sorted(
                (
                    SemanticArgumentV1(
                        slot_id=argument.slot_id,
                        ref_kind=argument.value.KIND,
                        target_fingerprint=lineage[argument.value],
                    )
                    for argument in ordered
                ),
                key=lambda item: item.slot_id,
            )
        )
    except KeyError as exc:
        raise ValueError(
            "action argument is unresolvable against the base table"
        ) from exc
    return PreparedOperatorActionV1(
        base_state_digest=state.state_digest,
        base_ast_digest=state.ast_digest,
        operator_fingerprint=declaration.fingerprint,
        semantic_action_id=semantic_action_id,
        semantic_arguments=semantic_arguments,
        effect=application.effect,
        proof=application.proof,
        dry_run=application,
    )


def _reject(
    kind: OperatorTransactionRejectionKind,
    base_state_digest: str,
    action_semantic_ids: tuple[str, ...],
    target_fingerprints: tuple[str, ...] = (),
) -> OperatorTransactionDecisionV1:
    return OperatorTransactionDecisionV1(
        rejection=OperatorTransactionRejectionV1(
            kind=kind,
            base_state_digest=base_state_digest,
            action_semantic_ids=tuple(sorted(set(action_semantic_ids))),
            target_fingerprints=tuple(sorted(set(target_fingerprints))),
        )
    )


def build_operator_transaction(
    *,
    library: OperatorLibraryV1,
    state: OperatorStateV1,
    reference_table: ReferenceTableV1,
    prepared_actions: tuple[PreparedOperatorActionV1, ...],
    provenance: ApplicationProvenanceV1,
    max_actions: int = OPERATOR_TRANSACTION_MAX_ACTIONS,
) -> OperatorTransactionDecisionV1:
    """Assemble already-prepared actions into one canonical transaction.

    Every check here is data-dependent (stale base, duplicate action,
    unknown footprint, conflict, cycle, fanout) — never a structural
    invariant of the built artifact itself (that lives in
    ``OperatorTransactionV1.__post_init__``, e.g. canonical ordering).
    """
    if max_actions <= 0:
        raise ValueError("max_actions must be positive")
    base_state_digest = state.state_digest
    base_ast_digest = state.ast_digest
    if reference_table.state_digest != base_state_digest:
        raise ValueError("reference table is stale for the transaction base state")
    if reference_table.request_id != provenance.request_id:
        raise ValueError("reference table and provenance request differ")

    all_ids = tuple(action.semantic_action_id for action in prepared_actions)
    if len(prepared_actions) > max_actions:
        return _reject(
            OperatorTransactionRejectionKind.FANOUT_EXCEEDED,
            base_state_digest,
            all_ids,
        )
    if any(
        action.base_state_digest != base_state_digest
        or action.base_ast_digest != base_ast_digest
        or action.dry_run.provenance.request_id != provenance.request_id
        for action in prepared_actions
    ):
        return _reject(
            OperatorTransactionRejectionKind.STALE_BASE,
            base_state_digest,
            all_ids,
        )
    if len(set(all_ids)) != len(all_ids):
        duplicates = {
            semantic_id
            for semantic_id in set(all_ids)
            if all_ids.count(semantic_id) > 1
        }
        return _reject(
            OperatorTransactionRejectionKind.DUPLICATE_ACTION,
            base_state_digest,
            tuple(duplicates),
        )

    lineage = _target_lineage(reference_table)
    declarations_by_fingerprint = {
        declaration.fingerprint: declaration for declaration in library.declarations
    }
    read_write_by_id: dict[str, OperatorReadWriteSetV1] = {}
    declaration_by_id: dict[str, AstOperatorV1] = {}
    for action in prepared_actions:
        declaration = declarations_by_fingerprint.get(action.operator_fingerprint)
        if declaration is None:
            return _reject(
                OperatorTransactionRejectionKind.UNKNOWN_FOOTPRINT,
                base_state_digest,
                all_ids,
            )
        read_write = derive_read_write_set(
            declaration, action.dry_run.arguments, action.effect, lineage
        )
        if read_write is None:
            return _reject(
                OperatorTransactionRejectionKind.UNKNOWN_FOOTPRINT,
                base_state_digest,
                all_ids,
            )
        read_write_by_id[action.semantic_action_id] = read_write
        declaration_by_id[action.semantic_action_id] = declaration

    sorted_actions = tuple(
        sorted(prepared_actions, key=lambda action: action.semantic_action_id)
    )
    conflict_edges: list[OperatorConflictEdgeV1] = []
    directed: dict[tuple[str, str], dict[str, set[str]]] = {}
    for left, right in itertools.combinations(sorted_actions, 2):
        left_rw = read_write_by_id[left.semantic_action_id]
        right_rw = read_write_by_id[right.semantic_action_id]
        write_write = left_rw.write_targets & right_rw.write_targets
        write_read = left_rw.write_targets & right_rw.read_targets
        read_write_overlap = right_rw.write_targets & left_rw.read_targets
        overlap = write_write | write_read | read_write_overlap
        if not overlap:
            continue
        left_decl = declaration_by_id[left.semantic_action_id]
        right_decl = declaration_by_id[right.semantic_action_id]
        mutually_commuting = (
            right_decl.operator_id in left_decl.commutes_with
            and left_decl.operator_id in right_decl.commutes_with
        )
        if not mutually_commuting:
            return _reject(
                OperatorTransactionRejectionKind.CONFLICT,
                base_state_digest,
                (left.semantic_action_id, right.semantic_action_id),
                tuple(overlap),
            )
        conflict_edges.append(
            OperatorConflictEdgeV1(
                left_semantic_id=left.semantic_action_id,
                right_semantic_id=right.semantic_action_id,
                target_fingerprints=tuple(sorted(overlap)),
                resolved_by_commutativity=True,
            )
        )

        def _record(predecessor: str, successor: str, targets: frozenset[str]) -> None:
            entry = directed.setdefault(
                (predecessor, successor), {"reasons": set(), "targets": set()}
            )
            entry["reasons"].add("write_before_read")
            entry["targets"].update(targets)

        if write_read:
            _record(left.semantic_action_id, right.semantic_action_id, write_read)
        if read_write_overlap:
            _record(
                right.semantic_action_id, left.semantic_action_id, read_write_overlap
            )
        remaining_write_write = write_write - write_read - read_write_overlap
        if remaining_write_write:
            ordered_pair = tuple(
                sorted((left.semantic_action_id, right.semantic_action_id))
            )
            entry = directed.setdefault(
                ordered_pair, {"reasons": set(), "targets": set()}
            )
            entry["reasons"].add("write_write_order")
            entry["targets"].update(remaining_write_write)

    dependency_edges = tuple(
        sorted(
            (
                OperatorDependencyEdgeV1(
                    predecessor_semantic_id=predecessor,
                    successor_semantic_id=successor,
                    reason="-".join(sorted(entry["reasons"])),
                    target_fingerprints=tuple(sorted(entry["targets"])),
                )
                for (predecessor, successor), entry in directed.items()
            ),
            key=lambda edge: (
                edge.predecessor_semantic_id,
                edge.successor_semantic_id,
            ),
        )
    )
    if _has_cycle(
        all_ids,
        tuple(
            (edge.predecessor_semantic_id, edge.successor_semantic_id)
            for edge in dependency_edges
        ),
    ):
        return _reject(
            OperatorTransactionRejectionKind.DEPENDENCY_CYCLE,
            base_state_digest,
            all_ids,
        )

    conflict_graph = tuple(
        sorted(
            conflict_edges,
            key=lambda edge: (edge.left_semantic_id, edge.right_semantic_id),
        )
    )
    composite_effect = _compose_effect(sorted_actions)
    sorted_ids = tuple(action.semantic_action_id for action in sorted_actions)
    expected_final_state_digest = _fingerprint(
        {
            "schema": "operator_transaction_expected_state/v1",
            "base_state_digest": base_state_digest,
            "composite_effect_fingerprint": composite_effect.fingerprint,
            "action_semantic_ids": list(sorted_ids),
        }
    )
    expected_final_ast_digest = _fingerprint(
        {
            "schema": "operator_transaction_expected_ast/v1",
            "base_ast_digest": base_ast_digest,
            "composite_effect_fingerprint": composite_effect.fingerprint,
            "action_semantic_ids": list(sorted_ids),
        }
    )
    proof = OperatorTransactionProofV1(
        proof_kind="transaction.prepared",
        checks=(
            "base.consistent",
            "footprints.exact",
            "conflicts.resolved",
            "dependency.acyclic",
            "order.canonical",
        ),
        transaction_result_digest=_fingerprint(
            {
                "schema": "operator_transaction_result/v1",
                "base_state_digest": base_state_digest,
                "prepared_action_fingerprints": [
                    action.fingerprint for action in sorted_actions
                ],
            }
        ),
        composite_effect_fingerprint=composite_effect.fingerprint,
    )
    transaction = OperatorTransactionV1(
        base_state_digest=base_state_digest,
        base_ast_digest=base_ast_digest,
        base_reference_table_fingerprint=reference_table.fingerprint,
        prepared_actions=sorted_actions,
        dependency_edges=dependency_edges,
        conflict_graph=conflict_graph,
        composite_effect=composite_effect,
        expected_final_state_digest=expected_final_state_digest,
        expected_final_ast_digest=expected_final_ast_digest,
        proof=proof,
        provenance=provenance,
    )
    return OperatorTransactionDecisionV1(transaction=transaction)


def _compose_effect(actions: tuple[PreparedOperatorActionV1, ...]) -> ActionEffectV1:
    return ActionEffectV1(
        consumed_roles=tuple(
            ref for action in actions for ref in action.effect.consumed_roles
        ),
        produced_roles=tuple(
            ref for action in actions for ref in action.effect.produced_roles
        ),
        consumed_binders=tuple(
            ref for action in actions for ref in action.effect.consumed_binders
        ),
        produced_binders=tuple(
            ref for action in actions for ref in action.effect.produced_binders
        ),
        scope_deltas=tuple(
            delta for action in actions for delta in action.effect.scope_deltas
        ),
        cardinality_deltas=tuple(
            delta
            for action in actions
            for delta in action.effect.cardinality_deltas
        ),
        property_deltas=tuple(
            delta for action in actions for delta in action.effect.property_deltas
        ),
        topology_deltas=tuple(
            delta for action in actions for delta in action.effect.topology_deltas
        ),
        compiler_coverage=CompilerCoverage.EXACT,
        estimated_completion_cost=sum(
            action.effect.estimated_completion_cost for action in actions
        ),
    )


def prepare_operator_transaction(
    *,
    pack: DslPack,
    library: OperatorLibraryV1,
    state: OperatorStateV1,
    reference_table: ReferenceTableV1,
    requests: tuple[tuple[str, tuple[BoundArgumentV1, ...]], ...],
    provenance: ApplicationProvenanceV1,
    max_actions: int = OPERATOR_TRANSACTION_MAX_ACTIONS,
) -> OperatorTransactionDecisionV1:
    """Dry-run every requested action against one base, then assemble it.

    Convenience composition of :func:`prepare_operator_action` (once per
    request) and :func:`build_operator_transaction` — never a parallel
    execution path. If any single request's dry run is outright rejected,
    the whole batch refuses as ``PARTIAL_PREPARATION`` rather than silently
    building a transaction from the requests that happened to succeed.
    """
    if max_actions <= 0:
        raise ValueError("max_actions must be positive")
    if len(requests) > max_actions:
        return _reject(
            OperatorTransactionRejectionKind.FANOUT_EXCEEDED,
            state.state_digest,
            (),
        )
    prepared: list[PreparedOperatorActionV1] = []
    for operator_id, arguments in requests:
        try:
            prepared.append(
                prepare_operator_action(
                    pack=pack,
                    library=library,
                    state=state,
                    reference_table=reference_table,
                    operator_id=operator_id,
                    arguments=arguments,
                    provenance=provenance,
                )
            )
        except ValueError:
            return _reject(
                OperatorTransactionRejectionKind.PARTIAL_PREPARATION,
                state.state_digest,
                tuple(action.semantic_action_id for action in prepared),
            )
    return build_operator_transaction(
        library=library,
        state=state,
        reference_table=reference_table,
        prepared_actions=tuple(prepared),
        provenance=provenance,
        max_actions=max_actions,
    )


def replay_operator_transaction(
    *,
    library: OperatorLibraryV1,
    state: OperatorStateV1,
    reference_table: ReferenceTableV1,
    prepared_actions: tuple[PreparedOperatorActionV1, ...],
    provenance: ApplicationProvenanceV1,
    recorded: OperatorTransactionDecisionV1,
    max_actions: int = OPERATOR_TRANSACTION_MAX_ACTIONS,
) -> OperatorTransactionDecisionV1:
    """Recompute the complete decision and require exact identity with ``recorded``."""
    replayed = build_operator_transaction(
        library=library,
        state=state,
        reference_table=reference_table,
        prepared_actions=prepared_actions,
        provenance=provenance,
        max_actions=max_actions,
    )
    if replayed.decision_id != recorded.decision_id:
        raise ValueError("operator transaction replay differs from recorded decision")
    return replayed


__all__ = [
    "OPERATOR_TRANSACTION_MAX_ACTIONS",
    "OperatorConflictEdgeV1",
    "OperatorDependencyEdgeV1",
    "OperatorReadWriteSetV1",
    "OperatorTransactionDecisionV1",
    "OperatorTransactionProofV1",
    "OperatorTransactionRejectionKind",
    "OperatorTransactionRejectionV1",
    "OperatorTransactionV1",
    "PreparedOperatorActionV1",
    "SemanticArgumentV1",
    "TargetFootprintV1",
    "build_operator_transaction",
    "derive_read_write_set",
    "prepare_operator_action",
    "prepare_operator_transaction",
    "replay_operator_transaction",
]
