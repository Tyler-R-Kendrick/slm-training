"""Sanitized, permutation-equivariant model boundary for operator policies (DSH3-22).

The operator runtime (``slm_training.dsl.operators``) carries semantic IDs,
opaque request-local surfaces, application/proof hashes, and successor
fingerprints that a learned policy must never see: any of those let a model
reconstruct which candidate is "the gold one" or peek at the outcome of an
action before it is chosen. This module defines the *only* allowed typed view
over a fresh ``ReferenceTableV1`` / ``OperatorLegalSetV1`` pair: an immutable,
row-indexed snapshot (``OperatorPolicyInputV1``) built once from a verified
freshness check and never re-derived from a stale table.

This module does not execute operators, does not change the runtime evidence
schemas in ``dsl/operators/``, and does not train anything: it is a read-only
projection consumed by future DSH3 M6/M7 learned-policy work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from slm_training.dsl.operators import (
    BindingPhase,
    CompilerFact,
    EffectDeltaKind,
    LegalSetCoverage,
    OperatorLibraryV1,
    OperatorSupportVerdict,
    RefKind,
    ReferenceDescriptorV1,
    ReferenceTableV1,
)
from slm_training.dsl.operators.legal_set import OperatorLegalSetV1

__all__ = [
    "FORBIDDEN_FIELD_NAMES",
    "ForbiddenFieldError",
    "OperatorActionViewV1",
    "OperatorArgumentSlotViewV1",
    "OperatorPolicyInputV1",
    "OperatorPolicyViewError",
    "ReferenceModelViewV1",
    "build_operator_policy_input",
    "validate_no_forbidden_fields",
]

# Every key name below identifies runtime evidence that would let a policy
# recover an opaque identity, a target/successor value, or a future outcome.
# The set is intentionally over-inclusive (recognizable substrings of the
# runtime's own field names) rather than an exact allowlist complement, so a
# renamed-but-still-forbidden field is still caught.
FORBIDDEN_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "semantic_fingerprint",
        "semantic_id",
        "candidate_id",
        "application_id",
        "application",
        "proof",
        "proof_fingerprint",
        "proof_checks",
        "proof_kind",
        "source_artifact_digest",
        "target",
        "target_ast",
        "target_fingerprint",
        "target_fingerprints",
        "source",
        "successor",
        "successor_fingerprint",
        "opaque_id",
        "request_id",
        "confirmation",
        "label",
        "labels",
        "planner_choice",
        "future_witness",
        "rejection",
        "rejection_samples",
        "before",
        "after",
        "before_state_digest",
        "before_ast_digest",
        "after_state_digest",
        "after_ast_digest",
        "compiler_result_digest",
        "effect_fingerprint",
    }
)


class OperatorPolicyViewError(ValueError):
    """Stable, machine-readable failure building or validating a policy view."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ForbiddenFieldError(OperatorPolicyViewError):
    """A forbidden field name was found in a value bound for model input."""

    def __init__(self, code: str, path: str) -> None:
        self.path = path
        super().__init__(code)


def validate_no_forbidden_fields(value: Any, *, _path: str = "$") -> None:
    """Recursively assert no key name in ``value`` is on the forbidden list.

    Walks dicts, lists, and tuples. Raises :class:`ForbiddenFieldError` on the
    first forbidden key it finds, at any nesting depth. This is defense in
    depth for the hand-authored views below, and the load-bearing assertion
    for the adversarial-injection tests: any future field added anywhere in
    this module's ``to_dict()`` output that collides with a forbidden name is
    caught even if the code that added it never re-reads this module's docs.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in FORBIDDEN_FIELD_NAMES:
                raise ForbiddenFieldError("policy_view.forbidden_field", f"{_path}.{key}")
            validate_no_forbidden_fields(item, _path=f"{_path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_no_forbidden_fields(item, _path=f"{_path}[{index}]")


@dataclass(frozen=True)
class ReferenceModelViewV1:
    """One allowlisted, row-local view over a fresh ``ReferenceDescriptorV1``.

    ``row`` is this entry's position in the owning ``OperatorPolicyInputV1``.
    ``parent_row`` is a row-local join into the same table (``None`` when the
    parent is not itself a resolvable reference in this table) — never the
    raw ``parent_fingerprint`` hash. No semantic, opaque, or request-local
    identity is a field here.
    """

    row: int
    ref_kind: RefKind
    value_type: str
    compiler_facts: tuple[CompilerFact, ...]
    has_parent: bool
    parent_row: int | None
    relative_position: int | None
    schema: str = "operator_reference_model_view/v1"

    def __post_init__(self) -> None:
        if self.row < 0:
            raise ValueError("row must be non-negative")
        if self.parent_row is not None and not self.has_parent:
            raise ValueError("parent_row requires has_parent")
        if self.parent_row is not None and self.parent_row < 0:
            raise ValueError("parent_row must be non-negative")
        if self.relative_position is not None and self.relative_position < 0:
            raise ValueError("relative_position must be non-negative")
        if self.ref_kind is not RefKind.INDEX and self.relative_position is not None:
            raise ValueError("only index references carry a relative position")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "row": self.row,
            "ref_kind": self.ref_kind.value,
            "value_type": self.value_type,
            "compiler_facts": sorted({fact.value for fact in self.compiler_facts}),
            "has_parent": self.has_parent,
            "parent_row": self.parent_row,
            "relative_position": self.relative_position,
        }


@dataclass(frozen=True)
class OperatorArgumentSlotViewV1:
    """One typed argument slot joined to its current candidate rows only."""

    slot_id: str
    ref_kind: RefKind
    binding_phase: BindingPhase
    required: bool
    repeated: bool
    candidate_rows: tuple[int, ...]
    domain_complete: bool
    schema: str = "operator_argument_slot_view/v1"

    def __post_init__(self) -> None:
        if len(set(self.candidate_rows)) != len(self.candidate_rows):
            raise ValueError("candidate_rows must be unique")
        if any(row < 0 for row in self.candidate_rows):
            raise ValueError("candidate_rows must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "slot_id": self.slot_id,
            "ref_kind": self.ref_kind.value,
            "binding_phase": self.binding_phase.value,
            "required": self.required,
            "repeated": self.repeated,
            "candidate_rows": sorted(self.candidate_rows),
            "domain_complete": self.domain_complete,
        }


@dataclass(frozen=True)
class OperatorActionViewV1:
    """One operator declaration's allowlisted, current-state-only view.

    Binds to reference rows only through ``OperatorArgumentSlotViewV1.
    candidate_rows``; carries no bound argument, no application, and no
    proof — this is the *legality/shape* view an action-kind/operator/
    typed-argument policy head scores over, not an executed decision.
    """

    row: int
    operator_id: str
    operator_version: str
    locality: str
    cost: float
    effect_signature: tuple[EffectDeltaKind, ...]
    argument_slots: tuple[OperatorArgumentSlotViewV1, ...]
    verdict: OperatorSupportVerdict
    coverage: LegalSetCoverage
    schema: str = "operator_action_view/v1"

    def __post_init__(self) -> None:
        if self.row < 0:
            raise ValueError("row must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "row": self.row,
            "operator_id": self.operator_id,
            "operator_version": self.operator_version,
            "locality": self.locality,
            "cost": self.cost,
            "effect_signature": sorted(kind.value for kind in self.effect_signature),
            "argument_slots": [slot.to_dict() for slot in self.argument_slots],
            "verdict": self.verdict.value,
            "coverage": self.coverage.value,
        }


@dataclass(frozen=True)
class OperatorPolicyInputV1:
    """The only allowed model input for DSH3 M6/M7 learned operator policies.

    Immutable snapshot of one fresh, verified ``(ReferenceTableV1,
    OperatorLegalSetV1)`` pair. Row order carries no semantics — a scorer
    must be equivariant under any permutation of ``reference_rows`` and
    ``action_rows`` (joins are by row index, which permutes along with its
    row). ``to_dict`` sorts by allowlisted content only, for reproducible
    evidence serialization; it never reveals the build-time opaque order.
    """

    reference_rows: tuple[ReferenceModelViewV1, ...]
    action_rows: tuple[OperatorActionViewV1, ...]
    ordinary_action_count: int
    coverage: LegalSetCoverage
    schema: str = "operator_policy_input/v1"

    def __post_init__(self) -> None:
        if self.ordinary_action_count < 0:
            raise ValueError("ordinary_action_count must be non-negative")
        n_refs = len(self.reference_rows)
        if tuple(view.row for view in self.reference_rows) != tuple(range(n_refs)):
            raise ValueError("reference_rows must be densely row-indexed from 0")
        if tuple(view.row for view in self.action_rows) != tuple(
            range(len(self.action_rows))
        ):
            raise ValueError("action_rows must be densely row-indexed from 0")
        for view in self.reference_rows:
            if view.parent_row is not None and view.parent_row >= n_refs:
                raise ValueError("parent_row must join an existing reference row")
        for action in self.action_rows:
            for slot in action.argument_slots:
                if any(row >= n_refs for row in slot.candidate_rows):
                    raise ValueError("candidate_rows must join existing reference rows")
        validate_no_forbidden_fields(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Canonicalize row order by allowlisted content, then remap joins.

        Two views built from opaque-ID/order permutations of the same
        underlying reference table serialize identically: rows are ordered by
        their own allowed content only (never by build-time row number), and
        every ``parent_row`` / ``candidate_rows`` join is remapped to the new
        canonical numbering so cross-references stay internally consistent.
        """

        def reference_key(index: int) -> tuple[Any, ...]:
            view = self.reference_rows[index]
            return (
                view.ref_kind.value,
                view.value_type,
                tuple(sorted({fact.value for fact in view.compiler_facts})),
                view.has_parent,
                view.relative_position is None,
                view.relative_position or 0,
            )

        canonical_order = sorted(range(len(self.reference_rows)), key=reference_key)
        new_row_of_old = {old: new for new, old in enumerate(canonical_order)}

        reference_payloads = []
        for new_row, old_row in enumerate(canonical_order):
            payload = self.reference_rows[old_row].to_dict()
            payload["row"] = new_row
            if payload["parent_row"] is not None:
                payload["parent_row"] = new_row_of_old[payload["parent_row"]]
            reference_payloads.append(payload)

        def action_key(action: OperatorActionViewV1) -> tuple[Any, ...]:
            return (action.operator_id, action.operator_version, action.verdict.value)

        action_payloads = []
        for new_row, action in enumerate(sorted(self.action_rows, key=action_key)):
            payload = action.to_dict()
            payload["row"] = new_row
            for slot_payload in payload["argument_slots"]:
                slot_payload["candidate_rows"] = sorted(
                    new_row_of_old[old_row] for old_row in slot_payload["candidate_rows"]
                )
            action_payloads.append(payload)

        return {
            "schema": self.schema,
            "reference_rows": reference_payloads,
            "action_rows": action_payloads,
            "ordinary_action_count": self.ordinary_action_count,
            "coverage": self.coverage.value,
        }


def _reference_view(
    row: int,
    descriptor: ReferenceDescriptorV1,
    row_by_semantic_fingerprint: Mapping[str, int],
) -> ReferenceModelViewV1:
    has_parent = descriptor.parent_fingerprint is not None
    parent_row = (
        row_by_semantic_fingerprint.get(descriptor.parent_fingerprint)
        if has_parent
        else None
    )
    return ReferenceModelViewV1(
        row=row,
        ref_kind=descriptor.ref_kind,
        value_type=descriptor.value_type,
        compiler_facts=descriptor.compiler_facts,
        has_parent=has_parent,
        parent_row=parent_row,
        relative_position=descriptor.position,
    )


def build_operator_policy_input(
    reference_table: ReferenceTableV1,
    legal_set: OperatorLegalSetV1,
    library: OperatorLibraryV1,
) -> OperatorPolicyInputV1:
    """Build the sanitized model input from one fresh, matching table+legal-set.

    Fails closed (``policy_view.stale_reference_table``) if ``legal_set`` was
    not built from this exact ``reference_table``, and
    (``policy_view.registry_mismatch``) if ``library`` did not produce
    ``legal_set`` — a stale or substituted table/registry can never produce a
    view, mirroring the runtime's own ``ref.stale_state`` / ``ref.cross_branch``
    guarantees.
    """
    if legal_set.reference_table_fingerprint != reference_table.fingerprint:
        raise OperatorPolicyViewError("policy_view.stale_reference_table")
    if legal_set.registry_fingerprint != library.registry_fingerprint:
        raise OperatorPolicyViewError("policy_view.registry_mismatch")

    row_by_semantic_fingerprint: dict[str, int] = {
        entry.descriptor.semantic_fingerprint: row
        for row, entry in enumerate(reference_table.entries)
    }
    reference_rows = tuple(
        _reference_view(row, entry.descriptor, row_by_semantic_fingerprint)
        for row, entry in enumerate(reference_table.entries)
    )
    row_by_ref: dict[tuple[RefKind, str], int] = {
        (entry.ref.KIND, entry.ref.opaque_id): row
        for row, entry in enumerate(reference_table.entries)
    }

    action_rows = tuple(
        OperatorActionViewV1(
            row=row,
            operator_id=entry.operator_id,
            operator_version=library.lookup(entry.operator_id).version,
            locality=library.lookup(entry.operator_id).locality,
            cost=library.lookup(entry.operator_id).cost,
            effect_signature=library.lookup(entry.operator_id).effect_signature,
            argument_slots=tuple(
                OperatorArgumentSlotViewV1(
                    slot_id=domain.slot_id,
                    ref_kind=domain.ref_kind,
                    binding_phase=BindingPhase.APPLICATION,
                    required=domain.required,
                    repeated=domain.repeated,
                    candidate_rows=tuple(
                        sorted(
                            row_by_ref[(candidate.KIND, candidate.opaque_id)]
                            for candidate in domain.candidates
                        )
                    ),
                    domain_complete=domain.complete,
                )
                for domain in entry.argument_domains
            ),
            verdict=entry.verdict,
            coverage=entry.coverage,
        )
        for row, entry in enumerate(legal_set.entries)
    )
    return OperatorPolicyInputV1(
        reference_rows=reference_rows,
        action_rows=action_rows,
        ordinary_action_count=len(legal_set.ordinary_nonoperator_actions),
        coverage=legal_set.coverage,
    )
