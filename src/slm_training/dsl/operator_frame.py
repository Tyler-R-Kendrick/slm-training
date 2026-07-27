"""OperatorFrameV1: deterministic edit facts for one operator application.

DSH3-11 (SLM-379): a natural-language *edit* prompt for an operator
application must paraphrase EXACT operator effects -- never let an LLM define
transformation semantics. This module is the DSH2-02
(:mod:`slm_training.dsl.semantic_frame`) analogue for DSH3 operators: it
derives, deterministically and fail-closed, the partition of facts an edit
prompt may state from the already-fixed compiler evidence
(:class:`~slm_training.dsl.operators.contracts.AstOperatorV1` declaration,
typed bound arguments, the before-state identity, and the
:class:`~slm_training.dsl.operators.contracts.ActionEffectV1`):

- **required edit facts** -- what MUST change: every declared effect delta
  (target + delta kind + before/after value) plus consumed/produced roles and
  binders. These come from the operator's own ``ActionEffectV1``; nothing is
  inferred or guessed.
- **forbidden edit facts** -- what MUST NOT change: every reference in the
  current :class:`~slm_training.dsl.operators.references.ReferenceTableV1`
  the effect does NOT touch. This claim is only sound when the effect's
  ``compiler_coverage`` is ``exact``; under ``bounded``/``approximate``
  coverage the forbidden set is empty rather than overclaimed (fail-closed).
- **state-reference facts** -- the nodes/refs the operator touches (bound
  arguments plus effect targets), each resolved through the reference table's
  typed descriptor tables. Only inference-visible compiler facts
  (``ReferenceDescriptorV1.compiler_facts``) are exposed; a reference that is
  not in the table cannot be described without guessing, so derivation raises
  :class:`UnsupportedOperatorFrameError` instead (stop rule).

Every fact carries exact provenance (argument slot id / effect delta kind /
descriptor fingerprint). No display name, natural language, token ID, or
learned representation participates in any fact or fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from slm_training.dsl.operators.contracts import (
    ActionEffectV1,
    AstOperatorV1,
    BoundArgumentV1,
    CompilerCoverage,
    OperatorRef,
    _canonical_json,
    _fingerprint,
)
from slm_training.dsl.operators.references import (
    ReferenceDescriptorV1,
    ReferenceTableV1,
)
from slm_training.dsl.operators.registry import OperatorStateV1

OperatorFactCategory = Literal["required", "forbidden"]

#: Fact kinds for required edit facts (from ActionEffectV1).
REQUIRED_FACT_KINDS: tuple[str, ...] = (
    "effect_delta",
    "consumed_role",
    "produced_role",
    "consumed_binder",
    "produced_binder",
)
#: Fact kind for forbidden edit facts (untouched references).
FORBIDDEN_FACT_KIND = "untouched_reference"
#: Fact kind for state-reference facts.
STATE_REFERENCE_FACT_KIND = "state_reference"


class UnsupportedOperatorFrameError(ValueError):
    """An operator frame fact could not be derived from compiler evidence.

    Fail-closed per the DSH3-11 stop rule: raised instead of guessing a
    description for a reference the typed descriptor tables do not cover, or
    deriving a frame from stale/mismatched state evidence.
    """


@dataclass(frozen=True)
class OperatorFactProvenanceV1:
    """Exact compiler provenance for one operator frame fact.

    ``source`` is which compiler artifact the fact came from (``effect`` for
    ActionEffectV1-derived facts, ``argument`` for bound-argument state
    references, ``reference_table`` for untouched-reference forbidden facts).
    ``slot_id`` and ``delta_kind`` are set only when meaningful.
    ``descriptor_fingerprint`` binds the fact to the reference table's typed
    descriptor entry whenever the fact names a reference.
    """

    source: str
    slot_id: str | None = None
    delta_kind: str | None = None
    descriptor_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "slot_id": self.slot_id,
            "delta_kind": self.delta_kind,
            "descriptor_fingerprint": self.descriptor_fingerprint,
        }


@dataclass(frozen=True)
class OperatorFactV1:
    """One atomic operator edit/state-reference fact with exact provenance."""

    category: OperatorFactCategory
    fact_kind: str
    ref: OperatorRef
    value: Any
    provenance: OperatorFactProvenanceV1

    def __post_init__(self) -> None:
        _canonical_json(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "fact_kind": self.fact_kind,
            "ref": self.ref.to_dict(),
            "value": self.value,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class OperatorArgumentFactV1:
    """One bound argument plus its resolved typed descriptor (state reference)."""

    slot_id: str
    ref: OperatorRef
    descriptor: ReferenceDescriptorV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "ref": self.ref.to_dict(),
            "descriptor": self.descriptor.to_dict(),
        }


@dataclass(frozen=True)
class OperatorFrameV1:
    """The full deterministic operator frame for one operator application."""

    operator_id: str
    operator_fingerprint: str
    operator_version: str
    before_state_digest: str
    before_ast_digest: str
    reference_table_fingerprint: str
    effect_fingerprint: str
    arguments: tuple[OperatorArgumentFactV1, ...]
    required_edit_facts: tuple[OperatorFactV1, ...]
    forbidden_edit_facts: tuple[OperatorFactV1, ...]
    state_reference_facts: tuple[OperatorFactV1, ...]
    compiler_coverage: str
    schema: str = "operator_frame/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "operator_id": self.operator_id,
            "operator_fingerprint": self.operator_fingerprint,
            "operator_version": self.operator_version,
            "before_state_digest": self.before_state_digest,
            "before_ast_digest": self.before_ast_digest,
            "reference_table_fingerprint": self.reference_table_fingerprint,
            "effect_fingerprint": self.effect_fingerprint,
            "compiler_coverage": self.compiler_coverage,
            "arguments": [argument.to_dict() for argument in self.arguments],
            "required_edit_facts": [
                fact.to_dict() for fact in self.required_edit_facts
            ],
            "forbidden_edit_facts": [
                fact.to_dict() for fact in self.forbidden_edit_facts
            ],
            "state_reference_facts": [
                fact.to_dict() for fact in self.state_reference_facts
            ],
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def equivalent_to(self, other: "OperatorFrameV1") -> bool:
        """Semantic equivalence: same operator, same fact content.

        Facts are compared by canonical content (their ``to_dict`` payloads),
        not by dataclass identity or incidental ordering.
        """

        def fact_set(facts: tuple[OperatorFactV1, ...]) -> frozenset[str]:
            return frozenset(_canonical_json(fact.to_dict()) for fact in facts)

        def arg_set(args: tuple[OperatorArgumentFactV1, ...]) -> frozenset[str]:
            return frozenset(_canonical_json(arg.to_dict()) for arg in args)

        return (
            self.operator_fingerprint == other.operator_fingerprint
            and self.effect_fingerprint == other.effect_fingerprint
            and self.before_state_digest == other.before_state_digest
            and arg_set(self.arguments) == arg_set(other.arguments)
            and fact_set(self.required_edit_facts)
            == fact_set(other.required_edit_facts)
            and fact_set(self.forbidden_edit_facts)
            == fact_set(other.forbidden_edit_facts)
            and fact_set(self.state_reference_facts)
            == fact_set(other.state_reference_facts)
        )


@dataclass(frozen=True)
class OperatorFactConflictV1:
    """An internally-contradictory fact pair in a (mutated/hand-built) frame."""

    ref: dict[str, str]
    reason: str
    facts: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": dict(self.ref),
            "reason": self.reason,
            "facts": [dict(fact) for fact in self.facts],
        }


def _ref_key(ref: OperatorRef) -> tuple[str, str, str]:
    return (ref.KIND.value, ref.request_id, ref.opaque_id)


def frame_conflicts(frame: OperatorFrameV1) -> tuple[OperatorFactConflictV1, ...]:
    """Detect refs both required to change and forbidden from changing.

    Empty for any frame produced by :func:`derive_operator_frame`; exists so a
    mutated or hand-assembled frame can be checked for self-consistency.
    """
    required_keys = {_ref_key(fact.ref) for fact in frame.required_edit_facts}
    conflicts: list[OperatorFactConflictV1] = []
    for fact in frame.forbidden_edit_facts:
        if _ref_key(fact.ref) in required_keys:
            conflicts.append(
                OperatorFactConflictV1(
                    ref=fact.ref.to_dict(),
                    reason=(
                        "reference is both required to change and forbidden "
                        "from changing"
                    ),
                    facts=(
                        *(
                            other.to_dict()
                            for other in frame.required_edit_facts
                            if _ref_key(other.ref) == _ref_key(fact.ref)
                        ),
                        fact.to_dict(),
                    ),
                )
            )
    return tuple(conflicts)


def _descriptor_for(
    reference_table: ReferenceTableV1, ref: OperatorRef
) -> ReferenceDescriptorV1:
    for entry in reference_table.entries:
        if (
            entry.ref.KIND is ref.KIND
            and entry.ref.opaque_id == ref.opaque_id
            and entry.ref.request_id == ref.request_id
        ):
            return entry.descriptor
    raise UnsupportedOperatorFrameError(
        f"reference {ref.KIND.value}:{ref.opaque_id!r} is absent from the "
        "typed descriptor tables; refusing to guess a description"
    )


def derive_operator_frame(
    declaration: AstOperatorV1,
    arguments: tuple[BoundArgumentV1, ...],
    state: OperatorStateV1,
    effect: ActionEffectV1,
    reference_table: ReferenceTableV1,
) -> OperatorFrameV1:
    """Deterministically derive an :class:`OperatorFrameV1` from compiler facts.

    Validates the argument binding against the declaration, requires the
    reference table to be fresh for ``state``, and resolves every touched
    reference through the typed descriptor tables. Fails closed with
    :class:`UnsupportedOperatorFrameError` on any reference the tables do not
    cover or on stale state evidence.
    """
    if reference_table.state_digest != state.state_digest:
        raise UnsupportedOperatorFrameError(
            "reference table is stale for the before state"
        )
    ordered = declaration.validate_arguments(arguments)
    request_ids = {argument.value.request_id for argument in ordered}
    request_ids.update(effect.request_ids)
    if len(request_ids) > 1:
        raise UnsupportedOperatorFrameError(
            "arguments and effect references span multiple requests"
        )
    if request_ids and request_ids != {reference_table.request_id}:
        raise UnsupportedOperatorFrameError(
            "arguments/effect request differs from the reference table"
        )

    argument_facts = tuple(
        OperatorArgumentFactV1(
            slot_id=argument.slot_id,
            ref=argument.value,
            descriptor=_descriptor_for(reference_table, argument.value),
        )
        for argument in ordered
    )

    required: list[OperatorFactV1] = []
    for delta in (
        *effect.scope_deltas,
        *effect.cardinality_deltas,
        *effect.property_deltas,
        *effect.topology_deltas,
    ):
        required.append(
            OperatorFactV1(
                category="required",
                fact_kind="effect_delta",
                ref=delta.target,
                value={"before": delta.before, "after": delta.after},
                provenance=OperatorFactProvenanceV1(
                    source="effect",
                    delta_kind=delta.kind.value,
                    descriptor_fingerprint=_descriptor_for(
                        reference_table, delta.target
                    ).fingerprint,
                ),
            )
        )
    for kind, refs in (
        ("consumed_role", effect.consumed_roles),
        ("produced_role", effect.produced_roles),
        ("consumed_binder", effect.consumed_binders),
        ("produced_binder", effect.produced_binders),
    ):
        for ref in refs:
            required.append(
                OperatorFactV1(
                    category="required",
                    fact_kind=kind,
                    ref=ref,
                    value=None,
                    provenance=OperatorFactProvenanceV1(
                        source="effect",
                        descriptor_fingerprint=_descriptor_for(
                            reference_table, ref
                        ).fingerprint,
                    ),
                )
            )

    touched = {_ref_key(fact.ref) for fact in required}
    touched.update(_ref_key(argument.ref) for argument in argument_facts)

    forbidden: list[OperatorFactV1] = []
    if effect.compiler_coverage is CompilerCoverage.EXACT:
        for entry in reference_table.entries:
            if _ref_key(entry.ref) in touched:
                continue
            forbidden.append(
                OperatorFactV1(
                    category="forbidden",
                    fact_kind=FORBIDDEN_FACT_KIND,
                    ref=entry.ref,
                    value=None,
                    provenance=OperatorFactProvenanceV1(
                        source="reference_table",
                        descriptor_fingerprint=entry.descriptor.fingerprint,
                    ),
                )
            )

    state_refs: list[OperatorFactV1] = []
    seen: set[tuple[str, str, str]] = set()
    for argument in argument_facts:
        key = _ref_key(argument.ref)
        if key in seen:
            continue
        seen.add(key)
        state_refs.append(
            OperatorFactV1(
                category="required",
                fact_kind=STATE_REFERENCE_FACT_KIND,
                ref=argument.ref,
                value={
                    "value_type": argument.descriptor.value_type,
                    "compiler_facts": sorted(
                        fact.value for fact in argument.descriptor.compiler_facts
                    ),
                },
                provenance=OperatorFactProvenanceV1(
                    source="argument",
                    slot_id=argument.slot_id,
                    descriptor_fingerprint=argument.descriptor.fingerprint,
                ),
            )
        )
    for fact in required:
        key = _ref_key(fact.ref)
        if key in seen:
            continue
        seen.add(key)
        descriptor = _descriptor_for(reference_table, fact.ref)
        state_refs.append(
            OperatorFactV1(
                category="required",
                fact_kind=STATE_REFERENCE_FACT_KIND,
                ref=fact.ref,
                value={
                    "value_type": descriptor.value_type,
                    "compiler_facts": sorted(
                        item.value for item in descriptor.compiler_facts
                    ),
                },
                provenance=OperatorFactProvenanceV1(
                    source="effect",
                    descriptor_fingerprint=descriptor.fingerprint,
                ),
            )
        )

    return OperatorFrameV1(
        operator_id=declaration.operator_id,
        operator_fingerprint=declaration.fingerprint,
        operator_version=declaration.version,
        before_state_digest=state.state_digest,
        before_ast_digest=state.ast_digest,
        reference_table_fingerprint=reference_table.fingerprint,
        effect_fingerprint=effect.fingerprint,
        arguments=argument_facts,
        required_edit_facts=tuple(required),
        forbidden_edit_facts=tuple(forbidden),
        state_reference_facts=tuple(state_refs),
        compiler_coverage=effect.compiler_coverage.value,
    )


__all__ = [
    "FORBIDDEN_FACT_KIND",
    "REQUIRED_FACT_KINDS",
    "STATE_REFERENCE_FACT_KIND",
    "OperatorArgumentFactV1",
    "OperatorFactConflictV1",
    "OperatorFactProvenanceV1",
    "OperatorFactV1",
    "OperatorFrameV1",
    "UnsupportedOperatorFrameError",
    "derive_operator_frame",
    "frame_conflicts",
]
