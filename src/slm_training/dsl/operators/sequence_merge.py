"""Conservative merge for verified multi-step branch application sequences.

Extends the one-step conservative branch merge (``merge.py``, DSH3-09) to
sequences of N chained AST-changing turns per branch, per DSH5-07. This
module is deliberately scoped, per that issue's own stop rule ("If sequence
collapse loses the precision required for exact effect/read-write
attribution, preserve the existing one-step merge bound and close sequence
merge as unsupported"):

* A sequence (``BranchApplicationSequenceV1``) is a chain of successful,
  replay-verified ``AST_EDIT`` applications on ONE branch from a common fork
  point. History-only turns (``UNDO``/``REDO``/``CHECKOUT_STATE``) never
  appear in a sequence at all -- a caller collapsing a real
  ``ConversationTraceV1`` excludes them before building one (they move the
  cursor over existing nodes; they do not add evidence a merge needs to
  reason about). ``TRANSACTION_COMMIT`` turns, and any ``FORK``/
  ``COPY_STATE`` turn occurring after a branch's starting point, are out of
  scope for this issue and are simply not representable here.
* Conflict classification collapses each branch's whole sequence into one
  base-relative ``ActionEffectV1`` plus lineage map (content-addressed by
  reference *descriptor fingerprint*, which survives every turn's reference
  table rebuild -- see ``_sequence_fingerprint_targets``), then reuses
  ``merge.classify_merge_effects`` unchanged. Any ref that cannot be traced
  back to a target that existed in ``base`` (for example a node created and
  only ever touched within one branch) is dropped from the composite
  effect: such content can never overlap the other branch, which forked
  from the same base and has no way to reference it.
* Unlike the one-step path, a composite-sequence conflict is never
  overridden by a declared ``commutes_with`` pair -- that override is a
  precise per-operator-pair claim about two single actions and does not
  generalize safely to two arbitrary sequences. This is a conservative
  narrowing relative to the one-step bound, not a defect: "Unknown overlap
  is a typed conflict, never optimistic merge" applies here at least as
  strongly as it does to one-step merge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from slm_training.dsl.operators.ast_merge import (
    MergeConflictKind,
    StructuralMergeConflict as _StructuralConflict,
    merge_ast_value as _merge_value,
)
from slm_training.dsl.operators.contracts import (
    ActionEffectV1,
    AstOperatorV1,
    CompilerCoverage,
    EffectDeltaKind,
    OperatorApplicationV1,
    OperatorRef,
    _fingerprint,
    _require_digest,
)
from slm_training.dsl.operators.conversation import ConversationStateNodeV1
from slm_training.dsl.operators.merge import (
    BranchAuthorityResolver,
    BranchEditV1,
    BranchMergeContinuationV1,
    MergeReferenceTableBuilder,
    _base_targets,
    _declaration,
    _effect_matches_declaration,
    _source_digest,
    _state_dict,
    classify_merge_effects,
)
from slm_training.dsl.operators.registry import OperatorStateV1
from slm_training.dsl.pack import DslPack


#: Internal stand-in used only to satisfy ``classify_merge_effects``'s
#: per-declaration ``effect_signature`` guard. The real, precise per-step
#: guard for a sequence runs in ``merge_branch_application_sequences``
#: before a composite effect is ever built, so every delta kind is legal
#: here -- that check has already run against each step's own declaration.
_PERMISSIVE_SEQUENCE_DECLARATION = AstOperatorV1(
    operator_id="sequence_composite_declaration",
    version="v1",
    domain="sequence",
    codomain="sequence",
    argument_slots=(),
    preconditions=(),
    effect_signature=tuple(EffectDeltaKind),
    locality="sequence",
    cost=0.0,
)


@dataclass(frozen=True)
class BranchApplicationSequenceV1:
    """N chained, replay-verified ``AST_EDIT`` applications on one branch.

    ``nodes`` has exactly ``len(applications) + 1`` entries: ``nodes[i]`` is
    the input state of ``applications[i]`` and ``nodes[i + 1]`` is its
    output. All nodes share one ``branch_digest``. This is the direct
    N-step generalization of ``merge.BranchEditV1`` -- a length-1 sequence
    carries exactly the same evidence as one ``BranchEditV1``.
    """

    nodes: tuple[ConversationStateNodeV1, ...]
    applications: tuple[OperatorApplicationV1, ...]
    schema: str = "branch_application_sequence/v1"

    def __post_init__(self) -> None:
        if not self.applications:
            raise ValueError(
                "branch application sequence requires at least one step"
            )
        if len(self.nodes) != len(self.applications) + 1:
            raise ValueError(
                "branch application sequence nodes must chain one more than "
                "applications"
            )
        branch_digest = self.nodes[0].branch_digest
        for index, application in enumerate(self.applications):
            input_node = self.nodes[index]
            output_node = self.nodes[index + 1]
            if (
                input_node.branch_digest != branch_digest
                or output_node.branch_digest != branch_digest
            ):
                raise ValueError(
                    "branch application sequence cannot cross branches"
                )
            if output_node.parent_state_id != input_node.state_id:
                raise ValueError("sequence step output must name its exact input")
            if not application.succeeded:
                raise ValueError("sequence step requires a successful application")
            if (
                application.before_state_digest != input_node.state.state_digest
                or application.before_ast_digest != input_node.state.ast_digest
                or application.after_state_digest != output_node.state.state_digest
                or application.after_ast_digest != output_node.state.ast_digest
            ):
                raise ValueError(
                    "sequence step application digests do not match its edge"
                )

    @property
    def branch_digest(self) -> str:
        return self.nodes[0].branch_digest

    @property
    def input_node(self) -> ConversationStateNodeV1:
        return self.nodes[0]

    @property
    def output_node(self) -> ConversationStateNodeV1:
        return self.nodes[-1]

    @property
    def application_ids(self) -> tuple[str, ...]:
        return tuple(
            application.application_id for application in self.applications
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "input_state_id": self.nodes[0].state_id,
            "output_state_id": self.nodes[-1].state_id,
            "branch_digest": self.branch_digest,
            "application_ids": list(self.application_ids),
            "applications": [
                application.to_dict() for application in self.applications
            ],
        }


def sequence_from_branch_edits(
    edits: tuple[BranchEditV1, ...],
) -> BranchApplicationSequenceV1:
    """Chain an ordered, already-validated list of single-step edits.

    Each ``BranchEditV1`` is individually proven (its own ``__post_init__``
    already checked its edge); this only proves the *chain*: edit ``i``'s
    output is edit ``i + 1``'s input. A caller collapsing a real
    ``ConversationTraceV1`` walks its turns, skips history-only turns,
    builds one ``BranchEditV1`` per ``AST_EDIT`` turn from consecutive
    ``ConversationStateNodeV1``/``OperatorApplicationV1`` evidence, and
    passes the ordered result here.
    """
    edits = tuple(edits)
    if not edits:
        raise ValueError("branch application sequence requires at least one step")
    for previous, current in zip(edits, edits[1:]):
        if current.input_node.state_id != previous.output_node.state_id:
            raise ValueError(
                "branch edits do not chain: each output must be the next input"
            )
    nodes = (edits[0].input_node, *(edit.output_node for edit in edits))
    applications = tuple(edit.application for edit in edits)
    return BranchApplicationSequenceV1(nodes=nodes, applications=applications)


@dataclass(frozen=True)
class BranchSequenceMergeConflictV1:
    kind: MergeConflictKind
    base_state_id: str
    branch_state_ids: tuple[str, ...]
    application_ids: tuple[str, ...]
    target_fingerprints: tuple[str, ...] = ()
    schema: str = "branch_sequence_merge_conflict/v1"

    def __post_init__(self) -> None:
        _require_digest(self.base_state_id, "base_state_id")
        if tuple(sorted(self.branch_state_ids)) != tuple(self.branch_state_ids):
            raise ValueError("branch state IDs must be canonical")
        if tuple(sorted(self.application_ids)) != tuple(self.application_ids):
            raise ValueError("application IDs must be canonical")
        for value in (
            *self.branch_state_ids,
            *self.application_ids,
            *self.target_fingerprints,
        ):
            _require_digest(value, "sequence merge conflict fingerprint")

    @property
    def conflict_id(self) -> str:
        return _fingerprint(self.to_dict(include_conflict_id=False))

    def to_dict(self, *, include_conflict_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": self.schema,
            "kind": self.kind.value,
            "base_state_id": self.base_state_id,
            "branch_state_ids": list(self.branch_state_ids),
            "application_ids": list(self.application_ids),
            "target_fingerprints": list(self.target_fingerprints),
        }
        if include_conflict_id:
            value["conflict_id"] = self.conflict_id
        return value


@dataclass(frozen=True)
class BranchSequenceMergeArtifactV1:
    pack_id: str
    base_state_id: str
    branch_state_ids: tuple[str, ...]
    application_ids: tuple[str, ...]
    merged_branch_digest: str
    merged_state: OperatorStateV1
    schema: str = "branch_sequence_merge/v1"

    def __post_init__(self) -> None:
        _require_digest(self.base_state_id, "base_state_id")
        _require_digest(self.merged_branch_digest, "merged_branch_digest")
        if tuple(sorted(self.branch_state_ids)) != tuple(self.branch_state_ids):
            raise ValueError("branch state IDs must be canonical")
        if tuple(sorted(self.application_ids)) != tuple(self.application_ids):
            raise ValueError("application IDs must be canonical")
        for value in (*self.branch_state_ids, *self.application_ids):
            _require_digest(value, "sequence merge artifact fingerprint")
        if self.merged_state.pack_id != self.pack_id:
            raise ValueError("merged state belongs to another pack")

    @property
    def merge_id(self) -> str:
        return _fingerprint(self.to_dict(include_merge_id=False))

    def to_dict(self, *, include_merge_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": self.schema,
            "pack_id": self.pack_id,
            "base_state_id": self.base_state_id,
            "branch_state_ids": list(self.branch_state_ids),
            "application_ids": list(self.application_ids),
            "merged_branch_digest": self.merged_branch_digest,
            "merged_state": _state_dict(self.merged_state),
        }
        if include_merge_id:
            value["merge_id"] = self.merge_id
        return value


@dataclass(frozen=True)
class BranchSequenceMergeDecisionV1:
    merge: BranchSequenceMergeArtifactV1 | None = None
    conflict: BranchSequenceMergeConflictV1 | None = None
    continuation: BranchMergeContinuationV1 | None = None
    schema: str = "branch_sequence_merge_decision/v1"

    def __post_init__(self) -> None:
        if (self.merge is None) == (self.conflict is None):
            raise ValueError("exactly one merge or conflict is required")
        if self.continuation is not None:
            if (
                self.merge is None
                or self.continuation.merge_id != self.merge.merge_id
            ):
                raise ValueError("continuation must bind its successful merge")

    @property
    def succeeded(self) -> bool:
        return self.merge is not None

    @property
    def decision_id(self) -> str:
        return _fingerprint(self.to_dict(include_decision_id=False))

    def to_dict(self, *, include_decision_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": self.schema,
            "merge": self.merge.to_dict() if self.merge is not None else None,
            "conflict": (
                self.conflict.to_dict() if self.conflict is not None else None
            ),
            "continuation": (
                self.continuation.to_dict()
                if self.continuation is not None
                else None
            ),
        }
        if include_decision_id:
            value["decision_id"] = self.decision_id
        return value


def sequence_merged_continuation_node(
    decision: BranchSequenceMergeDecisionV1,
) -> ConversationStateNodeV1:
    """Return the fresh merge root or reject a terminal v1 merge artifact."""
    if decision.continuation is None:
        raise ValueError("sequence_merge.continuation_unsupported_legacy")
    return decision.continuation.merged_node


def _sequence_fingerprint_targets(
    base: ConversationStateNodeV1, sequence: BranchApplicationSequenceV1
) -> dict[str, str]:
    """Base-relative target keyed by reference *descriptor fingerprint*.

    Descriptor fingerprints are content-addressed and independent of the
    opaque ref allocated to them in any one reference-table build, so this
    map -- built once from the sequence's immediate post-fork input node --
    stays valid for resolving the same surviving content in every later
    step's table, even though later tables reallocate opaque refs.
    """
    base_targets = _base_targets(base, sequence.nodes[0])
    return {
        entry.descriptor.fingerprint: base_targets[entry.ref]
        for entry in sequence.nodes[0].reference_table.entries
        if entry.ref in base_targets
    }


def _resolve_sequence_chain(
    ref: OperatorRef,
    node: ConversationStateNodeV1,
    fingerprint_targets: dict[str, str],
) -> tuple[str, ...] | None:
    """Walk ``ref``'s parent-fingerprint chain within ``node``'s own table.

    Returns ``None`` if ``ref`` is not even registered in ``node``'s own
    reference table -- a genuinely stale/forged reference, which the caller
    must keep *in* the composite effect but *out* of the lineage map so
    ``classify_merge_effects`` fails closed with ``STALE_REF``, exactly as
    the one-step path does for an unbound ref.

    Returns ``()`` if ``ref`` is registered but its content-addressed chain
    never reaches a target that existed in ``base`` (mirrors
    ``merge._base_target_lineage``'s all-or-nothing chain: every hop must
    resolve or the whole chain is discarded). This is a *legitimate*
    outcome for content this branch's own sequence created -- it cannot
    conflict with a branch that forked from the same base and has no way to
    reference content only this branch produced, so the caller drops it
    from the composite effect entirely rather than treating it as an error.
    """
    by_ref = {entry.ref: entry for entry in node.reference_table.entries}
    if ref not in by_ref:
        return None
    by_fingerprint = {
        entry.descriptor.fingerprint: entry for entry in node.reference_table.entries
    }
    chain: list[str] = []
    current = by_ref[ref]
    seen: set[Any] = set()
    while current is not None:
        if current.ref in seen:
            return ()
        seen.add(current.ref)
        target = fingerprint_targets.get(current.descriptor.fingerprint)
        if target is None:
            return ()
        chain.append(target)
        parent_fingerprint = current.descriptor.parent_fingerprint
        current = (
            by_fingerprint.get(parent_fingerprint)
            if parent_fingerprint is not None
            else None
        )
    return tuple(chain)


def _sequence_composite(
    base: ConversationStateNodeV1, sequence: BranchApplicationSequenceV1
) -> tuple[ActionEffectV1, dict[OperatorRef, tuple[str, ...]]] | None:
    """Collapse a branch's whole sequence into one base-relative effect.

    Returns ``None`` if any step's own effect is missing or not
    ``CompilerCoverage.EXACT`` -- an inexact or absent effect anywhere in
    the chain makes the whole sequence's base-relative footprint unprovable,
    so the caller must treat it as ``UNSUPPORTED_EFFECT``.
    """
    fingerprint_targets = _sequence_fingerprint_targets(base, sequence)
    lineage: dict[OperatorRef, tuple[str, ...]] = {}
    consumed_roles: list[Any] = []
    produced_roles: list[Any] = []
    consumed_binders: list[Any] = []
    produced_binders: list[Any] = []
    consumed_nodes: list[Any] = []
    produced_nodes: list[Any] = []
    scope_deltas: list[Any] = []
    cardinality_deltas: list[Any] = []
    property_deltas: list[Any] = []
    topology_deltas: list[Any] = []
    total_cost = 0.0

    for index, application in enumerate(sequence.applications):
        effect = application.effect
        if effect is None or effect.compiler_coverage is not CompilerCoverage.EXACT:
            return None
        input_node = sequence.nodes[index]

        def resolved(ref: OperatorRef) -> tuple[str, ...] | None:
            return _resolve_sequence_chain(ref, input_node, fingerprint_targets)

        for bucket, refs in (
            (consumed_roles, effect.consumed_roles),
            (produced_roles, effect.produced_roles),
            (consumed_binders, effect.consumed_binders),
            (produced_binders, effect.produced_binders),
            (consumed_nodes, effect.consumed_nodes),
            (produced_nodes, effect.produced_nodes),
        ):
            for ref in refs:
                chain = resolved(ref)
                if chain is None:
                    # Genuinely stale/forged ref: keep it in the composite
                    # effect but leave it out of lineage, so
                    # classify_merge_effects fails closed with STALE_REF.
                    bucket.append(ref)
                elif chain:
                    bucket.append(ref)
                    lineage[ref] = chain
                # else: legitimate intra-branch-only content -- drop it.
        for bucket, deltas in (
            (scope_deltas, effect.scope_deltas),
            (cardinality_deltas, effect.cardinality_deltas),
            (property_deltas, effect.property_deltas),
            (topology_deltas, effect.topology_deltas),
        ):
            for delta in deltas:
                chain = resolved(delta.target)
                if chain is None:
                    bucket.append(delta)
                elif chain:
                    bucket.append(delta)
                    lineage[delta.target] = chain
        total_cost += effect.estimated_completion_cost

    composite = ActionEffectV1(
        consumed_roles=tuple(consumed_roles),
        produced_roles=tuple(produced_roles),
        consumed_binders=tuple(consumed_binders),
        produced_binders=tuple(produced_binders),
        consumed_nodes=tuple(consumed_nodes),
        produced_nodes=tuple(produced_nodes),
        scope_deltas=tuple(scope_deltas),
        cardinality_deltas=tuple(cardinality_deltas),
        property_deltas=tuple(property_deltas),
        topology_deltas=tuple(topology_deltas),
        compiler_coverage=CompilerCoverage.EXACT,
        estimated_completion_cost=total_cost,
    )
    return composite, lineage


def _sequence_identity(
    left: BranchApplicationSequenceV1, right: BranchApplicationSequenceV1
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(sorted((left.output_node.state_id, right.output_node.state_id))),
        tuple(sorted((*left.application_ids, *right.application_ids))),
    )


def _sequence_conflict(
    kind: MergeConflictKind,
    base: ConversationStateNodeV1,
    left: BranchApplicationSequenceV1,
    right: BranchApplicationSequenceV1,
    targets: tuple[str, ...] = (),
) -> BranchSequenceMergeDecisionV1:
    branch_ids, application_ids = _sequence_identity(left, right)
    return BranchSequenceMergeDecisionV1(
        conflict=BranchSequenceMergeConflictV1(
            kind=kind,
            base_state_id=base.state_id,
            branch_state_ids=branch_ids,
            application_ids=application_ids,
            target_fingerprints=tuple(sorted(targets)),
        )
    )


def merge_branch_application_sequences(
    *,
    pack: DslPack,
    base: ConversationStateNodeV1,
    left: BranchApplicationSequenceV1,
    right: BranchApplicationSequenceV1,
    authority_resolver: BranchAuthorityResolver,
    reference_table_builder: MergeReferenceTableBuilder | None = None,
) -> BranchSequenceMergeDecisionV1:
    """Merge two verified multi-step branch sequences or return a typed conflict.

    Generalizes ``merge.merge_conversation_branches`` from one step per
    branch to N; see the module docstring for the exact, deliberately
    conservative scope of that generalization. Never raises for a
    legitimate conflict -- only for programmer errors or replay tamper.
    """
    if (
        left.nodes[0].state != base.state
        or right.nodes[0].state != base.state
        or left.branch_digest == right.branch_digest
    ):
        return _sequence_conflict(
            MergeConflictKind.UNSUPPORTED_EFFECT, base, left, right
        )

    for sequence in (left, right):
        for index, application in enumerate(sequence.applications):
            input_node = sequence.nodes[index]
            output_node = sequence.nodes[index + 1]
            branch_pack, library = authority_resolver(input_node)
            if branch_pack.pack_id != pack.pack_id:
                return _sequence_conflict(
                    MergeConflictKind.UNSUPPORTED_EFFECT, base, left, right
                )
            if (
                application.provenance.request_id
                != input_node.reference_table.request_id
                or application.provenance.source_artifact_digest
                != _source_digest(input_node.state.source)
            ):
                return _sequence_conflict(
                    MergeConflictKind.STALE_REF, base, left, right
                )
            if any(
                argument.value
                not in {entry.ref for entry in input_node.reference_table.entries}
                for argument in application.arguments
            ):
                return _sequence_conflict(
                    MergeConflictKind.STALE_REF, base, left, right
                )
            try:
                replayed = library.replay(
                    branch_pack, input_node.state, application
                )
            except Exception:  # noqa: BLE001 - external authority becomes typed refusal
                return _sequence_conflict(
                    MergeConflictKind.UNSUPPORTED_EFFECT, base, left, right
                )
            if replayed.state != output_node.state:
                return _sequence_conflict(
                    MergeConflictKind.UNSUPPORTED_EFFECT, base, left, right
                )
            declaration = _declaration(library, application)
            if (
                declaration is None
                or application.effect is None
                or not _effect_matches_declaration(declaration, application.effect)
            ):
                return _sequence_conflict(
                    MergeConflictKind.UNSUPPORTED_EFFECT, base, left, right
                )

    left_composed = _sequence_composite(base, left)
    right_composed = _sequence_composite(base, right)
    if left_composed is None or right_composed is None:
        return _sequence_conflict(
            MergeConflictKind.UNSUPPORTED_EFFECT, base, left, right
        )
    left_effect, left_lineage = left_composed
    right_effect, right_lineage = right_composed

    overlap = classify_merge_effects(
        left_declaration=_PERMISSIVE_SEQUENCE_DECLARATION,
        right_declaration=_PERMISSIVE_SEQUENCE_DECLARATION,
        left_effect=left_effect,
        right_effect=right_effect,
        left_lineage=left_lineage,
        right_lineage=right_lineage,
    )
    if overlap is not None:
        return _sequence_conflict(overlap[0], base, left, right, overlap[1])

    try:
        merged_ast = _merge_value(
            pack.backend.parse(base.state.source),
            pack.backend.parse(left.output_node.state.source),
            pack.backend.parse(right.output_node.state.source),
        )
        merged_state = OperatorStateV1.from_source(
            pack, pack.backend.serialize(merged_ast)
        )
    except _StructuralConflict as exc:
        return _sequence_conflict(exc.kind, base, left, right)
    except Exception:  # noqa: BLE001 - invalid merged AST is a typed refusal
        return _sequence_conflict(
            MergeConflictKind.UNSUPPORTED_EFFECT, base, left, right
        )

    branch_ids, application_ids = _sequence_identity(left, right)
    merged_branch_digest = _fingerprint(
        {
            "schema": "conversation_sequence_merge_branch/v1",
            "base_state_id": base.state_id,
            "branch_state_ids": branch_ids,
            "application_ids": application_ids,
        }
    )
    merge = BranchSequenceMergeArtifactV1(
        pack_id=pack.pack_id,
        base_state_id=base.state_id,
        branch_state_ids=branch_ids,
        application_ids=application_ids,
        merged_branch_digest=merged_branch_digest,
        merged_state=merged_state,
    )
    if reference_table_builder is None:
        return BranchSequenceMergeDecisionV1(merge=merge)
    allocation_seed = int(
        _fingerprint(
            {
                "schema": "branch_sequence_merge_reference_seed/v1",
                "merge_id": merge.merge_id,
            }
        )[:16],
        16,
    )
    try:
        merged_table = reference_table_builder(
            pack, merged_state, merged_branch_digest, allocation_seed
        )
        if (
            merged_table.state_digest != merged_state.state_digest
            or merged_table.branch_digest != merged_branch_digest
            or merged_table.request_id != base.reference_table.request_id
        ):
            raise ValueError("reference_table.identity_mismatch")
        continuation = BranchMergeContinuationV1(
            merge_id=merge.merge_id,
            allocation_seed=allocation_seed,
            merged_node=ConversationStateNodeV1(
                parent_state_id=None,
                branch_digest=merged_branch_digest,
                state=merged_state,
                reference_table=merged_table,
            ),
            reference_table_fingerprint=merged_table.fingerprint,
        )
    except Exception:  # noqa: BLE001 - context rebuild must fail closed
        return _sequence_conflict(
            MergeConflictKind.REFERENCE_REBUILD_FAILED, base, left, right
        )
    return BranchSequenceMergeDecisionV1(merge=merge, continuation=continuation)


def replay_branch_sequence_merge(
    *,
    pack: DslPack,
    base: ConversationStateNodeV1,
    left: BranchApplicationSequenceV1,
    right: BranchApplicationSequenceV1,
    authority_resolver: BranchAuthorityResolver,
    reference_table_builder: MergeReferenceTableBuilder | None = None,
    recorded: BranchSequenceMergeDecisionV1,
) -> BranchSequenceMergeDecisionV1:
    """Recompute the complete decision and require exact provenance identity."""
    replayed = merge_branch_application_sequences(
        pack=pack,
        base=base,
        left=left,
        right=right,
        authority_resolver=authority_resolver,
        reference_table_builder=reference_table_builder,
    )
    if replayed.decision_id != recorded.decision_id:
        raise ValueError(
            "branch sequence merge replay differs from recorded decision"
        )
    return replayed


__all__ = [
    "BranchApplicationSequenceV1",
    "BranchSequenceMergeArtifactV1",
    "BranchSequenceMergeConflictV1",
    "BranchSequenceMergeDecisionV1",
    "merge_branch_application_sequences",
    "replay_branch_sequence_merge",
    "sequence_from_branch_edits",
    "sequence_merged_continuation_node",
]
