from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BranchApplicationSequenceV1,
    BranchEditV1,
    CompilerCoverage,
    ConversationStateNodeV1,
    EffectDeltaKind,
    EffectDeltaV1,
    MergeConflictKind,
    NodeRef,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorStateV1,
    RefKind,
    ReferenceDescriptorV1,
    ReferenceResolutionError,
    RegisteredOperatorV1,
    branch_fingerprint,
    append_operator_turn,
    build_reference_table,
    clone_reference_table_for_branch,
    create_conversation_trace,
    enumerate_operator_legal_set,
    merge_branch_application_sequences,
    replay_branch_sequence_merge,
    replay_conversation_trace,
    sequence_from_branch_edits,
    sequence_merged_continuation_node,
)
from slm_training.dsl.operators.contracts import _fingerprint
from slm_training.dsl.pack import get_pack

SOURCE = (
    'root = Card([TextContent(":hero.title"), '
    'TextContent(":hero.body")], "clear")'
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _provenance(state: OperatorStateV1) -> ApplicationProvenanceV1:
    return ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.sequence_merge_fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(state.source),
        request_id="sequence-merge-request",
    )


def _effect(
    *,
    target,
    category: str,
    coverage: CompilerCoverage,
    consumed_nodes: tuple[NodeRef, ...] = (),
    produced_nodes: tuple[NodeRef, ...] = (),
) -> ActionEffectV1:
    delta = EffectDeltaV1(
        kind=EffectDeltaKind(category),
        target=target,
        before="before",
        after="after",
    )
    values = {f"{category}_deltas": (delta,)}
    return ActionEffectV1(
        **values,
        consumed_nodes=consumed_nodes,
        produced_nodes=produced_nodes,
        compiler_coverage=coverage,
    )


class _Fixture:
    """Builds multi-step ``BranchEditV1`` chains sharing one branch digest.

    Mirrors ``test_operator_merge.py``'s ``_Fixture`` exactly for a single
    step (``start_branch``); ``continue_branch`` is the new capability this
    test module needs -- it applies a second (or Nth) operator whose input
    is a previous step's *output* node on the same branch, exactly the
    shape ``BranchApplicationSequenceV1`` chains.
    """

    def __init__(self) -> None:
        self.pack = get_pack("openui")
        self.state = OperatorStateV1.from_source(self.pack, SOURCE)
        self.root_branch = branch_fingerprint(
            self.state.state_digest, _sha("sequence-merge-root")
        )
        descriptors = tuple(
            ReferenceDescriptorV1(
                ref_kind=RefKind.NODE,
                semantic_fingerprint=_sha(name),
                value_type=f"openui.{name}",
            )
            for name in ("title", "body")
        )
        self.root_table = build_reference_table(
            request_id="sequence-merge-request",
            state_digest=self.state.state_digest,
            branch_digest=self.root_branch,
            descriptors=descriptors,
            seed=1,
        )
        self.base = ConversationStateNodeV1(
            parent_state_id=None,
            branch_digest=self.root_branch,
            state=self.state,
            reference_table=self.root_table,
        )
        self.authorities: dict[str, tuple] = {}
        self._seed = 100

    def _next_seed(self) -> int:
        self._seed += 1
        return self._seed

    def start_branch(self, *, name: str, **kwargs) -> BranchEditV1:
        branch = branch_fingerprint(
            self.state.state_digest, _sha(f"sequence-merge-{name}")
        )
        table = clone_reference_table_for_branch(
            self.root_table, branch_digest=branch, seed=self._next_seed()
        )
        input_node = ConversationStateNodeV1(
            parent_state_id=self.base.state_id,
            branch_digest=branch,
            state=self.state,
            reference_table=table,
        )
        return self._apply_step(input_node, name=name, **kwargs)

    def continue_branch(
        self, previous: BranchEditV1, *, name: str, **kwargs
    ) -> BranchEditV1:
        return self._apply_step(previous.output_node, name=name, **kwargs)

    def _apply_step(
        self,
        input_node: ConversationStateNodeV1,
        *,
        name: str,
        target_name: str,
        replacement: str,
        before: str | None = None,
        category: str = "property",
        coverage: CompilerCoverage = CompilerCoverage.EXACT,
        operator_id: str | None = None,
        commutes_with: tuple[str, ...] = (),
        stale_effect_ref: bool = False,
        consumed_node: bool = False,
        produced_node: bool = False,
    ) -> BranchEditV1:
        branch = input_node.branch_digest
        target = next(
            entry.ref
            for entry in input_node.reference_table.entries
            if entry.descriptor.value_type == f"openui.{target_name}"
        )
        if stale_effect_ref:
            target = next(
                entry.ref
                for entry in self.root_table.entries
                if entry.descriptor.value_type == f"openui.{target_name}"
            )
        operator_id = operator_id or f"openui.seqmerge_{name}_{self._next_seed()}"
        declaration = AstOperatorV1(
            operator_id=operator_id,
            version="v1",
            domain="openui.ast",
            codomain="openui.ast",
            argument_slots=(),
            preconditions=(),
            effect_signature=(EffectDeltaKind(category),),
            locality="node",
            cost=1.0,
            commutes_with=commutes_with,
        )
        marker = before or f":hero.{target_name}"

        def execute(state, _arguments):
            if marker not in state.source:
                raise ValueError("fixture target is unavailable")
            return OperatorMutationV1(
                source=state.source.replace(marker, replacement),
                effect=_effect(
                    target=target,
                    category=category,
                    coverage=coverage,
                    consumed_nodes=(target,) if consumed_node else (),
                    produced_nodes=(target,) if produced_node else (),
                ),
            )

        library = OperatorLibraryV1((RegisteredOperatorV1(declaration, execute),))
        branch_pack = replace(self.pack, operator_library=library)
        applied = library.apply(
            branch_pack,
            input_node.state,
            operator_id,
            (),
            _provenance(input_node.state),
        )
        assert applied.succeeded and applied.state is not None
        output_table = build_reference_table(
            request_id=input_node.reference_table.request_id,
            state_digest=applied.state.state_digest,
            branch_digest=branch,
            descriptors=tuple(
                entry.descriptor for entry in input_node.reference_table.entries
            ),
            seed=self._next_seed(),
        )
        output_node = ConversationStateNodeV1(
            parent_state_id=input_node.state_id,
            branch_digest=branch,
            state=applied.state,
            reference_table=output_table,
        )
        self.authorities[input_node.state_id] = (branch_pack, library)
        return BranchEditV1(input_node, output_node, applied.application)

    def resolve(self, node):
        return self.authorities[node.state_id]

    def rebuild_merged_table(
        self,
        _pack: object,
        state: OperatorStateV1,
        branch_digest: str,
        seed: int,
    ):
        descriptors = tuple(
            ReferenceDescriptorV1(
                ref_kind=RefKind.NODE,
                semantic_fingerprint=_sha(f"canonical:{marker}"),
                value_type=f"openui.{marker.removeprefix(':hero.')}",
            )
            for marker in (":hero.heading", ":hero.copy")
            if marker in state.source
        )
        return build_reference_table(
            request_id="sequence-merge-request",
            state_digest=state.state_digest,
            branch_digest=branch_digest,
            descriptors=descriptors,
            seed=seed,
        )


def test_branch_application_sequence_requires_chained_nonempty_edits() -> None:
    fixture = _Fixture()
    with pytest.raises(ValueError, match="requires at least one step"):
        sequence_from_branch_edits(())

    left_first = fixture.start_branch(
        name="chain_left", target_name="title", replacement=":hero.mid"
    )
    right_first = fixture.start_branch(
        name="chain_right", target_name="body", replacement=":hero.other"
    )
    with pytest.raises(ValueError, match="do not chain"):
        sequence_from_branch_edits((left_first, right_first))

    sequence = sequence_from_branch_edits((left_first,))
    assert isinstance(sequence, BranchApplicationSequenceV1)
    assert sequence.application_ids == (left_first.application.application_id,)
    assert sequence.input_node == left_first.input_node
    assert sequence.output_node == left_first.output_node


def test_disjoint_two_step_and_one_step_sequences_merge_and_are_order_invariant() -> None:
    fixture = _Fixture()
    left_step1 = fixture.start_branch(
        name="left", target_name="title", replacement=":hero.mid"
    )
    left_step2 = fixture.continue_branch(
        left_step1, name="left2", target_name="title", before=":hero.mid",
        replacement=":hero.heading",
    )
    left = sequence_from_branch_edits((left_step1, left_step2))

    right_step1 = fixture.start_branch(
        name="right", target_name="body", replacement=":hero.copy"
    )
    right = sequence_from_branch_edits((right_step1,))

    decision = merge_branch_application_sequences(
        pack=fixture.pack,
        base=fixture.base,
        left=left,
        right=right,
        authority_resolver=fixture.resolve,
    )
    reversed_decision = merge_branch_application_sequences(
        pack=fixture.pack,
        base=fixture.base,
        left=right,
        right=left,
        authority_resolver=fixture.resolve,
    )

    assert decision.succeeded
    assert decision.decision_id == reversed_decision.decision_id
    assert decision.merge is not None
    assert ":hero.heading" in decision.merge.merged_state.source
    assert ":hero.copy" in decision.merge.merged_state.source
    assert (
        OperatorStateV1.from_source(fixture.pack, decision.merge.merged_state.source)
        == decision.merge.merged_state
    )
    assert decision.merge.application_ids == tuple(
        sorted(
            (
                left_step1.application.application_id,
                left_step2.application.application_id,
                right_step1.application.application_id,
            )
        )
    )
    assert replay_branch_sequence_merge(
        pack=fixture.pack,
        base=fixture.base,
        left=left,
        right=right,
        authority_resolver=fixture.resolve,
        recorded=decision,
    ).decision_id == decision.decision_id


def test_conflict_buried_in_a_later_step_is_still_detected() -> None:
    """A conflict on step 2 of a sequence must not be masked by step 1 being safe."""
    fixture = _Fixture()
    left_step1 = fixture.start_branch(
        name="buried_left", target_name="body", replacement=":hero.safe"
    )
    left_step2 = fixture.continue_branch(
        left_step1,
        name="buried_left2",
        target_name="title",
        replacement=":hero.left",
    )
    left = sequence_from_branch_edits((left_step1, left_step2))

    right_step1 = fixture.start_branch(
        name="buried_right", target_name="title", replacement=":hero.right"
    )
    right = sequence_from_branch_edits((right_step1,))

    decision = merge_branch_application_sequences(
        pack=fixture.pack,
        base=fixture.base,
        left=left,
        right=right,
        authority_resolver=fixture.resolve,
    )
    assert not decision.succeeded
    assert decision.conflict is not None
    assert decision.conflict.kind is MergeConflictKind.SAME_NODE_INCOMPATIBLE_EDIT
    assert len(decision.conflict.target_fingerprints) == 1


def test_delete_modify_conflict_detected_when_delete_step_is_not_first() -> None:
    fixture = _Fixture()
    left_step1 = fixture.start_branch(
        name="delmod_left", target_name="body", replacement=":hero.safe"
    )
    left_step2 = fixture.continue_branch(
        left_step1,
        name="delmod_left2",
        target_name="title",
        replacement=":hero.left",
        category="topology",
        consumed_node=True,
    )
    left = sequence_from_branch_edits((left_step1, left_step2))

    right_step1 = fixture.start_branch(
        name="delmod_right", target_name="title", replacement=":hero.right"
    )
    right = sequence_from_branch_edits((right_step1,))

    decision = merge_branch_application_sequences(
        pack=fixture.pack,
        base=fixture.base,
        left=left,
        right=right,
        authority_resolver=fixture.resolve,
    )
    assert decision.conflict is not None
    assert decision.conflict.kind is MergeConflictKind.DELETE_MODIFY


def test_inexact_coverage_anywhere_in_sequence_is_unsupported() -> None:
    fixture = _Fixture()
    left_step1 = fixture.start_branch(
        name="partial_left", target_name="body", replacement=":hero.safe"
    )
    left_step2 = fixture.continue_branch(
        left_step1,
        name="partial_left2",
        target_name="title",
        replacement=":hero.left",
        coverage=CompilerCoverage.APPROXIMATE,
    )
    left = sequence_from_branch_edits((left_step1, left_step2))

    right_step1 = fixture.start_branch(
        name="partial_right", target_name="title", replacement=":hero.right"
    )
    right = sequence_from_branch_edits((right_step1,))

    decision = merge_branch_application_sequences(
        pack=fixture.pack,
        base=fixture.base,
        left=left,
        right=right,
        authority_resolver=fixture.resolve,
    )
    assert decision.conflict is not None
    assert decision.conflict.kind is MergeConflictKind.UNSUPPORTED_EFFECT


def test_stale_ref_in_a_later_step_refuses_without_mutation() -> None:
    fixture = _Fixture()
    left_step1 = fixture.start_branch(
        name="stale_left", target_name="body", replacement=":hero.safe"
    )
    left_step2 = fixture.continue_branch(
        left_step1,
        name="stale_left2",
        target_name="title",
        replacement=":hero.left",
        stale_effect_ref=True,
    )
    left = sequence_from_branch_edits((left_step1, left_step2))

    right_step1 = fixture.start_branch(
        name="stale_right", target_name="title", replacement=":hero.right"
    )
    right = sequence_from_branch_edits((right_step1,))

    decision = merge_branch_application_sequences(
        pack=fixture.pack,
        base=fixture.base,
        left=left,
        right=right,
        authority_resolver=fixture.resolve,
    )
    assert decision.conflict is not None
    assert decision.conflict.kind is MergeConflictKind.STALE_REF


def test_sequence_merge_continuation_rebuilds_fresh_table_and_replays_followup_trace() -> None:
    fixture = _Fixture()
    left_step1 = fixture.start_branch(
        name="continuation_left", target_name="title", replacement=":hero.mid"
    )
    left_step2 = fixture.continue_branch(
        left_step1,
        name="continuation_left2",
        target_name="title",
        before=":hero.mid",
        replacement=":hero.heading",
    )
    left = sequence_from_branch_edits((left_step1, left_step2))

    right_step1 = fixture.start_branch(
        name="continuation_right", target_name="body", replacement=":hero.copy"
    )
    right = sequence_from_branch_edits((right_step1,))

    decision = merge_branch_application_sequences(
        pack=fixture.pack,
        base=fixture.base,
        left=left,
        right=right,
        authority_resolver=fixture.resolve,
        reference_table_builder=fixture.rebuild_merged_table,
    )

    assert decision.succeeded
    assert decision.continuation is not None
    node = sequence_merged_continuation_node(decision)
    assert node.state == decision.merge.merged_state  # type: ignore[union-attr]
    assert node.reference_table.branch_digest == (
        decision.merge.merged_branch_digest  # type: ignore[union-attr]
    )
    assert {
        entry.descriptor.semantic_fingerprint for entry in node.reference_table.entries
    } == {_sha("canonical::hero.heading"), _sha("canonical::hero.copy")}

    old_ref = left.nodes[0].reference_table.entries[0].ref
    with pytest.raises(ReferenceResolutionError, match="ref.stale_state"):
        left.nodes[0].reference_table.resolve(
            old_ref,
            state_digest=node.state.state_digest,
            branch_digest=left.nodes[0].branch_digest,
            expected_kind=RefKind.NODE,
        )
    with pytest.raises(ReferenceResolutionError, match="ref.cross_branch"):
        left.nodes[0].reference_table.resolve(
            old_ref,
            state_digest=left.nodes[0].state.state_digest,
            branch_digest=node.branch_digest,
            expected_kind=RefKind.NODE,
        )

    target = next(
        entry.ref
        for entry in node.reference_table.entries
        if entry.descriptor.value_type == "openui.heading"
    )
    declaration = AstOperatorV1(
        operator_id="openui.advance_after_sequence_merge",
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(),
        preconditions=(),
        effect_signature=(EffectDeltaKind.PROPERTY,),
        locality="node",
        cost=1.0,
    )

    def execute(state, _arguments):
        return OperatorMutationV1(
            source=state.source.replace(":hero.heading", ":hero.final"),
            effect=_effect(
                target=target, category="property", coverage=CompilerCoverage.EXACT
            ),
        )

    library = OperatorLibraryV1((RegisteredOperatorV1(declaration, execute),))
    pack = replace(fixture.pack, operator_library=library)
    provenance = _provenance(node.state)
    legal = enumerate_operator_legal_set(
        pack=pack,
        library=library,
        state=node.state,
        reference_table=node.reference_table,
        provenance=provenance,
    )
    action = legal.entries[0].legal_actions[0]
    applied = library.apply(
        pack, node.state, action.operator_id, action.arguments, provenance
    )
    assert applied.succeeded and applied.state is not None
    output_table = fixture.rebuild_merged_table(
        pack, applied.state, node.branch_digest, decision.continuation.allocation_seed
    )
    trace = create_conversation_trace(
        pack=pack,
        root_state=node.state,
        root_reference_table=node.reference_table,
        provenance=provenance,
    )
    trace = append_operator_turn(
        trace,
        pack=pack,
        library=library,
        application=applied.application,
        output_reference_table=output_table,
    )
    assert (
        replay_conversation_trace(pack=pack, library=library, trace=trace).state
        == applied.state
    )
    assert replay_branch_sequence_merge(
        pack=fixture.pack,
        base=fixture.base,
        left=left,
        right=right,
        authority_resolver=fixture.resolve,
        reference_table_builder=fixture.rebuild_merged_table,
        recorded=decision,
    ).decision_id == decision.decision_id
    assert merge_branch_application_sequences(
        pack=fixture.pack,
        base=fixture.base,
        left=right,
        right=left,
        authority_resolver=fixture.resolve,
        reference_table_builder=fixture.rebuild_merged_table,
    ).decision_id == decision.decision_id


def test_sequence_conflict_identity_is_deterministic_and_provenance_complete() -> None:
    fixture = _Fixture()
    left_step1 = fixture.start_branch(
        name="identity_left", target_name="title", replacement=":hero.mid"
    )
    left_step2 = fixture.continue_branch(
        left_step1,
        name="identity_left2",
        target_name="title",
        before=":hero.mid",
        replacement=":hero.left",
    )
    left = sequence_from_branch_edits((left_step1, left_step2))

    right_step1 = fixture.start_branch(
        name="identity_right", target_name="title", replacement=":hero.right"
    )
    right = sequence_from_branch_edits((right_step1,))

    decision = merge_branch_application_sequences(
        pack=fixture.pack,
        base=fixture.base,
        left=left,
        right=right,
        authority_resolver=fixture.resolve,
    )
    assert decision.conflict is not None
    assert decision.conflict.base_state_id == fixture.base.state_id
    assert decision.conflict.branch_state_ids == tuple(
        sorted((left.output_node.state_id, right.output_node.state_id))
    )
    assert decision.conflict.application_ids == tuple(
        sorted((*left.application_ids, *right.application_ids))
    )
    assert decision.conflict.conflict_id == _fingerprint(
        decision.conflict.to_dict(include_conflict_id=False)
    )
    assert replay_branch_sequence_merge(
        pack=fixture.pack,
        base=fixture.base,
        left=right,
        right=left,
        authority_resolver=fixture.resolve,
        recorded=decision,
    ).decision_id == decision.decision_id
