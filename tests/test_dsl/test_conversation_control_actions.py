from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    CompilerCoverage,
    EffectDeltaKind,
    EffectDeltaV1,
    MergeConflictKind,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorStateV1,
    RefKind,
    ReferenceDescriptorV1,
    RegisteredOperatorV1,
    append_operator_turn,
    branch_fingerprint,
    build_reference_table,
    checkout_conversation_state,
    create_conversation_trace,
    fork_conversation,
)
from slm_training.dsl.operators.control_actions import (
    ConversationControlKind,
    ConversationControlPolicyReportV1,
    ControlSupportVerdict,
    LegalControlActionV1,
    MergeCandidateV1,
    apply_conversation_control_action,
    deterministic_control_priority,
    enumerate_conversation_control_legal_set,
)
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
        compiler_id="openui.control_fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(state.source),
        request_id="control-request",
    )


def _effect(*, target, category: str, coverage: CompilerCoverage = CompilerCoverage.EXACT):
    delta = EffectDeltaV1(kind=EffectDeltaKind(category), target=target, before="before", after="after")
    return ActionEffectV1(**{f"{category}_deltas": (delta,)}, compiler_coverage=coverage)


class _ControlFixture:
    """Builds a real, forkable ``ConversationTraceV1`` with node-ref-bearing effects.

    Unlike ``test_operator_merge.py``'s fixture (which constructs
    ``BranchEditV1`` edges directly, never a real trace), this drives the
    actual ``conversation.py`` API end to end -- ``fork_conversation``,
    ``append_operator_turn``, ``checkout_conversation_state`` -- so the
    control-action legal-set/executor can be exercised against a genuine
    multi-branch trace.
    """

    def __init__(self) -> None:
        self.pack = get_pack("openui")
        self.root_state = OperatorStateV1.from_source(self.pack, SOURCE)
        self.root_branch = branch_fingerprint(
            self.root_state.state_digest, _sha("control-root")
        )
        self.root_table = build_reference_table(
            request_id="control-request",
            state_digest=self.root_state.state_digest,
            branch_digest=self.root_branch,
            descriptors=tuple(
                ReferenceDescriptorV1(
                    ref_kind=RefKind.NODE,
                    semantic_fingerprint=_sha(name),
                    value_type=f"openui.{name}",
                )
                for name in ("title", "body")
            ),
            seed=1,
        )
        self.authorities: dict[str, tuple] = {}

    def new_trace(self):
        trace = create_conversation_trace(
            pack=self.pack,
            root_state=self.root_state,
            root_reference_table=self.root_table,
            provenance=_provenance(self.root_state),
        )
        self.authorities[trace.root_state_id] = (self.pack, OperatorLibraryV1(()))
        return trace

    def resolve(self, node):
        return self.authorities[node.state_id]

    def edit(self, trace, *, target_name: str, replacement: str, category: str = "property", seed: int):
        current = trace.current
        target = next(
            entry.ref
            for entry in current.reference_table.entries
            if entry.descriptor.value_type == f"openui.{target_name}"
        )
        operator_id = f"openui.control_{target_name}_{seed}"
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
        )
        marker = f":hero.{target_name}"

        def execute(state, _arguments):
            return OperatorMutationV1(
                source=state.source.replace(marker, replacement),
                effect=_effect(target=target, category=category),
            )

        library = OperatorLibraryV1((RegisteredOperatorV1(declaration, execute),))
        branch_pack = replace(self.pack, operator_library=library)
        applied = library.apply(
            branch_pack, current.state, operator_id, (), _provenance(current.state)
        )
        assert applied.succeeded and applied.state is not None
        output_table = build_reference_table(
            request_id=current.reference_table.request_id,
            state_digest=applied.state.state_digest,
            branch_digest=current.branch_digest,
            descriptors=tuple(entry.descriptor for entry in current.reference_table.entries),
            seed=seed,
        )
        self.authorities[current.state_id] = (branch_pack, library)
        return append_operator_turn(
            trace,
            pack=branch_pack,
            library=library,
            application=applied.application,
            output_reference_table=output_table,
        )


def test_root_trace_legal_set_permits_only_stop_and_fork() -> None:
    fixture = _ControlFixture()
    trace = fixture.new_trace()
    legal_set = enumerate_conversation_control_legal_set(trace=trace)

    assert set(legal_set.legal_kinds) == {
        ConversationControlKind.STOP,
        ConversationControlKind.FORK,
    }
    undo_entry = next(
        e for e in legal_set.entries if e.kind is ConversationControlKind.UNDO
    )
    assert undo_entry.verdict is ControlSupportVerdict.UNSUPPORTED
    assert undo_entry.rejection_reasons == ("nothing_to_undo",)
    redo_entry = next(
        e for e in legal_set.entries if e.kind is ConversationControlKind.REDO
    )
    assert redo_entry.verdict is ControlSupportVerdict.UNSUPPORTED
    assert redo_entry.rejection_reasons == ("no_children",)
    checkout_entry = next(
        e for e in legal_set.entries if e.kind is ConversationControlKind.CHECKOUT
    )
    assert checkout_entry.verdict is ControlSupportVerdict.UNSUPPORTED
    merge_entry = next(
        e for e in legal_set.entries if e.kind is ConversationControlKind.MERGE_BRANCHES
    )
    assert merge_entry.verdict is ControlSupportVerdict.UNKNOWN
    assert merge_entry.rejection_reasons == ("no_candidates_proposed",)
    # Only STOP and FORK are legal at the root -- this is not a COMPLETE
    # singleton, so no forced action.
    assert legal_set.forced_action is None


def test_edit_undo_redo_checkout_matrix_matches_trace_shape() -> None:
    fixture = _ControlFixture()
    trace = fixture.new_trace()
    root_id = trace.root_state_id

    edited = fixture.edit(trace, target_name="title", replacement=":hero.mid", seed=2)
    child_id = edited.current_state_id

    at_child = enumerate_conversation_control_legal_set(trace=edited)
    assert ConversationControlKind.UNDO in at_child.legal_kinds
    assert ConversationControlKind.REDO not in at_child.legal_kinds
    checkout_targets = {
        action.target_state_id
        for entry in at_child.entries
        if entry.kind is ConversationControlKind.CHECKOUT
        for action in entry.legal_actions
    }
    assert checkout_targets == {root_id}

    undo_result = apply_conversation_control_action(
        edited,
        next(
            a
            for e in at_child.entries
            if e.kind is ConversationControlKind.UNDO
            for a in e.legal_actions
        ),
        provenance=_provenance(edited.current.state),
    )
    assert undo_result.succeeded
    assert undo_result.output_trace is not None
    back_at_root = undo_result.output_trace
    assert back_at_root.current_state_id == root_id

    at_root_after_undo = enumerate_conversation_control_legal_set(trace=back_at_root)
    redo_entry = next(
        e for e in at_root_after_undo.entries if e.kind is ConversationControlKind.REDO
    )
    assert redo_entry.verdict is ControlSupportVerdict.SUPPORTED
    assert [a.target_state_id for a in redo_entry.legal_actions] == [child_id]

    redo_result = apply_conversation_control_action(
        back_at_root, redo_entry.legal_actions[0], provenance=_provenance(back_at_root.current.state)
    )
    assert redo_result.succeeded
    assert redo_result.output_trace.current_state_id == child_id


def test_fork_creates_a_second_redo_target_visible_from_the_shared_parent() -> None:
    fixture = _ControlFixture()
    trace = fixture.new_trace()
    root_id = trace.root_state_id

    left_forked = fork_conversation(
        trace,
        branch_nonce_digest=_sha("left-branch"),
        reference_seed=11,
        provenance=_provenance(trace.current.state),
    )
    left_edited = fixture.edit(left_forked, target_name="title", replacement=":hero.left", seed=12)
    left_tip_id = left_edited.current_state_id

    back_to_root = checkout_conversation_state(
        left_edited, target_state_id=root_id, provenance=_provenance(left_edited.current.state)
    )
    right_forked = fork_conversation(
        back_to_root,
        branch_nonce_digest=_sha("right-branch"),
        reference_seed=21,
        provenance=_provenance(back_to_root.current.state),
    )
    right_edited = fixture.edit(right_forked, target_name="body", replacement=":hero.right", seed=22)
    right_tip_id = right_edited.current_state_id

    checked_back = checkout_conversation_state(
        right_edited, target_state_id=root_id, provenance=_provenance(right_edited.current.state)
    )
    legal_set = enumerate_conversation_control_legal_set(trace=checked_back)
    redo_entry = next(
        e for e in legal_set.entries if e.kind is ConversationControlKind.REDO
    )
    assert redo_entry.verdict is ControlSupportVerdict.SUPPORTED
    assert {a.target_state_id for a in redo_entry.legal_actions} == {
        left_forked.current_state_id,
        right_forked.current_state_id,
    }
    checkout_targets = {
        a.target_state_id
        for e in legal_set.entries
        if e.kind is ConversationControlKind.CHECKOUT
        for a in e.legal_actions
    }
    assert {left_tip_id, right_tip_id} <= checkout_targets


def test_disjoint_merge_branches_candidate_is_legal_and_executes() -> None:
    fixture = _ControlFixture()
    trace = fixture.new_trace()
    base_id = trace.root_state_id

    left_forked = fork_conversation(
        trace,
        branch_nonce_digest=_sha("disjoint-left"),
        reference_seed=11,
        provenance=_provenance(trace.current.state),
    )
    left_edited = fixture.edit(left_forked, target_name="title", replacement=":hero.left", seed=12)
    left_tip_id = left_edited.current_state_id

    back_to_root = checkout_conversation_state(
        left_edited, target_state_id=base_id, provenance=_provenance(left_edited.current.state)
    )
    right_forked = fork_conversation(
        back_to_root,
        branch_nonce_digest=_sha("disjoint-right"),
        reference_seed=21,
        provenance=_provenance(back_to_root.current.state),
    )
    right_edited = fixture.edit(right_forked, target_name="body", replacement=":hero.right", seed=22)
    right_tip_id = right_edited.current_state_id

    candidate = MergeCandidateV1(
        base_state_id=base_id, left_state_id=left_tip_id, right_state_id=right_tip_id
    )
    legal_set = enumerate_conversation_control_legal_set(
        trace=right_edited,
        merge_candidates=(candidate,),
        authority_resolver=fixture.resolve,
    )
    merge_entry = next(
        e for e in legal_set.entries if e.kind is ConversationControlKind.MERGE_BRANCHES
    )
    assert merge_entry.verdict is ControlSupportVerdict.SUPPORTED
    assert len(merge_entry.legal_actions) == 1
    assert merge_entry.legal_actions[0].merge_candidate == candidate

    result = apply_conversation_control_action(
        right_edited,
        merge_entry.legal_actions[0],
        provenance=_provenance(right_edited.current.state),
        authority_resolver=fixture.resolve,
    )
    assert result.succeeded
    assert result.merge_decision is not None and result.merge_decision.succeeded
    assert ":hero.left" in result.merge_decision.merge.merged_state.source
    assert ":hero.right" in result.merge_decision.merge.merged_state.source


def test_overlapping_merge_branches_candidate_is_illegal_with_typed_reason() -> None:
    fixture = _ControlFixture()
    trace = fixture.new_trace()
    base_id = trace.root_state_id

    left_forked = fork_conversation(
        trace,
        branch_nonce_digest=_sha("conflict-left"),
        reference_seed=11,
        provenance=_provenance(trace.current.state),
    )
    left_edited = fixture.edit(left_forked, target_name="title", replacement=":hero.left", seed=12)
    left_tip_id = left_edited.current_state_id

    back_to_root = checkout_conversation_state(
        left_edited, target_state_id=base_id, provenance=_provenance(left_edited.current.state)
    )
    right_forked = fork_conversation(
        back_to_root,
        branch_nonce_digest=_sha("conflict-right"),
        reference_seed=21,
        provenance=_provenance(back_to_root.current.state),
    )
    right_edited = fixture.edit(right_forked, target_name="title", replacement=":hero.right", seed=22)
    right_tip_id = right_edited.current_state_id

    candidate = MergeCandidateV1(
        base_state_id=base_id, left_state_id=left_tip_id, right_state_id=right_tip_id
    )
    legal_set = enumerate_conversation_control_legal_set(
        trace=right_edited,
        merge_candidates=(candidate,),
        authority_resolver=fixture.resolve,
    )
    merge_entry = next(
        e for e in legal_set.entries if e.kind is ConversationControlKind.MERGE_BRANCHES
    )
    assert merge_entry.verdict is ControlSupportVerdict.UNSUPPORTED
    assert merge_entry.legal_actions == ()
    assert merge_entry.rejection_reasons == (
        f"merge_conflict:{MergeConflictKind.SAME_NODE_INCOMPATIBLE_EDIT.value}",
    )


def test_merge_branches_without_authority_resolver_is_unknown_not_unsupported() -> None:
    fixture = _ControlFixture()
    trace = fixture.new_trace()
    base_id = trace.root_state_id
    left_forked = fork_conversation(
        trace, branch_nonce_digest=_sha("no-authority-left"), reference_seed=11,
        provenance=_provenance(trace.current.state),
    )
    left_edited = fixture.edit(left_forked, target_name="title", replacement=":hero.left", seed=12)
    back_to_root = checkout_conversation_state(
        left_edited, target_state_id=base_id, provenance=_provenance(left_edited.current.state)
    )
    right_forked = fork_conversation(
        back_to_root, branch_nonce_digest=_sha("no-authority-right"), reference_seed=21,
        provenance=_provenance(back_to_root.current.state),
    )
    right_edited = fixture.edit(right_forked, target_name="body", replacement=":hero.right", seed=22)

    candidate = MergeCandidateV1(
        base_state_id=base_id,
        left_state_id=left_edited.current_state_id,
        right_state_id=right_edited.current_state_id,
    )
    legal_set = enumerate_conversation_control_legal_set(
        trace=right_edited, merge_candidates=(candidate,)
    )
    merge_entry = next(
        e for e in legal_set.entries if e.kind is ConversationControlKind.MERGE_BRANCHES
    )
    assert merge_entry.verdict is ControlSupportVerdict.UNSUPPORTED
    assert merge_entry.rejection_reasons == ("no_authority_resolver",)


def test_stop_result_carries_only_a_typed_reason_and_never_mutates() -> None:
    fixture = _ControlFixture()
    trace = fixture.new_trace()
    legal_set = enumerate_conversation_control_legal_set(trace=trace)
    stop_action = next(
        a
        for e in legal_set.entries
        if e.kind is ConversationControlKind.STOP
        for a in e.legal_actions
    )
    result = apply_conversation_control_action(
        trace, stop_action, provenance=_provenance(trace.current.state), stop_reason="budget_exhausted"
    )
    assert result.succeeded
    assert result.stop_reason == "budget_exhausted"
    assert result.output_trace is None
    assert result.merge_decision is None
    assert trace.current_state_id == trace.root_state_id  # base trace untouched


def test_legal_control_action_rejects_mismatched_target_shape() -> None:
    with pytest.raises(ValueError, match="requires a target_state_id"):
        LegalControlActionV1(kind=ConversationControlKind.REDO)
    with pytest.raises(ValueError, match="carries no target or merge candidate"):
        LegalControlActionV1(kind=ConversationControlKind.STOP, target_state_id="x" * 64)
    with pytest.raises(ValueError, match="requires a merge_candidate"):
        LegalControlActionV1(kind=ConversationControlKind.MERGE_BRANCHES)


def test_deterministic_priority_bypasses_forced_and_empty_sets_then_orders_by_kind() -> None:
    fixture = _ControlFixture()
    trace = fixture.new_trace()

    root_only = enumerate_conversation_control_legal_set(trace=trace)
    decision = deterministic_control_priority(root_only)
    # STOP and FORK are both legal at the root -- not forced, and FORK
    # outranks STOP in the fixed priority order.
    assert decision.decision_kind == "scored"
    assert decision.action is not None
    assert decision.action.kind is ConversationControlKind.FORK

    edited = fixture.edit(trace, target_name="title", replacement=":hero.mid", seed=2)
    at_child = enumerate_conversation_control_legal_set(trace=edited)
    child_decision = deterministic_control_priority(at_child)
    assert child_decision.decision_kind == "scored"
    # UNDO, CHECKOUT, and FORK are all legal here; UNDO outranks both.
    assert child_decision.action.kind is ConversationControlKind.UNDO


def test_conversation_control_policy_report_reuses_calibration_helpers() -> None:
    stop_action = LegalControlActionV1(
        kind=ConversationControlKind.STOP, proof_checks=("always_legal",), serialized="CONTROL STOP"
    )
    undo_action = LegalControlActionV1(
        kind=ConversationControlKind.UNDO, proof_checks=("has_parent",), serialized="CONTROL UNDO"
    )
    report = ConversationControlPolicyReportV1.from_predictions(
        predicted_actions=[stop_action, undo_action, undo_action],
        expected_actions=[stop_action, undo_action, stop_action],
        stop_probabilities=[0.9, 0.1, 0.4],
        stop_targets=[1, 0, 0],
        premature=[False, False, True],
        late=[False, False, False],
    )
    assert report.command_accuracy == pytest.approx(2 / 3)
    assert report.argument_accuracy == pytest.approx(2 / 3)
    assert 0.0 <= report.stop_brier <= 1.0
    assert 0.0 <= report.stop_ece <= 1.0
    assert report.premature_stop_rate == pytest.approx(1 / 3)
    assert report.late_stop_rate == 0.0
