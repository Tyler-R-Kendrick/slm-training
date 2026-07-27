"""Tests for SLM-418 (DSH5-10) replay-grounded preference row extraction."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from slm_training.dsl.operators import (
    ActionEffectV1,
    AstOperatorV1,
    BindingPhase,
    BoundArgumentV1,
    CompilerCoverage,
    OperatorArgumentSlotV1,
    OperatorEventMemoryReportV1,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorStateV1,
    RefKind,
    ReferenceDescriptorV1,
    RegisteredOperatorV1,
    ReplayPreferenceRelation,
    append_operator_turn,
    branch_fingerprint,
    build_reference_table,
    checkout_conversation_state,
    create_conversation_trace,
    extract_merge_preference_row,
    extract_replay_preference_rows,
    fork_conversation,
    merge_conversation_branches,
    redo_conversation,
    serialize_operator_action,
    undo_conversation,
)
from slm_training.dsl.pack import get_pack
from tests.test_dsl.test_operator_conversation import _append, _fixture, _provenance
from tests.test_dsl.test_operator_merge import _Fixture as _MergeFixture
from tests.test_dsl.test_operator_merge import _provenance as _merge_provenance


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_PRONOUN_SOURCE = 'root = TextContent(":hero.title")'
_PRONOUN_OPERATOR_ID = "openui.fixture_pronoun_focus"
_PRONOUN_TABLE_SEED = 9
_PRONOUN_TARGETS = {0: ":hero.target_a", 1: ":hero.target_b"}


def _pronoun_descriptor(index: int) -> ReferenceDescriptorV1:
    return ReferenceDescriptorV1(
        ref_kind=RefKind.VALUE,
        semantic_fingerprint=_sha(f"pronoun-target-{index}"),
        value_type="openui.string",
    )


def _pronoun_focus_fixture():
    """Two always-legal ``VALUE`` refs for one operator: a genuine sibling pair.

    Unlike the zero-argument ``_fixture()`` every other pattern in this
    module reuses, pronoun-focus classification needs an operator whose
    argument choice is itself ambiguous -- two distinct, simultaneously
    legal refs the user could target -- so "continued focus" has a real
    "switched to the untouched sibling" alternative to be preferred over.
    Both refs stay legal (and keep the same opaque IDs) at every state this
    fixture reaches: they are allocated from a fixed seed + descriptor pair,
    which ``build_reference_table`` derives independently of any state
    digest, so focus overlap can be checked by direct ref equality.
    """
    base_pack = get_pack("openui")
    root_state = OperatorStateV1.from_source(base_pack, _PRONOUN_SOURCE)
    descriptors = tuple(_pronoun_descriptor(index) for index in (0, 1))
    branch = branch_fingerprint(root_state.state_digest, _sha("pronoun-branch"))

    def table_for(state):
        return build_reference_table(
            request_id="request-1",
            state_digest=state.state_digest,
            branch_digest=branch,
            descriptors=descriptors,
            seed=_PRONOUN_TABLE_SEED,
        )

    root_table = table_for(root_state)
    semantic_by_ref = {
        entry.ref: entry.descriptor.semantic_fingerprint for entry in root_table.entries
    }
    index_by_semantic = {
        descriptor.semantic_fingerprint: index
        for index, descriptor in enumerate(descriptors)
    }
    refs_by_index = {
        index_by_semantic[entry.descriptor.semantic_fingerprint]: entry.ref
        for entry in root_table.entries
    }

    def execute(_state, arguments):
        index = index_by_semantic[semantic_by_ref[arguments[0].value]]
        return OperatorMutationV1(
            source=f'root = TextContent("{_PRONOUN_TARGETS[index]}")',
            effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
        )

    declaration = AstOperatorV1(
        operator_id=_PRONOUN_OPERATOR_ID,
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(
            OperatorArgumentSlotV1("value", RefKind.VALUE, BindingPhase.APPLICATION),
        ),
        preconditions=(),
        effect_signature=(),
        locality="node",
        cost=1.0,
    )
    library = OperatorLibraryV1((RegisteredOperatorV1(declaration, execute),))
    pack = replace(base_pack, operator_library=library)
    trace = create_conversation_trace(
        pack=pack,
        root_state=root_state,
        root_reference_table=root_table,
        provenance=_provenance(root_state),
    )
    return pack, library, trace, table_for, refs_by_index


def _append_pronoun(pack, library, trace, ref, *, table_for):
    result = library.apply(
        pack,
        trace.current.state,
        _PRONOUN_OPERATOR_ID,
        (BoundArgumentV1("value", ref),),
        _provenance(trace.current.state),
    )
    assert result.succeeded
    assert result.state is not None
    return append_operator_turn(
        trace,
        pack=pack,
        library=library,
        application=result.application,
        output_reference_table=table_for(result.state),
    )


def test_edit_then_undo_yields_one_row_grounded_in_the_legal_set() -> None:
    pack, library, root = _fixture()
    edited, _application = _append(pack, library, root)
    undone = undo_conversation(edited, provenance=_provenance(edited.current.state))

    report = extract_replay_preference_rows(
        undone, pack=pack, library=library, provenance_for=_provenance
    )

    assert isinstance(report, OperatorEventMemoryReportV1)
    assert report.counts_by_relation == {
        ReplayPreferenceRelation.EDIT_THEN_UNDO.value: 1
    }
    row = report.rows[0]
    assert row.input_state_id == edited.current_state_id
    assert row.chosen_action == "undo"
    assert row.chosen_output_state_id == root.root_state_id
    assert row.rejected_action != row.chosen_action
    assert row.correction_reason == "user_undid_without_redo"


def test_undo_then_redo_yields_one_row_preferring_redo() -> None:
    pack, library, root = _fixture()
    edited, _application = _append(pack, library, root)
    original_child_id = edited.current_state_id
    undone = undo_conversation(edited, provenance=_provenance(edited.current.state))
    redone = redo_conversation(
        undone,
        target_state_id=original_child_id,
        provenance=_provenance(undone.current.state),
    )

    report = extract_replay_preference_rows(
        redone, pack=pack, library=library, provenance_for=_provenance
    )

    relations = [row.semantic_relation for row in report.rows]
    assert ReplayPreferenceRelation.UNDO_THEN_REDO in relations
    redo_row = next(
        row
        for row in report.rows
        if row.semantic_relation is ReplayPreferenceRelation.UNDO_THEN_REDO
    )
    assert redo_row.input_state_id == root.root_state_id
    assert redo_row.chosen_action == f"redo:{original_child_id}"
    assert redo_row.chosen_output_state_id == original_child_id
    assert redo_row.correction_reason == "user_redid_after_reconsidering"


def test_partial_rollback_yields_a_row_for_the_second_consecutive_undo() -> None:
    """A second undo, not preceded by an intervening edit, is its own pattern.

    root -> edit1 -> edit2 -> undo (edit2's state) -> undo (edit1's state):
    the first undo is edit-then-undo (edit2 was just made); the second undo
    -- taken instead of redo/checkout/a-new-edit at edit1's state -- is
    partial_rollback, distinct from both edit_then_undo and undo_then_redo.
    """
    pack, library, root = _fixture()
    edited_once, _application_one = _append(pack, library, root)
    after_edit1_id = edited_once.current_state_id
    edited_twice, _application_two = _append(pack, library, edited_once, seed=3)
    after_edit2_id = edited_twice.current_state_id

    undone_once = undo_conversation(
        edited_twice, provenance=_provenance(edited_twice.current.state)
    )
    assert undone_once.current_state_id == after_edit1_id
    undone_twice = undo_conversation(
        undone_once, provenance=_provenance(undone_once.current.state)
    )
    assert undone_twice.current_state_id == root.root_state_id

    report = extract_replay_preference_rows(
        undone_twice, pack=pack, library=library, provenance_for=_provenance
    )

    assert report.counts_by_relation == {
        ReplayPreferenceRelation.EDIT_THEN_UNDO.value: 1,
        ReplayPreferenceRelation.PARTIAL_ROLLBACK.value: 1,
    }
    rollback_row = next(
        row
        for row in report.rows
        if row.semantic_relation is ReplayPreferenceRelation.PARTIAL_ROLLBACK
    )
    assert rollback_row.input_state_id == after_edit1_id
    assert rollback_row.chosen_action == "undo"
    assert rollback_row.chosen_output_state_id == root.root_state_id
    assert rollback_row.rejected_action != "undo"
    assert rollback_row.correction_reason == (
        "user_continued_rollback_past_first_undo"
    )

    edit_row = next(
        row
        for row in report.rows
        if row.semantic_relation is ReplayPreferenceRelation.EDIT_THEN_UNDO
    )
    assert edit_row.input_state_id == after_edit2_id
    assert edit_row.chosen_output_state_id == after_edit1_id


def test_checkout_another_state_yields_a_row_preferring_checkout_over_undo() -> None:
    """checkout to the same destination undo would reach is its own pattern.

    root -> edit1 -> checkout(root): the user reached the exact same
    destination undo would have, but through the distinct checkout tool
    invocation -- the row records that choice, not an inferred undo.
    """
    pack, library, root = _fixture()
    edited, _application = _append(pack, library, root)
    after_edit_id = edited.current_state_id

    checked_out = checkout_conversation_state(
        edited,
        target_state_id=root.root_state_id,
        provenance=_provenance(edited.current.state),
    )
    assert checked_out.current_state_id == root.root_state_id

    report = extract_replay_preference_rows(
        checked_out, pack=pack, library=library, provenance_for=_provenance
    )

    assert report.counts_by_relation == {
        ReplayPreferenceRelation.CHECKOUT_ANOTHER_STATE.value: 1
    }
    row = report.rows[0]
    assert row.input_state_id == after_edit_id
    assert row.chosen_action == f"checkout:{root.root_state_id}"
    assert row.chosen_output_state_id == root.root_state_id
    assert row.rejected_action != row.chosen_action
    assert row.correction_reason == "user_checked_out_alternate_state"


def test_checkout_row_replays_independently_to_its_recorded_output_state() -> None:
    pack, library, root = _fixture()
    edited, _application = _append(pack, library, root)
    checked_out = checkout_conversation_state(
        edited,
        target_state_id=root.root_state_id,
        provenance=_provenance(edited.current.state),
    )

    report = extract_replay_preference_rows(
        checked_out, pack=pack, library=library, provenance_for=_provenance
    )

    row = report.rows[0]
    replayed = checkout_conversation_state(
        edited,
        target_state_id=root.root_state_id,
        provenance=_provenance(edited.current.state),
    )
    assert replayed.current_state_id == row.chosen_output_state_id


def test_fork_then_return_to_original_branch_yields_a_distinct_relation() -> None:
    """A checkout back across a fork boundary is not plain checkout-another-state.

    root -> edit (main branch, state D) -> fork (new branch, state F) ->
    checkout(D): the user opened a second branch at D, then explicitly
    returned to the pre-fork branch instead of continuing on the fork --
    that is fork_then_choose_one_branch, not checkout_another_state, even
    though the same ``checkout_conversation_state`` primitive is used.
    """
    pack, library, root = _fixture()
    edited, _application = _append(pack, library, root)
    main_branch_state_id = edited.current_state_id

    forked = fork_conversation(
        edited,
        branch_nonce_digest=_sha("fork-branch"),
        reference_seed=7,
        provenance=_provenance(edited.current.state),
    )
    assert forked.current.branch_digest != edited.current.branch_digest

    checked_out = checkout_conversation_state(
        forked,
        target_state_id=main_branch_state_id,
        provenance=_provenance(forked.current.state),
    )

    report = extract_replay_preference_rows(
        checked_out, pack=pack, library=library, provenance_for=_provenance
    )

    assert report.counts_by_relation == {
        ReplayPreferenceRelation.FORK_THEN_CHOOSE_ONE_BRANCH.value: 1
    }
    row = report.rows[0]
    assert row.input_state_id == forked.current_state_id
    assert row.chosen_action == f"checkout:{main_branch_state_id}"
    assert row.chosen_output_state_id == main_branch_state_id
    assert row.rejected_action != row.chosen_action
    assert row.correction_reason == "user_chose_a_different_branch_after_fork"


def test_checkout_between_two_forked_branches_is_fork_then_choose_one_branch() -> None:
    """Choosing between two genuinely divergent forks, not just undoing a fork.

    root -> edit (D) -> fork twice from D (branches F1, F2) -> checkout from
    F1 to F2: neither destination is the pre-fork branch, so this is the
    most literal reading of "fork then choose one branch."
    """
    pack, library, root = _fixture()
    edited, _application = _append(pack, library, root)

    fork_one = fork_conversation(
        edited,
        branch_nonce_digest=_sha("fork-one"),
        reference_seed=21,
        provenance=_provenance(edited.current.state),
    )
    # Return to the pre-fork state so the second fork also branches from D.
    back_to_edited = checkout_conversation_state(
        fork_one,
        target_state_id=edited.current_state_id,
        provenance=_provenance(fork_one.current.state),
    )
    fork_two = fork_conversation(
        back_to_edited,
        branch_nonce_digest=_sha("fork-two"),
        reference_seed=22,
        provenance=_provenance(back_to_edited.current.state),
    )
    assert fork_two.current.branch_digest not in (
        edited.current.branch_digest,
        fork_one.current.branch_digest,
    )

    checked_out = checkout_conversation_state(
        fork_two,
        target_state_id=fork_one.current_state_id,
        provenance=_provenance(fork_two.current.state),
    )

    report = extract_replay_preference_rows(
        checked_out, pack=pack, library=library, provenance_for=_provenance
    )

    # Two fork-boundary-crossing checkouts appear in this trace: the earlier
    # "return to D before opening the second fork" and this test's actual
    # subject, F2 -> F1. Both are correctly fork_then_choose_one_branch;
    # isolate the one this test is about by its exact input state.
    fork_choice_rows = [
        row
        for row in report.rows
        if row.semantic_relation is ReplayPreferenceRelation.FORK_THEN_CHOOSE_ONE_BRANCH
    ]
    assert len(fork_choice_rows) == 2
    row = next(
        row for row in fork_choice_rows if row.input_state_id == fork_two.current_state_id
    )
    assert row.chosen_action == f"checkout:{fork_one.current_state_id}"
    assert row.chosen_output_state_id == fork_one.current_state_id


def test_no_rows_without_any_history_operation() -> None:
    pack, library, root = _fixture()
    edited, _application = _append(pack, library, root)

    report = extract_replay_preference_rows(
        edited, pack=pack, library=library, provenance_for=_provenance
    )

    assert report.rows == ()
    assert report.counts_by_relation == {}


def test_rows_replay_independently_to_their_recorded_output_state() -> None:
    """Acceptance: every row shares one exact input state and replays.

    Re-derive the chosen action from ``row.input_state_id`` on ``edited``
    (independently of the trace the row was extracted from) and confirm it
    lands on exactly ``row.chosen_output_state_id`` -- never an assumed or
    merely-existing state.
    """
    pack, library, root = _fixture()
    edited, _application = _append(pack, library, root)
    undone = undo_conversation(edited, provenance=_provenance(edited.current.state))

    report = extract_replay_preference_rows(
        undone, pack=pack, library=library, provenance_for=_provenance
    )

    row = report.rows[0]
    assert row.input_state_id == edited.current_state_id
    replayed = undo_conversation(edited, provenance=_provenance(edited.current.state))
    assert replayed.current_state_id == row.chosen_output_state_id
    assert row.chosen_output_state_id == root.root_state_id


def test_report_carries_a_version_stamp() -> None:
    pack, library, root = _fixture()
    edited, _application = _append(pack, library, root)
    undone = undo_conversation(edited, provenance=_provenance(edited.current.state))

    report = extract_replay_preference_rows(
        undone, pack=pack, library=library, provenance_for=_provenance
    )

    assert report.version_stamp["stamp_schema"]
    assert report.version_stamp["components"]["dsl.operators.replay_preference"]
    assert report.to_dict()["version_stamp"] == report.version_stamp


def test_merge_success_yields_a_row_preferring_merge_over_checkout_or_undo() -> None:
    """A clean, disjoint-target merge is a legal candidate at the fork tip.

    Two branches fork from a common base and edit disjoint targets (title
    vs body) -- the merge succeeds, and the row records that the user chose
    ``merge:<pair>`` over the equally-legal alternative of just checking out
    to the sibling branch instead.
    """
    fixture = _MergeFixture()
    left = fixture.branch(name="left", target_name="title", replacement=":hero.heading")
    right = fixture.branch(name="right", target_name="body", replacement=":hero.copy")
    decision = merge_conversation_branches(
        pack=fixture.pack,
        base=fixture.base,
        left=left,
        right=right,
        authority_resolver=fixture.resolve,
        reference_table_builder=fixture.rebuild_merged_table,
    )
    assert decision.succeeded

    row = extract_merge_preference_row(
        left=left,
        right=right,
        decision=decision,
        authority_resolver=fixture.resolve,
        provenance_for=_merge_provenance,
    )

    assert row is not None
    assert row.semantic_relation is ReplayPreferenceRelation.MERGE_SUCCESS
    assert row.input_state_id == left.output_node.state_id
    assert row.chosen_action.startswith("merge:")
    assert row.chosen_output_state_id == decision.continuation.merged_node.state_id
    assert row.rejected_action != row.chosen_action
    assert row.correction_reason == "user_merged_diverged_branches"


def test_merge_success_row_replays_independently_to_the_same_merged_state() -> None:
    fixture = _MergeFixture()
    left = fixture.branch(name="left2", target_name="title", replacement=":hero.heading")
    right = fixture.branch(name="right2", target_name="body", replacement=":hero.copy")
    decision = merge_conversation_branches(
        pack=fixture.pack,
        base=fixture.base,
        left=left,
        right=right,
        authority_resolver=fixture.resolve,
        reference_table_builder=fixture.rebuild_merged_table,
    )

    row = extract_merge_preference_row(
        left=left,
        right=right,
        decision=decision,
        authority_resolver=fixture.resolve,
        provenance_for=_merge_provenance,
    )
    assert row is not None

    replayed = merge_conversation_branches(
        pack=fixture.pack,
        base=fixture.base,
        left=left,
        right=right,
        authority_resolver=fixture.resolve,
        reference_table_builder=fixture.rebuild_merged_table,
    )
    assert replayed.continuation is not None
    assert replayed.continuation.merged_node.state_id == row.chosen_output_state_id


def test_merge_conflict_never_yields_a_preference_row() -> None:
    """A conflicting merge is excluded from the ranking denominator entirely.

    Both branches edit the same target -- ``merge_conversation_branches``
    returns a typed ``SAME_NODE_INCOMPATIBLE_EDIT`` conflict, never a
    merged state. Per the module's honesty rule, this yields no row (there
    is no successor state to replay to and no recorded "chosen instead"
    action), rather than a fabricated preference.
    """
    fixture = _MergeFixture()
    left = fixture.branch(
        name="conflict_left", target_name="title", replacement=":hero.left"
    )
    right = fixture.branch(
        name="conflict_right", target_name="title", replacement=":hero.right"
    )
    decision = merge_conversation_branches(
        pack=fixture.pack,
        base=fixture.base,
        left=left,
        right=right,
        authority_resolver=fixture.resolve,
    )
    assert not decision.succeeded

    row = extract_merge_preference_row(
        left=left,
        right=right,
        decision=decision,
        authority_resolver=fixture.resolve,
        provenance_for=_merge_provenance,
    )

    assert row is None


def test_pronoun_focus_followup_prefers_continuing_the_touched_ref() -> None:
    """Two edits, both targeting the ref the first one touched, is the pattern.

    root -> edit(ref0) -> edit(ref0 again): at the second edit's decision
    state, ref1 was an equally legal target for the same operator (a genuine
    sibling), but the user's follow-up kept operating on the ref they had
    just touched -- exactly the "it" continuation the pattern models,
    without ever consulting any transcript text.
    """
    pack, library, trace, table_for, refs_by_index = _pronoun_focus_fixture()
    edited_once = _append_pronoun(pack, library, trace, refs_by_index[0], table_for=table_for)
    edited_twice = _append_pronoun(
        pack, library, edited_once, refs_by_index[0], table_for=table_for
    )

    report = extract_replay_preference_rows(
        edited_twice, pack=pack, library=library, provenance_for=_provenance
    )

    assert report.counts_by_relation == {
        ReplayPreferenceRelation.PRONOUN_FOCUS_FOLLOWUP.value: 1
    }
    row = report.rows[0]
    assert row.input_state_id == edited_once.current_state_id
    assert row.chosen_action == serialize_operator_action(
        _PRONOUN_OPERATOR_ID, (BoundArgumentV1("value", refs_by_index[0]),)
    )
    assert row.rejected_action == serialize_operator_action(
        _PRONOUN_OPERATOR_ID, (BoundArgumentV1("value", refs_by_index[1]),)
    )
    assert row.chosen_output_state_id == edited_twice.current_state_id
    assert row.correction_reason == (
        "user_continued_implicit_focus_over_sibling_target"
    )


def test_pronoun_focus_row_replays_independently_to_its_recorded_output_state() -> None:
    pack, library, trace, table_for, refs_by_index = _pronoun_focus_fixture()
    edited_once = _append_pronoun(pack, library, trace, refs_by_index[0], table_for=table_for)
    edited_twice = _append_pronoun(
        pack, library, edited_once, refs_by_index[0], table_for=table_for
    )

    report = extract_replay_preference_rows(
        edited_twice, pack=pack, library=library, provenance_for=_provenance
    )
    row = report.rows[0]

    replayed = _append_pronoun(
        pack, library, edited_once, refs_by_index[0], table_for=table_for
    )
    assert replayed.current_state_id == row.chosen_output_state_id


def test_switching_to_the_untouched_sibling_does_not_yield_a_pronoun_focus_row() -> None:
    """Switching targets is honestly out of scope: it is not asserted a mistake.

    root -> edit(ref0) -> edit(ref1): the follow-up explicitly moved to the
    sibling ref instead of continuing focus. This is the "exact named
    reference" case the issue's own matrix lists as distinct from "pronoun
    focus" -- no row is fabricated for it here.
    """
    pack, library, trace, table_for, refs_by_index = _pronoun_focus_fixture()
    edited_once = _append_pronoun(pack, library, trace, refs_by_index[0], table_for=table_for)
    edited_twice = _append_pronoun(
        pack, library, edited_once, refs_by_index[1], table_for=table_for
    )

    report = extract_replay_preference_rows(
        edited_twice, pack=pack, library=library, provenance_for=_provenance
    )

    assert report.rows == ()
    assert report.counts_by_relation == {}


def test_pronoun_focus_followup_needs_an_established_focus() -> None:
    """A first edit from the trace root has no predecessor to continue focus from.

    Only one ``AST_EDIT`` exists in this trace, so the turn-pair scan never
    fires -- there is nothing to compare the (nonexistent) follow-up
    against, confirming the pattern never fabricates a focus out of a single
    edit.
    """
    pack, library, trace, table_for, refs_by_index = _pronoun_focus_fixture()
    edited_once = _append_pronoun(pack, library, trace, refs_by_index[0], table_for=table_for)

    report = extract_replay_preference_rows(
        edited_once, pack=pack, library=library, provenance_for=_provenance
    )

    assert report.rows == ()

