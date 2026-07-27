"""Tests for SLM-418 (DSH5-10) replay-preference-row -> PreferencePair rendering."""

from __future__ import annotations

from slm_training.dsl.operators import (
    BoundArgumentV1,
    OperatorReplayPreferenceRowV1,
    ReplayPreferenceRelation,
    checkout_conversation_state,
    extract_merge_preference_row,
    extract_replay_preference_rows,
    merge_conversation_branches,
    redo_conversation,
    serialize_operator_action,
    undo_conversation,
)
from slm_training.harnesses.preference import PreferencePair
from slm_training.harnesses.preference.replay_pairs import (
    merge_node_resolver,
    render_replay_preference_pair,
    render_replay_preference_pairs,
)
from tests.test_dsl.test_operator_conversation import _append, _fixture, _provenance
from tests.test_dsl.test_operator_merge import _Fixture as _MergeFixture
from tests.test_dsl.test_operator_merge import _provenance as _merge_provenance
from tests.test_dsl.test_replay_preference import (
    _PRONOUN_OPERATOR_ID,
    _append_pronoun,
    _pronoun_focus_fixture,
)


def test_edit_then_undo_row_renders_a_pair_matching_the_trace_states() -> None:
    pack, library, root = _fixture()
    edited, _application = _append(pack, library, root)
    undone = undo_conversation(edited, provenance=_provenance(edited.current.state))

    report = extract_replay_preference_rows(
        undone, pack=pack, library=library, provenance_for=_provenance
    )
    row = report.rows[0]
    assert row.semantic_relation is ReplayPreferenceRelation.EDIT_THEN_UNDO

    pair = render_replay_preference_pair(
        row,
        resolve_node=undone.node,
        pack=pack,
        library=library,
        provenance_for=_provenance,
    )

    assert isinstance(pair, PreferencePair)
    assert pair.prompt == undone.node(row.input_state_id).state.source
    assert pair.chosen == undone.node(row.chosen_output_state_id).state.source
    # "undo" chosen -> the rejected side must be some other legal candidate's
    # own resulting text, never the same text as the chosen undo target.
    assert pair.chosen == root.node(root.root_state_id).state.source
    assert pair.rejected != pair.chosen
    assert pair.meta["pair_corpus"] == "replay_preference"
    assert pair.meta["semantic_relation"] == "edit_then_undo"
    assert pair.meta["chosen_action"] == "undo"
    assert pair.meta["rejected_action"] == row.rejected_action


def test_undo_then_redo_row_declines_a_degenerate_collapse() -> None:
    """The fixture's rejected alternative here happens to reproduce ``chosen`` verbatim.

    ``_fixture()``'s operator is deterministic and zero-argument, so at the
    root state its only candidate (the alphabetically-first rejected
    action, since ``"OPERATOR ..."`` sorts before ``"checkout:..."``/
    ``"undo"``) reapplies the exact same transition ``redo`` already
    reached -- title -> body either way. Rendering that as a preference
    pair would tell the model the identical text is simultaneously
    preferred and rejected, so the renderer's dedup guard must decline it
    rather than emit a self-contradictory pair.
    """
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
    row = next(
        r
        for r in report.rows
        if r.semantic_relation is ReplayPreferenceRelation.UNDO_THEN_REDO
    )

    pair = render_replay_preference_pair(
        row,
        resolve_node=redone.node,
        pack=pack,
        library=library,
        provenance_for=_provenance,
    )

    assert pair is None


def test_checkout_another_state_row_renders_via_the_operator_reapplication_branch() -> None:
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
    row = next(
        r
        for r in report.rows
        if r.semantic_relation is ReplayPreferenceRelation.CHECKOUT_ANOTHER_STATE
    )

    pair = render_replay_preference_pair(
        row,
        resolve_node=checked_out.node,
        pack=pack,
        library=library,
        provenance_for=_provenance,
    )

    assert pair is not None
    assert pair.chosen == checked_out.node(row.chosen_output_state_id).state.source
    assert pair.rejected != pair.chosen


def test_rejected_undo_action_resolves_to_the_input_states_own_parent() -> None:
    """Directly exercises the ``"undo"``-rejected branch (a control lookup, no apply).

    Constructed directly rather than relying on the fixture's own
    alphabetical tie-break (its ``"OPERATOR ..."`` candidate always sorts
    first), so this proves the parent-lookup path itself, independent of
    which branch a real extracted row happens to land on. root -> edit1 ->
    edit2, so ``edit2``'s parent (edit1, "body") is textually distinct from
    the row's chosen destination (root, "title").
    """
    pack, library, root = _fixture()
    edited_once, _application1 = _append(pack, library, root)
    edited_twice, _application2 = _append(pack, library, edited_once, seed=3)

    contrived = OperatorReplayPreferenceRowV1(
        input_state_id=edited_twice.current_state_id,
        chosen_action=f"checkout:{root.root_state_id}",
        rejected_action="undo",
        chosen_output_state_id=root.root_state_id,
        semantic_relation=ReplayPreferenceRelation.CHECKOUT_ANOTHER_STATE,
        correction_reason="user_checked_out_alternate_state",
        legal_set_fingerprint="0" * 64,
    )

    pair = render_replay_preference_pair(
        contrived,
        resolve_node=edited_twice.node,
        pack=pack,
        library=library,
        provenance_for=_provenance,
    )

    assert pair is not None
    assert pair.chosen == root.node(root.root_state_id).state.source
    assert pair.rejected == edited_once.node(edited_once.current_state_id).state.source
    assert pair.rejected != pair.chosen


def test_pronoun_focus_followup_row_renders_rejected_by_reapplying_the_sibling() -> None:
    pack, library, trace, table_for, refs_by_index = _pronoun_focus_fixture()
    edited_once = _append_pronoun(pack, library, trace, refs_by_index[0], table_for=table_for)
    edited_twice = _append_pronoun(
        pack, library, edited_once, refs_by_index[0], table_for=table_for
    )

    report = extract_replay_preference_rows(
        edited_twice, pack=pack, library=library, provenance_for=_provenance
    )
    row = report.rows[0]
    assert row.semantic_relation is ReplayPreferenceRelation.PRONOUN_FOCUS_FOLLOWUP
    assert row.rejected_action == serialize_operator_action(
        _PRONOUN_OPERATOR_ID, (BoundArgumentV1("value", refs_by_index[1]),)
    )

    pair = render_replay_preference_pair(
        row,
        resolve_node=edited_twice.node,
        pack=pack,
        library=library,
        provenance_for=_provenance,
    )

    assert pair is not None
    assert pair.chosen == edited_twice.node(row.chosen_output_state_id).state.source
    # The rejected side was never applied to any trace node -- it is only
    # derivable by re-running the sibling legal action through the same
    # pack-authorized executor, which is exactly what the renderer does.
    replayed_sibling = library.apply(
        pack,
        edited_once.current.state,
        _PRONOUN_OPERATOR_ID,
        (BoundArgumentV1("value", refs_by_index[1]),),
        _provenance(edited_once.current.state),
    )
    assert replayed_sibling.succeeded and replayed_sibling.state is not None
    assert pair.rejected == replayed_sibling.state.source
    assert pair.rejected != pair.chosen


def test_merge_success_row_renders_via_merge_node_resolver() -> None:
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

    pack, library = fixture.resolve(left.input_node)
    pair = render_replay_preference_pair(
        row,
        resolve_node=merge_node_resolver(left, right, decision),
        pack=pack,
        library=library,
        provenance_for=_merge_provenance,
    )

    assert pair is not None
    assert pair.prompt == left.output_node.state.source
    assert pair.chosen == decision.continuation.merged_node.state.source
    assert pair.rejected != pair.chosen
    assert pair.meta["semantic_relation"] == "merge_success"


def test_rejected_merge_action_is_never_rendered() -> None:
    """Defensive branch: a hypothetical rejected ``merge:<pair>`` renders ``None``.

    Not reachable through today's single-merge-candidate extraction (see
    module docstring), but the renderer must never guess a merged state
    instead of honestly declining.
    """
    pack, library, root = _fixture()
    edited, _application = _append(pack, library, root)

    contrived = OperatorReplayPreferenceRowV1(
        input_state_id=edited.current_state_id,
        chosen_action="undo",
        rejected_action=f"merge:{edited.current_state_id},{root.root_state_id}",
        chosen_output_state_id=root.root_state_id,
        semantic_relation=ReplayPreferenceRelation.EDIT_THEN_UNDO,
        correction_reason="user_undid_without_redo",
        legal_set_fingerprint="0" * 64,
    )

    pair = render_replay_preference_pair(
        contrived,
        resolve_node=edited.node,
        pack=pack,
        library=library,
        provenance_for=_provenance,
    )

    assert pair is None


def test_render_replay_preference_pairs_batches_a_full_report() -> None:
    pack, library, root = _fixture()
    edited, _application = _append(pack, library, root)
    undone = undo_conversation(edited, provenance=_provenance(edited.current.state))

    report = extract_replay_preference_rows(
        undone, pack=pack, library=library, provenance_for=_provenance
    )

    pairs = render_replay_preference_pairs(
        report.rows,
        resolve_node=undone.node,
        pack=pack,
        library=library,
        provenance_for=_provenance,
    )

    assert len(pairs) == len(report.rows)
    assert all(isinstance(pair, PreferencePair) for pair in pairs)
