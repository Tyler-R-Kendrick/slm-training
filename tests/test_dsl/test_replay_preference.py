"""Tests for SLM-418 (DSH5-10) replay-grounded preference row extraction."""

from __future__ import annotations

import hashlib

from slm_training.dsl.operators import (
    OperatorEventMemoryReportV1,
    ReplayPreferenceRelation,
    checkout_conversation_state,
    extract_replay_preference_rows,
    fork_conversation,
    redo_conversation,
    undo_conversation,
)
from tests.test_dsl.test_operator_conversation import _append, _fixture, _provenance


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
