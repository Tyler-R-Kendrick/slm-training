"""Tests for SLM-418 (DSH5-10) replay-grounded preference row extraction."""

from __future__ import annotations

from slm_training.dsl.operators import (
    OperatorEventMemoryReportV1,
    ReplayPreferenceRelation,
    extract_replay_preference_rows,
    redo_conversation,
    undo_conversation,
)
from tests.test_dsl.test_operator_conversation import _append, _fixture, _provenance


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
