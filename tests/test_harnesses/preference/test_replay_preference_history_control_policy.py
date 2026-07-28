"""Tests for SLM-418 (DSH5-10) ninth slice: sibling history-control policy view.

See ``docs/design/dsh5-10-replay-preference-rows.md``'s "Ninth slice" and
``slm_training.harnesses.preference.replay_preference_history_control_policy``'s
module docstring for the full disposition.
"""

from __future__ import annotations

import pytest

from slm_training.dsl.operators import ReplayPreferenceRelation
from slm_training.harnesses.preference.replay_preference_context_view_variants import (
    _merge_session,
    _provenance,
    _rollback_chain_trace,
    synthesize_bounded_session_corpus,
)
from slm_training.harnesses.preference.replay_preference_history_control_policy import (
    ControlActionKind,
    HistoryControlRowRepresentability,
    adapt_replay_row_to_history_control_policy_input,
    build_history_control_policy_input,
    build_history_control_representability_report,
    classify_replay_row_history_control_representability,
    evaluate_history_control_representability,
    synthesize_history_control_sessions,
)
from slm_training.models.operator_policy_view import OperatorPolicyViewError


def _rows_by_relation() -> dict[str, list]:
    sessions = synthesize_bounded_session_corpus()
    by_relation: dict[str, list] = {}
    for session in sessions:
        for row in session.report.rows:
            by_relation.setdefault(row.semantic_relation.value, []).append(row)
    return by_relation


# --------------------------------------------------------------------------- #
# classify_replay_row_history_control_representability: real fixture rows.
# --------------------------------------------------------------------------- #
def test_pronoun_focus_followup_rows_are_not_this_module_s_domain() -> None:
    rows = _rows_by_relation()[ReplayPreferenceRelation.PRONOUN_FOCUS_FOLLOWUP.value]
    assert rows
    for row in rows:
        assert (
            classify_replay_row_history_control_representability(row)
            is HistoryControlRowRepresentability.NOT_REPRESENTABLE_OPERATOR_CHOSEN
        )


@pytest.mark.parametrize(
    "relation",
    [
        ReplayPreferenceRelation.EDIT_THEN_UNDO,
        ReplayPreferenceRelation.UNDO_THEN_REDO,
        ReplayPreferenceRelation.PARTIAL_ROLLBACK,
        ReplayPreferenceRelation.CHECKOUT_ANOTHER_STATE,
        ReplayPreferenceRelation.FORK_THEN_CHOOSE_ONE_BRANCH,
        ReplayPreferenceRelation.MERGE_SUCCESS,
    ],
)
def test_the_other_six_relations_chosen_side_is_a_control_token(relation) -> None:
    """Chosen action is representable (a real control-domain row can be built).

    The eighth slice's ``TypedOperatorPolicyScorer`` adapter classified every
    one of these rows ``NOT_REPRESENTABLE_CONTROL_ACTION`` because it can
    build no row for a control action at all. This module's sibling view can
    always build a row for the *chosen* side; the real, independently
    verified limit is the *rejected* side, asserted below.
    """
    rows = _rows_by_relation()[relation.value]
    assert rows
    for row in rows:
        classification = classify_replay_row_history_control_representability(row)
        assert classification is not HistoryControlRowRepresentability.NOT_REPRESENTABLE_OPERATOR_CHOSEN


def test_real_corpus_has_zero_same_domain_pairs_every_rejected_side_is_an_operator() -> None:
    """The real, empirically verified finding this slice's docstring claims.

    Every one of the 47 non-``PRONOUN_FOCUS_FOLLOWUP`` rows in the real
    synthetic corpus classifies ``CHOSEN_CONTROL_REJECTED_OPERATOR``, never
    ``BOTH_CONTROL`` -- ``_pick_rejected`` always picks the operator token
    because it sorts first lexicographically whenever one is legal, which it
    always is in this fixture.
    """
    by_relation = _rows_by_relation()
    for relation, rows in by_relation.items():
        if relation == ReplayPreferenceRelation.PRONOUN_FOCUS_FOLLOWUP.value:
            continue
        for row in rows:
            assert (
                classify_replay_row_history_control_representability(row)
                is HistoryControlRowRepresentability.CHOSEN_CONTROL_REJECTED_OPERATOR
            )


def test_build_history_control_representability_report_matches_the_real_full_corpus() -> None:
    sessions = synthesize_bounded_session_corpus()
    rows = tuple(row for session in sessions for row in session.report.rows)
    report = build_history_control_representability_report(rows)

    assert report.total_rows == 56
    assert report.counts_by_representability == {
        "not_representable_operator_chosen": 9,
        "chosen_control_rejected_cross_domain_operator": 47,
    }
    assert set(report.counts_by_relation["pronoun_focus_followup"]) == {
        "not_representable_operator_chosen"
    }
    for relation, counts in report.counts_by_relation.items():
        if relation != "pronoun_focus_followup":
            assert set(counts) == {"chosen_control_rejected_cross_domain_operator"}


# --------------------------------------------------------------------------- #
# build_history_control_policy_input / adapt_replay_row_to_history_control_policy_input
# --------------------------------------------------------------------------- #
def test_adapt_replay_row_builds_a_real_control_row_for_the_chosen_action() -> None:
    sessions = synthesize_history_control_sessions()
    session = next(
        s
        for s in sessions
        if s.rows
        and s.rows[0].semantic_relation is ReplayPreferenceRelation.EDIT_THEN_UNDO
    )
    row = session.rows[0]

    result = adapt_replay_row_to_history_control_policy_input(
        row, session.trace, library=session.library, pack=session.pack,
        provenance_for=_provenance,
    )

    assert (
        result.representability
        is HistoryControlRowRepresentability.CHOSEN_CONTROL_REJECTED_OPERATOR
    )
    assert result.policy_input is not None
    assert result.accepted_control_row is not None
    accepted_view = result.policy_input.control_rows[result.accepted_control_row]
    assert accepted_view.control_kind is ControlActionKind.UNDO
    # The rejected action is a cross-domain operator token, so no control row
    # represents it -- this is the real, honestly reported limit.
    assert result.rejected_control_row is None


def test_build_history_control_policy_input_never_carries_the_opaque_state_id() -> None:
    sessions = synthesize_history_control_sessions()
    session = next(s for s in sessions if s.rows)
    row = session.rows[0]
    from slm_training.dsl.operators import legal_set_at

    legal_set = legal_set_at(
        session.trace,
        pack=session.pack,
        library=session.library,
        state_id=row.input_state_id,
        provenance_for=_provenance,
    )
    policy_input = build_history_control_policy_input(legal_set)

    assert len(policy_input.control_rows) == len(legal_set.ordinary_nonoperator_actions)
    for view in policy_input.control_rows:
        payload = view.to_dict()
        assert set(payload) == {"schema", "row", "control_kind"}


def test_adapt_replay_row_reports_foreign_state_for_a_merge_success_row() -> None:
    """A ``MERGE_SUCCESS`` row is grounded on a ``BranchEditV1`` tip, not any trace."""
    merge_row, _left, _right = _merge_session()
    unrelated_trace, unrelated_pack, unrelated_library = _rollback_chain_trace(1)

    result = adapt_replay_row_to_history_control_policy_input(
        merge_row,
        unrelated_trace,
        library=unrelated_library,
        pack=unrelated_pack,
        provenance_for=_provenance,
    )

    assert (
        result.representability
        is HistoryControlRowRepresentability.NOT_REPRESENTABLE_FOREIGN_STATE
    )
    assert result.policy_input is None
    assert result.accepted_control_row is None


def test_adapt_replay_row_skips_pronoun_rows_without_fabricating() -> None:
    from slm_training.harnesses.preference.replay_preference_typed_policy_adapter import (
        synthesize_pronoun_focus_sessions,
    )

    sessions = synthesize_pronoun_focus_sessions()
    session = next(s for s in sessions if s.rows)
    row = session.rows[0]

    result = adapt_replay_row_to_history_control_policy_input(
        row, session.trace, library=session.library, pack=session.pack,
        provenance_for=_provenance,
    )
    assert (
        result.representability
        is HistoryControlRowRepresentability.NOT_REPRESENTABLE_OPERATOR_CHOSEN
    )
    assert result.policy_input is None


def test_adapt_replay_row_fails_closed_on_a_stale_legal_set_fingerprint() -> None:
    from dataclasses import replace

    sessions = synthesize_history_control_sessions()
    session = next(s for s in sessions if s.rows)
    row = session.rows[0]
    stale_row = replace(row, legal_set_fingerprint="0" * 64)

    with pytest.raises(OperatorPolicyViewError):
        adapt_replay_row_to_history_control_policy_input(
            stale_row, session.trace, library=session.library, pack=session.pack,
            provenance_for=_provenance,
        )


# --------------------------------------------------------------------------- #
# evaluate_history_control_representability: the real end-to-end report.
# --------------------------------------------------------------------------- #
def test_evaluate_history_control_representability_reports_the_real_zero_pairing_result() -> None:
    report = evaluate_history_control_representability()

    assert report.same_domain_pair_count == 0
    assert report.cross_domain_pair_count == 46
    assert report.verdict == "no_same_domain_pairs_fixture_scale"
    assert "always the operator action" in report.note
    assert report.representability.total_rows == 46
    assert report.representability.counts_by_representability == {
        "chosen_control_rejected_cross_domain_operator": 46
    }
    assert report.version_stamp["components"] == {
        "harness.preference.replay_preference_history_control_policy": "v1"
    }
