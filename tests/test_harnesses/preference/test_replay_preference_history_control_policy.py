"""Tests for SLM-418 (DSH5-10) ninth/tenth slices: sibling history-control policy view.

See ``docs/design/dsh5-10-replay-preference-rows.md``'s "Ninth slice"/"Tenth
slice" and
``slm_training.harnesses.preference.replay_preference_history_control_policy``'s
module docstring for the full disposition. The tenth slice's own
``_pick_rejected`` same-domain-preferring change flips several of this
module's real numbers (documented inline on each affected test), and adds
``train_history_control_pairwise_scorer`` coverage at the bottom of this
file.
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
    train_history_control_pairwise_scorer,
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


def test_real_corpus_now_has_same_domain_pairs_since_the_tenth_slice() -> None:
    """The tenth slice's own re-run of this exact finding, now flipped.

    The ninth slice's ``_pick_rejected`` always picked a cross-domain
    operator token as ``rejected`` (a serialized operator token sorts
    before every control-token prefix). The tenth slice changed
    ``_pick_rejected`` to prefer a same-domain alternative when the legal
    set actually offers one -- and every decision state in this real
    synthetic corpus does offer one (at least a ``checkout:<state>``
    candidate, per ``_available_history_actions``). Every one of the 47
    non-``PRONOUN_FOCUS_FOLLOWUP`` rows now classifies ``BOTH_CONTROL``,
    never ``CHOSEN_CONTROL_REJECTED_OPERATOR``. See
    ``test_replay_preference.py``'s own direct ``_pick_rejected`` unit
    tests for the underlying mechanism, and "Tenth slice" in
    ``docs/design/dsh5-10-replay-preference-rows.md`` for why this
    doesn't, by itself, mean the resulting pairs are a *hard* preference
    signal -- see this file's own scorer tests below.
    """
    by_relation = _rows_by_relation()
    for relation, rows in by_relation.items():
        if relation == ReplayPreferenceRelation.PRONOUN_FOCUS_FOLLOWUP.value:
            continue
        for row in rows:
            assert (
                classify_replay_row_history_control_representability(row)
                is HistoryControlRowRepresentability.BOTH_CONTROL
            )


def test_build_history_control_representability_report_matches_the_real_full_corpus() -> None:
    sessions = synthesize_bounded_session_corpus()
    rows = tuple(row for session in sessions for row in session.report.rows)
    report = build_history_control_representability_report(rows)

    assert report.total_rows == 56
    assert report.counts_by_representability == {
        "not_representable_operator_chosen": 9,
        "representable_both_control": 47,
    }
    assert set(report.counts_by_relation["pronoun_focus_followup"]) == {
        "not_representable_operator_chosen"
    }
    for relation, counts in report.counts_by_relation.items():
        if relation != "pronoun_focus_followup":
            assert set(counts) == {"representable_both_control"}


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

    # Tenth slice: _pick_rejected now prefers a same-domain alternative, and
    # this fixture always has one (at least a checkout candidate), so this
    # row -- like every non-pronoun row in the real corpus -- is now
    # BOTH_CONTROL, not CHOSEN_CONTROL_REJECTED_OPERATOR.
    assert result.representability is HistoryControlRowRepresentability.BOTH_CONTROL
    assert result.policy_input is not None
    assert result.accepted_control_row is not None
    accepted_view = result.policy_input.control_rows[result.accepted_control_row]
    assert accepted_view.control_kind is ControlActionKind.UNDO
    # The rejected action is now a same-domain control token too, so a
    # control row represents it as well.
    assert result.rejected_control_row is not None
    rejected_view = result.policy_input.control_rows[result.rejected_control_row]
    assert row.rejected_action.startswith("checkout:")
    assert rejected_view.control_kind is ControlActionKind.CHECKOUT


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
def test_evaluate_history_control_representability_reports_the_real_same_domain_result() -> None:
    """Tenth slice: this exact probe, re-run, now reports the flipped finding.

    ``_pick_rejected``'s same-domain preference (this slice) means every
    one of the 46 trace-grounded, non-pronoun, non-merge rows in the real
    corpus now pairs a control chosen action against a control rejected
    action -- see ``docs/design/dsh5-10-replay-preference-rows.md``'s
    "Tenth slice" for why this genuinely moves ``same_domain_pair_count``
    off zero without, by itself, proving a hard preference signal exists
    (see the scorer tests below for that separate question).
    """
    report = evaluate_history_control_representability()

    assert report.same_domain_pair_count == 46
    assert report.cross_domain_pair_count == 0
    assert report.verdict == "same_domain_pairs_found_fixture_scale"
    assert "same-domain" in report.note
    assert report.representability.total_rows == 46
    assert report.representability.counts_by_representability == {
        "representable_both_control": 46
    }
    assert report.version_stamp["components"] == {
        "harness.preference.replay_preference_history_control_policy": "v2"
    }


# --------------------------------------------------------------------------- #
# Tenth slice: train_history_control_pairwise_scorer -- the stretch goal the
# ninth slice's own "Named next lever" pointed at, now reachable.
# --------------------------------------------------------------------------- #
def test_train_history_control_pairwise_scorer_reports_the_real_lexicographic_artifact() -> None:
    """The real, measured, honest result -- not forced positive.

    ``same_domain_pair_count`` moved off zero, so a pairwise scorer over
    ``HistoryControlPolicyInputV1.control_rows`` can now be trained. Its
    held-out accuracy is real and positive (1.0 on n=2 held-out pairs), but
    ``rejected_kind_counts`` shows the rejected side is *always* "checkout"
    across the whole corpus -- ``_pick_rejected``'s same-domain preference
    still breaks ties within a domain by plain lexicographic sort, and
    "checkout:" sorts first among control prefixes whenever legal, which
    this corpus's trace shape makes true almost everywhere. So this is
    correctly reported as a lexicographic-tiebreak artifact, not a hard
    preference signal -- same rigor as the eighth slice's zero-margin
    finding, just for a positive number this time.
    """
    report = train_history_control_pairwise_scorer()

    assert report.train_pair_count == 44
    assert report.held_out_pair_count == 2
    assert report.mean_margin > 0
    assert report.pairwise_margin_accuracy == 1.0
    assert report.rejected_kind_counts == {"checkout": 46}
    assert report.verdict == "discrimination_is_lexicographic_tiebreak_artifact_fixture_scale"
    assert "NOT evidence of a real hard preference signal" in report.note
    assert report.version_stamp["components"] == {
        "harness.preference.replay_preference_history_control_policy": "v2"
    }


def test_train_history_control_pairwise_scorer_weights_penalize_checkout() -> None:
    """Direct evidence for the artifact: the trained weight on CHECKOUT is negative.

    ``ControlActionKind`` order is ``(UNDO, REDO, CHECKOUT, MERGE)`` --
    since ``checkout`` is the rejected side of every single training pair,
    a linear pairwise scorer only needs a negative CHECKOUT weight (and
    non-negative others) to fit the corpus, exactly what a real trained run
    produces.
    """
    report = train_history_control_pairwise_scorer()

    undo_weight, redo_weight, checkout_weight, merge_weight = report.weights
    assert checkout_weight < 0
    assert undo_weight > 0
    assert redo_weight > 0
    assert merge_weight == 0.0  # MERGE never appears as chosen or rejected in this corpus
