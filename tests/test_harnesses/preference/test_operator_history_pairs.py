"""Tests for SLM-418 (DSH5-10) sixth slice: context-view / turn-depth ablation.

Covers ``slm_training.dsl.operators.replay_preference_context_views`` (the
structural context-view/turn-depth dimensions) and
``slm_training.harnesses.preference.replay_preference_context_view_variants``
(the bounded, fixture-scale matched context-view comparison harness). See
``docs/design/dsh5-10-replay-preference-rows.md`` for the full disposition.
"""

from __future__ import annotations

import pytest

from slm_training.dsl.operators import (
    ReplayPreferenceRelation,
    undo_conversation,
)
from slm_training.dsl.operators.replay_preference_context_views import (
    TURN_DEPTHS,
    ContextView,
    action_kind_of,
    build_ablation_report,
    build_context_view_window,
    receipts_from_trace,
    state_lookup_from_trace,
)
from slm_training.harnesses.preference.local_decisions import split_for_group
from slm_training.harnesses.preference.replay_preference_context_view_variants import (
    _provenance,
    _rollback_chain_trace,
    synthesize_bounded_session_corpus,
    train_replay_preference_context_view_variants,
)


# --------------------------------------------------------------------------- #
# action_kind_of: structural prefix classification.
# --------------------------------------------------------------------------- #
def test_action_kind_of_classifies_every_control_prefix_and_defaults_to_operator() -> None:
    assert action_kind_of("undo") == "undo"
    assert action_kind_of("redo:abc123") == "redo"
    assert action_kind_of("checkout:abc123") == "checkout"
    assert action_kind_of("merge:abc:def") == "merge"
    assert action_kind_of("openui.some_operator(value=abc)") == "operator"


# --------------------------------------------------------------------------- #
# build_context_view_window: each view's precise, distinct construction.
# --------------------------------------------------------------------------- #
def test_current_state_only_never_exposes_any_history_receipt() -> None:
    trace, _pack, _library = _rollback_chain_trace(4)
    receipts = receipts_from_trace(trace)
    state_lookup = state_lookup_from_trace(trace)
    decision = receipts[-1]

    for depth in TURN_DEPTHS:
        window = build_context_view_window(
            state_lookup=state_lookup,
            receipts=receipts,
            decision_state_id=decision.input_state_id,
            chosen_output_state_id=decision.output_state_id,
            view=ContextView.CURRENT_STATE_ONLY,
            turn_depth=depth,
        )
        assert window.visible_receipt_ids == ()


def test_recent_receipts_window_grows_with_turn_depth_and_caps_at_available_history() -> None:
    # 8 novel edits then a full rollback: the fourth undo in the rollback has
    # exactly 8 + 3 = 11 receipts strictly before it (8 edits, 3 prior undos).
    trace, _pack, _library = _rollback_chain_trace(8)
    receipts = receipts_from_trace(trace)
    state_lookup = state_lookup_from_trace(trace)
    decision = receipts[11]  # the fourth undo (turn_index 11)
    assert decision.action_kind == "undo"

    counts = {}
    for depth in TURN_DEPTHS:
        window = build_context_view_window(
            state_lookup=state_lookup,
            receipts=receipts,
            decision_state_id=decision.input_state_id,
            chosen_output_state_id=decision.output_state_id,
            view=ContextView.STATE_PLUS_RECENT_RECEIPTS,
            turn_depth=depth,
        )
        counts[depth] = len(window.visible_receipt_ids)

    assert counts[1] == 1
    assert counts[2] == 2
    assert counts[4] == 4
    assert counts[8] == 8
    # 11 available, but depth=16 requests more than exist: caps, never fabricates.
    assert counts[16] == 11


def test_recent_receipts_most_recent_kind_reflects_the_immediately_prior_undo() -> None:
    """Regression test for the revisited-decision-state disambiguation bug.

    A rollback chain's later partial-rollback rows share the *content* of an
    earlier forward-edit's input state (both visit the same state_id at
    different points in the turn sequence). Positioning must resolve to
    *this row's own* turn via the exact (input, output) edge, not merely the
    first receipt whose input_state_id matches -- otherwise the window
    silently reports the wrong (earlier, edit-only) history.
    """
    trace, pack, library = _rollback_chain_trace(8)
    from slm_training.dsl.operators import extract_replay_preference_rows

    report = extract_replay_preference_rows(
        trace, pack=pack, library=library, provenance_for=_provenance
    )
    receipts = receipts_from_trace(trace)
    state_lookup = state_lookup_from_trace(trace)

    rollback_rows = [
        row
        for row in report.rows
        if row.semantic_relation is ReplayPreferenceRelation.PARTIAL_ROLLBACK
    ]
    assert len(rollback_rows) == 7  # chain_length - 1

    for row in rollback_rows:
        window = build_context_view_window(
            state_lookup=state_lookup,
            receipts=receipts,
            decision_state_id=row.input_state_id,
            chosen_output_state_id=row.chosen_output_state_id,
            view=ContextView.STATE_PLUS_RECENT_RECEIPTS,
            turn_depth=1,
        )
        assert window.visible_action_kinds() == ("undo",)


def test_retrieved_events_is_ancestry_not_recency() -> None:
    """Ancestry (parent_state_id lineage) differs from chronological recency.

    After a full rollback, the *most recent* receipt before a rollback-chain
    decision is an "undo" (see the test above), but the *ancestry* (the
    single-branch lineage that originally built the state graph) is the
    forward edit chain -- "operator" kind -- since a state's parent_state_id
    is fixed at creation and never changed by later undo/redo navigation.
    """
    trace, pack, library = _rollback_chain_trace(8)
    from slm_training.dsl.operators import extract_replay_preference_rows

    report = extract_replay_preference_rows(
        trace, pack=pack, library=library, provenance_for=_provenance
    )
    receipts = receipts_from_trace(trace)
    state_lookup = state_lookup_from_trace(trace)
    row = next(
        r
        for r in report.rows
        if r.semantic_relation is ReplayPreferenceRelation.PARTIAL_ROLLBACK
    )

    recent = build_context_view_window(
        state_lookup=state_lookup,
        receipts=receipts,
        decision_state_id=row.input_state_id,
        chosen_output_state_id=row.chosen_output_state_id,
        view=ContextView.STATE_PLUS_RECENT_RECEIPTS,
        turn_depth=1,
    )
    retrieved = build_context_view_window(
        state_lookup=state_lookup,
        receipts=receipts,
        decision_state_id=row.input_state_id,
        chosen_output_state_id=row.chosen_output_state_id,
        view=ContextView.STATE_PLUS_RETRIEVED_EVENTS,
        turn_depth=1,
    )
    assert recent.visible_action_kinds() == ("undo",)
    assert retrieved.visible_action_kinds() == ("operator",)
    assert recent.visible_receipt_ids != retrieved.visible_receipt_ids


def test_full_trace_view_saturates_regardless_of_turn_depth() -> None:
    trace, _pack, _library = _rollback_chain_trace(4)
    receipts = receipts_from_trace(trace)
    state_lookup = state_lookup_from_trace(trace)
    decision = receipts[-1]

    counts = {
        depth: len(
            build_context_view_window(
                state_lookup=state_lookup,
                receipts=receipts,
                decision_state_id=decision.input_state_id,
                chosen_output_state_id=decision.output_state_id,
                view=ContextView.FULL_TRACE_PLUS_STATE,
                turn_depth=depth,
            ).visible_receipt_ids
        )
        for depth in TURN_DEPTHS
    }
    assert len(set(counts.values())) == 1  # identical at every depth


def test_last_three_text_history_is_fixed_width_and_state_blind() -> None:
    trace, _pack, _library = _rollback_chain_trace(8)
    receipts = receipts_from_trace(trace)
    state_lookup = state_lookup_from_trace(trace)
    decision = receipts[11]

    for depth in TURN_DEPTHS:
        window = build_context_view_window(
            state_lookup=state_lookup,
            receipts=receipts,
            decision_state_id=decision.input_state_id,
            chosen_output_state_id=decision.output_state_id,
            view=ContextView.LAST_THREE_TEXT_HISTORY,
            turn_depth=depth,
        )
        assert len(window.visible_receipt_ids) == 3
        # State-blind by construction: no state id substring ever appears.
        for token in window.visible_receipt_ids:
            assert token.startswith("kind:")
            assert decision.input_state_id not in token
            assert decision.output_state_id not in token


def test_unknown_turn_depth_is_rejected() -> None:
    trace, _pack, _library = _rollback_chain_trace(2)
    receipts = receipts_from_trace(trace)
    state_lookup = state_lookup_from_trace(trace)
    decision = receipts[-1]
    with pytest.raises(ValueError):
        build_context_view_window(
            state_lookup=state_lookup,
            receipts=receipts,
            decision_state_id=decision.input_state_id,
            chosen_output_state_id=decision.output_state_id,
            view=ContextView.CURRENT_STATE_ONLY,
            turn_depth=3,
        )


# --------------------------------------------------------------------------- #
# OperatorEventMemoryAblationReportV1: structural grid, not a trained result.
# --------------------------------------------------------------------------- #
def test_ablation_report_covers_every_row_view_and_depth_combination() -> None:
    trace, pack, library = _rollback_chain_trace(2)
    from slm_training.dsl.operators import extract_replay_preference_rows

    report = extract_replay_preference_rows(
        trace, pack=pack, library=library, provenance_for=_provenance
    )
    ablation = build_ablation_report(
        report,
        state_lookup=state_lookup_from_trace(trace),
        receipts=receipts_from_trace(trace),
    )
    assert len(ablation.cells) == len(report.rows) * len(list(ContextView)) * len(TURN_DEPTHS)
    assert ablation.version_stamp["components"]["dsl.operators.replay_preference_context_views"]


# --------------------------------------------------------------------------- #
# synthesize_bounded_session_corpus: coverage and split-integrity.
# --------------------------------------------------------------------------- #
def test_synthetic_corpus_covers_all_seven_named_relations() -> None:
    sessions = synthesize_bounded_session_corpus()
    seen = {
        row.semantic_relation
        for session in sessions
        for row in session.report.rows
    }
    assert seen == set(ReplayPreferenceRelation)


def test_synthetic_corpus_session_split_matches_split_for_group() -> None:
    sessions = synthesize_bounded_session_corpus()
    assert sessions  # non-empty
    for session in sessions:
        assert session.split == split_for_group(session.group_id)
    # Adversarial control: "conversation variants stay in one split" -- every
    # row in a session inherits exactly that one session's split (there is no
    # per-row split field to drift, but assert group ids are pairwise
    # distinct, so no two sessions silently share -- and therefore split --
    # one group).
    group_ids = [session.group_id for session in sessions]
    assert len(group_ids) == len(set(group_ids))


def test_synthetic_corpus_has_both_train_and_held_out_sessions() -> None:
    sessions = synthesize_bounded_session_corpus()
    splits = {session.split for session in sessions}
    assert splits == {"train", "held_out"}


# --------------------------------------------------------------------------- #
# train_replay_preference_context_view_variants: the bounded comparison.
# --------------------------------------------------------------------------- #
def test_train_variants_covers_every_view_depth_cell_with_real_numbers() -> None:
    sessions = synthesize_bounded_session_corpus()
    report = train_replay_preference_context_view_variants(sessions)

    assert len(report.cells) == len(list(ContextView)) * len(TURN_DEPTHS)
    for cell in report.cells:
        assert 0.0 <= cell.pairwise_preference_accuracy <= 1.0
        assert cell.held_out_pair_count > 0
        assert cell.train_pair_count > 0
        assert len(cell.weights) == 2

    # "no_non_baseline_held_out_data" is deliberately excluded: the full
    # synthetic corpus always has non-baseline held-out pairs, so accepting
    # it here would make a real corpus/split regression indistinguishable
    # from a genuine fixture-scale result.
    assert report.held_out_benefit["verdict"] in {
        "benefit_observed_fixture_scale",
        "no_benefit_fixture_scale",
    }
    assert 0.0 < report.undo_family_rate < 1.0
    assert report.row_count == sum(len(session.report.rows) for session in sessions)
    assert "structurally inapplicable" in report.unintended_mutation_note
    assert "CAP0/CAP1/CAP2" in report.cap_retention_note


def test_train_variants_is_deterministic_across_repeated_runs() -> None:
    sessions = synthesize_bounded_session_corpus()
    first = train_replay_preference_context_view_variants(sessions)
    second = train_replay_preference_context_view_variants(sessions)
    assert first.held_out_benefit == second.held_out_benefit
    for a, b in zip(first.cells, second.cells, strict=True):
        assert a.pairwise_preference_accuracy == b.pairwise_preference_accuracy
        assert a.weights == b.weights


def test_train_variants_refuses_a_corpus_missing_a_split() -> None:
    sessions = synthesize_bounded_session_corpus()
    train_only = tuple(session for session in sessions if session.split == "train")
    assert train_only  # sanity: the fixture corpus does have train sessions
    with pytest.raises(ValueError):
        train_replay_preference_context_view_variants(train_only)


# --------------------------------------------------------------------------- #
# Spot-check: the corpus's own rows really do replay independently.
#
# This does not re-derive OperatorEventMemoryReportV1's own acceptance
# criterion (unchanged by this slice, already proven by
# tests/test_dsl/test_replay_preference.py) -- it is a direct, real check
# that this harness's *synthetic* traces produce rows with that same
# property, using the same pattern those tests use.
# --------------------------------------------------------------------------- #
def test_a_rollback_chain_row_replays_independently_to_its_recorded_output_state() -> None:
    """Same acceptance criterion as replay_preference.py's own tests.

    Independently re-derive the chosen action from a trace object whose
    ``current`` is exactly the row's input state (built the same way the
    harness's own corpus builder does, one bump earlier) and confirm it
    lands on exactly ``chosen_output_state_id`` -- never merely an assumed
    or already-existing state.
    """
    from slm_training.dsl.operators import (
        branch_fingerprint,
        create_conversation_trace,
        extract_replay_preference_rows,
    )
    from slm_training.harnesses.preference.replay_preference_context_view_variants import (
        _bump,
        _counter_pack_and_root,
        _sha,
        _table,
    )

    pack, library, root_state = _counter_pack_and_root()
    branch = branch_fingerprint(root_state.state_digest, _sha("replay-spot-check"))
    root_table = _table(root_state, branch, seed=1)
    trace = create_conversation_trace(
        pack=pack, root_state=root_state, root_reference_table=root_table, provenance=_provenance(root_state)
    )
    edited = _bump(pack, library, trace, seed=2)
    undone = undo_conversation(edited, provenance=_provenance(edited.current.state))

    report = extract_replay_preference_rows(
        undone, pack=pack, library=library, provenance_for=_provenance
    )
    row = next(
        r
        for r in report.rows
        if r.semantic_relation is ReplayPreferenceRelation.EDIT_THEN_UNDO
    )
    assert row.chosen_action == "undo"
    assert row.input_state_id == edited.current_state_id

    replayed = undo_conversation(edited, provenance=_provenance(edited.current.state))
    assert replayed.current_state_id == row.chosen_output_state_id
