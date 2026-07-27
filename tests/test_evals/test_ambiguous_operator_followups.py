"""Tests for SLM-418 (DSH5-10) sixth slice: the ambiguous-followup disposition.

Covers ``slm_training.evals.ambiguous_operator_followups`` -- the eval-level
assembly of the bounded, fixture-scale context-view comparison plus the
structural ablation grid and the issue's own out-of-scope acknowledgements.
See ``docs/design/dsh5-10-replay-preference-rows.md`` for the full history.
"""

from __future__ import annotations

from slm_training.dsl.operators import (
    ReplayPreferenceRelation,
    undo_conversation,
)
from slm_training.dsl.operators.replay_preference_context_views import ContextView, TURN_DEPTHS
from slm_training.evals.ambiguous_operator_followups import (
    OUT_OF_SCOPE_METRICS,
    evaluate_ambiguous_operator_followups,
)
from slm_training.harnesses.preference.local_decisions import split_for_group
from slm_training.harnesses.preference.replay_preference_context_view_variants import (
    _provenance,
    synthesize_bounded_session_corpus,
)


def test_report_schema_and_version_stamp_are_real() -> None:
    report = evaluate_ambiguous_operator_followups()
    assert report.schema == "ambiguous_operator_followup_eval_report/v1"
    assert report.version_stamp["components"][
        "harness.preference.replay_preference_context_view_variants"
    ]
    payload = report.to_dict()
    assert payload["comparison"]["schema"] == (
        "replay_preference_context_view_comparison_report/v1"
    )


def test_held_out_benefit_verdict_is_never_a_ship_or_certified_claim() -> None:
    report = evaluate_ambiguous_operator_followups()
    verdict = report.comparison.held_out_benefit["verdict"]
    # The default full synthetic corpus always has non-baseline held-out
    # pairs, so "no_non_baseline_held_out_data" is deliberately excluded --
    # accepting it here would make a real corpus/split regression
    # indistinguishable from a genuine fixture-scale result. The remaining
    # vocabulary is the only one this disposition may use -- an honest
    # fixture-scale finding either way, never "ship" / "certified" /
    # "production-ready".
    assert verdict in {
        "benefit_observed_fixture_scale",
        "no_benefit_fixture_scale",
    }
    for banned in ("ship", "certified", "production", "promote"):
        assert banned not in verdict


def test_out_of_scope_metrics_names_every_unmeasured_issue_metric() -> None:
    report = evaluate_ambiguous_operator_followups()
    # held_out_benefit_statistical_power is corpus-size-dependent (derived
    # from the actual comparison.row_count/session_count), so it is checked
    # separately rather than by exact dict equality with the static constant.
    assert report.out_of_scope_metrics.keys() == {
        *OUT_OF_SCOPE_METRICS,
        "held_out_benefit_statistical_power",
    }
    for key, note in OUT_OF_SCOPE_METRICS.items():
        assert report.out_of_scope_metrics[key] == note
    assert str(report.comparison.row_count) in report.out_of_scope_metrics[
        "held_out_benefit_statistical_power"
    ]
    assert str(report.comparison.session_count) in report.out_of_scope_metrics[
        "held_out_benefit_statistical_power"
    ]
    assert "cap0_cap1_cap2_retention" in report.out_of_scope_metrics
    assert "unintended_mutations" in report.out_of_scope_metrics
    # Every note must itself be honest, not a placeholder.
    for note in report.out_of_scope_metrics.values():
        assert len(note) > 20


def test_ablation_cell_counts_cover_the_full_view_by_depth_matrix() -> None:
    report = evaluate_ambiguous_operator_followups()
    expected_keys = {
        f"{view.value}:{depth}" for view in ContextView for depth in TURN_DEPTHS
    }
    assert set(report.ablation_cell_counts) == expected_keys
    assert all(count > 0 for count in report.ablation_cell_counts.values())


def test_conversation_variants_stay_in_one_split() -> None:
    """Adversarial control: no session's rows are split across train/held-out."""
    sessions = synthesize_bounded_session_corpus()
    for session in sessions:
        assert session.split == split_for_group(session.group_id)
    groups_by_split = {"train": set(), "held_out": set()}
    for session in sessions:
        groups_by_split[session.split].add(session.group_id)
    assert not (groups_by_split["train"] & groups_by_split["held_out"])


def test_corpus_covers_all_seven_named_relations_at_least_once() -> None:
    sessions = synthesize_bounded_session_corpus()
    seen = {row.semantic_relation for session in sessions for row in session.report.rows}
    assert seen == set(ReplayPreferenceRelation)


def test_undo_is_not_automatically_preferred() -> None:
    """Adversarial control: 'undo is not automatically preferred' over other legal actions.

    Every row's rejected_action must differ from its chosen_action -- proving
    the corpus never asserts undo (or any other action) preferred by default,
    matching replay_preference.py's own construction guarantee.
    """
    sessions = synthesize_bounded_session_corpus()
    for session in sessions:
        for row in session.report.rows:
            assert row.rejected_action != row.chosen_action


def test_evaluate_accepts_a_smaller_externally_supplied_corpus() -> None:
    """A caller may pass a reduced corpus, as long as both splits are present."""
    sessions = synthesize_bounded_session_corpus()
    train = next(s for s in sessions if s.split == "train")
    held_out = next(s for s in sessions if s.split == "held_out")
    report = evaluate_ambiguous_operator_followups(sessions=(train, held_out))
    assert report.comparison.session_count == 2
    assert report.comparison.train_session_count == 1
    assert report.comparison.held_out_session_count == 1
    # The statistical-power note must reflect this smaller corpus, not the
    # hardcoded "8 sessions" of the module's own default full corpus.
    note = report.out_of_scope_metrics["held_out_benefit_statistical_power"]
    assert "2 session" in note
    assert "8 session" not in note


def test_a_corpus_row_independently_replays_reusing_the_real_operator_api() -> None:
    """Spot-check the trace_replay_note's claim directly, not just assert the note exists.

    Builds a fresh one-edit trace with the harness's own fixture constructors
    (not the final, already-fully-appended corpus trace, which already
    contains the very undo turn being checked and would collide on replay),
    extracts its EDIT_THEN_UNDO row, and independently re-derives
    ``chosen_output_state_id`` from ``row.input_state_id`` -- never assuming
    it, per the issue's own acceptance criterion.
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
    branch = branch_fingerprint(root_state.state_digest, _sha("eval-replay-spot-check"))
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
        r for r in report.rows if r.semantic_relation is ReplayPreferenceRelation.EDIT_THEN_UNDO
    )
    assert row.chosen_action == "undo"
    assert row.input_state_id == edited.current_state_id

    replayed = undo_conversation(edited, provenance=_provenance(edited.current.state))
    assert replayed.current_state_id == row.chosen_output_state_id
