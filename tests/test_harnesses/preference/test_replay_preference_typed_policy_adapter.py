"""Tests for SLM-418 (DSH5-10) eighth slice: TypedOperatorPolicyScorer adapter.

See ``docs/design/dsh5-10-replay-preference-rows.md``'s "Eighth slice" and
``slm_training.harnesses.preference.replay_preference_typed_policy_adapter``'s
module docstring for the full disposition.
"""

from __future__ import annotations

import pytest
import torch

from slm_training.dsl.operators import ReplayPreferenceRelation
from slm_training.harnesses.experiments.typed_operator_policy import (
    TypedOperatorPolicyScorer,
    train_typed_operator_policy,
)
from slm_training.harnesses.preference.replay_preference_context_view_variants import (
    _provenance,
    synthesize_bounded_session_corpus,
)
from slm_training.harnesses.preference.replay_preference_typed_policy_adapter import (
    ReplayRowRepresentability,
    adapt_replay_row_to_typed_policy_input,
    argument_logit_margin,
    build_representability_report,
    classify_replay_row_representability,
    evaluate_typed_policy_argument_probe,
    synthesize_pronoun_focus_sessions,
)
from slm_training.models.operator_policy_view import OperatorPolicyViewError


# --------------------------------------------------------------------------- #
# classify_replay_row_representability: every relation, real fixture rows.
# --------------------------------------------------------------------------- #
def _rows_by_relation() -> dict[str, list]:
    sessions = synthesize_bounded_session_corpus()
    by_relation: dict[str, list] = {}
    for session in sessions:
        for row in session.report.rows:
            by_relation.setdefault(row.semantic_relation.value, []).append(row)
    return by_relation


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
def test_control_action_relations_are_not_representable(relation) -> None:
    rows = _rows_by_relation()[relation.value]
    assert rows
    for row in rows:
        assert (
            classify_replay_row_representability(row)
            is ReplayRowRepresentability.NOT_REPRESENTABLE_CONTROL_ACTION
        )


def test_pronoun_focus_followup_rows_are_representable() -> None:
    rows = _rows_by_relation()[ReplayPreferenceRelation.PRONOUN_FOCUS_FOLLOWUP.value]
    assert rows
    for row in rows:
        assert (
            classify_replay_row_representability(row)
            is ReplayRowRepresentability.ARGUMENT_CHOICE
        )


def test_build_representability_report_matches_the_real_full_corpus() -> None:
    sessions = synthesize_bounded_session_corpus()
    rows = tuple(row for session in sessions for row in session.report.rows)
    report = build_representability_report(rows)

    assert report.total_rows == 56
    assert report.counts_by_representability == {
        "representable_argument_choice": 9,
        "not_representable_control_action": 47,
    }
    # Representability is a relation-level structural fact, not a per-row
    # coincidence: PRONOUN_FOCUS_FOLLOWUP is entirely representable, every
    # other relation entirely is not.
    assert set(report.counts_by_relation["pronoun_focus_followup"]) == {
        "representable_argument_choice"
    }
    for relation, counts in report.counts_by_relation.items():
        if relation != "pronoun_focus_followup":
            assert set(counts) == {"not_representable_control_action"}


# --------------------------------------------------------------------------- #
# adapt_replay_row_to_typed_policy_input: real construction + integrity check.
# --------------------------------------------------------------------------- #
def test_adapt_replay_row_builds_a_policy_input_for_a_representable_row() -> None:
    sessions = synthesize_pronoun_focus_sessions()
    session = next(s for s in sessions if s.rows)
    row = session.rows[0]

    result = adapt_replay_row_to_typed_policy_input(
        row,
        session.trace,
        library=session.library,
        pack=session.pack,
        provenance_for=_provenance,
    )

    assert result.representability is ReplayRowRepresentability.ARGUMENT_CHOICE
    assert result.policy_input is not None
    assert result.accepted_action_row is not None
    assert len(result.policy_input.action_rows) == 1
    assert result.accepted_argument_rows != result.rejected_argument_rows
    # Only argument choice differs; both share the same action row and slot.
    (chosen_slot, chosen_row) = result.accepted_argument_rows[0]
    (rejected_slot, rejected_row) = result.rejected_argument_rows[0]
    assert chosen_slot == rejected_slot
    assert chosen_row != rejected_row


def test_adapt_replay_row_skips_non_representable_rows_without_fabricating() -> None:
    sessions = synthesize_bounded_session_corpus()
    session = next(
        s
        for s in sessions
        if s.report.rows and s.report.rows[0].semantic_relation
        is ReplayPreferenceRelation.EDIT_THEN_UNDO
    )
    row = session.report.rows[0]
    # This corpus's ReplayPreferenceSessionV1 does not retain trace/pack/
    # library (see PronounFocusSessionV1's docstring); rebuild them the same
    # way synthesize_bounded_session_corpus does for this one session type.
    from slm_training.harnesses.preference.replay_preference_context_view_variants import (
        _rollback_chain_trace,
    )

    trace, pack, library = _rollback_chain_trace(1)
    result = adapt_replay_row_to_typed_policy_input(
        row, trace, library=library, pack=pack, provenance_for=_provenance
    )
    assert result.representability is ReplayRowRepresentability.NOT_REPRESENTABLE_CONTROL_ACTION
    assert result.policy_input is None
    assert result.accepted_action_row is None
    assert result.accepted_argument_rows == ()
    assert result.rejected_argument_rows == ()
    with pytest.raises(OperatorPolicyViewError):
        result.to_typed_example(row_id="x")


def test_adapt_replay_row_fails_closed_on_a_stale_legal_set_fingerprint() -> None:
    sessions = synthesize_pronoun_focus_sessions()
    session = next(s for s in sessions if s.rows)
    row = session.rows[0]
    from dataclasses import replace

    stale_row = replace(row, legal_set_fingerprint="0" * 64)
    with pytest.raises(OperatorPolicyViewError):
        adapt_replay_row_to_typed_policy_input(
            stale_row,
            session.trace,
            library=session.library,
            pack=session.pack,
            provenance_for=_provenance,
        )


# --------------------------------------------------------------------------- #
# argument_logit_margin: the real structural-invariance finding.
# --------------------------------------------------------------------------- #
def test_argument_logit_margin_is_exactly_zero_regardless_of_weights() -> None:
    """The real, reproducible eighth-slice finding.

    Both PRONOUN_FOCUS_FOLLOWUP candidate reference rows are structurally
    identical (same ref_kind, value_type, no parent, same compiler_facts) --
    they differ only in an opaque identity ``OperatorFeatureEncoder``
    deliberately never encodes. Their embeddings are therefore identical for
    *any* weight setting, so the argument head's chosen-vs-rejected margin
    is exactly zero for every representable row, before and after training,
    across independent random seeds -- not a training-signal shortfall, a
    representational-collapse ceiling this DSH3 head cannot cross for this
    fixture without a context/history input it structurally does not carry.
    """
    sessions = synthesize_pronoun_focus_sessions()
    results = []
    for session in sessions:
        for row in session.rows:
            results.append(
                adapt_replay_row_to_typed_policy_input(
                    row,
                    session.trace,
                    library=session.library,
                    pack=session.pack,
                    provenance_for=_provenance,
                )
            )
    assert results

    for seed in range(3):
        torch.manual_seed(seed)
        scorer = TypedOperatorPolicyScorer.from_examples(
            tuple(
                result.to_typed_example(row_id=str(index))
                for index, result in enumerate(results)
            ),
            dim=16,
        )
        for result in results:
            ref_embeddings = scorer.encoder.encode_reference_rows(result.policy_input)
            chosen_row = result.accepted_argument_rows[0][1]
            rejected_row = result.rejected_argument_rows[0][1]
            assert torch.allclose(ref_embeddings[chosen_row], ref_embeddings[rejected_row])
            assert argument_logit_margin(scorer, result) == 0.0


def test_training_makes_zero_progress_on_the_representable_subset() -> None:
    """Real training run: loss is exactly unchanged across every step.

    Direct evidence the gradient is exactly zero for this data, not merely
    small -- consistent with the identical-embedding finding above.
    """
    sessions = synthesize_pronoun_focus_sessions()
    train_results = [
        adapt_replay_row_to_typed_policy_input(
            row, session.trace, library=session.library, pack=session.pack,
            provenance_for=_provenance,
        )
        for session in sessions
        if session.split == "train"
        for row in session.rows
    ]
    torch.manual_seed(0)
    examples = tuple(
        result.to_typed_example(row_id=str(index))
        for index, result in enumerate(train_results)
    )
    scorer = TypedOperatorPolicyScorer.from_examples(examples, dim=16)
    history = train_typed_operator_policy(scorer, examples, steps=50, learning_rate=0.05)
    assert history[0] == pytest.approx(history[-1])


# --------------------------------------------------------------------------- #
# evaluate_typed_policy_argument_probe: the real end-to-end report.
# --------------------------------------------------------------------------- #
def test_evaluate_typed_policy_argument_probe_reports_the_real_tied_result() -> None:
    report = evaluate_typed_policy_argument_probe(seed=0)

    assert report.representability.total_rows == 9
    assert report.representability.counts_by_representability == {
        "representable_argument_choice": 9
    }
    assert report.train_example_count == 8
    assert report.held_out_pair_count == 1
    # The structural-invariance finding above predicts an exact tie.
    assert report.pairwise_margin_accuracy == 0.5
    assert report.mean_margin == 0.0
    assert "diagnostic probe only" in report.gating_note
    assert report.version_stamp["components"] == {
        "harness.preference.replay_preference_typed_policy_adapter": "v1"
    }
