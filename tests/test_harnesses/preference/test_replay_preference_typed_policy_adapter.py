"""Tests for SLM-418 (DSH5-10) eighth/ninth slices: TypedOperatorPolicyScorer adapter.

See ``docs/design/dsh5-10-replay-preference-rows.md``'s "Eighth slice" and
"Ninth slice" and
``slm_training.harnesses.preference.replay_preference_typed_policy_adapter``'s
module docstring for the full disposition.
"""

from __future__ import annotations

import pytest
import torch

from slm_training.dsl.operators import (
    ReplayPreferenceRelation,
    refs_touched_by_preceding_turn,
)
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
from slm_training.models.operator_feature_encoder import (
    FixtureOperatorDecisionV1,
    permute_fixture_decision,
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
    # Ninth slice: the bounded history bit is attached and is bounded.
    assert result.recently_touched_rows
    assert len(result.recently_touched_rows) < len(
        result.policy_input.reference_rows
    )
    assert all(
        result.policy_input.reference_rows[reference_row].recently_touched
        for reference_row in result.recently_touched_rows
    )
    assert sum(
        row.recently_touched for row in result.policy_input.reference_rows
    ) == len(result.recently_touched_rows)


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
# recently_touched: what the history bit actually tags (ninth slice).
# --------------------------------------------------------------------------- #
def _all_representable_results() -> list:
    sessions = synthesize_pronoun_focus_sessions()
    return [
        adapt_replay_row_to_typed_policy_input(
            row,
            session.trace,
            library=session.library,
            pack=session.pack,
            provenance_for=_provenance,
        )
        for session in sessions
        for row in session.rows
    ]


def test_recently_touched_tags_the_focus_ref_and_only_the_focus_ref() -> None:
    """The tag is the preceding turn's own bound refs -- measured, not assumed.

    On this fixture that means the chosen candidate is tagged and the
    rejected candidate is not, for every representable row. That is *not* a
    coincidence and *not* leakage from the outcome: it is
    ``extract_replay_preference_rows``'s own pronoun-focus predicate, which
    selects a rejected sibling precisely because its refs do not intersect
    the focus set. The consequence -- that a positive margin on this fixture
    is a representability/learnability result and never a generalization
    result -- is what the adapter's ``recently_touched_note`` says verbatim.
    """
    results = _all_representable_results()
    assert results

    for result in results:
        tagged = set(result.recently_touched_rows)
        chosen_row = result.accepted_argument_rows[0][1]
        rejected_row = result.rejected_argument_rows[0][1]
        assert chosen_row in tagged
        assert rejected_row not in tagged
        # Bounded: the preceding application's arity, never the whole table.
        assert 0 < len(tagged) < len(result.policy_input.reference_rows)


def test_rows_with_no_preceding_edit_are_simply_untagged() -> None:
    """``refs_touched_by_preceding_turn`` is total, not fail-closed-on-absence.

    A state no recorded turn produced (the root) has no preceding edit; that
    is an ordinary answer, so the accessor returns an empty set rather than
    raising, and nothing is tagged.
    """
    sessions = synthesize_pronoun_focus_sessions()
    session = next(s for s in sessions if s.rows)
    root_state_id = session.trace.state_nodes[0].state_id
    assert refs_touched_by_preceding_turn(session.trace, root_state_id) == frozenset()
    assert refs_touched_by_preceding_turn(session.trace, "no-such-state") == frozenset()


def test_recently_touched_survives_a_row_order_permutation_of_a_real_view() -> None:
    """Permutation-equivariance regression on real adapter output.

    Replays the SLM-397 opaque-ID/row-order permutation control at the
    encoder layer (``permute_fixture_decision``, the repo's own helper) over
    a *real* replay-preference view, with real embeddings: the chosen row's
    embedding must be unchanged, and must stay distinct from the rejected
    row's -- whose only difference from it is this bit.
    """
    results = _all_representable_results()
    torch.manual_seed(0)
    scorer = TypedOperatorPolicyScorer.from_examples(
        tuple(
            result.to_typed_example(row_id=str(index))
            for index, result in enumerate(results)
        ),
        dim=16,
    )

    for result in results:
        chosen_row = result.accepted_argument_rows[0][1]
        rejected_row = result.rejected_argument_rows[0][1]
        decision = FixtureOperatorDecisionV1(view=result.policy_input, gold_row=chosen_row)
        permuted = permute_fixture_decision(decision, seed=31)
        # Guard against a vacuous pass: the permutation must really have
        # moved the tagged row to a different index.
        assert permuted.gold_row != chosen_row
        assert permuted.view.to_dict() == result.policy_input.to_dict()

        original = scorer.encoder.encode_reference_rows(result.policy_input)
        reshuffled = scorer.encoder.encode_reference_rows(permuted.view)
        assert torch.allclose(
            original[chosen_row], reshuffled[permuted.gold_row], atol=1e-6
        )
        assert not torch.allclose(original[chosen_row], original[rejected_row])


# --------------------------------------------------------------------------- #
# argument_logit_margin: the eighth slice's flatness, and what broke it.
# --------------------------------------------------------------------------- #
def test_candidate_embeddings_no_longer_collapse() -> None:
    """The eighth slice's finding, re-run: it is now false, and only because
    of the new field.

    The eighth slice measured ``torch.allclose(ref[chosen], ref[rejected])``
    as ``True`` at every seed *before any training*, which forced the
    argument margin to exactly 0.0 for any weights. With
    ``recently_touched`` carried through the sanitized boundary the two rows
    are no longer identical inputs, so the margin is a real (small,
    seed-dependent, either-signed) number at initialization rather than an
    exact structural zero.
    """
    results = _all_representable_results()
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
            assert not torch.allclose(
                ref_embeddings[chosen_row], ref_embeddings[rejected_row]
            )
            assert argument_logit_margin(scorer, result) != 0.0


def test_training_now_makes_real_progress_on_the_representable_subset() -> None:
    """Real training run: the gradient is no longer exactly zero.

    The eighth slice's counterpart asserted ``history[0] == history[-1]`` to
    full float precision. The same run over the same 8 train rows now
    descends.
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
    assert history[-1] < history[0]
    assert history[0] == pytest.approx(0.3303639590740204, abs=1e-9)
    assert history[-1] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# evaluate_typed_policy_argument_probe: the real end-to-end report.
# --------------------------------------------------------------------------- #
def test_evaluate_typed_policy_argument_probe_reports_the_real_result() -> None:
    report = evaluate_typed_policy_argument_probe(seed=0)

    assert report.representability.total_rows == 9
    assert report.representability.counts_by_representability == {
        "representable_argument_choice": 9
    }
    assert report.train_example_count == 8
    assert report.held_out_pair_count == 1
    # Ninth slice: a real, non-zero, correctly-signed held-out margin at n=1.
    assert report.pairwise_margin_accuracy == 1.0
    assert report.mean_margin > 0.0
    assert report.mean_margin == pytest.approx(63.911468505859375, rel=1e-6)
    # ...reported alongside the exact measurement of its own tautology risk.
    assert report.recently_touched_diagnostics == {
        "representable_rows": 9,
        "tagged_reference_rows": 9,
        "chosen_rows_tagged": 9,
        "rejected_rows_tagged": 0,
    }
    assert "definitionally aligned with the label" in report.recently_touched_note
    assert "never evidence that the feature generalizes" in report.recently_touched_note
    assert "diagnostic probe only" in report.gating_note
    assert report.version_stamp["components"] == {
        "harness.preference.replay_preference_typed_policy_adapter": "v2"
    }


@pytest.mark.parametrize("seed", [0, 1, 2, 7])
def test_probe_margin_sign_is_stable_across_independent_seeds(seed: int) -> None:
    """n=1 held out: the *sign* is the claim, never the magnitude."""
    report = evaluate_typed_policy_argument_probe(seed=seed)
    assert report.pairwise_margin_accuracy == 1.0
    assert report.mean_margin > 0.0
