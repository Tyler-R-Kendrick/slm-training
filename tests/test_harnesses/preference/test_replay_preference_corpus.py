"""Tests for SLM-418 (DSH5-10) seventh slice: writing the bounded
replay-preference corpus to real ``PreferencePair`` corpus files.

Covers ``slm_training.harnesses.preference.replay_preference_corpus``. See
``docs/design/dsh5-10-replay-preference-rows.md`` "Seventh slice" for the
full disposition.
"""

from __future__ import annotations

import pytest

from slm_training.dsl.operators import ReplayPreferenceRelation
from slm_training.harnesses.preference import load_pairs
from slm_training.harnesses.preference.replay_preference_context_view_variants import (
    synthesize_bounded_session_corpus,
)
from slm_training.harnesses.preference.replay_preference_corpus import (
    ReplayPreferenceCorpusBuildReportV1,
    replay_preference_pairs_for_split,
    write_replay_preference_corpus,
)

_ROLLBACK_FAMILY = {
    ReplayPreferenceRelation.EDIT_THEN_UNDO.value,
    ReplayPreferenceRelation.UNDO_THEN_REDO.value,
    ReplayPreferenceRelation.PARTIAL_ROLLBACK.value,
}


# --------------------------------------------------------------------------- #
# replay_preference_pairs_for_split
# --------------------------------------------------------------------------- #
def test_train_split_converts_every_train_row_with_nothing_skipped() -> None:
    sessions = synthesize_bounded_session_corpus()
    train_row_count = sum(
        len(session.report.rows) for session in sessions if session.split == "train"
    )
    pairs, skipped = replay_preference_pairs_for_split(sessions, "train")
    assert not skipped
    assert len(pairs) == train_row_count
    assert train_row_count > 0


def test_held_out_split_converts_exactly_the_two_held_out_sessions() -> None:
    sessions = synthesize_bounded_session_corpus()
    held_out_sessions = [session for session in sessions if session.split == "held_out"]
    assert {session.group_id for session in held_out_sessions} == {
        "rollback_chain_1",
        "checkout_and_fork",
    }
    held_out_row_count = sum(len(session.report.rows) for session in held_out_sessions)

    pairs, skipped = replay_preference_pairs_for_split(sessions, "held_out")
    assert not skipped
    assert len(pairs) == held_out_row_count
    relations = {pair.meta["semantic_relation"] for pair in pairs}
    assert relations == {
        ReplayPreferenceRelation.EDIT_THEN_UNDO.value,
        ReplayPreferenceRelation.UNDO_THEN_REDO.value,
        ReplayPreferenceRelation.CHECKOUT_ANOTHER_STATE.value,
        ReplayPreferenceRelation.FORK_THEN_CHOOSE_ONE_BRANCH.value,
    }


def test_merge_success_row_converts_via_the_direct_state_lookup_path() -> None:
    """MERGE_SUCCESS has no trace (session.trace is None); its row must still
    convert, via this module's own direct state_lookup path, not
    ``preference_pairs_from_trace`` (which would skip it, per the sixth
    slice's own documented convention).
    """
    sessions = synthesize_bounded_session_corpus()
    merge_session = next(s for s in sessions if s.group_id == "merge_success")
    assert merge_session.trace is None
    assert merge_session.split == "train"  # sanity: exercised by the train split below

    pairs, skipped = replay_preference_pairs_for_split(sessions, "train")
    assert not skipped
    merge_pairs = [
        pair for pair in pairs if pair.meta["semantic_relation"] == "merge_success"
    ]
    assert len(merge_pairs) == 1
    merge_pair = merge_pairs[0]
    assert merge_pair.chosen.startswith("merge:")
    assert merge_pair.prompt.strip()
    assert merge_pair.meta["schema"] == "operator_replay_preference_pair/v1"


def test_every_converted_pair_has_a_real_prompt_and_distinct_actions() -> None:
    sessions = synthesize_bounded_session_corpus()
    pairs, _skipped = replay_preference_pairs_for_split(sessions, "train")
    assert pairs
    for pair in pairs:
        assert pair.prompt.strip()
        assert pair.chosen != pair.rejected
        assert pair.meta["schema"] == "operator_replay_preference_pair/v1"
        assert pair.chosen_score == 0.0
        assert pair.rejected_score == 0.0


def test_unknown_split_yields_nothing_rather_than_raising() -> None:
    sessions = synthesize_bounded_session_corpus()
    pairs, skipped = replay_preference_pairs_for_split(sessions, "train")
    held_out_pairs, held_out_skipped = replay_preference_pairs_for_split(
        sessions, "held_out"
    )
    total_rows = sum(len(session.report.rows) for session in sessions)
    assert len(pairs) + len(held_out_pairs) == total_rows
    assert not skipped and not held_out_skipped


# --------------------------------------------------------------------------- #
# write_replay_preference_corpus
# --------------------------------------------------------------------------- #
def test_write_replay_preference_corpus_writes_a_loadable_jsonl(tmp_path) -> None:
    out_path = tmp_path / "train_pairs.jsonl"
    report = write_replay_preference_corpus(out_path, "train")

    assert isinstance(report, ReplayPreferenceCorpusBuildReportV1)
    assert report.split == "train"
    assert report.pair_count > 0
    assert report.skipped_count == 0
    assert out_path.exists()

    loaded = load_pairs(out_path)
    assert len(loaded) == report.pair_count
    assert report.version_stamp["components"]["harness.preference.replay_preference_corpus"]


def test_write_replay_preference_corpus_held_out_split(tmp_path) -> None:
    out_path = tmp_path / "held_out_pairs.jsonl"
    report = write_replay_preference_corpus(out_path, "held_out")
    assert report.pair_count == 4  # rollback_chain_1 (2) + checkout_and_fork (2)
    assert report.session_count == 2
    loaded = load_pairs(out_path)
    assert len(loaded) == 4


def test_write_replay_preference_corpus_refuses_to_write_an_empty_split(tmp_path) -> None:
    sessions = synthesize_bounded_session_corpus()
    held_out_only = tuple(session for session in sessions if session.split == "held_out")
    assert held_out_only  # sanity
    out_path = tmp_path / "empty.jsonl"
    with pytest.raises(ValueError):
        # Asking for the "train" split against a held-out-only session list
        # produces zero pairs; must fail closed, never write an empty file.
        write_replay_preference_corpus(out_path, "train", sessions=held_out_only)
    assert not out_path.exists()
