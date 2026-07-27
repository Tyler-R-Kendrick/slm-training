"""Tests for the VAR3-03 (SLM-433) real TurnDispositionHead corpus builder."""

from __future__ import annotations

import pytest

from slm_training.data.flow.turn_disposition_corpus import (
    TIER_GOLD_CHEAP_PROBABILITY,
    TIER_MARGINS,
    TurnDispositionCorpusRowV1,
    build_turn_disposition_corpus,
)
from slm_training.models.turn_disposition_head import TurnDispositionLabel


def test_out_of_scope_rows_are_provably_derived_from_empty_legal_set() -> None:
    """Adversarial control: the corpus builder must never mislabel a
    non-empty-legal-set turn as out_of_scope."""
    corpus = build_turn_disposition_corpus(n_per_tier=4, seed=1)
    oos_rows = [row for row in corpus.rows if row.turn_kind == "out_of_scope"]
    assert oos_rows
    assert all(row.legal_action_count == 0 for row in oos_rows)
    assert corpus.quality_report.out_of_scope_never_from_nonempty_set is True


def test_row_post_init_rejects_out_of_scope_with_nonempty_legal_set() -> None:
    with pytest.raises(ValueError, match="out_of_scope rows must be provably derived"):
        TurnDispositionCorpusRowV1(
            row_id="bad",
            trace_id="trace0",
            tier=0,
            turn_index=0,
            turn_kind="out_of_scope",
            is_state_query=False,
            legal_action_count=1,  # non-empty: must be rejected
            coverage_complete=True,
            forced_singleton=True,
            gold_pick=None,
            features=(),
            target=None,
            split="train",
        )


def test_row_post_init_rejects_nonempty_kind_with_empty_legal_set() -> None:
    with pytest.raises(ValueError, match="requires a non-empty legal set"):
        TurnDispositionCorpusRowV1(
            row_id="bad2",
            trace_id="trace0",
            tier=0,
            turn_index=0,
            turn_kind="forced_emit",
            is_state_query=False,
            legal_action_count=0,
            coverage_complete=True,
            forced_singleton=False,
            gold_pick=None,
            features=(),
            target=None,
            split="train",
        )


def test_answer_rows_carry_is_state_query_and_no_target() -> None:
    """answer is a derived, non-learned fact -- never a model prediction --
    exactly as derive_turn_disposition requires."""
    corpus = build_turn_disposition_corpus(n_per_tier=4, seed=2)
    answer_rows = [row for row in corpus.rows if row.turn_kind == "answer"]
    assert answer_rows
    assert all(row.is_state_query for row in answer_rows)
    assert all(row.target is None for row in answer_rows)
    # Precedence proof: these rows are ALSO forced-singleton legal sets, yet
    # I1 resolves them as answer, never forced-emit -- proving answer
    # outranks the forced-singleton bypass.
    assert all(row.forced_singleton for row in answer_rows)


def test_forced_emit_rows_are_not_state_query_and_have_no_target() -> None:
    corpus = build_turn_disposition_corpus(n_per_tier=4, seed=3)
    forced_rows = [row for row in corpus.rows if row.turn_kind == "forced_emit"]
    assert forced_rows
    assert all(not row.is_state_query for row in forced_rows)
    assert all(row.forced_singleton for row in forced_rows)
    assert all(row.target is None for row in forced_rows)


def test_scored_rows_have_real_cost_derived_features_and_target() -> None:
    corpus = build_turn_disposition_corpus(n_per_tier=4, seed=4)
    scored_rows = [row for row in corpus.rows if row.turn_kind == "scored"]
    assert scored_rows
    for row in scored_rows:
        assert row.legal_action_count == 2
        assert row.target is not None
        margin, cheap_cost, exp_cost = row.features
        assert margin == pytest.approx(exp_cost - cheap_cost)
        assert margin in TIER_MARGINS
        # Hindsight-correct label: EMIT iff the naive (lower-cost) pick
        # matches the real, ground-truth accepted action for this trace.
        expected_label = (
            TurnDispositionLabel.EMIT
            if row.gold_pick == "cheap"
            else TurnDispositionLabel.CLARIFY
        )
        assert row.target.label is expected_label
        assert row.target.forced_singleton is False


def test_split_is_leakage_safe_at_trace_level() -> None:
    corpus = build_turn_disposition_corpus(n_per_tier=10, train_fraction=0.7, seed=5)
    train_traces = {row.trace_id for row in corpus.by_split("train")}
    heldout_traces = {row.trace_id for row in corpus.by_split("heldout")}
    assert train_traces
    assert heldout_traces
    assert train_traces.isdisjoint(heldout_traces)
    # Every trace's four rows must stay entirely within one split.
    split_by_trace: dict[str, set[str]] = {}
    for row in corpus.rows:
        split_by_trace.setdefault(row.trace_id, set()).add(row.split)
    assert all(len(splits) == 1 for splits in split_by_trace.values())


def test_both_splits_are_represented_in_every_tier() -> None:
    corpus = build_turn_disposition_corpus(n_per_tier=10, train_fraction=0.7, seed=6)
    for tier in range(len(TIER_MARGINS)):
        tier_rows = [row for row in corpus.rows if row.tier == tier]
        splits_present = {row.split for row in tier_rows}
        assert splits_present == {"train", "heldout"}


def test_corpus_is_deterministic_given_same_seed() -> None:
    corpus_a = build_turn_disposition_corpus(n_per_tier=6, seed=7)
    corpus_b = build_turn_disposition_corpus(n_per_tier=6, seed=7)
    assert [row.to_dict() for row in corpus_a.rows] == [
        row.to_dict() for row in corpus_b.rows
    ]


def test_quality_report_counts_match_rows() -> None:
    corpus = build_turn_disposition_corpus(n_per_tier=5, seed=8)
    report = corpus.quality_report
    assert report.total_rows == len(corpus.rows)
    assert sum(report.rows_by_kind.values()) == report.total_rows
    assert sum(report.rows_by_split.values()) == report.total_rows
    assert sum(report.rows_by_tier.values()) == report.total_rows


def test_n_per_tier_too_small_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_per_tier must be at least 2"):
        build_turn_disposition_corpus(n_per_tier=1, seed=9)


def test_tier_gold_cheap_probability_matches_tier_margins_length() -> None:
    assert len(TIER_GOLD_CHEAP_PROBABILITY) == len(TIER_MARGINS)
