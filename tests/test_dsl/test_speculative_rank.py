"""Decode invariant I3: deterministic speculative ranking over symbol tables.

Ranking is a lever; legality is not. Every test here checks one of those two
halves — that the scorer orders sensibly and reproducibly, and that it can
never add a candidate the completion forest did not already prove legal.
"""

from __future__ import annotations

import json

import pytest

from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionForest,
    CompletionPath,
)
from slm_training.dsl.grammar.fastpath.speculative_rank import (
    NGRAM_TABLE_SCHEMA,
    NgramTableV1,
    SpeculativeRankerV1,
    build_ngram_table,
    load_ranker,
    speculative_span,
)


def _table(order: int = 3) -> NgramTableV1:
    # "1 2 3" is the corpus habit; "1 2 9" appears once.
    return build_ngram_table([[1, 2, 3]] * 8 + [[1, 2, 9]], order=order)


def _forest(paths: list[list[int]], coverage: str = "complete") -> CompletionForest:
    return CompletionForest(
        paths=tuple(CompletionPath(token_ids=tuple(p), kind="action") for p in paths),
        coverage=coverage,  # type: ignore[arg-type]
    )


def test_ngram_table_prefers_the_corpus_continuation() -> None:
    table = _table()
    assert table.log_prob([1, 2], 3) > table.log_prob([1, 2], 9)


def test_unseen_continuation_scores_finite_not_negative_infinity() -> None:
    """A total order needs finite scores, or margins stop being comparable."""
    table = _table()
    score = table.log_prob([1, 2], 4242)
    assert score == pytest.approx(score)  # not nan
    assert score > float("-inf")


def test_backoff_reaches_the_unigram_level() -> None:
    table = _table()
    # Context never observed: backoff must still separate a frequent token
    # from one that never appears at all.
    assert table.log_prob([777, 888], 3) > table.log_prob([777, 888], 4242)


def test_table_round_trips_through_json() -> None:
    table = _table()
    restored = NgramTableV1.from_dict(json.loads(json.dumps(table.to_dict())))
    assert restored.order == table.order
    assert restored.corpus_fingerprint == table.corpus_fingerprint
    assert restored.log_prob([1, 2], 3) == pytest.approx(table.log_prob([1, 2], 3))


def test_table_write_and_load_round_trip(tmp_path) -> None:
    table = _table()
    path = table.write(tmp_path / "nested" / "table.json")
    assert json.loads(path.read_text())["schema"] == NGRAM_TABLE_SCHEMA
    assert NgramTableV1.load(path).corpus_fingerprint == table.corpus_fingerprint


def test_ranker_orders_legal_paths_without_changing_membership() -> None:
    ranker = SpeculativeRankerV1(table=_table())
    paths = (
        CompletionPath(token_ids=(9,), kind="a"),
        CompletionPath(token_ids=(3,), kind="b"),
    )
    choice = ranker.choose([1, 2], paths)
    assert choice is not None
    assert choice.best_index == 1  # token 3 is the corpus habit
    scored = ranker.score_paths([1, 2], paths)
    assert set(scored) == {(9,), (3,)}


def test_ranking_is_reproducible_regardless_of_input_order() -> None:
    ranker = SpeculativeRankerV1(table=_table())
    a = CompletionPath(token_ids=(3,), kind="a")
    b = CompletionPath(token_ids=(9,), kind="b")
    forward = ranker.choose([1, 2], (a, b))
    reverse = ranker.choose([1, 2], (b, a))
    assert forward is not None and reverse is not None
    assert forward.scores[forward.best_index] == pytest.approx(
        reverse.scores[reverse.best_index]
    )


def test_zero_margin_never_commits_without_the_model() -> None:
    """The safe default: supply an ordering, let the model decide."""
    ranker = SpeculativeRankerV1(table=_table(), margin=0.0)
    choice = ranker.choose(
        [1, 2],
        (
            CompletionPath(token_ids=(3,), kind="a"),
            CompletionPath(token_ids=(9,), kind="b"),
        ),
    )
    assert choice is not None
    assert not choice.confident


def test_confident_margin_commits() -> None:
    ranker = SpeculativeRankerV1(table=_table(), margin=0.5)
    choice = ranker.choose(
        [1, 2],
        (
            CompletionPath(token_ids=(3,), kind="a"),
            CompletionPath(token_ids=(9,), kind="b"),
        ),
    )
    assert choice is not None
    assert choice.confident
    assert choice.margin >= 0.5


def test_span_commits_forced_singletons_with_no_ranker() -> None:
    """A singleton domain needs no scorer and no forward — I2 inside a span."""
    sequence = {0: [[7]], 1: [[8]], 2: [[9]]}

    def build(prefix: list[int]) -> CompletionForest:
        return _forest(sequence[len(prefix)])

    span = speculative_span([], build, max_tokens=3)
    assert span.token_ids == (7, 8, 9)
    assert span.forced_tokens == 3
    assert span.ranked_tokens == 0


def test_span_stops_at_a_branch_point_without_a_ranker() -> None:
    def build(prefix: list[int]) -> CompletionForest:
        return _forest([[7]]) if not prefix else _forest([[8], [9]])

    span = speculative_span([], build, max_tokens=4)
    assert span.token_ids == (7,)
    assert span.stop_reason == "no_ranker"


def test_span_refuses_to_speculate_past_an_incomplete_proof() -> None:
    """A partial completion forest is not a proof; speculation must stop."""

    def build(prefix: list[int]) -> CompletionForest:
        return _forest([[7]]) if not prefix else _forest([[8]], coverage="partial")

    span = speculative_span([], build, max_tokens=4, ranker=SpeculativeRankerV1(_table()))
    assert span.token_ids == (7,)
    assert span.stop_reason == "incomplete_coverage"


def test_span_stops_on_an_empty_domain_rather_than_inventing_a_token() -> None:
    def build(prefix: list[int]) -> CompletionForest:
        return _forest([[7]]) if not prefix else _forest([])

    span = speculative_span([], build, max_tokens=4)
    assert span.token_ids == (7,)
    assert span.stop_reason == "empty_domain"


def test_span_only_ever_emits_tokens_the_forest_offered() -> None:
    """The legality half: nothing in a span came from outside the domain."""
    domains = {0: [[1]], 1: [[2]], 2: [[3], [9]], 3: [[5]]}

    def build(prefix: list[int]) -> CompletionForest:
        return _forest(domains[len(prefix)])

    span = speculative_span(
        [], build, ranker=SpeculativeRankerV1(_table(), margin=0.1), max_tokens=4
    )
    emitted = list(span.token_ids)
    for index, token in enumerate(emitted):
        offered = {path[0] for path in domains[index]}
        assert token in offered


def test_span_respects_its_token_budget() -> None:
    def build(prefix: list[int]) -> CompletionForest:
        return _forest([[1, 1, 1]])

    span = speculative_span([], build, max_tokens=2)
    assert span.token_ids == ()
    assert span.stop_reason == "budget"


def test_load_ranker_is_none_when_unconfigured() -> None:
    assert load_ranker(None) is None


def test_load_ranker_rejects_an_empty_table(tmp_path) -> None:
    """A run must be able to name the table that ranked it."""
    path = tmp_path / "empty.json"
    build_ngram_table([], order=2).write(path)
    with pytest.raises(ValueError, match="empty"):
        load_ranker(path)


def test_unsupported_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="schema"):
        NgramTableV1.from_dict({"schema": "something/v9", "order": 2, "counts": []})
