from __future__ import annotations

import pytest

from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionForest,
    CompletionPath,
)
from slm_training.dsl.grammar.fastpath.lattice_search import (
    LatticeSearchState,
    rank_forest,
    refine_hard_paths,
)


def _paths() -> tuple[CompletionPath, CompletionPath]:
    return (
        CompletionPath((11,), "component"),
        CompletionPath((12,), "component"),
    )


def test_soft_scores_only_order_hard_candidates() -> None:
    paths = _paths()
    ranked = rank_forest(
        CompletionForest(paths, "complete"), {(11,): -100.0, (12,): 2.0}
    )
    assert [path.token_ids for path in ranked.paths] == [(12,), (11,)]
    assert set(ranked.paths) == set(paths)


def test_signature_is_deterministic_across_soft_ordering() -> None:
    paths = _paths()
    forest = CompletionForest(paths, "complete")
    assert rank_forest(forest, {(11,): 2.0}).signature == rank_forest(
        forest, {(12,): 2.0}
    ).signature


def test_hard_refinement_cannot_add_candidates() -> None:
    left, right = _paths()
    assert refine_hard_paths((left, right), (right,)) == (right,)
    with pytest.raises(ValueError, match="cannot add candidates"):
        refine_hard_paths((left,), (left, right))


def test_bottom_rolls_back_and_records_local_nogood() -> None:
    first, second = _paths()
    prefix = [1, 7]
    state = LatticeSearchState(backtrack_limit=2)
    chosen = state.choose(
        prefix,
        rank_forest(CompletionForest((first, second), "complete")),
    )
    assert chosen == first
    assert rank_forest(CompletionForest((), "none")).is_bottom
    assert state.rollback() == (prefix, second)
    assert (tuple(prefix), (11,)) in state.nogoods
    reranked = rank_forest(
        CompletionForest((first, second), "complete"),
        prefix=tuple(prefix),
        nogoods=frozenset(state.nogoods),
    )
    assert reranked.paths == (second,)


def test_backtracking_is_bounded() -> None:
    state = LatticeSearchState(backtrack_limit=1)
    first, second = _paths()
    state.choose(
        [1], rank_forest(CompletionForest((first, second), "complete"))
    )
    assert state.rollback() == ([1], second)
    assert state.rollback() is None
