from __future__ import annotations

import pytest

from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionForest,
    CompletionPath,
)
from slm_training.dsl.grammar.fastpath.lattice_search import (
    LatticeSearchState,
    StagnationTracker,
    TrajectoryCandidate,
    deduplicate_semantic_candidates,
    path_key,
    rank_forest,
    refine_hard_paths,
    select_trajectory_candidate,
    trajectory_orders,
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
    assert (
        rank_forest(forest, {(11,): 2.0}).signature
        == rank_forest(forest, {(12,): 2.0}).signature
    )


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
    assert state.rollback() == (prefix, None)
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
    state.choose([1], rank_forest(CompletionForest((first, second), "complete")))
    assert state.rollback(local_nogoods=False) == ([1], second)
    assert state.rollback() is None


def test_local_nogood_reprojects_at_exact_stable_prefix() -> None:
    first, second = _paths()
    state = LatticeSearchState(backtrack_limit=2)
    prefix = [1, 9]
    state.choose(prefix, rank_forest(CompletionForest((first, second), "complete")))

    restored, direct = state.rollback(local_nogoods=True) or (None, first)

    assert restored == prefix
    assert direct is None
    assert state.nogoods == {(tuple(prefix), path_key(first))}


def test_stagnation_requires_same_state_without_progress() -> None:
    tracker = StagnationTracker(patience=1)
    assert not tracker.observe("state", 4)
    assert not tracker.observe("state", 5)
    assert tracker.observe("state", 5)


def test_trajectories_are_seeded_and_never_add_illegal_paths() -> None:
    paths = _paths()
    ranked = rank_forest(CompletionForest(paths, "complete"))
    left = trajectory_orders(ranked, width=8, noise=2.0, seed=7)
    right = trajectory_orders(ranked, width=8, noise=2.0, seed=7)
    assert left == right
    assert left
    assert all(set(order) == set(paths) for order in left)


def test_semantic_deduplication_keeps_first_candidate() -> None:
    candidates = ("root = Card()", "root=Card()", "root = Stack([])")
    unique = deduplicate_semantic_candidates(
        candidates, lambda text: "card" if "Card" in text else "stack"
    )
    assert unique == ("root = Card()", "root = Stack([])")


def test_selector_never_prefers_invalid_over_valid() -> None:
    invalid = TrajectoryCandidate("invalid", False, True, 100.0, 1)
    valid = TrajectoryCandidate("valid", True, True, -100.0, 9, "ast")

    selected, unique = select_trajectory_candidate(
        (invalid, valid), semantic_dedup=True
    )

    assert selected == valid
    assert unique == 1


def test_gram_selector_deduplicates_valid_ast_fingerprints() -> None:
    candidates = (
        TrajectoryCandidate("first", True, True, 1.0, 5, "same"),
        TrajectoryCandidate("duplicate", True, True, 2.0, 4, "same"),
        TrajectoryCandidate("other", True, True, 0.5, 3, "other"),
    )

    selected, unique = select_trajectory_candidate(candidates, semantic_dedup=True)

    assert selected is candidates[0]
    assert unique == 2
