"""Regression tests for eval score policies (EFS1-03, SLM-110)."""

from __future__ import annotations

import math

import pytest

from slm_training.evals.score_policy import (
    CandidatePath,
    ContentFloorPolicy,
    GrammarAlignedMassPolicy,
    MinimumMassRemaskPolicy,
    RawCumulativePolicy,
    SemanticLengthNormPolicy,
    build_policy,
    compare_policies,
    rank_candidates,
)


def _path(
    candidate_id: str,
    log_probs: list[float],
    *,
    removed_mass: list[float] | None = None,
    semantic_mask: list[float] | None = None,
) -> CandidatePath:
    return CandidatePath(
        candidate_id=candidate_id,
        token_ids=tuple(range(len(log_probs))),
        log_probs=tuple(log_probs),
        removed_mass=tuple(removed_mass) if removed_mass is not None else None,
        semantic_mask=tuple(semantic_mask) if semantic_mask is not None else None,
    )


def test_raw_cumulative_prefers_longer_negative_nll() -> None:
    # log-probabilities are negative; summing more of them yields a lower score.
    short = _path("short", [-1.0])
    long = _path("long", [-1.0, -1.0])
    policy = RawCumulativePolicy()
    assert policy.score(long) < policy.score(short)


def test_semantic_length_norm_reduces_length_bias() -> None:
    # Both candidates have -1.0 per semantic decision; raw cumulative favors short.
    short = _path("short", [-1.0], semantic_mask=[1.0])
    long = _path("long", [-1.0, -1.0], semantic_mask=[1.0, 1.0])
    raw = RawCumulativePolicy()
    norm = SemanticLengthNormPolicy(alpha=1.0)
    assert raw.score(short) > raw.score(long)
    assert norm.score(short) == pytest.approx(norm.score(long))


def test_grammar_aligned_mass_penalizes_removed_mass() -> None:
    # Same cumulative log-prob; candidate with more removed mass scores worse.
    low_mass = _path("low", [-1.0, -1.0], removed_mass=[0.1, 0.1])
    high_mass = _path("high", [-1.0, -1.0], removed_mass=[0.9, 0.9])
    policy = GrammarAlignedMassPolicy(beta=1.0)
    assert policy.score(low_mass) > policy.score(high_mass)


def test_minimum_mass_remask_penalizes_low_retained_mass() -> None:
    same = _path("same", [-1.0, -1.0], removed_mass=[0.5, 0.5], semantic_mask=[0.0, 1.0])
    policy = MinimumMassRemaskPolicy(gamma=1.0)
    score = policy.score(same)
    assert math.isfinite(score)
    # Low retained mass (log(0.5) < 0) is a penalty on the semantic position.
    assert score < sum(same.log_probs)


def test_content_floor_rejects_empty_candidates() -> None:
    empty = _path("empty", [-0.5], semantic_mask=[0.0])
    populated = _path("pop", [-1.0, -1.0], semantic_mask=[1.0, 1.0])
    policy = ContentFloorPolicy(min_semantic_decisions=1)
    assert policy.score(empty) == float("-inf")
    assert math.isfinite(policy.score(populated))


def test_rank_candidates_orders_by_score() -> None:
    paths = [
        _path("b", [-2.0, -2.0]),
        _path("a", [-1.0]),
        _path("c", [-3.0]),
    ]
    ranked = rank_candidates(paths, RawCumulativePolicy())
    assert [cid for cid, _ in ranked] == ["a", "c", "b"]


def test_compare_policies_reports_rank_changes() -> None:
    # Empty wins raw cumulative because it is shorter; length norm flips to populated.
    empty = _path("empty", [-1.0])
    populated = _path("populated", [-1.0, -1.0], semantic_mask=[1.0, 1.0])
    comparison = compare_policies(
        [populated, empty],
        [RawCumulativePolicy(), SemanticLengthNormPolicy(alpha=1.0)],
    )
    assert comparison["rankings"]["raw_cumulative"][0] == "empty"
    assert comparison["rankings"]["semantic_length_norm"][0] == "populated"


def test_build_policy_by_name() -> None:
    policy = build_policy("semantic_length_norm", alpha=0.75)
    assert policy.name == "semantic_length_norm"
    assert policy.alpha == 0.75  # type: ignore[attr-defined]


def test_unknown_policy_raises() -> None:
    with pytest.raises(ValueError):
        build_policy("no_such_policy")


def test_candidate_path_validates_lengths() -> None:
    with pytest.raises(ValueError):
        CandidatePath(
            candidate_id="bad",
            token_ids=(0, 1),
            log_probs=(-1.0,),
        )
