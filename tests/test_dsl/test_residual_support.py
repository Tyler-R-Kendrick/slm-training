"""Residual multi-hole predicate + legal-set ranking (claim family B)."""

from __future__ import annotations

import math

import pytest

from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
from slm_training.dsl.grammar.fastpath.residual_support import (
    ResidualSupportResult,
    joint_multi_hole_support,
    rank_inside_legal_residual,
)
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer


@pytest.fixture(scope="module")
def tok() -> DSLNativeTokenizer:
    return DSLNativeTokenizer.build()


def test_joint_multi_hole_support_admits_valid_prefix(tok: DSLNativeTokenizer) -> None:
    engine = OpenUIIncrementalEngine()
    ids = list(tok.encode("root = Card([", add_special=False))
    result = joint_multi_hole_support(engine, tok, ids, gamma_leaf_filters=True)
    assert isinstance(result, ResidualSupportResult)
    assert result.admitted is True
    assert result.authority == "honest_overapprox"
    assert result.soft_legality is False
    assert result.gamma_leaf_filters is True


def test_joint_multi_hole_support_rejects_garbage(tok: DSLNativeTokenizer) -> None:
    engine = OpenUIIncrementalEngine()
    # Unbalanced / illegal fragment that is not a valid incomplete prefix.
    garbage = list(tok.encode(")))", add_special=False))
    result = joint_multi_hole_support(
        engine, tok, garbage, gamma_leaf_filters=False
    )
    assert result.admitted is False
    assert result.authority == "exact"
    assert result.soft_legality is False


def test_soft_legality_forbidden() -> None:
    with pytest.raises(ValueError, match="soft legality"):
        ResidualSupportResult(
            admitted=True,
            authority="exact",
            gamma_leaf_filters=False,
            reason="x",
            soft_legality=True,
        )


def test_rank_inside_legal_residual_never_picks_illegal() -> None:
    scores = (0.1, 9.9, 0.5)
    legal = (True, False, True)
    winner, masked = rank_inside_legal_residual(scores, legal)
    assert winner == 2
    assert math.isinf(masked[1]) and masked[1] < 0
    assert masked[0] == 0.1
    assert masked[2] == 0.5


def test_rank_inside_legal_residual_empty_legal_set() -> None:
    winner, masked = rank_inside_legal_residual((1.0, 2.0), (False, False))
    assert winner is None
    assert all(math.isinf(x) and x < 0 for x in masked)
