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


def test_masked_canvas_never_claims_exact(tok: DSLNativeTokenizer) -> None:
    """A canvas that still contains holes is a left-prefix over-approximation.

    Even with Γ-internal filtering (gamma_leaf_filters=False), the probe only
    validates the span left of the first hole — the fixed suffix is unseen —
    so the authority label must never be "exact".
    """
    engine = OpenUIIncrementalEngine()
    ids = list(tok.encode("root = Card([", add_special=False))
    canvas = [*ids, tok.mask_id, *tok.encode("]", add_special=False)]
    result = joint_multi_hole_support(engine, tok, canvas, gamma_leaf_filters=False)
    assert result.authority == "left_prefix_overapprox"
    assert result.reason.startswith("left_prefix_")


def test_left_prefix_admits_jointly_impossible_suffix(tok: DSLNativeTokenizer) -> None:
    """The a-[HOLE]-c counterexample, proven by bounded enumeration.

    Canvas: `root [HOLE] \n )` — the left prefix `root` is a valid incomplete
    prefix, so the probe admits the canvas. But NO single-token fill makes
    `root <X> \n )` a valid incomplete prefix (proven below by enumerating the
    whole vocabulary — the newline in the fixed suffix defeats even
    comment-byte fills, which would otherwise swallow a same-line suffix), so
    the joint fixed canvas is uncompletable: the probe result is an
    over-approximation exactly as the authority label states.
    """
    engine = OpenUIIncrementalEngine()
    root_ids = list(tok.encode("root", add_special=False))
    close_ids = list(tok.encode("\n)", add_special=False))
    canvas = [*root_ids, tok.mask_id, *close_ids]

    result = joint_multi_hole_support(engine, tok, canvas)
    assert result.admitted is True  # pins the over-approximation
    assert result.authority == "left_prefix_overapprox"

    # Bounded oracle: every single-token fill of the hole yields an invalid
    # prefix, so no completion preserves the fixed suffix.
    probe = OpenUIIncrementalEngine()
    special = {tok.mask_id, tok.pad_id, tok.bos_id, tok.eos_id}
    joint_possible = False
    for fill in range(tok.vocab_size):
        if fill in special:
            continue
        filled = [*root_ids, fill, *close_ids]
        text = tok.decode(filled, preserve_trailing_newline=True)
        if probe.set_prefix(text):
            joint_possible = True
            break
    assert joint_possible is False


def test_empty_left_span_admits_unconditionally(tok: DSLNativeTokenizer) -> None:
    """Pin: a canvas starting with a hole has an empty left span and the
    probe admits it without consulting the grammar at all."""
    engine = OpenUIIncrementalEngine()
    canvas = [tok.mask_id, *tok.encode(")))", add_special=False)]
    result = joint_multi_hole_support(engine, tok, canvas)
    assert result.admitted is True
    assert result.authority == "left_prefix_overapprox"


def test_multi_region_rejects_the_counterexample(tok: DSLNativeTokenizer) -> None:
    """The exact checker proves what the left-prefix probe cannot see:
    `root [HOLE] \\n )` has NO fill (visible or epsilon) preserving the
    fixed suffix, while admit_fill admits it."""
    from slm_training.dsl.grammar.fastpath.residual_support import (
        multi_region_support,
    )

    root_ids = list(tok.encode("root", add_special=False))
    close_ids = list(tok.encode("\n)", add_special=False))
    canvas = [*root_ids, tok.mask_id, *close_ids]

    left_prefix = joint_multi_hole_support(OpenUIIncrementalEngine(), tok, canvas)
    exact = multi_region_support(tok, canvas)
    assert left_prefix.admitted is True  # the over-approximation
    assert exact.admitted is False
    assert exact.authority == "exact_multi_region"
    assert exact.reason == "multi_region_unsupported"


def test_multi_region_supports_a_completable_canvas(tok: DSLNativeTokenizer) -> None:
    """`root [HOLE] Card(` is completable (fill `=`) — exact checker agrees
    with the left-prefix probe here, proving it is not merely stricter."""
    from slm_training.dsl.grammar.fastpath.residual_support import (
        multi_region_support,
    )

    root_ids = list(tok.encode("root", add_special=False))
    tail_ids = list(tok.encode(" Card(", add_special=False))
    canvas = [*root_ids, tok.mask_id, *tail_ids]
    exact = multi_region_support(tok, canvas)
    assert exact.admitted is True
    assert exact.authority == "exact_multi_region"


def test_multi_region_multiple_holes(tok: DSLNativeTokenizer) -> None:
    """Two holes with fixed structure between them resolve jointly."""
    from slm_training.dsl.grammar.fastpath.residual_support import (
        multi_region_support,
    )

    # root [HOLE] Card( [HOLE] )  — fills `=` and e.g. a symbol/empty arg.
    a = list(tok.encode("root", add_special=False))
    b = list(tok.encode(" Card(", add_special=False))
    c = list(tok.encode(")", add_special=False))
    canvas = [*a, tok.mask_id, *b, tok.mask_id, *c]
    exact = multi_region_support(tok, canvas)
    assert exact.admitted is True


def test_multi_region_budget_exhaustion_is_unknown(tok: DSLNativeTokenizer) -> None:
    from slm_training.dsl.grammar.fastpath.residual_support import (
        multi_region_support,
    )

    a = list(tok.encode("root", add_special=False))
    close_ids = list(tok.encode("\n)", add_special=False))
    canvas = [*a, tok.mask_id, *close_ids]
    starved = multi_region_support(tok, canvas, node_budget=1)
    assert starved.admitted is False
    assert starved.authority == "unknown"
    assert starved.reason == "multi_region_budget_exhausted"
