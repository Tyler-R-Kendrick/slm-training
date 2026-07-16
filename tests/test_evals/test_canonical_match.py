"""Canonical exact-match eval tests (D2)."""

from __future__ import annotations

from slm_training.evals.canonical_match import canonical_exact_match_rate, to_canonical

HERO = 'root = Stack([hero], "column")\nhero = Card([t])\nt = TextContent(":x")'
HERO_RENAMED = (
    'root = Stack([box], "column")\nbox = Card([label])\nlabel = TextContent(":x")'
)
CTA = 'root = Stack([cta])\ncta = Button(":c")'


def test_canonical_match_rescues_alpha_renamed_prediction() -> None:
    # Prediction is correct but alpha-renamed vs gold: surface misses, canonical hits.
    report = canonical_exact_match_rate([(HERO_RENAMED, HERO)])
    assert report["n"] == 1
    assert report["surface_exact_match"] == 0.0
    assert report["canonical_exact_match"] == 1.0
    assert report["canonicalization_rescued"] == 1


def test_canonical_match_counts_true_mismatch() -> None:
    report = canonical_exact_match_rate([(CTA, HERO)])
    assert report["canonical_exact_match"] == 0.0
    assert report["canonicalization_rescued"] == 0


def test_canonical_match_surface_and_canonical_agree_on_identity() -> None:
    report = canonical_exact_match_rate([(HERO, HERO)])
    assert report["surface_exact_match"] == 1.0
    assert report["canonical_exact_match"] == 1.0
    assert report["canonicalization_rescued"] == 0


def test_to_canonical_returns_none_on_unparseable() -> None:
    assert to_canonical("nonsense (((") is None
    assert to_canonical(HERO) is not None
