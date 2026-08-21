"""Wilcoxon / sign-test paired screening decisions."""

from __future__ import annotations

from slm_training.autoresearch.paired_stats import (
    exact_sign_test,
    paired_screening_test,
    wilcoxon_signed_rank_p,
)


def test_wilcoxon_all_positive_matches_sign_flip_count() -> None:
    deltas = [1.0, 2.0, 3.0, 4.0, 5.0]
    w_plus, p, n = wilcoxon_signed_rank_p(deltas)
    assert n == 5
    assert w_plus == 15.0
    # One of 32 sign patterns is all-positive; two-sided = 2/32.
    assert abs(p - 2 / 32) < 1e-12


def test_sign_test_known_vector() -> None:
    pos, neg, p = exact_sign_test([1, 1, 1, 1, -1])
    assert pos == 4 and neg == 1
    # 2 * C(5,0..1) / 32 = 2 * 6 / 32 = 12/32.
    assert float(p) == 12 / 32


def test_ties_dropped_and_insufficient_nontied_is_mechanism_no_effect() -> None:
    result = paired_screening_test(
        [0.0, 0.0, 0.0, 0.1, -0.1], min_nontied_pairs=5
    )
    assert result.n_ties == 3
    assert result.n_nontied == 2
    assert result.verdict == "mechanism_no_effect"
    assert "nontied_pairs" in result.reason


def test_all_ties_is_mechanism_no_effect() -> None:
    result = paired_screening_test([0.0] * 6)
    assert result.verdict == "mechanism_no_effect"


def test_wilcoxon_win_on_consistent_positive_deltas() -> None:
    # n=12 > sign-test fallback; all-positive is extreme.
    deltas = [float(i) for i in range(1, 13)]
    result = paired_screening_test(deltas, min_nontied_pairs=5)
    assert result.kind == "wilcoxon_signed_rank"
    assert result.verdict == "win"
    assert result.p_value < 0.05


def test_sign_test_fallback_for_tiny_n() -> None:
    deltas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    result = paired_screening_test(deltas, min_nontied_pairs=5)
    assert result.kind == "sign_test"
    assert result.verdict == "win"
