"""Paired screening tests: Wilcoxon signed-rank with an exact sign-test fallback.

Ties are dropped. A cycle with fewer than ``min_nontied_pairs`` non-tied
deltas is ``mechanism_no_effect`` (not a loss). Stdlib only — no scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from statistics import NormalDist
from typing import Literal, Sequence

from slm_training.autoresearch.evidence_ledger import (
    parse_alpha,
    sign_test_min_two_sided_p,
)

PairedTestKind = Literal["wilcoxon_signed_rank", "sign_test"]
PairedVerdict = Literal["win", "loss", "inconclusive", "mechanism_no_effect"]

DEFAULT_MIN_NONTIED_PAIRS = 5
DEFAULT_ALPHA = Fraction(1, 20)
# Exact Wilcoxon enumeration is 2^n; cap keeps the walk bounded.
_WILCOXON_EXACT_MAX_N = 16
_SIGN_TEST_MAX_N = 8
_STANDARD_NORMAL = NormalDist()

MECHANISM_NO_EFFECT = "mechanism_no_effect"


@dataclass(frozen=True)
class PairedTestResult:
    kind: PairedTestKind
    n_pairs: int
    n_nontied: int
    n_ties: int
    statistic: float
    p_value: float
    alpha: str
    verdict: PairedVerdict
    reason: str
    promotion_authority: Literal[False] = False


def _two_sided_binom_p(n_pos: int, n_neg: int) -> Fraction:
    n = n_pos + n_neg
    if n <= 0:
        return Fraction(1)
    k = n_pos
    # Two-sided exact binomial p=1/2: 2 * min(P(X<=k), P(X>=k)), capped at 1.
    left = sum(math.comb(n, i) for i in range(0, k + 1))
    right = sum(math.comb(n, i) for i in range(k, n + 1))
    tail = min(left, right)
    return min(Fraction(1), Fraction(2 * tail, 2**n))


def exact_sign_test(
    deltas: Sequence[float],
    *,
    alpha: Fraction | str | int = DEFAULT_ALPHA,
) -> tuple[int, int, Fraction]:
    """Return (n_pos, n_neg, two-sided p) after dropping zeros."""

    pos = neg = 0
    for raw in deltas:
        if raw > 0:
            pos += 1
        elif raw < 0:
            neg += 1
    p = _two_sided_binom_p(pos, neg)
    _ = parse_alpha(alpha)
    return pos, neg, p


def _average_ranks(abs_values: Sequence[float]) -> list[float]:
    order = sorted(range(len(abs_values)), key=lambda i: abs_values[i])
    ranks = [0.0] * len(abs_values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs_values[order[j + 1]] == abs_values[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _wilcoxon_w_plus(signed: Sequence[float]) -> tuple[float, list[float], list[int]]:
    abs_vals = [abs(v) for v in signed]
    ranks = _average_ranks(abs_vals)
    signs = [1 if v > 0 else -1 for v in signed]
    w_plus = sum(r for r, s in zip(ranks, signs) if s > 0)
    return w_plus, ranks, signs


def _wilcoxon_exact_p(ranks: Sequence[float], w_plus: float) -> float:
    n = len(ranks)
    total = 1 << n
    extreme = 0
    # Two-sided: count assignments whose W+ is at least as far from the
    # midpoint as the observed W+.
    mid = sum(ranks) / 2.0
    observed_dev = abs(w_plus - mid)
    for mask in range(total):
        w = 0.0
        for i, rank in enumerate(ranks):
            if mask & (1 << i):
                w += rank
        if abs(w - mid) + 1e-12 >= observed_dev:
            extreme += 1
    return extreme / total


def _wilcoxon_normal_p(ranks: Sequence[float], w_plus: float) -> float:
    n = len(ranks)
    mean = n * (n + 1) / 4.0
    # Tie-corrected variance of Wilcoxon W+.
    var = n * (n + 1) * (2 * n + 1) / 24.0
    # Group tied abs ranks.
    from collections import Counter

    counts = Counter(ranks)
    tie_adj = sum(t * t * t - t for t in counts.values()) / 48.0
    var -= tie_adj
    if var <= 0:
        return 1.0
    z = (w_plus - mean) / math.sqrt(var)
    # Two-sided.
    return float(2 * (1.0 - _STANDARD_NORMAL.cdf(abs(z))))


def wilcoxon_signed_rank_p(deltas: Sequence[float]) -> tuple[float, float, int]:
    """Return (W+, two-sided p, n_nontied) on non-zero deltas."""

    signed = [float(v) for v in deltas if v != 0]
    n = len(signed)
    if n == 0:
        return 0.0, 1.0, 0
    w_plus, ranks, _signs = _wilcoxon_w_plus(signed)
    if n <= _WILCOXON_EXACT_MAX_N:
        p = _wilcoxon_exact_p(ranks, w_plus)
    else:
        p = _wilcoxon_normal_p(ranks, w_plus)
    return w_plus, min(1.0, max(0.0, p)), n


def paired_screening_test(
    deltas: Sequence[float],
    *,
    alpha: Fraction | str | int = DEFAULT_ALPHA,
    min_nontied_pairs: int = DEFAULT_MIN_NONTIED_PAIRS,
    kind: PairedTestKind = "wilcoxon_signed_rank",
) -> PairedTestResult:
    """Decision on per-record paired deltas (positive = candidate better)."""

    alpha_f = parse_alpha(alpha)
    values = [float(v) for v in deltas]
    n_pairs = len(values)
    ties = sum(1 for v in values if v == 0)
    nontied = n_pairs - ties
    if nontied < max(1, int(min_nontied_pairs)):
        return PairedTestResult(
            kind=kind,
            n_pairs=n_pairs,
            n_nontied=nontied,
            n_ties=ties,
            statistic=0.0,
            p_value=1.0,
            alpha=str(alpha_f),
            verdict="mechanism_no_effect",
            reason=f"{MECHANISM_NO_EFFECT}:nontied_pairs={nontied}<{min_nontied_pairs}",
        )

    use_sign = kind == "sign_test" or nontied <= _SIGN_TEST_MAX_N
    if use_sign:
        pos, neg, p_frac = exact_sign_test(values, alpha=alpha_f)
        statistic = float(pos - neg)
        p_value = float(p_frac)
        used: PairedTestKind = "sign_test"
        # Sign-test floor: if even the all-agree p cannot beat alpha, inconclusive.
        if sign_test_min_two_sided_p(nontied) > alpha_f:
            return PairedTestResult(
                kind=used,
                n_pairs=n_pairs,
                n_nontied=nontied,
                n_ties=ties,
                statistic=statistic,
                p_value=p_value,
                alpha=str(alpha_f),
                verdict="inconclusive",
                reason="sign_test_undecidable",
            )
    else:
        statistic, p_value, _n = wilcoxon_signed_rank_p(values)
        used = "wilcoxon_signed_rank"

    if p_value > float(alpha_f):
        return PairedTestResult(
            kind=used,
            n_pairs=n_pairs,
            n_nontied=nontied,
            n_ties=ties,
            statistic=statistic,
            p_value=p_value,
            alpha=str(alpha_f),
            verdict="inconclusive",
            reason="paired_p_above_alpha",
        )
    mean_delta = sum(v for v in values if v != 0) / nontied
    verdict: PairedVerdict = "win" if mean_delta > 0 else "loss"
    return PairedTestResult(
        kind=used,
        n_pairs=n_pairs,
        n_nontied=nontied,
        n_ties=ties,
        statistic=statistic,
        p_value=p_value,
        alpha=str(alpha_f),
        verdict=verdict,
        reason=f"paired_{used}_{verdict}",
    )


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_MIN_NONTIED_PAIRS",
    "MECHANISM_NO_EFFECT",
    "PairedTestResult",
    "PairedTestKind",
    "PairedVerdict",
    "exact_sign_test",
    "paired_screening_test",
    "wilcoxon_signed_rank_p",
]
