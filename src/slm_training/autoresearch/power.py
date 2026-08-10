"""Pure power/decidability arithmetic for pre-run experiment screening.

Answers, *before any run is spent*, two independent questions about a
proposed measurement design:

1. **Attainability** — can the design's decision statistic reject at
   ``alpha`` at all?  This is combinatorics, not an empirical question:
   an exact test over a finite arrangement space has a hard floor on the
   smallest p-value it can ever produce.
2. **Sensitivity** — is the design's minimum detectable effect (MDE) at
   the declared budget no larger than the preregistered minimum effect?

Both questions are the §4-R1 successor approach to RC1 of
``docs/design/harness-evolution-architecture-review-20260809.md``:
screening at n=3 is below the decidability floor (paired sign-test
minimum two-sided p = 0.25), and ``minimum_effect: 0.01`` is ~33x below
measurement granularity.

Paired vs unpaired reconciliation
---------------------------------

Two distinct exact tests appear in this repo and in the review doc, and
they have different p-value floors.  Both are exposed here behind
``paired``:

* **Paired** (``paired=True``): the sign test / sign-flip permutation
  test over ``n`` paired differences — ``2^n`` sign arrangements, so the
  minimum two-sided p is ``2 * (1/2)^n`` (n=3 -> 0.25, the number RC1
  cites).  This is the variant the climb loop actually uses: the credit
  engine decides screening arms with a paired sign test over signed
  per-document effects (``credit_engine._two_sided_p_from_signed_effects``),
  and darkfactory phase 1 shipped the exact arithmetic in
  ``evidence_ledger.sign_test_min_two_sided_p`` — reused here, not
  re-implemented.
* **Unpaired** (``paired=False``, the default of these public
  signatures): the two-sample permutation test over ``n_a`` vs ``n_b``
  independent observations — ``C(n_a + n_b, n_a)`` distinct labelings,
  so the minimum one-sided p is ``1 / C(n_a + n_b, n_a)`` and the
  minimum two-sided p is ``2 / C(n_a + n_b, n_a)`` (for a resolving
  statistic whose mirror labeling realizes the opposite extreme).  At
  n=3 vs n=3 that is one-sided 1/20 = 0.05 and two-sided 2/20 = 0.1.

Sided convention: every default in this module is **two-sided**, matching
the loop's two-sided sign test; ``sided=1`` is available where noted.

Normal-approximation assumptions (``min_detectable_effect`` /
``required_n_for_effect``): two-sample z-power with known common ``sd``,
balanced arms of size ``n``, two-sided ``alpha``.  ``paired=True`` swaps
in the one-sample form where ``sd`` is the SD of the paired differences.
These are approximations used for budgeting, not for inference — the
attainability floor above is exact.
"""

from __future__ import annotations

import math
from fractions import Fraction
from statistics import NormalDist

from pydantic import BaseModel, ConfigDict, Field

from slm_training.autoresearch.evidence_ledger import (
    parse_alpha,
    required_sign_test_n,
    sign_test_min_two_sided_p,
)

__all__ = [
    "DecidabilityReport",
    "is_decidable",
    "min_attainable_n",
    "min_attainable_p",
    "min_detectable_effect",
    "required_n_for_effect",
]

_STANDARD_NORMAL = NormalDist()


def _z(quantile: float) -> float:
    return _STANDARD_NORMAL.inv_cdf(quantile)


def min_attainable_p(
    n_a: int,
    n_b: int,
    test: str = "permutation",
    *,
    paired: bool = False,
    sided: int = 2,
) -> float:
    """Smallest p-value the design can ever produce (exact combinatorics).

    Unpaired (default): two-sample permutation test over ``n_a`` vs
    ``n_b`` observations; ``C(n_a + n_b, n_a)`` labelings give a
    one-sided floor of ``1/C`` and a two-sided floor of ``min(1, 2/C)``.
    n=3 vs n=3: one-sided 0.05, two-sided 0.1.

    Paired: sign test / sign-flip permutation over ``min(n_a, n_b)``
    pairs; two-sided floor ``2 * (1/2)^n`` (n=3 -> 0.25, per the review
    doc), one-sided ``(1/2)^n``.  Reuses the darkfactory phase 1 exact
    arithmetic (``evidence_ledger.sign_test_min_two_sided_p``).

    ``n_a <= 0`` or ``n_b <= 0`` carries no evidence and returns 1.0.
    """
    if test not in {"permutation", "sign"}:
        raise ValueError(f"unsupported test: {test!r}")
    if sided not in (1, 2):
        raise ValueError(f"sided must be 1 or 2, got {sided!r}")
    if n_a <= 0 or n_b <= 0:
        return 1.0
    if paired or test == "sign":
        n_pairs = min(n_a, n_b)
        two_sided = sign_test_min_two_sided_p(n_pairs)
        floor = two_sided if sided == 2 else two_sided / 2
    else:
        arrangements = math.comb(n_a + n_b, n_a)
        floor = min(Fraction(1), Fraction(sided, arrangements))
    return float(floor)


def min_attainable_n(
    alpha: float = 0.05, *, paired: bool = False, sided: int = 2
) -> int:
    """Smallest balanced per-arm ``n`` whose p-value floor is <= ``alpha``.

    For the paired two-sided case this is exactly darkfactory phase 1's
    ``evidence_ledger.required_sign_test_n`` (reused).  For the other
    variants the same "smallest feasible n" search is run against
    :func:`min_attainable_p`.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if paired and sided == 2:
        return required_sign_test_n(parse_alpha(alpha))
    n = 1
    while min_attainable_p(n, n, paired=paired, sided=sided) > alpha:
        n += 1
    return n


def min_detectable_effect(
    n: int,
    sd: float,
    alpha: float = 0.05,
    power: float = 0.8,
    *,
    paired: bool = False,
) -> float:
    """Minimum detectable effect under the normal approximation.

    Unpaired (default): balanced two-sample z-test with known common
    ``sd`` and ``n`` per arm — ``MDE = (z_{1-alpha/2} + z_{power}) * sd
    * sqrt(2/n)``.  Paired: one-sample form on ``n`` paired differences
    whose SD is ``sd`` — ``MDE = (z_{1-alpha/2} + z_{power}) * sd /
    sqrt(n)``.  Two-sided ``alpha``; an approximation for budgeting, not
    inference.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if sd < 0:
        raise ValueError("sd must be non-negative")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    z_total = _z(1 - alpha / 2) + _z(power)
    scale = 1.0 / math.sqrt(n) if paired else math.sqrt(2.0 / n)
    return z_total * sd * scale


def required_n_for_effect(
    effect: float,
    sd: float,
    alpha: float = 0.05,
    power: float = 0.8,
    *,
    paired: bool = False,
) -> int:
    """Smallest per-arm ``n`` whose MDE is <= ``effect`` (normal approx).

    Inverse of :func:`min_detectable_effect`: unpaired
    ``n = ceil(2 * ((z_total * sd) / effect)^2)``, paired
    ``n = ceil(((z_total * sd) / effect)^2)``; never less than 1.
    """
    if effect <= 0:
        raise ValueError("effect must be positive")
    if sd < 0:
        raise ValueError("sd must be non-negative")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    if sd == 0:
        return 1
    z_total = _z(1 - alpha / 2) + _z(power)
    ratio = (z_total * sd) / effect
    factor = 1.0 if paired else 2.0
    return max(1, math.ceil(factor * ratio * ratio))


class DecidabilityReport(BaseModel):
    """Pre-run verdict on whether a design can possibly return an answer."""

    model_config = ConfigDict(frozen=True)

    decidable: bool
    min_attainable_p: float
    min_detectable_effect: float
    required_n_for_effect: int
    reasons: list[str] = Field(default_factory=list)


def is_decidable(
    n: int,
    sd: float,
    minimum_effect: float,
    alpha: float = 0.05,
    paired: bool = False,
) -> DecidabilityReport:
    """Full decidability report for a balanced design of per-arm size ``n``.

    Decidable iff BOTH hold: the exact p-value floor
    (:func:`min_attainable_p`, two-sided) is <= ``alpha``, AND the
    normal-approximation MDE at ``n`` is <= ``minimum_effect``.  The
    report's ``required_n_for_effect`` already folds in the attainability
    floor (:func:`min_attainable_n`), so re-planning to that ``n``
    yields a decidable design on both axes.

    Never raises on degenerate inputs (``n <= 0``, ``sd <= 0``,
    ``minimum_effect <= 0``, out-of-range ``alpha``): they produce an
    undecidable report with explanatory reasons instead, so callers in
    the loop can fail closed without try/except.
    """
    reasons: list[str] = []
    if not 0 < alpha < 1:
        reasons.append(f"alpha={alpha!r} is not in (0, 1); design is undecidable")
        return DecidabilityReport(
            decidable=False,
            min_attainable_p=1.0,
            min_detectable_effect=math.inf,
            required_n_for_effect=0,
            reasons=reasons,
        )
    variant = "paired sign test" if paired else "two-sample permutation test"
    if n <= 0:
        reasons.append(f"n={n} carries no evidence (min_attainable_p=1.0)")
        p_floor = 1.0
    else:
        p_floor = min_attainable_p(n, n, paired=paired)
        if p_floor > alpha:
            reasons.append(
                f"min_attainable_p={p_floor:.6g} > alpha={alpha:g} at n={n} "
                f"({variant}): no result at this n can ever reject"
            )
    if minimum_effect <= 0:
        reasons.append(
            f"minimum_effect={minimum_effect!r} must be positive to power a design"
        )
        return DecidabilityReport(
            decidable=False,
            min_attainable_p=p_floor,
            min_detectable_effect=math.inf,
            required_n_for_effect=0,
            reasons=reasons,
        )
    if sd < 0:
        reasons.append(f"sd={sd!r} must be non-negative")
        return DecidabilityReport(
            decidable=False,
            min_attainable_p=p_floor,
            min_detectable_effect=math.inf,
            required_n_for_effect=0,
            reasons=reasons,
        )
    mde = min_detectable_effect(max(1, n), sd, alpha, paired=paired)
    required_n = max(
        required_n_for_effect(minimum_effect, sd, alpha, paired=paired),
        min_attainable_n(alpha, paired=paired),
    )
    if mde > minimum_effect:
        reasons.append(
            f"min_detectable_effect={mde:.6g} > minimum_effect={minimum_effect:g} "
            f"at n={n}, sd={sd:.6g}: required_n_for_effect={required_n}"
        )
    decidable = not reasons
    if decidable:
        reasons.append(
            f"decidable at n={n} ({variant}): min_attainable_p={p_floor:.6g} <= "
            f"alpha={alpha:g} and min_detectable_effect={mde:.6g} <= "
            f"minimum_effect={minimum_effect:g}"
        )
    return DecidabilityReport(
        decidable=decidable,
        min_attainable_p=p_floor,
        min_detectable_effect=mde,
        required_n_for_effect=required_n,
        reasons=reasons,
    )
