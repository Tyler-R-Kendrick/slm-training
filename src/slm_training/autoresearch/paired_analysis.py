"""Paired evidence eligibility and default-off sequential diagnostics.

Numerical fixed-sample primitives and thresholds remain in paired_stats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .evidence_ledger import parse_alpha


@dataclass(frozen=True)
class PairedSelection:
    """Locked ordered cases and declared independent root units, if available."""

    record_ids: Sequence[str]
    root_ids: Mapping[str, str] | None = None

    def __post_init__(self):
        if isinstance(self.record_ids, (str, bytes)):
            raise ValueError("selected cases must be a sequence of identities")
        cases = tuple(self.record_ids)
        if any(not isinstance(case, str) or not case for case in cases):
            raise ValueError("selected case identities must be nonempty strings")
        if len(set(cases)) != len(cases):
            raise ValueError("duplicate selected case identities")
        object.__setattr__(self, "record_ids", cases)
        if self.root_ids is not None:
            object.__setattr__(self, "root_ids", MappingProxyType(dict(self.root_ids)))


def paired_record_screening(
    control: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    direction: str,
    alpha: Fraction | str | int,
    min_nontied_pairs: int,
    kind: str,
    minimum_effect: float,
    selection: PairedSelection | None,
) -> dict[str, Any]:
    """Paired screening verdict on per-record maps.

    ``win`` requires ``p < alpha`` on the paired test **and** a median
    improvement strictly above ``minimum_effect``. Everything else — an
    undecidable pair count, ``p >= alpha``, a significant loss, or a
    significant but sub-threshold gain — is not a win. The result is a plain
    JSON-serialisable dict for delivery records.
    """

    from .paired_stats import _finite, paired_record_deltas, paired_screening_test

    required_ids = selection.record_ids if selection is not None else None
    root_ids = selection.root_ids if selection is not None else None
    effect = _finite(minimum_effect)
    if effect is None or effect < 0:
        raise ValueError("minimum_effect must be finite and nonnegative")
    pairs = paired_record_deltas(
        control, candidate, direction=direction, required_ids=required_ids
    )
    complete = pairs.n_pairs > 0 and not (
        pairs.n_missing_control or pairs.n_missing_candidate
    )
    roots = None
    if root_ids is not None:
        expected = (
            set(required_ids)
            if required_ids is not None
            else set(control) | set(candidate)
        )
        if set(root_ids) != expected or any(
            not isinstance(value, str) or not value for value in root_ids.values()
        ):
            raise ValueError("root identity coverage does not match selected cases")
        roots = set(root_ids.values())
    independent = roots is not None and len(roots) == len(root_ids)
    test = paired_screening_test(
        pairs.deltas,
        alpha=alpha,
        min_nontied_pairs=min_nontied_pairs,
        kind=kind,
    )
    median_delta = pairs.median_delta
    alpha_f = float(parse_alpha(alpha))
    win = bool(
        complete
        and independent
        and test.verdict == "win"
        and test.p_value < alpha_f
        and median_delta is not None
        and median_delta > float(minimum_effect)
    )
    return {
        "kind": test.kind,
        "direction": direction,
        "n_pairs": pairs.n_pairs,
        "n_nontied": test.n_nontied,
        "n_ties": test.n_ties,
        "n_missing_control": pairs.n_missing_control,
        "n_missing_candidate": pairs.n_missing_candidate,
        "statistic": test.statistic,
        "p_value": test.p_value,
        "alpha": test.alpha,
        "min_nontied_pairs": int(min_nontied_pairs),
        "minimum_effect": float(minimum_effect),
        "median_delta": median_delta,
        "mean_delta": pairs.mean_delta,
        "paired_sd": pairs.sd,
        "verdict": test.verdict if complete and independent else "inconclusive",
        "reason": (
            "measurement_incomplete:required_pairs_missing"
            if not complete
            else "design_infeasible:independent_units_not_declared"
            if roots is None
            else "design_infeasible:related_roots_require_locked_block_analysis"
            if not independent
            else test.reason
        ),
        "diagnostic_complete": complete,
        "independence_contract_met": independent,
        "independence": "declared_root_families"
        if roots is not None
        else "assumed_record_units_not_verified",
        "root_family_n": len(roots) if roots is not None else None,
        "selection_locked": required_ids is not None,
        "paired_record_ids": list(pairs.record_ids),
        "win": win,
        "promotion_authority": False,
    }


def bernoulli_mixture_interval(
    successes: int, trials: int, *, alpha
) -> tuple[float, float]:
    """Beta(1,1) likelihood-mixture confidence sequence for IID Bernoulli units.

    Invert B(s+1,n-s+1)/(p**s*(1-p)**(n-s)) < 1/alpha. Ville's
    inequality provides time-uniform coverage under the stated sampling law.
    This is not an interval for an unbounded mean NLL difference. See Howard
    et al. arXiv:1810.08240, conjugate-mixture construction; no new dependency.
    """
    if (
        type(successes) is not int
        or type(trials) is not int
        or not 0 <= successes <= trials
    ):
        raise ValueError("invalid Bernoulli counts")
    threshold = -math.log(float(parse_alpha(alpha)))
    if trials == 0:
        return 0.0, 1.0
    failures = trials - successes
    log_beta = (
        math.lgamma(successes + 1) + math.lgamma(failures + 1) - math.lgamma(trials + 2)
    )

    def log_mixture(p):
        if p == 0:
            return math.inf if successes else log_beta
        if p == 1:
            return math.inf if failures else log_beta
        return log_beta - successes * math.log(p) - failures * math.log1p(-p)

    center = successes / trials
    low, high = 0.0, center
    for _ in range(60):
        middle = (low + high) / 2
        if log_mixture(middle) > threshold:
            low = middle
        else:
            high = middle
    lower = low if successes else 0.0  # Outward rounding, never shrink coverage.
    low, high = center, 1.0
    for _ in range(60):
        middle = (low + high) / 2
        if log_mixture(middle) > threshold:
            high = middle
        else:
            low = middle
    return lower, high if failures else 1.0


def compare_fixed_sequential_signs(
    deltas: Sequence[float], *, unit_ids: Sequence[str], iid_units_declared: bool, alpha
) -> dict[str, Any]:
    """Default-off offline analysis comparator; never changes an existing verdict.

    The sign probability endpoint is explicit. IID/root independence is a
    caller's locked design obligation, not something a numeric vector proves.
    Trials sharing an ancestor yield only a conditional-on-ancestor claim.
    """
    from .paired_stats import _finite_deltas, paired_screening_test

    values = _finite_deltas(deltas)
    if (
        not values
        or len(values) != len(unit_ids)
        or len(set(unit_ids)) != len(unit_ids)
    ):
        raise ValueError("unique planned independent units required")
    if not iid_units_declared or any(
        not isinstance(value, str) or not value for value in unit_ids
    ):
        raise ValueError("IID sign sampling law must be explicitly declared")
    path = []
    successes = trials = 0
    for index, value in enumerate(values):
        if value:
            trials += 1
            successes += value > 0
        lower, upper = bernoulli_mixture_interval(successes, trials, alpha=alpha)
        path.append(
            {
                "observations": index + 1,
                "non_ties": trials,
                "lower": lower,
                "upper": upper,
                "verdict": "benefit"
                if lower > 0.5
                else "harm"
                if upper < 0.5
                else "inconclusive",
            }
        )
    fixed = paired_screening_test(values, kind="sign_test", alpha=alpha)
    return {
        "schema_version": "sequential_sign_comparison/v1",
        "default_enabled": False,
        "endpoint": "candidate_better_probability_conditional_on_non_tie",
        "units": "probability",
        "unit_ids": list(unit_ids),
        "iid_units_declared": True,
        "alpha": str(parse_alpha(alpha)),
        "fixed_horizon_verdict": fixed.verdict,
        "fixed_horizon_p": fixed.p_value,
        "sequential_path": path,
        "promotion_authority": False,
        "holdout_reuse_control": False,
        "mean_nll_inference": False,
    }
