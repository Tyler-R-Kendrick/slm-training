"""Generic promotion-protocol engine (DSL-agnostic).

The frozen promotion checks extracted verbatim from
``slm_training.harnesses.experiments.promotion``, which remains the policy
owner: the hard-category tuple, the DSL-touching data-integrity scan, and the
ship-gate binding. The engine takes those specifics as parameters
(``hard_categories``, ``gate_evaluator``) instead of importing any metric- or
DSL-specific policy.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from slm_training.harness_core.efficiency_gain import efficiency_gain_lcb

GateEvaluator = Callable[
    [dict[str, dict[str, Any]], dict[str, dict[str, float]] | None],
    dict[str, Any],
]


@dataclass(frozen=True)
class PromotionCriteria:
    category_regression_tolerance: float = 0.02
    require_rank_stable_top2: bool = True
    eg_time_lcb_min: float = 1.0
    # Capacity growth must pay for itself on the same terms as wall time
    # (AGENTS.md VI). Wall time alone is not a size budget: a wider model can
    # hold its latency and still buy its loss with parameters.
    eg_params_lcb_min: float = 1.0
    ship_gate_policy: dict[str, dict[str, float]] | None = None


def _category_mean(report: dict[str, Any], name: str) -> float | None:
    cat = (report.get("categories") or {}).get(name) or {}
    mean = (cat.get("aggregate") or {}).get("mean_nll")
    return float(mean) if mean is not None else None


def check_category_regression(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    categories: Sequence[str],
    tolerance: float = 0.02,
) -> dict[str, Any]:
    """No hard category may regress more than ``tolerance`` relatively."""
    details: dict[str, Any] = {}
    ok = True
    for name in categories:
        base = _category_mean(baseline, name)
        cand = _category_mean(candidate, name)
        if base is None or cand is None:
            details[name] = {"pass": False, "reason": "missing"}
            ok = False
            continue
        # Higher NLL is worse; allow cand <= base * (1 + tol).
        limit = base * (1.0 + tolerance) if base > 0 else base + tolerance
        passed = cand <= limit
        details[name] = {
            "pass": passed,
            "baseline": base,
            "candidate": cand,
            "limit": limit,
        }
        ok = ok and passed
    return {"pass": ok, "categories": details}


def _point_order_key(point_id: str) -> tuple[Any, ...]:
    """Order ladder point ids by their embedded numbers, not lexically.

    Point ids look like ``d192_h6_...``; a plain string sort ranks
    ``"d96" > "d64" > "d192"``, which silently picks the wrong two rungs.
    """
    numbers = tuple(int(n) for n in re.findall(r"\d+", point_id))
    return (numbers, point_id) if numbers else ((), point_id)


def select_smallest_sufficient(
    candidates: Sequence[Mapping[str, Any]],
    *,
    params_of: Callable[[Mapping[str, Any]], int | None],
    loss_of: Callable[[Mapping[str, Any]], float | None],
    tolerance: float = 0.02,
) -> Mapping[str, Any] | None:
    """Pick the smallest candidate whose loss reaches the frontier.

    Goal invariant VI: the deliverable is the smartest output at the smallest
    model. Selecting by loss alone across a ladder that spans widths promotes
    the widest rung by construction, because loss falls with capacity. This
    takes the best achieved loss, admits everything within ``tolerance`` of
    it, and returns the smallest admitted model — so capacity is spent only
    where it actually buys something outside the noise band.
    """
    scored: list[tuple[Mapping[str, Any], float]] = []
    for cand in candidates:
        loss = loss_of(cand)
        if loss is None or not math.isfinite(float(loss)):
            continue
        scored.append((cand, float(loss)))
    if not scored:
        return None
    best_loss = min(loss for _, loss in scored)
    limit = (
        best_loss * (1.0 + tolerance) if best_loss > 0 else best_loss + tolerance
    )
    eligible = [(c, loss) for c, loss in scored if loss <= limit]
    # Unknown parameter counts sort last: an unmeasured model never displaces a
    # measured smaller one.
    def key(item: tuple[Mapping[str, Any], float]) -> tuple[int, int, float]:
        cand, loss = item
        params = params_of(cand)
        return (0, int(params), loss) if params else (1, 0, loss)

    return min(eligible, key=key)[0]


def check_rank_stability(
    rankings: dict[str, list[str]],
    *,
    top_k: int = 1,
) -> dict[str, Any]:
    """Top candidate must be stable across the largest two ladder points."""
    if not rankings:
        return {"pass": False, "reason": "empty_rankings"}
    keys = sorted(rankings.keys(), key=_point_order_key, reverse=True)[:2]
    if len(keys) < 2:
        return {"pass": False, "reason": "need_two_ladder_points", "keys": keys}
    tops = [tuple((rankings[k] or [])[:top_k]) for k in keys]
    stable = len(set(tops)) == 1 and all(tops)
    return {
        "pass": stable,
        "keys": keys,
        "tops": {k: list(t) for k, t in zip(keys, tops)},
    }


def check_parameter_efficiency(
    *,
    baseline_params: int | None,
    candidate_params: int | None,
    eg_params_by_seed: Sequence[float] | None,
    eg_params_lcb_min: float = 1.0,
) -> dict[str, Any]:
    """A candidate that grew must show the growth paid for itself.

    Holding or shrinking capacity always passes — this check never penalizes a
    smaller model. It engages only on growth, and then fails closed: growing
    without a measured size-normalized gain is an unproven purchase, not a
    capability result, and is exactly the pattern that lets a scaled-up model
    be promoted on quality it bought rather than earned.
    """
    detail: dict[str, Any] = {
        "baseline_params": baseline_params,
        "candidate_params": candidate_params,
        "min": eg_params_lcb_min,
    }
    if baseline_params is None or candidate_params is None:
        if eg_params_by_seed is None:
            return {**detail, "pass": False, "reason": "parameter_counts_unmeasured"}
        grew = True
    else:
        grew = int(candidate_params) > int(baseline_params)
        detail["growth_ratio"] = (
            float(candidate_params) / float(baseline_params)
            if baseline_params
            else None
        )
    if not grew:
        return {**detail, "pass": True, "reason": "capacity_not_increased"}
    if eg_params_by_seed is None:
        return {
            **detail,
            "pass": False,
            "reason": "capacity_increased_without_parameter_efficiency_evidence",
        }
    mean, lcb, ucb = efficiency_gain_lcb(eg_params_by_seed)
    return {
        **detail,
        "pass": bool(lcb >= eg_params_lcb_min and math.isfinite(lcb)),
        "mean": mean,
        "lcb": lcb,
        "ucb": ucb,
    }


def evaluate_promotion(
    *,
    integrity: dict[str, Any] | None = None,
    baseline_loss_report: dict[str, Any] | None = None,
    candidate_loss_report: dict[str, Any] | None = None,
    rankings: dict[str, list[str]] | None = None,
    eg_time_by_seed: Sequence[float] | None = None,
    eg_params_by_seed: Sequence[float] | None = None,
    baseline_trainable_params: int | None = None,
    candidate_trainable_params: int | None = None,
    ship_suites: dict[str, dict[str, Any]] | None = None,
    criteria: PromotionCriteria | None = None,
    hard_categories: Sequence[str],
    gate_evaluator: GateEvaluator,
) -> dict[str, Any]:
    """Return ``{promotable, checks, failures}`` mirroring ship-gates shape."""
    crit = criteria or PromotionCriteria()
    checks: dict[str, Any] = {}
    failures: list[str] = []

    if integrity is not None:
        checks["integrity"] = integrity
        if not integrity.get("pass"):
            failures.append("integrity")

    if baseline_loss_report is not None and candidate_loss_report is not None:
        cat = check_category_regression(
            baseline_loss_report,
            candidate_loss_report,
            categories=hard_categories,
            tolerance=crit.category_regression_tolerance,
        )
        checks["category_regression"] = cat
        if not cat["pass"]:
            failures.append("category_regression")
        base_w = (baseline_loss_report.get("aggregate") or {}).get("weighted_nll")
        cand_w = (candidate_loss_report.get("aggregate") or {}).get("weighted_nll")
        nll_improved = (
            base_w is not None
            and cand_w is not None
            and float(cand_w) < float(base_w)
        )
        checks["weighted_nll_improved"] = {
            "pass": nll_improved,
            "baseline": base_w,
            "candidate": cand_w,
        }
        if not nll_improved:
            failures.append("weighted_nll_improved")

    if rankings is not None and crit.require_rank_stable_top2:
        stab = check_rank_stability(rankings)
        checks["rank_stability"] = stab
        if not stab["pass"]:
            failures.append("rank_stability")

    if eg_time_by_seed is not None:
        mean, lcb, ucb = efficiency_gain_lcb(eg_time_by_seed)
        eg_ok = lcb >= crit.eg_time_lcb_min and math.isfinite(lcb)
        checks["eg_time"] = {
            "pass": eg_ok,
            "mean": mean,
            "lcb": lcb,
            "ucb": ucb,
            "min": crit.eg_time_lcb_min,
        }
        if not eg_ok:
            failures.append("eg_time")

    if (
        baseline_trainable_params is not None
        and candidate_trainable_params is not None
    ) or eg_params_by_seed is not None:
        par = check_parameter_efficiency(
            baseline_params=baseline_trainable_params,
            candidate_params=candidate_trainable_params,
            eg_params_by_seed=eg_params_by_seed,
            eg_params_lcb_min=crit.eg_params_lcb_min,
        )
        checks["eg_params"] = par
        if not par["pass"]:
            failures.append("eg_params")

    if ship_suites is not None:
        gates = gate_evaluator(ship_suites, crit.ship_gate_policy)
        checks["ship_gates"] = gates
        if not gates.get("pass"):
            failures.append("ship_gates")

    comparative_checks = {
        "weighted_nll_improved",
        "rank_stability",
        "eg_time",
        "eg_params",
        "ship_gates",
    }
    if not comparative_checks & checks.keys():
        checks["sufficient_evidence"] = {
            "pass": False,
            "reason": "promotion requires quality, rank, efficiency, or ship evidence",
        }
        failures.append("sufficient_evidence")

    return {
        "promotable": not failures,
        "checks": checks,
        "failures": failures,
    }
