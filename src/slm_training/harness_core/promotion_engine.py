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
from dataclasses import dataclass
from typing import Any, Callable, Sequence

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


def check_rank_stability(
    rankings: dict[str, list[str]],
    *,
    top_k: int = 1,
) -> dict[str, Any]:
    """Top candidate must be stable across the largest two ladder points."""
    if not rankings:
        return {"pass": False, "reason": "empty_rankings"}
    # Prefer keys that look like the largest points (sorted descending).
    keys = sorted(rankings.keys(), reverse=True)[:2]
    if len(keys) < 2:
        return {"pass": False, "reason": "need_two_ladder_points", "keys": keys}
    tops = [tuple((rankings[k] or [])[:top_k]) for k in keys]
    stable = len(set(tops)) == 1 and all(tops)
    return {
        "pass": stable,
        "keys": keys,
        "tops": {k: list(t) for k, t in zip(keys, tops)},
    }


def evaluate_promotion(
    *,
    integrity: dict[str, Any] | None = None,
    baseline_loss_report: dict[str, Any] | None = None,
    candidate_loss_report: dict[str, Any] | None = None,
    rankings: dict[str, list[str]] | None = None,
    eg_time_by_seed: Sequence[float] | None = None,
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

    if ship_suites is not None:
        gates = gate_evaluator(ship_suites, crit.ship_gate_policy)
        checks["ship_gates"] = gates
        if not gates.get("pass"):
            failures.append("ship_gates")

    comparative_checks = {
        "weighted_nll_improved",
        "rank_stability",
        "eg_time",
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
