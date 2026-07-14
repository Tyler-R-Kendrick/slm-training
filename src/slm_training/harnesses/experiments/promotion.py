"""Promotion-protocol evaluation (P1c)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from slm_training.harnesses.experiments.efficiency_gain import efficiency_gain_lcb
from slm_training.harnesses.model_build.ship_gates import (
    DEFAULT_SHIP_GATES,
    evaluate_ship_gates,
)

HARD_CATEGORIES = ("binding", "structural", "repair")


@dataclass(frozen=True)
class PromotionCriteria:
    category_regression_tolerance: float = 0.02
    require_rank_stable_top2: bool = True
    eg_time_lcb_min: float = 1.0
    ship_gate_policy: dict[str, dict[str, float]] | None = None


def check_data_integrity(
    train_dir: Path | str,
    test_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Lightweight integrity: train manifest exists + optional leakage scan."""
    from slm_training.data.leakage import find_leakage, load_train_fingerprints
    from slm_training.dsl.schema import load_jsonl

    train_dir = Path(train_dir)
    manifest = train_dir / "manifest.json"
    records_path = train_dir / "records.jsonl"
    failures: list[str] = []
    if not records_path.exists():
        failures.append("missing_train_records")
    if not manifest.exists():
        failures.append("missing_train_manifest")
    leakage_hits = 0
    if test_dir is not None and manifest.exists():
        fps = load_train_fingerprints(manifest)
        suites_root = Path(test_dir) / "suites"
        if suites_root.exists():
            for suite_dir in sorted(suites_root.iterdir()):
                rec_path = suite_dir / "records.jsonl"
                if not rec_path.exists():
                    continue
                for record in load_jsonl(rec_path):
                    hits = find_leakage(record, fps)
                    leakage_hits += len(hits)
        if leakage_hits:
            failures.append(f"leakage_hits:{leakage_hits}")
    return {
        "pass": not failures,
        "failures": failures,
        "leakage_hits": leakage_hits,
    }


def _category_mean(report: dict[str, Any], name: str) -> float | None:
    cat = (report.get("categories") or {}).get(name) or {}
    mean = (cat.get("aggregate") or {}).get("mean_nll")
    return float(mean) if mean is not None else None


def check_category_regression(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    tolerance: float = 0.02,
) -> dict[str, Any]:
    """No hard category may regress more than ``tolerance`` relatively."""
    details: dict[str, Any] = {}
    ok = True
    for name in HARD_CATEGORIES:
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
        gates = evaluate_ship_gates(
            ship_suites,
            thresholds=crit.ship_gate_policy or DEFAULT_SHIP_GATES,
        )
        checks["ship_gates"] = gates
        if not gates.get("pass"):
            failures.append("ship_gates")

    return {
        "promotable": not failures,
        "checks": checks,
        "failures": failures,
    }


def register_promoted_checkpoint(
    checkpoint_dir: Path | str,
    *,
    source: Path | str | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Copy/link the mid-trained anchor to ``promoted.pt`` (P1d)."""
    import shutil

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dest = checkpoint_dir / "promoted.pt"
    if source is not None:
        source = Path(source)
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)
    meta_path = checkpoint_dir / "promoted.json"
    payload = {"kind": "promoted_anchor", **(meta or {})}
    meta_path.write_text(
        __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return dest
