"""Cross-farm cost projection helpers."""

from __future__ import annotations

import json

from gpu_multi_farm.models import FarmCostEstimate, FarmListResult, Offer


def cheapest_offer(results: dict[str, FarmListResult], gpu_type: str | None = None) -> dict[str, Offer | None]:
    out: dict[str, Offer | None] = {}
    for farm, result in results.items():
        offers = result.offers
        if gpu_type:
            needle = gpu_type.lower()
            offers = [o for o in offers if needle in o.gpu_type.lower()]
        out[farm] = min(offers, key=lambda o: o.price_per_hr) if offers else None
    return out


def project_costs(
    results: dict[str, FarmListResult],
    hours: int,
    gpu_type: str,
    model_size_gb: float,
    overhead: float,
) -> dict:
    """Project training cost per farm and recommend the cheapest available."""
    from pathlib import Path

    _ = model_size_gb  # reserved for future VRAM/disk sizing heuristics
    bench = Path("outputs/cactus/bench.json")
    if bench.exists():
        try:
            data = json.loads(bench.read_text(encoding="utf-8"))
            overhead = float(data.get("overhead") or overhead)
        except Exception:  # noqa: BLE001
            pass
    estimates: list[FarmCostEstimate] = []
    per_farm: dict[str, dict] = {}

    for farm, result in results.items():
        if result.error and not result.offers:
            est = FarmCostEstimate(
                farm=farm,
                price_per_hr=None,
                hours=hours,
                compute_cost=None,
                overhead_multiplier=overhead,
                total_cost=None,
                gpu_type=gpu_type,
                available=False,
                note=result.error,
            )
            estimates.append(est)
            per_farm[farm] = est.to_dict()
            continue

        offers = [
            o
            for o in result.offers
            if gpu_type.lower() in o.gpu_type.lower()
        ] or list(result.offers)
        if not offers:
            est = FarmCostEstimate(
                farm=farm,
                price_per_hr=None,
                hours=hours,
                compute_cost=None,
                overhead_multiplier=overhead,
                total_cost=None,
                gpu_type=gpu_type,
                available=False,
                note="no matching offers",
            )
            estimates.append(est)
            per_farm[farm] = est.to_dict()
            continue

        best = min(offers, key=lambda o: o.price_per_hr)
        compute = best.price_per_hr * hours
        total = compute * overhead
        est = FarmCostEstimate(
            farm=farm,
            price_per_hr=best.price_per_hr,
            hours=hours,
            compute_cost=round(compute, 4),
            overhead_multiplier=overhead,
            total_cost=round(total, 4),
            gpu_type=best.gpu_type,
            available=True,
            note=f"offer_id={best.offer_id}; spot={best.spot}",
        )
        estimates.append(est)
        per_farm[farm] = est.to_dict()

    available = [e for e in estimates if e.available and e.total_cost is not None]
    recommended = None
    reason = None
    if available:
        best_est = min(available, key=lambda e: e.total_cost or float("inf"))
        recommended = best_est.farm
        savings = None
        if len(available) > 1:
            worst = max(available, key=lambda e: e.total_cost or 0.0)
            if worst.total_cost and best_est.total_cost and worst.total_cost > 0:
                savings = round(100.0 * (1.0 - best_est.total_cost / worst.total_cost), 1)
        reason = (
            f"{best_est.farm} (${best_est.total_cost} projected incl. cactus overhead "
            f"{overhead}x"
            + (f"; ~{savings}% vs highest" if savings is not None else "")
            + ")"
        )

    return {
        "hours": hours,
        "gpu_type": gpu_type,
        "model_size_gb": model_size_gb,
        "cactus_overhead": overhead,
        "farms": per_farm,
        "recommended": recommended,
        "recommended_reason": reason,
    }
