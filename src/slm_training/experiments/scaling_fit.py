"""Power-law scaling fits L(C) = A·C^(-α) + E (P1c)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence


@dataclass(frozen=True)
class ScalingObservation:
    track: str
    candidate_id: str
    point_id: str
    seed: int
    loss: float
    cost_time_s: float
    cost_flops: float | None = None
    cost_nfe: int | None = None
    trainable_params: int | None = None


CostKey = Literal["flops", "time", "nfe"]


def _cost(obs: ScalingObservation, cost_key: CostKey) -> float | None:
    if cost_key == "time":
        return float(obs.cost_time_s) if obs.cost_time_s > 0 else None
    if cost_key == "flops":
        return float(obs.cost_flops) if obs.cost_flops and obs.cost_flops > 0 else None
    if cost_key == "nfe":
        return float(obs.cost_nfe) if obs.cost_nfe and obs.cost_nfe > 0 else None
    return None


def fit_power_law(
    observations: Sequence[ScalingObservation],
    *,
    cost_key: CostKey = "time",
    irreducible: float | None = None,
) -> dict[str, float]:
    """Fit ``L(C) = A·C^(-α) + E`` via grid search on α + closed-form A,E.

    Pure Python — no scipy/numpy dependency.
    """
    points: list[tuple[float, float]] = []
    for obs in observations:
        c = _cost(obs, cost_key)
        if c is None or obs.loss is None or not math.isfinite(obs.loss):
            continue
        points.append((c, float(obs.loss)))
    if len(points) < 2:
        raise ValueError("need ≥2 observations to fit a scaling law")

    # Fix E as min(loss)*0.5 when not provided (irreducible floor).
    e0 = (
        float(irreducible)
        if irreducible is not None
        else max(0.0, min(y for _, y in points) * 0.5)
    )
    best: dict[str, float] | None = None
    for alpha_i in range(1, 81):
        alpha = alpha_i / 20.0  # 0.05 .. 4.0
        # L - E = A · C^{-α}  →  log(L-E) = log A - α log C
        xs: list[float] = []
        ys: list[float] = []
        for c, loss in points:
            resid = loss - e0
            if resid <= 1e-12:
                continue
            xs.append(math.log(c))
            ys.append(math.log(resid))
        if len(xs) < 2:
            continue
        # Force slope = -α: log A = mean(ys + α xs)
        log_a = sum(y + alpha * x for x, y in zip(xs, ys)) / len(xs)
        a = math.exp(log_a)
        sse = 0.0
        sst = 0.0
        mean_y = sum(loss for _, loss in points) / len(points)
        for c, loss in points:
            pred = a * (c ** (-alpha)) + e0
            sse += (loss - pred) ** 2
            sst += (loss - mean_y) ** 2
        r2 = 1.0 - (sse / sst if sst > 0 else 0.0)
        cand = {"A": a, "alpha": alpha, "E": e0, "r2": r2, "n": float(len(points))}
        if best is None or cand["r2"] > best["r2"]:
            best = cand
    if best is None:
        raise ValueError("scaling fit failed")
    return best


def predict_loss(fit: dict[str, float], cost: float) -> float:
    return float(fit["A"]) * (float(cost) ** (-float(fit["alpha"]))) + float(fit["E"])


def invert_loss(fit: dict[str, float], loss: float) -> float:
    """Solve for C given L: C = ((L - E) / A)^(-1/α)."""
    a, alpha, e = float(fit["A"]), float(fit["alpha"]), float(fit["E"])
    resid = float(loss) - e
    if resid <= 0 or a <= 0:
        return float("inf")
    return (resid / a) ** (-1.0 / alpha)


def observation_from_summary(
    summary: dict[str, Any],
    *,
    candidate_id: str,
    point_id: str,
    seed: int,
    loss: float | None = None,
    cost_nfe: int | None = None,
    cost_flops: float | None = None,
) -> ScalingObservation:
    track = dict(summary.get("track") or {})
    tel = dict(summary.get("telemetry") or {})
    cost_time_s = float(tel.get("total_ms") or 0.0) / 1000.0
    weighted = loss
    if weighted is None:
        weighted = summary.get("best_weighted_nll")
    if weighted is None:
        final = summary.get("final_loss_eval") or {}
        weighted = (final.get("weighted_nll") if isinstance(final, dict) else None)
    return ScalingObservation(
        track=str(track.get("context_backend") or "unknown"),
        candidate_id=candidate_id,
        point_id=point_id,
        seed=seed,
        loss=float(weighted if weighted is not None else math.inf),
        cost_time_s=cost_time_s,
        cost_flops=cost_flops,
        cost_nfe=cost_nfe,
        trainable_params=track.get("trainable_params"),
    )
