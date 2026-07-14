"""Efficiency-gain metrics EG_x (P1c)."""

from __future__ import annotations

import math
from typing import Sequence

from slm_training.experiments.scaling_fit import (
    CostKey,
    ScalingObservation,
    invert_loss,
)


def efficiency_gain(
    baseline_fit: dict[str, float],
    candidate: ScalingObservation,
    *,
    cost_key: CostKey = "time",
) -> float | None:
    """EG_x = f_x^{-1}(L_candidate) / C_candidate."""
    if cost_key == "time":
        c = candidate.cost_time_s
    elif cost_key == "flops":
        c = candidate.cost_flops
    else:
        c = float(candidate.cost_nfe) if candidate.cost_nfe is not None else None
    if c is None or c <= 0 or not math.isfinite(candidate.loss):
        return None
    baseline_cost = invert_loss(baseline_fit, candidate.loss)
    if not math.isfinite(baseline_cost) or baseline_cost <= 0:
        return None
    return float(baseline_cost) / float(c)


def efficiency_gain_lcb(
    gains: Sequence[float],
    *,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Mean and normal-approx LCB/UCB over seed-level EG values."""
    vals = [float(g) for g in gains if g is not None and math.isfinite(float(g))]
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return (mean, mean, mean)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    std = math.sqrt(var)
    # z for common confidence levels; default ~1.96 for 95%.
    z = 1.96 if confidence >= 0.95 else 1.64 if confidence >= 0.9 else 1.0
    se = std / math.sqrt(len(vals))
    return (mean, mean - z * se, mean + z * se)
