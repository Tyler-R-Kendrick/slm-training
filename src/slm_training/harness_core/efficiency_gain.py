"""Efficiency-gain metrics EG_x (P1c)."""

from __future__ import annotations

import math
from typing import Sequence

from slm_training.harness_core.scaling_fit import (
    CostKey,
    ScalingObservation,
    invert_loss,
    observation_cost,
)


def efficiency_gain(
    baseline_fit: dict[str, float],
    candidate: ScalingObservation,
    *,
    cost_key: CostKey = "time",
) -> float | None:
    """EG_x = f_x^{-1}(L_candidate) / C_candidate.

    ``cost_key="params"`` yields the size-normalized gain: EG_params <= 1 means
    the candidate reached its loss by spending capacity the baseline curve
    would have spent less of — a scaling purchase, not a capability gain.
    """
    c = observation_cost(candidate, cost_key)
    if c is None or c <= 0 or not math.isfinite(candidate.loss):
        return None
    baseline_cost = invert_loss(baseline_fit, candidate.loss)
    if not math.isfinite(baseline_cost) or baseline_cost <= 0:
        return None
    return float(baseline_cost) / float(c)


# Two-sided Student-t critical values by (confidence, degrees of freedom).
# EG_params/EG_time gates run on seed counts in the low single digits, where
# the normal approximation is severely anticonservative (e.g. 1.96 vs the
# true 12.7 at df=1). Values are the standard textbook table; no scipy
# dependency is added for it. beyond df=30 the table's largest entry is
# reused, which is *wider* than the asymptotic z — a conservative bound.
_STUDENT_T_TABLE: dict[float, dict[int, float]] = {
    0.90: {
        1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895,
        8: 1.860, 9: 1.833, 10: 1.812, 11: 1.796, 12: 1.782, 13: 1.771,
        14: 1.761, 15: 1.753, 16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729,
        20: 1.725, 21: 1.721, 22: 1.717, 23: 1.714, 24: 1.711, 25: 1.708,
        26: 1.706, 27: 1.703, 28: 1.701, 29: 1.699, 30: 1.697,
    },
    0.95: {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    },
    0.99: {
        1: 63.657, 2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032, 6: 3.707, 7: 3.499,
        8: 3.355, 9: 3.250, 10: 3.169, 11: 3.106, 12: 3.055, 13: 3.012,
        14: 2.977, 15: 2.947, 16: 2.921, 17: 2.898, 18: 2.878, 19: 2.861,
        20: 2.845, 21: 2.831, 22: 2.819, 23: 2.807, 24: 2.797, 25: 2.787,
        26: 2.779, 27: 2.771, 28: 2.763, 29: 2.756, 30: 2.750,
    },
}


def _student_t_critical(df: int, confidence: float) -> float:
    """Two-sided critical value for ``df`` degrees of freedom.

    ``confidence`` is bucketed down to the largest tabulated level not
    exceeding it (floor 0.90 — no caller in this codebase requests a wider
    interval than that, and floor-bucketing keeps the result conservative
    rather than silently interpolating a laxer one).
    """
    if confidence >= 0.99:
        table = _STUDENT_T_TABLE[0.99]
    elif confidence >= 0.95:
        table = _STUDENT_T_TABLE[0.95]
    else:
        table = _STUDENT_T_TABLE[0.90]
    bounded_df = max(1, min(df, max(table)))
    return table[bounded_df]


def efficiency_gain_lcb(
    gains: Sequence[float],
    *,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Mean and exact Student-t LCB/UCB over seed-level EG values.

    Requires at least two finite observations: a single seed carries no
    estimate of dispersion, so treating it as a zero-width interval would let
    a bare point estimate satisfy any ``lcb >= threshold`` promotion gate
    without evidence of repeatability. At n=1 the bounds are ``nan`` (callers
    already gate on ``math.isfinite(lcb)`` alongside the threshold compare,
    so this fails closed automatically).
    """
    vals = [float(g) for g in gains if g is not None and math.isfinite(float(g))]
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return (mean, float("nan"), float("nan"))
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    std = math.sqrt(var)
    se = std / math.sqrt(len(vals))
    t = _student_t_critical(len(vals) - 1, confidence)
    return (mean, mean - t * se, mean + t * se)
