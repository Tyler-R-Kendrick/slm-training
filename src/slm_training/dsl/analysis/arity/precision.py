"""Residual-precision reference functions and scale-mode guards (CAP0-03)."""
from __future__ import annotations

import math
from enum import Enum


class ResidualScaleMode(str, Enum):
    """How a residual scale was obtained; determines which claims are valid."""

    GEOMETRIC_BALANCED = "geometric_balanced"
    LEARNED_INDEPENDENT = "learned_independent"
    OTHER = "other"


def minimum_margin_trit_planes(error_radius: float, margin: float) -> int:
    """Minimum R such that error_radius/(3**R - 1) < margin/2.

    Uses the strict division-free integer predicate:
        2 * error_radius < margin * (3**R - 1)

    Regression: error_radius=4, margin=1 -> R=3 (R=2 is equality).
    """
    if margin <= 0:
        raise ValueError("margin must be positive")
    if error_radius < 0:
        raise ValueError("error_radius must be non-negative")
    if error_radius == 0:
        return 0
    lhs = 2 * error_radius
    R = 1
    while True:
        rhs = margin * (3 ** R - 1)
        if lhs < rhs:
            return R
        R += 1


def ternary_ecoc_width(actions: int, *, detect_single_trit_error: bool) -> int:
    """Ternary ECOC label width for b legal actions.

    * No guaranteed detection: ceil(log_3 b) trits.
    * Single-trit detection: ceil(log_3 b) + 1 trits (adds one parity/check trit).
    """
    if actions <= 0:
        raise ValueError("actions must be positive")
    base = max(0, math.ceil(math.log(actions, 3)))
    if actions == 1:
        base = 0
    return base + (1 if detect_single_trit_error else 0)


def balanced_ternary_levels(R: int) -> int:
    """Number of distinct consecutive integer states for R balanced-ternary digits.

    Only valid under GEOMETRIC_BALANCED mode.
    """
    if R < 0:
        raise ValueError("R must be non-negative")
    return 3 ** R


def assert_geometric_only(mode: ResidualScaleMode) -> None:
    """Guard: raise if a caller tries to apply analytic grid claims to learned scales."""
    if mode != ResidualScaleMode.GEOMETRIC_BALANCED:
        raise ValueError(
            f"analytic 3^R grid claim is invalid for scale mode {mode.value}"
        )
