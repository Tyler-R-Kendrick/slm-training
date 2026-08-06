"""Dual-regime thrash: isolate (causal OFAT) vs climb (residual on sticky recipe).

Pure functions only — drivers pass synthetic or live champion/timeout inputs.
Does not train, eval, or weaken constrained-decode policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "REGIME_ISOLATE",
    "REGIME_CLIMB",
    "REGIME_TIMEOUT_DECODE_RESIDUAL",
    "DECODE_RESIDUAL_SLUGS",
    "CLIMB_BASELINE_STATUSES",
    "ThrashRegimeDecision",
    "is_compiler_ms_timeout_signal",
    "select_climb_baseline_entry",
    "decide_screening_regime",
    "lever_overlay",
    "compose_isolate_control_levers",
    "compose_climb_control_levers",
    "compose_treatment_levers",
    "select_decode_residual_slug",
    "select_recommended_slug_for_regime",
]

REGIME_ISOLATE = "isolate"
REGIME_CLIMB = "climb"
REGIME_TIMEOUT_DECODE_RESIDUAL = "timeout_decode_residual"

# Registered thrash arms whose extras are decode-cost / completeness levers.
# Production thrash never routes timeouts to unconstrained / compiler_decode_mode=off.
DECODE_RESIDUAL_SLUGS: tuple[str, ...] = (
    "bounds",
    "canvas",
    "both",
    "cached-compiler-decision-margin",
)

# Queue statuses that may supply a sticky climb baseline recipe.
CLIMB_BASELINE_STATUSES: frozenset[str] = frozenset(
    {
        "confirmed",
        "climb_accepted",
        "promoted",
    }
)

# Measurement / thrash-budget keys never inherited into climb lever identity.
# Screening re-samples seed, steps, decode timeout, and generate_batch_size.
_NON_LEVER_KEYS: frozenset[str] = frozenset(
    {
        "seed",
        "steps",
        "decode_timeout_seconds",
        "eval_suites",
        "generate_batch_size",
    }
)


@dataclass(frozen=True)
class ThrashRegimeDecision:
    """Which thrash regime applies for the next screening cycle."""

    regime: str
    base_regime: str
    """Underlying science/product baseline: isolate or climb (never timeout alone)."""
    climb_baseline: Mapping[str, Any] | None
    timeout_residual: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "base_regime": self.base_regime,
            "timeout_residual": self.timeout_residual,
            "reason": self.reason,
            "has_climb_baseline": self.climb_baseline is not None,
            "climb_baseline_keys": sorted(self.climb_baseline or ()),
        }


def is_compiler_ms_timeout_signal(
    *,
    reasons: Sequence[str] | None = None,
    decode_outcome_detail: str | None = None,
    incomplete: bool = False,
    decode_timeout_count: int | float | None = None,
) -> bool:
    """True when prior work is incomplete and compiler_ms dominated decode timeout."""

    detail = str(decode_outcome_detail or "")
    if "timeout_dominant_phase=compiler_ms" in detail:
        return True
    if "compiler_ms" in detail and "timeout_dominant" in detail:
        return True
    reason_blob = " ".join(str(r) for r in (reasons or ()))
    if "timeout_dominant_phase=compiler_ms" in reason_blob:
        return True
    if incomplete and decode_timeout_count is not None:
        try:
            if float(decode_timeout_count) > 0 and (
                "compiler_ms" in reason_blob or "decode_timeout" in reason_blob
            ):
                # Prefer explicit dominant-phase when present; incomplete+timeout
                # alone is not enough without a compiler_ms marker.
                if "compiler_ms" in reason_blob:
                    return True
        except (TypeError, ValueError):
            pass
    return False


def select_climb_baseline_entry(
    queue_entries: Sequence[Mapping[str, Any]] | None,
) -> Mapping[str, Any] | None:
    """Most recent queue row with sticky champion knobs for climb control."""

    if not queue_entries:
        return None
    # Prefer product-accepted, then confirmed waiting for promote.
    for status in ("climb_accepted", "promoted", "confirmed"):
        for row in reversed(list(queue_entries)):
            if str(row.get("status") or "") != status:
                continue
            knobs = row.get("knobs")
            if isinstance(knobs, dict) and knobs:
                return row
    return None


def decide_screening_regime(
    *,
    climb_baseline_knobs: Mapping[str, Any] | None,
    compiler_ms_timeout: bool,
) -> ThrashRegimeDecision:
    """Decide isolate vs climb vs timeout residual routing for screening thrash."""

    has_climb = bool(climb_baseline_knobs)
    base = REGIME_CLIMB if has_climb else REGIME_ISOLATE
    if compiler_ms_timeout:
        return ThrashRegimeDecision(
            regime=REGIME_TIMEOUT_DECODE_RESIDUAL,
            base_regime=base,
            climb_baseline=dict(climb_baseline_knobs) if has_climb else None,
            timeout_residual=True,
            reason=(
                "prior_incomplete_compiler_ms_timeout;"
                f"baseline={base};prefer_decode_residual"
            ),
        )
    if has_climb:
        return ThrashRegimeDecision(
            regime=REGIME_CLIMB,
            base_regime=REGIME_CLIMB,
            climb_baseline=dict(climb_baseline_knobs),
            timeout_residual=False,
            reason="sticky_champion_baseline_available",
        )
    return ThrashRegimeDecision(
        regime=REGIME_ISOLATE,
        base_regime=REGIME_ISOLATE,
        climb_baseline=None,
        timeout_residual=False,
        reason="no_climb_baseline_causal_ofat",
    )


def lever_overlay(
    baseline: Mapping[str, Any] | None,
    residual: Mapping[str, Any] | None,
    *,
    strip_measurement_from_residual: bool = False,
) -> dict[str, Any]:
    """Merge residual onto baseline; residual wins on key collision.

    Measurement keys (seed/steps/timeouts) are stripped from the baseline so
    screening can re-sample thrash budgets. Residuals may still carry
    intentional step/batch changes (e.g. thrash ``steps`` arm) unless
    ``strip_measurement_from_residual`` is set.
    """

    out: dict[str, Any] = {}
    if baseline:
        out.update({k: v for k, v in baseline.items() if k not in _NON_LEVER_KEYS})
    if residual:
        if strip_measurement_from_residual:
            out.update(
                {k: v for k, v in residual.items() if k not in _NON_LEVER_KEYS}
            )
        else:
            out.update(dict(residual))
    return out


def compose_isolate_control_levers(
    *,
    precursor_extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Isolate control: empty or precursor package only (zeroed otherwise)."""

    return lever_overlay(None, precursor_extras)


def compose_climb_control_levers(
    climb_baseline: Mapping[str, Any],
) -> dict[str, Any]:
    """Climb control: full sticky champion lever set."""

    return lever_overlay(climb_baseline, None)


def compose_treatment_levers(
    *,
    control_levers: Mapping[str, Any],
    residual_extras: Mapping[str, Any],
) -> dict[str, Any]:
    """Treatment = control baseline + residual thrash extras only."""

    return lever_overlay(control_levers, residual_extras)


def select_decode_residual_slug(
    *,
    skip: Iterable[str] | None = None,
    bank_slugs: Sequence[str] | None = None,
    residual_slugs: Sequence[str] = DECODE_RESIDUAL_SLUGS,
) -> str | None:
    """First open decode residual slug, or None if all closed."""

    closed = set(skip or ())
    allowed = set(bank_slugs) if bank_slugs is not None else None
    for slug in residual_slugs:
        if slug in closed:
            continue
        if allowed is not None and slug not in allowed:
            continue
        return slug
    return None


def select_recommended_slug_for_regime(
    *,
    decision: ThrashRegimeDecision,
    cycle: int,
    skip: Iterable[str] | None,
    bank_slugs: Sequence[str],
    isolate_selector: Any,
) -> str:
    """Pick recommended arm: decode residual under timeout, else isolate rotation.

    ``isolate_selector`` is ``(cycle, skip) -> slug`` (e.g. bank rotation).
    """

    skip_set = set(skip or ())
    if decision.timeout_residual:
        residual = select_decode_residual_slug(
            skip=skip_set,
            bank_slugs=bank_slugs,
        )
        if residual is not None:
            return residual
        # Fall through if residual bank exhausted — still isolate rotation.
    return str(isolate_selector(cycle, skip_set))
