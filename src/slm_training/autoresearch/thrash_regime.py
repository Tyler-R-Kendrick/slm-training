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
    "TIMEOUT_CAUSE_BUDGET",
    "TIMEOUT_CAUSE_SLOW_DECODE",
    "TIMEOUT_CAUSE_NONE",
    "DECODE_RESIDUAL_SLUGS",
    "LATENCY_PRIMARY_LEAF",
    "DECODE_COST_LEVER_CATEGORIES",
    "DECODE_COST_MODEL_LEVERS",
    "TRAINING_LOSS_LEVER_SUFFIX",
    "TRAINING_DURATION_LEVERS",
    "LATENCY_HYPOTHESIS_SLUGS",
    "STEPS_FACTOR_KEY",
    "CLIMB_BASELINE_STATUSES",
    "ThrashRegimeDecision",
    "is_compiler_ms_timeout_signal",
    "classify_timeout_cause",
    "select_climb_baseline_entry",
    "decide_screening_regime",
    "lever_overlay",
    "compose_isolate_control_levers",
    "compose_climb_control_levers",
    "compose_treatment_levers",
    "select_decode_residual_slug",
    "select_recommended_slug_for_regime",
    "is_latency_only_arm",
]

REGIME_ISOLATE = "isolate"
REGIME_CLIMB = "climb"
REGIME_TIMEOUT_DECODE_RESIDUAL = "timeout_decode_residual"

# Why the predecessor timed out. A ``budget_timeout`` means the measured
# per-record work (p95) exceeded the per-record timeout the recipe could afford
# under the arm wall: the budget was infeasible, so the fix is recalibrating
# n_probe / timeout from measured cost (Pareto rule), never a decode-cost
# residual arm (measured effect of those arms under budget timeouts: 0.0).
# ``slow_decode_timeout`` means the budget was feasible (p95 <= timeout) and a
# tail of records still exceeded it: a genuine decode-cost residual candidate.
TIMEOUT_CAUSE_BUDGET = "budget_timeout"
TIMEOUT_CAUSE_SLOW_DECODE = "slow_decode_timeout"
TIMEOUT_CAUSE_NONE = "none"

# Registered thrash arms whose extras are decode-cost / completeness levers.
# Production thrash never routes timeouts to unconstrained / compiler_decode_mode=off.
DECODE_RESIDUAL_SLUGS: tuple[str, ...] = (
    "bounds",
    "canvas",
    "both",
    "cached-compiler-decision-margin",
)

# Latency-only arm bank gate (RC7, docs/design/autotrain-recovery-2-p9-20260902.md).
# Arms whose every knob is a decode/run *cost* lever cannot change trained
# weights, so they are drawn only when the screening role primary leaf is the
# latency metric. Under a quality/NLL primary they are a guaranteed null
# (evidence_ledger.v1: ``bounds`` n_complete=50, mean_delta=0.0,
# m2_delta=0.0; ``canvas`` n_complete=5, all null).
LATENCY_PRIMARY_LEAF = "latency_ms_p50"
# ``lever_catalog()`` categories that never touch the training objective.
DECODE_COST_LEVER_CATEGORIES: frozenset[str] = frozenset({"decode", "run"})
# Catalog-``model`` levers that only reshape decode work (no weight effect).
DECODE_COST_MODEL_LEVERS: frozenset[str] = frozenset({"compact_active_canvas"})
# ``lever_catalog()`` labels ``compiler_*`` levers "decode" by prefix, but a
# ``*_loss_weight`` enters the training objective (twotower.py alignment and
# decision-token losses). The suffix outranks the prefix for cost classification.
TRAINING_LOSS_LEVER_SUFFIX = "_loss_weight"
# ``steps`` is catalog-``run`` but changes weights (more optimizer updates), so
# it is a training-duration lever, never a pure cost lever.
TRAINING_DURATION_LEVERS: frozenset[str] = frozenset({"steps"})
# Training-lever arms whose preregistered hypothesis is a latency / cost claim;
# they ride with the latency bank (``batch1`` ledger: n_complete=11,
# n_positive=0, mean_delta 0.015; ``steps`` x2 is a depth-confound cost
# control, never a quality lever).
LATENCY_HYPOTHESIS_SLUGS: tuple[str, ...] = ("batch1", "steps")
# Private bank key materialized as ``steps``.
STEPS_FACTOR_KEY = "_steps_factor"


def is_latency_only_arm(
    extras: Mapping[str, Any] | None,
    *,
    lever_categories: Mapping[str, str],
) -> bool:
    """True when every public knob of ``extras`` is a decode/run cost lever.

    ``lever_categories`` maps lever name -> ``lever_catalog()`` category.
    ``_steps_factor`` counts as ``steps``. A ``*_loss_weight`` lever is always
    a training lever even when its catalog category says ``decode``, and
    ``steps`` is a training-duration lever even though its category is ``run``.
    An arm with no public knobs is not latency-only (a no-op, not a cost arm).
    """

    public: list[str] = []
    for key in (extras or {}):
        name = str(key)
        if name == STEPS_FACTOR_KEY:
            public.append("steps")
        elif not name.startswith("_"):
            public.append(name)
    if not public:
        return False
    for name in public:
        if name.endswith(TRAINING_LOSS_LEVER_SUFFIX) or name in TRAINING_DURATION_LEVERS:
            return False
        if name in DECODE_COST_MODEL_LEVERS:
            continue
        if lever_categories.get(name) not in DECODE_COST_LEVER_CATEGORIES:
            return False
    return True


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
        "eval_limit",
        "eval_partial_scoreboard",
        "eval_max_records_this_run",
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


def classify_timeout_cause(
    *,
    p95_seconds: float | None,
    timeout_seconds: float | None,
    timeout_count: int | float | None = None,
) -> str:
    """Classify a predecessor's decode timeouts as budget-bound or tail-slow.

    ``budget_timeout`` when the measured per-record p95 exceeds the timeout
    that was applied (the recipe could not fit the work: recalibrate budget).
    ``slow_decode_timeout`` when p95 fits the timeout yet records timed out
    (feasible budget, slow tail: decode residual arms stay eligible).
    ``none`` when there were no timeouts or the cause is undecidable because
    the measured p95 or the applied timeout is unknown.
    """

    try:
        count = float(timeout_count) if timeout_count is not None else None
    except (TypeError, ValueError):
        count = None
    if count is not None and count <= 0:
        return TIMEOUT_CAUSE_NONE
    if p95_seconds is None or timeout_seconds is None:
        return TIMEOUT_CAUSE_NONE
    try:
        p95 = float(p95_seconds)
        timeout = float(timeout_seconds)
    except (TypeError, ValueError):
        return TIMEOUT_CAUSE_NONE
    if p95 <= 0 or timeout <= 0:
        return TIMEOUT_CAUSE_NONE
    if count is None and p95 <= timeout:
        # No timeout evidence at all: a fitting p95 is simply "no timeout".
        return TIMEOUT_CAUSE_NONE
    return TIMEOUT_CAUSE_BUDGET if p95 > timeout else TIMEOUT_CAUSE_SLOW_DECODE


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
    climb_champion_available: bool = False,
    timeout_cause: str | None = None,
) -> ThrashRegimeDecision:
    """Decide isolate vs climb vs timeout residual routing for screening thrash.

    ``timeout_cause`` (see :func:`classify_timeout_cause`) gates the residual
    route: a ``budget_timeout`` never routes into ``DECODE_RESIDUAL_SLUGS`` —
    the budget feedback loop recalibrates n_probe / timeout instead.
    """

    has_climb = bool(climb_baseline_knobs) or bool(climb_champion_available)
    climb_payload = (
        dict(climb_baseline_knobs)
        if isinstance(climb_baseline_knobs, Mapping)
        else ({} if climb_champion_available else None)
    )
    base = REGIME_CLIMB if has_climb else REGIME_ISOLATE
    budget_bound = str(timeout_cause or "") == TIMEOUT_CAUSE_BUDGET
    if compiler_ms_timeout and budget_bound:
        return ThrashRegimeDecision(
            regime=base,
            base_regime=base,
            climb_baseline=climb_payload if has_climb else None,
            timeout_residual=False,
            reason=(
                "prior_incomplete_budget_timeout;"
                f"baseline={base};recalibrate_budget_not_residual"
            ),
        )
    if compiler_ms_timeout:
        return ThrashRegimeDecision(
            regime=REGIME_TIMEOUT_DECODE_RESIDUAL,
            base_regime=base,
            climb_baseline=climb_payload if has_climb else None,
            timeout_residual=True,
            reason=(
                "prior_incomplete_compiler_ms_timeout;"
                f"baseline={base};prefer_decode_residual"
            ),
        )
    if has_climb:
        reason = (
            "sticky_champion_baseline_available"
            if climb_baseline_knobs
            else "climb_champion_checkpoint_available"
        )
        return ThrashRegimeDecision(
            regime=REGIME_CLIMB,
            base_regime=REGIME_CLIMB,
            climb_baseline=climb_payload,
            timeout_residual=False,
            reason=reason,
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
