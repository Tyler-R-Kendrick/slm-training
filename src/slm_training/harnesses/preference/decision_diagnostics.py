"""Bounded objective-geometry diagnostics for local decisions (LDI0-03).

A single cumulative wall-time budget (default and hard cap two minutes) is shared
across every diagnostic stage, mirroring ``CampaignBudget`` /
``autoresearch.engine.execute_commands``. On expiry the run reports ``expired`` and
produces **no result artifact** — the E285/E286 lesson that runtime expiry is a
stopped run, never a result. A future Tier-2 trainable-subspace pass refuses an
unauthorized full-parameter request (``not_authorized``) rather than replaying the
invalid E285 full-parameter profile.

This module writes no model, checkpoint, or training result. Elapsed time is read
through the module-level ``time`` reference so a deterministic fake clock can be
injected in tests (``monkeypatch.setattr(decision_diagnostics.time, "monotonic",
...)``).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from slm_training.levers import MAX_RUN_MINUTES

DEFAULT_WALL_MINUTES = float(MAX_RUN_MINUTES)
MAX_WALL_MINUTES = float(MAX_RUN_MINUTES)
DiagnosticStatus = Literal["completed", "expired", "not_authorized"]

__all__ = [
    "DEFAULT_WALL_MINUTES",
    "MAX_WALL_MINUTES",
    "Deadline",
    "DiagnosticBudget",
    "not_authorized_report",
    "run_bounded_stages",
    "tier1_objective_geometry",
    "tier2_subspace_gradients",
    "write_diagnostic_report",
]


@dataclass(frozen=True)
class DiagnosticBudget:
    """A cumulative wall-time budget, defaulted and hard-capped at two minutes."""

    max_wall_minutes: float = DEFAULT_WALL_MINUTES

    def __post_init__(self) -> None:
        if not 0.0 < float(self.max_wall_minutes) <= MAX_WALL_MINUTES:
            raise ValueError(f"max_wall_minutes must be in (0, {MAX_RUN_MINUTES}]")

    @property
    def seconds(self) -> float:
        return float(self.max_wall_minutes) * 60.0


class Deadline:
    """A monotonic cumulative deadline shared across diagnostic stages."""

    def __init__(self, budget: DiagnosticBudget) -> None:
        self._budget = budget
        self._start = time.monotonic()
        self._deadline = self._start + budget.seconds

    def remaining(self) -> float:
        return max(0.0, self._deadline - time.monotonic())

    def expired(self) -> bool:
        return time.monotonic() >= self._deadline

    def elapsed(self) -> float:
        return time.monotonic() - self._start


def run_bounded_stages(
    stages: Sequence[tuple[str, Callable[[], Any]]],
    *,
    budget: DiagnosticBudget | None = None,
) -> dict[str, Any]:
    """Run named stages under one cumulative deadline.

    Returns a report whose ``status`` is ``completed`` or ``expired``. On expiry
    ``result`` is ``None`` — no partial result is ever represented as a diagnostic
    result — and ``stage_telemetry`` records exactly which stages ran before the
    stop, so a rejected stage is visible rather than silently retained.
    """
    budget = budget or DiagnosticBudget()
    deadline = Deadline(budget)
    telemetry: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for name, stage in stages:
        if deadline.expired():
            return _report("expired", budget, deadline, telemetry, None)
        started = time.monotonic()
        output = stage()
        telemetry.append({"stage": name, "seconds": time.monotonic() - started})
        results[name] = output
    if deadline.expired():
        return _report("expired", budget, deadline, telemetry, None)
    return _report("completed", budget, deadline, telemetry, results)


def not_authorized_report(reason: str, *, budget: DiagnosticBudget | None = None) -> dict[str, Any]:
    """Report a refused stage (e.g. a full-parameter Tier-2 request) with no result."""
    budget = budget or DiagnosticBudget()
    return {
        "status": "not_authorized",
        "max_wall_minutes": budget.max_wall_minutes,
        "elapsed_wall_seconds": 0.0,
        "stage_telemetry": [],
        "reason": reason,
        "result": None,
    }


def _report(
    status: DiagnosticStatus,
    budget: DiagnosticBudget,
    deadline: Deadline,
    telemetry: list[dict[str, Any]],
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "status": status,
        "max_wall_minutes": budget.max_wall_minutes,
        "elapsed_wall_seconds": deadline.elapsed(),
        "stage_telemetry": telemetry,
        "result": result,
    }


def write_diagnostic_report(path: Path | str, report: dict[str, Any]) -> None:
    """Atomically write a diagnostic report (mkstemp + fsync + os.replace).

    An ``expired`` / ``not_authorized`` report carries ``result: None`` — the
    artifact records the stop without a diagnostic result.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def tier1_objective_geometry(
    per_state_views: Sequence[Sequence[Any]],
    *,
    budget: DiagnosticBudget | None = None,
) -> dict[str, Any]:
    """Read-only objective geometry over already-materialized objective views.

    ``per_state_views[i]`` is the sequence of ``ObjectiveView`` objects for one state
    (one view per materializer / objective). The pass runs inside the bounded runner,
    so it obeys the same cumulative deadline as every other diagnostic stage. It reads
    ``good_action_ids`` / ``bad_action_ids`` off each view (via ``getattr`` so it stays
    decoupled from ``decision_events_v2``) and reports, per corpus:

    - ``objective_contradictions`` — states where some action is scored *good* by one
      view and *bad* by another. A contradiction means the objective is not well-posed
      on that state; it is the logit-space shadow of the E284 objective conflict.
    - ``mean_good_set_overlap`` — mean pairwise Jaccard of the per-state good-action
      sets, a coarse measure of how much the views agree on what is preferred
      (``None`` when no state carries two or more views).

    This computes no gradient, trains nothing, writes no model, and makes no
    model-quality claim — it is geometry over inputs the caller already materialized.
    """

    def _analyze() -> dict[str, Any]:
        contradictions = 0
        overlaps: list[float] = []
        for views in per_state_views:
            good_sets = [set(getattr(view, "good_action_ids", ())) for view in views]
            bad_sets = [set(getattr(view, "bad_action_ids", ())) for view in views]
            good_union: set[int] = set().union(*good_sets) if good_sets else set()
            bad_union: set[int] = set().union(*bad_sets) if bad_sets else set()
            if good_union & bad_union:
                contradictions += 1
            for i in range(len(good_sets)):
                for j in range(i + 1, len(good_sets)):
                    left, right = good_sets[i], good_sets[j]
                    union = left | right
                    overlaps.append(len(left & right) / len(union) if union else 1.0)
        return {
            "states": len(per_state_views),
            "objective_contradictions": contradictions,
            "mean_good_set_overlap": (
                sum(overlaps) / len(overlaps) if overlaps else None
            ),
        }

    return run_bounded_stages([("objective_geometry", _analyze)], budget=budget)


def tier2_subspace_gradients(
    *,
    trainable_parameter_subset: Sequence[str] | None,
    budget: DiagnosticBudget | None = None,
) -> dict[str, Any]:
    """Tier-2 adapter-subspace gradient interface (refuses full-parameter).

    A Tier-2 gradient pass is authorized only over an explicit trainable-parameter
    subset (named adapter tensors). An empty or ``None`` subset is a full-parameter
    request, refused as ``not_authorized`` rather than replaying the invalid E285
    full-parameter profile that blew the runtime envelope. When a subset is supplied
    the runner records a bounded *plan* only — the gradient computation is deferred to
    a model stage this module never runs, so no gradient is computed and no model is
    written here.
    """
    subset = tuple(trainable_parameter_subset or ())
    if not subset:
        return not_authorized_report(
            "full-parameter Tier-2 is not authorized; provide an explicit adapter subset",
            budget=budget,
        )

    def _plan() -> dict[str, Any]:
        return {
            "parameter_subset_size": len(subset),
            "parameter_subset": list(subset),
            "gradients": "deferred_to_model_stage",
        }

    return run_bounded_stages([("subspace_plan", _plan)], budget=budget)
