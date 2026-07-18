"""Bounded objective-geometry diagnostics for local decisions (LDI0-03).

A single cumulative wall-time budget (default and hard cap five minutes) is shared
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

DEFAULT_WALL_MINUTES = 5.0
MAX_WALL_MINUTES = 5.0
DiagnosticStatus = Literal["completed", "expired", "not_authorized"]

__all__ = [
    "DEFAULT_WALL_MINUTES",
    "MAX_WALL_MINUTES",
    "Deadline",
    "DiagnosticBudget",
    "not_authorized_report",
    "run_bounded_stages",
    "write_diagnostic_report",
]


@dataclass(frozen=True)
class DiagnosticBudget:
    """A cumulative wall-time budget, defaulted and hard-capped at five minutes."""

    max_wall_minutes: float = DEFAULT_WALL_MINUTES

    def __post_init__(self) -> None:
        if not 0.0 < float(self.max_wall_minutes) <= MAX_WALL_MINUTES:
            raise ValueError("max_wall_minutes must be in (0, 5]")

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
