"""Tests for bounded local-decision diagnostics (LDI0-03).

Exercises the cumulative wall-time deadline with a deterministic fake clock: an
expired run produces no result artifact, and a full-parameter request is refused
as not_authorized.
"""

from __future__ import annotations

import json

import pytest

from slm_training.harnesses.preference import decision_diagnostics as dd
from slm_training.harnesses.preference.decision_diagnostics import (
    DiagnosticBudget,
    not_authorized_report,
    run_bounded_stages,
    write_diagnostic_report,
)


class _FakeClock:
    """Deterministic monotonic clock: pops queued readings, then holds the last."""

    def __init__(self, values: list[float]) -> None:
        self._values = list(values)
        self._last = self._values[-1] if self._values else 0.0

    def __call__(self) -> float:
        if self._values:
            self._last = self._values.pop(0)
        return self._last


def test_budget_capped_at_five_minutes() -> None:
    assert DiagnosticBudget().max_wall_minutes == 5.0
    with pytest.raises(ValueError, match=r"\(0, 5\]"):
        DiagnosticBudget(max_wall_minutes=6.0)
    with pytest.raises(ValueError):
        DiagnosticBudget(max_wall_minutes=0.0)


def test_completed_run_reports_all_stages(monkeypatch) -> None:
    monkeypatch.setattr(dd.time, "monotonic", _FakeClock([0.0]))
    report = run_bounded_stages([("a", lambda: {"x": 1}), ("b", lambda: {"y": 2})])
    assert report["status"] == "completed"
    assert report["result"] == {"a": {"x": 1}, "b": {"y": 2}}
    assert [stage["stage"] for stage in report["stage_telemetry"]] == ["a", "b"]


def test_deadline_expiry_produces_no_result(monkeypatch) -> None:
    # start=0 -> deadline=300s; the next monotonic read is 500 -> expired before any stage.
    monkeypatch.setattr(dd.time, "monotonic", _FakeClock([0.0, 500.0]))
    ran: list[str] = []
    report = run_bounded_stages(
        [("a", lambda: ran.append("a")), ("b", lambda: ran.append("b"))]
    )
    assert report["status"] == "expired"
    assert report["result"] is None
    assert ran == []


def test_not_authorized_report_has_no_result() -> None:
    report = not_authorized_report("full-parameter Tier-2 is not authorized")
    assert report["status"] == "not_authorized"
    assert report["result"] is None
    assert "not authorized" in report["reason"]


def test_report_write_is_atomic(tmp_path) -> None:
    path = tmp_path / "diag.json"
    write_diagnostic_report(path, {"status": "expired", "result": None})
    assert json.loads(path.read_text())["result"] is None
