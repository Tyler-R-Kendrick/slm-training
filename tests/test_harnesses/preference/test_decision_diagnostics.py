"""Tests for bounded local-decision diagnostics (LDI0-03).

Exercises the cumulative wall-time deadline with a deterministic fake clock: an
expired run produces no result artifact, and a full-parameter request is refused
as not_authorized.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from slm_training.harnesses.preference import decision_diagnostics as dd
from slm_training.harnesses.preference.decision_diagnostics import (
    DiagnosticBudget,
    not_authorized_report,
    run_bounded_stages,
    tier1_objective_geometry,
    tier2_subspace_gradients,
    write_diagnostic_report,
)


def _view(good: tuple[int, ...], bad: tuple[int, ...]) -> SimpleNamespace:
    """A getattr-compatible stand-in for ObjectiveView (good/bad action ids)."""
    return SimpleNamespace(good_action_ids=tuple(good), bad_action_ids=tuple(bad))


class _FakeClock:
    """Deterministic monotonic clock: pops queued readings, then holds the last."""

    def __init__(self, values: list[float]) -> None:
        self._values = list(values)
        self._last = self._values[-1] if self._values else 0.0

    def __call__(self) -> float:
        if self._values:
            self._last = self._values.pop(0)
        return self._last


def test_budget_capped_at_three_minutes() -> None:
    assert DiagnosticBudget().max_wall_minutes == 3.0
    with pytest.raises(ValueError, match=r"\(0, 3\]"):
        DiagnosticBudget(max_wall_minutes=3.1)
    with pytest.raises(ValueError):
        DiagnosticBudget(max_wall_minutes=0.0)


def test_completed_run_reports_all_stages(monkeypatch) -> None:
    monkeypatch.setattr(dd.time, "monotonic", _FakeClock([0.0]))
    report = run_bounded_stages([("a", lambda: {"x": 1}), ("b", lambda: {"y": 2})])
    assert report["status"] == "completed"
    assert report["result"] == {"a": {"x": 1}, "b": {"y": 2}}
    assert [stage["stage"] for stage in report["stage_telemetry"]] == ["a", "b"]


def test_deadline_expiry_produces_no_result(monkeypatch) -> None:
    # start=0 -> deadline=180s; the next monotonic read is 500 -> expired before any stage.
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


def test_tier1_objective_geometry_flags_contradiction(monkeypatch) -> None:
    monkeypatch.setattr(dd.time, "monotonic", _FakeClock([0.0]))
    # One state, two views: action 4 is good in the first view but bad in the second.
    per_state = [[_view((4,), (9,)), _view((5,), (4,))]]
    report = tier1_objective_geometry(per_state)
    assert report["status"] == "completed"
    geometry = report["result"]["objective_geometry"]
    assert geometry["states"] == 1
    assert geometry["objective_contradictions"] == 1
    assert geometry["mean_good_set_overlap"] == 0.0


def test_tier1_objective_geometry_agrees_when_views_match(monkeypatch) -> None:
    monkeypatch.setattr(dd.time, "monotonic", _FakeClock([0.0]))
    per_state = [[_view((4,), (9,)), _view((4,), (9,))]]
    geometry = tier1_objective_geometry(per_state)["result"]["objective_geometry"]
    assert geometry["objective_contradictions"] == 0
    assert geometry["mean_good_set_overlap"] == 1.0


def test_tier1_objective_geometry_respects_deadline(monkeypatch) -> None:
    # start=0 -> deadline=180s; the next read is 500 -> expired before the stage runs.
    monkeypatch.setattr(dd.time, "monotonic", _FakeClock([0.0, 500.0]))
    report = tier1_objective_geometry([[_view((4,), (9,))]])
    assert report["status"] == "expired"
    assert report["result"] is None


def test_tier2_refuses_full_parameter_request() -> None:
    for subset in (None, []):
        report = tier2_subspace_gradients(trainable_parameter_subset=subset)
        assert report["status"] == "not_authorized"
        assert report["result"] is None
        assert "adapter subset" in report["reason"]


def test_tier2_plans_over_explicit_subset(monkeypatch) -> None:
    monkeypatch.setattr(dd.time, "monotonic", _FakeClock([0.0]))
    report = tier2_subspace_gradients(
        trainable_parameter_subset=["adapter.0.weight", "adapter.0.bias"]
    )
    assert report["status"] == "completed"
    plan = report["result"]["subspace_plan"]
    assert plan["parameter_subset_size"] == 2
    assert plan["gradients"] == "deferred_to_model_stage"
