"""Regression tests for CAP3-04 mixed-precision allocation."""

from __future__ import annotations

from slm_training.harnesses.quantization.allocation import allocate_mixed_precision
from slm_training.harnesses.quantization.sensitivity import (
    GroupFormatPoint,
    SensitivityReport,
)


def _report(points: list[GroupFormatPoint]) -> SensitivityReport:
    return SensitivityReport(
        version="cap3-04-v1",
        run_id="test-run",
        timestamp="2026-01-01T00:00:00Z",
        checkpoint_id="toy",
        calibration_manifest_sha="sha",
        grouping_policy_version="test-v1",
        formats=tuple(sorted({p.format_id for p in points})),
        sample_count=8,
        gradient_proxies={},
        points=points,
    )


def _point(group_id: str, format_id: str, total_bytes: int, mean_regret: float, cvar: float = 0.0) -> GroupFormatPoint:
    return GroupFormatPoint(
        group_id=group_id,
        format_id=format_id,
        group_size=8,
        packed_bytes=total_bytes,
        total_bytes=total_bytes,
        sample_count=8,
        top1_accuracy=0.0,
        teacher_top1_accuracy=0.0,
        action_flip_rate=0.0,
        kl_to_teacher=0.0,
        margin_preservation=0.0,
        mean_regret=mean_regret,
        cvar90_regret=cvar,
        status="ok",
    )


def test_knapsack_respects_byte_budget() -> None:
    points = [
        _point("g1", "fp16", 100, 0.0),
        _point("g1", "int4", 50, 0.1),
        _point("g2", "fp16", 100, 0.0),
        _point("g2", "int4", 50, 0.2),
    ]
    manifest = allocate_mixed_precision(_report(points), 120)
    assert manifest.solver_status == "optimal"
    assert manifest.total_bytes <= 120
    assert all(c.bytes <= 120 for c in manifest.allocation)


def test_knapsack_prefers_lower_cost() -> None:
    points = [
        _point("g1", "cheap", 50, 0.1),
        _point("g1", "dear", 50, 0.5),
        _point("g2", "cheap", 50, 0.2),
        _point("g2", "dear", 50, 0.1),
    ]
    manifest = allocate_mixed_precision(_report(points), 100)
    assert manifest.solver_status == "optimal"
    choices = {c.group_id: c.format_id for c in manifest.allocation}
    # g1 cheap regret 0.1 < dear 0.5; g2 dear regret 0.1 < cheap 0.2.
    assert choices["g1"] == "cheap"
    assert choices["g2"] == "dear"


def test_tail_max_filters_options() -> None:
    points = [
        _point("g1", "risky", 50, 0.05, cvar=0.9),
        _point("g1", "safe", 50, 0.1, cvar=0.1),
    ]
    manifest = allocate_mixed_precision(_report(points), 100, tail_max=0.5)
    assert manifest.solver_status == "optimal"
    choices = {c.group_id: c.format_id for c in manifest.allocation}
    assert choices["g1"] == "safe"


def test_uniform_baseline_finds_same_format() -> None:
    points = [
        _point("g1", "fmt", 50, 0.1),
        _point("g2", "fmt", 50, 0.2),
    ]
    manifest = allocate_mixed_precision(_report(points), 100)
    assert manifest.baselines["uniform"] is not None
    assert len(manifest.baselines["uniform"]) == 2
    assert all(c.format_id == "fmt" for c in manifest.baselines["uniform"])


def test_random_baseline_is_deterministic() -> None:
    points = [
        _point("g1", "a", 10, 0.1),
        _point("g1", "b", 20, 0.2),
        _point("g2", "a", 10, 0.1),
        _point("g2", "b", 20, 0.2),
    ]
    m1 = allocate_mixed_precision(_report(points), 100, random_seed=7)
    m2 = allocate_mixed_precision(_report(points), 100, random_seed=7)
    assert m1.baselines["random"] is not None
    assert m2.baselines["random"] is not None
    c1 = [(c.group_id, c.format_id) for c in m1.baselines["random"]]
    c2 = [(c.group_id, c.format_id) for c in m2.baselines["random"]]
    assert c1 == c2


def test_infeasible_budget_reports_status() -> None:
    points = [
        _point("g1", "fmt", 100, 0.1),
        _point("g2", "fmt", 100, 0.1),
    ]
    manifest = allocate_mixed_precision(_report(points), 50)
    assert manifest.solver_status == "infeasible"
    assert not manifest.allocation
