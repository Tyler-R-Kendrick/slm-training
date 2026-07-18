"""CAP3-04: mixed-precision allocation solver.

Deterministic multiple-choice knapsack allocation over group/format sensitivity
points, plus uniform/random/hand-hybrid baselines.  Latency numbers are modeled,
not measured, until valid per-format kernel data exists.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from slm_training.harnesses.quantization.sensitivity import (
    GroupFormatPoint,
    SensitivityReport,
)


CAP3_04_ALLOC_VERSION = "cap3-04-alloc-v1"


@dataclass
class AllocationChoice:
    """One selected group/format in an allocation."""

    group_id: str
    format_id: str
    group_size: int
    bytes: int
    cost: float
    mean_regret: float
    cvar90_regret: float
    kl_to_teacher: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "format_id": self.format_id,
            "group_size": self.group_size,
            "bytes": self.bytes,
            "cost": self.cost,
            "mean_regret": self.mean_regret,
            "cvar90_regret": self.cvar90_regret,
            "kl_to_teacher": self.kl_to_teacher,
        }


@dataclass
class AllocationManifest:
    """Versioned mixed-precision allocation result."""

    version: str
    run_id: str
    timestamp: str
    sensitivity_run_id: str
    budget_bytes: int
    tail_max: float | None
    objective: str
    allocation: list[AllocationChoice]
    baselines: dict[str, list[AllocationChoice] | None]
    solver_status: str
    total_bytes: int
    total_cost: float
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        def baseline_to_dict(b: list[AllocationChoice] | None) -> dict[str, Any] | None:
            if b is None:
                return None
            return {
                "choices": [c.as_dict() for c in b],
                "total_bytes": sum(c.bytes for c in b),
                "total_cost": sum(c.cost for c in b),
            }

        return {
            "version": self.version,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "sensitivity_run_id": self.sensitivity_run_id,
            "budget_bytes": self.budget_bytes,
            "tail_max": self.tail_max,
            "objective": self.objective,
            "allocation": [c.as_dict() for c in self.allocation],
            "baselines": {k: baseline_to_dict(v) for k, v in self.baselines.items()},
            "solver_status": self.solver_status,
            "total_bytes": self.total_bytes,
            "total_cost": self.total_cost,
            "notes": self.notes,
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, default=str)


def _point_cost(point: GroupFormatPoint, objective: str) -> float:
    if objective == "kl_to_teacher":
        return point.kl_to_teacher
    if objective == "weighted":
        return 0.5 * point.mean_regret + 0.5 * point.kl_to_teacher
    return point.mean_regret


def _group_options(
    report: SensitivityReport,
    objective: str,
    tail_max: float | None,
) -> dict[str, list[GroupFormatPoint]]:
    """Group points by group_id, filtering excluded/infeasible options."""
    options: dict[str, list[GroupFormatPoint]] = {}
    for point in report.points:
        if point.status != "ok":
            continue
        if tail_max is not None and point.cvar90_regret > tail_max:
            continue
        options.setdefault(point.group_id, []).append(point)
    return options


def _choices_from_assignment(
    assignment: dict[str, GroupFormatPoint],
    objective: str,
) -> list[AllocationChoice]:
    return [
        AllocationChoice(
            group_id=gid,
            format_id=p.format_id,
            group_size=p.group_size,
            bytes=p.total_bytes,
            cost=_point_cost(p, objective),
            mean_regret=p.mean_regret,
            cvar90_regret=p.cvar90_regret,
            kl_to_teacher=p.kl_to_teacher,
        )
        for gid, p in sorted(assignment.items())
    ]


def _uniform_baseline(
    options: dict[str, list[GroupFormatPoint]],
    budget_bytes: int,
    objective: str,
) -> list[AllocationChoice] | None:
    """Assign the same feasible format to every group with options."""
    best: list[AllocationChoice] | None = None
    best_cost = float("inf")
    candidates: set[str] = set()
    for opts in options.values():
        for p in opts:
            candidates.add(p.format_id)
    for fmt in candidates:
        assignment: dict[str, GroupFormatPoint] = {}
        for gid, opts in options.items():
            match = next((p for p in opts if p.format_id == fmt), None)
            if match is None:
                break
            assignment[gid] = match
        if len(assignment) != len(options):
            continue
        total_bytes = sum(p.total_bytes for p in assignment.values())
        if total_bytes > budget_bytes:
            continue
        total_cost = sum(_point_cost(p, objective) for p in assignment.values())
        if total_cost < best_cost:
            best_cost = total_cost
            best = _choices_from_assignment(assignment, objective)
    return best


def _random_baseline(
    options: dict[str, list[GroupFormatPoint]],
    budget_bytes: int,
    objective: str,
    seed: int = 0,
) -> list[AllocationChoice] | None:
    """One deterministic random feasible assignment."""
    rng = random.Random(seed)
    group_ids = sorted(options)
    assignment: dict[str, GroupFormatPoint] = {}
    remaining = budget_bytes
    for gid in group_ids:
        feasible = [p for p in options[gid] if p.total_bytes <= remaining]
        if not feasible:
            return None
        choice = rng.choice(feasible)
        assignment[gid] = choice
        remaining -= choice.total_bytes
    return _choices_from_assignment(assignment, objective)


def _hand_hybrid_baseline(
    options: dict[str, list[GroupFormatPoint]],
    budget_bytes: int,
    objective: str,
    overrides: dict[str, str] | None = None,
) -> list[AllocationChoice] | None:
    """Default hand-designed hybrid: backbone ternary, local head int8."""
    overrides = overrides or {
        "local_head/scorer": "int8",
        "local_head/embeddings": "ternary",
    }
    assignment: dict[str, GroupFormatPoint] = {}
    total_bytes = 0
    for gid in sorted(options):
        fmt_id = overrides.get(gid, "ternary")
        match = next((p for p in options[gid] if p.format_id == fmt_id), None)
        if match is None:
            return None
        total_bytes += match.total_bytes
        if total_bytes > budget_bytes:
            return None
        assignment[gid] = match
    return _choices_from_assignment(assignment, objective)


def allocate_mixed_precision(
    report: SensitivityReport,
    budget_bytes: int,
    *,
    objective: str = "mean_regret",
    tail_max: float | None = None,
    hand_overrides: dict[str, str] | None = None,
    random_seed: int = 0,
    run_id: str | None = None,
) -> AllocationManifest:
    """Solve the mixed-precision allocation and compare with baselines."""
    if run_id is None:
        run_id = f"cap3-04-alloc-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    options = _group_options(report, objective, tail_max)
    group_ids = sorted(options)
    notes: list[str] = []

    if not group_ids:
        return AllocationManifest(
            version=CAP3_04_ALLOC_VERSION,
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sensitivity_run_id=report.run_id,
            budget_bytes=budget_bytes,
            tail_max=tail_max,
            objective=objective,
            allocation=[],
            baselines={},
            solver_status="infeasible",
            total_bytes=0,
            total_cost=0.0,
            notes=["no feasible group/format options after tail/budget filters"],
        )

    # Multiple-choice knapsack DP over reachable byte totals.
    # dp[i] maps total bytes after processing first i groups to (cost, chosen_point).
    dp: list[dict[int, tuple[float, GroupFormatPoint | None]]] = [
        {0: (0.0, None)}
    ]
    for gid in group_ids:
        opts = options[gid]
        prev = dp[-1]
        next_dp: dict[int, tuple[float, GroupFormatPoint | None]] = {}
        for prev_bytes, (prev_cost, _) in prev.items():
            for point in opts:
                new_bytes = prev_bytes + point.total_bytes
                if new_bytes > budget_bytes:
                    continue
                new_cost = prev_cost + _point_cost(point, objective)
                if new_bytes not in next_dp or new_cost < next_dp[new_bytes][0]:
                    next_dp[new_bytes] = (new_cost, point)
        dp.append(next_dp)
        if not next_dp:
            notes.append(f"no reachable state after group {gid!r}")
            break

    allocation_choices: list[AllocationChoice] = []
    solver_status = "optimal"
    total_bytes = 0
    total_cost = 0.0

    final_dp = dp[-1]
    if final_dp:
        best_bytes, (best_cost, _) = min(final_dp.items(), key=lambda x: x[1][0])
        # Backtrack through layers.
        assignment: dict[str, GroupFormatPoint] = {}
        cur_bytes = best_bytes
        for i in range(len(group_ids), 0, -1):
            point = dp[i][cur_bytes][1]
            if point is None:
                break
            gid = group_ids[i - 1]
            assignment[gid] = point
            cur_bytes -= point.total_bytes
        total_bytes = best_bytes
        total_cost = best_cost
        allocation_choices = _choices_from_assignment(assignment, objective)
    else:
        solver_status = "infeasible"
        notes.append("knapsack DP found no feasible allocation under budget")

    baselines: dict[str, list[AllocationChoice] | None] = {
        "uniform": _uniform_baseline(options, budget_bytes, objective),
        "random": _random_baseline(options, budget_bytes, objective, seed=random_seed),
        "hand_hybrid": _hand_hybrid_baseline(options, budget_bytes, objective, hand_overrides),
    }

    if not allocation_choices:
        solver_status = "infeasible"

    return AllocationManifest(
        version=CAP3_04_ALLOC_VERSION,
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        sensitivity_run_id=report.run_id,
        budget_bytes=budget_bytes,
        tail_max=tail_max,
        objective=objective,
        allocation=allocation_choices,
        baselines=baselines,
        solver_status=solver_status,
        total_bytes=total_bytes,
        total_cost=total_cost,
        notes=notes,
    )
