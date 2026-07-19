"""CAP4-02 fixture: run adaptive residual-plane schedules on synthetic traces.

Generates random held-out legal-action traces, routes them through a small
``ResidualTritPlaneHead`` under eight schedules, and reports plane-count
statistics and agreement with the full ``uniform_max`` baseline.  This is a
wiring fixture only; no ship gate, checkpoint, or latency claim is made.
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from slm_training.models.local_action_head import ResidualTritPlaneHead, StateContext
from slm_training.models.quantization.adaptive_planes import (
    AdaptivePlaneRoutingContext,
    PlaneRouter,
    PlaneScheduler,
    ScheduleMode,
    make_schedule_spec,
)
from slm_training.versioning import build_version_stamp


SCHEDULES: list[ScheduleMode] = [
    "uniform_1",
    "uniform_max",
    "structural_floor",
    "floor_plus_entropy",
    "floor_plus_margin",
    "floor_plus_sensitivity",
    "floor_plus_learned_router",
    "oracle_min_planes",
]


def _make_traces(
    n: int,
    hidden_dim: int,
    max_actions: int,
    *,
    seed: int = 11,
) -> tuple[torch.Tensor, list[StateContext], list[list[str]]]:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    hidden = torch.randn(n, hidden_dim)
    action_pool = [f"action_{i:03d}" for i in range(max_actions)]
    contexts: list[StateContext] = []
    legals: list[list[str]] = []
    for _ in range(n):
        branch_count = rng.randint(1, 10)
        legal = rng.sample(action_pool, branch_count)
        forced = branch_count == 1
        sensitivity = {"slot": rng.random(), "template": rng.random()} if not forced else None
        contexts.append(
            StateContext(
                state_family_id="fixture",
                branch_count=branch_count,
                forced=forced,
                sensitivity=sensitivity,
                completion_support_size=rng.randint(1, branch_count) if not forced else None,
            )
        )
        legals.append(legal)
    return hidden, contexts, legals


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def _run_schedule(
    head: ResidualTritPlaneHead,
    hidden: torch.Tensor,
    contexts: list[StateContext],
    legals: list[list[str]],
    schedule_id: ScheduleMode,
    baseline_actions: list[str | None] | None,
) -> dict[str, Any]:
    spec = make_schedule_spec(
        schedule_id,
        max_planes=head.R,
        grouping_policy="compact" if schedule_id == "floor_plus_learned_router" else "whole_batch",
    )
    router: PlaneRouter | None = None
    if schedule_id == "floor_plus_learned_router":
        router = PlaneRouter()
    scheduler = PlaneScheduler(spec, router=router)
    ctx = AdaptivePlaneRoutingContext(
        head,
        scheduler,
        grouping_policy=spec.grouping_policy,
        stability_patience=0,
    )

    results = ctx.route_batch(hidden, contexts, legals)
    planes = [r.planes_used for r in results]
    actions = [r.action_identity for r in results]

    metrics: dict[str, Any] = {
        "n": len(results),
        "avg_planes": sum(planes) / len(planes) if planes else 0.0,
        "p50_planes": _percentile(planes, 50),
        "p95_planes": _percentile(planes, 95),
        "max_planes": max(planes) if planes else 0,
        "forced_count": sum(1 for r in results if r.decision_kind == "forced"),
        "abstain_count": sum(1 for r in results if r.decision_kind == "abstain"),
        "fallback_count": sum(
            1 for r in results if r.telemetry.get("fallback_triggered")
        ),
    }

    if baseline_actions is not None:
        flips = sum(
            1 for a, b in zip(actions, baseline_actions) if a != b
        )
        metrics["flips_vs_uniform_max"] = flips
        metrics["agreement_vs_uniform_max"] = 1.0 - flips / len(actions) if actions else 1.0

    return {
        "schedule_id": schedule_id,
        "grouping_policy": spec.grouping_policy,
        "metrics": metrics,
        "plane_histogram": {str(k): planes.count(k) for k in sorted(set(planes))},
    }


def main() -> int:
    hidden_dim = 16
    max_actions = 32
    n = 128
    R = 4

    hidden, contexts, legals = _make_traces(n, hidden_dim, max_actions)
    head = ResidualTritPlaneHead(
        hidden_dim,
        max_actions=max_actions,
        R=R,
        scale_mode="geometric_balanced",
        residual_normalization="none",
    )
    head.eval()

    # Establish a full-precision-equivalent baseline with all planes.
    uniform_max_spec = make_schedule_spec("uniform_max", max_planes=R)
    uniform_max_ctx = AdaptivePlaneRoutingContext(
        head, PlaneScheduler(uniform_max_spec), stability_patience=0
    )
    baseline_results = uniform_max_ctx.route_batch(hidden, contexts, legals)
    baseline_actions = [r.action_identity for r in baseline_results]

    schedule_results: list[dict[str, Any]] = []
    for schedule_id in SCHEDULES:
        schedule_results.append(
            _run_schedule(
                head, hidden, contexts, legals, schedule_id, baseline_actions
            )
        )

    avg_planes_by_schedule = {
        r["schedule_id"]: r["metrics"]["avg_planes"]
        for r in schedule_results
    }

    result: dict[str, Any] = {
        "recipe": {
            "hidden_dim": hidden_dim,
            "max_actions": max_actions,
            "n": n,
            "R": R,
            "scale_mode": head.residual_stack.scale_mode,
        },
        "schedules": schedule_results,
        "comparison": {
            "avg_planes_by_schedule": avg_planes_by_schedule,
            "baseline": "uniform_max",
        },
        "version_stamp": build_version_stamp("model.quantization", "matrix.perf"),
        "caveats": [
            "wiring fixture only; no ship gate, checkpoint, or latency claim",
            "head is randomly initialized, not trained, so semantic quality is not meaningful",
            "plane counts are algorithmic savings; no incremental packed-plane kernel is used",
            "energy and wall-clock numbers are not measured on-device",
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out_dir = Path("outputs/runs/cap4-02-adaptive-plane")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"adaptive_plane_fixture_{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(json.dumps(avg_planes_by_schedule, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
