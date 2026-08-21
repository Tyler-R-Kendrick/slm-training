"""Small durable primitives for fail-closed continuous-loop progress."""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import fmean, stdev

__all__ = [
    "BlockerRule",
    "BLOCKER_RULES",
    "append_power_deltas",
    "allocate_screening_suite_id",
    "lease_covers",
    "read_power_evidence",
    "power_evidence_summary",
]


class BlockerRule:
    __slots__ = ("owner", "postcondition", "max_attempts", "terminal")

    def __init__(self, owner: str, postcondition: str, max_attempts: int, terminal: str):
        self.owner = owner
        self.postcondition = postcondition
        self.max_attempts = max_attempts
        self.terminal = terminal


BLOCKER_RULES = {
    "loop_stalled_no_campaign": BlockerRule("autotrain", "campaign_initialized", 3, "escalate"),
    "heal_postcondition_failed": BlockerRule("autotrain", "declared_artifact_changed", 3, "escalate"),
    "vacuous_pass": BlockerRule("autotrain", "campaign_initialized_or_typed_action", 3, "escalate"),
}


def allocate_screening_suite_id(root: Path, n: int, *, prefix: str = "e938_role_safe_all_targets_smoke") -> str:
    """Return the next unused suite id; frozen snapshots are never reused."""
    root = Path(root)
    existing = {p.name for p in root.glob(f"{prefix}*_v1") if p.is_dir()}
    candidate = 1
    while f"{prefix}{n}_v{candidate}" in existing:
        candidate += 1
    return f"{prefix}{n}_v{candidate}"


def append_power_deltas(path: Path, *, cycle: str, metric: str, deltas: list[float], costs: list[float] | None = None) -> int:
    """Append paired observations and return the new observation count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    costs = costs or [0.0] * len(deltas)
    if len(costs) != len(deltas):
        raise ValueError("costs and deltas must have equal length")
    with path.open("a", encoding="utf-8") as fh:
        for delta, cost in zip(deltas, costs):
            fh.write(json.dumps({"cycle": cycle, "metric": metric, "delta": float(delta), "wall_seconds": float(cost)}, sort_keys=True) + "\n")
    return sum(1 for _ in path.open(encoding="utf-8"))


def read_power_evidence(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and isinstance(row.get("delta"), (int, float)):
            rows.append(row)
    return rows


def power_evidence_summary(path: Path, *, min_pairs: int = 100, min_cycles: int = 10) -> dict:
    rows = read_power_evidence(path)
    cycles = {str(row.get("cycle")) for row in rows}
    ready = len(rows) >= min_pairs and len(cycles) >= min_cycles
    values = [float(row["delta"]) for row in rows]
    return {
        "paired_deltas": len(rows),
        "cycles": len(cycles),
        "ready": ready,
        "sd": stdev(values) if len(values) > 1 else None,
        "mean": fmean(values) if values else None,
        "mean_abs_cost_seconds": fmean(float(row.get("wall_seconds", 0.0)) for row in rows) if rows else None,
    }


def lease_covers(path: Path, dirty_path: str, *, now: float | None = None) -> bool:
    """Honor only an unexpired lease that explicitly covers the dirty path."""
    try:
        lease = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    now = time.time() if now is None else now
    if float(lease.get("expires_at", 0)) <= now:
        return False
    prefixes = lease.get("path_prefixes") or lease.get("paths") or []
    return any(str(dirty_path).replace("\\", "/").startswith(str(prefix).rstrip("/") + "/") or str(dirty_path) == str(prefix).rstrip("/") for prefix in prefixes)
