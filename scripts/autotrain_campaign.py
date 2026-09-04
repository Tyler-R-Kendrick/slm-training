"""Identifying the campaign a cycle belongs to.

One responsibility: resolving campaign identity and lineage -- which campaign a
cycle sits in, which experiment started it, and the component version it stamps.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.autotrain_io import (
    read_json,
)
from slm_training.autoresearch.hillclimb import (
    DEFAULT_MAX_CUMULATIVE_EPOCHS,
)


def campaign_started_experiment(root: Path, campaign_id: object) -> bool:
    """Return whether a reserved champion attempt reached actual execution."""
    if not isinstance(campaign_id, str) or not campaign_id:
        return False
    path = root / campaign_id / "events.jsonl"
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("event_type") == "experiment_started":
            return True
    return False


PROMOTE_AUTHORITY_HARNESS_COMPONENT = "harness.autoresearch.experiment_campaign"


def experiment_campaign_component_version() -> str:
    """Current continuous/promote harness component version from versions.json."""
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "slm_training"
        / "resources"
        / "versions.json"
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        components = payload.get("components") or {}
        row = components.get(PROMOTE_AUTHORITY_HARNESS_COMPONENT) or {}
        version = str(row.get("version") or "").strip()
        return version or "unknown"
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return "unknown"


def lineage_campaign_ids(
    root: Path,
    predecessor_campaign_id: str | None,
    *,
    max_cycles: int | None = None,
) -> list[str]:
    """Return one loop's content-linked campaign chain, oldest first."""

    cursor = predecessor_campaign_id
    seen: set[str] = set()
    loop_id: str | None = None
    chain: list[str] = []
    lineage = itertools.count() if max_cycles is None else range(max(0, max_cycles))
    for _ in lineage:
        if not cursor or cursor in seen:
            break
        seen.add(cursor)
        camp_dir = root / cursor
        campaign = read_json(camp_dir / "campaign.json")
        handoff = read_json(camp_dir / "cycle_handoff.json")
        campaign_loop = str(campaign.get("loop_id") or handoff.get("loop_id") or "")
        if loop_id is None:
            loop_id = campaign_loop or None
        elif campaign_loop and campaign_loop != loop_id:
            break
        chain.append(cursor)
        cursor = str(campaign.get("predecessor_campaign_id") or "")
    return list(reversed(chain))


def warm_start_policy(policy: Any) -> dict[str, Any]:
    measurement = getattr(policy, "measurement", None)
    if not isinstance(measurement, Mapping):
        measurement = {}
    raw = measurement.get("warm_start") if isinstance(measurement, Mapping) else None
    block = dict(raw) if isinstance(raw, Mapping) else {}
    block.setdefault("enabled", True)
    block.setdefault("max_cumulative_epochs", DEFAULT_MAX_CUMULATIVE_EPOCHS)
    # With no confirmed champion, the loop's first complete control checkpoint
    # seeds a ``baseline_seed`` champion so both arms warm start from the same
    # weights (directive §2.6). It never counts as confirmed / promoted.
    block.setdefault("seed_from_baseline_control", True)
    return block


def campaign_power_feasibility(
    camp_dir: Path, candidate_id: str
) -> dict[str, Any] | None:
    """Locked pre-run power report from the candidate's campaign manifest."""
    if not candidate_id:
        return None
    report = read_json(camp_dir / "manifests" / f"{candidate_id}.json").get(
        "power_feasibility"
    )
    return report if isinstance(report, dict) else None


def campaign_at_cycle(root: Path, loop_id: str, cycle_index: int) -> str | None:
    matches: list[str] = []
    for path in root.glob("*/campaign.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if (
            data.get("loop_id") == loop_id
            and int(data.get("cycle_index") or 0) == cycle_index
        ):
            matches.append(str(data.get("campaign_id")))
    if len(matches) > 1:
        raise RuntimeError(f"multiple campaigns claim loop cycle {cycle_index}")
    return matches[0] if matches else None


def experiment_artifact(camp_dir: Path, experiment_id: str) -> dict[str, Any]:
    for path in (camp_dir / "artifacts" / "experiments").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("experiment_id") == experiment_id:
            return payload
    raise RuntimeError(f"frozen replay experiment is missing: {experiment_id}")


def campaign_id(loop_id: str, cycle: int, *, date: str | None = None) -> str:
    """Loop-scoped identity; historical campaign ids remain readable."""

    slug = re.sub(r"[^A-Za-z0-9]+", "-", loop_id).strip("-").lower()[:24]
    digest = hashlib.sha256(loop_id.encode("utf-8")).hexdigest()[:8]
    day = date or time.strftime("%Y%m%d")
    return f"continuous-loop-{day}-{slug or 'loop'}-{digest}-c{cycle}"
