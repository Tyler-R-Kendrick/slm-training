"""Truthful status from verified runtime history; heartbeat is never a model win."""

from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path

from slm_training.autoresearch.runtime.activity_process import controller_status
from slm_training.autoresearch.runtime.activity_projection import ActivityProjection
from slm_training.autoresearch.storage import CampaignStore, loop_result_rows


def runtime_store(root: Path, loop_id: str) -> CampaignStore:
    """Confine operator-supplied IDs before touching any controller state."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", loop_id):
        raise ValueError("invalid_loop_id")
    parent = root.resolve() / "loops"
    path = parent / loop_id
    if parent.is_symlink() or path.is_symlink() or (path / "runtime").is_symlink():
        raise ValueError("loop_state_symlink")
    return CampaignStore("runtime", path)


def loop_status(root: Path, loop_id: str) -> dict:
    store = runtime_store(root, loop_id)
    events = store.verify_event_chain()
    try:
        controller = controller_status(store, now=time.time())
    except (OSError, ValueError, KeyError):
        controller = {"alive": False, "reason": "controller_identity_unverifiable"}
    states = ActivityProjection(store).read()
    transitions = [
        event for event in events if event["event_type"] == "activity_transition"
    ]
    progress = [
        event
        for event in transitions
        if event["detail"]["operation"] in ("finish", "reconcile_finish")
        and event["detail"].get("outputs")
    ]
    # Existing rows retain their historical claim class. A finished experiment
    # is not necessarily a complete identity-paired comparison.
    rows = loop_result_rows(root, loop_id)
    by_kind = Counter()
    for state in states.values():
        by_kind[state.spec.kind] += state.charged_seconds
    return {
        "schema_version": "autonomy_status/v1",
        "loop_id": loop_id,
        "active": bool(controller.get("alive")),
        "controller": controller,
        "activity_counts": dict(Counter(state.status for state in states.values())),
        "attempted_activities": sum(state.attempts for state in states.values()),
        "operational_artifact_advances": len(progress),
        "recorded_experiment_results": len(rows),
        "verified_paired_comparisons": None,
        "last_verified_comparison": None,
        "paired_evidence_status": "not_established_by_runtime_history",
        "scientific_progress": False,
        "resource_seconds_by_kind": dict(by_kind),
        "waiting": [
            {
                "activity_id": state.spec.activity_id,
                "family": state.spec.family,
                "status": state.status,
                "action": state.action,
                "wake": state.wake.model_dump(mode="json") if state.wake else None,
            }
            for state in states.values()
            if state.status.startswith("waiting")
        ],
        "source_identities": sorted(
            {state.spec.source_digest for state in states.values()}
        ),
        "champion_state": "not_inferred_from_operational_activity",
        "ship_state": "not_inferred_from_operational_activity",
    }
