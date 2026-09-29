"""Truthful status from verified runtime history; heartbeat is never a model win."""

from __future__ import annotations

import hashlib
import json
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
    comparisons = _comparisons(root, loop_id)
    monitors = [
        e["detail"] for e in events if e["event_type"] == "operations_monitor_checked"
    ]
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
        "verified_paired_comparisons": len(comparisons),
        "last_verified_comparison": comparisons[-1] if comparisons else None,
        "paired_evidence_status": "validated_diagnostic_artifacts"
        if comparisons
        else "not_established_by_runtime_history",
        "last_operational_progress_event": progress[-1]["event_id"]
        if progress
        else None,
        "last_monitor_check": monitors[-1] if monitors else None,
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


def _comparisons(root: Path, loop_id: str) -> list[dict]:
    """Project immutable native measurement evidence; diagnostics are not wins."""
    from slm_training.autoresearch.storage import _sha, loop_campaigns

    results = []
    for campaign in loop_campaigns(root, loop_id, last=None):
        store = CampaignStore(campaign.campaign_id, root)
        for event in store.verify_event_chain():
            if event["event_type"] != "supervised_measurement_consumed":
                continue
            digest = event["artifact_sha256"]
            path = (
                store.root
                / "artifacts/supervised_measurement_result"
                / f"{digest}.json"
            )
            result = json.loads(path.read_text())
            if path.is_symlink() or _sha(result) != digest:
                raise ValueError("comparison_artifact_integrity_failure")
            _validate_comparison(store, result)
            results.append(
                {
                    "campaign_id": campaign.campaign_id,
                    "artifact_sha256": digest,
                    "evidence_class": result["evidence_class"],
                    "confirmation_complete": False,
                    "promotion_allowed": False,
                }
            )
    return results


def _validate_comparison(store, result):
    from scripts.autotrain_measurement import measurement_is_complete
    from scripts.autotrain_metrics import read_paired_nll

    if (
        result.get("schema") != "supervised_measurement_result/v1"
        or result.get("diagnostic_complete") is not True
        or result.get("confirmation_complete") is not False
        or result.get("promotion_allowed") is not False
        or result.get("ship_eligible") is not False
        or result.get("operation", {}).get("campaign_id") != store.campaign_id
        or set(result.get("arms", {})) != {"control", "candidate"}
        or not measurement_is_complete(result.get("decision", {}))
    ):
        raise ValueError("comparison_contract_incomplete")
    for name, arm in result["arms"].items():
        if not arm.get("agentv_artifacts"):
            raise ValueError("comparison_agentv_evidence_missing")
        directory = store.root / "runs" / name
        paths = {
            str(directory / "scoreboard.json"): arm["scoreboard_sha256"],
            str(directory / "loss_suites.json"): arm["loss_report_sha256"],
            **arm["agentv_artifacts"],
        }
        for filename, expected in paths.items():
            path = Path(filename)
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(store.root.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                raise ValueError("comparison_referenced_artifact_changed")
    paired, counts, failures = read_paired_nll(
        store.root / "runs/control", store.root / "runs/candidate"
    )
    selected = result.get("selection", {}).get("selected_record_ids", [])
    if (
        failures
        or not paired
        or counts != result["paired_counts"]
        or len(selected) != 6
        or len(set(selected)) != 6
        or paired.get("selected_record_ids") != selected
        or any(set(paired[arm]) != set(selected) for arm in ("control", "candidate"))
    ):
        raise ValueError("comparison_paired_evidence_changed")
