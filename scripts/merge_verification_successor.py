"""Carry only unspent verification resources into a source-bound successor."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.merge_verification_evidence import digest


def _predecessor_state(runtime, events, predecessor_id, grant_for_plan):
    locks = [
        event
        for event in events
        if event["event_type"] == "release_verification_locked"
        and event.get("experiment_id") == predecessor_id
    ]
    if len(locks) != 1:
        raise ValueError("release_verification_predecessor_lock_missing_or_ambiguous")
    lock = locks[0]
    old_path = Path(lock["detail"]["artifact"]).resolve()
    if not old_path.is_relative_to(runtime.store.root.resolve()):
        raise ValueError("release_verification_predecessor_artifact_outside_store")
    old_plan = json.loads(old_path.read_text())
    old_digest = lock["detail"]["plan_digest"]
    if digest(old_plan) != old_digest or old_plan.get("activity_id") != predecessor_id:
        raise ValueError("release_verification_predecessor_plan_invalid")
    state = runtime.snapshot().get(predecessor_id)
    if (
        state is None
        or state.spec.input_digest != old_digest
        or state.spec.source_digest != old_plan["source_digest"]
        or state.spec.environment_digest != old_plan["environment_digest"]
        or state.spec.grant != grant_for_plan(old_plan)
    ):
        raise ValueError("release_verification_predecessor_activity_mismatch")
    return old_digest, state


def _existing_successor(runtime, events, plan, predecessor_id, state):
    existing = [
        event
        for event in events
        if event["event_type"] == "release_verification_successor_activated"
        and event["detail"].get("predecessor_activity_id") == predecessor_id
    ]
    if existing:
        if (
            len(existing) != 1
            or existing[0]["detail"].get("successor_activity_id")
            != plan["activity_id"]
        ):
            raise ValueError("release_verification_successor_already_forked")
        artifact = Path(existing[0]["detail"]["artifact"]).resolve()
        if not artifact.is_relative_to(runtime.store.root.resolve()):
            raise ValueError("release_verification_successor_artifact_outside_store")
        successor = json.loads(artifact.read_text())
        if digest(successor) != existing[0]["detail"].get("successor_plan_digest"):
            raise ValueError("release_verification_successor_plan_invalid")
        for key in (
            "source",
            "source_digest",
            "environment_digest",
            "runtime_digest",
            "runtime_roots",
            "base_ref",
            "step_seconds",
            "state_dir",
            "local_feedback",
            "require_js_runtime",
            "identity",
            "activity_id",
        ):
            if successor.get(key) != plan.get(key):
                raise ValueError("release_verification_successor_binding_changed")
        if state.status not in {"cancelled", "succeeded"}:
            runtime.cancel(
                predecessor_id, reason="release_verification_successor_activated"
            )
        return successor
    return None


def _remaining_successor(plan, predecessor_id, old_digest, state, grant_for_plan):
    grant = state.spec.grant
    total_seconds = grant.total_seconds - state.charged_seconds
    max_attempts = grant.max_attempts - state.attempts
    successor = {
        **plan,
        "successor_of": predecessor_id,
        "predecessor_plan_digest": old_digest,
        "total_seconds": total_seconds,
        "max_invocations": max_attempts,
    }
    successor_grant = grant_for_plan(successor)
    minimum_seconds = (
        successor_grant.interrupt_seconds
        + successor_grant.kill_grace_seconds
        + successor_grant.finalization_reserve_seconds
    )
    if max_attempts < 1 or total_seconds < minimum_seconds:
        raise ValueError("release_verification_predecessor_grant_exhausted")
    return successor


def release_successor_plan(runtime, plan: dict, predecessor_id: str, grant_for_plan):
    if plan["activity_id"] == predecessor_id:
        raise ValueError("release_verification_successor_must_be_new_activity")
    events = runtime.store.verify_event_chain()
    old_digest, state = _predecessor_state(
        runtime, events, predecessor_id, grant_for_plan
    )
    successor = _existing_successor(runtime, events, plan, predecessor_id, state)
    if successor is not None:
        return successor
    if state.status not in {"waiting_retry", "waiting_capability", "waiting_dependency"}:
        raise ValueError("release_verification_predecessor_not_resumable")
    successor = _remaining_successor(
        plan, predecessor_id, old_digest, state, grant_for_plan
    )
    artifact = runtime.store.write_artifact("release_verification_successors", successor)
    runtime.store.append_event(
        "release_verification_successor_activated",
        experiment_id=plan["activity_id"],
        idempotency_key="release-verification-successor:" + predecessor_id,
        detail={
            "predecessor_activity_id": predecessor_id,
            "predecessor_plan_digest": old_digest,
            "successor_activity_id": plan["activity_id"],
            "successor_plan_digest": digest(successor),
            "artifact": str(artifact),
        },
    )
    runtime.cancel(predecessor_id, reason="release_verification_successor_activated")
    return successor
