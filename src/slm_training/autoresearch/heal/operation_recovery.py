"""Durable bridge from failed controller operations to the existing repair seam.

Raw stderr is untrusted and is never interpreted as permission to edit source.
An approved operation-specific reproducer must independently reproduce the fault.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from slm_training.harness_core.activity_contract import (
    ActivityOutcome,
    ActivitySpec,
    ResourceGrant,
    contract_digest,
)

from .operation_failure_signature import capture_failure_signature
from .operation_diagnosis import diagnose_operation as diagnose_operation
from .repair_release import verified_activation_handoff


class SuccessorGrantExhausted(ValueError):
    """The verified repair cannot authorize more logical attempts or compute."""


def record_operation_failure(runtime, lease, request, result, *, outcome):
    """Persist before finish; return a controller-owned pending diagnosis.

    The caller owns finish/resource charging and scheduling repair children.
    stdout/stderr hashes retain exact observations without copying secrets/log
    instructions into a prompt. Exception/argparse text never sets the class.
    """
    outcome = ActivityOutcome(outcome)
    if outcome in {ActivityOutcome.SUCCEEDED, ActivityOutcome.CANCELLED}:
        raise ValueError("not a repairable failed operation")
    observation = {
        "returncode": result.returncode,
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "outcome": outcome.value,
        "process_outcome": result.outcome.value,
        "truncated": result.stdout_truncated or result.stderr_truncated,
        "launch_error": bool(result.launch_error),
    }
    source = request.get("cwd")
    signature = capture_failure_signature(result, Path(source)) if source else None
    if signature is not None:
        observation["failure_signature"] = signature.model_dump(mode="json")
    pending = {
        "kind": "repair_harness",
        "blocker_code": "controller_operation_failure",
        "affected_activity_id": lease.activity_id,
        "original_operation": request["operation"],
        "original_request_digest": contract_digest(request),
        "reason": "controller operation requires bounded diagnosis",
        "observed_outcome": outcome.value,
        "failure_observation": observation,
    }
    artifact = runtime.store.write_artifact(
        "operation_failures", {"request": request, "pending": pending}
    )
    runtime.store.append_event(
        "operation_repair_requested",
        experiment_id=lease.activity_id,
        artifact_sha256=artifact.stem,
        idempotency_key="operation-failure:" + lease.token,
        detail={"request": request, "pending": pending, "fence": lease.token},
    )
    return pending


def record_driver_reconciliation(runtime, lease, request, driver_pending):
    """Queue an exact unresolved cursor without inventing a process failure."""
    from scripts.autotrain_pending import validate_pending

    outcome, wake = validate_pending(driver_pending)
    blocker = driver_pending.get("blocker") or {}
    if (
        request.get("operation") != "driver"
        or outcome != ActivityOutcome.CAPABILITY
        or driver_pending.get("reason") != "driver_attempt_requires_reconciliation"
        or blocker.get("kind") != "repair_harness"
        or blocker.get("blocker_code") != driver_pending["reason"]
        or blocker.get("required_capability") != "driver_continuation_reconciliation"
        or blocker.get("input_digest") != driver_pending.get("input_digest")
        or wake.source != "driver_cycle_checkpoint"
    ):
        raise ValueError("untrusted driver reconciliation pending")
    pending = {
        "kind": "repair_harness",
        "blocker_code": blocker["blocker_code"],
        "required_capability": blocker["required_capability"],
        "unmet_predicate": "driver_command_cursor_reconciled",
        "input_digest": blocker["input_digest"],
        "pending_digest": contract_digest(driver_pending),
        "affected_activity_id": lease.activity_id,
        "original_operation": "driver",
        "original_request_digest": contract_digest(request),
        "reason": driver_pending["reason"],
        "observed_outcome": outcome.value,
    }
    artifact = runtime.store.write_artifact(
        "operation_failures", {"request": request, "pending": pending, "driver_pending": driver_pending}
    )
    runtime.store.append_event(
        "operation_repair_requested",
        experiment_id=lease.activity_id,
        artifact_sha256=artifact.stem,
        idempotency_key="driver-reconciliation:" + lease.token,
        detail={"request": request, "pending": pending, "fence": lease.token},
    )
    return pending


def pending_operation_repairs(runtime) -> list[dict]:
    """Recover the repair queue from the canonical events, including after crash."""
    states = runtime.snapshot()
    latest = {}
    events = runtime.store.verify_event_chain()
    for event in events:
        if event["event_type"] == "operation_repair_requested":
            latest[event["experiment_id"]] = event["detail"]
    jobs = []
    for activity, detail in latest.items():
        state = states.get(activity)
        if state is None or state.status not in {
            "waiting_repair", "waiting_dependency", "waiting_capability",
        }:
            continue
        if (state.status == "waiting_capability"
                and detail["pending"].get("blocker_code")
                != "driver_attempt_requires_reconciliation"):
            continue
        original = detail["request"]
        if (
            detail["pending"].get("observed_outcome")
            == ActivityOutcome.WALL_BUDGET.value
        ):
            continue  # Remaining grant/cursor reconciliation belongs to the driver.
        # Do not spawn recursive repair-of-repair chains. The original durable
        # job remains visible and capability/diagnosis receipts explain its wait.
        if original["operation"] == "repair":
            continue
        jobs.append(
            {
                **original,
                "operation": "repair",
                "hard_pending": [detail["pending"]],
                "campaign_id": original.get("campaign_id")
                or "repair-" + original["loop_id"],
                "max_heal_attempts": original.get("max_heal_attempts", 1),
                "playbooks_enabled": original.get("playbooks_enabled", False),
            }
        )
    serviced = {
        event["experiment_id"]: index
        for index, event in enumerate(events)
        if event["event_type"] == "operation_repair_serviced"
    }
    return sorted(
        jobs,
        key=lambda job: serviced.get(
            job["hard_pending"][0]["affected_activity_id"], -1
        ),
    )


def wake_verified_operation(runtime, handoff, *, cwd):
    """Replace failed execution under a verified source, preserving logical work.

    Historical name retained for callers; the old immutable activity is cancelled,
    never woken under a different source. Return a pre-registered successor request
    with its remaining grant. Parent run_operation must use successor_activity_id
    and its registered spec, not synthesize a fresh/default resource allowance.
    """
    from slm_training.harness_core.execution_release import (
        runtime_source_identity,
    )

    checked = verified_activation_handoff(runtime.store, handoff)
    activation = checked.get("activation_id", checked["publication_id"])
    if (
        Path(cwd).resolve() != Path(checked["successor_execution"]).resolve()
        or runtime_source_identity(Path(cwd)) != checked["source_digest"]
        or Path.cwd().resolve() != Path(cwd).resolve()
        or not Path(__file__).resolve().is_relative_to(Path(cwd).resolve())
    ):
        raise ValueError("successor_not_active_in_this_execution")
    activity = checked["resume_activity_id"]
    events = runtime.store.verify_event_chain()
    plans = [
        e["detail"]
        for e in events
        if e["event_type"] == "operation_successor_planned"
        and e["detail"]["handoff"] == checked
    ]
    plan = plans[-1] if plans else _successor_plan(runtime, checked, events)
    runtime.store.append_event(
        "operation_successor_planned",
        experiment_id=activity,
        idempotency_key="operation-plan:" + activation,
        detail=plan,
    )
    # Cancel before register: crash recovery replays the plan without allowing
    # old/new executions to overlap or losing the remainder of the logical grant.
    runtime.cancel(activity, reason="replaced_by_verified_release:" + activation)
    runtime.register(ActivitySpec.model_validate(plan["spec"]))
    runtime.store.append_event(
        "operation_successor_activated",
        experiment_id=activity,
        idempotency_key="operation-activation:" + activation,
        detail={
            "handoff": checked,
            "successor_request_digest": contract_digest(plan["request"]),
        },
    )
    return plan["request"]


def _successor_plan(runtime, checked, events):
    activity = checked["resume_activity_id"]
    rows = [
        e
        for e in events
        if e["event_type"] == "operation_repair_requested"
        and e["experiment_id"] == activity
    ]
    if not rows:
        raise ValueError("original_operation_request_missing")
    original = rows[-1]["detail"]["request"]
    state = runtime.snapshot()[activity]
    if state.spec.input_digest != contract_digest(original) or state.status in {
        "running",
        "succeeded",
        "cancelled",
    }:
        raise ValueError("original_operation_identity_changed")
    seconds = state.spec.grant.total_seconds - state.charged_seconds
    attempts = state.spec.grant.max_attempts - state.attempts
    if (
        attempts <= 0
        or seconds
        < state.spec.grant.attempt_seconds
        + state.spec.grant.finalization_reserve_seconds
    ):
        raise SuccessorGrantExhausted(
            "successor_resource_grant_exhausted:max_attempts_or_seconds"
        )
    remaining = ResourceGrant.model_validate(
        {
            **state.spec.grant.model_dump(),
            "total_seconds": seconds,
            "max_attempts": attempts,
        }
    )
    prior = original.get("logical_continuation", {})
    activation = checked.get("activation_id", checked["publication_id"])
    environment = _successor_environment(state.spec.environment_digest, checked)
    successor_id = (
        "successor-"
        + contract_digest({"activity": activity, "publication": activation})[:24]
    )
    request = {
        **original,
        "cwd": checked["successor_execution"],
        "source_digest": checked["source_digest"],
        "environment_digest": environment,
        "successor_activity_id": successor_id,
        "resource_grant": remaining.model_dump(mode="json"),
        "logical_continuation": {
            "schema_version": "operation_continuation/v1",
            "predecessor_activity_id": activity,
            "predecessor_request_digest": contract_digest(original),
            "logical_activity_id": prior.get("logical_activity_id", activity),
            "logical_request_digest": prior.get(
                "logical_request_digest", contract_digest(original)
            ),
            "logical_resource_grant": prior.get(
                "logical_resource_grant", state.spec.grant.model_dump(mode="json")
            ),
            "prior_charged_seconds": prior.get("prior_charged_seconds", 0)
            + state.charged_seconds,
            "prior_attempts": prior.get("prior_attempts", 0) + state.attempts,
            "publication_id": checked["publication_id"],
            "activation_id": activation,
            "scientific_replicate_increment": 0,
        },
    }
    spec = ActivitySpec.model_validate(
        {
            **state.spec.model_dump(mode="json"),
            "activity_id": successor_id,
            "source_digest": checked["source_digest"],
            "environment_digest": environment,
            "input_digest": contract_digest(request),
            "output_namespace": "attempts/" + successor_id,
            "grant": remaining.model_dump(mode="json"),
        }
    )
    return {
        "handoff": checked,
        "request": request,
        "spec": spec.model_dump(mode="json"),
    }


def _successor_environment(original, checked):
    """Only the recorded path relocation may change an operation environment."""
    from scripts.merge_verification_evidence import digest, environment_identity

    transition = checked.get("environment_transition")
    if transition is None:
        return original
    current = digest(environment_identity())
    if (
        transition["predecessor_digest"] != original
        or transition["successor_digest"] != current
    ):
        raise ValueError("successor_environment_transition_mismatch")
    return current
