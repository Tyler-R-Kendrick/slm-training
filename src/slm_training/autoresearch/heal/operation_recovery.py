"""Durable bridge from failed controller operations to the existing repair seam.

Raw stderr is untrusted and is never interpreted as permission to edit source.
An approved operation-specific reproducer must independently reproduce the fault.
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path

from slm_training.harness_core.activity_contract import (
    ActivityOutcome,
    ActivitySpec,
    ResourceGrant,
    contract_digest,
)
from slm_training.harness_core.bounded_process import ProcessOutcome
from slm_training.levers import KILL_GRACE_SECONDS

from .isolation import IsolationSpec, IsolationUnavailable, run_isolated
from .isolation_workspace import manifest_digest, private_snapshot, tree_manifest
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


def pending_operation_repairs(runtime) -> list[dict]:
    """Recover the repair queue from the canonical events, including after crash."""
    states = runtime.snapshot()
    latest = {}
    for event in runtime.store.verify_event_chain():
        if event["event_type"] == "operation_repair_requested":
            latest[event["experiment_id"]] = event["detail"]
    jobs = []
    for activity, detail in latest.items():
        state = states.get(activity)
        if state is None or state.status not in {
            "waiting_repair",
            "waiting_dependency",
        }:
            continue
        original = detail["request"]
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
    return jobs


def diagnose_operation(pending, context, config, journal):
    """One bounded independent probe, then reuse its identity-bound disposition.

    Returns (resolved pending or None, reason, probe_ran). Probe and agent launch
    occupy different invocations. No recipe means an explicit capability wait.
    """
    operation = pending["original_operation"]
    code = config.operation_recipes.get(operation)
    recipe = config.recipes.get(code)
    unavailable = None
    if pending.get("observed_outcome") not in {
        ActivityOutcome.UNKNOWN_FAILURE.value,
        ActivityOutcome.CODE_FAILURE.value,
    }:
        unavailable = (
            "typed_outcome_requires_original_owner:" + pending["observed_outcome"]
        )
    elif recipe is None:
        unavailable = "operation_reproducer_not_configured:" + operation
    if unavailable:
        return None, unavailable, False
    identity = contract_digest(
        {
            "request": pending["original_request_digest"],
            "source": context.source_digest,
            "environment": context.environment_digest,
            "config": config.digest(),
        }
    )
    events = journal.verify_event_chain()
    matches = [e for e in events if e["detail"].get("diagnosis_id") == identity]
    completed = [
        e for e in matches if e["event_type"] == "operation_diagnosis_finished"
    ]
    if completed:
        detail = completed[-1]["detail"]
        return detail["resolved"], detail["reason"], False
    if any(e["detail"].get("attempt_id") == context.attempt_id for e in matches):
        return None, "interrupted_diagnosis_requires_reconciliation", False
    grant = config.grant
    if grant is None or grant.expires_at <= time.time():
        return None, "diagnosis_grant_missing_or_expired", False
    seconds = min(10.0, grant.interrupt_seconds)
    reserved = sum(
        e["detail"]["reserved_seconds"]
        for e in events
        if e["event_type"] == "operation_diagnosis_started"
        and e["detail"]["grant_digest"] == grant.digest()
    )
    if (
        len(matches) >= grant.max_attempts
        or reserved + seconds + KILL_GRACE_SECONDS > grant.total_seconds
    ):
        return None, "diagnosis_grant_exhausted", False
    journal.append_event(
        "operation_diagnosis_started",
        detail={
            "diagnosis_id": identity,
            "attempt_id": context.attempt_id,
            "grant_digest": grant.digest(),
            "reserved_seconds": seconds + KILL_GRACE_SECONDS,
        },
    )
    resolved, reason, spent = _probe(pending, context, config, recipe, code, seconds)
    journal.append_event(
        "operation_diagnosis_finished",
        detail={
            "diagnosis_id": identity,
            "resolved": resolved,
            "reason": reason,
            "spent_seconds": spent,
            "scientific_outcome": False,
        },
    )
    return resolved, reason, True


def _probe(pending, context, config, recipe, code, seconds):
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix="operation-diagnose-", dir=context.source.parent
    ) as directory:
        snapshot = private_snapshot(context.source, Path(directory) / "source")
        if manifest_digest(tree_manifest(snapshot)) != context.source_digest:
            return None, "diagnosis_source_identity_changed", time.monotonic() - started
        try:
            result = run_isolated(
                IsolationSpec(
                    snapshot,
                    runtime_roots=tuple(Path(p) for p in config.runtime_roots),
                    timeout_seconds=seconds,
                ),
                recipe.original.argv,
            )
        except IsolationUnavailable:
            return None, "diagnosis_isolation_unavailable", time.monotonic() - started
    matches = (
        result.outcome == ProcessOutcome.COMPLETED
        and result.returncode == recipe.failure_returncode
        and not (result.stdout_truncated or result.stderr_truncated)
        and hashlib.sha256(result.stdout.encode()).hexdigest()
        == recipe.failure_stdout_sha256
        and hashlib.sha256(result.stderr.encode()).hexdigest()
        == recipe.failure_stderr_sha256
    )
    if not matches:
        return None, "original_fault_not_reproduced", result.duration_seconds
    observed = pending.get("failure_observation", {})
    if (
        observed.get("returncode") != recipe.failure_returncode
        or observed.get("stdout_sha256") != recipe.failure_stdout_sha256
        or observed.get("stderr_sha256") != recipe.failure_stderr_sha256
        or observed.get("truncated", True)
    ):
        return (
            None,
            "operation_observation_not_covered_by_recipe",
            result.duration_seconds,
        )
    resolved = {
        **pending,
        "blocker_code": code,
        "unmet_predicate": recipe.original.check_id,
        "required_capability": "source_repair",
    }
    return resolved, "original_fault_reproduced", result.duration_seconds


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
        idempotency_key="operation-plan:" + checked["publication_id"],
        detail=plan,
    )
    # Cancel before register: crash recovery replays the plan without allowing
    # old/new executions to overlap or losing the remainder of the logical grant.
    runtime.cancel(
        activity, reason="replaced_by_verified_release:" + checked["publication_id"]
    )
    runtime.register(ActivitySpec.model_validate(plan["spec"]))
    runtime.store.append_event(
        "operation_successor_activated",
        experiment_id=activity,
        idempotency_key="operation-activation:" + checked["publication_id"],
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
    successor_id = (
        "successor-"
        + contract_digest(
            {"activity": activity, "publication": checked["publication_id"]}
        )[:24]
    )
    request = {
        **original,
        "cwd": checked["successor_execution"],
        "source_digest": checked["source_digest"],
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
            "scientific_replicate_increment": 0,
        },
    }
    spec = ActivitySpec.model_validate(
        {
            **state.spec.model_dump(mode="json"),
            "activity_id": successor_id,
            "source_digest": checked["source_digest"],
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
