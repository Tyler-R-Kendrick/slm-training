"""Bounded, source-pinned diagnosis for failed controller operations."""

from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path

from slm_training.harness_core.activity_contract import ActivityOutcome, contract_digest
from slm_training.harness_core.bounded_process import ProcessOutcome
from slm_training.levers import KILL_GRACE_SECONDS

from .grant_accounting import append_budget_reservation, diagnosis_budget_exhausted
from .isolation import IsolationSpec, IsolationUnavailable, run_isolated
from .isolation_workspace import manifest_digest, private_snapshot, tree_manifest
from .operation_failure_signature import (
    OperationFailureSignature,
    capture_failure_signature,
)


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
            "failure_observation": pending.get("failure_observation"),
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
    exhausted = diagnosis_budget_exhausted(
        events,
        grant,
        journal,
        pending.get("affected_activity_id", ""),
        code,
        seconds + KILL_GRACE_SECONDS,
    )
    appended = (
        False
        if exhausted
        else append_budget_reservation(
            journal,
            events,
            "operation_diagnosis_started",
            detail={
                "diagnosis_id": identity,
                "original_request_digest": pending["original_request_digest"],
                "affected_activity_id": pending.get("affected_activity_id", ""),
                "blocker_code": code,
                "attempt_id": context.attempt_id,
                "grant_digest": grant.digest(),
                "grant_id": grant.grant_id,
                "grant_accounting_digest": grant.accounting_digest(),
                "grant_record": grant.model_dump(mode="json"),
                "reserved_seconds": seconds + KILL_GRACE_SECONDS,
            },
        )
    )
    if exhausted or not appended:
        reason = (
            "diagnosis_grant_exhausted" if exhausted else "diagnosis_reservation_raced"
        )
        return None, reason, False
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
    expected_signature = recipe.original_failure_signature
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
        probe_signature = None
        if expected_signature is not None:
            # Bubblewrap exposes snapshot at /workspace. Test/host adapters
            # may report physical snapshot path instead.
            probe_signature = capture_failure_signature(result, Path("/workspace"))
            if probe_signature is None:
                probe_signature = capture_failure_signature(result, snapshot)
    matches = (
        result.outcome == ProcessOutcome.COMPLETED
        and result.returncode == recipe.failure_returncode
        and not (result.stdout_truncated or result.stderr_truncated)
        and hashlib.sha256(result.stdout.encode()).hexdigest()
        == recipe.failure_stdout_sha256
        and hashlib.sha256(result.stderr.encode()).hexdigest()
        == recipe.failure_stderr_sha256
        and (expected_signature is None or probe_signature == expected_signature)
    )
    if not matches:
        return None, "original_fault_not_reproduced", result.duration_seconds
    observed = pending.get("failure_observation", {})
    signature = recipe.original_failure_signature
    try:
        signature_matches = (
            OperationFailureSignature.model_validate(
                observed.get("failure_signature"), strict=True
            )
            == signature
            if signature is not None
            else observed.get("stdout_sha256") == recipe.failure_stdout_sha256
            and observed.get("stderr_sha256") == recipe.failure_stderr_sha256
        )
    except ValueError:
        signature_matches = False
    if (
        observed.get("returncode")
        != (
            signature.returncode if signature is not None else recipe.failure_returncode
        )
        or not signature_matches
        or observed.get("truncated", True)
        or observed.get("launch_error", False)
        or observed.get("process_outcome", ProcessOutcome.COMPLETED.value)
        != ProcessOutcome.COMPLETED.value
        or not all(
            isinstance(observed.get(key), str)
            and len(observed[key]) == 64
            and all(c in "0123456789abcdef" for c in observed[key])
            for key in ("stdout_sha256", "stderr_sha256")
        )
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
