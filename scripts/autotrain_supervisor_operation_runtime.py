"""Leased parent-side execution for one durable supervisor operation."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from scripts.autotrain_supervisor_operations import (
    _cancel_after_identity_drift,
    _operation_result_state,
    _operation_state,
    _reconcile_stale_operation,
    _replay_committed_operation,
    validate_operation_identity,
)


def run_operation(runtime, request: dict, *, sequence: int, log_event) -> dict | None:
    """Lease, execute and validate one operation; failure is never a model loss."""
    from scripts.merge_verification_evidence import digest
    from scripts.autotrain_pending import recover_driver_request
    from slm_training.harness_core.activity_contract import ActivityOutcome
    from slm_training.autoresearch.runtime.activity_runtime import StaleLease

    request = recover_driver_request(runtime, request)
    identity = digest(request)
    state = _operation_state(runtime, request, sequence, identity)
    activity_id = state.spec.activity_id
    if state.status == "succeeded":
        return _replay_committed_operation(runtime, request, state)
    lease = runtime.claim_next(
        capabilities={"local_process", "controller_publication"},
        activity_id=activity_id,
    )
    if lease is None:
        log_event(
            {
                "event": "operation_waiting",
                "activity_id": activity_id,
                "state": state.status,
                "wake": state.wake.model_dump() if state.wake else None,
            }
        )
        return None
    attempt_dir = runtime.attempt_dir(lease)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    request_path, output_path = (
        attempt_dir / "request.json",
        attempt_dir / "result.json",
    )
    execution_request = {
        **request,
        "lease": lease.model_dump(mode="json"),
        "parent_event": runtime.store.verify_event_chain()[-1]["event_id"],
    }
    runtime.store._replace_durable(
        request_path, json.dumps(execution_request, sort_keys=True)
    )
    started = time.monotonic()
    controller = Path(__file__).resolve().parents[1]
    bootstrap = "import runpy,sys;sys.path[:0]=[sys.argv.pop(1),sys.argv.pop(1)];runpy.run_module('scripts.run_autotrain_supervisor',run_name='__main__')"
    result = runtime.run(
        lease,
        [
            sys.executable,
            "-c",
            bootstrap,
            str(controller),
            str(controller / "src"),
            "--operation-request",
            str(request_path),
            "--operation-output",
            str(output_path),
        ],
        cwd=controller,
    )
    from scripts.run_autotrain_supervisor import _source_identity

    if _cancel_after_identity_drift(
        runtime,
        request,
        lease,
        _source_identity,
        output_path,
        started,
        activity_id,
        log_event,
    ):
        return None
    outcome, payload, pending_wait, wake = _operation_result_state(
        runtime, lease, request, identity, execution_request, result, output_path
    )
    outputs = (
        {"result.json": hashlib.sha256(output_path.read_bytes()).hexdigest()}
        if payload is not None
        else {}
    )
    try:
        validate_operation_identity(request, _source_identity, "before parent commit")
    except ValueError as exc:
        output_path.unlink(missing_ok=True)
        outcome, payload, outputs, pending_wait = (
            ActivityOutcome.CANCELLED,
            None,
            {},
            None,
        )
        wake = None
        identity_error = exc
    else:
        identity_error = None
    try:
        if (
            (not pending_wait or _source_repair_pending(payload))
            and outcome
            not in {
                ActivityOutcome.SUCCEEDED,
                ActivityOutcome.CANCELLED,
            }
        ):
            from slm_training.autoresearch.heal.operation_recovery import (
                record_operation_failure,
            )

            record_operation_failure(
                runtime,
                lease,
                request,
                result,
                outcome=ActivityOutcome.WALL_BUDGET
                if outcome == ActivityOutcome.RETRY
                else outcome,
            )
        runtime.finish(
            lease,
            outcome=outcome,
            outputs=outputs,
            spent_seconds=time.monotonic() - started,
            wake=wake
            if outcome not in {ActivityOutcome.SUCCEEDED, ActivityOutcome.CANCELLED}
            else None,
        )
    except StaleLease:
        # Uncommitted child output is not evidence. Reconcile adopts only a
        # previously verified output receipt; all other states retry later.
        return _reconcile_stale_operation(
            runtime, request, lease, activity_id, log_event
        )
    if isinstance(identity_error, ValueError):
        log_event(
            {
                "event": "operation_identity_drift",
                "activity_id": activity_id,
                "operation": request["operation"],
                "error": str(identity_error),
            }
        )
        return None
    log_event(
        {
            "event": "operation_finished",
            "activity_id": activity_id,
            "operation": request["operation"],
            "outcome": outcome.value,
            "returncode": result.returncode,
            "spent_seconds": result.duration_seconds,
        }
    )
    return payload


def _source_repair_pending(payload):
    """Keep the original request for verified source succession, not repair authority."""
    from slm_training.autoresearch.heal.classify import classify_blocker

    pending = (payload or {}).get("pending") or {}
    blocker = pending.get("blocker") or pending.get("readiness", {}).get("blocker") or {}
    return (
        blocker.get("kind") == "repair_harness"
        and blocker.get("required_capability") in {
            "source_repair", "configured_source_repair", "bounded_measurement_repair",
        }
        and classify_blocker("repair_harness", "", code=blocker.get("blocker_code") or "") == "code"
    )
