"""Bounded trusted-controller operations; untrusted repairs use heal.isolation."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path


def _operation_state(runtime, request, sequence, identity):
    """Register immutable inputs or reuse the verified successor's remaining grant."""
    from slm_training.harness_core.activity_contract import ActivitySpec, ResourceGrant

    successor = None
    if request.get("successor_activity_id"):
        successor = runtime.snapshot().get(request["successor_activity_id"])
        if (
            successor is None
            or successor.spec.input_digest != identity
            or successor.spec.source_digest != request["source_digest"]
            or successor.spec.grant.model_dump(mode="json")
            != request.get("resource_grant")
        ):
            raise ValueError("unregistered or altered verified successor request")
    activity_id = f"supervisor-{sequence}-{request['operation']}-{identity[:16]}"
    unfinished = [
        s
        for s in runtime.snapshot().values()
        if s.spec.input_digest == identity
        and s.status not in {"succeeded", "cancelled"}
    ]
    if unfinished:
        activity_id = unfinished[-1].spec.activity_id
    if successor is not None:
        activity_id = successor.spec.activity_id
    return runtime.register(
        successor.spec
        if successor is not None
        else ActivitySpec(
            activity_id=activity_id,
            family=request["loop_id"],
            kind="control",
            source_digest=request["source_digest"],
            capabilities=("local_process", "controller_publication")
            if request["operation"]
            in {"driver", "repair", "promotion_eval", "inspect", "closeout"}
            else ("local_process",),
            environment_digest=request["environment_digest"],
            input_digest=identity,
            grant=ResourceGrant.model_validate_json(request["driver_argv"][request["driver_argv"].index("--continuation-grant") + 1])
            if "--continuation-grant" in request.get("driver_argv", ()) else ResourceGrant(),
            output_namespace=f"attempts/{activity_id}",
        )
    )


def _operation_payload(output_path: Path, request: dict):
    from scripts.merge_verification_evidence import digest

    envelope = json.loads(output_path.read_text())
    if (
        envelope["schema_version"] != "supervisor_operation/v1"
        or envelope["request_digest"] != digest(request)
        or envelope["operation"] != request["operation"]
        or not isinstance(envelope["payload"], dict)
    ):
        raise ValueError("operation output identity mismatch")
    payload = envelope["payload"]
    if request["operation"] == "driver":
        if type(payload.get("returncode")) is not int or payload["returncode"] not in {
            0,
            10,
        }:
            raise ValueError(
                "driver output does not attest successful execution or a typed yield"
            )
        if payload["returncode"] == 10:
            from scripts.autotrain_pending import validate_pending

            validate_pending(payload.get("pending"))
    return payload


def interpret_operation_result(result, output_path: Path, execution_request: dict):
    """Validate the envelope and preserve typed operational incompleteness."""
    from scripts.autotrain_controller_repair import repair_payload_outcome
    from slm_training.harness_core.activity_contract import ActivityOutcome
    from slm_training.harness_core.bounded_process import ProcessOutcome

    if result.cancelled:
        return ActivityOutcome.CANCELLED, None
    if result.progress_stalled:
        return ActivityOutcome.UNKNOWN_FAILURE, None
    if result.timed_out:
        return ActivityOutcome.WALL_BUDGET, None
    if result.outcome != ProcessOutcome.COMPLETED:
        return ActivityOutcome.UNKNOWN_FAILURE, None
    if result.returncode == 0 and output_path.is_file():
        try:
            payload = _operation_payload(output_path, execution_request)
            outcome = (
                repair_payload_outcome(payload)
                if execution_request["operation"] == "repair"
                else ActivityOutcome.SUCCEEDED
            )
            if (
                execution_request["operation"] == "driver"
                and payload["returncode"] == 10
            ):
                from scripts.autotrain_pending import validate_pending

                outcome, _ = validate_pending(payload["pending"])
            return outcome, payload
        except (ValueError, KeyError, TypeError):
            pass
    return ActivityOutcome.UNKNOWN_FAILURE, None


def run_operation(runtime, request: dict, *, sequence: int, log_event) -> dict | None:
    """Lease, execute and validate one operation; failure is never a model loss."""
    from scripts.merge_verification_evidence import digest
    from scripts.autotrain_pending import verification_wait, recover_driver_request
    from slm_training.harness_core.activity_contract import (
        ActivityOutcome,
        WakeCondition,
    )
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
    request_path, output_path = attempt_dir / "request.json", attempt_dir / "result.json"
    execution_request = {**request, "lease": lease.model_dump(mode="json"),
                         "parent_event": runtime.store.verify_event_chain()[-1]["event_id"]}
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
    outcome, payload = interpret_operation_result(
        result, output_path, execution_request
    )
    # Reentry reconciles existing effects under the same grant; no result is inferred.
    if result.timed_out and not result.progress_stalled:
        outcome = ActivityOutcome.RETRY
    outputs = {"result.json": hashlib.sha256(output_path.read_bytes()).hexdigest()} if payload is not None else {}
    pending_wait = verification_wait(runtime, lease, payload)
    if payload and payload.get("returncode") == 10 and request["operation"] == "driver":
        from scripts.autotrain_pending import validate_pending

        pending_wait = validate_pending(payload["pending"])
    if pending_wait:
        outcome, wake = pending_wait
    else:
        wake = WakeCondition(
            predicate="original operation produces valid output",
            source="independent_repair_verification",
            identity_digest=identity,
        )
    try:
        if not pending_wait and outcome not in {
            ActivityOutcome.SUCCEEDED,
            ActivityOutcome.CANCELLED,
        }:
            from slm_training.autoresearch.heal.operation_recovery import (
                record_operation_failure,
            )

            record_operation_failure(runtime, lease, request, result,
                outcome=ActivityOutcome.WALL_BUDGET if outcome == ActivityOutcome.RETRY else outcome)
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
        runtime.reconcile()
        state = runtime.snapshot().get(activity_id)
        if state is not None and state.status == "succeeded":
            payload = _replay_committed_operation(runtime, request, state)
            _record_stale_operation_recovery(
                runtime, request, lease, state.status, log_event,
                committed_receipt_replayed=True,
            )
            return payload
        _record_stale_operation_recovery(
            runtime, request, lease,
            state.status if state is not None else "missing",
            log_event,
            committed_receipt_replayed=False,
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


def _replay_committed_operation(runtime, request, state):
    """Return only the exact, already-committed receipt for this operation."""
    from scripts.run_autotrain_supervisor import _source_identity

    validate_operation_identity(request, _source_identity, "before replay")
    completed = next(
        row
        for row in reversed(runtime.store.verify_event_chain())
        if row["event_type"] == "activity_transition"
        and row["experiment_id"] == state.spec.activity_id
        and row["detail"]["operation"] in {"finish", "reconcile_finish"}
    )
    attempt = completed["detail"]["lease"]["attempt_id"]
    path = runtime.store.root / state.spec.output_namespace / attempt / "result.json"
    if (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != state.outputs["result.json"]
    ):
        raise ValueError("committed operation artifact changed")
    return json.loads(path.read_text())["payload"]


def _record_stale_operation_recovery(
    runtime, request, lease, status, log_event, *, committed_receipt_replayed
):
    from scripts.merge_verification_evidence import digest

    detail = {
        "activity_id": lease.activity_id,
        "attempt_id": lease.attempt_id,
        "request_digest": digest(request),
        "lease_expires_at": lease.expires_at,
        "reconciled_status": status,
        "committed_receipt_replayed": committed_receipt_replayed,
    }
    event = runtime.store.append_event(
        "operation_stale_lease_reconciled",
        experiment_id=lease.activity_id,
        detail=detail,
        idempotency_key="operation-stale-lease:" + lease.token,
    )
    log_event({"event": "operation_stale_lease_reconciled", **detail,
               "event_id": event["event_id"]})


def repair_operation(request, *, cwd, root, loop_id, handle_hard_pending):
    from scripts.autotrain_controller_repair import (
        dispatch_controller_repairs,
    )

    payload = handle_hard_pending(
        request["hard_pending"],
        cwd=cwd,
        root=root,
        loop_id=loop_id,
        campaign_id=request["campaign_id"],
        max_heal_attempts=request["max_heal_attempts"],
        playbooks_enabled=request["playbooks_enabled"],
        log_event=lambda _: None,
    )
    payload["agent_repairs"] = dispatch_controller_repairs(
        request,
        cwd=cwd,
        root=root,
        loop_id=loop_id,
    )
    return payload


def operation_publication_scope(request, root, loop_id):
    from slm_training.autoresearch.runtime.activity_publication import (
        DelegatedPublisher,
    )
    from slm_training.harness_core.activity_contract import ActivityLease
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.harness_core.checkpoint_publication import (
        champion_publication_scope,
    )

    lease = ActivityLease.model_validate(request["lease"])
    journal = CampaignStore("runtime", root / "loops" / loop_id)
    publisher = DelegatedPublisher(journal, request["source_digest"])
    return champion_publication_scope(publisher, lease, loop_dir=root / "loops" / loop_id)


def validate_operation_identity(request, source_identity, boundary):
    """Bind execution/replay to current source and the measured runtime environment."""
    from scripts.merge_verification_evidence import digest, environment_identity

    if source_identity(Path(request["cwd"])) != request["source_digest"]:
        raise ValueError("operation source changed " + boundary)
    if digest(environment_identity()) != request["environment_digest"]:
        raise ValueError("operation environment changed " + boundary)


def operation_main(
    request_path: Path, output_path: Path, *, source_identity,
    load_continuous, handle_hard_pending, write_family_closures,
) -> int:
    """Run a pinned bounded controller child; untrusted repairs use heal.isolation."""
    from scripts.autotrain_operation_worker import operation_main as run_worker

    return run_worker(
        request_path,
        output_path,
        source_identity=source_identity,
        load_continuous=load_continuous,
        handle_hard_pending=handle_hard_pending,
        write_family_closures=write_family_closures,
    )
