"""Bounded trusted-controller operations; untrusted repairs use heal.isolation."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from scripts.autotrain_cycle_execution import driver_operation as driver_operation


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
    if result.timed_out or result.progress_stalled:
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
    from scripts.autotrain_pending import verification_wait
    from slm_training.harness_core.activity_contract import (
        ActivityOutcome,
        WakeCondition,
    )

    identity = digest(request)
    state = _operation_state(runtime, request, sequence, identity)
    activity_id = state.spec.activity_id
    if state.status == "succeeded":
        # Reconcile a committed effect without executing it again.
        completed = next(
            row
            for row in reversed(runtime.store.verify_event_chain())
            if row["event_type"] == "activity_transition"
            and row["experiment_id"] == activity_id
            and row["detail"]["operation"] in {"finish", "reconcile_finish"}
        )
        attempt = completed["detail"]["lease"]["attempt_id"]
        path = (
            runtime.store.root / state.spec.output_namespace / attempt / "result.json"
        )
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != state.outputs["result.json"]
        ):
            raise ValueError("committed operation artifact changed")
        return json.loads(path.read_text())["payload"]
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
    result = runtime.run(
        lease,
        [
            sys.executable,
            "-m",
            "scripts.run_autotrain_supervisor",
            "--operation-request",
            str(request_path),
            "--operation-output",
            str(output_path),
        ],
        cwd=Path(request["cwd"]),
    )
    outcome, payload = interpret_operation_result(
        result, output_path, execution_request
    )
    outputs = (
        {"result.json": hashlib.sha256(output_path.read_bytes()).hexdigest()}
        if payload is not None
        else {}
    )
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
    if not pending_wait and outcome not in {
        ActivityOutcome.SUCCEEDED,
        ActivityOutcome.CANCELLED,
    }:
        from slm_training.autoresearch.heal.operation_recovery import (
            record_operation_failure,
        )

        record_operation_failure(runtime, lease, request, result, outcome=outcome)
    runtime.finish(
        lease,
        outcome=outcome,
        outputs=outputs,
        spent_seconds=time.monotonic() - started,
        wake=wake
        if outcome not in {ActivityOutcome.SUCCEEDED, ActivityOutcome.CANCELLED}
        else None,
    )
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
    return champion_publication_scope(
        publisher, lease, loop_dir=root / "loops" / loop_id
    )


def operation_main(
    request_path: Path,
    output_path: Path,
    *,
    source_identity,
    load_continuous,
    handle_hard_pending,
    write_family_closures,
) -> int:
    """Run one pinned trusted-controller operation in a bounded child.

    This is not the untrusted repair sandbox. Repair code uses heal.isolation.
    Heavy imports, playbooks and their failure paths stay outside the parent.
    """
    from scripts.merge_verification_evidence import digest
    from slm_training.autoresearch.storage import CampaignStore

    request = json.loads(request_path.read_text())
    cwd, root = Path(request["cwd"]), Path(request["root"])
    if source_identity(cwd) != request["source_digest"]:
        raise ValueError("operation source changed before launch")
    continuous = load_continuous()
    operation = request["operation"]
    loop_id = request["loop_id"]
    if operation == "inspect":
        with operation_publication_scope(request, root, loop_id):
            report = continuous.self_heal_unblock_loop(
                cwd=cwd, root=root, loop_id=loop_id
            )
            campaign_id = continuous._latest_cycle(root, loop_id)[1]
            pending_promotion = False
            if campaign_id:
                from scripts.autotrain_promotion_chunks import load_ledger
                from scripts.autotrain_promotion_finalize import finalization_pending

                store = CampaignStore(campaign_id, root)
                ledger = load_ledger(store)
                pending_promotion = bool(
                    ledger
                    and (
                        finalization_pending(store, ledger)
                        or any(
                            arm["status"] in {"pending", "running", "invocation_yield"}
                            for arm in ledger["arms"].values()
                        )
                    )
                )
            payload = {
                "report": report,
                "parked": continuous._check_regime_parked(
                    root=root, loop_id=loop_id, cwd=cwd
                ),
                "campaign_id": campaign_id,
                "promotion_pending": pending_promotion,
            }
    elif operation == "promotion_eval":
        from scripts.autotrain_promotion_chunks import resume_chunks
        from scripts.autotrain_promotion_finalize import (
            finalize_promotion,
            finalization_pending,
        )

        with operation_publication_scope(request, root, loop_id):
            ledger = resume_chunks(
                {
                    "cwd": cwd,
                    "root": root,
                    "loop_id": loop_id,
                    "campaign_id": request["campaign_id"],
                },
                stage_runner=continuous._stage_command,
                scoreboard=continuous._promotion_scoreboard_state,
            )
            store = CampaignStore(request["campaign_id"], root)
            if finalization_pending(store, ledger):
                finalize_promotion(store, cwd, continuous, ledger)
        payload = {"campaign_id": request["campaign_id"], "ledger": ledger}
    elif operation == "repair":
        with operation_publication_scope(request, root, loop_id):
            payload = repair_operation(
                request,
                cwd=cwd,
                root=root,
                loop_id=loop_id,
                handle_hard_pending=handle_hard_pending,
            )
    elif operation == "driver":
        payload = driver_operation(request, continuous, cwd, root, loop_id)
    elif operation == "closeout":
        events = []
        with operation_publication_scope(request, root, loop_id):
            write_family_closures(events.append)
        payload = {"events": events}
    else:
        raise ValueError("unsupported supervisor operation")
    if source_identity(cwd) != request["source_digest"]:
        raise ValueError("operation source changed while running")
    CampaignStore._replace_durable(
        output_path,
        json.dumps(
            {
                "schema_version": "supervisor_operation/v1",
                "request_digest": digest(request),
                "operation": operation,
                "payload": payload,
            },
            sort_keys=True,
            allow_nan=False,
        ),
    )
    return 0
