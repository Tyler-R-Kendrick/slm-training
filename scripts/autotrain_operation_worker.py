"""Child-side dispatcher for pinned supervisor operations."""

from __future__ import annotations

import json
from pathlib import Path


def operation_main(
    request_path: Path,
    output_path: Path,
    *,
    source_identity,
    load_continuous,
    handle_hard_pending,
    write_family_closures,
) -> int:
    """Run a pinned bounded controller child; untrusted repairs use heal.isolation."""
    from scripts.merge_verification_evidence import digest
    from scripts.autotrain_supervisor_operations import (
        operation_publication_scope,
        repair_operation,
        validate_operation_identity,
    )
    from slm_training.autoresearch.storage import CampaignStore

    request = json.loads(request_path.read_text())
    cwd, root = Path(request["cwd"]), Path(request["root"])
    validate_operation_identity(request, source_identity, "before launch")
    operation = request["operation"]
    continuous = load_continuous() if operation in {"inspect", "promotion_eval", "driver"} else None
    loop_id = request["loop_id"]
    if operation == "inspect":
        with operation_publication_scope(request, root, loop_id):
            report = continuous.self_heal_unblock_loop(
                cwd=cwd, root=root, loop_id=loop_id,
                campaign_id=loop_id if request.get("locked_diagnostic") else None,
                locked_plan=bool(request.get("locked_diagnostic")),
            )
            campaign_id = (loop_id if request.get("locked_diagnostic") else
                           continuous._latest_cycle(root, loop_id)[1])
            pending_promotion = False
            if campaign_id and not request.get("locked_diagnostic"):
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
                "parked": None if request.get("locked_diagnostic") else continuous._check_regime_parked(
                    root=root, loop_id=loop_id, cwd=cwd),
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
        from scripts.autotrain_cycle_execution import driver_operation

        payload = driver_operation(request, continuous, cwd, root, loop_id)
    elif operation == "closeout":
        events = []
        with operation_publication_scope(request, root, loop_id):
            write_family_closures(events.append)
        payload = {"events": events}
    else:
        raise ValueError("unsupported supervisor operation")
    validate_operation_identity(request, source_identity, "while running")
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
