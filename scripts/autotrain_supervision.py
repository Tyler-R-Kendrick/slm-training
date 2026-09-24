"""Supervisor queue policy over canonical activity outcomes, not scientific verdicts."""

from __future__ import annotations

import json
import time
from pathlib import Path

_MAX_HARD_RETRY_SECONDS = 60.0
_NO_CAMPAIGN_THRESHOLD = 5
_STALL_KIND = "loop_stalled_no_campaign"


def watchdog_no_campaign(
    *,
    root: Path,
    loop_id: str,
    passes_without_campaign: int,
    total_no_campaign_passes: int,
    campaign_id: str | None,
    log_event,
    threshold: int = _NO_CAMPAIGN_THRESHOLD,
) -> float | None:
    """Return governed backoff above threshold, retaining fingerprint and counts."""
    if passes_without_campaign < threshold:
        return None
    try:
        from slm_training.autoresearch.heal.escalation import EscalationLedger

        ledger = EscalationLedger.load(root, loop_id)
        record = ledger.observe(
            kind=_STALL_KIND,
            reason="supervised passes without a new campaign",
            blocker_class="unknown",
            campaign_id=campaign_id or "unknown",
            owner_skill="autotrain",
        )
        ledger.escalate(
            record.fingerprint,
            note=(
                f"supervisor watchdog: consecutive={passes_without_campaign} "
                f"total_no_campaign_passes={total_no_campaign_passes} "
                f"threshold={threshold}"
            ),
        )
        ledger.save()
        backoff = float(ledger.records[record.fingerprint].next_backoff_seconds)
        log_event(
            {
                "event": _STALL_KIND,
                "passes": passes_without_campaign,
                "total_no_campaign_passes": total_no_campaign_passes,
                "seen_count": record.seen_count,
                "next_backoff_seconds": backoff,
            }
        )
        return backoff
    except Exception as exc:  # noqa: BLE001
        log_event({"event": "watchdog_error", "error": repr(exc)})
        return None


def handle_pending(args, runtime, common, inspection, cycle, log_event, run_operation):
    report = inspection["report"]
    register_delivery_waits(
        runtime, common, report.get("delivery_waits", []), log_event
    )
    if inspection.get("promotion_pending"):
        run_operation(
            runtime,
            {
                **common,
                "operation": "promotion_eval",
                "campaign_id": inspection["campaign_id"],
            },
            sequence=cycle,
            log_event=log_event,
        )
        return "wait"
    parked = inspection["parked"]
    if parked and not report.get("hard_pending"):
        log_event(
            {
                "event": "regime_parked",
                "status": parked,
                "cycle": cycle,
                "soft_healed": list(report.get("soft_healed") or []),
            }
        )
        if args.exit_on_park:
            return "stop"
        # Park waits for its existing unblock/identity predicate.
        runtime.cancel_event.wait(max(1.0, float(args.park_backoff_seconds)))
        return "wait"
    if report.get("hard_pending"):
        outcome = (
            run_operation(
                runtime,
                {
                    **common,
                    "operation": "repair",
                    "hard_pending": report["hard_pending"],
                    "campaign_id": str(report.get("predecessor_campaign_id") or ""),
                    "max_heal_attempts": int(args.max_heal_attempts),
                    "playbooks_enabled": not args.no_playbooks,
                },
                sequence=cycle,
                log_event=log_event,
            )
            or {}
        )
        log_event(
            {
                "event": "hard_pending_heal",
                "cycle": cycle,
                "hard_pending": report["hard_pending"],
                **outcome,
            }
        )
        if outcome.get("any_healed"):
            # A shifting blocker can mint fingerprints; sleep to bound heal-spin.
            runtime.cancel_event.wait(max(0.5, float(args.soft_backoff_seconds)))
            return "wait"
        runtime.cancel_event.wait(
            max(
                1.0,
                min(
                    _MAX_HARD_RETRY_SECONDS,
                    float(outcome.get("sleep_seconds") or args.hard_backoff_seconds),
                ),
            )
        )
        return "wait"

    return "run"


def register_delivery_waits(runtime, common, waits, log_event):
    """Reconcile scoped delivery independently of the trainer."""
    import hashlib
    from scripts.merge_verification_evidence import digest
    from slm_training.harness_core.activity_contract import ActivitySpec
    from slm_training.autoresearch.runtime.operations_control import consume_delivery, load_delivery_host

    path = Path(common["delivery_config"]) if common.get("delivery_config") else None
    if path and hashlib.sha256(path.read_bytes()).hexdigest() != common["delivery_config_digest"]:
        raise ValueError("delivery configuration changed")
    host = load_delivery_host(path)
    results = []
    for wait in waits:
        if wait.get("required_capability") != "authorized_github_connector_delivery":
            raise ValueError("unsupported delivery capability")
        kind = wait.get("kind", "document")
        if kind not in {"document", "workspace", "verified_repair_source"}:
            raise ValueError("unsupported delivery dependency kind")
        source_digest = common["source_digest"]
        if kind == "verified_repair_source":
            from slm_training.autoresearch.heal.repair_delivery import resolve_source_delivery
            source_digest = resolve_source_delivery(runtime.store, wait)["successor_source_digest"]
        artifact = runtime.store.write_artifact("delivery_requests", wait)
        activity_id = f"{kind}-delivery-" + digest(wait)[:24]
        existing = runtime.snapshot().get(activity_id)
        # A delivered repair survives its fresh-process source transition. The
        # writer still independently checks the current host/gate environment.
        environment = (existing.spec.environment_digest
                       if existing and kind == "verified_repair_source"
                       else common["environment_digest"])
        runtime.register(
            ActivitySpec(
                activity_id=activity_id,
                family=common["loop_id"],
                kind="delivery",
                source_digest=source_digest,
                environment_digest=environment,
                input_digest=digest(wait),
                output_namespace=f"attempts/{activity_id}",
                capabilities=("authorized_github_connector_delivery",),
            )
        )
        result = consume_delivery(runtime, activity_id, wait, host)
        results.append(result)
        log_event({"event": "delivery_wait", "activity_id": activity_id,
                   "request_artifact": artifact.stem, "state": result["state"],
                   "reason": result.get("reason")})
    return results


def post_cycle(
    args,
    runtime,
    common,
    cycle,
    after_campaign,
    run_operation,
    log_event,
    watchdog_backoff,
):
    # Post-cycle unblock regardless of exit code.
    run_operation(
        runtime,
        {**common, "operation": "closeout", "campaign_id": after_campaign},
        sequence=cycle,
        log_event=log_event,
    )
    try:
        observed = run_operation(
            runtime,
            {**common, "operation": "inspect"},
            sequence=cycle + 1,
            log_event=log_event,
        )
        report = observed["report"] if observed else {}
        log_event({"event": "post_cycle_unblock", "cycle": cycle, **report})
        if report.get("hard_pending"):
            sleep_seconds = max(1.0, float(args.hard_backoff_seconds))
        else:
            sleep_seconds = max(0.5, float(args.soft_backoff_seconds))
    except Exception as exc:  # noqa: BLE001
        log_event({"event": "post_cycle_unblock_error", "error": repr(exc)})
        sleep_seconds = max(1.0, float(args.soft_backoff_seconds))
    if watchdog_backoff is not None:
        # Honor escalation backoff when a no-campaign stall is proven.
        sleep_seconds = max(sleep_seconds, float(watchdog_backoff))
    return min(60.0, sleep_seconds)


def pre_cycle(runtime, common, cycle, log_event, run_operation):
    """Dispatch repair before importing a possibly broken inspection owner."""
    from slm_training.autoresearch.heal.operation_recovery import (
        pending_operation_repairs,
    )
    from scripts.autotrain_repair_activation import VerifiedRestart, recover_release
    from scripts.autotrain_verification import drain_source_verification
    from scripts.autotrain_pending import drain_driver_pending

    try:
        drain_source_verification(runtime, common, log_event, cycle=cycle)
        drain_driver_pending(runtime, common, cycle, log_event, run_operation)
        for repair_request in pending_operation_repairs(runtime)[:1]:
            runtime.store.append_event("operation_repair_serviced",
                experiment_id=repair_request["hard_pending"][0]["affected_activity_id"])
            run_operation(runtime, repair_request, sequence=cycle, log_event=log_event)
        recover_release(
            runtime,
            common,
            sequence=cycle,
            log_event=log_event,
            run_operation=run_operation,
        )
        inspection = run_operation(
            runtime,
            {**common, "operation": "inspect"},
            sequence=cycle,
            log_event=log_event,
        )
        if inspection is not None:
            log_event(
                {"event": "pre_cycle_unblock", "cycle": cycle, **inspection["report"]}
            )
        return inspection
    except VerifiedRestart:
        raise
    except Exception as exc:  # noqa: BLE001 — retained as a scoped operational failure
        log_event({"event": "pre_cycle_unblock_error", "error": repr(exc)})
        return None


def _driver_argv(args):
    cmd = ["--loop-id", args.loop_id, "--root", str(args.root),
           "--supervised", "--max-cycles", "1", "--train-version", args.train_version,
           "--steps", str(args.steps), "--primary-metric", args.primary_metric]
    if getattr(args, "continuation_grant", None):
        cmd.extend(["--continuation-grant", args.continuation_grant])
    return cmd


def observe_driver_progress(runtime, args, cycle, before, driver, log_event, watchdog):
    """Replay operational counters; a timeout/missing output is not progress."""
    prior = next((row["detail"] for row in reversed(runtime.store.verify_event_chain())
                  if row["event_type"] == "supervisor_driver_progress"), {})
    if prior.get("sequence") != cycle:
        after = driver["campaign_id"] if driver is not None else before
        progress = driver is not None and (after != before or bool(driver.get("completion")))
        prior = {"sequence": cycle, "campaign_id": after,
                 "passes_without_campaign": 0 if progress else prior.get("passes_without_campaign", 0) + 1,
                 "total_no_campaign_passes": prior.get("total_no_campaign_passes", 0) + int(not progress)}
        runtime.store.append_event("supervisor_driver_progress", detail=prior,
                                   idempotency_key=f"supervisor-driver-progress:{cycle}")
    return watchdog(root=Path(args.root).resolve(), loop_id=args.loop_id,
                    log_event=log_event, **{key: value for key, value in prior.items() if key != "sequence"})


def supervise(args, runtime, common: dict, *, run_operation, watchdog) -> int:
    root = Path(common["root"])
    log_dir = root / "loops" / args.loop_id
    log_dir.mkdir(parents=True, exist_ok=True)
    supervisor_log = log_dir / "supervisor.jsonl"

    def log_event(event: dict) -> None:
        event = {
            **event,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "loop_id": args.loop_id,
        }
        with supervisor_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, sort_keys=True) + "\n")
        print(json.dumps(event, sort_keys=True), flush=True)

    # Invocation count is operational only; it never changes a training recipe.
    prior = [
        row
        for row in runtime.store.verify_event_chain()
        if row["event_type"] == "supervisor_pass"
    ]
    cycle = len(prior)
    first_cycle = cycle
    while (
        not runtime.cancel_event.is_set()
        and (args.max_cycles == 0 or cycle - first_cycle < args.max_cycles)
        and (
            getattr(args, "stop_after_pass", None) is None
            or cycle < args.stop_after_pass
        )
    ):
        cycle += 1
        runtime.store.append_event(
            "supervisor_pass",
            detail={"sequence": cycle},
            idempotency_key=f"supervisor-pass:{cycle}",
        )
        inspection = pre_cycle(runtime, common, cycle, log_event, run_operation)
        if inspection is None:
            backoff = observe_driver_progress(
                runtime, args, cycle, None, None, log_event, watchdog)
            runtime.cancel_event.wait(min(
                60.0, max(1.0, args.hard_backoff_seconds, backoff or 0.0)))
            continue
        pending = handle_pending(
            args, runtime, common, inspection, cycle, log_event, run_operation
        )
        if pending == "stop":
            return 0
        if pending == "wait":
            continue

        cmd = _driver_argv(args)
        before_campaign = inspection["campaign_id"]
        log_event({"event": "start_driver", "cycle": cycle, "cmd": cmd})
        driver = run_operation(
            runtime,
            {
                **common,
                "operation": "driver",
                "driver_argv": cmd,
                "predecessor_campaign_id": before_campaign,
            },
            sequence=cycle,
            log_event=log_event,
        )
        if driver is None:
            backoff = observe_driver_progress(runtime, args, cycle, before_campaign, driver, log_event, watchdog)
            runtime.cancel_event.wait(min(60.0, max(1.0, args.hard_backoff_seconds, backoff or 0)))
            continue
        log_event({"event": "driver_exit", "cycle": cycle,
                   "returncode": driver["returncode"]})
        if driver["returncode"] == 10:
            from scripts.autotrain_pending import validate_pending

            outcome, wake = validate_pending(driver["pending"])
            log_event(
                {
                    "event": "driver_pending",
                    "cycle": cycle,
                    "outcome": outcome.value,
                    "wake": wake.model_dump(mode="json"),
                }
            )
            # A bounded continuation is not a finished campaign. Its activity
            # retains the original grant; neither closeout nor a new receipt
            # resets scientific stagnation or authorizes another trial.
            runtime.cancel_event.wait(min(60.0, max(0.5, args.soft_backoff_seconds)))
            continue
        after_campaign = driver["campaign_id"]
        watchdog_backoff = observe_driver_progress(
            runtime, args, cycle, before_campaign, driver, log_event, watchdog)
        runtime.cancel_event.wait(
            post_cycle(
                args,
                runtime,
                common,
                cycle,
                after_campaign,
                run_operation,
                log_event,
                watchdog_backoff,
            )
        )
    return 0
