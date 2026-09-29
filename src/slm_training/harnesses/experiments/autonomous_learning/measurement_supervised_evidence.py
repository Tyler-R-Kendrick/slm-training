"""Collect native supervisor evidence; no substituted child or scientific verdict."""

from pathlib import Path
import math

from scripts.autotrain_cycle_context import read_artifact
from scripts.autotrain_cycle_execution import completed_cycle_since
from scripts.autotrain_measurement import measurement_is_complete
from scripts.autotrain_metrics import read_paired_nll
from slm_training.evals.measurement_identity import content_digest
from slm_training.versioning import build_version_stamp

from .cli import _write
from .measurement_fixture_evidence import checked_arm, write_result_docs
from .measurement_fixture_plan import COMPONENTS, _read, _sha


def completed_operation(journal, campaign_id, before, source_digest):
    """Read a committed driver result from verified ordinary-supervisor history."""
    from scripts.autotrain_supervisor_operations import _operation_payload
    from slm_training.autoresearch.runtime.activity_projection import ActivityProjection

    events = journal.verify_event_chain()
    current = [event for event in events if event["event_id"] not in before]
    if not any(event["event_type"] == "supervisor_pass" for event in current):
        return None
    states = ActivityProjection(journal).read()
    for event in reversed(current):
        if event["event_type"] != "activity_transition":
            continue
        detail = event["detail"]
        if detail["operation"] not in {"finish", "reconcile_finish"}:
            continue
        state = states[event["experiment_id"]]
        if state.status != "succeeded" or "result.json" not in state.outputs:
            continue
        directory = journal.root / state.spec.output_namespace / detail["lease"]["attempt_id"]
        result_path = directory / "result.json"
        if _sha(result_path) != state.outputs["result.json"]:
            raise ValueError("committed supervisor result changed")
        request = _read(directory / "request.json")
        if request["operation"] != "driver":
            continue
        payload = _operation_payload(result_path, request)
        if payload.get("campaign_id") != campaign_id or payload["returncode"] != 0:
            continue
        if state.spec.source_digest != source_digest or request["source_digest"] != source_digest:
            raise ValueError("supervisor result belongs to another source")
        return payload
    return None


def stage_receipts(store, arms):
    receipts = {}
    for name in arms:
        terminal = [
            event
            for event in store.verify_event_chain()
            if event["event_type"] == "experiment_finished"
            and event["experiment_id"] == name
        ]
        if len(terminal) != 1:
            raise ValueError("supervised arm requires one actual terminal outcome")
        outcome = read_artifact(store, "outcomes", terminal[0]["artifact_sha256"])
        if outcome["status"] != "completed":
            raise ValueError("supervised arm outcome is incomplete")
        for index, stage in enumerate(outcome["stage_telemetry"]):
            if stage.get("executed") is False:
                continue
            seconds = stage.get("duration_seconds")
            if (
                not stage.get("command")
                or type(stage.get("exit_code")) is not int
                or stage["exit_code"] not in {0, 8, 10}
                or type(seconds) not in (int, float)
                or not math.isfinite(seconds)
                or seconds < 0
                or any(stage.get(key) for key in ("timed_out", "interrupted", "killed"))
            ):
                raise ValueError("supervised workload lacks a valid executed stage")
            receipts[f"{name}-{index}"] = {
                "outcome_sha256": terminal[0]["artifact_sha256"],
                "command_sha256": content_digest(stage["command"]),
                "exit_code": stage["exit_code"],
                "seconds": stage["duration_seconds"],
            }
    return receipts


def require_six_pairs(paired, failures):
    if (
        failures
        or paired is None
        or any(len(paired.get(name, {})) != 6 for name in ("control", "candidate"))
    ):
        raise ValueError("supervised fixture lacks all six actual paired observations")


def collect_supervised(store, plan, operation):
    """Require actual terminal driver/arm evidence before publishing a fixture report."""
    if (
        type(operation.get("returncode")) is not int
        or operation["returncode"] != 0
        or operation.get("campaign_id") != store.campaign_id
        or not isinstance(operation.get("completion"), dict)
    ):
        raise ValueError("supervised fixture lacks actual same-campaign completion")
    loop_id = store.load_campaign().loop_id
    completion = completed_cycle_since(
        Path.cwd(), store.root.parent, loop_id, store.campaign_id, frozenset()
    )
    if completion != operation["completion"]:
        raise ValueError("supervisor completion evidence changed")
    if _sha(store.root / "cycle_handoff.json") != operation["handoff_digest"]:
        raise ValueError("supervisor handoff evidence changed")
    arms = {name: checked_arm(plan, name) for name in plan["arms"]}
    paired, counts, failures = read_paired_nll(
        arms["control"]["run_dir"], arms["candidate"]["run_dir"]
    )
    require_six_pairs(paired, failures)
    decision = _read(store.root / "sdlc_delivery.json")
    if (
        not measurement_is_complete(decision)
        or decision["primary_metric"] != plan["primary"]["metric"]
    ):
        raise ValueError(
            "native driver disposition did not complete the locked endpoint"
        )
    receipts = stage_receipts(store, arms)
    manifest_sha256s = {
        name: arm["manifest_sha256"] for name, arm in plan["arms"].items()
    }
    identity = {
        "source_digest": plan["execution_context"]["source"],
        "plan_sha256": content_digest(plan),
        "operation_sha256": content_digest(operation),
        "completion_sha256": content_digest(operation["completion"]),
        "handoff_sha256": operation["handoff_digest"],
        "manifest_sha256s": manifest_sha256s,
    }
    result = {
        "schema": "supervised_measurement_result/v1",
        "source_digest": identity["source_digest"],
        "execution_identity": {**identity, "identity_sha256": content_digest(identity)},
        "operation": operation,
        "selection": plan["inputs"]["selection"],
        "manifest_sha256s": manifest_sha256s,
        "decision": decision,
        "paired_counts": counts,
        "command_receipts": receipts,
        "charged_child_seconds": sum(row["seconds"] for row in receipts.values()),
        "arms": {
            name: {
                "agentv_artifacts": arm["agentv_artifacts"],
                "trainable_parameters": plan["inputs"]["arms"][name][
                    "trainable_parameters"
                ],
                "scoreboard_sha256": _sha(arm["run_dir"] / "scoreboard.json"),
                "loss_report_sha256": _sha(arm["run_dir"] / "loss_suites.json"),
            }
            for name, arm in arms.items()
        },
        "evidence_class": "actual_supervisor_driver_cmd_run_loss_decode_agentv_disposition",
        "preparation": "controlled prelocked public fixture, not autonomous proposal selection",
        "new_training": False,
        "diagnostic_complete": True,
        "decoded_probe_complete": True,
        "confirmation_complete": False,
        "promotion_allowed": False,
        "ship_eligible": False,
        "version_stamp": build_version_stamp(*COMPONENTS),
    }
    return _persist_result(store, result)


def _persist_result(store, result):
    """Replay the committed result after a crash; never mint another observation."""
    previous = [event for event in store.verify_event_chain()
                if event["event_type"] == "supervised_measurement_consumed"]
    if previous:
        if len(previous) != 1:
            raise ValueError("ambiguous supervised measurement consumption")
        retained = read_artifact(store, "supervised_measurement_result", previous[0]["artifact_sha256"])
        result["version_stamp"]["stamped_at"] = retained["version_stamp"]["stamped_at"]
        if content_digest(result) != content_digest(retained):
            raise ValueError("committed supervised measurement changed")
    artifact = store.write_artifact("supervised_measurement_result", result)
    store.append_event(
        "supervised_measurement_consumed",
        artifact_sha256=artifact.stem,
        idempotency_key="supervised-measurement-consumed",
    )
    _write(store.root / "supervised_measurement_result.json", result)
    write_result_docs(store.campaign_id, result)
    return result


def run_supervised(store):
    """Exercise the public supervisor, including inspection and queue dispatch."""
    from dataclasses import asdict
    from slm_training.harness_core.bounded_process import run_bounded_process
    from slm_training.levers import INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS
    from slm_training.autoresearch.storage import CampaignStore, _sha as payload_sha
    from . import measurement_bundle as bundle
    import sys
    import time

    started = time.monotonic()
    plan = bundle.load_contract(store, store.root / "supervised_measurement.json")
    campaign = store.load_campaign()
    loop_id = campaign.loop_id
    journal = CampaignStore("runtime", store.root.parent / "loops" / loop_id)
    recovered = completed_operation(journal, store.campaign_id, frozenset(),
                                    plan["execution_context"]["source"])
    if recovered is not None:
        collect_supervised(store, plan, recovered)
        return dict(payload=recovered, contract_sha256=payload_sha(plan),
                    returncode=0, invocation_sha256=None, recovered=True)
    before = {event["event_id"] for event in journal.verify_event_chain()}
    grant = campaign.budget.continuation_grant
    grant_args = ["--continuation-grant", grant.model_dump_json()] if grant else []
    # Leave derived cleanup/finalization headroom inside the outer command cap.
    seconds = INTERRUPT_AFTER_SECONDS - (time.monotonic() - started) - 2 * KILL_GRACE_SECONDS
    if seconds <= 0:
        raise TimeoutError("supervised measurement preparation exhausted invocation budget")
    result = run_bounded_process(
        [sys.executable, "-m", "scripts.run_autotrain_supervisor",
         "--root", str(store.root.parent), "--loop-id", loop_id,
         "--train-version", plan["inputs"]["train_version"], "--max-cycles", "1", *grant_args],
        cwd=Path.cwd(), interrupt_after_seconds=seconds,
        kill_grace_seconds=KILL_GRACE_SECONDS,
    )
    receipt = store.write_artifact("supervised_measurement_invocation", asdict(result))
    store.append_event("supervised_measurement_invoked", artifact_sha256=receipt.stem)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    complete = result.returncode == 0 and not any((
        result.timed_out, result.interrupted, result.killed, result.cancelled,
        result.progress_stalled,
    ))
    operation = None
    if complete:
        bundle.load_contract(store, store.root / "supervised_measurement.json")
        operation = completed_operation(journal, store.campaign_id, before,
                                        plan["execution_context"]["source"])
        if operation is not None:
            collect_supervised(store, plan, operation)
    return dict(payload=operation, contract_sha256=payload_sha(plan),
                returncode=0 if operation is not None else (10 if complete else 1),
                invocation_sha256=receipt.stem)
