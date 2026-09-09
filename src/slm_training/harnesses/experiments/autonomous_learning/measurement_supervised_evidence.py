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
    result = {
        "schema": "supervised_measurement_result/v1",
        "operation": operation,
        "selection": plan["inputs"]["selection"],
        "manifest_sha256s": {
            name: arm["manifest_sha256"] for name, arm in plan["arms"].items()
        },
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
    artifact = store.write_artifact("supervised_measurement_result", result)
    store.append_event(
        "supervised_measurement_consumed",
        artifact_sha256=artifact.stem,
        idempotency_key="supervised-measurement-consumed",
    )
    _write(store.root / "supervised_measurement_result.json", result)
    write_result_docs(store.campaign_id, result)
    return result
