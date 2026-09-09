"""Fenced, resource-charged independent readiness recheck of a waiting driver."""

import hashlib
import json
import sys
import time
from pathlib import Path

from slm_training.autoresearch.storage import CampaignStore, _sha
from slm_training.harness_core.activity_contract import ActivityOutcome, ActivitySpec, WakeCondition, contract_digest
from slm_training.harness_core.bounded_process import ProcessOutcome


def _probe_request(runtime, common, job, repaired):
    readiness = job.get("probe") or job["payload"]["readiness"]
    store = CampaignStore(readiness["input_campaign_id"], Path(common["root"]))
    kind = readiness.get("input_kind", "matrix_readiness_inputs")
    if kind not in {"matrix_readiness_inputs", "driver_pending_inputs"}:
        raise ValueError("unsupported frozen driver probe input")
    path = store.root / "artifacts" / kind / (readiness["input_artifact_sha256"] + ".json")
    inputs = json.loads(path.read_text())
    if _sha(inputs) != readiness["input_artifact_sha256"] or str(path) != readiness["input_path"]:
        raise ValueError("frozen pending input path/digest mismatch")
    if inputs["cwd"] != common["cwd"] or inputs["loop_id"] != common["loop_id"]:
        raise ValueError("pending readiness belongs to another runtime")
    argv = [sys.executable, "-m", "scripts.autotrain_readiness_probe", "--input", str(path),
            "--input-digest", path.stem, "--root", common["root"]]
    activity = readiness.get("readiness_campaign_id")
    if activity:
        argv += ["--request-digest", readiness["request_digest"], "--readiness-campaign-id", activity]
        store = CampaignStore(activity, Path(common["root"]))
    trigger = contract_digest({"dependency_events": [e["event_id"] for e in store.verify_event_chain()],
                               "repair_result": repaired, "repair_config": common.get("repair_config_digest")})
    return argv, trigger


def _probe_state(runtime, common, job, trigger):
    identity = job["pending_digest"]
    state = runtime.register(ActivitySpec(activity_id="driver-readiness-" + identity,
        family=common["loop_id"], kind="verify", source_digest=common["source_digest"],
        environment_digest=common["environment_digest"], input_digest=identity,
        output_namespace="attempts/driver-readiness-" + identity))
    checks = [e for e in runtime.store.verify_event_chain() if e["event_type"] == "driver_readiness_checked"
              and e["experiment_id"] == job["activity_id"]]
    if state.status == "waiting_dependency" and checks:
        previous = checks[-1]["detail"]
        if previous["trigger"] == trigger and not previous["ready"]:
            return None
        # Only controller-owned dependency advancement (or crash reconciliation
        # of a positive probe) wakes verification. Its finite grant is unchanged.
        runtime.wake(state.spec.activity_id, evidence=state.wake)
    return runtime.claim_next(activity_id=state.spec.activity_id, capabilities={"local_process"})


def _publish_probe(common, job, evidence):
    readiness = job.get("probe") or job["payload"]["readiness"]
    if evidence.get("candidate"):
        activity = CampaignStore(readiness["readiness_campaign_id"], Path(common["root"]))
        observed = evidence["observation"]
        if observed.get("request_sha256") != readiness["request_digest"] or observed.get("ready") is not True:
            raise ValueError("independent probe request identity mismatch")
        artifact = activity.write_artifact("data_readiness", observed)
        key = "data-ready:" + readiness["request_digest"] + ":" + evidence["candidate"]["manifest_sha256"]
        if not any(e.get("idempotency_key") == key for e in activity.verify_event_chain()):
            activity.append_event("screening_successor_ready", artifact_sha256=artifact.stem,
                detail={"request_sha256": readiness["request_digest"], "loop_id": common["loop_id"],
                        "data_root": common["cwd"], "candidate": evidence["candidate"], "measurement_complete": False},
                idempotency_key=key)


def _accept_probe(runtime, job, evidence):
    wake = WakeCondition.model_validate(job["payload"]["wake"])
    state = runtime.snapshot()[job["activity_id"]]
    if state.status not in {"waiting_dependency", "waiting_capability"} or state.wake != wake:
        raise ValueError("original driver no longer owns this wait")
    runtime.wake(job["activity_id"], evidence=wake)
    artifact = runtime.store.write_artifact("driver_readiness_verified", evidence)
    runtime.store.append_event("driver_pending_resolved", experiment_id=job["activity_id"],
        artifact_sha256=artifact.stem, detail={"pending_digest": job["pending_digest"]},
        idempotency_key="driver-pending-resolved:" + job["activity_id"] + ":" + job["pending_digest"])


def recheck_driver_pending(runtime, common, job, *, repaired, log_event):
    argv, trigger = _probe_request(runtime, common, job, repaired)
    lease = _probe_state(runtime, common, job, trigger)
    if lease is None:
        return False
    started = time.monotonic()
    result = runtime.run(lease, argv, cwd=Path(common["cwd"]))
    probe = job.get("probe") or job["payload"]["readiness"]
    evidence = {"ready": False, "reason": "bounded_readiness_probe_incomplete"}
    if result.outcome == ProcessOutcome.COMPLETED and not result.stdout_truncated:
        try:
            evidence = json.loads(result.stdout.splitlines()[-1])
            if (evidence.get("schema_version") != "matrix_readiness_probe/v1"
                    or evidence.get("input_digest") != probe["input_artifact_sha256"]
                    or evidence.get("ready") is not True or result.returncode != 0):
                evidence["ready"] = False
        except (IndexError, ValueError, AttributeError, TypeError):
            evidence = {"ready": False, "reason": "malformed_readiness_probe_output"}
    path = runtime.attempt_dir(lease) / "readiness.json"
    runtime.store._replace_durable(path, json.dumps(evidence, sort_keys=True))
    runtime.store.append_event("driver_readiness_checked", experiment_id=job["activity_id"],
        detail={"trigger": trigger, "ready": evidence["ready"], "attempt_id": lease.attempt_id})
    if evidence["ready"]:
        with runtime.publication(lease):
            _publish_probe(common, job, evidence)
    runtime.finish(lease, outcome=ActivityOutcome.DEPENDENCY,
        outputs={path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
        spent_seconds=time.monotonic() - started, wake=WakeCondition.model_validate(job["payload"]["wake"]))
    if evidence["ready"]:
        _accept_probe(runtime, job, evidence)
    log_event({"event": "driver_readiness_rechecked", "activity_id": job["activity_id"], "ready": evidence["ready"]})
    return evidence["ready"]
