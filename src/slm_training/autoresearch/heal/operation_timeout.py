"""Narrow host-authorized successor for a historically misclassified timeout.

Host CLI authorization is authority; receipt hashes only bind its exact evidence.
Never infer operator intent from an empty cancellation reason or signal number.
"""

import hashlib
import json
from pathlib import Path

from slm_training.harness_core.activity_contract import contract_digest
from slm_training.harness_core.host_process_identity import process_identity


def _bound_json(reference):
    path = Path(reference["path"])
    raw = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("timeout evidence changed")
    return json.loads(raw)


def verified_timeout_handoff(runtime, authorization, *, cwd):
    with runtime._transaction():
        proof, identity = authorization
        if (proof["schema_version"] != "host_timeout_recovery/v1"
                or proof["cause"] != "host_run_timeout"
                or Path(proof["journal_root"]).resolve() != runtime.store.root.resolve()
                or contract_digest(proof) != identity):
            raise ValueError("timeout recovery authority binding mismatch")
        events = runtime.store.verify_event_chain()
        cancelled = next(e for e in events if e["event_id"] == proof["cancel_event_id"])
        state = runtime.snapshot()[cancelled["experiment_id"]]
        if (state.status != "cancelled" or state.lease is not None
                or contract_digest(state) != proof["state_digest"]
                or cancelled["event_type"] != "activity_transition"
                or cancelled["detail"]["operation"] != "finish"
                or cancelled["detail"]["outcome"] != "cancelled"
                or cancelled["detail"].get("cancel_reason")
                or cancelled["detail"]["sequence"] != state.sequence):
            raise ValueError("timeout recovery cancellation changed")
        _check_successor_consumption(events, state.spec.activity_id, identity)
        lease = cancelled["detail"]["lease"]
        _dead_process(lease["owner_identity"])
        _dead_process(state.worker_identity)
        _host_receipts(proof, state)
        rows = [e for e in events if e["event_type"] == "operation_repair_requested"
                and e["experiment_id"] == state.spec.activity_id]
        original = rows[-1]["detail"]["request"]
        if (contract_digest(original) != proof["request_digest"]
                or state.spec.input_digest != proof["request_digest"]
                or Path(original["cwd"]).resolve() != Path(cwd).resolve()):
            raise ValueError("timeout original request changed")
        yielded = next(e for e in events if e["event_id"] == proof["yield_event_id"])
        claim = next(e for e in events if e["event_type"] == "activity_transition"
                     and e["detail"]["operation"] == "claim" and e["detail"].get("lease") == lease)
        _cancellation_provenance(events, claim, proof)
        if not (events.index(claim) < events.index(yielded) < events.index(cancelled)):
            raise ValueError("timeout yield outside cancelled lease")
        if yielded["event_type"] != "driver_yielded":
            raise ValueError("timeout recovery requires durable driver yield")
        path = runtime.store.root / "artifacts/driver_pending" / (yielded["artifact_sha256"] + ".json")
        pending = json.loads(path.read_text())
        from slm_training.harness_core.driver_pending import validate_pending
        from slm_training.harness_core.activity_contract import ActivityOutcome

        outcome, _ = validate_pending(pending)
        if (contract_digest(pending) != yielded["artifact_sha256"]
                or yielded["detail"].get("pending_digest") != yielded["artifact_sha256"]
                or outcome != ActivityOutcome.YIELDED):
            raise ValueError("timeout yielded artifact changed")
        _pending_binding(runtime.store, original, pending)
        validate_timeout_cursor(original, pending)
        return dict(schema_version="operation_timeout_handoff/v1", publication_id=identity,
                    activation_id=identity, resume_activity_id=state.spec.activity_id,
                    resume_campaign_id=pending["campaign_id"],
                    successor_execution=str(Path(cwd).resolve()), source_digest=state.spec.source_digest,
                    timeout_authorization=proof, pending=pending)


def _dead_process(identity):
    if identity and process_identity(int(identity.split(":")[1])) == identity:
        raise ValueError("timeout predecessor still alive")


def _host_receipts(proof, state):
    from slm_training.levers import INTERRUPT_AFTER_SECONDS

    receipt = _bound_json(proof["host_result"])
    preflight = _bound_json(proof["host_preflight"])
    previous = preflight["retained_activity"]
    if (receipt["returncode"] != 124 or receipt["lock_timeout"]
            or not receipt["preflight_passed"] or receipt["elapsed_seconds"] < INTERRUPT_AFTER_SECONDS
            or Path(receipt["command"][-1]).resolve() != Path(proof["host_result"]["path"]).parent.resolve()
            or Path(proof["host_preflight"]["path"]).parent.resolve()
            != Path(proof["host_result"]["path"]).parent.resolve()
            or previous["spec"] != state.spec.model_dump(mode="json")
            or previous["attempts"] + 1 != state.attempts
            or previous["charged_seconds"] > state.charged_seconds):
        raise ValueError("host timeout receipt does not bind cancelled attempt")


def _cancellation_provenance(events, claim, proof):
    subsequent = events[events.index(claim):]
    if any(e["event_type"] == "supervisor_stop_requested" and
           e["detail"].get("cause") == "explicit_cancel" for e in subsequent):
        raise ValueError("explicit user cancellation cannot recover as timeout")
    legacy = [e["event_id"] for e in subsequent if e["event_type"] == "activity_transition"
              and e["detail"].get("operation") == "cancel"
              and e["detail"].get("cancel_reason") == "explicit supervisor stop"]
    if legacy != proof["legacy_cancel_event_ids"]:
        raise ValueError("legacy ambiguous cancellation requires exact host authorization")
    if any(e["event_type"] == "activity_transition" and e["detail"].get("operation") == "cancel"
           and e["detail"].get("cancel_reason") not in {"explicit supervisor stop", ""}
           for e in subsequent):
        raise ValueError("additional cancellation provenance blocks timeout recovery")


def _pending_binding(journal, original, pending):
    from scripts.autotrain_cycle_lock import active_reference

    root, loop = Path(original["root"]).resolve(), original["loop_id"]
    expected = root / "loops" / loop / "runtime"
    reference = active_reference(journal)
    if (journal.root.resolve() != expected or Path(loop).name != loop
            or reference != {"campaign_id": pending["campaign_id"], "input_digest": pending["input_digest"]}):
        raise ValueError("timeout yield does not belong to original active driver")
    argv = original["driver_argv"]
    path = Path(argv[argv.index("--locked-preregistration") + 1])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != argv[argv.index("--locked-prereg-sha256") + 1]:
        raise ValueError("timeout preregistration changed")
    plan = json.loads(raw)
    if (plan["campaign_id"] != pending["campaign_id"] or Path(plan["campaign_root"]).resolve() != root
            or Path(plan["source_path"]).resolve() != Path(original["cwd"]).resolve()
            or plan["source_digest"] != original["source_digest"]):
        raise ValueError("timeout preregistration belongs to another workload")


def validate_timeout_cursor(original, pending):
    from scripts.autotrain_cycle_lock import load_context, read_artifact
    from types import SimpleNamespace
    from scripts.autotrain_cycle_context import verify_inputs, CycleJournal
    from scripts.autotrain_cursor_reconcile import committed_prefix_reconcilable
    from slm_training.autoresearch.storage import CampaignStore

    store = CampaignStore(pending["campaign_id"], Path(original["root"]))
    events = store.verify_event_chain()
    if not any(e["event_type"] == "driver_cycle_locked" for e in events):
        raise ValueError("missing original driver lock")
    value = load_context(store, pending["input_digest"])
    verify_inputs(store, Path(original["cwd"]), value)
    latest = [e for e in events if e["event_type"] == "driver_cycle_checkpoint"][-1]
    if latest["artifact_sha256"] != pending["wake"]["identity_digest"]:
        raise ValueError("timeout cursor changed")
    state = read_artifact(store, "driver_cycle_state", latest["artifact_sha256"])
    CycleJournal._validate(SimpleNamespace(state=state, value=value))
    if (state.get("inflight") is not None or state.get("repair_required")) and not (
            committed_prefix_reconcilable(store, value, state)):
        raise ValueError("timeout cursor requires independent reconciliation")


def _check_successor_consumption(events, activity, identity):
    # Planning already reserves this predecessor remainder. A crash before
    # activation must not permit another authorized proof to mint it again.
    plans = [e["detail"]["handoff"] for e in events
             if e["event_type"] == "operation_successor_planned"
             and e["detail"]["handoff"]["resume_activity_id"] == activity]
    if any(p.get("schema_version") != "operation_timeout_handoff/v1"
           or p.get("activation_id") != identity for p in plans):
        raise ValueError("timeout predecessor already consumed by successor plan")
