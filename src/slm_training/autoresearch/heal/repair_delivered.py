"""Fenced delivered materialization, separate from historical repair acceptance."""

from __future__ import annotations

import hashlib
import json
import tempfile
from functools import reduce
from pathlib import Path

from slm_training.harness_core.activity_contract import ActivityEvent, contract_digest, reduce_activity
from slm_training.harness_core.execution_environment import verification_environment
from slm_training.harness_core.execution_release import (
    MARKER, _delivered_provenance, prepare_delivered_release, runtime_git_provenance,
    runtime_source_identity,
)

from .isolation_workspace import checked_path
from .repair_delivery import _artifact, resolve_source_delivery


def _base_handoff(store, wait):
    from .repair_release import verified_activation_handoff

    rows = [row["detail"]["handoff"] for row in store.verify_event_chain()
            if row["event_type"] == "repair_release_accepted"
            and row["detail"]["publication_id"] == wait["publication_id"]]
    if len(rows) != 1 or rows[0].get("source_delivery") != wait:
        raise ValueError("delivered_activation_repair_acceptance_missing")
    return verified_activation_handoff(store, rows[0])


def _environment_transition(runtime, accepted, execution):
    from scripts.merge_verification_evidence import digest, environment_identity

    current = environment_identity()
    original = runtime.snapshot()[accepted["resume_activity_id"]]
    before = digest(current)
    if before != original.spec.environment_digest:
        raise ValueError("delivered_activation_predecessor_environment_changed")
    variables = verification_environment()
    if digest(variables) != current["execution_environment_sha256"]:
        raise ValueError("delivered_activation_environment_capture_changed")
    variables["PYTHONPATH"] = str(execution / "src")
    variables.pop("PYTHONHOME", None)
    successor = {**current, "execution_environment_sha256": digest(variables)}
    return {"predecessor_digest": before, "successor_digest": digest(successor)}


def _reference(artifact, activation_id):
    return {"artifact_sha256": artifact.stem, "activation_id": activation_id}


def _record(runtime, lease, wait, reconciliation):
    from scripts.github_source_delivery import source_completion
    from .repair_release import _durable_tree, _sync

    from slm_training.harness_core.source_authority import authority_record, authority_reference, publish_source_authority

    store = runtime.store
    request = json.loads((runtime.attempt_dir(lease) / "request.json").read_text())
    if request.get("lease") != lease.model_dump(mode="json"):
        raise ValueError("delivered_activation_request_lease_mismatch")
    subject = resolve_source_delivery(store, wait)
    remote = source_completion(store, wait, reconciliation)
    accepted = _base_handoff(store, wait)
    reconciliation = {key: reconciliation[key] for key in ("verification_artifact", "receipt")}
    activation_id = contract_digest({"publication_id": wait["publication_id"],
                                    "receipt": Path(reconciliation["verification_artifact"]).stem,
                                    "request": contract_digest(request)})
    prior = [event for event in store.verify_event_chain()
             if event["event_type"] == "delivered_repair_activation_recorded"
             and event["detail"]["activation_id"] == activation_id]
    if prior:
        ref = {"artifact_sha256": prior[0]["artifact_sha256"], "activation_id": activation_id}
        payload = _validated_payload(store, ref)
        if payload["delivery_activity_id"] != lease.activity_id or payload["reconciliation"] != reconciliation:
            raise ValueError("delivered_activation_intent_conflict")
        authority = authority_record(runtime, lease, kind="repair_delivery", request=request,
                                     remote=remote, execution=payload["execution"])
        if authority_reference(authority) != payload["manifest"].get("source_authority"):
            raise ValueError("delivered_activation_authority_conflict")
        publish_source_authority(runtime, lease, authority)
        return ref
    source = Path(subject["successor_execution"])
    root = Path(subject["successor"]["release"]).parent.parent
    stage = Path(tempfile.mkdtemp(prefix="delivered-", dir=root))
    outputs = Path(json.loads((source / MARKER).read_text())["outputs"])
    destinations = (stage / "release", stage / "execution", outputs)
    transition = _environment_transition(runtime, accepted, destinations[1])
    authority = authority_record(runtime, lease, kind="repair_delivery", request=request,
                                 remote=remote, execution=destinations[1])
    manifest = prepare_delivered_release((source, subject["successor_source_digest"]),
        destinations, (remote["merge_commit_object"], remote["merge_sha"]),
        source_authority=authority_reference(authority))
    _durable_tree(stage)
    _sync(root)
    if _environment_transition(runtime, accepted, destinations[1]) != transition:
        raise ValueError("delivered_activation_environment_changed_during_copy")
    payload = {"schema_version": "delivered_repair_activation/v1", "activation_id": activation_id,
               "accepted_handoff": accepted, "wait": wait, "reconciliation": reconciliation,
               "delivery_activity_id": lease.activity_id, "environment_transition": transition,
               "execution": str(destinations[1]), "manifest": manifest}
    # Recheck the lease after potentially slow copies, before recording authority.
    with runtime.publication(lease):
        publish_source_authority(runtime, lease, authority)
        artifact = store.write_artifact("delivered_repair_activations", payload)
        store.append_event("delivered_repair_activation_recorded", artifact_sha256=artifact.stem,
            experiment_id=lease.activity_id, idempotency_key="delivered-activation:" + activation_id,
            detail={"activation_id": activation_id, "publication_id": wait["publication_id"]})
    return _reference(artifact, activation_id)


def record_delivered_activation(runtime, lease, wait, reconciliation):
    """Called during independent reconciliation, before delivery activity finish.

    Nested runtime.publication is supported. A recorded materialization alone
    never permits activation: resolve_delivered_activation also requires the
    committed delivery output carrying this exact reference.
    """
    with runtime.publication(lease) as state:
        if (state.spec.kind != "delivery" or state.spec.input_digest != contract_digest(wait)
                or state.spec.source_digest != resolve_source_delivery(runtime.store, wait)["successor_source_digest"]
                or "authorized_github_connector_delivery" not in state.spec.capabilities):
            raise ValueError("delivered_activation_delivery_lease_mismatch")
        return _record(runtime, lease, wait, reconciliation)


def _validated_payload(store, reference):
    from scripts.github_source_delivery import source_completion

    payload = _artifact(store, "delivered_repair_activations", reference["artifact_sha256"])
    expected_ref = {"artifact_sha256": reference["artifact_sha256"], "activation_id": payload["activation_id"]}
    expected_event = {"activation_id": payload["activation_id"], "publication_id": payload["wait"]["publication_id"]}
    if (reference != expected_ref or payload["schema_version"] != "delivered_repair_activation/v1"
            or not any(event["event_type"] == "delivered_repair_activation_recorded"
                       and event["artifact_sha256"] == reference["artifact_sha256"]
                       and event["experiment_id"] == payload["delivery_activity_id"]
                       and event["detail"] == expected_event for event in store.verify_event_chain())):
        raise ValueError("delivered_activation_not_recorded")
    if _base_handoff(store, payload["wait"]) != payload["accepted_handoff"]:
        raise ValueError("delivered_activation_accepted_handoff_changed")
    remote = source_completion(store, payload["wait"], payload["reconciliation"])
    accepted = payload["accepted_handoff"]
    expected = _delivered_provenance((Path(accepted["successor_execution"]), accepted["source_digest"]),
                                    (remote["merge_commit_object"], remote["merge_sha"]))
    execution = Path(payload["execution"])
    if (runtime_source_identity(execution) != accepted["source_digest"]
            or runtime_git_provenance(execution) != expected
            or json.loads((execution / MARKER).read_text()) != payload["manifest"]):
        raise ValueError("delivered_activation_materialization_changed")
    return payload


def _committed_delivery(store, payload, reference):
    activity = payload["delivery_activity_id"]
    transitions = (ActivityEvent.model_validate(row["detail"]) for row in store.verify_event_chain()
                   if row["event_type"] == "activity_transition" and row["experiment_id"] == activity)
    state = reduce(reduce_activity, transitions, None)
    if (state is None or state.status != "succeeded" or state.spec.kind != "delivery"
            or state.spec.source_digest != payload["accepted_handoff"]["source_digest"]
            or state.spec.input_digest != contract_digest(payload["wait"])):
        raise ValueError("delivered_activation_requires_committed_delivery")
    terminal = next(event for event in reversed(store.verify_event_chain())
                    if event["event_type"] == "activity_transition" and event["experiment_id"] == activity
                    and event["detail"]["operation"] in {"finish", "reconcile_finish"})
    attempt = Path(state.spec.output_namespace) / terminal["detail"]["lease"]["attempt_id"]
    for name in ("receipt.json", "domain-reconciliation.json"):
        path = checked_path(store.root, str(attempt / name))
        if hashlib.sha256(path.read_bytes()).hexdigest() != state.outputs.get(name):
            raise ValueError("delivered_activation_committed_output_changed")
    proof = json.loads(path.read_text())
    if (proof.get("delivered_activation") != reference
            or any(proof.get(key) != value for key, value in payload["reconciliation"].items())):
        raise ValueError("delivered_activation_committed_reference_mismatch")


def resolve_delivered_activation(store, reference):
    """Return an authenticated new handoff only after committed delivery success."""
    payload = _validated_payload(store, reference)
    _committed_delivery(store, payload, reference)
    return {**payload["accepted_handoff"], "activation_id": payload["activation_id"],
            "delivered_source": reference, "successor_execution": payload["execution"],
            "restart_cwd": payload["execution"], "environment_transition": payload["environment_transition"]}
