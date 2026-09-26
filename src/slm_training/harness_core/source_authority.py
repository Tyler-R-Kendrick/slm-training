"""Branch membership from committed controller observations, never marker claims.

The caller supplies the trusted CampaignStore and expected source/repository.
No journal location or executable is discovered from an untrusted reference.
"""

from __future__ import annotations

import hashlib
import json
from functools import reduce
from pathlib import Path

from .activity_contract import ActivityEvent, contract_digest, reduce_activity
from .execution_release import _runtime_manifest
from .github_delivery_tree import git_object, source_entries, tree_sha

_KINDS = {"initial_release": ("verify", "authorized_github_connector_read"),
          "repair_delivery": ("delivery", "authorized_github_connector_delivery")}


def authority_reference(payload):
    return {"kind": payload["kind"], "artifact_sha256": contract_digest(payload)}


def authority_record(runtime, lease, *, kind, request, remote, execution):
    """Normalize only controller-authenticated observations under a live lease."""
    with runtime.publication(lease) as state:
        payload = {
            "schema_version": "source_authority/v1", "kind": kind,
            "activity_id": lease.activity_id, "input_digest": state.spec.input_digest,
            "source_digest": state.spec.source_digest, "request": request,
            "remote": remote, "execution": str(Path(execution).resolve()),
        }
        _binding(payload, state.spec)
        return payload


def publish_source_authority(runtime, lease, payload):
    """Publication is inert until this exact output commits with activity success."""
    with runtime.publication(lease) as state:
        if payload["activity_id"] != lease.activity_id:
            raise ValueError("source_authority_activity_mismatch")
        _binding(payload, state.spec)
        artifact = runtime.store.write_artifact("source_authorities", payload)
        reference = authority_reference(payload)
        if artifact.stem != reference["artifact_sha256"]:
            raise ValueError("source_authority_digest_mismatch")
        runtime.store.append_event("source_authority_recorded", artifact_sha256=artifact.stem,
            experiment_id=lease.activity_id, idempotency_key="source-authority:" + artifact.stem,
            detail={"kind": payload["kind"], "input_digest": payload["input_digest"]})
        runtime.store._replace_durable(runtime.attempt_dir(lease) / "source-authority.json",
                                      json.dumps(payload, sort_keys=True))
        return reference


def _binding(payload, spec):
    kind = payload["kind"]
    if kind not in _KINDS:
        raise ValueError("unknown_source_authority_kind")
    activity_kind, capability = _KINDS[kind]
    request, remote = payload["request"], payload["remote"]
    original = request if kind == "initial_release" else request["wait"]
    if (payload["schema_version"] != "source_authority/v1" or spec.kind != activity_kind
            or payload["activity_id"] != spec.activity_id
            or capability not in spec.capabilities or spec.input_digest != contract_digest(original)
            or payload["input_digest"] != spec.input_digest
            or payload["source_digest"] != spec.source_digest
            or request["source_digest"] != spec.source_digest
            or remote["source_digest"] != spec.source_digest
            or request["repository"] != remote["repository"]
            or remote.get("base_branch", "main") != request.get("base_branch", "main")):
        raise ValueError("source_authority_request_binding_mismatch")
    if kind == "initial_release":
        if (request["schema_version"] != "initial_source_request/v1"
                or remote["merge_sha"] != request["commit"]
                or remote["candidate_git_tree"] != request["tree"]
                or remote["ref_before"] != remote["ref_after"]
                or remote["ref_before"] != {"ref": "refs/heads/" + request["base_branch"],
                                           "object": {"type": "commit", "sha": request["commit"]}}
                or payload["execution"] != request["execution"]):
            raise ValueError("initial_source_observation_mismatch")
    elif (request["schema_version"] != "connector_delivery_request/v1"
          or request["operation"] != "deliver_and_reconcile_squash_merge"
          or remote["base_ref"] != request["base_ref"]):
        raise ValueError("repair_source_observation_mismatch")
    raw = remote["merge_commit_object"]
    if (git_object("commit", raw.encode()) != remote["merge_sha"]
            or raw.split("\n\n", 1)[0].splitlines()[0] != "tree " + remote["candidate_git_tree"]):
        raise ValueError("source_authority_commit_tree_mismatch")


def _read(root, relative):
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("unsafe_source_authority_path")
    return path.read_bytes()


def _committed_payload(store, reference):
    if (not isinstance(reference, dict) or set(reference) != {"kind", "artifact_sha256"}
            or reference["kind"] not in _KINDS):
        raise ValueError("source_authority_reference_required")
    sha = reference["artifact_sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("invalid_source_authority_digest")
    payload = json.loads(_read(store.root, Path("artifacts/source_authorities") / (sha + ".json")))
    if authority_reference(payload) != reference:
        raise ValueError("source_authority_artifact_changed")
    events = store.verify_event_chain()
    recorded = [e for e in events if e["event_type"] == "source_authority_recorded"
                and e["artifact_sha256"] == sha]
    if (len(recorded) != 1 or recorded[0]["experiment_id"] != payload["activity_id"]
            or recorded[0]["detail"] != {"kind": payload["kind"], "input_digest": payload["input_digest"]}):
        raise ValueError("source_authority_not_fenced")
    transitions = [ActivityEvent.model_validate(e["detail"]) for e in events
                   if e["event_type"] == "activity_transition" and e["experiment_id"] == payload["activity_id"]]
    state = reduce(reduce_activity, transitions, None)
    if state is None or state.status != "succeeded":
        raise ValueError("source_authority_requires_committed_success")
    _binding(payload, state.spec)
    terminal = next(e for e in reversed(transitions) if e.operation in {"finish", "reconcile_finish"})
    attempt = Path(state.spec.output_namespace) / terminal.lease.attempt_id
    raw = _read(store.root, attempt / "source-authority.json")
    if (hashlib.sha256(raw).hexdigest() != state.outputs.get("source-authority.json")
            or json.loads(raw) != payload):
        raise ValueError("source_authority_committed_output_changed")
    if payload["kind"] == "repair_delivery":
        if payload["request"].get("lease") != terminal.lease.model_dump(mode="json"):
            raise ValueError("source_authority_terminal_lease_mismatch")
        _repair_outputs(store, attempt, state.outputs, payload, events)
    return payload


def _repair_outputs(store, attempt, outputs, payload, events):
    values = {}
    for name in ("request.json", "receipt.json", "domain-reconciliation.json"):
        raw = _read(store.root, attempt / name)
        if hashlib.sha256(raw).hexdigest() != outputs.get(name):
            raise ValueError("source_authority_delivery_output_changed")
        values[name] = json.loads(raw)
    if (values["request.json"] != payload["request"]
            or values["receipt.json"].get("request_digest") != contract_digest(payload["request"])
            or values["receipt.json"].get("provider") != "github_connector"
            or values["domain-reconciliation.json"]["receipt"]["remote"] != payload["remote"]):
        raise ValueError("source_authority_delivery_binding_mismatch")
    reconciliation = values["domain-reconciliation.json"]
    proof = reconciliation["receipt"]
    sha = contract_digest(proof)
    path = Path("artifacts/verified_repair_source_delivery_receipts") / (sha + ".json")
    wait = payload["request"]["wait"]
    if (Path(reconciliation["verification_artifact"]) != store.root / path
            or json.loads(_read(store.root, path)) != proof
            or proof.get("wait") != wait
            or proof.get("schema_version") != "verified_repair_source_delivery_receipt/v1"
            or not any(e["event_type"] == "verified_repair_source_delivered"
                       and e["artifact_sha256"] == sha and e["experiment_id"] == payload["activity_id"]
                       and e["detail"] == {"publication_id": wait["publication_id"],
                           "request_artifact_sha256": wait["artifact_sha256"]} for e in events)):
        raise ValueError("source_authority_repair_receipt_not_fenced")
    reference = reconciliation["delivered_activation"]
    activation = json.loads(_read(store.root, Path("artifacts/delivered_repair_activations") /
                                  (reference["artifact_sha256"] + ".json")))
    if (contract_digest(activation) != reference["artifact_sha256"]
            or activation["activation_id"] != reference["activation_id"]
            or activation["execution"] != payload["execution"]
            or activation["delivery_activity_id"] != payload["activity_id"]
            or activation["reconciliation"] != {key: reconciliation[key] for key in ("verification_artifact", "receipt")}
            or activation["manifest"].get("source_authority") != authority_reference(payload)
            or not any(e["event_type"] == "delivered_repair_activation_recorded"
                       and e["artifact_sha256"] == reference["artifact_sha256"]
                       and e["experiment_id"] == payload["activity_id"] for e in events)):
        raise ValueError("source_authority_repair_activation_mismatch")


def resolve_source_authority(store, reference, *, execution, expected_repository,
                             expected_commit, expected_source_digest):
    """Resolve fresh against caller-selected trust root; a copied marker is inert."""
    payload = _committed_payload(store, reference)
    execution = Path(execution).resolve()
    manifest = _runtime_manifest(execution)
    remote = payload["remote"]
    if (manifest is None or manifest.get("source_authority") != reference
            or payload["execution"] != str(execution)
            or manifest["source_digest"] != expected_source_digest
            or payload["source_digest"] != expected_source_digest
            or remote["repository"] != expected_repository
            or remote["merge_sha"] != expected_commit
            or manifest["git_provenance"] != {"integration_commit": expected_commit,
                "upstream_commit": expected_commit, "code_dirty": False}
            or tree_sha(source_entries(execution, manifest["files"])) != remote["candidate_git_tree"]):
        raise ValueError("source_authority_materialization_binding_mismatch")
    return {"repository": expected_repository, "ref": "refs/heads/" + remote.get("base_branch", "main"),
            "commit": expected_commit, "tree": remote["candidate_git_tree"],
            "source_digest": expected_source_digest, "reference": dict(reference)}


def require_main_source_authority(store, reference, *, execution, expected_repository,
                                  expected_commit, expected_source_digest):
    result = resolve_source_authority(store, reference, execution=execution,
        expected_repository=expected_repository, expected_commit=expected_commit,
        expected_source_digest=expected_source_digest)
    if result["ref"] != "refs/heads/main":
        raise ValueError("source_authority_main_membership_required")
    return result
