"""Accepted local repair subjects; remote lineage and delivery need separate grants.

Only the controller publication path calls this owner. Artifact hashes bind
content; verified journal events and the fenced publisher supply authority.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from slm_training.harness_core.activity_contract import contract_digest
from slm_training.harness_core.execution_release import (
    MARKER, _files, runtime_git_provenance, runtime_source_identity,
)
from slm_training.lineage.records import canonical_json

from .isolation_workspace import checked_path, manifest_digest, tree_manifest
from .repair_acceptance import (
    SourceVerificationGate, VerificationWorkspace, patch_manifest_digest,
)
from .repair_scope import require_routine_scope
from .repair_contracts import RepairDispatchResult, RepairRequest


def _artifact(journal, kind, digest):
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError("repair_delivery_artifact_digest_invalid")
    path = checked_path(journal.root, f"artifacts/{kind}/{digest}.json")
    value = json.loads(path.read_text())
    if hashlib.sha256(canonical_json(value).encode()).hexdigest() != digest:
        raise ValueError("repair_delivery_artifact_changed")
    return value


def _accepted_artifacts(journal, request, result):
    events = journal.verify_event_chain()
    expected = (
        ("repair_started", "repair_requests", request.digest(), request),
        ("repair_dispatch", "repair_dispatch", result.digest(), result),
    )
    refs = {}
    for event_kind, kind, digest, model in expected:
        if not any(event["event_type"] == event_kind
                   and event.get("artifact_sha256") == digest for event in events):
            raise ValueError("repair_delivery_acceptance_event_missing")
        if _artifact(journal, kind, digest) != model.model_dump(mode="json"):
            raise ValueError("repair_delivery_accepted_artifact_mismatch")
        refs[kind] = digest
    refs["repair_verification"] = result.verification.evidence_digest
    evidence = _artifact(journal, "repair_verification", refs["repair_verification"])
    matches = {}
    for event in events:
        if event["event_type"] != "repair_source_verification_prepared":
            continue
        digest = event["artifact_sha256"]
        value = _artifact(journal, "repair_source_verification_inputs", digest)
        if (value.get("request_digest") == request.digest()
                and value.get("proposal_digest") == result.proposal.digest()
                and value.get("verification_identity") == evidence["source_verification"]["identity"]):
            matches[digest] = value
    if len(matches) != 1:
        raise ValueError("repair_delivery_source_input_missing_or_ambiguous")
    digest, inputs = next(iter(matches.items()))
    refs["repair_source_verification_inputs"] = digest
    return refs, evidence, inputs


def _source_gate(journal, inputs, evidence, workspace, historical):
    identity_inputs = {key: inputs[key] for key in
                       ("request_digest", "proposal_digest", "config_digest",
                        "verification_identity")}
    directory = journal.root / "source_verification" / contract_digest(identity_inputs)
    manifest = checked_path(journal.root, str(directory.relative_to(journal.root) / "manifest.json"))
    if json.loads(manifest.read_text()) != inputs:
        raise ValueError("repair_delivery_source_input_changed")
    if contract_digest(inputs["binding"]) != inputs["verification_identity"]:
        raise ValueError("repair_delivery_source_binding_changed")
    gate = SourceVerificationGate(directory / "root", directory / "cache",
                                  inputs["base_ref"], inputs["verification_identity"])
    summary = _historical_gate(gate, inputs) if historical else gate.read(workspace)
    if summary is None or summary != evidence["source_verification"]:
        raise ValueError("repair_delivery_source_verification_changed")
    return {"root": str(gate.root.resolve()), "state_dir": str(gate.state_dir.resolve()),
            "base_ref": gate.base_ref, "identity": gate.identity}


def _historical_gate(gate, inputs):
    """Authenticate old evidence without claiming its environment is current."""
    from scripts.merge_verification import _summary
    from scripts.merge_verification_evidence import ReceiptCache, source_identity, validate_cached_state

    if not (gate.state_dir / "issuer.key").is_file():
        raise ValueError("repair_delivery_historical_gate_missing")
    state = ReceiptCache(gate.state_dir, gate.root).load(gate.identity)
    if state is None or source_identity(gate.root) != inputs["binding"]["candidate_tree_sha256"]:
        raise ValueError("repair_delivery_historical_source_changed")
    validate_cached_state(state, inputs["binding"])
    return _summary(state)


def delivery_inputs(request, result, journal, predecessor, candidate, *, historical=False):
    """Resolve exact accepted evidence, never a worker path or 'latest' result."""
    execution, expected_digest = predecessor
    if journal.campaign_id != request.campaign_id or runtime_source_identity(execution) != expected_digest:
        raise ValueError("repair_delivery_predecessor_mismatch")
    marker = json.loads((execution / MARKER).read_text())
    base = Path(marker["release"]).resolve(strict=True)
    provenance = runtime_git_provenance(execution)
    refs, evidence, inputs = _accepted_artifacts(journal, request, result)
    before, after = tree_manifest(base), tree_manifest(candidate)
    source_digest, candidate_digest = manifest_digest(before), manifest_digest(after)
    pairs = (
        (source_digest, request.blocker.source_digest),
        (candidate_digest, result.proposal.tree_digest),
        (inputs["source_snapshot_digest"], source_digest),
        (inputs["candidate_snapshot_digest"], candidate_digest),
        (evidence["source_snapshot_digest"], source_digest),
        (evidence["candidate_snapshot_digest"], candidate_digest),
        (patch_manifest_digest(before, after), result.proposal.patch_digest),
    )
    if any(actual != expected for actual, expected in pairs):
        raise ValueError("repair_delivery_scope_identity_mismatch")
    changes = {path: {"before": before.get(path), "after": after.get(path)}
               for path in sorted(before.keys() | after.keys())
               if before.get(path) != after.get(path)}
    changed_files = tuple(path for path in changes
                          if not (path not in before and (candidate / path).is_dir()))
    require_routine_scope(base, candidate, changed_files, request.allowed_paths,
                          request.semantics_preserving_paths,
                          request_digest=request.digest())
    workspace = VerificationWorkspace(base, candidate,
        tuple(Path(path) for path in inputs["binding"]["runtime_roots"]))
    gate = _source_gate(journal, inputs, evidence, workspace, historical)
    if (tree_manifest(base) != before or tree_manifest(candidate) != after
            or runtime_source_identity(execution) != expected_digest
            or runtime_git_provenance(execution) != provenance):
        raise ValueError("repair_delivery_inputs_changed")
    return {
        "campaign_id": request.campaign_id, "campaign_store": str(journal.root.resolve()),
        "artifacts": refs, "request_digest": request.digest(),
        "proposal_digest": result.proposal.digest(), "verification_digest": result.verification.digest(),
        "blocked_activity_id": request.blocked_activity_id,
        "original_input_digest": request.blocker.input_digest,
        "predecessor": {"execution": str(execution.resolve()), "release": str(base),
                        "runtime_source_digest": expected_digest,
                        "source_snapshot_digest": source_digest, "git_provenance": provenance},
        "scope": {"allowed_paths": list(request.allowed_paths),
                  "semantics_preserving_paths": list(request.semantics_preserving_paths),
                  "patch_digest": result.proposal.patch_digest, "changes": changes},
        "source_verification": gate,
    }


def _delivery_payload(selected, inputs):
    candidate = Path(selected["verified_source"])
    if manifest_digest(tree_manifest(candidate)) != selected["verified_tree_digest"]:
        raise ValueError("repair_delivery_frozen_candidate_changed")
    if runtime_source_identity(Path(selected["execution"])) != selected["runtime_source_digest"]:
        raise ValueError("repair_delivery_successor_changed")
    files = _files(candidate)
    if contract_digest(files) != selected["runtime_source_digest"]:
        raise ValueError("repair_delivery_successor_manifest_mismatch")
    return {
        "schema_version": "verified_repair_source_delivery/v1", **inputs,
        "publication_id": selected["publication_id"],
        "pointer_digest": contract_digest(selected),
        "predecessor_publication_id": selected.get("predecessor_publication_id"),
        "successor_source_digest": selected["runtime_source_digest"],
        "successor_execution": selected["execution"],
        "successor": {"execution": selected["execution"], "release": selected["release"],
                      "verified_source": str(candidate),
                      "runtime_source_digest": selected["runtime_source_digest"],
                      "source_snapshot_digest": selected["verified_tree_digest"],
                      "files": files},
    }


def publish_delivery(store, selected, inputs):
    """Persist a subject before acceptance; only its accepted handoff grants use."""
    payload = _delivery_payload(selected, inputs)
    artifact = store.write_artifact("verified_repair_source_delivery", payload)
    return {"kind": "verified_repair_source", "publication_id": selected["publication_id"],
            "artifact_sha256": artifact.stem,
            "required_capability": "authorized_github_connector_delivery"}


def _accepted_pointer(store, wait, payload):
    expected = {"kind": "verified_repair_source", "publication_id": payload["publication_id"],
                "artifact_sha256": wait["artifact_sha256"],
                "required_capability": "authorized_github_connector_delivery"}
    if wait != expected or payload["schema_version"] != "verified_repair_source_delivery/v1":
        raise ValueError("repair_delivery_dependency_mismatch")
    events = [event for event in store.verify_event_chain()
              if event.get("detail", {}).get("publication_id") == payload["publication_id"]]
    intents = [event["detail"]["pointer"] for event in events
               if event["event_type"] == "repair_release_intent"]
    accepted = [event["detail"] for event in events
                if event["event_type"] == "repair_release_accepted"]
    if len(intents) != 1 or len(accepted) != 1:
        raise ValueError("repair_delivery_publication_acceptance_missing")
    pointer, receipt = intents[0], accepted[0]
    if (contract_digest(pointer) != payload["pointer_digest"]
            or receipt["pointer_digest"] != payload["pointer_digest"]
            or receipt["handoff"].get("source_delivery") != wait
            or receipt["handoff"]["source_digest"] != pointer["runtime_source_digest"]
            or receipt["handoff"]["successor_execution"] != pointer["execution"]):
        raise ValueError("repair_delivery_publication_binding_mismatch")
    return pointer


def resolve_source_delivery(runtime_store, wait):
    """Authenticate a historical accepted subject, never grant remote delivery.

    The caller uses successor_source_digest for activity registration. A later
    current pointer does not invalidate this immutable accepted publication.
    Remote comparison-base coverage and current delivery authority remain the
    consumer's responsibility; the synthetic source gate is not that authority.
    """
    from slm_training.autoresearch.storage import CampaignStore

    payload = _artifact(runtime_store, "verified_repair_source_delivery", wait["artifact_sha256"])
    pointer = _accepted_pointer(runtime_store, wait, payload)
    directory = Path(payload["campaign_store"])
    journal = CampaignStore(payload["campaign_id"], directory.parent)
    if journal.root.resolve() != directory:
        raise ValueError("repair_delivery_campaign_namespace_mismatch")
    refs = payload["artifacts"]
    request = RepairRequest.model_validate_json(json.dumps(_artifact(journal, "repair_requests", refs["repair_requests"])))
    result = RepairDispatchResult.model_validate_json(json.dumps(_artifact(journal, "repair_dispatch", refs["repair_dispatch"])))
    if (result.status != "verified" or result.verification is None or result.proposal is None
            or pointer["verification_digest"] != result.verification.digest()
            or pointer["publication_id"] != contract_digest({"request": request.digest(), "proposal": result.proposal.digest()})):
        raise ValueError("repair_delivery_accepted_result_mismatch")
    predecessor = payload["predecessor"]
    inputs = delivery_inputs(request, result, journal,
        (Path(predecessor["execution"]), predecessor["runtime_source_digest"]),
        Path(pointer["verified_source"]), historical=True)
    if canonical_json(_delivery_payload(pointer, inputs)) != canonical_json(payload):
        raise ValueError("repair_delivery_subject_changed")
    return payload
