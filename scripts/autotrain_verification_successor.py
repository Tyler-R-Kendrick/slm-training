"""Rebind a saved repair proposal when its verifier environment has drifted."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.merge_verification_evidence import digest
from scripts.verify_merge_ready import merge_gate_steps
from slm_training.harness_core.activity_contract import ResourceGrant


def _remaining_grant(state):
    original = state.spec.grant.model_dump(mode="json")
    original["total_seconds"] -= state.charged_seconds
    original["max_attempts"] -= state.attempts
    if (
        original["max_attempts"] < 1
        or original["total_seconds"]
        < original["interrupt_seconds"] + original["kill_grace_seconds"]
        + original["finalization_reserve_seconds"]
    ):
        return None
    return ResourceGrant.model_validate(original)


def _materialize(runtime, predecessor, grant):
    from slm_training.autoresearch.heal.dispatch import _prior_result
    from slm_training.autoresearch.heal.recovery_dispatch import (
        RecoveryContext,
        load_recovery_config,
    )
    from slm_training.autoresearch.heal.repair_contracts import RepairRequest
    from slm_training.autoresearch.heal.isolation_workspace import (
        manifest_digest, private_snapshot, tree_manifest,
    )
    from slm_training.autoresearch.heal.repair_source_workspace import prepare_source_verification
    from slm_training.autoresearch.heal.repair_acceptance import VerificationWorkspace
    from slm_training.autoresearch.storage import CampaignStore

    common = predecessor["_successor_common"]
    config_path = Path(common["repair_config"])
    config = load_recovery_config(
        config_path, expected_sha256=common["repair_config_digest"]
    )
    if config is None or config.source_verification_grant is None:
        raise ValueError("source_verification_successor_grant_missing")
    config = config.model_copy(update={"source_verification_grant": grant})
    campaign_root = Path(predecessor["manifest_path"]).resolve().parents[2]
    manifest = json.loads(Path(predecessor["manifest_path"]).read_text())
    identity_fields = {
        "request_digest": predecessor["request_digest"],
        "proposal_digest": predecessor["proposal_digest"],
        "verification_identity": predecessor["verification_identity"],
        "candidate_snapshot_digest": predecessor["candidate_snapshot_digest"],
        "base_ref": predecessor["base_ref"],
    }
    if any(manifest.get(key) != value for key, value in identity_fields.items()):
        raise ValueError("source_verification_predecessor_manifest_mismatch")
    journal = CampaignStore(predecessor["campaign_id"], campaign_root.parent)
    request_path = journal.root / "artifacts" / "repair_requests" / (
        predecessor["request_digest"] + ".json"
    )
    request = RepairRequest.model_validate_json(request_path.read_text())
    if request.digest() != predecessor["request_digest"]:
        raise ValueError("source_verification_request_digest_mismatch")
    result = _prior_result(journal, request, journal.verify_event_chain())
    proposal = result.proposal if result is not None else None
    if proposal is None or proposal.digest() != predecessor["proposal_digest"]:
        raise ValueError("source_verification_proposal_unavailable")
    roots = tuple(str(Path(path).resolve()) for path in predecessor["runtime_roots"])
    candidate = campaign_root / "repair_workspaces" / request.digest() / "candidate"
    base_seed = Path(predecessor["root"]).resolve().parent / "base"
    base = candidate.parent / "verification-bases" / manifest["source_snapshot_digest"]
    base.parent.mkdir(parents=True, exist_ok=True)
    if not base.exists():
        private_snapshot(base_seed, base)
    if manifest_digest(tree_manifest(base)) != manifest["source_snapshot_digest"]:
        raise ValueError("source_verification_predecessor_base_mismatch")
    context = RecoveryContext(
        root=campaign_root.parent,
        loop_id=common["loop_id"],
        campaign_id=request.campaign_id,
        source=base,
        source_digest=manifest["source_snapshot_digest"],
        environment_digest=request.blocker.environment_digest,
        fence=request.fence,
        parent_event=request.parent_event,
        attempt_id=request.attempt_id,
    )
    gate = prepare_source_verification(
        context, config, request, proposal,
        VerificationWorkspace(base, candidate, tuple(Path(path) for path in roots)),
    )
    return journal, request, proposal, config, gate, roots


def _activation(runtime, predecessor_event, predecessor, request, proposal, dependency):
    artifact = runtime.store.write_artifact("source_verification_requests", dependency)
    detail = {
        "predecessor_dependency_digest": predecessor_event["detail"]["dependency_digest"],
        "predecessor_identity": predecessor["verification_identity"],
        "successor_dependency_digest": artifact.stem,
        "successor_identity": dependency["verification_identity"],
        "successor_activity_id": dependency["activity_id"],
        "request_digest": request.digest(),
        "proposal_digest": proposal.digest(),
    }
    runtime.store.append_event(
        "source_verification_successor_activated",
        experiment_id=predecessor_event["experiment_id"],
        artifact_sha256=artifact.stem,
        idempotency_key="source-verification-successor:" + digest(detail),
        detail=detail,
    )
    return artifact


def plan_successor(runtime, event, predecessor, common):
    """Persist a fresh verifier identity using only the predecessor's residual grant."""
    from scripts.autotrain_verification import load_dependency

    predecessor_digest = event["detail"]["dependency_digest"]
    for row in runtime.store.verify_event_chain():
        if (
            row["event_type"] == "source_verification_successor_activated"
            and row["detail"].get("predecessor_dependency_digest") == predecessor_digest
        ):
            state = runtime.snapshot().get(predecessor["activity_id"])
            if state and state.status not in {"cancelled", "succeeded"}:
                runtime.cancel(state.spec.activity_id,
                               reason="source_verification_identity_changed")
            return {"status": "successor_activated",
                    "activity_id": row["detail"]["successor_activity_id"],
                    "verification_identity": row["detail"]["successor_identity"]}
    state = runtime.snapshot().get(predecessor["activity_id"])
    if state is None:
        raise ValueError("source_verification_predecessor_activity_missing")
    from scripts import autotrain_verification as owner

    binding = owner.verification_binding(
        Path(predecessor["root"]), predecessor["base_ref"], merge_gate_steps(),
        isolated=True, runtimes=tuple(Path(path) for path in predecessor["runtime_roots"]),
    )
    if digest(binding) == predecessor["verification_identity"]:
        return None
    grant = _remaining_grant(state)
    if grant is None:
        runtime.store.append_event(
            "source_verification_successor_wait",
            experiment_id=event["experiment_id"],
            idempotency_key="source-verification-successor-exhausted:" + predecessor_digest,
            detail={"predecessor_dependency_digest": predecessor_digest,
                    "reason": "predecessor_verification_grant_exhausted"},
        )
        return None
    materialized = _materialize(
        runtime, {**predecessor, "_successor_common": common}, grant
    )
    _, request, proposal, config, gate, roots = materialized
    current = load_dependency(runtime.store, event)
    if any(
        current.get(key) != predecessor.get(key)
        for key in ("request_digest", "proposal_digest", "candidate_snapshot_digest")
    ):
        raise ValueError("source_verification_successor_request_changed")
    from slm_training.autoresearch.heal import recovery_dispatch

    dependency = recovery_dispatch._verification_dependency(
        request, proposal, gate, grant, roots
    )
    successor = runtime.store.write_artifact("source_verification_requests", dependency)
    runtime.store.append_event(
        "source_verification_requested",
        experiment_id=event["experiment_id"],
        artifact_sha256=successor.stem,
        idempotency_key="source-verification:" + event["experiment_id"] + ":" + successor.stem,
        detail={"dependency_digest": successor.stem,
                "repair_activity_id": event["experiment_id"]},
    )
    from scripts.autotrain_verification import dependency_plan, register_dependency

    register_dependency(runtime, dependency_plan(dependency))
    _activation(runtime, event, predecessor, request, proposal, dependency)
    if state.status not in {"cancelled", "succeeded"}:
        runtime.cancel(state.spec.activity_id, reason="source_verification_identity_changed")
    return {"status": "successor_activated", "activity_id": dependency["activity_id"],
            "verification_identity": dependency["verification_identity"]}
