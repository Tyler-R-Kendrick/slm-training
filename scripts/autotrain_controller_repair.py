"""Trusted repair-operation composition for the existing supervisor child."""

from __future__ import annotations

import hashlib
from pathlib import Path

from slm_training.autoresearch.runtime.activity_publication import DelegatedPublisher
from slm_training.harness_core.activity_contract import (
    ActivityLease,
    ActivityOutcome,
)
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.train_data.readiness import bootstrap_screening_request

from slm_training.autoresearch.heal.classify import classify_blocker
from slm_training.autoresearch.heal.recovery_dispatch import (
    RecoveryContext,
    dispatch_hard_pending,
    load_recovery_config,
)
from slm_training.autoresearch.heal.repair_release import (
    pinned_repair_source, verified_release_callback, source_verification_callback,
)


def dispatch_controller_repairs(
    request, *, cwd: Path, root: Path, loop_id: str
) -> list[dict]:
    """Execute only the approved repair seam, using immutable input and fencing."""
    lease = ActivityLease.model_validate(request["lease"])
    store = CampaignStore("runtime", root / "loops" / loop_id)
    publisher = DelegatedPublisher(store, request["source_digest"])

    def fence_valid(token):
        if token != lease.token:
            return False
        with publisher.publication(lease):
            return True

    path = Path(request["repair_config"]) if request.get("repair_config") else None
    if (
        path is not None
        and hashlib.sha256(path.read_bytes()).hexdigest()
        != request["repair_config_digest"]
    ):
        raise ValueError("repair authority changed during invocation")
    config = load_recovery_config(path)
    source, source_digest = cwd, request["source_digest"]
    publish = None
    if config is not None and config.grant is not None:
        source, source_digest = pinned_repair_source(cwd, request["source_digest"])
        releases = Path(
            request.get("release_root") or source.parent / "verified-releases"
        )
        publish = verified_release_callback(
            publisher, lease, destinations=(releases, root)
        )
    context = RecoveryContext(
        root=root,
        loop_id=loop_id,
        campaign_id=request["campaign_id"] or "repair-" + loop_id,
        source=source,
        source_digest=source_digest,
        environment_digest=request["environment_digest"],
        fence=lease.token,
        parent_event=request["parent_event"],
        attempt_id=lease.attempt_id,
    )
    results = []
    for pending in request["hard_pending"]:
        kind = classify_blocker(
            str(pending.get("kind") or ""),
            str(pending.get("reason") or ""),
            code=pending.get("blocker_code"),
        )
        if kind not in {"code", "unknown"}:
            continue
        result = dispatch_hard_pending(
            pending,
            context,
            config=config,
            fence_valid=fence_valid,
            publish_verified=publish,
            source_verification=source_verification_callback(context, config) if config is not None else None,
        )
        results.append(result)
        if result.get("release_handoff") or result.get("verification_dependency"):
            break  # Remaining repairs need this successor, not its predecessor.
    return results


def repair_payload_outcome(payload: dict) -> ActivityOutcome:
    """Operational completion is not a repaired predicate or scientific result."""
    results = payload.get("agent_repairs", [])
    if any(row.get("release_handoff") for row in results):
        return ActivityOutcome.SUCCEEDED
    if not results:
        return (
            ActivityOutcome.SUCCEEDED
            if payload.get("any_healed")
            else ActivityOutcome.CAPABILITY
        )
    statuses = {row.get("status") for row in results}
    if statuses == {"cancelled"}:
        return ActivityOutcome.CANCELLED
    if "waiting_capability" in statuses:
        return ActivityOutcome.CAPABILITY
    if "waiting_verification" in statuses:
        configured = all((row.get("verification_dependency") or {}).get("verification_identity")
                         for row in results if row.get("status") == "waiting_verification")
        return ActivityOutcome.DEPENDENCY if configured else ActivityOutcome.CAPABILITY
    return (ActivityOutcome.YIELDED if statuses <= {"waiting_diagnosis", "rejected"}
            else ActivityOutcome.DEPENDENCY)


def dispatch_screening_rebuild(
    *, cwd, root, loop_id, campaign_id, train_version=None,
    eval_version=None, minimum=None, readiness_context=None,
):
    """Use the frozen data action; never append tracked seeds or guess ancestry.

    Publication restores only data readiness. The caller must still resolve a
    compatible successor experiment; this does not acknowledge its measurement.
    """
    if not campaign_id:
        if readiness_context is None:
            return None
        bootstrap = bootstrap_screening_request(
            cwd=cwd, root=root, loop_id=loop_id, train_version=train_version,
            eval_version=eval_version, minimum=minimum, context=readiness_context)
        return _dispatch_data_request(cwd=cwd, root=root, loop_id=loop_id,
            campaign_id=bootstrap["campaign_id"], blocker=bootstrap, screening=True)
    from slm_training.autoresearch.schemas import AutotrainCycleHandoffV1
    from slm_training.autoresearch.storage import autotrain_action_sha256, pending_autotrain_actions

    handoff_path = Path(root) / campaign_id / "cycle_handoff.json"
    action_id = None
    if handoff_path.is_file():
        handoff = AutotrainCycleHandoffV1.model_validate_json(handoff_path.read_text())
        if handoff.loop_id != loop_id or handoff.campaign_id != campaign_id:
            raise ValueError("screening repair handoff identity mismatch")
        for _, action in pending_autotrain_actions(root, handoff):
            if action.kind == "rebuild_data" and action.blocker_code == "screening_suite_volume":
                action_id = autotrain_action_sha256(action)
                break
    blocker = {"kind": "rebuild_data", "blocker_code": "screening_suite_volume",
               "reason": "restore locked screening suite volume", "campaign_id": campaign_id,
               "data_action_id": action_id}
    return _dispatch_data_request(cwd=cwd, root=root, loop_id=loop_id,
                                  campaign_id=campaign_id, blocker=blocker, screening=True,
                                  screening_context=(train_version, eval_version, minimum),
                                  readiness_context=readiness_context)


def _resolve_data_request(*, cwd, root, campaign_id, blocker, screening,
                          screening_context=None, readiness_context=None):
    from slm_training.autoresearch.heal.playbooks import data_rebuild
    from slm_training.harnesses.train_data.readiness import (
        build_screening_request, lock_readiness_request, screening_request_context,
    )

    store = CampaignStore(campaign_id, root)
    train_version, eval_version, minimum = screening_context or (None, None, None)
    experiment_id = blocker.get("experiment_id")
    try:
        request = data_rebuild.load_readiness_request(blocker, root=root, campaign_id=campaign_id)
        if readiness_context is not None:
            from slm_training.data.readiness_contract import DataReadinessRequest
            expected = DataReadinessRequest.model_validate({**request.model_dump(),
                **screening_request_context(readiness_context)})
            if (expected != request
                    or (minimum is not None and (type(minimum) is not int or minimum != request.minimum_unique_cases))
                    or (eval_version is not None and eval_version != request.original.dataset_id)
                    or (train_version is not None and not any(
                        ref.dataset_id == train_version for ref in request.training_ancestors))):
                raise ValueError("locked data readiness context changed")
        return request
    except (OSError, TypeError, ValueError) as exc:
        load_error = exc
        can_build = screening and blocker.get("data_action_id") and all(
            value is not None for value in (train_version, eval_version, minimum)
        )
        if can_build:
            try:
                if experiment_id is None:
                    experiments = {event["experiment_id"] for event in store.verify_event_chain()
                                   if event["event_type"] == "experiment_campaign_locked"}
                    if len(experiments) != 1:
                        raise ValueError("missing or ambiguous data request experiment identity")
                    experiment_id = experiments.pop()
                request = build_screening_request(
                    root=Path(cwd), campaign_id=campaign_id,
                    action_id=str(blocker["data_action_id"]),
                    train_version=str(train_version), eval_version=str(eval_version),
                    minimum=minimum, context=readiness_context,
                )
                lock_readiness_request(store, request, experiment_id=experiment_id)
            except (OSError, TypeError, ValueError, StopIteration) as build_exc:
                load_error = build_exc
            else:
                return data_rebuild.load_readiness_request(
                    blocker, root=root, campaign_id=campaign_id
                )
        store.append_event(
            "screening_repair_wait" if screening else "data_repair_wait", status="waiting_capability",
            detail={"unblock_predicate": "committed exact data readiness request",
                    "wake_source": "data_readiness_request_locked",
                    "owner": "controller", "executor": "bootstrap_screening_request" if screening else "lock_readiness_request",
                    "needed_capability": "locked_data_readiness_context",
                    "data_action_id": blocker.get("data_action_id"),
                    "reason": type(load_error).__name__},
            idempotency_key="data-repair:missing-contract:" + str(blocker.get("data_action_id", "screening")))
        return None


def _dispatch_data_request(*, cwd, root, loop_id, campaign_id, blocker, screening=False,
                           screening_context=None, readiness_context=None):
    from slm_training.autoresearch.heal import run_playbooks
    from slm_training.autoresearch.heal.playbooks import data_rebuild
    from slm_training.harnesses.train_data.readiness import check_readiness, snapshot_input

    store = CampaignStore(campaign_id, root)
    request = _resolve_data_request(
        cwd=cwd, root=root, campaign_id=campaign_id, blocker=blocker,
        screening=screening, screening_context=screening_context, readiness_context=readiness_context,
    )
    if request is None:
        return None
    blocker = {**blocker, "data_readiness_request": request.model_dump(mode="json")}
    if screening and (request.original.kind != "eval" or request.purpose != "screening_volume"):
        raise ValueError("screening repair resolved a different data prerequisite")
    destination = Path(cwd) / "outputs/data" / request.original.kind / request.successor_id
    if not destination.exists():
        receipts = run_playbooks(root=root, loop_id=loop_id, campaign_id=campaign_id,
                                  blockers=[blocker], cwd=cwd,
                                  playbooks=(data_rebuild.PLAYBOOK,))
        if not receipts or receipts[-1].outcome != "healed":
            return None
    candidate = snapshot_input(Path(cwd), destination, exposure=request.original.exposure)
    observation = check_readiness(request, root=Path(cwd), candidate=candidate)
    if not observation["ready"]:
        return None
    event_type = "screening_successor_ready" if request.original.kind == "eval" else "data_successor_ready"
    key = f"data-ready:{request.sha256}:{candidate.manifest_sha256}"
    if any(event.get("idempotency_key") == key for event in store.verify_event_chain()):
        return None  # Same valid artifact is not new operational progress.
    artifact = store.write_artifact("data_readiness", observation)
    store.append_event(event_type, artifact_sha256=artifact.stem,
                       detail={"request_sha256": request.sha256,
                               "loop_id": loop_id, "data_root": str(Path(cwd).resolve()),
                               "candidate": candidate.model_dump(),
                               "measurement_complete": False}, idempotency_key=key)
    return event_type


def dispatch_data_actions(*, cwd, root, loop_id, campaign_id):
    """One action cannot discharge another action's dataset, validity or volume."""
    from slm_training.autoresearch.schemas import AutotrainCycleHandoffV1
    from slm_training.autoresearch.storage import autotrain_action_sha256, pending_autotrain_actions
    from slm_training.autoresearch.heal.data_predicate_receipt import complete_data_action

    if not campaign_id:
        return None
    path = Path(root) / campaign_id / "cycle_handoff.json"
    if not path.is_file():
        return None
    handoff = AutotrainCycleHandoffV1.model_validate_json(path.read_text())
    if handoff.loop_id != loop_id or handoff.campaign_id != campaign_id:
        raise ValueError("data repair handoff identity mismatch")
    advances = []
    for index, action in pending_autotrain_actions(root, handoff):
        if action.kind != "rebuild_data":
            continue
        blocker = {"kind": "rebuild_data", "reason": action.reason,
                   "campaign_id": campaign_id, "data_action_id": autotrain_action_sha256(action)}
        advance = _dispatch_data_request(cwd=cwd, root=root, loop_id=loop_id,
                                         campaign_id=campaign_id, blocker=blocker)
        if complete_data_action(cwd=cwd, root=root, handoff=handoff, action_index=index):
            advance = "rebuild_data"
        advances.append(advance)
    return next((result for result in advances if result is not None), None)


def verified_training_successors(*, cwd, root, loop_id, campaign_id):
    """Recover accepted action results, never a producer's readiness alone."""
    from slm_training.data.readiness_receipt import current_successors

    current = {request.sha256: candidate for request, candidate, _ in current_successors(
        CampaignStore(campaign_id, root), cwd=cwd, loop_id=loop_id, kind="train")}
    journal = CampaignStore("runtime", Path(root) / "loops" / loop_id)
    accepted = {}
    for event in journal.verify_event_chain():
        detail = event["detail"]
        if (event["event_type"] != "data_successor_accepted"
                or detail.get("campaign_id") != campaign_id
                or detail.get("request_sha256") not in current):
            continue
        verified = _accepted_data_successor(event, root=root, loop_id=loop_id, cwd=cwd)
        candidate = current[verified["request_sha256"]]
        accepted[candidate.dataset_id] = candidate
    return list(accepted.values())


def preserve_workspace(*, cwd, root, loop_id, reason, paths=()):
    """Record delivery-dependent work without modifying WIP or calling it healed."""
    if root is None or loop_id is None:
        return None
    store = CampaignStore("runtime", Path(root) / "loops" / loop_id)
    payload = {"schema": "workspace_delivery_request/v1", "cwd": str(Path(cwd).resolve()),
               "loop_id": loop_id, "reason": reason, "paths": sorted(set(paths)),
               "required_capability": "authorized_github_connector_delivery",
               "unblock_predicate": "independently verified immutable successor release is available",
               "wake_source": "successor_release_published", "source_modified": False}
    artifact = store.write_artifact("workspace_delivery_requests", payload)
    store.append_event("workspace_delivery_wait", status="waiting_capability",
                       artifact_sha256=artifact.stem, detail=payload,
                       idempotency_key="workspace-delivery:" + artifact.stem)
    return {**payload, "kind": "workspace", "state": "waiting_capability",
            "artifact_sha256": artifact.stem}


def _accepted_data_successor(event, *, root, loop_id, cwd):
    """Validate both event ownership and what the actual data resolver will read."""
    from slm_training.data.readiness_receipt import validate_data_action_evidence
    from slm_training.autoresearch.schemas import AutotrainCycleHandoffV1
    from slm_training.autoresearch.storage import autotrain_action_sha256
    from slm_training.data.store import DataStore
    from slm_training.harnesses.train_data.readiness import snapshot_input

    detail = event["detail"]
    store = CampaignStore(detail["campaign_id"], root)
    handoff = AutotrainCycleHandoffV1.model_validate_json((store.root / "cycle_handoff.json").read_text())
    index = detail["action_index"]
    if type(index) is not int or not 0 <= index < len(handoff.actions):
        raise ValueError("invalid data successor action index")
    path = store.root / "artifacts/data_action_evidence" / f"{event['artifact_sha256']}.json"
    evidence, request = validate_data_action_evidence(store, handoff, autotrain_action_sha256(handoff.actions[index]), path)
    if (evidence.loop_id != loop_id or evidence.data_root != str(Path(cwd).resolve())
            or request.original.model_dump() != detail["original"]
            or request.sha256 != detail["request_sha256"]
            or evidence.candidate.model_dump() != detail["candidate"]):
        raise ValueError("data successor selection identity mismatch")
    for snapshot in (request.original, evidence.candidate):
        resolved = DataStore(cwd).resolve(snapshot.kind, snapshot.dataset_id).path
        if snapshot_input(Path(cwd), resolved, exposure=snapshot.exposure, records=snapshot.records) != snapshot:
            raise ValueError("data successor resolver identity mismatch")
    return {"campaign_id": handoff.campaign_id, "evidence_sha256": path.stem,
            "request_sha256": request.sha256, "original": detail["original"],
            "candidate": evidence.candidate.model_dump(),
            "readiness_dependencies": [ref.model_dump() for ref in
                (request.scored_suites if request.original.kind == "train" else request.training_ancestors)]}


def resolve_repaired_data(*, root, loop_id, cwd, train_version, eval_version, intent="screening"):
    """Only fresh screening designs follow unique, currently verified successors."""
    versions = {"train": train_version, "eval": eval_version}
    applied, edges = [], {}
    if intent == "screening":
        journal = CampaignStore("runtime", Path(root) / "loops" / loop_id)
        for event in journal.verify_event_chain():
            if event["event_type"] == "data_successor_accepted":
                original = event["detail"]["original"]
                edges.setdefault((original["kind"], original["dataset_id"]), []).append(event)
    for kind, version in versions.items():
        seen = {version}
        while candidates := edges.get((kind, version)):
            if len(candidates) != 1:
                raise ValueError("ambiguous data successor selection")
            accepted = _accepted_data_successor(candidates[0], root=root, loop_id=loop_id, cwd=cwd)
            version = accepted["candidate"]["dataset_id"]
            if version in seen:
                raise ValueError("cyclic data successor selection")
            seen.add(version)
            versions[kind] = version
            applied.append(accepted)
    _require_selection_dependencies(cwd, versions, applied)
    return {"train_version": versions["train"], "eval_version": versions["eval"], "successions": applied}


def _require_selection_dependencies(cwd, versions, applied):
    from slm_training.data.store import DataStore
    from slm_training.harnesses.train_data.readiness import snapshot_input

    for accepted in applied:
        other = "eval" if accepted["candidate"]["kind"] == "train" else "train"
        dependencies = [ref for ref in accepted["readiness_dependencies"]
                        if ref["kind"] == other and ref["dataset_id"] == versions[other]]
        if not dependencies:
            raise ValueError("combined data successors require new cross-lineage readiness evidence")
        for ref in dependencies:
            actual = DataStore(cwd).resolve(other, versions[other]).path
            if snapshot_input(Path(cwd), actual, exposure=ref["exposure"], records=ref["records"]).model_dump() != ref:
                raise ValueError("data successor readiness dependency identity mismatch")
