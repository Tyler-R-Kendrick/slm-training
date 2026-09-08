"""Controller publication of independently checked data action evidence."""
from pathlib import Path

from slm_training.data.readiness_receipt import DataActionEvidence, current_successors
from slm_training.harness_core.checkpoint_publication import controller_artifact_publication


def complete_data_action(*, cwd, root, handoff, action_index):
    """Controller-owned closure; retry after publication rechecks without rebuilding."""
    from slm_training.autoresearch.schemas import AutotrainActionReceiptV1
    from slm_training.autoresearch.storage import (
        CampaignStore, append_autotrain_action_receipt, autotrain_action_sha256,
        bind_autotrain_action_evidence,
    )

    action = handoff.actions[action_index]
    if action.kind != "rebuild_data":
        raise ValueError("data evidence cannot waive another prerequisite")
    store = CampaignStore(handoff.campaign_id, root)
    action_sha = autotrain_action_sha256(action)
    matches = current_successors(store, cwd=cwd, loop_id=handoff.loop_id, action_sha=action_sha)
    if not matches:
        return False
    if len(matches) != 1:
        raise ValueError("ambiguous data action successor")
    request, candidate, event = matches[0]
    evidence = DataActionEvidence(loop_id=handoff.loop_id, campaign_id=handoff.campaign_id,
        action_sha256=action_sha, request_sha256=request.sha256, candidate=candidate,
        ready_event_id=event["event_id"], data_root=str(Path(cwd).resolve()))
    path = store.write_artifact("data_action_evidence", evidence)
    uris = (path.relative_to(store.root).as_posix(),)
    bound = bind_autotrain_action_evidence(root, handoff, action, uris)
    journal = CampaignStore("runtime", Path(root) / "loops" / handoff.loop_id)
    _publish_acceptance(journal, path.stem, request, candidate, action_index)
    append_autotrain_action_receipt(root, AutotrainActionReceiptV1(
        loop_id=handoff.loop_id, campaign_id=handoff.campaign_id, action_index=action_index,
        action_sha256=action_sha, action_kind="rebuild_data", status="completed",
        evidence_uris=uris, evidence=bound))
    return True


def _publish_acceptance(journal, artifact_sha, request, candidate, action_index):
    key = "data-successor:" + request.campaign_id + ":" + request.sha256
    detail = {"campaign_id": request.campaign_id, "action_index": action_index,
              "original": request.original.model_dump(), "candidate": candidate.model_dump(),
              "request_sha256": request.sha256}
    # Admission/model preparation stays outside the runtime's short publication lock.
    with controller_artifact_publication(journal.root) as fence:
        previous = next((e for e in journal.verify_event_chain() if e.get("idempotency_key") == key), None)
        if previous is not None:
            if {k: v for k, v in previous["detail"].items() if k != "fence"} != detail:
                raise ValueError("accepted data successor identity changed")
            detail = previous["detail"]  # Preserve the original fence, including legacy absence.
        else:
            detail = {**detail, "fence": fence}
        journal.append_event("data_successor_accepted", artifact_sha256=artifact_sha,
                             detail=detail, idempotency_key=key)
