"""Logical repair lookup independent of transient activity leases and retries."""

from __future__ import annotations

import hashlib

from slm_training.autoresearch.heal.dispatch import _prior_result
from slm_training.autoresearch.heal.repair_contracts import RepairRequest, VerificationBinding
from slm_training.lineage.records import canonical_json


def logical_job_digest(request: RepairRequest, config_digest: str) -> str:
    payload = request.model_dump(mode="json", exclude={"attempt_id", "fence", "parent_event"})
    payload["blocker"].pop("evidence")
    return hashlib.sha256(canonical_json({"request": payload, "config": config_digest}).encode()).hexdigest()


def resumable_proposal(journal, request: RepairRequest, config_digest: str):
    """Read actual verified history and content-bound objects, never latest files."""
    job = logical_job_digest(request, config_digest)
    events = journal.verify_event_chain()
    for event in reversed(events):
        if event["event_type"] != "repair_job_attempt" or event.get("detail", {}).get("job_digest") != job:
            continue
        digest = event["detail"]["request_digest"]
        path = journal.root / "artifacts/repair_requests" / (digest + ".json")
        original = RepairRequest.model_validate_json(path.read_text())
        if original.digest() != digest or logical_job_digest(original, config_digest) != job:
            raise ValueError("repair_job_request_integrity_failure")
        result = _prior_result(journal, original, events)
        if result and result.status == "waiting_verification" and result.proposal is not None:
            return original, result
    journal.write_artifact("repair_requests", request)
    journal.append_event("repair_job_attempt", idempotency_key="repair-job:" + request.digest(),
                        detail={"job_digest": job, "request_digest": request.digest()})
    return request, None


def bind_verification(journal, request, proposal, context) -> VerificationBinding:
    if request.grant is None:
        raise ValueError("verification_grant_missing")
    binding = VerificationBinding(request_digest=request.digest(), proposal_digest=proposal.digest(),
        grant_digest=request.grant.digest(), fence=context.fence,
        attempt_id=context.attempt_id, parent_event=context.parent_event)
    journal.write_artifact("repair_verification_bindings", binding)
    journal.append_event("repair_verification_bound", idempotency_key="verification-binding:"+binding.digest(),
                        detail=binding.model_dump(mode="json"))
    return binding

