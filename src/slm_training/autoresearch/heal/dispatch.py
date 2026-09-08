"""Governed repair dispatch using the canonical CampaignStore event history.

Caller must hold its activity lease. No worker can close a blocker, write an
action acknowledgment, or activate a source release through this module.
"""

from __future__ import annotations

import time
from typing import Callable

from slm_training.autoresearch.heal.agent_executor import AgentCancelled, CodexExecutor
from slm_training.autoresearch.heal.repair_contracts import (
    RepairDispatchResult,
    RepairProposal,
    RepairRequest,
    RepairVerification,
    VerificationBinding,
)
from slm_training.autoresearch.storage import CampaignStore
from slm_training.levers import KILL_GRACE_SECONDS


def _record(
    journal: CampaignStore, request: RepairRequest, result: RepairDispatchResult
) -> RepairDispatchResult:
    artifact = journal.write_artifact("repair_dispatch", result)
    journal.append_event(
        "repair_dispatch",
        experiment_id=request.activity_id,
        status=result.status,
        artifact_sha256=artifact.stem,
        detail={
            "request_digest": request.digest(),
            "attempt_id": request.attempt_id,
            "fingerprint": request.blocker.fingerprint(),
            "fence": result.verification.fence
            if result.verification
            else request.fence,
            "original_request_fence": request.fence,
            "spent_seconds": result.spent_seconds,
        },
    )
    return result


def _history(journal: CampaignStore) -> list[dict]:
    return journal.verify_event_chain()


def _prior_result(
    journal: CampaignStore, request: RepairRequest, events: list[dict]
) -> RepairDispatchResult | None:
    for event in reversed(events):
        if (
            event["event_type"] == "repair_dispatch"
            and event.get("detail", {}).get("request_digest") == request.digest()
        ):
            sha = event["artifact_sha256"]
            path = journal.root / "artifacts" / "repair_dispatch" / f"{sha}.json"
            result = RepairDispatchResult.model_validate_json(path.read_text())
            if result.digest() != sha:
                raise ValueError("repair result integrity failure")
            if result.status == "verified":
                return RepairDispatchResult(
                    status="waiting_verification",
                    request_digest=request.digest(),
                    reason="historical_verification_requires_current_predicate",
                    proposal=result.proposal,
                )
            return result
    return None


def dispatch_repair(
    request: RepairRequest,
    *,
    executor: CodexExecutor | None,
    journal: CampaignStore,
    fence_valid: Callable[[str], bool],
    progress: Callable[[], None] = lambda: None,
    cancelled: Callable[[], bool] = lambda: False,
) -> RepairDispatchResult:
    """Dispatch one leased attempt; retries never silently execute it again.

    A started attempt without a committed result is an uncertain external effect:
    reconciliation/diagnosis must resolve it before issuing another attempt.
    Waiting capability requests may be polled without consuming attempt budget.
    Runtime owns cross-campaign grants and passes one stable journal per blocker.
    """
    if journal.campaign_id != request.campaign_id or not fence_valid(request.fence):
        raise ValueError("repair campaign or lease mismatch")
    events = _history(journal)
    previous = _prior_result(journal, request, events)
    if previous and previous.status != "waiting_capability":
        return previous
    fields = {"request_digest": request.digest()}
    if request.blocker.blocker_class not in {
        "code",
        "environment",
        "data",
        "formal_infra",
    }:
        result = RepairDispatchResult(
            status="waiting_capability",
            reason=f"owner_required:{request.blocker.blocker_class}",
            **fields,
        )
        return _record(journal, request, result) if result != previous else result
    reason = (
        "agent_adapter_not_configured"
        if executor is None
        else executor.capability(request)
    )
    if reason:
        result = RepairDispatchResult(
            status="waiting_capability", reason=reason, **fields
        )
        return _record(journal, request, result) if result != previous else result
    starts = [
        event
        for event in events
        if event["event_type"] == "repair_started"
        and event.get("detail", {}).get("fingerprint") == request.blocker.fingerprint()
    ]
    if any(event["detail"]["attempt_id"] == request.attempt_id for event in starts):
        return _record(
            journal,
            request,
            RepairDispatchResult(
                status="waiting_diagnosis",
                reason="started_attempt_requires_reconciliation",
                **fields,
            ),
        )
    assert request.grant is not None and executor is not None
    reserved = sum(float(event["detail"]["reserved_seconds"]) for event in starts)
    reserved += sum(
        float(event["detail"]["reserved_seconds"])
        for event in events
        if event["event_type"] == "operation_diagnosis_started"
        and event["detail"]["grant_digest"] == request.grant.digest()
    )
    allocation = request.grant.interrupt_seconds + KILL_GRACE_SECONDS
    if (
        len(starts) >= request.grant.max_attempts
        or reserved + allocation > request.grant.total_seconds
    ):
        return _record(
            journal,
            request,
            RepairDispatchResult(
                status="waiting_capability", reason="repair_grant_exhausted", **fields
            ),
        )
    artifact = journal.write_artifact("repair_requests", request)
    journal.append_event(
        "repair_started",
        experiment_id=request.activity_id,
        artifact_sha256=artifact.stem,
        detail={
            "fingerprint": request.blocker.fingerprint(),
            "attempt_id": request.attempt_id,
            "request_digest": request.digest(),
            "fence": request.fence,
            "reserved_seconds": allocation,
        },
    )
    try:
        result = executor.execute(request, progress=progress, cancelled=cancelled)
    except AgentCancelled:
        result = RepairDispatchResult(
            status="cancelled", reason="controller_cancelled", **fields
        )
    except (Exception, SystemExit) as exc:
        # Deliberately omit exception text: provider output may contain secrets.
        result = RepairDispatchResult(
            status="waiting_diagnosis",
            reason=f"executor_fault:{type(exc).__name__}",
            **fields,
        )
    if not fence_valid(request.fence):
        raise ValueError("stale repair output quarantined; lease revoked")
    return _record(journal, request, result)


def accept_verification(
    request: RepairRequest,
    proposal: RepairProposal,
    verification: RepairVerification,
    *,
    journal: CampaignStore,
    authenticated: Callable[[RepairVerification], bool],
    fence_valid: Callable[[str], bool],
    binding: VerificationBinding | None = None,
) -> RepairDispatchResult:
    """Validate trusted channel evidence; do not accept worker-provided booleans.

    `authenticated` belongs to the independent verifier/controller. It must
    resolve its own stored result by digest, not trust a caller's signature key.
    Successful return still requires the publisher to CAS the verified release.
    """
    grant = request.grant
    current_fence = binding.fence if binding is not None else request.fence
    binding_valid = _binding_valid(binding, request, proposal, verification, journal)
    identities_match = (
        journal.campaign_id == request.campaign_id
        and proposal.request_digest == request.digest()
        and verification.request_digest == request.digest()
        and verification.proposal_digest == proposal.digest()
        and verification.manifest_digest == request.verification_manifest_digest
        and verification.source_digest == request.blocker.source_digest
        and verification.environment_digest == request.blocker.environment_digest
        and verification.input_digest == request.blocker.input_digest
        and verification.release_digest == proposal.tree_digest
        and verification.fence == current_fence
        and binding_valid
        and grant is not None
        and verification.grant_id == grant.grant_id
        and grant.expires_at > time.time()
    )
    restored = all(
        (
            verification.original_failure_reproduced,
            verification.original_predicate_restored,
            verification.required_checks_passed,
            verification.protected_surfaces_unchanged,
        )
    )
    accepted = (
        identities_match
        and restored
        and proposal.classification == "implementation"
        and fence_valid(current_fence)
        and authenticated(verification)
    )
    result = RepairDispatchResult(
        status="verified" if accepted else "rejected",
        request_digest=request.digest(),
        reason="verified_release_requires_publication"
        if accepted
        else "verification_rejected",
        proposal=proposal,
        verification=verification,
    )
    if not fence_valid(current_fence):
        raise ValueError("stale verifier output quarantined")
    return _record(journal, request, result)


def _binding_valid(binding, request, proposal, verification, journal) -> bool:
    if binding is None:
        return verification.authority_binding_digest is None
    return bool(
        request.grant is not None
        and binding.request_digest == request.digest()
        and binding.proposal_digest == proposal.digest()
        and binding.grant_digest == request.grant.digest()
        and verification.authority_binding_digest == binding.digest()
        and any(
            event["event_type"] == "repair_verification_bound"
            and event.get("detail") == binding.model_dump(mode="json")
            for event in journal.verify_event_chain()
        )
    )
