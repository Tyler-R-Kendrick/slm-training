"""Controller-only local successor publication, fenced by ActivityRuntime.

Verified work, source activation and scientific promotion are separate events.
This publisher never imports candidate code, edits shared Git refs, or starts a service.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

from slm_training.harness_core.activity_contract import contract_digest
from slm_training.autoresearch.heal.isolation_workspace import (
    manifest_digest,
    private_snapshot,
    tree_manifest,
)
from slm_training.autoresearch.heal.repair_contracts import (
    RepairDispatchResult,
    RepairRequest,
)
from slm_training.harness_core.execution_release import (
    MARKER,
    prepare_release,
    runtime_source_identity,
)
from slm_training.harness_core.lineage.store import _atomic_write


def source_verification_callback(context, config):
    """Resolve full source coverage from pinned inputs, never execute its queue.

    Pass this controller-owned callback as ``source_verification=`` to
    ``dispatch_hard_pending``. A missing separate ResourceGrant returns no gate;
    the source verifier never borrows the agent's repair allowance. The resulting
    dependency supplies the finite verifier's root/cache/base/identity and grant.
    """
    from .repair_source_workspace import prepare_source_verification

    def resolve(request, proposal, workspace):
        if config.source_verification_grant is None:
            return None
        return prepare_source_verification(context, config, request, proposal, workspace)

    return resolve


def _authorize(request, result, lease, authenticated):
    if not authenticated(result):
        raise ValueError("untrusted_release_result")
    request = RepairRequest.model_validate(request.model_dump())
    result = RepairDispatchResult.model_validate(result.model_dump())
    receipt, proposal = result.verification, result.proposal
    if (
        result.status != "verified"
        or receipt is None
        or proposal is None
        or request.grant is None
    ):
        raise ValueError("release_verification_identity_or_predicate_mismatch")
    if proposal.classification != "implementation":
        raise ValueError("successor_measurement_or_policy_authority_required")
    pairs = (
        (result.request_digest, request.digest()),
        (proposal.request_digest, request.digest()),
        (receipt.request_digest, request.digest()),
        (receipt.proposal_digest, proposal.digest()),
        (receipt.fence, lease.token),
        (receipt.grant_id, request.grant.grant_id),
        (receipt.release_digest, proposal.tree_digest),
        (receipt.manifest_digest, request.verification_manifest_digest),
        (receipt.source_digest, request.blocker.source_digest),
        (receipt.environment_digest, request.blocker.environment_digest),
        (receipt.input_digest, request.blocker.input_digest),
    )
    predicates = (
        receipt.original_failure_reproduced,
        receipt.original_predicate_restored,
        receipt.required_checks_passed,
        receipt.protected_surfaces_unchanged,
        request.grant.expires_at > time.time(),
        lease.expires_at > time.time(),
    )
    if not all(predicates) or any(actual != expected for actual, expected in pairs):
        raise ValueError("release_verification_identity_or_predicate_mismatch")


def _sync(path: Path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _durable_tree(root: Path):
    for directory, _, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                _sync(path)
        _sync(Path(directory))


def _materialize(candidate, outputs, publication_id, releases, verified_digest):
    releases = releases.resolve()
    if releases == outputs.resolve() or releases.is_relative_to(outputs.resolve()):
        raise ValueError("release_root_must_be_outside_run_outputs")
    releases.mkdir(parents=True, exist_ok=True)
    # An interrupted stage is retained for diagnosis, never selected by 'latest'.
    stage = Path(tempfile.mkdtemp(prefix="stage-", dir=releases))
    frozen = private_snapshot(candidate, stage / "verified")
    if manifest_digest(tree_manifest(frozen)) != verified_digest:
        raise ValueError("verified_candidate_changed_during_materialization")
    manifest = prepare_release(frozen, stage / "release", stage / "execution", outputs)
    if manifest_digest(tree_manifest(frozen)) != verified_digest:
        raise ValueError("verified_candidate_changed_during_materialization")
    if manifest_digest(tree_manifest(candidate)) != verified_digest:
        raise ValueError("verified_candidate_changed_during_materialization")
    _durable_tree(stage)
    _sync(releases)
    return {
        "schema_version": "verified_local_release_pointer/v1",
        "publication_id": publication_id,
        "runtime_source_digest": manifest["source_digest"],
        "release": str(stage / "release"),
        "execution": str(stage / "execution"),
        "outputs": str(outputs.resolve()),
    }


def publish_verified_repair(
    request: RepairRequest,
    result: RepairDispatchResult,
    *,
    runtime,
    lease,
    candidate: Path,
    destinations: tuple[Path, Path],
    expected_previous: str | None,
    authenticated,
) -> dict:
    """Publish under current lease; return restart handoff, never activate it.

    ``authenticated`` is provided by trusted controller/verifier code, never the
    patch worker. ``expected_previous`` is the observed prior publication ID.
    Crash after intent or after pointer is reconciled under the same logical ID.
    A different lease needs a newly verified receipt, not an edited old receipt.
    ``destinations`` is (controller-owned release root, shared run outputs).
    """
    _authorize(request, result, lease, authenticated)
    release_root, outputs = destinations
    assert result.proposal is not None and result.verification is not None
    if manifest_digest(tree_manifest(candidate)) != result.proposal.tree_digest:
        raise ValueError("verified_candidate_changed")
    publication_id = contract_digest(
        {"request": request.digest(), "proposal": result.proposal.digest()}
    )
    pointer = runtime.store.root / "source_release_pointer.json"
    with runtime.publication(lease):
        current = json.loads(pointer.read_text()) if pointer.exists() else None
        if current and current["publication_id"] == publication_id:
            selected = _intent_or_stage(
                runtime, result, candidate, outputs, publication_id, release_root
            )
            if selected != current:
                raise ValueError(
                    "publication_pointer_differs_from_authoritative_intent"
                )
        else:
            if (current["publication_id"] if current else None) != expected_previous:
                raise ValueError("source_release_CAS_conflict")
            selected = _intent_or_stage(
                runtime, result, candidate, outputs, publication_id, release_root
            )
            if (
                runtime_source_identity(Path(selected["execution"]))
                != selected["runtime_source_digest"]
            ):
                raise ValueError("staged_release_changed")
            _authorize(request, result, lease, authenticated)
            _atomic_write(pointer, selected)
            _sync(pointer.parent)
        if (
            runtime_source_identity(Path(selected["execution"]))
            != selected["runtime_source_digest"]
        ):
            raise ValueError("published_release_changed")
        handoff = _handoff(request, selected)
        runtime.store.append_event(
            "repair_release_accepted",
            idempotency_key="release-accepted:" + publication_id,
            detail={
                "publication_id": publication_id,
                "pointer_digest": contract_digest(selected),
                "scientific_promotion": False,
                "handoff": handoff,
            },
        )
    return handoff


def _handoff(request, selected):
    return {
        "status": "successor_release_published",
        "publication_id": selected["publication_id"],
        "successor_execution": selected["execution"],
        "source_digest": selected["runtime_source_digest"],
        "resume_activity_id": request.blocked_activity_id,
        "resume_campaign_id": request.campaign_id,
        "original_request_digest": request.digest(),
        "original_input_digest": request.blocker.input_digest,
        "restart_argv_prefix": [
            sys.executable,
            "-m",
            "scripts.run_autotrain_supervisor",
        ],
        "restart_cwd": selected["execution"],
        "requires_fresh_process": True,
        "service_activated": False,
        "measurement_reuse": "same_locked_semantics_only",
        "promotion_authority": False,
    }


def pinned_repair_source(cwd: Path, expected_digest: str) -> tuple[Path, str]:
    """Resolve validated immutable input, never hash live Git or linked outputs.

    A legacy checkout requires the canonical release-preparation operation first.
    Its runtime digest and the isolation byte/mode digest are distinct identities.
    """
    if runtime_source_identity(cwd) != expected_digest:
        raise ValueError("repair_requires_pinned_execution_release")
    marker = json.loads((cwd / MARKER).read_text())
    source = Path(marker["release"]).resolve(strict=True)
    digest = manifest_digest(tree_manifest(source))
    if runtime_source_identity(cwd) != expected_digest:
        raise ValueError("repair_source_changed_during_resolution")
    return source, digest


def verified_release_callback(publisher, lease, *, destinations):
    """Trusted dispatch callback; authority and stores never enter the workload.

    At most one successor is published per invocation. Remaining repairs must
    rebase on that successor rather than overwrite it from the original source.
    """
    with publisher.publication(lease):
        pointer = publisher.store.root / "source_release_pointer.json"
        previous = (
            json.loads(pointer.read_text())["publication_id"]
            if pointer.exists()
            else None
        )
    published = False

    def publish(request, accepted_result, candidate):
        nonlocal published
        if published:
            raise ValueError("successor_requires_fresh_repair_operation")
        handoff = publish_verified_repair(
            request,
            accepted_result,
            runtime=publisher,
            lease=lease,
            candidate=candidate,
            destinations=destinations,
            expected_previous=previous,
            authenticated=lambda supplied: supplied is accepted_result,
        )
        published = True
        return handoff

    return publish


def verified_activation_handoff(store, handoff: dict) -> dict:
    """Check controller-issued publication before a parent-owned fresh restart.

    This function starts nothing and marks no blocker healed. Hashes supplement
    controller ownership, not worker authentication. Legacy acceptance events
    lacking a handoff cannot authorize activation under this contract.
    """
    pointer = json.loads((store.root / "source_release_pointer.json").read_text())
    events = store.verify_event_chain()
    identity = pointer["publication_id"]
    matching = [
        event for event in events if event["detail"].get("publication_id") == identity
    ]
    intents = [e for e in matching if e["event_type"] == "repair_release_intent"]
    accepted = [e for e in matching if e["event_type"] == "repair_release_accepted"]
    if not intents or not accepted:
        raise ValueError("release_activation_requires_controller_acceptance")
    receipt = accepted[-1]["detail"]
    if (
        intents[-1]["detail"]["pointer"] != pointer
        or receipt.get("pointer_digest") != contract_digest(pointer)
        or receipt.get("handoff") != handoff
        or receipt.get("scientific_promotion") is not False
    ):
        raise ValueError("release_activation_identity_mismatch")
    if not handoff.get("resume_activity_id") or not handoff.get("resume_campaign_id"):
        raise ValueError("release_activation_requires_original_activity")
    if (
        runtime_source_identity(Path(pointer["execution"]))
        != pointer["runtime_source_digest"]
    ):
        raise ValueError("release_activation_source_changed")
    return dict(handoff)


def _intent_or_stage(runtime, result, candidate, outputs, publication_id, release_root):
    events = runtime.store.verify_event_chain()
    prior = [
        event
        for event in events
        if event["event_type"] == "repair_release_intent"
        and event["detail"]["publication_id"] == publication_id
    ]
    if prior:
        selected = prior[-1]["detail"]["pointer"]
        if (
            selected["verified_tree_digest"] != result.proposal.tree_digest
            or Path(selected["outputs"]).resolve() != outputs.resolve()
            or not Path(selected["release"])
            .resolve()
            .is_relative_to(release_root.resolve())
        ):
            raise ValueError("publication_intent_identity_conflict")
        return selected
    selected = _materialize(
        candidate, outputs, publication_id, release_root, result.proposal.tree_digest
    )
    selected["verified_tree_digest"] = result.proposal.tree_digest
    selected["verification_digest"] = result.verification.digest()
    runtime.store.append_event(
        "repair_release_intent",
        idempotency_key="release-intent:" + publication_id,
        detail={"publication_id": publication_id, "pointer": selected},
    )
    return selected
