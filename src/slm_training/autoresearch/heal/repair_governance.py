"""Controller-owned metadata reconciliation after an isolated repair exits."""

from __future__ import annotations

import hashlib
import json

from slm_training.autoresearch.heal.dispatch import _record
from slm_training.autoresearch.heal.isolation_workspace import (
    manifest_digest,
    patch_manifest_digest,
    tree_manifest,
)
from slm_training.autoresearch.heal.repair_acceptance import proposal_patch_digest
from slm_training.autoresearch.heal.repair_scope import apply_version_overlay, require_routine_scope
from slm_training.lineage.records import canonical_json

_REGISTRY = "src/slm_training/resources/versions.json"


def _recorded_overlay(journal, request_digest, proposal_digest):
    for event in reversed(journal.verify_event_chain()):
        if event["event_type"] != "repair_governance_overlay":
            continue
        path = journal.root / "artifacts/repair_governance_overlays" / (
            event["artifact_sha256"] + ".json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if hashlib.sha256(canonical_json(payload).encode()).hexdigest() != event["artifact_sha256"]:
            raise ValueError("repair governance artifact integrity failure")
        if (payload.get("request_digest"), payload.get("governed_proposal_digest")) == (
            request_digest, proposal_digest,
        ):
            return True
    return False


def reconcile_version_overlay(request, result, workspace, journal):
    """Apply governance after worker teardown and retain the worker proposal."""
    proposal = result.proposal
    assert proposal is not None
    base = tree_manifest(workspace.base)
    candidate = tree_manifest(workspace.candidate)
    tree = manifest_digest(candidate)
    patch = proposal_patch_digest(workspace.base, workspace.candidate)
    before, after = base, candidate
    changed = tuple(sorted(
        path for path in before.keys() | after.keys()
        if before.get(path) != after.get(path)
        and not (path not in before and (workspace.candidate / path).is_dir())
    ))
    if (tree, patch) == (proposal.tree_digest, proposal.patch_digest) and _REGISTRY in changed:
        require_routine_scope(
            workspace.base, workspace.candidate, changed,
            request.allowed_paths, request.semantics_preserving_paths,
            request_digest=request.digest(),
        )
        if not _recorded_overlay(journal, request.digest(), proposal.digest()):
            raise ValueError("controller version overlay has no journal provenance")
        return result
    if (tree, patch) != (proposal.tree_digest, proposal.patch_digest):
        worker = dict(candidate)
        worker[_REGISTRY] = base[_REGISTRY]
        require_routine_scope(
            workspace.base, workspace.candidate, changed,
            request.allowed_paths, request.semantics_preserving_paths,
            request_digest=request.digest(),
        )
        if (
            manifest_digest(worker), patch_manifest_digest(base, worker)
        ) != (proposal.tree_digest, proposal.patch_digest):
            raise ValueError("repair proposal differs from candidate before governance")
    else:
        require_routine_scope(
            workspace.base, workspace.candidate, changed,
            request.allowed_paths, request.semantics_preserving_paths,
        )
    if _REGISTRY not in changed and not apply_version_overlay(
        workspace.base, workspace.candidate, changed, request.digest()
    ):
        return result
    governed = proposal.model_copy(update={
        "tree_digest": manifest_digest(tree_manifest(workspace.candidate)),
        "patch_digest": proposal_patch_digest(workspace.base, workspace.candidate),
    })
    updated = result.model_copy(update={"proposal": governed})
    artifact = journal.write_artifact("repair_governance_overlays", {
        "schema_version": "repair_governance_overlay/v1",
        "request_digest": request.digest(),
        "worker_proposal_digest": proposal.digest(),
        "governed_proposal_digest": governed.digest(),
        "candidate_digest": governed.tree_digest,
    })
    journal.append_event(
        "repair_governance_overlay",
        artifact_sha256=artifact.stem,
        idempotency_key="repair-governance-overlay:" + governed.digest(),
    )
    return _record(journal, request, updated)
