"""Accepted repair delivery: controller proof to an exact remote-base successor.

The heal owner authenticates publication and scope. This bridge authenticates
the *delivery* candidate against that proof; repair's synthetic-base gate is
never substituted for the canonical gate over the true remote base.
"""

from pathlib import Path

from slm_training.harness_core.activity_contract import contract_digest
from slm_training.harness_core.execution_release import _files, runtime_source_identity
from slm_training.harness_core.github_delivery_remote import fetch, sha, verify_tree
from slm_training.harness_core.github_delivery_tree import source_tree, source_entries, tree_sha
from slm_training.harness_core.github_git_snapshot import commit_object


def reader_url_allowed(repository, arguments):
    """Only immutable Git commits/trees used by independent reconciliation."""
    import re

    if set(arguments) != {"url"} or not isinstance(arguments["url"], str):
        return False
    prefix = "https://api.github.com/repos/" + repository + "/git/"
    return re.fullmatch(re.escape(prefix) +
                        r"(?:commits/[a-f0-9]{40}|trees/[a-f0-9]{40}\?recursive=1)",
                        arguments["url"]) is not None


def source_binding(store, wait, host):
    from scripts.merge_verification_evidence import source_paths
    from slm_training.autoresearch.heal.repair_delivery import resolve_source_delivery

    subject = resolve_source_delivery(store, wait)
    execution = Path(subject["successor_execution"]).resolve(strict=True)
    predecessor = Path(subject["predecessor"]["execution"]).resolve(strict=True)
    plan = host.verification_plan or {}
    if (
        subject["successor_source_digest"] != host.source_digest
        or runtime_source_identity(execution) != host.source_digest
        or runtime_source_identity(predecessor) != subject["predecessor"]["runtime_source_digest"]
        or Path(plan.get("execution", "")).resolve() != execution
    ):
        raise ValueError("accepted_source_delivery_execution_binding_mismatch")
    source = Path(plan["source"]).resolve(strict=True)
    # A documentation successor is a separate materialized, gated dependency.
    # Do not infer extra document authority from a repair's allowed-path scope.
    if plan.get("delivery_documents_sha256"):
        raise ValueError("source_delivery_requires_exact_accepted_candidate")
    tree = source_tree(
        source, host.base_ref,
        predecessor=(predecessor, _files(predecessor)),
        candidate=(execution, _files(execution)),
        allowed_paths=subject["scope"]["changes"], paths=source_paths(source),
    )
    return {**tree, "publication_id": wait["publication_id"],
            "artifact_sha256": wait["artifact_sha256"], "source_digest": host.source_digest,
            "repository": host.repository, "base_ref": host.base_ref,
            "verification_identity": plan["identity"],
            "verification_plan_sha256": contract_digest(plan)}


async def verify_source_remote(connector, binding, proposal, required_checks):
    from slm_training.autoresearch.runtime.operations_reconciliation import _remote_delivery

    # This existing domain predicate binds PR identity, reviews, checks and main
    # ancestry. Full-tree reads additionally cover links, modes and deletions.
    remote = await _remote_delivery(
        connector, binding["repository"], proposal, binding["files"], required_checks
    )
    base = await fetch(connector, binding["repository"], "git/commits/" + sha(binding["base_ref"]))
    if base["tree"]["sha"] != binding["base_git_tree"]:
        raise ValueError("source_delivery_remote_base_tree_mismatch")
    for revision in (proposal["verified_head_sha"], proposal["merge_sha"]):
        commit = await fetch(connector, binding["repository"], "git/commits/" + sha(revision))
        if ([row["sha"] for row in commit["parents"]] != [binding["base_ref"]]
                or commit["tree"]["sha"] != binding["candidate_git_tree"]):
            raise ValueError("source_delivery_remote_parent_or_tree_mismatch")
    await verify_tree(connector, binding["repository"], binding["candidate_git_tree"])
    return {**remote, "merge_commit_object": commit_object(commit, proposal["merge_sha"]),
            **{name: binding[name] for name in (
        "publication_id", "artifact_sha256", "source_digest", "base_ref",
        "base_git_tree", "candidate_git_tree", "verification_identity", "verification_plan_sha256",
    )}}


async def reconcile_source_delivery(runtime, lease, wait, proposal, host, connector):
    binding = source_binding(runtime.store, wait, host)
    remote = await verify_source_remote(connector, binding, proposal, host.required_checks)
    if source_binding(runtime.store, wait, host) != binding:
        raise ValueError("accepted_source_changed_during_remote_reconciliation")
    with runtime.publication(lease):
        proof = {"schema_version": "verified_repair_source_delivery_receipt/v1",
                 "wait": wait, "remote": remote}
        artifact = runtime.store.write_artifact("verified_repair_source_delivery_receipts", proof)
        runtime.store.append_event(
            "verified_repair_source_delivered", artifact_sha256=artifact.stem,
            experiment_id=lease.activity_id, idempotency_key="source-delivered:" + artifact.stem,
            detail={"publication_id": wait["publication_id"],
                    "request_artifact_sha256": wait["artifact_sha256"]},
        )
        from slm_training.autoresearch.heal.repair_delivered import record_delivered_activation

        reconciliation = {"verification_artifact": str(artifact), "receipt": proof}
        reconciliation["delivered_activation"] = record_delivered_activation(
            runtime, lease, wait, reconciliation
        )
    return reconciliation


def source_completion(store, wait, reconciliation):
    """Activation boundary: authenticate accepted subject and fenced remote proof.

    The parent additionally requires the delivery activity's committed success.
    This function never edits the accepted handoff or immutable release marker.
    """
    import json
    import re
    from slm_training.autoresearch.heal.repair_delivery import resolve_source_delivery
    from slm_training.autoresearch.storage import _sha

    subject = resolve_source_delivery(store, wait)
    proof = reconciliation["receipt"]
    identity = _sha(proof)
    path = store.root / "artifacts/verified_repair_source_delivery_receipts" / (identity + ".json")
    if (Path(reconciliation["verification_artifact"]) != path
            or path.is_symlink() or json.loads(path.read_text()) != proof
            or proof.get("schema_version") != "verified_repair_source_delivery_receipt/v1"
            or proof.get("wait") != wait):
        raise ValueError("source_delivery_completion_artifact_mismatch")
    expected = {"publication_id": wait["publication_id"], "request_artifact_sha256": wait["artifact_sha256"]}
    if not any(event["event_type"] == "verified_repair_source_delivered"
               and event.get("artifact_sha256") == identity and event["detail"] == expected
               for event in store.verify_event_chain()):
        raise ValueError("source_delivery_completion_not_fenced")
    remote = proof["remote"]
    for key, value in {"publication_id": wait["publication_id"],
                       "artifact_sha256": wait["artifact_sha256"],
                       "source_digest": subject["successor_source_digest"]}.items():
        if remote.get(key) != value:
            raise ValueError("source_delivery_completion_publication_mismatch")
    for name in ("base_ref", "base_git_tree", "candidate_git_tree", "head_sha", "merge_sha"):
        sha(remote[name])
    from slm_training.harness_core.github_delivery_tree import git_object
    if git_object("commit", remote["merge_commit_object"].encode()) != remote["merge_sha"]:
        raise ValueError("source_delivery_completion_commit_object_mismatch")
    if any(not re.fullmatch(r"[a-f0-9]{64}", remote.get(key, ""))
           for key in ("verification_identity", "verification_plan_sha256")):
        raise ValueError("source_delivery_completion_gate_mismatch")
    execution = Path(subject["successor_execution"])
    if tree_sha(source_entries(execution, _files(execution))) != remote["candidate_git_tree"]:
        raise ValueError("source_delivery_completion_candidate_tree_mismatch")
    return dict(remote)
