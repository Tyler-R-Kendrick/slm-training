"""Controller-owned remote document verification and bounded reader transport."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from slm_training.autoresearch.action_dependencies import _materialized_delivery
from .operations_verification import require_local_gate as require_local_gate


def validate_completion(store, wait, reconciliation, *, completed=False):
    if wait.get("kind") == "verified_repair_source":
        from scripts.github_source_delivery import source_completion
        source_completion(store, wait, reconciliation)
        if completed:
            from slm_training.autoresearch.heal.repair_delivered import resolve_delivered_activation
            resolve_delivered_activation(store, reconciliation["delivered_activation"])
    return reconciliation


async def _connector_read(connector, tool, **arguments):
    """Read through the controller's configured GitHub connector, never a worker."""
    result = await connector(tool, arguments)
    if result.get("isError") or not isinstance(result.get("structuredContent"), dict):
        raise ValueError("connector_read_failed")
    return result["structuredContent"]


def _changes_requested(reviews):
    latest = {}
    for row in sorted(
        reviews, key=lambda r: (r.get("submitted_at", ""), r.get("id", 0))
    ):
        state = row.get("state")
        if state in {"COMMENTED", "PENDING"}:
            continue
        author = (row.get("user") or {}).get("login")
        if not author or state not in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}:
            raise ValueError("remote_review_identity_or_state_missing")
        latest[author.casefold()] = state
    return "CHANGES_REQUESTED" in latest.values()


async def _remote_delivery(connector, repository, proposal, files, required_checks):
    import asyncio

    if (
        required_checks is None
        or type(proposal.get("pr_number")) is not int
        or proposal["pr_number"] <= 0
    ):
        raise ValueError("remote_delivery_requires_pr_and_trusted_required_checks")
    number = proposal["pr_number"]
    pr = await _connector_read(
        connector,
        "github_get_pr_info",
        repository_full_name=repository,
        pr_number=number,
    )
    if (
        pr.get("merged") is not True
        or pr.get("base") != "main"
        or pr.get("head_sha") != proposal.get("verified_head_sha")
        or pr.get("merge_commit_sha") != proposal.get("merge_sha")
    ):
        raise ValueError("remote_merge_identity_mismatch")
    checks, threads, reviews, ancestry = await asyncio.gather(
        _connector_read(
            connector,
            "github_get_commit_combined_status",
            repo_full_name=repository,
            commit_sha=pr["head_sha"],
        ),
        _connector_read(
            connector,
            "github_list_pull_request_review_threads",
            repo_full_name=repository,
            pr_number=number,
        ),
        _connector_read(
            connector,
            "github_list_pull_request_reviews",
            repo_full_name=repository,
            pr_number=number,
        ),
        _connector_read(
            connector,
            "github_compare_commits",
            repo_full_name=repository,
            base=pr["merge_commit_sha"],
            head="main",
        ),
    )
    statuses = {row["context"]: row["state"] for row in checks["statuses"]}
    if any(statuses.get(name) != "success" for name in required_checks):
        raise ValueError("remote_required_checks_incomplete")
    if (
        any(row.get("isResolved") is not True for row in threads["review_threads"])
        or _changes_requested(reviews["reviews"])
        or ancestry.get("status") not in {"ahead", "identical"}
        or ancestry.get("merge_base_commit", {}).get("sha") != pr["merge_commit_sha"]
    ):
        raise ValueError("remote_review_or_ancestry_incomplete")
    contents = await asyncio.gather(
        *[
            _connector_read(
                connector,
                "github_fetch_file",
                repository_full_name=repository,
                path=name,
                ref=pr["merge_commit_sha"],
                encoding="utf-8",
            )
            for name in files
        ]
    )
    if any(
        result.get("encoding") != "utf-8" or result.get("content") != expected
        for result, expected in zip(contents, files.values(), strict=True)
    ):
        raise ValueError("remote_document_content_mismatch")
    return {
        "repository": repository,
        "pr_number": number,
        "head_sha": pr["head_sha"],
        "merge_sha": pr["merge_commit_sha"],
        "required_checks": list(required_checks),
        "checks": statuses,
        "files": {
            name: hashlib.sha256(value.encode()).hexdigest()
            for name, value in files.items()
        },
    }


def _delivery_subject(root, wait):
    import re
    from slm_training.autoresearch.schemas import AutotrainCycleHandoffV1
    from slm_training.autoresearch.storage import CampaignStore

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", wait.get("campaign_id", "")
    ):
        raise ValueError("invalid_delivery_campaign")
    store = CampaignStore(wait["campaign_id"], root)
    handoff = AutotrainCycleHandoffV1.model_validate_json(
        (store.root / "cycle_handoff.json").read_text()
    )
    index = wait.get("action_index")
    if type(index) is not int or not 0 <= index < len(handoff.actions):
        raise ValueError("invalid_delivery_action_index")
    action = handoff.actions[index]
    materialized = _materialized_delivery(store, handoff)
    if (
        action.kind != "document"
        or action.dependency_scope != "delivery"
        or materialized is None
        or {**materialized, "action_index": index} != wait
    ):
        raise ValueError("delivery_action_or_materialization_changed")
    path = (
        store.root
        / "artifacts/delivery_documents"
        / f"{materialized['artifact_sha256']}.json"
    )
    return store, handoff, action, json.loads(path.read_text())


async def reconcile_document_delivery(
    root,
    wait,
    proposal,
    *,
    repository,
    required_checks,
    connector,
    publish_context=None,
):
    """Fresh connector reads discharge only the exact current document action.

    connector is the trusted controller's async (tool_name, args) capability,
    not a subprocess response or an agent-supplied callback. No host capability
    means no receipt. An immutable verification remains historical evidence.
    """
    import asyncio
    from slm_training.levers import INTERRUPT_AFTER_SECONDS
    from slm_training.autoresearch.schemas import AutotrainActionReceiptV1
    from slm_training.autoresearch.storage import (
        append_autotrain_action_receipt,
        autotrain_action_sha256,
        bind_autotrain_action_evidence,
        pending_autotrain_actions,
    )

    store, handoff, action, materialized = _delivery_subject(root, wait)
    if connector is None:
        raise ValueError("independent_connector_read_capability_unavailable")
    async with asyncio.timeout(INTERRUPT_AFTER_SECONDS):
        remote = await _remote_delivery(
            connector, repository, proposal, materialized["files"], required_checks
        )
    _, current, _, current_materialized = _delivery_subject(root, wait)
    if current != handoff or current_materialized != materialized:
        raise ValueError("delivery_changed_during_remote_verification")
    from contextlib import nullcontext

    with publish_context() if publish_context is not None else nullcontext():
        proof = {
            "schema": "connector_document_verification/v1",
            "wait": wait,
            "action_sha256": autotrain_action_sha256(action),
            "remote": remote,
        }
        artifact = store.write_artifact("connector_document_verifications", proof)
        store.append_event(
            "connector_document_verified",
            artifact_sha256=artifact.stem,
            idempotency_key="connector-document:" + artifact.stem,
        )
        uri = str(artifact.relative_to(store.root))
        receipt = AutotrainActionReceiptV1(
            loop_id=handoff.loop_id,
            campaign_id=handoff.campaign_id,
            action_index=wait["action_index"],
            action_sha256=autotrain_action_sha256(action),
            action_kind="document",
            status="completed",
            evidence_uris=(uri,),
            evidence=bind_autotrain_action_evidence(root, handoff, action, (uri,)),
        )
        if any(
            index == wait["action_index"]
            for index, _ in pending_autotrain_actions(root, handoff)
        ):
            append_autotrain_action_receipt(root, receipt)
        return {
            "verification_artifact": str(artifact),
            "receipt": receipt.model_dump(mode="json"),
        }


READ_TOOLS = frozenset(
    {
        "github_get_pr_info",
        "github_get_commit_combined_status",
        "github_list_pull_request_review_threads",
        "github_list_pull_request_reviews",
        "github_compare_commits",
        "github_fetch_file",
        "github_fetch",
    }
)


def reader_ready(host, *, seconds=30.0):
    if (
        host.read_command is None
        or len(host.read_command) != 1
        or host.reader_sha256 is None
        or host.required_checks is None
    ):
        return False
    reader = Path(host.read_command[0])
    try:
        if (
            not reader.is_absolute()
            or reader.is_symlink()
            or reader.resolve() == Path(host.command[0]).resolve()
            or hashlib.sha256(reader.read_bytes()).hexdigest() != host.reader_sha256
        ):
            return False
        require_local_gate(host, seconds=seconds)
    except (OSError, ValueError, KeyError):
        return False
    return True


def reconcile_with_host(runtime, lease, wait, proposal, host):
    """Reachable bounded read transport; no writer response is a remote read."""
    import asyncio
    from scripts.github_source_preparation import delivery_environment
    from slm_training.harness_core.activity_contract import contract_digest
    from slm_training.harness_core.bounded_process import ProcessOutcome

    import time
    from slm_training.levers import KILL_GRACE_SECONDS

    remaining = lease.expires_at - time.time() - KILL_GRACE_SECONDS - 1
    if (
        remaining <= 0
        or wait.get("kind") not in {"document", "verified_repair_source"}
        or not reader_ready(host, seconds=remaining)
    ):
        raise ValueError("independent_connector_read_capability_unavailable")
    root = runtime.store.root.parents[2]
    attempt = runtime.attempt_dir(lease)
    sequence = 0

    async def connector(tool, arguments):
        nonlocal sequence
        if tool not in READ_TOOLS:
            raise ValueError("connector_reader_method_not_allowed")
        sequence += 1
        request = {
            "schema_version": "connector_read_request/v1",
            "tool": tool,
            "arguments": arguments,
            "lease": lease.model_dump(mode="json"),
            "wait": wait,
        }
        input_path, output_path = (
            attempt / f"read-{sequence}-{kind}.json" for kind in ("request", "response")
        )
        runtime.store._replace_durable(input_path, json.dumps(request))
        env = delivery_environment(host)
        result = runtime.run(
            lease,
            [
                *host.read_command,
                "--request",
                str(input_path),
                "--output",
                str(output_path),
            ],
            cwd=attempt,
            env=env,
        )
        if result.outcome != ProcessOutcome.COMPLETED or result.returncode != 0:
            raise ValueError("independent_connector_read_failed")
        if output_path.is_symlink() or output_path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("unsafe_connector_read_response")
        response = json.loads(output_path.read_text())
        if (
            response.get("request_digest") != contract_digest(request)
            or response.get("provider") != "github_connector"
            or response.get("read_only") is not True
        ):
            raise ValueError("independent_connector_read_binding_mismatch")
        return response["result"]

    if wait.get("kind") == "verified_repair_source":
        from scripts.github_source_delivery import reconcile_source_delivery

        operation = reconcile_source_delivery(runtime, lease, wait, proposal, host, connector)
    else:
        operation = reconcile_document_delivery(
            root,
            wait,
            proposal,
            repository=host.repository,
            required_checks=host.required_checks,
            connector=connector,
            publish_context=lambda: runtime.publication(lease),
        )
    result = asyncio.run(operation)

    runtime.store._replace_durable(
        attempt / "domain-reconciliation.json", json.dumps(result)
    )
    return validate_completion(runtime.store, wait, result)
