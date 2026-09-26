"""Pinned host entrypoint: --mode writer|reader --config PATH --config-sha256 SHA.

DeliveryHost commands must be distinct immutable launchers pinning mode and
import roots. The config digest comes from trusted service configuration (not
campaign input); this avoids a hash cycle through DeliveryHost executable pins.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from slm_training.autoresearch.runtime.activity_projection import ActivityProjection
from slm_training.autoresearch.runtime.operations_delivery import DeliveryHost
from slm_training.autoresearch.runtime.operations_reconciliation import (
    READ_TOOLS,
    _delivery_subject,
    _remote_delivery,
    require_local_gate,
)
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.activity_contract import ActivityLease, contract_digest
from slm_training.harness_core.github_connector import (
    ConnectorConfig,
    ConnectorRejected,
    DeliveryWaiting,
    host_connector,
)
from slm_training.harness_core.github_document_delivery import (
    DocumentDelivery,
    WRITE_TOOLS,
    WRITER_READ_TOOLS,
)
from slm_training.levers import INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS


class AdapterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    host: DeliveryHost
    transport: ConnectorConfig
    runtime_root: str
    state_dir: str
    reviewers: tuple[str, ...]
    review_note_authors: tuple[str, ...] = Field(min_length=1)


def load_config(path, digest):
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("unsafe_host_config")
    raw = path.read_bytes()
    if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("trusted_host_config_changed")
    return AdapterConfig.model_validate_json(raw)


def current_lease(config, request):
    lease = ActivityLease.model_validate(request["lease"])
    host = config.host
    if not host.authorized or min(host.expires_at, lease.expires_at) <= time.time():
        raise DeliveryWaiting("delivery_grant_or_lease_expired")
    root = Path(config.runtime_root).resolve()
    state = (
        ActivityProjection(CampaignStore(root.name, root.parent))
        .read()
        .get(lease.activity_id)
    )
    initial = request.get("source_request")
    if initial is not None:
        from scripts.github_source_authority import initial_source_request
        if initial != initial_source_request(host):
            raise ValueError("initial_source_host_binding_mismatch")
    if (
        state is None
        or state.lease != lease
        or state.status != "running"
        or state.spec.kind != ("verify" if initial is not None else "delivery")
        or state.spec.source_digest != host.source_digest
        or ("authorized_github_connector_read" if initial is not None else
            "authorized_github_connector_delivery") not in state.spec.capabilities
        or initial is not None and state.spec.input_digest != contract_digest(initial)
    ):
        raise ValueError("connector_lease_not_current")
    return state


def document_binding(config, request):
    state = current_lease(config, request)
    host, wait = config.host, request["wait"]
    if host.required_checks is None:
        raise DeliveryWaiting("trusted_required_checks_missing")
    identity = "delivery:" + contract_digest(
        {
            "input": contract_digest(wait),
            "source": host.source_digest,
            "repository": host.repository,
            "base_ref": host.base_ref,
            **({"base_branch": host.base_branch} if host.base_branch != "main" else {}),
        }
    )
    expected = {
        "schema_version": "connector_delivery_request/v1",
        "operation": "deliver_and_reconcile_squash_merge",
        "repository": host.repository,
        "base_ref": host.base_ref,
        **({"base_branch": host.base_branch} if host.base_branch != "main" else {}),
        "source_digest": host.source_digest,
        "runtime_root": str(Path(config.runtime_root).resolve()),
        "grant_expires_at": host.expires_at,
        "idempotency_key": identity,
    }
    if any(
        request.get(k) != v for k, v in expected.items()
    ) or request.get("base_branch", "main") != host.base_branch or state.spec.input_digest != contract_digest(wait):
        raise ValueError("connector_delivery_request_binding_mismatch")
    data = _subject_binding(config, wait)
    marker = "slm-delivery-" + identity.split(":")[1]
    return {
        "repository": host.repository, "base_ref": host.base_ref,
        **({"base_branch": host.base_branch} if host.base_branch != "main" else {}),
        "source_digest": host.source_digest, "wait": wait, **data,
        "branch": "autotrain/" + marker, "marker": marker,
        "message": data.get("title", "Publish measured campaign documents") + "\n\n" + marker,
        "required_checks": list(host.required_checks),
        "reviewers": list(config.reviewers),
        "review_note_authors": list(config.review_note_authors),
        "host_policy_digest": contract_digest(config),
    }


def _subject_binding(config, wait):
    host = config.host
    if wait.get("kind") == "verified_repair_source":
        from scripts.github_source_delivery import source_binding

        root = Path(config.runtime_root).resolve()
        return {**source_binding(CampaignStore(root.name, root.parent), wait, host),
                "title": "Publish verified source repair"}
    _, handoff, _, materialized = _delivery_subject(
        Path(config.runtime_root).resolve().parents[2], wait
    )
    allowed = {
        f"docs/design/{handoff.campaign_id}-results.{suffix}"
        for suffix in ("md", "json")
    }
    ledger = (
        "src/slm_training/resources/experiments/autotrain_climb/evidence_ledger.v1.json"
    )
    if ledger not in materialized["files"]:
        raise DeliveryWaiting("legacy_document_ledger_rematerialization_required")
    allowed.add(ledger)
    if handoff.checkpoint_documentation_required:
        allowed.update({"README.md", "docs/MODEL_CARD.md"})
    if set(materialized["files"]) != allowed:
        raise ValueError("document_materialization_exceeds_authorized_scope")
    from scripts.merge_verification_evidence import source_paths
    from slm_training.harness_core.github_delivery_tree import document_tree

    if not host.verification_plan:
        raise DeliveryWaiting("authenticated_local_merge_gate_missing")
    approved = host.verification_plan.get("delivery_documents_sha256")
    hashes = {
        name: hashlib.sha256(value.encode()).hexdigest()
        for name, value in materialized["files"].items()
    }
    if approved != hashes:
        raise DeliveryWaiting("exact_document_successor_scope_and_full_gate_required")
    source = Path(host.verification_plan["source"])
    candidate_tree = document_tree(
        source, host.base_ref, materialized["files"], source_paths(source)
    )
    return {"files": materialized["files"], "candidate_git_tree": candidate_tree}


async def write_request(config, request, connector):
    binding = document_binding(config, request)
    directory = Path(config.state_dir)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.is_symlink() or directory.stat().st_mode & 0o077:
        raise ValueError("delivery_journal_requires_private_host_directory")
    path = directory / (binding["marker"] + ".json")
    lock = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.is_symlink():
            raise ValueError("unsafe_delivery_journal")
        journal = json.loads(path.read_text()) if path.exists() else {}

        def validate(*, gate=False):
            if document_binding(config, request) != binding:
                raise ValueError("document_materialization_changed")
            if gate:
                require_local_gate(config.host)

        async def verify(proposal):
            if request["wait"].get("kind") == "verified_repair_source":
                from scripts.github_source_delivery import verify_source_remote

                await verify_source_remote(connector, binding, proposal, config.host.required_checks)
                return
            await _remote_delivery(
                connector,
                config.host.repository,
                proposal,
                binding["files"],
                config.host.required_checks, base_branch=config.host.base_branch,
            )

        writer = DocumentDelivery(
            binding,
            journal,
            connector=connector,
            transport=config.transport,
            save=lambda data: CampaignStore._replace_durable(path, json.dumps(data)),
            validate=validate,
            verify=verify,
        )
        proposal = await writer.run()
    finally:
        os.close(lock)
    return {
        "schema_version": "connector_delivery_receipt/v1",
        "provider": "github_connector",
        "request_digest": contract_digest(request),
        "repository": config.host.repository,
        "source_digest": config.host.source_digest,
        "merged": True,
        "checks_verified": True,
        "reviews_resolved": True,
        "content_verified": True,
        **proposal,
    }


async def read_request(config, request, connector):
    if request.get("schema_version") != "connector_read_request/v1":
        raise ValueError("connector_read_schema_mismatch")
    current_lease(config, request)
    tool, arguments = request["tool"], request["arguments"]
    if tool not in READ_TOOLS:
        raise ValueError("connector_reader_method_not_allowed")
    repositories = [
        arguments[k]
        for k in ("repository_full_name", "repo_full_name")
        if k in arguments
    ]
    if tool == "github_fetch":
        from scripts.github_source_delivery import reader_url_allowed

        initial = request.get("source_request")
        from urllib.parse import quote
        initial_ref = ("https://api.github.com/repos/" + config.host.repository
                       + "/git/ref/heads/" + quote(config.host.base_branch, safe="/"))
        if not (initial is not None and arguments == {"url": initial_ref}) and not reader_url_allowed(config.host.repository, arguments):
            raise ValueError("connector_reader_repository_mismatch")
    elif repositories != [config.host.repository]:
        raise ValueError("connector_reader_repository_mismatch")
    result = await connector(tool, arguments)
    current_lease(config, request)
    if result.get("isError") or not isinstance(result.get("structuredContent"), dict):
        raise ConnectorRejected("connector_read_rejected")
    return {
        "provider": "github_connector",
        "read_only": True,
        "request_digest": contract_digest(request),
        "result": result,
    }


async def execute(args):
    config = load_config(args.config, args.config_sha256)
    if args.request.is_symlink() or args.request.stat().st_size > 1024 * 1024:
        raise ValueError("unsafe_connector_request")
    request = json.loads(args.request.read_text())
    from scripts.github_source_preparation import delivery_host

    root = Path(config.runtime_root).resolve()
    config = config.model_copy(update={"host": delivery_host(
        CampaignStore(root.name, root.parent), request.get("wait", {}), config.host)})
    allowed = READ_TOOLS if args.mode == "reader" else WRITE_TOOLS | WRITER_READ_TOOLS
    try:
        remaining = (
            min(config.host.expires_at, request["lease"]["expires_at"])
            - time.time()
            - KILL_GRACE_SECONDS
        )
        if remaining <= 0:
            raise DeliveryWaiting("delivery_grant_or_lease_expired")
        seconds = min(INTERRUPT_AFTER_SECONDS, remaining)
        async with asyncio.timeout(seconds):
            async with host_connector(
                config.transport,
                allowed_tools=allowed,
                timeout_seconds=seconds,
            ) as connector:
                result = await (
                    read_request if args.mode == "reader" else write_request
                )(config, request, connector)
        code = 0
    except (DeliveryWaiting, ConnectorRejected, TimeoutError) as error:
        result = {
            "schema_version": "connector_delivery_wait/v1",
            "provider": "github_connector",
            "request_digest": contract_digest(request),
            "state": "waiting_capability",
            "reason": str(error) or "connector_deadline_exceeded",
        }
        code = 20 if isinstance(error, ConnectorRejected) else 10
    if args.output.is_symlink():
        raise ValueError("unsafe_connector_output")
    CampaignStore._replace_durable(args.output, json.dumps(result))
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("writer", "reader"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return asyncio.run(execute(args))
    except Exception as error:
        # Transport and validation exceptions can contain credential-bearing
        # URLs or config values. Never render their messages or traceback.
        import sys

        print(
            json.dumps(
                {
                    "state": "waiting_capability",
                    "reason": "connector_host_or_request_failed",
                    "error_type": type(error).__name__,
                }
            ),
            file=sys.stderr,
        )
        return 20


if __name__ == "__main__":
    raise SystemExit(main())


async def prepare_initial_release(runtime, lease, config):
    """Controller API using the same configured, schema-pinned read transport."""
    from scripts.github_source_authority import initial_source_request, record_initial_release

    request = {"lease": lease.model_dump(mode="json"), "source_request": initial_source_request(config.host)}
    current_lease(config, request)
    seconds = min(INTERRUPT_AFTER_SECONDS, config.host.expires_at - time.time(),
                  lease.expires_at - time.time() - KILL_GRACE_SECONDS - 1)
    async with host_connector(config.transport, allowed_tools={"github_fetch"}, timeout_seconds=seconds) as connector:
        async def read(tool, arguments):
            current_lease(config, request)
            response = await read_request(config, {**request, "schema_version": "connector_read_request/v1",
                                                  "tool": tool, "arguments": arguments}, connector)
            return response["result"]
        return await record_initial_release(runtime, lease, config.host, read)
