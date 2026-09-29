"""Bounded connector delivery; independent remote reads precede acknowledgement."""

from __future__ import annotations
import time
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .operations_reconciliation import validate_completion


class DeliveryHost(BaseModel):
    """Operator-provided trusted host adapter; never load this from repair output."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    command: tuple[str, ...] = Field(min_length=1)
    executable_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    repository: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    base_branch: str = Field(default="main", exclude_if=lambda value: value == "main")
    base_ref: str = Field(pattern=r"^[a-f0-9]{40}$")
    source_digest: str = Field(pattern=r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$")
    expires_at: float = Field(gt=0)
    authorized: bool
    accepted_source_successors: bool = False
    read_command: tuple[str, ...] | None = None
    reader_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    # None means policy unknown. An explicit empty tuple permits local-only CI.
    required_checks: tuple[str, ...] | None = None
    verification_plan: dict | None = None


    @field_validator("base_branch")
    @classmethod
    def valid_base_branch(cls, value):
        import re

        if (not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_./-]*", value)
                or ".." in value or value.startswith("refs/")
                or any(not part or part.startswith(".") or part.endswith((".", ".lock"))
                       for part in value.split("/"))):
            raise ValueError("invalid_delivery_base_branch")
        return value


def load_delivery_host(path: Path | None) -> DeliveryHost | None:
    if path is None:
        return None
    if path.is_symlink() or path.stat().st_size > 65536:
        raise ValueError("unsafe_delivery_host_config")
    return DeliveryHost.model_validate_json(path.read_text())


def _delivery_capable(host, state):
    import hashlib

    if host is None or not host.authorized or host.expires_at <= time.time():
        return False
    executable = Path(host.command[0])
    if (
        not executable.is_absolute()
        or executable.is_symlink()
        or len(host.command) != 1
    ):
        raise ValueError("delivery_host_requires_pinned_absolute_executable")
    if hashlib.sha256(executable.read_bytes()).hexdigest() != host.executable_sha256:
        raise ValueError("delivery_host_executable_changed")
    if host.source_digest != state.spec.source_digest:
        raise ValueError("delivery_host_source_grant_mismatch")
    from slm_training.autoresearch.runtime.operations_reconciliation import reader_ready

    return reader_ready(host)


def _delivery_receipt(path, request):
    import json
    import re
    from slm_training.harness_core.activity_contract import contract_digest

    if path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise ValueError("unsafe_connector_receipt")
    receipt = json.loads(path.read_text())
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != "connector_delivery_receipt/v1"
        or receipt.get("request_digest") != contract_digest(request)
        or receipt.get("provider") != "github_connector"
        or receipt.get("repository") != request["repository"]
        or receipt.get("source_digest") != request["source_digest"]
        or any(
            receipt.get(key) is not True
            for key in (
                "merged",
                "checks_verified",
                "reviews_resolved",
                "content_verified",
            )
        )
        or not re.fullmatch(r"[a-f0-9]{40}", str(receipt.get("merge_sha", "")))
        or not re.fullmatch(r"[a-f0-9]{40}", str(receipt.get("verified_head_sha", "")))
    ):
        raise ValueError("connector_delivery_receipt_not_verified")
    return receipt


def _publishable(wait):
    kind = wait.get("kind", "document")
    if kind not in {"document", "workspace", "verified_repair_source"}:
        raise ValueError("unsupported_delivery_dependency_kind")
    return kind != "workspace"


def _connector_wait(result, path, request):
    import json
    from slm_training.harness_core.bounded_process import ProcessOutcome
    from slm_training.harness_core.activity_contract import contract_digest

    if result.outcome != ProcessOutcome.COMPLETED or result.returncode not in {10, 20} or path.is_symlink() or not path.is_file():
        return None
    if path.stat().st_size > 1024 * 1024:
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    valid = isinstance(value, dict) and all(value.get(key) == expected for key, expected in {
        "schema_version": "connector_delivery_wait/v1", "provider": "github_connector",
        "request_digest": contract_digest(request), "state": "waiting_capability",
    }.items()) and bool(value.get("reason"))
    return value if valid else None


def _document_capable(runtime, wait, host, state):
    import hashlib
    from .operations_reconciliation import _delivery_subject

    capable = _delivery_capable(host, state)
    if wait.get("kind") == "verified_repair_source":
        from scripts.github_source_verification import source_capable

        return source_capable(runtime, wait, host, capable) and _delivery_capable(host, state)
    if not capable:
        return False
    _, _, _, materialized = _delivery_subject(runtime.store.root.parents[2], wait)
    expected = {name: hashlib.sha256(content.encode()).hexdigest()
                for name, content in materialized["files"].items()}
    if (host.verification_plan or {}).get("delivery_documents_sha256") != expected:
        raise ValueError("delivery_document_successor_gate_required")
    return True



def _require_same_base_branch(runtime, state, host):
    """A changed host cannot retarget an existing attempt or replay its receipt."""
    import json

    if host is None:
        return
    attempts = runtime.store.root / state.spec.output_namespace
    for path in attempts.glob("*/request.json"):
        if json.loads(path.read_text()).get("base_branch", "main") != host.base_branch:
            raise ValueError("delivery_base_branch_changed_requires_new_activity")


def consume_delivery(runtime, activity_id: str, wait: dict, host: DeliveryHost | None):
    """Consume a registered wait through a trusted, bounded connector adapter.

    Adapter protocol: --request JSON --output JSON. It must reconcile the stable
    idempotency key with GitHub BEFORE writing, independently verify content,
    required checks and reviews at the exact head, and squash-merge using that
    head. Its connector receipt is trusted host testimony, never agent evidence.
    No CLI or credentials implicitly provide this configured capability.
    """
    import hashlib
    import json
    from slm_training.harness_core.activity_contract import (
        ActivityOutcome,
        WakeCondition,
        contract_digest,
    )

    state = runtime.snapshot()[activity_id]
    if (
        state.spec.kind != "delivery"
        or state.spec.input_digest != contract_digest(wait)
        or "authorized_github_connector_delivery" not in state.spec.capabilities
    ):
        raise ValueError("delivery_request_binding_mismatch")
    if not _publishable(wait):
        # A dirty-path inventory does not authorize publishing unrelated WIP.
        # No writer may run before an immutable scoped successor is available.
        runtime.claim_next(activity_id=activity_id, capabilities={"local_process"})
        runtime.store.append_event(
            "delivery_precondition_wait", experiment_id=activity_id,
            idempotency_key="workspace-successor:" + state.spec.input_digest,
            detail={"predicate": "immutable_workspace_successor_and_reconciliation_required",
                    "input_digest": state.spec.input_digest, "writer_started": False},
        )
        return {
            "state": runtime.snapshot()[activity_id].status,
            "activity_id": activity_id,
            "reason": "immutable_workspace_successor_and_reconciliation_required",
        }
    _require_same_base_branch(runtime, state, host)
    if state.status == "succeeded":
        terminal = next(
            e
            for e in reversed(runtime.store.verify_event_chain())
            if e["event_type"] == "activity_transition"
            and e["experiment_id"] == activity_id
            and e["detail"]["operation"] in {"finish", "reconcile_finish"}
        )
        attempt = (
            runtime.store.root
            / state.spec.output_namespace
            / terminal["detail"]["lease"]["attempt_id"]
        )
        output = attempt / "receipt.json"
        if (
            output.is_symlink()
            or hashlib.sha256(output.read_bytes()).hexdigest()
            != state.outputs["receipt.json"]
        ):
            raise ValueError("committed_delivery_receipt_changed")
        proof = attempt / "domain-reconciliation.json"
        if (
            proof.is_symlink()
            or hashlib.sha256(proof.read_bytes()).hexdigest()
            != state.outputs[proof.name]
        ):
            raise ValueError("committed_delivery_domain_proof_changed")
        return {
            "state": "succeeded",
            "activity_id": activity_id,
            "receipt": json.loads(output.read_text()),
            "reconciliation": validate_completion(runtime.store, wait, json.loads(proof.read_text()), completed=True),
        }
    from scripts.github_source_preparation import delivery_host, delivery_environment
    host = delivery_host(runtime.store, wait, host)
    capable = _document_capable(runtime, wait, host, state)
    lease = runtime.claim_next(
        activity_id=activity_id,
        capabilities={
            "local_process",
            *({"authorized_github_connector_delivery"} if capable else set()),
        },
    )
    if lease is None:
        return {
            "state": runtime.snapshot()[activity_id].status,
            "activity_id": activity_id,
            "reason": "configured_writer_reader_and_authenticated_local_gate_required"
            if not capable
            else "activity_not_runnable",
        }
    request = {
        "schema_version": "connector_delivery_request/v1",
        "repository": host.repository,
        "base_ref": host.base_ref,
        **({"base_branch": host.base_branch} if host.base_branch != "main" else {}),
        "source_digest": state.spec.source_digest,
        "idempotency_key": "delivery:"
        + contract_digest(
            {
                "input": state.spec.input_digest,
                "source": state.spec.source_digest,
                "repository": host.repository,
                "base_ref": host.base_ref,
                **({"base_branch": host.base_branch} if host.base_branch != "main" else {}),
            }
        ),
        "operation": "deliver_and_reconcile_squash_merge",
        "wait": wait,
        "runtime_root": str(runtime.store.root.resolve()),
        "lease": lease.model_dump(mode="json"),
        "grant_expires_at": host.expires_at,
    }
    attempt = runtime.attempt_dir(lease)
    attempt.mkdir(parents=True, exist_ok=True)
    request_path, output_path = attempt / "request.json", attempt / "receipt.json"
    runtime.store._replace_durable(request_path, json.dumps(request, sort_keys=True))
    started = time.monotonic()
    env = delivery_environment(host)
    result = runtime.run(
        lease,
        [*host.command, "--request", str(request_path), "--output", str(output_path)],
        cwd=attempt,
        env=env,
    )
    waiting = _connector_wait(result, output_path, request)
    outcome = ActivityOutcome.CAPABILITY if waiting else ActivityOutcome.DELIVERY_FAILURE
    receipt, reconciliation, publication_pending, outcome = _reconcile_receipt(
        runtime, lease, wait, host, result, output_path, request, outcome)
    if result.cancelled:
        outcome = ActivityOutcome.CANCELLED
    outputs = (
        {
            str(path.relative_to(attempt)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in attempt.glob("*.json")
        }
        if receipt
        else {}
    )
    runtime.finish(
        lease,
        outcome=outcome,
        outputs=outputs,
        spent_seconds=time.monotonic() - started,
        wake=WakeCondition.model_validate(publication_pending["wake"]) if publication_pending else (WakeCondition(
            predicate="connector reconciles remote delivery",
            source="authorized_delivery_receipt",
            identity_digest=state.spec.input_digest,
        )
        if outcome in {ActivityOutcome.DELIVERY_FAILURE, ActivityOutcome.CAPABILITY}
        else None),
    )
    return {
        "state": runtime.snapshot()[activity_id].status,
        "activity_id": activity_id,
        "receipt": receipt,
        "reconciliation": reconciliation,
        **({"reason": waiting["reason"]} if waiting else {}),
        **({"pending": publication_pending} if publication_pending else {}),
    }


def _reconcile_receipt(runtime, lease, wait, host, result, output_path, request, outcome):
    from slm_training.harness_core.activity_contract import ActivityOutcome
    from slm_training.harness_core.bounded_process import ProcessOutcome
    receipt, reconciliation = None, None
    publication_pending = None
    from scripts.autotrain_source_publication import SourcePublicationPrerequisite
    if (
        result.outcome == ProcessOutcome.COMPLETED
        and result.returncode == 0
        and time.time() < host.expires_at
    ):
        try:
            receipt = _delivery_receipt(output_path, request)
            from slm_training.autoresearch.runtime.operations_reconciliation import (
                reconcile_with_host,
            )

            reconciliation = reconcile_with_host(runtime, lease, wait, receipt, host)
            outcome = ActivityOutcome.SUCCEEDED
        except SourcePublicationPrerequisite as pending:
            receipt = None
            publication_pending = pending.pending
            outcome = ActivityOutcome.DEPENDENCY
        except (OSError, ValueError, TypeError, KeyError, TimeoutError):
            receipt = None
    return receipt, reconciliation, publication_pending, outcome
