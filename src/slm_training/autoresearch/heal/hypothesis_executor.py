"""Noninteractive hypothesis worker on the existing isolated execution owner.

Output remains a proposal; AgentHypothesisProvider owns canonical validation.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Literal

from pydantic import Field

from slm_training.harness_core.activity_contract import Contract, ResourceGrant, contract_digest
from ..search.evidence import Digest
from ..search.proposals import proposal_inputs
from .agent_executor import probe_codex
from .isolation import IsolationSpec, probe_isolation, run_isolated
from .isolation_workspace import manifest_digest, private_snapshot, tree_manifest


class ProposalCapabilityUnavailable(RuntimeError):
    """The caller parks this proposal job, not independent training families."""


class ResearchGrant(Contract):
    grant_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    executable: str = Field(min_length=1)
    executable_sha256: Digest
    expires_at: float = Field(gt=0)
    resources: ResourceGrant
    network: Literal["none", "approved_provider_only"] = "none"
    purpose: Literal["hypothesis_proposal"] = "hypothesis_proposal"


class CodexHypothesisExecutor:
    """One attempt, controlled by an already leased outer ActivityRuntime job.

    Reuses the repair isolation and CLI probe, not its permissions. No host
    credentials or provider network are mounted by this reference backend.
    """

    def __init__(
        self,
        *,
        source: Path,
        attempt_root: Path,
        source_digest: str,
        grant: ResearchGrant | None,
        runtime_roots: tuple[Path, ...],
        context: dict,
        store=None,
    ):
        self.source, self.attempt_root, self.source_digest = (
            source,
            attempt_root,
            source_digest,
        )
        self.grant, self.runtime_roots = grant, runtime_roots
        self.instructions, self.owner_contract = (
            context["instructions"],
            context["owner_contract"],
        )
        self.supported_levers, self.store = context["supported_levers"], store
        self.output_schema = context.get("output_schema")

    def capability(self) -> str | None:
        grant = self.grant
        if grant is None:
            return "research_grant_missing"
        if not isinstance(self.output_schema, dict) or not self.output_schema:
            return "proposal_output_schema_missing"
        if grant.expires_at <= time.time():
            return "research_grant_expired"
        if grant.network != "none":
            return "provider_egress_not_supported_by_isolation_backend"
        capability = probe_isolation()
        if not capability.available:
            return f"isolation_unavailable:{capability.reason}"
        return self._executable_capability(grant)

    @staticmethod
    def _executable_capability(grant):
        try:
            if (
                hashlib.sha256(Path(grant.executable).read_bytes()).hexdigest()
                != grant.executable_sha256
            ):
                return "agent_executable_changed"
        except OSError:
            return "agent_executable_unavailable"
        probe = probe_codex(grant.executable)
        return None if probe["available"] else probe["reason"]

    def execute(self, campaign, evidence, sources, feedback=()) -> dict:
        inputs = proposal_inputs(
            campaign,
            evidence,
            sources,
            feedback,
            instructions=self.instructions,
            owner_contract=self.owner_contract,
            source_digest=self.source_digest,
            supported_levers=self.supported_levers,
        )
        reason = self.capability()
        if reason:
            if self.store is not None:
                self.store.append_event(
                    "proposal_waiting_capability",
                    idempotency_key=f"proposal-wait:{contract_digest(inputs)}:{reason}",
                    detail={
                        "request_digest": contract_digest(inputs),
                        "reason": reason,
                        "wake_source": "research_capability_configuration",
                    },
                )
            raise ProposalCapabilityUnavailable(reason)
        result = self._run(inputs)
        matrix = json.loads(result)
        if (
            not isinstance(matrix, dict)
            or matrix.get("campaign_id") != campaign.campaign_id
            or matrix.get("evidence_snapshot_id") != evidence.snapshot_id
        ):
            raise ValueError("proposal campaign/evidence identity mismatch")
        return matrix

    def _run(self, inputs):
        assert self.grant is not None
        self.attempt_root.mkdir(parents=True, exist_ok=True)
        # Attempt allocation belongs to Runtime. Reusing a path never re-executes.
        workspace = private_snapshot(self.source, self.attempt_root / "candidate")
        if manifest_digest(tree_manifest(workspace)) != self.source_digest:
            raise ValueError("proposal release differs from pinned source")
        input_dir, output_dir = (
            workspace / "proposal-input",
            workspace / "proposal-output",
        )
        input_dir.mkdir()
        output_dir.mkdir()
        (input_dir / "instructions.json").write_text(json.dumps(inputs, sort_keys=True))
        (input_dir / "schema.json").write_text(json.dumps(self.output_schema))
        executable = self._mapped_executable()
        argv = (
            executable,
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--config",
            "model_provider=" + json.dumps(self.grant.provider),
            "--ephemeral",
            "--sandbox",
            "workspace-write",
            "--json",
            "--cd",
            "/workspace",
            "--output-schema",
            "/workspace/proposal-input/schema.json",
            "--output-last-message",
            "/workspace/proposal-output/matrix.json",
            "Read /workspace/proposal-input/instructions.json. Produce only the requested hypothesis matrix.",
        )
        result = run_isolated(
            IsolationSpec(
                workspace,
                writable_paths=("proposal-output",),
                runtime_roots=self.runtime_roots,
                timeout_seconds=self.grant.resources.interrupt_seconds,
            ),
            argv,
        )
        if self.store is not None:
            self.store.append_event(
                "proposal_attempt_observed",
                detail={
                    "request_digest": contract_digest(inputs),
                    "outcome": result.outcome.value,
                    "returncode": result.returncode,
                    "spent_seconds": result.duration_seconds,
                },
            )
        if result.outcome.value != "completed" or result.returncode != 0:
            raise RuntimeError("proposal_execution_incomplete")
        output = output_dir / "matrix.json"
        if (
            output.is_symlink()
            or not output.is_file()
            or output.stat().st_size > 1_000_000
        ):
            raise ValueError("proposal_output_missing_or_invalid")
        return output.read_text(encoding="utf-8")

    def _mapped_executable(self):
        executable = Path(self.grant.executable).resolve()
        for index, root in enumerate(self.runtime_roots):
            if executable.is_relative_to(root.resolve()):
                return f"/runtime/{index}/{executable.relative_to(root.resolve()).as_posix()}"
        raise ProposalCapabilityUnavailable("executable_outside_approved_runtime")

