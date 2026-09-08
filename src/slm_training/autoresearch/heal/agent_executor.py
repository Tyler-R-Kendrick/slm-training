"""Configured Codex noninteractive execution inside a controller-owned sandbox.

The sandbox runner owns mounts, credential grant, input/output files and process
cancellation. There is deliberately no unsandboxed subprocess fallback.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from slm_training.autoresearch.heal.repair_contracts import (
    RepairDispatchResult,
    RepairProposal,
    RepairRequest,
)
from slm_training.autoresearch.heal.repair_prompt import repair_prompt
from slm_training.harness_core.bounded_process import (
    ProcessOutcome,
    run_bounded_process,
)
from slm_training.levers import KILL_GRACE_SECONDS

REQUIRED_FLAGS = (
    "--json",
    "--sandbox",
    "--output-schema",
    "--output-last-message",
    "--ignore-user-config",
    "--ignore-rules",
    "--ephemeral",
    "--skip-git-repo-check",
)


@dataclass(frozen=True)
class AgentRun:
    outcome: str
    returncode: int | None
    seconds: float
    final_json: str


class AgentCancelled(Exception):
    """Controller cancellation; isolation owns bounded child teardown."""


class IsolatedAgentRunner(Protocol):
    """Trusted host adapter, never constructed from a worker payload.

    run mounts input below /input and private candidate source at /workspace,
    outputs below /output; captures final JSON only after child teardown. It
    enforces grant egress/deadline, never exposes controller state, and calls
    progress for operational heartbeat (not scientific progress).
    """

    def capability(self, request: RepairRequest) -> str | None: ...

    def run(
        self,
        request: RepairRequest,
        argv: tuple[str, ...],
        *,
        inputs: dict,
        progress: Callable[[], None],
        cancelled: Callable[[], bool],
    ) -> AgentRun: ...


def probe_codex(executable: str) -> dict:
    """Read-only version/help probe. Never authenticates or submits model work."""
    responses = []
    for args in (("--version",), ("exec", "--help")):
        result = run_bounded_process(
            (executable, *args),
            interrupt_after_seconds=10,
            kill_grace_seconds=KILL_GRACE_SECONDS,
            max_output_bytes=16000,
        )
        if result.outcome != ProcessOutcome.COMPLETED or result.returncode != 0:
            return {"available": False, "reason": "cli_probe_failed"}
        responses.append(result.stdout)
    missing = [flag for flag in REQUIRED_FLAGS if flag not in responses[1]]
    return {
        "available": not missing,
        "version": responses[0].strip(),
        "missing_flags": missing,
        "reason": "unsupported_cli" if missing else "ok",
    }


class CodexExecutor:
    def __init__(
        self, runner: IsolatedAgentRunner | None, *, instructions: str, contract: str
    ) -> None:
        self.runner = runner
        self.instructions = instructions
        self.contract = contract

    def capability(self, request: RepairRequest) -> str | None:
        grant = request.grant
        if grant is None or grant.expires_at <= time.time():
            return "agent_grant_missing" if grant is None else "agent_grant_expired"
        if request.blocker.blocker_class not in grant.repair_classes:
            return "repair_class_not_granted"
        if self.runner is None:
            return "isolation_backend_unavailable"
        failure = self.runner.capability(request)
        if failure:
            return failure
        return _cli_failure(grant.executable, grant.executable_sha256)

    def execute(
        self,
        request: RepairRequest,
        *,
        progress: Callable[[], None],
        cancelled: Callable[[], bool],
    ) -> RepairDispatchResult:
        reason = self.capability(request)
        if reason:
            return RepairDispatchResult(
                status="waiting_capability",
                request_digest=request.digest(),
                reason=reason,
            )
        if cancelled():
            return RepairDispatchResult(
                status="cancelled",
                request_digest=request.digest(),
                reason="controller_cancelled",
            )
        assert request.grant is not None and self.runner is not None
        argv = (
            request.grant.executable,
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--config",
            "model_provider=" + json.dumps(request.grant.provider),
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
            "--json",
            "--cd",
            "/workspace",
            "--output-schema",
            "/input/proposal-schema.json",
            "--output-last-message",
            "/output/proposal.json",
            "Read /input/repair-instructions.json and execute only that scoped repair task.",
        )
        result = self.runner.run(
            request,
            argv,
            inputs={
                "repair-instructions.json": repair_prompt(
                    request, instructions=self.instructions, contract=self.contract
                ),
                "proposal-schema.json": RepairProposal.model_json_schema(),
            },
            progress=progress,
            cancelled=cancelled,
        )
        fields = {"request_digest": request.digest(), "spent_seconds": result.seconds}
        if result.outcome != "completed" or result.returncode != 0:
            stopped = result.outcome == "cancelled"
            return RepairDispatchResult(
                status="cancelled" if stopped else "waiting_diagnosis",
                reason="controller_cancelled"
                if stopped
                else "agent_execution_incomplete",
                **fields,
            )
        try:
            proposal = RepairProposal.model_validate_json(result.final_json)
        except ValueError:
            return RepairDispatchResult(
                status="rejected", reason="agent_output_invalid", **fields
            )
        if proposal.request_digest != request.digest():
            return RepairDispatchResult(
                status="rejected", reason="proposal_request_mismatch", **fields
            )
        return RepairDispatchResult(
            status="waiting_verification",
            proposal=proposal,
            reason="independent_verification_required",
            **fields,
        )


def _cli_failure(executable: str, expected_digest: str) -> str | None:
    try:
        digest = hashlib.sha256(Path(executable).read_bytes()).hexdigest()
    except OSError:
        return "agent_executable_unavailable"
    if digest != expected_digest:
        return "agent_executable_changed"
    probe = probe_codex(executable)
    return None if probe["available"] else str(probe["reason"])
