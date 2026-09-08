"""Map the Codex execution protocol to the canonical isolation backend."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable

from slm_training.autoresearch.heal.agent_executor import AgentRun
from slm_training.autoresearch.heal.isolation import (
    IsolationSpec,
    probe_isolation,
    run_isolated,
)
from slm_training.autoresearch.heal.isolation_workspace import (
    manifest_digest,
    private_snapshot,
    tree_manifest,
)
from slm_training.autoresearch.heal.repair_contracts import RepairRequest


class BubblewrapAgentRunner:
    """Controller-constructed runner; private attempt root is never mounted.

    The initial backend has no provider-network or credential mount capability.
    Such requests wait explicitly; it must not expose host credentials to make
    a remote model work. Runtime installations are explicit readonly mounts.
    """

    def __init__(
        self, *, source: Path, attempt_root: Path, runtime_roots: tuple[Path, ...] = ()
    ) -> None:
        self.source = source
        self.attempt_root = attempt_root
        self.runtime_roots = runtime_roots

    def capability(self, request: RepairRequest) -> str | None:
        if request.grant is None:
            return "agent_grant_missing"
        if request.grant.network != "none":
            return "provider_egress_not_supported_by_isolation_backend"
        capability = probe_isolation()
        if not capability.available:
            return f"isolation_unavailable:{capability.reason}"
        executable = Path(request.grant.executable).resolve()
        if not any(
            executable.is_relative_to(root.resolve()) for root in self.runtime_roots
        ):
            return "agent_executable_outside_approved_runtime"
        return None

    def _argv(self, argv: tuple[str, ...]) -> tuple[str, ...]:
        executable = Path(argv[0]).resolve()
        for index, root in enumerate(self.runtime_roots):
            if executable.is_relative_to(root.resolve()):
                mapped = f"/runtime/{index}/{executable.relative_to(root.resolve()).as_posix()}"
                break
        else:
            raise ValueError("agent executable not in approved runtime")
        return (
            mapped,
            *(
                arg.replace("/input/", "/workspace/repair-input/").replace(
                    "/output/", "/workspace/repair-output/"
                )
                for arg in argv[1:]
            ),
        )

    def run(
        self,
        request: RepairRequest,
        argv: tuple[str, ...],
        *,
        inputs: dict,
        progress: Callable[[], None],
        cancelled: Callable[[], bool],
    ) -> AgentRun:
        reason = self.capability(request)
        if reason:
            raise ValueError(reason)
        # Digest names cannot traverse; an existing attempt is reconciliation,
        # never an overwrite of previous patch or execution evidence.
        attempt = self.attempt_root / request.digest()
        attempt.mkdir(parents=True, exist_ok=False)
        workspace = private_snapshot(self.source, attempt / "candidate")
        baseline = tree_manifest(workspace)
        if manifest_digest(baseline) != request.blocker.source_digest:
            raise ValueError("repair source differs from pinned release")
        _create_regression_mounts(workspace, request.allowed_paths)
        input_dir, output_dir = workspace / "repair-input", workspace / "repair-output"
        input_dir.mkdir()
        output_dir.mkdir()
        # Only the fixed result file is writable; directory mounts stay forbidden.
        (output_dir / "proposal.json").touch(exist_ok=False)
        (input_dir / "baseline-manifest.json").write_text(
            json.dumps(baseline, sort_keys=True), encoding="utf-8"
        )
        for name, payload in inputs.items():
            if name not in {"repair-instructions.json", "proposal-schema.json"}:
                raise ValueError("unexpected controller input name")
            (input_dir / name).write_text(
                json.dumps(payload, sort_keys=True), encoding="utf-8"
            )
        assert request.grant is not None

        finished, cancel_event = threading.Event(), threading.Event()

        watcher = threading.Thread(
            target=_watch_cancellation,
            args=(cancelled, finished, cancel_event),
            daemon=True,
        )
        watcher.start()

        def heartbeat(_pid: int) -> None:
            progress()

        spec = IsolationSpec(
            workspace,
            writable_paths=(*request.allowed_paths, "repair-output/proposal.json"),
            runtime_roots=self.runtime_roots,
            timeout_seconds=request.grant.interrupt_seconds,
        )
        try:
            result = run_isolated(
                spec,
                self._argv(argv),
                on_start=heartbeat,
                on_heartbeat=heartbeat,
                cancel_event=cancel_event,
            )
        finally:
            finished.set()
            watcher.join(timeout=0.1)
            # Preserve protocol artifacts separately from the source proposal.
            # Only these two controller-created paths are moved, never removed.
            input_dir.rename(attempt / "input")
            output_dir.rename(attempt / "output")
        output = attempt / "output" / "proposal.json"
        final = ""
        if (
            output.is_file()
            and not output.is_symlink()
            and output.stat().st_size <= 65536
        ):
            final = output.read_text(encoding="utf-8")
        return AgentRun(
            result.outcome.value, result.returncode, result.duration_seconds, final
        )


def _watch_cancellation(cancelled, finished, cancel_event) -> None:
    while not finished.is_set():
        try:
            if cancelled():
                cancel_event.set()
                return
        except Exception:
            cancel_event.set()
            return
        finished.wait(0.05)


def _create_regression_mounts(workspace: Path, paths: tuple[str, ...]) -> None:
    for relative in paths:
        target = workspace / relative
        if relative.startswith("tests/") and target.exists():
            raise ValueError("existing regression tests are protected")
        if not target.exists():
            if not relative.startswith("tests/") or not relative.endswith(".py"):
                raise ValueError("only declared new regression files may be created")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch(exist_ok=False)
