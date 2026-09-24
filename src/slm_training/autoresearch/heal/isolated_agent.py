"""Map the Codex execution protocol to the canonical isolation backend."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import threading
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Callable

from slm_training.autoresearch.heal.agent_executor import AgentRun
from slm_training.autoresearch.heal.isolation import (
    IsolationSpec,
    probe_isolation,
    run_isolated,
)
from slm_training.autoresearch.heal.isolation_workspace import (
    checked_path,
    manifest_digest,
    private_snapshot,
    tree_manifest,
)
from slm_training.autoresearch.heal.repair_contracts import RepairRequest
from slm_training.harness_core.codex_subscription import SubscriptionCancelled
from slm_training.harness_core.provider_bridge import ProviderBridge


class BubblewrapAgentRunner:
    """Controller-constructed runner; private attempt root is never mounted.

    Provider access uses a granted fixed host proxy over one Unix socket.
    Credentials stay with that host proxy. Runtime mounts remain readonly.
    """

    def __init__(
        self, *, source: Path, attempt_root: Path, runtime_roots: tuple[Path, ...] = (),
        codex_subscription_home: str | None = None,
    ) -> None:
        self.source = source
        self.attempt_root = attempt_root
        self.runtime_roots = runtime_roots
        self.codex_subscription_home = codex_subscription_home

    def capability(self, request: RepairRequest) -> str | None:
        if request.grant is None:
            return "agent_grant_missing"
        if request.grant.network != "none" and request.grant.provider_endpoint is None:
            return "provider_egress_not_supported_by_isolation_backend"
        endpoint = request.grant.provider_endpoint
        if endpoint and endpoint.authentication == "codex_subscription" and not self.codex_subscription_home:
            return "host_codex_subscription_not_configured"
        capability = probe_isolation()
        if not capability.available:
            return f"isolation_unavailable:{capability.reason}"
        executable = Path(request.grant.executable).resolve()
        if not any(
            executable.is_relative_to(root.resolve()) for root in self.runtime_roots
        ):
            return "agent_executable_outside_approved_runtime"
        return self._credential_mount_failure()

    def _credential_mount_failure(self):
        if self.codex_subscription_home is None:
            return None
        home = Path(self.codex_subscription_home)
        if not home.is_absolute():
            return "host_codex_subscription_home_must_be_absolute"
        protected = (home.resolve(), (home / "auth.json").resolve())
        exposed = (self.source, *self.runtime_roots,
                   *(Path(path) for path in ("/usr", "/bin", "/sbin", "/lib", "/lib64")))
        if any(secret.is_relative_to(root.resolve()) for secret in protected for root in exposed):
            return "host_codex_subscription_credentials_exposed_to_worker"
        return None

    def _argv(self, argv: tuple[str, ...]) -> tuple[str, ...]:
        executable = Path(argv[0]).resolve()
        for index, root in enumerate(self.runtime_roots):
            if executable.is_relative_to(root.resolve()):
                mapped = f"/runtime/{index}/{executable.relative_to(root.resolve()).as_posix()}"
                break
        else:
            raise ValueError("agent executable not in approved runtime")
        mapped_argv = (
            mapped,
            *(
                arg.replace("/input/", "/workspace/repair-input/").replace(
                    "/output/", "/workspace/repair-output/"
                )
                for arg in argv[1:]
            ),
        )
        if len(argv) > 1 and argv[1] == "exec":
            return ("/bin/sh", "-c", 'exec "$@" < /workspace/repair-input/repair-instructions.json',
                    "slm-repair-stdin", *mapped_argv)
        return mapped_argv

    def run(
        self,
        request: RepairRequest,
        argv: tuple[str, ...],
        *,
        inputs: dict,
        progress: Callable[[], None],
        cancelled: Callable[[], bool],
    ) -> AgentRun:
        started = time.monotonic()
        reason = self.capability(request)
        if reason:
            raise ValueError(reason)
        assert request.grant is not None
        deadline = min(started + request.grant.interrupt_seconds,
                       time.monotonic() + request.grant.expires_at - time.time())
        # Digest names cannot traverse; an existing attempt is reconciliation,
        # never an overwrite of previous patch or execution evidence.
        attempt = self.attempt_root / request.digest()
        attempt.mkdir(parents=True, exist_ok=False)
        workspace = private_snapshot(self.source, attempt / "candidate")
        baseline = tree_manifest(workspace)
        if manifest_digest(baseline) != request.blocker.source_digest:
            raise ValueError("repair source differs from pinned release")
        _create_regression_mounts(workspace, request.allowed_paths)
        input_dir, output_dir = _stage_inputs(workspace, baseline, inputs)
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
            codex_mount_target=True,
        )
        result, outcome, returncode = None, "unknown_failure", None
        try:
            endpoint = request.grant.provider_endpoint
            seconds = _remaining(deadline)
            subscription = (self.codex_subscription_home, request.grant.executable)
            bridge = ProviderBridge(endpoint, seconds, cancel_event, subscription=subscription) if endpoint else nullcontext(None)
            with bridge as provider_socket:
                result = run_isolated(
                    replace(spec, provider_socket=provider_socket, timeout_seconds=_remaining(deadline)),
                    self._argv(argv),
                    on_start=heartbeat,
                    on_heartbeat=heartbeat,
                    cancel_event=cancel_event,
                )
                outcome, returncode = result.outcome.value, result.returncode
        except SubscriptionCancelled:
            outcome, returncode = "cancelled", None
        except TimeoutError:
            outcome, returncode = "timed_out", None
        except KeyboardInterrupt:
            outcome, returncode = "interrupted", None
            raise
        finally:
            finished.set()
            watcher.join(timeout=0.1)
            # Preserve protocol artifacts separately from the source proposal.
            # Only these two controller-created paths are moved, never removed.
            input_dir.rename(attempt / "input")
            output_dir.rename(attempt / "output")
            _record_execution(attempt, outcome, returncode, result)
        return AgentRun(
            outcome, returncode, time.monotonic() - started, _proposal_text(attempt)
        )


def _stage_inputs(workspace, baseline, inputs):
    prepared = tree_manifest(workspace)
    input_dir, output_dir = workspace / "repair-input", workspace / "repair-output"
    input_dir.mkdir()
    output_dir.mkdir()
    # Only the fixed result file is writable; directory mounts stay forbidden.
    (output_dir / "proposal.json").touch(exist_ok=False)
    (input_dir / "baseline-manifest.json").write_text(json.dumps(baseline, sort_keys=True))
    (input_dir / "prepared-manifest.json").write_text(json.dumps(prepared, sort_keys=True))
    shutil.copyfile(
        Path(__file__).with_name("isolation_workspace.py"),
        input_dir / "proposal-digest.py",
    )
    (input_dir / "proposal-digest.py").chmod(0o444)
    for name, payload in inputs.items():
        if name not in {"repair-instructions.json", "proposal-schema.json"}:
            raise ValueError("unexpected controller input name")
        if name == "repair-instructions.json":
            payload = {**payload, "digest_contract": payload["digest_contract"] + (
                " After all edits, run `/usr/bin/python3 -I -S "
                "repair-input/proposal-digest.py` "
                "from /workspace and use its tree_digest and patch_digest verbatim. The "
                "read-only controller command projects only exact request.allowed_paths onto "
                "the trusted post-preparation manifest. Never edit or reimplement it. The "
                "independent host still hashes the full physical candidate and rejects "
                "protected changes."
            )}
        (input_dir / name).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return input_dir, output_dir


def _proposal_text(attempt):
    output = attempt / "output" / "proposal.json"
    if output.is_file() and not output.is_symlink() and output.stat().st_size <= 65536:
        return output.read_text(encoding="utf-8")
    return ""


def _record_execution(attempt, outcome, returncode, result):
    """Bounded diagnostic tails, written after teardown outside worker mounts."""
    directory = attempt / "execution"
    directory.mkdir(mode=0o700)
    receipt = {"schema_version": "repair_agent_execution/v1",
               "outcome": outcome, "returncode": returncode,
               "process_result_available": result is not None, "acceptance_authority": False}
    for stream in ("stdout", "stderr"):
        data = getattr(result, stream).encode() if result is not None else b""
        (directory / (stream + ".txt")).write_bytes(data)
        receipt[stream + "_sha256"] = hashlib.sha256(data).hexdigest()
        receipt[stream + "_truncated"] = getattr(result, stream + "_truncated") if result is not None else False
    (directory / "receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")


def _remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("repair_grant_expired_during_setup")
    return seconds


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
    targets = [checked_path(workspace, relative, must_exist=False) for relative in paths]
    # Validate the entire grant before changing any mode or creating a file.
    for relative, target in zip(paths, targets):
        if relative.startswith("tests/") and target.exists():
            raise ValueError("existing regression tests are protected")
        if target.exists():
            info = target.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("only private regular files may be writable")
        else:
            if not relative.startswith("tests/") or not relative.endswith(".py"):
                raise ValueError("only declared new regression files may be created")
    for target in targets:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch(exist_ok=False)
        # The frozen source digest was checked before this private preparation.
        # Keep these scoped mode changes in the candidate/proposal identities.
        target.chmod(stat.S_IMODE(target.stat().st_mode) | stat.S_IWUSR)
