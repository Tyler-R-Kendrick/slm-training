"""Controller-owned Linux Bubblewrap boundary for untrusted repair workloads.

No network, credentials, host home, shared Git metadata or control store is
mounted. Missing isolation is an explicit capability failure, never a fallback.
"""

from __future__ import annotations

import math
import shutil
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from slm_training.harness_core.bounded_process import (
    BoundedProcessResult,
    ProcessOutcome,
    run_bounded_process,
)
from slm_training.levers import INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS

from .isolation_workspace import (
    IsolationViolation,
    checked_path,
    scope_changes,
    tree_manifest,
)
from .repair_scope import repair_classification

# Executed with isolated system Python, before candidate imports. Limits are
# hard limits inherited by descendants. Stdin never inherits a controller pipe.
_WORKLOAD_BOOTSTRAP = (
    "import os,resource,sys; "
    "limit=int(sys.argv[1]); resource.setrlimit(resource.RLIMIT_FSIZE,(limit,limit)); "
    "resource.setrlimit(resource.RLIMIT_CORE,(0,0)); "
    "resource.setrlimit(resource.RLIMIT_NOFILE,(128,128)); "
    "os.dup2(os.open('/dev/null',os.O_RDONLY),0); "
    "os.execvpe(sys.argv[2],sys.argv[2:],os.environ)"
)


class IsolationUnavailable(RuntimeError):
    """A required OS capability is absent; only this activity should wait."""


@dataclass(frozen=True)
class IsolationCapability:
    available: bool
    backend: str
    reason: str
    executable: str | None


@dataclass(frozen=True)
class IsolationSpec:
    """Trusted in-memory controller grant; never deserialize from agent output.

    Runtime roots map read-only to /runtime/0, /runtime/1, etc. The caller must
    supply dedicated runtime installations, not arbitrary user directories.
    Workspaces are private copies produced by private_snapshot, never live repos.
    """

    workspace: Path
    writable_paths: tuple[str, ...] = ()
    runtime_roots: tuple[Path, ...] = ()
    timeout_seconds: float = INTERRUPT_AFTER_SECONDS
    scratch_bytes: int = 64 * 1024 * 1024
    pythonpath: str = "/workspace/src"
    environment: tuple[tuple[str, str], ...] = ()


def _base_command(binary: str) -> list[str]:
    command = [
        binary,
        "--unshare-all",
        "--unshare-user",
        "--disable-userns",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--clearenv",
        "--ro-bind",
        "/usr",
        "/usr",
    ]
    for name in ("bin", "sbin", "lib", "lib64"):
        path = Path("/") / name
        if path.is_symlink():
            command.extend(("--symlink", str(path.readlink()), str(path)))
        elif path.is_dir():
            command.extend(("--ro-bind", str(path), str(path)))
    return command + ["--proc", "/proc", "--dev", "/dev"]


def probe_isolation() -> IsolationCapability:
    """Probe the real namespaces/flags; availability alone conveys no grant."""
    binary = shutil.which("bwrap")
    if binary is None:
        return IsolationCapability(False, "bubblewrap", "bwrap_not_installed", None)
    result = run_bounded_process(
        _base_command(binary) + ["--", "/bin/true"],
        interrupt_after_seconds=min(5, INTERRUPT_AFTER_SECONDS),
        kill_grace_seconds=KILL_GRACE_SECONDS,
        env={"PATH": "/usr/bin:/bin"},
    )
    available = result.outcome == ProcessOutcome.COMPLETED and result.returncode == 0
    return IsolationCapability(
        available,
        "bubblewrap",
        "available"
        if available
        else (result.launch_error or result.stderr or result.outcome.value),
        binary,
    )


def _runtime_mounts(spec: IsolationSpec) -> list[str]:
    args: list[str] = []
    forbidden = {"/", "/home", "/tmp", "/run", "/var", "/etc", "/root"}
    for index, root in enumerate(spec.runtime_roots):
        root = root.resolve(strict=True)
        if (
            str(root) in forbidden
            or root == Path.home()
            or len(root.parts) < 3
            or spec.workspace.resolve().is_relative_to(root)
            or root.is_relative_to(spec.workspace.resolve())
        ):
            raise IsolationViolation("runtime mount is broad or overlaps workspace")
        # Reject mounts containing host sockets or special files. Runtime
        # symlinks may reference the read-only system runtime, never host home.
        if not root.is_dir():
            raise IsolationViolation("runtime root must be a dedicated directory")
        _validate_runtime(root)
        args.extend(("--ro-bind", str(root), f"/runtime/{index}"))
    return args


def _validate_runtime(root: Path) -> None:
    import os
    import stat

    if root.name in {".git", ".codex", ".ssh", "outputs", "loops"}:
        raise IsolationViolation("credentials/control metadata cannot be a runtime")
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                target = path.resolve()
                if not (target.is_relative_to(root) or target.is_relative_to("/usr")):
                    raise IsolationViolation(
                        f"runtime link escapes approved runtime: {path}"
                    )
            elif not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise IsolationViolation(f"special runtime file: {path}")


def build_isolated_command(
    spec: IsolationSpec, argv: Sequence[str], binary: str
) -> list[str]:
    """Build only fixed mounts; argv is never shell-interpolated."""
    if not argv or not all(isinstance(arg, str) and "\0" not in arg for arg in argv):
        raise IsolationViolation("invalid workload argv")
    if (
        not math.isfinite(spec.timeout_seconds)
        or not 0 < spec.timeout_seconds <= INTERRUPT_AFTER_SECONDS
    ):
        raise IsolationViolation("timeout exceeds canonical interrupt budget")
    if not 0 < spec.scratch_bytes <= 256 * 1024 * 1024:
        raise IsolationViolation("scratch allocation outside bounded grant")
    workspace = spec.workspace.resolve(strict=True)
    if (workspace / ".git").exists():
        raise IsolationViolation(
            "workspace contains Git metadata; use private_snapshot"
        )
    tree_manifest(workspace)
    command = _base_command(binary) + _runtime_mounts(spec)
    command += ["--ro-bind", str(workspace), "/workspace"]
    for relative in spec.writable_paths:
        path = checked_path(workspace, relative)
        if relative.split("/", 1)[0].startswith("."):
            raise IsolationViolation("metadata cannot be a writable source mount")
        if path.is_dir():
            raise IsolationViolation("source repair needs exact file mounts")
        if repair_classification((relative,)) == "trust_policy_change":
            raise IsolationViolation("protected surface cannot be a writable mount")
        command.extend(("--bind", str(path), f"/workspace/{relative}"))
    for path in ("/tmp", "/scratch"):
        command.extend(("--size", str(spec.scratch_bytes), "--tmpfs", path))
    return command + [
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        "--setenv",
        "HOME",
        "/scratch",
        "--setenv",
        "PYTHONPATH",
        spec.pythonpath,
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "TMPDIR",
        "/tmp",
        *sum((["--setenv", key, value] for key, value in spec.environment), []),
        "--chdir",
        "/workspace",
        "--",
        "/usr/bin/python3",
        "-I",
        "-S",
        "-c",
        _WORKLOAD_BOOTSTRAP,
        str(spec.scratch_bytes),
        *argv,
    ]


def run_isolated(
    spec: IsolationSpec,
    argv: Sequence[str],
    *,
    on_start: Callable[[int], None] | None = None,
    on_heartbeat: Callable[[int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> BoundedProcessResult:
    """All worker/verifier/failure paths share this process and scope boundary."""
    started = time.monotonic()
    capability = probe_isolation()
    if not capability.available:
        raise IsolationUnavailable(capability.reason)
    command = build_isolated_command(spec, argv, capability.executable or "")
    before = tree_manifest(spec.workspace)
    remaining = spec.timeout_seconds - (time.monotonic() - started)
    if remaining <= 0:
        raise IsolationUnavailable("isolation_preparation_budget_exhausted")
    try:
        result = run_bounded_process(
            command,
            interrupt_after_seconds=remaining,
            kill_grace_seconds=KILL_GRACE_SECONDS,
            env={"PATH": "/usr/bin:/bin"},
            on_start=on_start,
            on_heartbeat=on_heartbeat,
            cancel_event=cancel_event,
        )
        return replace(result, duration_seconds=time.monotonic() - started)
    finally:
        changed = scope_changes(
            before, tree_manifest(spec.workspace), spec.writable_paths
        )
        if changed:
            raise IsolationViolation(f"snapshot scope changed: {changed}")
