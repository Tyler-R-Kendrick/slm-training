"""Controller-owned Linux Bubblewrap boundary for untrusted repair workloads.

No network, credentials, host home, shared Git metadata or control store is
mounted. Missing isolation is an explicit capability failure, never a fallback.
"""

from __future__ import annotations

import json
import math
import shutil
import stat
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from slm_training.harness_core.bounded_process import (
    BoundedProcessResult,
    ProcessOutcome,
    run_bounded_process,
)
from slm_training.levers import (
    HARNESS_FINALIZATION_RESERVE_SECONDS, INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS,
)

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
    writable_dirs: tuple[str, ...] = ()
    runtime_roots: tuple[Path, ...] = ()
    timeout_seconds: float = INTERRUPT_AFTER_SECONDS
    scratch_bytes: int = 64 * 1024 * 1024
    pythonpath: str = "/workspace/src"
    environment: tuple[tuple[str, str], ...] = ()
    provider_socket: Path | None = None
    codex_mount_target: bool = False

def _base_command(binary: str) -> list[str]:
    command = [
        binary,
        "--unshare-all",
        "--unshare-user",
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
    return command + ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]

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
    roots = tuple(root.resolve(strict=True) for root in spec.runtime_roots)
    if any(root.is_relative_to(Path("/nix/store")) for root in roots):
        args.extend(("--dir", "/nix", "--dir", "/nix/store"))
    for index, root in enumerate(roots):
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
        _validate_runtime(root, roots)
        if root.is_relative_to(Path("/runtime")):
            args.extend(("--ro-bind", str(root), str(root)))
        args.extend(("--ro-bind", str(root), f"/runtime/{index}"))
        if root.is_relative_to(Path("/nix/store")):
            args.extend(("--ro-bind", str(root), str(root)))
    return args

def _validate_runtime(root: Path, approved: tuple[Path, ...]) -> None:
    import os
    import stat

    if root.name in {".git", ".codex", ".ssh", "outputs", "loops"}:
        raise IsolationViolation("credentials/control metadata cannot be a runtime")
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            info = path.lstat()
            mode = info.st_mode
            if stat.S_ISLNK(mode):
                target = path.resolve()
                if not (
                    target.is_relative_to(root)
                    or target.is_relative_to("/usr")
                    or any(target.is_relative_to(other) for other in approved)
                    or any(
                        target.is_relative_to(Path("/runtime") / str(index))
                        for index in range(len(approved))
                    )
                ):
                    raise IsolationViolation(
                        f"runtime link escapes approved runtime: {path}"
                    )
            elif stat.S_ISREG(mode) and info.st_nlink != 1:
                raise IsolationViolation(f"hardlinked runtime file: {path}")
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
    command += _node_module_mounts(workspace, spec.runtime_roots)
    command += _writable_mounts(spec, workspace)
    for path in ("/tmp", "/scratch"):
        command.extend(("--size", str(spec.scratch_bytes), "--tmpfs", path))
    if spec.provider_socket is not None:
        command.extend(_provider_mounts(spec.provider_socket))
        argv = ("/usr/bin/python3", "-I", "-S", "/provider/bridge.py", "/provider/socket", *argv)
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


def _provider_mounts(path: Path) -> list[str]:
    import stat
    from slm_training.harness_core import provider_bridge

    if not path.is_absolute() or path.is_symlink() or not stat.S_ISSOCK(path.stat().st_mode):
        raise IsolationViolation("provider transport must be a controller-owned Unix socket")
    return ["--dir", "/provider", "--ro-bind", str(path), "/provider/socket",
            "--ro-bind", str(Path(provider_bridge.__file__)), "/provider/bridge.py"]


def _mount_readonly_siblings(
    command: list[str], parent: Path, workspace: Path, allowed: set[str]
) -> None:
    for sibling in sorted(parent.iterdir()):
        sibling_relative = sibling.relative_to(workspace).as_posix()
        if sibling_relative not in allowed:
            command.extend(("--ro-bind", str(sibling), f"/workspace/{sibling_relative}"))


def _writable_mounts(spec: IsolationSpec, workspace: Path) -> list[str]:
    command: list[str] = []
    parents: dict[str, set[str]] = {}
    for relative in spec.writable_paths:
        path = checked_path(workspace, relative)
        if relative.split("/", 1)[0].startswith("."):
            raise IsolationViolation("metadata cannot be a writable source mount")
        if path.is_dir():
            raise IsolationViolation("source repair needs exact file grants")
        if repair_classification((relative,)) == "trust_policy_change":
            raise IsolationViolation("protected surface cannot be a writable mount")
        parent = path.parent
        parent_relative = parent.relative_to(workspace).as_posix()
        if parent == workspace:
            raise IsolationViolation("writable source parent cannot be workspace root")
        parents.setdefault(parent_relative, set()).add(relative)
    for relative in sorted(parents):
        parent = checked_path(workspace, relative)
        command.extend(("--bind", str(parent), f"/workspace/{relative}"))
        _mount_readonly_siblings(command, parent, workspace, parents[relative])
    for relative in spec.writable_dirs:
        path = checked_path(workspace, relative)
        if not path.is_dir() or relative.split("/", 1)[0].startswith("."):
            raise IsolationViolation("writable directory is not a disposable directory")
        command.extend(("--tmpfs", f"/workspace/{relative}"))
    return command

def _node_module_mounts(workspace: Path, runtimes: tuple[Path, ...]) -> list[str]:
    mounts: list[str] = []
    for root in (path for path in runtimes if path.name == "node_modules" and path.is_dir()):
        manifest = root.parent / "package.json"
        try:
            package_name = json.loads(manifest.read_text()).get("name")
        except (OSError, ValueError):
            continue
        candidates = (
            workspace / "candidate",
            workspace / "candidate/src/apps/openui_bridge",
        )
        target = next(
            (
                path
                for path in candidates
                if (path / "package.json").is_file()
                and json.loads((path / "package.json").read_text()).get("name")
                == package_name
            ),
            None,
        )
        if target is None:
            continue
        destination = target / "node_modules"
        if destination.is_symlink():
            raise IsolationViolation("node_modules mountpoint cannot be a symlink")
        destination.mkdir(exist_ok=True)
        mounts.extend(("--ro-bind", str(root), f"/workspace/{destination.relative_to(workspace)}"))
    return mounts

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
    with _codex_mount_target(spec):
        return _run_checked(spec, command, started, (on_start, on_heartbeat, cancel_event))


@contextmanager
def _codex_mount_target(spec):
    """Empty readonly target for nested Codex; never copy host Git metadata.

    build_isolated_command already refused a preexisting Git directory. Remove
    the controller-created target only after the bounded process tree exits.
    """
    target = spec.workspace / ".git" if spec.codex_mount_target else None
    if target is not None:
        target.mkdir()
    try:
        yield
    finally:
        if target is not None:
            target.rmdir()


def _run_checked(spec, command, started, callbacks):
    on_start, on_heartbeat, cancel_event = callbacks
    before = tree_manifest(spec.workspace)
    remaining = spec.timeout_seconds - (time.monotonic() - started)
    if spec.codex_mount_target:
        # Agent deadline includes termination and the mandatory post-run scope hash.
        remaining -= KILL_GRACE_SECONDS + HARNESS_FINALIZATION_RESERVE_SECONDS
        if remaining <= 0:
            raise TimeoutError("repair deadline cannot fund execution and cleanup")
    if remaining <= 0:
        # A tiny diagnostic timeout may be consumed by mandatory setup; keep a
        # minimal child slice so cleanup/termination is still exercised.
        remaining = 0.001
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
    finally:
        after = tree_manifest(spec.workspace)
        for relative in spec.writable_paths:
            path = spec.workspace / relative
            if path.is_symlink():
                raise IsolationViolation("exact writable source path must remain a regular file")
            path = checked_path(spec.workspace, relative, must_exist=False)
            if path.exists():
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise IsolationViolation("exact writable source path must remain a private regular file")
        changed = scope_changes(
            before,
            after,
            spec.writable_paths,
            recursive=spec.writable_dirs,
        )
        if changed:
            raise IsolationViolation(f"snapshot scope changed: {changed}")
    return replace(result, duration_seconds=time.monotonic() - started)
