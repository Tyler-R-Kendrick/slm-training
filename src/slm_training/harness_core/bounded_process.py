"""Run a subprocess tree within explicit interrupt and kill budgets."""

from __future__ import annotations

import ast
import math
import operator
import os
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import BinaryIO


def _process_start_identity(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


class OwnedProcessTree:
    """Remember observed descendants even if the root later exits/reparents them."""

    def __init__(self, root: int):
        self.root = root
        self.identities: dict[int, str] = {}
        self.refresh()

    def refresh(self):
        pending = [self.root, *self.identities]
        seen: set[int] = set()
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            current = _process_start_identity(pid)
            if current is None or (pid in self.identities and self.identities[pid] != current):
                continue
            self.identities[pid] = current
            try:
                for task in Path(f"/proc/{pid}/task").iterdir():
                    pending.extend(int(value) for value in (task / "children").read_text().split())
            except (FileNotFoundError, ProcessLookupError):
                continue

    def alive(self) -> bool:
        return any(_process_start_identity(pid) == start for pid, start in self.identities.items())

    def signal(self, sig: int, *, excluding_group: int | None = None):
        self.refresh()
        for pid, start in self.identities.items():
            if _process_start_identity(pid) == start:
                try:
                    if excluding_group is not None and os.getpgid(pid) == excluding_group:
                        continue
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass


class ProcessOutcome(str, Enum):
    """Terminal outcome of a bounded subprocess run."""

    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"
    KILLED = "killed"
    LAUNCH_FAILED = "launch_failed"
    CANCELLED = "cancelled"
    STALLED = "stalled"


@dataclass(frozen=True)
class BoundedProcessResult:
    """Captured result from :func:`run_bounded_process`."""

    command: tuple[str, ...]
    outcome: ProcessOutcome
    returncode: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    interrupted: bool = False
    killed: bool = False
    launch_error: str | None = None
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    cancelled: bool = False
    progress_stalled: bool = False


class _Observer:
    """Trusted telemetry callbacks never run on the deadline-enforcing thread."""

    def __init__(self, process, on_start, on_heartbeat, progress_probe, interval):
        self.done = threading.Event()
        self.stopped = threading.Event()
        self.error: BaseException | None = None
        self.last_progress = time.monotonic()
        self.digest: str | None = None
        self.tree = OwnedProcessTree(process.pid)
        self.thread = threading.Thread(target=self._observe,
            args=(process, on_start, on_heartbeat, progress_probe, interval), daemon=True)

    def _observe(self, process, on_start, on_heartbeat, probe, interval):
        try:
            if on_start:
                on_start(process.pid)
            while not self.stopped.is_set():
                if probe:
                    digest = probe()
                    if digest and digest != self.digest:
                        self.digest = digest
                        self.last_progress = time.monotonic()
                if self.stopped.wait(interval):
                    break
                if on_heartbeat:
                    on_heartbeat(process.pid)
        except BaseException as exc:
            self.error = exc
        finally:
            self.done.set()


def _wait_owned(process, observer, *, deadline, grace, cancel_event, progress_timeout):
    timed_out = cancelled = stalled = interrupted = killed = False
    tree = observer.tree
    while True:
        tree.refresh()
        now = time.monotonic()
        if observer.error is not None:
            raise observer.error
        cancelled = cancel_event is not None and cancel_event.is_set()
        stalled = progress_timeout is not None and now - observer.last_progress >= progress_timeout
        timed_out = now >= deadline
        if cancelled or stalled or timed_out:
            interrupted, killed = _stop_and_reap(process, grace, tree=tree)
            break
        if process.poll() is not None:
            observer.stopped.set()
            if observer.done.is_set():
                # A zero-exit parent cannot strand its owned descendants.
                if _process_group_exists(process.pid) or tree.alive():
                    interrupted, killed = _stop_and_reap(process, grace, tree=tree)
                break
        time.sleep(min(0.01, max(0, deadline - now)))
    return timed_out, cancelled, stalled, interrupted, killed


def _policy_limits():
    """Read canonical expressions without importing DSL-coupled lever discovery."""
    policy = Path(__file__).parents[1] / "levers.py"
    expressions = {
        node.target.id: node.value
        for node in ast.parse(policy.read_text()).body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name):
            return value(expressions[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult):
            operation = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
            return operation[type(node.op)](value(node.left), value(node.right))
        raise ValueError("unsupported canonical run-policy expression")

    return value(expressions["INTERRUPT_AFTER_SECONDS"]), value(expressions["KILL_GRACE_SECONDS"])


INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS = _policy_limits()


def _validate_budget(interrupt, grace, heartbeat, progress_timeout):
    if not math.isfinite(interrupt) or not 0 < interrupt <= INTERRUPT_AFTER_SECONDS:
        raise ValueError("interrupt budget must be positive and within canonical cap")
    if not math.isfinite(grace) or not 0 <= grace <= KILL_GRACE_SECONDS:
        raise ValueError("kill grace must be non-negative and within canonical cap")
    if not math.isfinite(heartbeat) or heartbeat <= 0:
        raise ValueError("heartbeat interval must be finite and positive")
    if progress_timeout is not None and (not math.isfinite(progress_timeout) or progress_timeout <= 0):
        raise ValueError("progress timeout must be finite and positive")


def _tail(stream: BinaryIO, limit: int) -> tuple[str, bool]:
    stream.flush()
    size = stream.seek(0, os.SEEK_END)
    stream.seek(max(0, size - limit))
    return stream.read(limit).decode("utf-8", errors="replace"), size > limit


def _signal_process_group(process: subprocess.Popen[bytes], sig: int) -> bool:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return False
    except OSError:
        if process.poll() is not None:
            return False
        try:
            process.send_signal(sig)
        except ProcessLookupError:
            return False
    return True


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # killpg(0) also succeeds for zombie-only groups. Those members cannot run
    # or handle a signal; their parent owns reaping, not this process runner.
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            except (FileNotFoundError, ProcessLookupError):
                continue
            if int(fields[2]) == process_group_id and fields[0] not in {"Z", "X"}:
                return True
    except (OSError, ValueError, IndexError):
        return True  # Unavailable process inspection must not waive cleanup.
    return False


def _stop_and_reap(
    process: subprocess.Popen[bytes], kill_grace_seconds: float, *, tree: OwnedProcessTree | None = None
) -> tuple[bool, bool]:
    tree = tree or OwnedProcessTree(process.pid)
    interrupted = _signal_process_group(process, signal.SIGINT)
    tree.signal(signal.SIGINT, excluding_group=process.pid)
    deadline = time.monotonic() + kill_grace_seconds
    process.poll()
    while (_process_group_exists(process.pid) or tree.alive()) and time.monotonic() < deadline:
        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        process.poll()
    if not _process_group_exists(process.pid) and not tree.alive():
        process.wait()
        return interrupted, False
    killed = _signal_process_group(process, signal.SIGKILL)
    if tree.alive():
        tree.signal(signal.SIGKILL)
        killed = True
    process.wait()
    return interrupted, killed


def run_bounded_process(
    command: Sequence[str],
    *,
    interrupt_after_seconds: float,
    kill_grace_seconds: float,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    max_output_bytes: int = 8_000,
    on_start: Callable[[int], None] | None = None,
    on_heartbeat: Callable[[int], None] | None = None,
    heartbeat_interval_seconds: float = 1.0,
    cancel_event: threading.Event | None = None,
    progress_probe: Callable[[], str | None] | None = None,
    progress_timeout_seconds: float | None = None,
) -> BoundedProcessResult:
    """Run ``command``, stopping its process group when the budget expires.

    Callers supply the canonical budget values owned by their policy layer.
    Output is disk-backed while the command runs and only bounded tails are
    retained in memory.
    """

    argv = tuple(command)
    if not argv:
        raise ValueError("command must not be empty")
    _validate_budget(interrupt_after_seconds, kill_grace_seconds,
                     heartbeat_interval_seconds, progress_timeout_seconds)
    if max_output_bytes <= 0:
        raise ValueError("max_output_bytes must be positive")
    if progress_timeout_seconds is not None and progress_probe is None:
        raise ValueError("progress timeout requires a relevant artifact/progress probe")

    started = time.monotonic()
    with (
        tempfile.TemporaryFile() as stdout_file,
        tempfile.TemporaryFile() as stderr_file,
    ):
        try:
            process = subprocess.Popen(
                argv,
                cwd=Path(cwd) if cwd is not None else None,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
        except OSError as exc:
            return BoundedProcessResult(
                command=argv,
                outcome=ProcessOutcome.LAUNCH_FAILED,
                returncode=None,
                stdout="",
                stderr="",
                duration_seconds=time.monotonic() - started,
                launch_error=f"{type(exc).__name__}: {exc}",
            )

        observer = _Observer(process, on_start, on_heartbeat, progress_probe,
                             heartbeat_interval_seconds)
        try:
            observer.thread.start()
            timed_out, cancelled, stalled, interrupted, killed = _wait_owned(
                process, observer, deadline=started + interrupt_after_seconds,
                grace=kill_grace_seconds, cancel_event=cancel_event,
                progress_timeout=progress_timeout_seconds)
        except BaseException:
            _stop_and_reap(process, kill_grace_seconds, tree=observer.tree)
            raise
        finally:
            observer.stopped.set()

        stdout, stdout_truncated = _tail(stdout_file, max_output_bytes)
        stderr, stderr_truncated = _tail(stderr_file, max_output_bytes)
        if cancelled:
            outcome = ProcessOutcome.CANCELLED
        elif stalled:
            outcome = ProcessOutcome.STALLED
        elif killed:
            outcome = ProcessOutcome.KILLED
        elif interrupted:
            outcome = ProcessOutcome.INTERRUPTED
        elif timed_out:
            outcome = ProcessOutcome.TIMED_OUT
        else:
            outcome = ProcessOutcome.COMPLETED
        return BoundedProcessResult(
            command=argv,
            outcome=outcome,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=time.monotonic() - started,
            timed_out=timed_out,
            interrupted=interrupted,
            killed=killed,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            cancelled=cancelled,
            progress_stalled=stalled,
        )
