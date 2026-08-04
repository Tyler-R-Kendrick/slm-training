"""Run a subprocess tree within explicit interrupt and kill budgets."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import BinaryIO


class ProcessOutcome(str, Enum):
    """Terminal outcome of a bounded subprocess run."""

    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"
    KILLED = "killed"
    LAUNCH_FAILED = "launch_failed"


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
    return True


def _stop_and_reap(
    process: subprocess.Popen[bytes], kill_grace_seconds: float
) -> tuple[bool, bool]:
    interrupted = _signal_process_group(process, signal.SIGINT)
    deadline = time.monotonic() + kill_grace_seconds
    process.poll()
    while _process_group_exists(process.pid) and time.monotonic() < deadline:
        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        process.poll()
    if not _process_group_exists(process.pid):
        process.wait()
        return interrupted, False
    killed = _signal_process_group(process, signal.SIGKILL)
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
) -> BoundedProcessResult:
    """Run ``command``, stopping its process group when the budget expires.

    Callers supply the canonical budget values owned by their policy layer.
    Output is disk-backed while the command runs and only bounded tails are
    retained in memory.
    """

    argv = tuple(command)
    if not argv:
        raise ValueError("command must not be empty")
    if interrupt_after_seconds <= 0:
        raise ValueError("interrupt_after_seconds must be positive")
    if kill_grace_seconds < 0:
        raise ValueError("kill_grace_seconds must be non-negative")
    if max_output_bytes <= 0:
        raise ValueError("max_output_bytes must be positive")
    if heartbeat_interval_seconds <= 0:
        raise ValueError("heartbeat_interval_seconds must be positive")

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

        timed_out = False
        interrupted = False
        killed = False
        try:
            if on_start is not None:
                on_start(process.pid)
            interrupt_deadline = time.monotonic() + interrupt_after_seconds
            while process.poll() is None:
                remaining = interrupt_deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    interrupted, killed = _stop_and_reap(process, kill_grace_seconds)
                    break
                try:
                    process.wait(timeout=min(heartbeat_interval_seconds, remaining))
                except subprocess.TimeoutExpired:
                    if on_heartbeat is not None:
                        on_heartbeat(process.pid)
        except BaseException:
            _stop_and_reap(process, kill_grace_seconds)
            raise

        stdout, stdout_truncated = _tail(stdout_file, max_output_bytes)
        stderr, stderr_truncated = _tail(stderr_file, max_output_bytes)
        if killed:
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
        )
