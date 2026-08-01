from __future__ import annotations

import sys
import time
from pathlib import Path

from slm_training.harness_core.bounded_process import (
    ProcessOutcome,
    _stop_and_reap,
    run_bounded_process,
)


def _run_python(code: str, **kwargs: object):
    return run_bounded_process(
        [sys.executable, "-c", code],
        interrupt_after_seconds=2,
        kill_grace_seconds=0.2,
        **kwargs,
    )


def test_completed_process_captures_output_and_returncode() -> None:
    result = _run_python(
        "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"
    )

    assert result.outcome is ProcessOutcome.COMPLETED
    assert result.returncode == 3
    assert result.stdout == "out\n"
    assert result.stderr == "err\n"
    assert not result.timed_out


def test_launch_failure_is_a_typed_result() -> None:
    result = run_bounded_process(
        ["/definitely/not/a/real/executable"],
        interrupt_after_seconds=1,
        kill_grace_seconds=0.1,
    )

    assert result.outcome is ProcessOutcome.LAUNCH_FAILED
    assert result.returncode is None
    assert result.launch_error is not None
    assert "FileNotFoundError" in result.launch_error


def test_timeout_interrupts_process_group_before_killing() -> None:
    result = run_bounded_process(
        [
            sys.executable,
            "-c",
            "import signal,time; "
            "signal.signal(signal.SIGINT, lambda *_: exit(17)); "
            "time.sleep(30)",
        ],
        interrupt_after_seconds=0.25,
        kill_grace_seconds=0.5,
    )

    assert result.outcome is ProcessOutcome.INTERRUPTED
    assert result.returncode == 17
    assert result.timed_out
    assert result.interrupted
    assert not result.killed


def test_timeout_kills_process_group_after_grace(tmp_path: Path) -> None:
    survivor = tmp_path / "grandchild-survived"
    grandchild = (
        "import signal,time,pathlib; "
        "signal.signal(signal.SIGINT, signal.SIG_IGN); "
        "time.sleep(1.0); "
        f"pathlib.Path({str(survivor)!r}).write_text('alive')"
    )
    parent = (
        "import signal,subprocess,sys,time; "
        "signal.signal(signal.SIGINT, lambda *_: sys.exit(23)); "
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
        "time.sleep(30)"
    )

    result = run_bounded_process(
        [sys.executable, "-c", parent],
        interrupt_after_seconds=0.2,
        kill_grace_seconds=0.1,
    )
    time.sleep(0.8)

    assert result.outcome is ProcessOutcome.KILLED
    assert result.returncode == 23
    assert result.timed_out
    assert result.interrupted
    assert result.killed
    assert not survivor.exists()


def test_zero_grace_reaps_already_exited_child_without_kill() -> None:
    import subprocess

    process = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        start_new_session=True,
    )
    time.sleep(0.1)

    interrupted, killed = _stop_and_reap(process, 0)

    assert interrupted
    assert not killed
    assert process.returncode == 0


def test_captured_output_is_bounded_to_tail() -> None:
    result = _run_python(
        "import sys; print('a' * 100 + 'TAIL'); "
        "print('b' * 100 + 'ERRTAIL', file=sys.stderr)",
        max_output_bytes=16,
    )

    assert len(result.stdout.encode()) <= 16
    assert len(result.stderr.encode()) <= 16
    assert result.stdout.endswith("TAIL\n")
    assert result.stderr.endswith("ERRTAIL\n")
    assert result.stdout_truncated
    assert result.stderr_truncated


def test_process_callbacks_expose_pid_and_refresh_heartbeat() -> None:
    started: list[int] = []
    heartbeats: list[int] = []
    result = run_bounded_process(
        [sys.executable, "-c", "import time; time.sleep(0.12)"],
        interrupt_after_seconds=1,
        kill_grace_seconds=0.1,
        heartbeat_interval_seconds=0.03,
        on_start=started.append,
        on_heartbeat=heartbeats.append,
    )

    assert result.outcome is ProcessOutcome.COMPLETED
    assert len(started) == 1
    assert heartbeats
    assert set(heartbeats) == set(started)
