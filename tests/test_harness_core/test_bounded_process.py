from __future__ import annotations

import sys
import time
from pathlib import Path

from slm_training.harness_core.bounded_process import (
    ProcessOutcome,
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
        "time.sleep(0.4); "
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
    time.sleep(0.35)

    assert result.outcome is ProcessOutcome.KILLED
    assert result.returncode == 23
    assert result.timed_out
    assert result.interrupted
    assert result.killed
    assert not survivor.exists()


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
