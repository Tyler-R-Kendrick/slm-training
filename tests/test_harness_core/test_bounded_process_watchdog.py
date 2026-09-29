from __future__ import annotations

import sys
import threading
import time

import pytest

from slm_training.harness_core.bounded_process import ProcessOutcome, run_bounded_process


def run(code, **kwargs):
    return run_bounded_process([sys.executable, "-c", code], interrupt_after_seconds=.25,
                               kill_grace_seconds=.1, heartbeat_interval_seconds=.01, **kwargs)


@pytest.mark.parametrize("callback", ["on_start", "on_heartbeat", "progress_probe"])
def test_hung_telemetry_callback_cannot_disable_deadline(callback):
    release = threading.Event()
    started = time.monotonic()
    try:
        result = run("import time; time.sleep(30)", **{callback: lambda *_: release.wait(5)})
    finally:
        release.set()
    assert time.monotonic() - started < 2
    assert result.timed_out and result.interrupted


def test_zero_exit_parent_does_not_leave_grandchild(tmp_path):
    survivor = tmp_path / "escaped"
    child = f"import time,pathlib; time.sleep(.5); pathlib.Path({str(survivor)!r}).touch()"
    result = run(f"import subprocess,sys; subprocess.Popen([sys.executable, '-c', {child!r}])")
    time.sleep(.5)
    assert not survivor.exists()
    assert result.interrupted or result.killed


def test_nested_session_is_killed_but_unrelated_child_survives(tmp_path):
    import subprocess
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"], start_new_session=True)
    survivor = tmp_path / "nested-escaped"
    child = ("import signal,time,pathlib; signal.signal(signal.SIGINT, signal.SIG_IGN); "
             f"time.sleep(.7); pathlib.Path({str(survivor)!r}).touch()")
    parent = ("import subprocess,sys,time; "
              f"subprocess.Popen([sys.executable, '-c', {child!r}], start_new_session=True); time.sleep(5)")
    try:
        result = run(parent)
        time.sleep(.7)
        assert result.killed
        assert not survivor.exists()
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=1)


def test_cancellation_and_unchanged_progress_are_distinct():
    cancel = threading.Event()
    cancel.set()
    assert run("import time; time.sleep(30)", cancel_event=cancel).outcome == ProcessOutcome.CANCELLED
    result = run("import time; time.sleep(30)", progress_probe=lambda: "same-row-digest",
                 progress_timeout_seconds=.06)
    assert result.outcome == ProcessOutcome.STALLED and result.progress_stalled


def test_system_exit_in_observer_cleans_up_owned_process(tmp_path):
    survivor = tmp_path / "escaped"
    def fail(_pid):
        raise SystemExit(7)
    with pytest.raises(SystemExit):
        run(f"import time,pathlib; time.sleep(.5); pathlib.Path({str(survivor)!r}).touch()", on_start=fail)
    time.sleep(.5)
    assert not survivor.exists()


@pytest.mark.parametrize("budget", [float("nan"), float("inf"), 171, -1])
def test_nonfinite_or_excessive_budget_rejected(budget):
    with pytest.raises(ValueError):
        run_bounded_process([sys.executable, "-c", "pass"], interrupt_after_seconds=budget,
                            kill_grace_seconds=.1)



def test_output_flood_does_not_block_deadline():
    result = run("import os\nwhile True: os.write(1, b'x' * 65536)", max_output_bytes=128)
    assert result.timed_out
    assert result.stdout_truncated and len(result.stdout.encode()) == 128


def test_tree_does_not_signal_reused_pid(monkeypatch):
    from slm_training.harness_core import bounded_process, process_tree

    tree = process_tree.OwnedProcessTree.__new__(process_tree.OwnedProcessTree)
    tree.root = 999999
    tree.identities = {999999: "original-start"}
    monkeypatch.setattr(bounded_process, "_process_start_identity", lambda _pid: "replacement-start")
    signals = []
    monkeypatch.setattr(bounded_process.os, "kill", lambda *args: signals.append(args))
    tree.signal(9)
    assert signals == [] and not tree.alive()
    assert tree.identities == {999999: "original-start"}


def test_observer_start_failure_reaps_launched_worker(monkeypatch):
    from slm_training.harness_core import bounded_process

    processes = []
    popen = bounded_process.subprocess.Popen

    def launch(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    def fail_start(_thread):
        raise RuntimeError("injected observer start failure")

    monkeypatch.setattr(bounded_process.subprocess, "Popen", launch)
    monkeypatch.setattr(threading.Thread, "start", fail_start)
    try:
        with pytest.raises(RuntimeError, match="observer start failure"):
            run("import time; time.sleep(30)")
        assert len(processes) == 1 and processes[0].poll() is not None
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=1)


def test_budget_limits_follow_canonical_expressions(monkeypatch):
    from slm_training.harness_core import bounded_process
    from slm_training import levers

    assert bounded_process._policy_limits() == (levers.INTERRUPT_AFTER_SECONDS, levers.KILL_GRACE_SECONDS)
    monkeypatch.setattr(bounded_process.Path, "read_text", lambda _path:
        "MAX_RUN_MINUTES: int = 2\nKILL_GRACE_SECONDS: int = 7\n"
        "MAX_RUN_SECONDS: int = MAX_RUN_MINUTES * 60\n"
        "INTERRUPT_AFTER_SECONDS: int = MAX_RUN_SECONDS - KILL_GRACE_SECONDS\n")
    assert bounded_process._policy_limits() == (113, 7)


def test_unsupported_budget_policy_fails_closed(monkeypatch):
    from slm_training.harness_core import bounded_process

    monkeypatch.setattr(bounded_process.Path, "read_text", lambda _path:
        "KILL_GRACE_SECONDS: int = 10\nINTERRUPT_AFTER_SECONDS: int = arbitrary()\n")
    with pytest.raises(ValueError, match="unsupported canonical run-policy"):
        bounded_process._policy_limits()


def test_formal_bootstrap_runs_without_project_dependencies():
    import subprocess

    result = subprocess.run(
        [sys.executable, "-S", "-c",
         "import runpy; ns=runpy.run_path('scripts/verify_formal_contracts.py', run_name='bootstrap'); "
         "result=ns['_run_formal'](['/bin/true'], timeout_seconds=1); assert result.returncode == 0"],
        check=False, capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("live", [False, True])
def test_orphan_group_distinguishes_zombies_from_live_workers(tmp_path, live):
    """A separate subreaper holds real orphan zombies without changing pytest ownership."""
    import subprocess

    child = r"""
import os, signal, sys, time
from pathlib import Path
pid = os.fork()
if pid == 0:
    if sys.argv[1] == 'live':
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        Path(sys.argv[2]).touch()
        time.sleep(30)
    os._exit(0)
while True:
    state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
    if (sys.argv[1] == 'zombie' and state == 'Z') or Path(sys.argv[2]).exists():
        break
    time.sleep(.005)
print(pid, flush=True)
os._exit(0)
"""
    controller = r"""
import ctypes, os, signal, sys
from pathlib import Path
from slm_training.harness_core.bounded_process import run_bounded_process, ProcessOutcome
assert ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0
result = run_bounded_process([sys.executable, '-c', sys.argv[1], sys.argv[2], sys.argv[3]],
    interrupt_after_seconds=3, kill_grace_seconds=.1, heartbeat_interval_seconds=.01)
pid = int(result.stdout.strip())
try:
    assert result.returncode == 0 and not result.timed_out, result
    if sys.argv[2] == 'live':
        assert result.outcome == ProcessOutcome.KILLED and result.killed, result
    else:
        assert Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z'
        os.killpg(os.getpgid(pid), 0)  # Kernel group existence alone is insufficient.
        assert result.outcome == ProcessOutcome.COMPLETED, result
        assert not result.interrupted and not result.killed, result
finally:
    os.kill(pid, signal.SIGKILL)
    reaped, status = os.waitpid(pid, 0)
assert reaped == pid
if sys.argv[2] == 'live':
    assert os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGKILL
else:
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
"""
    result = subprocess.run([sys.executable, "-c", controller, child,
        "live" if live else "zombie", str(tmp_path / "ready")],
        capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_group_liveness_remains_conservative_without_proc_access(monkeypatch):
    from slm_training.harness_core import bounded_process

    monkeypatch.setattr(bounded_process.os, "killpg", lambda *_: None)

    def denied(_path):
        raise PermissionError("injected unreadable process table")

    monkeypatch.setattr(bounded_process.Path, "iterdir", denied)
    assert bounded_process._process_group_exists(12345)
