"""Shared helpers for persistent subprocess bridges."""

from __future__ import annotations

import queue
import subprocess
import threading


def readline_with_timeout(
    proc: subprocess.Popen[str],
    timeout_s: float,
    *,
    error_message: str,
) -> str:
    """Read one subprocess line without allowing a blocked pipe to hang."""
    assert proc.stdout is not None
    result: queue.Queue[str | BaseException] = queue.Queue(maxsize=1)

    def _read() -> None:
        try:
            result.put(proc.stdout.readline())
        except BaseException as exc:  # pragma: no cover - defensive pipe failure
            result.put(exc)

    threading.Thread(target=_read, daemon=True).start()
    try:
        value = result.get(timeout=max(0.001, timeout_s))
    except queue.Empty as exc:
        raise subprocess.TimeoutExpired(proc.args, timeout_s) from exc
    if isinstance(value, BaseException):
        raise RuntimeError(error_message) from value
    return value
