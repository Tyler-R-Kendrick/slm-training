"""Shared helpers for bridges and repo-local path resolution."""

from __future__ import annotations

import queue
import subprocess
import threading
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Locate the repository root by walking up from this file.

    Prefer an anchor file over fixed ``parents[N]`` so packages can nest
    deeper without breaking bridge / fixture paths.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    # Fallback: src/slm_training → repo (package root's parent's parent).
    return here.parents[1]


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
