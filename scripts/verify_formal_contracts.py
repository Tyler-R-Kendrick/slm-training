#!/usr/bin/env python3
"""Build and audit the pinned Lean formal-contract package."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEAN_ROOT = ROOT / "src" / "slm_training" / "formal" / "lean"
FORBIDDEN_SOURCE = re.compile(r"\b(?:sorry|admit|axiom)\b")


def _run_formal(command: list[str], *, timeout_seconds: float):
    """Run a Lean command with the canonical process-group and project lock."""

    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from slm_training.formal.checkers import (
        FormalProjectLock,
        ProcessOutcome,
        formal_process_budget,
        run_formal_process,
    )

    total, _interrupt_after, _grace = formal_process_budget(timeout_seconds)
    with FormalProjectLock(LEAN_ROOT, timeout_seconds=total) as lock:
        result = run_formal_process(
            command,
            cwd=LEAN_ROOT,
            timeout_seconds=max(0.001, total - lock.wait_seconds),
        )
    if result.timed_out:
        raise subprocess.TimeoutExpired(command, total, output=result.stdout, stderr=result.stderr)
    if result.outcome is ProcessOutcome.LAUNCH_FAILED:
        raise OSError(result.launch_error or "formal command launch failed")
    if result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


def _canonical_run_seconds() -> int:
    """Read the hard cap without importing the not-yet-installed package."""
    path = ROOT / "src" / "slm_training" / "levers.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "MAX_RUN_MINUTES"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
        ):
            return node.value.value * 60
    raise ValueError(f"MAX_RUN_MINUTES must be an integer literal in {path}")


def main() -> int:
    sources = (LEAN_ROOT / "OpenUIProofs.lean",) + tuple(
        sorted((LEAN_ROOT / "OpenUIProofs").rglob("*.lean"))
    )
    forbidden = [
        str(path.relative_to(ROOT))
        for path in sources
        if FORBIDDEN_SOURCE.search(path.read_text(encoding="utf-8"))
    ]
    if forbidden:
        raise SystemExit(f"formal proof placeholders or axioms found: {forbidden}")
    deadline = time.monotonic() + _canonical_run_seconds() - 1

    def remaining() -> float:
        return max(0.001, deadline - time.monotonic())

    _run_formal(["lake", "build", "OpenUIProofs"], timeout_seconds=remaining())
    audit = _run_formal(
        ["lake", "env", "lean", "OpenUIProofs/Axioms.lean"],
        timeout_seconds=remaining(),
    )
    output = f"{audit.stdout}\n{audit.stderr}"
    if "sorryAx" in output:
        raise SystemExit("formal theorem audit contains Lean sorryAx")
    print("formal contracts: build and axiom audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
