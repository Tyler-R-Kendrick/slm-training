#!/usr/bin/env python3
"""Build and audit the pinned Lean formal-contract package."""

from __future__ import annotations

import ast
import fcntl
import hashlib
import importlib.util
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEAN_ROOT = ROOT / "src" / "slm_training" / "formal" / "lean"
FORBIDDEN_SOURCE = re.compile(r"\b(?:sorry|admit|axiom)\b")


def _run_formal(command: list[str], *, timeout_seconds: float):
    """Run Lean without importing DSL-heavy package initializers in CI."""

    source = ROOT / "src" / "slm_training" / "harness_core" / "bounded_process.py"
    spec = importlib.util.spec_from_file_location("_formal_bounded_process", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load bounded process runner: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    total, interrupt_after, grace = _formal_process_budget(timeout_seconds)
    lock_path = Path(tempfile.gettempdir()) / (
        "slm-formal-"
        + hashlib.sha256(str(LEAN_ROOT.resolve()).encode()).hexdigest()[:24]
        + ".lock"
    )
    started = time.monotonic()
    with lock_path.open("a+") as handle:
        deadline = started + total
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("formal project lock timed out")
                time.sleep(min(0.05, remaining))
        result = module.run_bounded_process(
            command,
            cwd=LEAN_ROOT,
            interrupt_after_seconds=min(
                interrupt_after, max(0.001, deadline - time.monotonic())
            ),
            kill_grace_seconds=grace,
        )
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    if result.timed_out:
        raise subprocess.TimeoutExpired(
            command, total, output=result.stdout, stderr=result.stderr
        )
    if result.outcome.value == "launch_failed":
        raise OSError(result.launch_error or "formal command launch failed")
    if result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


def _formal_process_budget(timeout_seconds: float) -> tuple[float, float, float]:
    """Derive the same cap values without importing ``slm_training``."""

    policy = ast.parse((ROOT / "src" / "slm_training" / "levers.py").read_text())
    values: dict[str, float] = {}
    expressions: dict[str, ast.expr] = {}
    for node in ast.walk(policy):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            value_node = node.value
        else:
            continue
        if value_node is not None:
            if isinstance(target, ast.Name) and target.id in {
                "MAX_RUN_MINUTES",
                "MAX_RUN_SECONDS",
                "INTERRUPT_AFTER_SECONDS",
                "KILL_GRACE_SECONDS",
            }:
                expressions[target.id] = value_node

    def evaluate(name: str) -> float:
        if name in values:
            return values[name]
        expression = expressions[name]
        if isinstance(expression, ast.Constant) and isinstance(
            expression.value, (int, float)
        ):
            value = float(expression.value)
        elif isinstance(expression, ast.Name):
            value = evaluate(expression.id)
        elif isinstance(expression, ast.BinOp) and isinstance(
            expression.op, (ast.Mult, ast.Sub, ast.Add)
        ):
            left = _evaluate_node(expression.left, evaluate)
            right = _evaluate_node(expression.right, evaluate)
            value = (
                left * right
                if isinstance(expression.op, ast.Mult)
                else left - right
                if isinstance(expression.op, ast.Sub)
                else left + right
            )
        else:
            raise ValueError(f"unsupported formal policy expression: {name}")
        values[name] = value
        return value

    values.setdefault("MAX_RUN_MINUTES", evaluate("MAX_RUN_MINUTES"))
    for name in ("MAX_RUN_SECONDS", "INTERRUPT_AFTER_SECONDS", "KILL_GRACE_SECONDS"):
        evaluate(name)
    total = min(values["MAX_RUN_SECONDS"], max(0.001, float(timeout_seconds)))
    grace = min(values["KILL_GRACE_SECONDS"], total * 0.1)
    interrupt = min(values["INTERRUPT_AFTER_SECONDS"], max(0.001, total - grace))
    return total, interrupt, grace


def _evaluate_node(node: ast.expr, resolve) -> float:
    if isinstance(node, ast.Name):
        return resolve(node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    raise ValueError("unsupported formal policy expression node")


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
