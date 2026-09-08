"""Controller policy for routine source repair; never worker-supplied authority."""

from __future__ import annotations

import ast
from pathlib import Path

from .isolation_workspace import IsolationViolation, relative_path


_PROTECTED = (
    ".",
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    "RTK.md",
    "pyproject.toml",
    "package.json",
    "package-lock.json",
    "pytest.ini",
    "setup.cfg",
    "setup.py",
    "tox.ini",
    "conftest.py",
    "src/slm_training/resources/",
    "src/slm_training/dsl/",
    "src/slm_training/formal/",
    "src/leverproof_lean/",
    "src/slm_training/harness_core/",
    "src/slm_training/autoresearch/",
    "src/slm_training/levers.py",
    "src/slm_training/__init__.py",
    "src/slm_training/versioning.py",
    "scripts/verify",
    "scripts/check",
    "scripts/run_autotrain",
    "docs/",
    "src/slm_training/models/",
    "src/slm_training/evals/",
    "src/slm_training/harnesses/model_build/eval_policy.py",
    "src/slm_training/harnesses/model_build/ship_gates.py",
)
_MEASUREMENT = (
    "src/slm_training/evals/",
    "src/slm_training/harnesses/model_build/eval",
    "src/slm_training/harnesses/model_build/ship_gates",
    "src/slm_training/harnesses/model_build/loss",
    "scripts/evaluate",
    "scripts/autotrain_metrics",
    "scripts/autotrain_measurement",
)


def repair_classification(paths: tuple[str, ...]) -> str:
    """Conservative review routing; no claim to prove semantic equivalence."""
    normalized = tuple(relative_path(path) for path in paths)
    if any(path.startswith(_PROTECTED) for path in normalized):
        return "trust_policy_change"
    if any(path.startswith(_MEASUREMENT) for path in normalized):
        return "measurement_semantic_change"
    return "implementation_repair"


def require_routine_scope(
    base: Path, candidate: Path, changed: tuple[str, ...], allowed: tuple[str, ...],
    equivalence_paths: tuple[str, ...] = (),
) -> None:
    """Only exact granted paths; existing tests cannot be edited or deleted."""
    if not changed:
        raise IsolationViolation("no candidate change")
    if not set(changed).issubset({relative_path(path) for path in allowed}):
        raise IsolationViolation("candidate changes exceed exact repair grant")
    classification = repair_classification(changed)
    semantic_paths = tuple(path for path in changed if path.startswith(_MEASUREMENT))
    if classification == "trust_policy_change" or (
        semantic_paths and not set(semantic_paths).issubset(set(equivalence_paths))
    ):
        raise IsolationViolation("change requires separate science/trust review")
    new_tests = []
    for relative in changed:
        path = candidate / relative
        if not path.is_file() or path.is_symlink():
            raise IsolationViolation("routine repair may not delete or link source")
        if relative.startswith("tests/"):
            if not path.name.startswith("test_") or path.suffix != ".py":
                raise IsolationViolation("only new regression modules may be added")
            if (base / relative).exists():
                raise IsolationViolation("existing regression tests are protected")
            _require_regression(path)
            new_tests.append(relative)
    if not new_tests:
        raise IsolationViolation("repair must add a reproducer-based regression")


def _require_regression(path: Path) -> None:
    tree = ast.parse(path.read_text())
    forbidden = {"skip", "skipif", "xfail", "importorskip", "exit", "_exit"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in forbidden:
            raise IsolationViolation("new test contains skip/xfail/exit escape")
        if isinstance(node, ast.Name) and node.id in forbidden:
            raise IsolationViolation("new test contains skip/xfail/exit escape")
    if not any(isinstance(node, ast.Assert) and not isinstance(node.test, ast.Constant)
               for node in ast.walk(tree)):
        raise IsolationViolation("new regression has no assertion")
