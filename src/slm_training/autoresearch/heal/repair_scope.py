"""Controller policy for routine source repair; never worker-supplied authority."""

from __future__ import annotations

import ast
import json
import os
import re
import stat
import tempfile
from datetime import UTC, datetime
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
_VERSION_REGISTRY = "src/slm_training/resources/versions.json"
_NUMERIC_VERSION = re.compile(r"v([0-9]+)")
_SEMVER = re.compile(r"([0-9]+)\.([0-9]+)\.([0-9]+)")
_SUFFIX_VERSION = re.compile(r"(.*(?:_|-)v)([0-9]+)")


def _component_for_path(path: str, components: dict) -> str | None:
    claims = sorted(
        ((claim, name) for name, row in components.items() for claim in row.get("paths", ())),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for claim, name in claims:
        if path == claim or claim.endswith("/") and path.startswith(claim):
            return name
    return None


def _next_version(value: object) -> str:
    version = str(value)
    if match := _NUMERIC_VERSION.fullmatch(version):
        return f"v{int(match.group(1)) + 1}"
    if match := _SEMVER.fullmatch(version):
        return f"{match.group(1)}.{match.group(2)}.{int(match.group(3)) + 1}"
    if match := _SUFFIX_VERSION.fullmatch(version):
        return f"{match.group(1)}{int(match.group(2)) + 1}"
    raise IsolationViolation("autonomous repair cannot deterministically bump component version")


def _version_overlay(base: Path, changed: tuple[str, ...], request_digest: str, date: str) -> dict | None:
    path = base / _VERSION_REGISTRY
    if not path.is_file():
        return None
    registry = json.loads(path.read_text(encoding="utf-8"))
    components = registry.get("components", {})
    touched = sorted({
        component for relative in changed
        if (component := _component_for_path(relative, components)) is not None
    })
    if not touched:
        return None
    for component in touched:
        row = components[component]
        version = _next_version(row.get("version"))
        row["version"] = version
        row["history"].insert(0, {
            "date": date,
            "note": (
                f"Controller-owned autonomous repair {request_digest[:16]} reconciled "
                f"the independently verified implementation change."
            ),
            "version": version,
        })
    return registry


def apply_version_overlay(
    base: Path, candidate: Path, changed: tuple[str, ...], request_digest: str,
) -> bool:
    """Bump watched components after worker teardown; workers cannot edit policy."""
    if _VERSION_REGISTRY in changed:
        raise IsolationViolation("worker changed the protected version registry")
    date = datetime.now(UTC).date().isoformat()
    registry = _version_overlay(base, changed, request_digest, date)
    if registry is None:
        return False
    target = candidate / _VERSION_REGISTRY
    mode = stat.S_IMODE(target.stat().st_mode)
    descriptor, raw = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(registry, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def worker_changes_with_valid_overlay(
    base: Path, candidate: Path, changed: tuple[str, ...], request_digest: str,
) -> tuple[str, ...]:
    """Remove only the exact deterministic controller registry overlay."""
    if _VERSION_REGISTRY not in changed:
        return changed
    worker = tuple(path for path in changed if path != _VERSION_REGISTRY)
    try:
        base_registry = json.loads((base / _VERSION_REGISTRY).read_text(encoding="utf-8"))
        touched = sorted({
            component for path in worker
            if (component := _component_for_path(path, base_registry["components"])) is not None
        })
        if not touched:
            raise ValueError("overlay without a watched repair path")
        actual = json.loads((candidate / _VERSION_REGISTRY).read_text(encoding="utf-8"))
        date = actual["components"][touched[0]]["history"][0]["date"]
        datetime.fromisoformat(date)
    except (OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise IsolationViolation("invalid controller version registry overlay") from exc
    expected = _version_overlay(base, worker, request_digest, date)
    if expected is None or actual != expected:
        raise IsolationViolation("invalid controller version registry overlay")
    return worker


def _authorized_changes(base, candidate, changed, request_digest):
    return (
        worker_changes_with_valid_overlay(base, candidate, changed, request_digest)
        if request_digest is not None
        else changed
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
    equivalence_paths: tuple[str, ...] = (), *, request_digest: str | None = None,
) -> None:
    """Only exact granted paths; existing tests cannot be edited or deleted."""
    changed = _authorized_changes(base, candidate, changed, request_digest)
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
