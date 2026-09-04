"""Per-function complexity and single-responsibility signals, via ruff.

Ruff already implements mccabe cyclomatic complexity (``C901``) and the pylint
function-shape rules; reimplementing them here would create a second, silently
diverging definition of "too complex". So ruff stays the single source of truth
and this module only aggregates its JSON output into ratchet counts.

See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
from collections import Counter
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

#: Rules that measure how much responsibility one function has taken on.
#: The rules are selected here rather than in ``[tool.ruff.lint] select`` so
#: that bare ``ruff check .`` stays a fast correctness pass; their *thresholds*
#: live in ``[tool.ruff.lint.mccabe]`` / ``[tool.ruff.lint.pylint]`` in
#: pyproject, so this gate and a local ``ruff check --select ...`` agree.
RULES: dict[str, str] = {
    "C901": "cyclomatic complexity",
    "PLR0911": "return statements",
    "PLR0912": "branches",
    "PLR0913": "arguments",
    "PLR0915": "statements",
}

#: Roots that ruff is asked to scan. Mirrors the repo-owned source tree.
LINT_ROOTS: tuple[str, ...] = ("src", "scripts", "tests")


class RuffUnavailableError(RuntimeError):
    """Raised when ruff is not on PATH, so complexity cannot be measured."""


@dataclass(frozen=True)
class Violation:
    """One ruff finding attributed to a file and rule."""

    path: str
    rule: str
    function: str
    line: int


def _ruff_executable() -> str:
    found = shutil.which("ruff")
    if found is None:
        raise RuffUnavailableError(
            "ruff is not on PATH; install the dev extra (`pip install -e '.[dev]'`) "
            "or run the gate with --skip-complexity"
        )
    return found


def ruff_version() -> str:
    """The ruff build these counts came from.

    Recorded in the baseline because the counts are only comparable within one
    ruff version: pyproject allows ``ruff>=0.9,<0.16``, and a minor bump that
    adds or refines a rule shifts every number at once. Without this, that
    shows up as hundreds of unexplained regressions.
    """

    result = subprocess.run(
        [_ruff_executable(), "--version"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def run_ruff(*, root: Path, roots: tuple[str, ...] = LINT_ROOTS) -> list[Violation]:
    """Collect complexity findings for every rule in :data:`RULES`."""

    root = root.resolve()
    result = subprocess.run(
        [
            _ruff_executable(),
            "check",
            "--no-cache",
            "--force-exclude",
            "--output-format=json",
            f"--select={','.join(sorted(RULES))}",
            *roots,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    # Ruff exits 1 when it reports findings, which is the expected path here;
    # anything else (2+) means the invocation itself failed.
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"ruff failed ({result.returncode}): {result.stderr.strip()}"
        )

    findings = json.loads(result.stdout or "[]")
    violations: list[Violation] = []
    for finding in findings:
        code = finding.get("code")
        if code not in RULES:
            continue
        path = Path(finding["filename"])
        relative = path.relative_to(root) if path.is_absolute() else path
        violations.append(
            Violation(
                path=relative.as_posix(),
                rule=code,
                function=_function_of(finding, root=root),
                line=int(finding.get("location", {}).get("row", 0)),
            )
        )
    return violations


def _function_of(finding: dict, *, root: Path) -> str:
    """Name the function a finding sits in.

    ``C901`` quotes the name in its message (``` `_run` is too complex ```) but
    the pylint rules do not, so fall back to locating the innermost enclosing
    definition by line number.
    """

    message = str(finding.get("message", ""))
    if "`" in message:
        return message.split("`")[1]
    path = Path(finding["filename"])
    row = int(finding.get("location", {}).get("row", 0))
    return _enclosing_definition(root / path if not path.is_absolute() else path, row)


@lru_cache(maxsize=None)
def _definition_spans(path: Path) -> tuple[tuple[int, int, str], ...]:
    """Every ``def``/``class`` in a file as ``(start, end, name)``, innermost last."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return ()
    spans = [
        (node.lineno, node.end_lineno or node.lineno, node.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]
    # Widest first, so a later match is always the more deeply nested one.
    return tuple(sorted(spans, key=lambda span: span[1] - span[0], reverse=True))


def _enclosing_definition(path: Path, row: int) -> str:
    innermost = "<module>"
    for start, end, name in _definition_spans(path):
        if start <= row <= end:
            innermost = name
    return innermost


def counts_by_file(violations: list[Violation]) -> dict[str, int]:
    """Total complexity findings per file -- the ratchet dimension."""

    tally = Counter(item.path for item in violations)
    return dict(sorted(tally.items()))


def counts_by_rule(violations: list[Violation]) -> dict[str, int]:
    """Findings per rule, for the human-readable summary."""

    tally = Counter(item.rule for item in violations)
    return {rule: tally.get(rule, 0) for rule in sorted(RULES)}


def worst_functions(violations: list[Violation], *, limit: int = 10) -> list[str]:
    """The functions carrying the most distinct complexity findings."""

    tally = Counter((item.path, item.function) for item in violations)
    return [
        f"{count} findings  {path}::{function}"
        for (path, function), count in tally.most_common(limit)
    ]
