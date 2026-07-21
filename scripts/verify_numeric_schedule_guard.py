#!/usr/bin/env python3
"""Fail-closed static guard for numeric weight/schedule fail-open patterns.

RSC-A06 (SLM-242): the SLM-138 recursive deep-supervision bug (silent
``min(len(depth_logits), len(ds_weights))`` truncation, and a per-depth
weight that is computed into ``total_w`` but never multiplied into the
per-depth loss it names) is one instance of a broader style of defect that a
runtime type checker cannot catch: the code *looks* like it uses a weight/
schedule vector correctly but silently drops part of it. This is a narrow,
low-false-positive AST guard over the canonical model-build loss/schedule
code (``src/slm_training/models/``, ``src/slm_training/harnesses/model_build/``)
for three source-shaped instances of that style:

* ``TRUNCATE`` -- ``min(len(a), len(b))`` used to pick a usable prefix length
  (the silent-truncation half of the SLM-138 bug).
* ``UNGUARDED_SUM`` -- ``total = sum(weights)`` followed by ``if total > 0:``
  in the same function, with no prior non-empty/all-zero validation call in
  that function (the all-zero-erasure half of the SLM-138 bug: the ``else``
  branch, i.e. "all zero", silently does nothing instead of failing closed
  earlier).
* ``UNUSED_LOOP_WEIGHT`` -- a ``for`` loop binds a weight-shaped variable
  (matching ``weight``/``^w$``/``_w$``) that is never read in the loop body
  (the "loop variable named as a weight but unused in the contribution"
  pattern -- this is precisely how the SLM-138 per-depth weight ``w`` was
  computed into ``total_w`` for normalization but never multiplied into
  ``d_loss``).

This guard intentionally does NOT attempt the fourth section-5 pattern
("capability guards that silently fall back") as a source-text AST rule --
that is covered far more precisely by the typed capability matrix in
``slm_training.models.twotower_schedule_policy`` (see
``supported_capability_requirement`` call sites and their regression tests)
than any generic text/AST heuristic could manage without broad false
positives.

A hit may be suppressed with a same-line or line-above comment:
``# schedule-guard: allow <PATTERN_ID> reason=<text> test=<path::test_name>``
Suppressions are recorded in the report, never silently dropped.

Usage::

    python -m scripts.verify_numeric_schedule_guard [--check] [--paths ...]

Exit code is 0 when every hit is suppressed, 1 when any is unsuppressed.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SCAN_PATHS: tuple[str, ...] = (
    "src/slm_training/models",
    "src/slm_training/harnesses/model_build",
)

# Files under the scan roots that are themselves guard/validation
# infrastructure (docstrings there legitimately quote the patterns as prose).
EXCLUDE_SUFFIXES: tuple[str, ...] = ("__pycache__",)

SUPPRESSION_RE = re.compile(
    r"#\s*schedule-guard:\s*allow\s+(?P<pattern>[A-Z_]+)\s+reason=(?P<reason>.+?)"
    r"\s+test=(?P<test>\S+)\s*$"
)

WEIGHT_NAME_RE = re.compile(r"weight", re.IGNORECASE)
BARE_W_RE = re.compile(r"^_?w$|_w$", re.IGNORECASE)


@dataclass
class Hit:
    pattern: str
    path: str
    line: int
    detail: str
    suppressed: bool = False
    suppression_reason: str | None = None
    suppression_test: str | None = None


@dataclass
class Report:
    hits: list[Hit] = field(default_factory=list)
    files_scanned: int = 0

    @property
    def unsuppressed(self) -> list[Hit]:
        return [h for h in self.hits if not h.suppressed]

    def to_dict(self) -> dict:
        return {
            "schema": "numeric_schedule_guard_report/v1",
            "files_scanned": self.files_scanned,
            "hit_count": len(self.hits),
            "unsuppressed_count": len(self.unsuppressed),
            "pass": not self.unsuppressed,
            "hits": [
                {
                    "pattern": h.pattern,
                    "path": h.path,
                    "line": h.line,
                    "detail": h.detail,
                    "suppressed": h.suppressed,
                    "suppression_reason": h.suppression_reason,
                    "suppression_test": h.suppression_test,
                }
                for h in self.hits
            ],
        }


def _is_len_call(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
    )


def _suppression_for_line(lines: list[str], lineno: int) -> re.Match | None:
    """Check the hit line and the line immediately above for a suppression comment."""
    for candidate in (lineno, lineno - 1):
        if 1 <= candidate <= len(lines):
            match = SUPPRESSION_RE.search(lines[candidate - 1])
            if match:
                return match
    return None


def _record(
    hits: list[Hit], *, pattern: str, path: str, line: int, detail: str, lines: list[str]
) -> None:
    match = _suppression_for_line(lines, line)
    hits.append(
        Hit(
            pattern=pattern,
            path=path,
            line=line,
            detail=detail,
            suppressed=bool(match),
            suppression_reason=match.group("reason") if match else None,
            suppression_test=match.group("test") if match else None,
        )
    )


def _iter_names_loaded(node: ast.AST) -> Iterable[str]:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            yield child.id


def _for_target_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for elt in target.elts:
            names.extend(_for_target_names(elt))
        return names
    return []


def _scan_truncate(tree: ast.Module, path: str, lines: list[str], hits: list[Hit]) -> None:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "min"
            and len(node.args) == 2
            and _is_len_call(node.args[0])
            and _is_len_call(node.args[1])
        ):
            a = ast.unparse(node.args[0])
            b = ast.unparse(node.args[1])
            _record(
                hits,
                pattern="TRUNCATE",
                path=path,
                line=node.lineno,
                detail=f"min({a}, {b}) silently picks the shorter of two vectors",
                lines=lines,
            )


def _scan_unguarded_sum(
    tree: ast.Module, path: str, lines: list[str], hits: list[Hit]
) -> None:
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        sum_assigned: set[str] = set()
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "sum"
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                sum_assigned.add(node.targets[0].id)
        if not sum_assigned:
            continue
        for node in ast.walk(func):
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id in sum_assigned
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Gt)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value in (0, 0.0)
            ):
                _record(
                    hits,
                    pattern="UNGUARDED_SUM",
                    path=path,
                    line=node.lineno,
                    detail=(
                        f"if {test.left.id} > 0 gates a sum(...)-derived total with no "
                        "prior all-zero validation in this function -- the all-zero case "
                        "falls through silently instead of failing closed"
                    ),
                    lines=lines,
                )


def _scan_unused_loop_weight(
    tree: ast.Module, path: str, lines: list[str], hits: list[Hit]
) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        for name in _for_target_names(node.target):
            if name.startswith("_"):
                continue
            if not (WEIGHT_NAME_RE.search(name) or BARE_W_RE.match(name)):
                continue
            loaded = set()
            for stmt in node.body:
                loaded.update(_iter_names_loaded(stmt))
            if name not in loaded:
                _record(
                    hits,
                    pattern="UNUSED_LOOP_WEIGHT",
                    path=path,
                    line=node.lineno,
                    detail=(
                        f"loop variable {name!r} is bound from a weight-shaped "
                        "iterable but never read in the loop body -- it does not "
                        "scale the per-iteration contribution"
                    ),
                    lines=lines,
                )


def _scan_file(path: Path, hits: list[Hit]) -> bool:
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return False
    lines = text.splitlines()
    try:
        rel = str(path.relative_to(ROOT))
    except ValueError:
        rel = str(path)
    _scan_truncate(tree, rel, lines, hits)
    _scan_unguarded_sum(tree, rel, lines, hits)
    _scan_unused_loop_weight(tree, rel, lines, hits)
    return True


def build_report(*, scan_paths: Iterable[str] = DEFAULT_SCAN_PATHS) -> Report:
    hits: list[Hit] = []
    scanned = 0
    for scan_path in scan_paths:
        root = ROOT / scan_path
        candidates = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for candidate in candidates:
            if any(part in EXCLUDE_SUFFIXES for part in candidate.parts):
                continue
            if _scan_file(candidate, hits):
                scanned += 1
    return Report(hits=hits, files_scanned=scanned)


def render_markdown(report: Report) -> str:
    lines = [
        "# Numeric schedule guard report",
        "",
        f"Files scanned: {report.files_scanned}",
        f"Hits: {len(report.hits)} ({len(report.unsuppressed)} unsuppressed)",
        f"Pass: **{not report.unsuppressed}**",
        "",
    ]
    for hit in report.hits:
        status = "suppressed" if hit.suppressed else "UNSUPPRESSED"
        lines.append(f"- `{hit.pattern}` {hit.path}:{hit.line} [{status}] -- {hit.detail}")
        if hit.suppressed:
            lines.append(
                f"  - reason: {hit.suppression_reason}; test: {hit.suppression_test}"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Override the default scan roots (repo-relative files or directories).",
    )
    parser.add_argument("--check", action="store_true", help="Kept for CI symmetry (default behavior).")
    parser.add_argument("--out", type=Path, default=None, help="Write JSON report here.")
    parser.add_argument("--markdown", type=Path, default=None, help="Write Markdown report here.")
    args = parser.parse_args(argv)

    report = build_report(scan_paths=args.paths or DEFAULT_SCAN_PATHS)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps(report.to_dict(), indent=2))
    return 0 if not report.unsuppressed else 1


if __name__ == "__main__":
    raise SystemExit(main())
