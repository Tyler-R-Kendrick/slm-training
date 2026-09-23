"""Static modularity / duplication census for the Python surface.

Measures the four debt classes that
:doc:`docs/design/modularity-refactor-strategy` proposes to retire, so the
strategy stays evidence-backed instead of anecdotal:

``serialization``
    Hand-written ``to_dict`` / ``from_dict`` / ``to_json`` bodies that are
    mechanically derivable from the declaring dataclass' field annotations.
``reporting``
    Hand-assembled markdown emitters, counted across every naming convention
    in use (``render_markdown``, ``_build_markdown``, ``_markdown``, ...).
``lifecycle``
    Independent re-implementations of the preregistered-campaign protocol
    (``run_campaign`` / ``build_campaign_manifest`` / ``lock_campaign`` / ...).
``primitives``
    Copy-pasted micro-helpers. For the reproducibility-critical subset
    (canonical JSON, hashing, clocks) the audit also reports how many
    *behaviourally distinct* implementations exist, because a divergent
    ``_canonical_json`` silently forks every downstream ``version_stamp``.

The audit is static (``ast`` only): no torch, no imports of the audited
modules, no network, so it is safe to run in CI and cheap enough to run per
commit.

Usage::

    python -m scripts.audit_modularity                  # human-readable census
    python -m scripts.audit_modularity --json           # machine-readable
    python -m scripts.audit_modularity --check-divergence
    python -m scripts.audit_modularity --top 25 --paths src scripts tests
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = ("src", "scripts")
EXCLUDED_PARTS = frozenset({"node_modules", ".venv", ".git", "__pycache__"})

# Debt classes: label -> method names that constitute it.
SERIALIZATION = ("to_dict", "from_dict", "to_json", "from_json", "as_dict")
REPORTING = (
    "render_markdown",
    "_render_markdown",
    "_build_markdown",
    "_markdown",
    "_write_markdown",
    "_markdown_report",
    "render_evidence_markdown",
)
LIFECYCLE = (
    "run_campaign",
    "run_experiment",
    "run_fixture_campaign",
    "build_campaign_manifest",
    "lock_campaign",
    "write_campaign_lock",
    "load_campaign_lock",
    "write_evidence",
    "build_manifest",
    "validate_manifest",
    "plan_only_preview",
    "patch_preregistry",
)
PRIMITIVES = (
    "_now",
    "_utc_now",
    "_today_yyyymmdd",
    "_repo_root",
    "_sha",
    "_sha256",
    "_canonical_json",
    "_stable_json",
    "stable_hash",
    "fingerprint",
    "_config_sha",
    "_source_commit",
)
# Primitives whose divergence forks reproducibility rather than merely costing
# lines: every one of these feeds a fingerprint or a version_stamp.
REPRO_CRITICAL = frozenset(
    {
        "_canonical_json",
        "_stable_json",
        "stable_hash",
        "fingerprint",
        "_sha",
        "_sha256",
        "_now",
        "_utc_now",
        "_source_commit",
    }
)

DEBT_CLASSES: Mapping[str, Sequence[str]] = {
    "serialization": SERIALIZATION,
    "reporting": REPORTING,
    "lifecycle": LIFECYCLE,
    "primitives": PRIMITIVES,
}


@dataclass(frozen=True)
class Definition:
    """One function/method definition found in the tree."""

    name: str
    path: str
    lineno: int
    lines: int
    normalized: str
    mechanical: bool


@dataclass
class Census:
    definitions: list[Definition] = field(default_factory=list)
    dataclasses: int = 0
    annotated_fields: int = 0
    files_scanned: int = 0
    parse_failures: list[str] = field(default_factory=list)

    def by_name(self) -> dict[str, list[Definition]]:
        out: dict[str, list[Definition]] = defaultdict(list)
        for d in self.definitions:
            out[d.name].append(d)
        return out


def _iter_python_files(paths: Iterable[str]) -> Iterator[Path]:
    for rel in paths:
        base = REPO_ROOT / rel
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if EXCLUDED_PARTS.isdisjoint(path.parts):
                yield path


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _normalize(node: ast.AST) -> str:
    """Docstring-stripped, canonically-unparsed body: identical text == identical behaviour."""
    body = _strip_docstring(list(getattr(node, "body", [])))
    if not body:
        body = [ast.Pass()]
    try:
        return ast.unparse(ast.Module(body=body, type_ignores=[]))
    except Exception:  # pragma: no cover - unparse is total in practice
        return "<unparseable>"


def _is_mechanical(node: ast.FunctionDef) -> bool:
    """True when the body is a single ``return {...}`` over constant keys.

    Such a body carries no information beyond the field list, so a shared codec
    derives it from annotations with zero behaviour change.
    """
    body = _strip_docstring(list(node.body))
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    value = body[0].value
    if not isinstance(value, ast.Dict):
        return False
    return all(isinstance(key, ast.Constant) for key in value.keys if key is not None)


def collect(paths: Sequence[str]) -> Census:
    census = Census()
    watched = {n for names in DEBT_CLASSES.values() for n in names}
    for path in _iter_python_files(paths):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            census.parse_failures.append(f"{path.relative_to(REPO_ROOT)}: {exc}")
            continue
        census.files_scanned += 1
        rel = str(path.relative_to(REPO_ROOT))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                decorators = [ast.unparse(d) for d in node.decorator_list]
                if any("dataclass" in d for d in decorators):
                    census.dataclasses += 1
                    census.annotated_fields += sum(
                        1 for b in node.body if isinstance(b, ast.AnnAssign)
                    )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name not in watched:
                    continue
                end = node.end_lineno or node.lineno
                census.definitions.append(
                    Definition(
                        name=node.name,
                        path=rel,
                        lineno=node.lineno,
                        lines=end - node.lineno + 1,
                        normalized=_normalize(node),
                        mechanical=(
                            isinstance(node, ast.FunctionDef) and _is_mechanical(node)
                        ),
                    )
                )
    return census


def _class_rows(census: Census) -> dict[str, dict[str, Any]]:
    by_name = census.by_name()
    rows: dict[str, dict[str, Any]] = {}
    for label, names in DEBT_CLASSES.items():
        defs = [d for n in names for d in by_name.get(n, ())]
        variants = defaultdict(set)
        for d in defs:
            variants[d.name].add(d.normalized)
        rows[label] = {
            "definitions": len(defs),
            "lines": sum(d.lines for d in defs),
            "files": len({d.path for d in defs}),
            "mechanical_definitions": sum(1 for d in defs if d.mechanical),
            "mechanical_lines": sum(d.lines for d in defs if d.mechanical),
            "distinct_implementations": {
                k: len(v) for k, v in sorted(variants.items())
            },
        }
    return rows


def divergences(census: Census) -> list[dict[str, Any]]:
    """Reproducibility-critical helpers with more than one behaviour."""
    by_name = census.by_name()
    out: list[dict[str, Any]] = []
    for name in sorted(REPRO_CRITICAL):
        defs = by_name.get(name, [])
        if not defs:
            continue
        variants: dict[str, list[Definition]] = defaultdict(list)
        for d in defs:
            variants[d.normalized].append(d)
        if len(variants) <= 1:
            continue
        ordered = sorted(variants.items(), key=lambda kv: -len(kv[1]))
        out.append(
            {
                "helper": name,
                "copies": len(defs),
                "distinct_implementations": len(variants),
                "variants": [
                    {
                        "copies": len(group),
                        "body": body,
                        "example": group[0].path,
                    }
                    for body, group in ordered
                ],
            }
        )
    return out


def build_report(paths: Sequence[str], top: int) -> dict[str, Any]:
    census = collect(paths)
    by_name = census.by_name()
    ranked = sorted(
        (
            {
                "name": name,
                "definitions": len(defs),
                "lines": sum(d.lines for d in defs),
                "distinct_implementations": len({d.normalized for d in defs}),
            }
            for name, defs in by_name.items()
        ),
        key=lambda r: -r["lines"],
    )
    classes = _class_rows(census)
    return {
        "schema": "audit_modularity/v1",
        "paths": list(paths),
        "files_scanned": census.files_scanned,
        "parse_failures": census.parse_failures,
        "dataclasses": census.dataclasses,
        "annotated_fields": census.annotated_fields,
        "classes": classes,
        "total_duplicated_lines": sum(c["lines"] for c in classes.values()),
        "top_methods": ranked[:top],
        "divergences": divergences(census),
    }


def render(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Modularity audit")
    add("")
    add(
        f"Scanned {report['files_scanned']} files under "
        f"{', '.join(report['paths'])}: {report['dataclasses']} dataclasses "
        f"declaring {report['annotated_fields']} annotated fields."
    )
    add("")
    add(f"{'debt class':<16}{'defs':>7}{'lines':>9}{'files':>7}  note")
    add("-" * 68)
    for label, row in report["classes"].items():
        note = ""
        if label == "serialization" and row["definitions"]:
            pct = row["mechanical_definitions"] / row["definitions"] * 100
            note = (
                f"{pct:.0f}% mechanically derivable ({row['mechanical_lines']} lines)"
            )
        elif label == "primitives":
            forked = sum(1 for v in row["distinct_implementations"].values() if v > 1)
            note = f"{forked} helpers have >1 implementation"
        add(
            f"{label:<16}{row['definitions']:>7}{row['lines']:>9}"
            f"{row['files']:>7}  {note}"
        )
    add("-" * 68)
    add(f"{'TOTAL':<16}{'':>7}{report['total_duplicated_lines']:>9}")
    add("")
    add("## Largest duplicated methods")
    add("")
    add(f"{'method':<30}{'defs':>7}{'lines':>9}{'impls':>7}")
    add("-" * 53)
    for row in report["top_methods"]:
        add(
            f"{row['name']:<30}{row['definitions']:>7}"
            f"{row['lines']:>9}{row['distinct_implementations']:>7}"
        )
    div = report["divergences"]
    add("")
    add("## Reproducibility-critical divergence")
    add("")
    if not div:
        add("None: every fingerprint/clock helper has a single implementation.")
    else:
        add(
            "These helpers feed fingerprints and version stamps. More than one "
            "implementation means two harnesses can hash the same payload to "
            "different digests."
        )
        add("")
        for entry in div:
            add(
                f"- `{entry['helper']}`: {entry['copies']} copies, "
                f"**{entry['distinct_implementations']} distinct implementations** "
                f"(e.g. {entry['variants'][0]['example']})"
            )
    if report["parse_failures"]:
        add("")
        add("## Parse failures")
        add("")
        for failure in report["parse_failures"]:
            add(f"- {failure}")
    return "\n".join(lines)


REQUIRED_PYTHON = (3, 12)


def _interpreter_warning() -> str | None:
    """The audit parses source with the running ``ast``.

    Below the project's ``requires-python`` floor, 3.12-only syntax (PEP 695
    generics, PEP 701 f-strings) fails to parse and those files drop silently
    out of the census, under-reporting the debt.
    """
    if sys.version_info[:2] >= REQUIRED_PYTHON:
        return None
    running = ".".join(str(p) for p in sys.version_info[:3])
    required = ".".join(str(p) for p in REQUIRED_PYTHON)
    return (
        f"WARNING: running Python {running}; this project requires >={required}. "
        "Files using 3.12-only syntax will be reported as parse failures and "
        "excluded from the census. Re-run with the project interpreter for "
        "accurate totals."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--paths",
        nargs="+",
        default=list(DEFAULT_PATHS),
        help=f"repo-relative roots to scan (default: {' '.join(DEFAULT_PATHS)})",
    )
    parser.add_argument(
        "--top", type=int, default=20, help="how many duplicated methods to rank"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument(
        "--check-divergence",
        action="store_true",
        help="exit 1 when a reproducibility-critical helper has >1 implementation",
    )
    args = parser.parse_args(argv)

    warning = _interpreter_warning()
    if warning:
        print(warning, file=sys.stderr)

    report = build_report(args.paths, args.top)
    report["interpreter"] = ".".join(str(p) for p in sys.version_info[:3])
    report["interpreter_warning"] = warning
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render(report))

    if args.check_divergence and report["divergences"]:
        forked = ", ".join(d["helper"] for d in report["divergences"])
        print(
            f"\nFAIL: divergent reproducibility-critical helpers: {forked}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
