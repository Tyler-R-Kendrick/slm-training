"""Move large JSON-shaped pytest parameter tables into external case files."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.casefiles import (
    REPO_ROOT,
    SCHEMA,
    load_case_document,
    resource_path_for_test,
)

MIN_INLINE_CHARS = 100


@dataclass(frozen=True)
class Candidate:
    function: str
    start: int
    end: int
    parameters: list[str]
    values: list[Any]


def _domain_shaped(value: Any) -> bool:
    if isinstance(value, (str, dict)):
        return True
    return isinstance(value, (list, tuple)) and any(
        _domain_shaped(item) for item in value
    )


def _parameters(value: Any) -> list[str] | None:
    if isinstance(value, str):
        result = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    ):
        result = list(value)
    else:
        return None
    return result or None


def _offsets(text: str) -> list[int]:
    result = [0]
    for line in text.splitlines(keepends=True):
        result.append(result[-1] + len(line))
    return result


def candidates(path: Path) -> list[Candidate]:
    text = path.read_text(encoding="utf-8")
    offsets = _offsets(text)
    result: list[Candidate] = []
    for node in ast.walk(ast.parse(text, filename=str(path))):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if (
                not isinstance(decorator, ast.Call)
                or len(decorator.args) < 2
                or not isinstance(decorator.func, ast.Attribute)
                or decorator.func.attr != "parametrize"
            ):
                continue
            try:
                parameters = _parameters(ast.literal_eval(decorator.args[0]))
            except (TypeError, ValueError, SyntaxError):
                continue
            if not parameters:
                continue
            segment = ast.get_source_segment(text, decorator.args[1]) or ""
            try:
                values = ast.literal_eval(decorator.args[1])
                json.dumps(values, allow_nan=False)
            except (TypeError, ValueError, SyntaxError):
                continue
            if (
                len(segment) < MIN_INLINE_CHARS
                or not isinstance(values, (list, tuple))
                or not _domain_shaped(values)
            ):
                continue
            result.append(
                Candidate(
                    function=node.name,
                    start=offsets[decorator.args[1].lineno - 1]
                    + decorator.args[1].col_offset,
                    end=offsets[decorator.args[1].end_lineno - 1]
                    + decorator.args[1].end_col_offset,
                    parameters=parameters,
                    values=list(values),
                )
            )
    return sorted(result, key=lambda item: item.start)


def _rows(candidate: Candidate) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(candidate.values, 1):
        if len(candidate.parameters) == 1:
            values = [value]
        elif isinstance(value, (list, tuple)) and len(value) == len(
            candidate.parameters
        ):
            values = list(value)
        else:
            raise ValueError(
                f"{candidate.function}: case {index} does not match "
                f"{len(candidate.parameters)} parameters"
            )
        rows.append({"id": f"case-{index:03d}", "values": values})
    return rows


def extract(path: Path, selected: list[Candidate]) -> None:
    resource = resource_path_for_test(path)
    if resource.exists():
        document = load_case_document(resource)
    else:
        document = {
            "schema": SCHEMA,
            "module": path.relative_to(REPO_ROOT).as_posix(),
            "case_sets": {},
        }
    for candidate in selected:
        if candidate.function in document["case_sets"]:
            raise ValueError(f"{resource}: duplicate case set {candidate.function!r}")
        document["case_sets"][candidate.function] = {
            "parameters": candidate.parameters,
            "cases": _rows(candidate),
        }
    resource.parent.mkdir(parents=True, exist_ok=True)
    resource.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    text = path.read_text(encoding="utf-8")
    for candidate in reversed(selected):
        replacement = f'case_values(__file__, "{candidate.function}")'
        text = text[: candidate.start] + replacement + text[candidate.end :]
    if "from tests.casefiles import case_values" not in text:
        marker = "import pytest\n"
        if marker not in text:
            raise ValueError(f"{path}: cannot find pytest import")
        text = text.replace(
            marker, marker + "\nfrom tests.casefiles import case_values\n", 1
        )
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="perform the extraction")
    parser.add_argument("paths", nargs="*", help="test files or directories")
    args = parser.parse_args(argv)
    roots = (
        [REPO_ROOT / path for path in args.paths]
        if args.paths
        else [REPO_ROOT / "tests"]
    )
    files = sorted(
        {
            file
            for root in roots
            for file in ([root] if root.is_file() else root.rglob("*.py"))
            if file.name != "casefiles.py"
        }
    )
    found = [(path, candidates(path)) for path in files]
    found = [(path, rows) for path, rows in found if rows]
    if args.write:
        for path, rows in found:
            extract(path, rows)
    for path, rows in found:
        names = ", ".join(row.function for row in rows)
        print(f"{path.relative_to(REPO_ROOT)}: {names}")
    return 0 if args.write or not found else 1


if __name__ == "__main__":
    raise SystemExit(main())
