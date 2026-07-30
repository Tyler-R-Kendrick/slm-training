"""Strict external test-case resources with opt-in expectation refresh."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = REPO_ROOT / "src/slm_training/resources/test_cases"
SCHEMA = "test_cases/v1"
REFRESH_ENV = "SLM_REFRESH_TEST_CASES"


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be an object")
    return value


def resource_path_for_test(test_file: str | Path) -> Path:
    test_path = Path(test_file).resolve()
    try:
        relative = test_path.relative_to(REPO_ROOT / "tests")
    except ValueError as exc:
        raise ValueError(f"test file is outside tests/: {test_path}") from exc
    return CASE_ROOT / relative.with_suffix(".json")


def test_path_for_resource(resource: str | Path) -> Path:
    path = Path(resource).resolve()
    try:
        relative = path.relative_to(CASE_ROOT)
    except ValueError as exc:
        raise ValueError(f"case resource is outside {CASE_ROOT}: {path}") from exc
    return REPO_ROOT / "tests" / relative.with_suffix(".py")


def load_case_document(path: str | Path) -> dict[str, Any]:
    resource = Path(path)
    value = _read_json(resource)
    if value.get("schema") != SCHEMA:
        raise ValueError(f"{resource}: schema must be {SCHEMA!r}")
    module = value.get("module")
    if not isinstance(module, str) or not module.startswith("tests/"):
        raise ValueError(f"{resource}: module must be a tests/ path")
    expected_module = test_path_for_resource(resource).relative_to(REPO_ROOT).as_posix()
    if module != expected_module:
        raise ValueError(
            f"{resource}: module {module!r} does not match {expected_module!r}"
        )
    case_sets = value.get("case_sets")
    if not isinstance(case_sets, dict) or not case_sets:
        raise ValueError(f"{resource}: case_sets must be a non-empty object")
    for set_name, case_set in case_sets.items():
        if not isinstance(set_name, str) or not set_name:
            raise ValueError(f"{resource}: case-set names must be non-empty strings")
        if not isinstance(case_set, dict):
            raise ValueError(f"{resource}: case set {set_name!r} must be an object")
        parameters = case_set.get("parameters")
        if parameters is not None and (
            not isinstance(parameters, list)
            or not parameters
            or any(not isinstance(name, str) or not name for name in parameters)
            or len(parameters) != len(set(parameters))
        ):
            raise ValueError(
                f"{resource}: case set {set_name!r} parameters must be unique strings"
            )
        cases = case_set.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"{resource}: case set {set_name!r} must have cases")
        ids: set[str] = set()
        for row in cases:
            if not isinstance(row, dict):
                raise ValueError(f"{resource}: {set_name!r} cases must be objects")
            case_id = row.get("id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"{resource}: every case needs a non-empty id")
            if case_id in ids:
                raise ValueError(
                    f"{resource}: duplicate case id {case_id!r} in {set_name!r}"
                )
            ids.add(case_id)
            if ("values" in row) == ("input" in row):
                raise ValueError(
                    f"{resource}: {set_name!r}/{case_id} needs exactly one of "
                    "values or input"
                )
            if "values" in row and not isinstance(row["values"], list):
                raise ValueError(
                    f"{resource}: {set_name!r}/{case_id} values must be a list"
                )
            if "values" in row and (
                parameters is None or len(row["values"]) != len(parameters)
            ):
                raise ValueError(
                    f"{resource}: {set_name!r}/{case_id} values must match parameters"
                )
            if "input" in row and parameters is not None:
                raise ValueError(
                    f"{resource}: {set_name!r}/{case_id} snapshots cannot declare "
                    "parameters"
                )
            if "input" in row and "expected" not in row:
                raise ValueError(
                    f"{resource}: {set_name!r}/{case_id} snapshot needs expected"
                )
            invariant = row.get("invariant")
            if invariant is not None and not isinstance(invariant, dict):
                raise ValueError(
                    f"{resource}: {set_name!r}/{case_id} invariant must be an object"
                )
    return value


def _case_set(test_file: str | Path, set_name: str) -> tuple[Path, dict[str, Any]]:
    path = resource_path_for_test(test_file)
    document = load_case_document(path)
    try:
        return path, document["case_sets"][set_name]
    except KeyError as exc:
        raise ValueError(f"{path}: missing case set {set_name!r}") from exc


def case_values(test_file: str | Path, set_name: str) -> list[Any]:
    """Return pytest parameters while keeping the test signature unchanged."""
    import pytest

    _, case_set = _case_set(test_file, set_name)
    return [pytest.param(*row["values"], id=row["id"]) for row in case_set["cases"]]


def _contains_subset(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains_subset(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_contains_subset(got, want) for got, want in zip(actual, expected))
        )
    return actual == expected


def _json_safe(value: Any) -> Any:
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=False)
    decoded = json.loads(encoded)
    if isinstance(decoded, float) and not math.isfinite(decoded):
        raise ValueError("expected value must be finite JSON")
    return decoded


def _write_expected(path: Path, set_name: str, case_id: str, expected: Any) -> None:
    document = load_case_document(path)
    rows = document["case_sets"][set_name]["cases"]
    row = next((item for item in rows if item["id"] == case_id), None)
    if row is None:
        raise ValueError(f"{path}: missing {set_name!r}/{case_id!r}")
    row["expected"] = _json_safe(expected)
    text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


@dataclass
class SnapshotCase:
    id: str
    input: Any
    expected: Any
    invariant: dict[str, Any]
    _path: Path
    _set_name: str

    def expect(self, actual: Any) -> None:
        """Check immutable invariants, then compare or refresh the full snapshot."""
        value = _json_safe(actual)
        assert _contains_subset(value, self.invariant), (
            f"{self.id}: invariant drifted\n"
            f"required subset={self.invariant!r}\nactual={value!r}"
        )
        if os.getenv(REFRESH_ENV) == "1":
            _write_expected(self._path, self._set_name, self.id, value)
            self.expected = value
            return
        assert value == self.expected, f"{self.id}: expected snapshot drifted"


def snapshot_cases(test_file: str | Path, set_name: str) -> list[SnapshotCase]:
    path, case_set = _case_set(test_file, set_name)
    result: list[SnapshotCase] = []
    for row in case_set["cases"]:
        if "input" not in row:
            raise ValueError(f"{path}: {set_name!r}/{row['id']} is not a snapshot")
        invariant = row.get("invariant") or {}
        if not invariant:
            raise ValueError(
                f"{path}: refreshable snapshot {set_name!r}/{row['id']} "
                "needs a non-empty invariant"
            )
        result.append(
            SnapshotCase(
                id=row["id"],
                input=row["input"],
                expected=row["expected"],
                invariant=invariant,
                _path=path,
                _set_name=set_name,
            )
        )
    return result


def all_case_files() -> list[Path]:
    return sorted(CASE_ROOT.rglob("*.json")) if CASE_ROOT.is_dir() else []
