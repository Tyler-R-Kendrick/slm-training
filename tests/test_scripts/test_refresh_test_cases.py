from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.refresh_test_cases import _gate_weakenings, _review, _without_expected
from tests import casefiles


def _document(*, cases: list[dict] | None = None) -> dict:
    return {
        "schema": "test_cases/v1",
        "module": "tests/test_example.py",
        "case_sets": {
            "examples": {
                "cases": cases
                or [
                    {
                        "id": "one",
                        "input": {"value": 1},
                        "expected": {"value": 2, "kind": "result"},
                        "invariant": {"kind": "result"},
                    }
                ]
            }
        },
    }


def _case_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path
    case_root = root / "src/slm_training/resources/test_cases"
    test_path = root / "tests/test_example.py"
    test_path.parent.mkdir(parents=True)
    test_path.write_text("", encoding="utf-8")
    resource = case_root / "test_example.json"
    resource.parent.mkdir(parents=True)
    resource.write_text(json.dumps(_document(), indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(casefiles, "REPO_ROOT", root)
    monkeypatch.setattr(casefiles, "CASE_ROOT", case_root)
    return resource


def test_snapshot_refresh_changes_only_expected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resource = _case_file(tmp_path, monkeypatch)
    before = casefiles.load_case_document(resource)
    case = casefiles.snapshot_cases(tmp_path / "tests/test_example.py", "examples")[0]
    monkeypatch.setenv(casefiles.REFRESH_ENV, "1")

    case.expect({"value": 3, "kind": "result"})

    after = casefiles.load_case_document(resource)
    assert after["case_sets"]["examples"]["cases"][0]["expected"]["value"] == 3
    assert _without_expected(before) == _without_expected(after)


def test_snapshot_refresh_never_weakens_invariant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _case_file(tmp_path, monkeypatch)
    case = casefiles.snapshot_cases(tmp_path / "tests/test_example.py", "examples")[0]
    monkeypatch.setenv(casefiles.REFRESH_ENV, "1")

    with pytest.raises(AssertionError, match="invariant drifted"):
        case.expect({"value": 3, "kind": "different"})


def test_case_document_rejects_duplicate_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resource = _case_file(tmp_path, monkeypatch)
    value = _document()
    value["case_sets"]["examples"]["cases"].append(
        value["case_sets"]["examples"]["cases"][0]
    )
    resource.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case id"):
        casefiles.load_case_document(resource)


def test_case_document_rejects_parameter_arity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resource = _case_file(tmp_path, monkeypatch)
    value = _document(cases=[{"id": "one", "values": [1], "expected": {"value": 2}}])
    value["case_sets"]["examples"]["parameters"] = ["left", "right"]
    resource.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="values must match parameters"):
        casefiles.load_case_document(resource)


def test_review_blocks_case_removal(tmp_path: Path) -> None:
    before = _document()
    after = _document(
        cases=[
            {
                "id": "two",
                "values": ["replacement"],
            }
        ]
    )

    with pytest.raises(ValueError, match="removal or rename"):
        _review(
            tmp_path / "cases.json",
            before,
            after,
            before_text=json.dumps(before),
            after_text=json.dumps(after),
        )


def test_gate_review_blocks_only_weakening() -> None:
    before = {
        "default_min_suite_n": 20,
        "suites": {"smoke": {"quality": 0.5}},
    }
    assert (
        _gate_weakenings(
            before,
            {
                "default_min_suite_n": 20,
                "suites": {"smoke": {"quality": 0.6}},
            },
        )
        == []
    )
    assert _gate_weakenings(
        before,
        {
            "default_min_suite_n": 19,
            "suites": {"smoke": {"quality": 0.4}},
        },
    ) == [
        "default_min_suite_n decreased",
        "lowered threshold smoke.quality",
    ]
