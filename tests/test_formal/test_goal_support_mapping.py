"""PGS-F01: Python ↔ Lean finite goal-support structural mapping tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.formal.goal_support_mapping import (
    GOAL_SUPPORT_FORMAL_SCHEMA,
    TerminalFailureSetV1,
    evaluate_structural_case,
    hits_all,
    subset_minimal_exact,
)
from slm_training.dsl.solver.goal_support import (
    GoalDomainActionPartitionsV1,
    classify_domain_adequacy,
)

_GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "slm_training"
    / "resources"
    / "formal"
    / "goal_support_golden_cases.v1.json"
)


def _golden_cases() -> list[dict]:
    payload = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    assert payload["schema"] == GOAL_SUPPORT_FORMAL_SCHEMA
    return list(payload["cases"])


@pytest.mark.parametrize("case", _golden_cases(), ids=lambda item: item["id"])
def test_golden_structural_mapping(case: dict) -> None:
    got = evaluate_structural_case(case["input"])
    for key, expected in case["expect"].items():
        assert got[key] == expected, (case["id"], key, got[key], expected)


def test_subset_minimal_core_properties() -> None:
    failures = (
        TerminalFailureSetV1(terminal=0, atoms=(10,)),
        TerminalFailureSetV1(terminal=1, atoms=(11,)),
    )
    core = (10, 11)
    assert hits_all(core, failures)
    assert subset_minimal_exact(core, failures)
    assert not subset_minimal_exact((10, 11, 12), failures) or hits_all(
        (10, 11, 12), failures
    )
    assert subset_minimal_exact((10, 11, 12), failures) is False


def test_unknown_unobserved_never_certified_removed() -> None:
    got = evaluate_structural_case(
        {
            "schema": GOAL_SUPPORT_FORMAL_SCHEMA,
            "partitions": {
                "legal": [1, 2, 3],
                "supported": [],
                "unsupported": [1],
                "unknown": [2],
                "unobserved": [3],
            },
            "evidence": [
                {"action": 1, "partition": "unsupported", "replay_ok": True, "hard_profile": True},
                {"action": 2, "partition": "unknown", "replay_ok": False, "hard_profile": True},
                {"action": 3, "partition": "unobserved", "replay_ok": False, "hard_profile": True},
            ],
            "inputs": {
                "selected": 2,
                "hard_profile": True,
                "cap_applied": False,
                "domain_complete": False,
                "all_unsupported_replay_valid": False,
                "obstruction_present": False,
            },
            "failures": [],
            "core": [],
            "required_constraints": [],
            "satisfied_constraints": [],
            "live": [],
            "removed": [],
        }
    )
    assert got["certified_removal"] == [1]
    assert 2 not in got["certified_removal"]
    assert 3 not in got["certified_removal"]


def test_formal_classifier_matches_production_without_requiring_a_core() -> None:
    partitions = GoalDomainActionPartitionsV1(
        supported=(), unsupported=("a",), unknown=(), unobserved=()
    )
    assert classify_domain_adequacy(
        partitions,
        "a",
        all_unsupported_replay_valid=True,
        hard_profile=True,
        cap_applied=False,
        domain_complete=True,
        obstruction_summary=None,
    ) == "domain_inadequate_under_bounds"
    assert classify_domain_adequacy(
        partitions,
        "a",
        all_unsupported_replay_valid=True,
        hard_profile=True,
        cap_applied=False,
        domain_complete=False,
        obstruction_summary=None,
    ) == "coverage_unknown"
