"""KERN-05 / SLM-523: strict closure progress + finite stabilization parity tests."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.formal.bound_ast import (
    BOUND_CLOSURE_LIVE_UPPER,
    BOUND_CLOSURE_STRICT_REMOVALS,
    assert_registered_bound_ast_id,
    evaluate_bound,
)
from slm_training.formal.closure_stabilization import (
    BOUND_AST_STRICT_REMOVALS,
    QueryResultV1,
    assignment_search_bound,
    batch_covers_removable,
    build_stabilization_evidence,
    close_pass_results,
    closure_complexity_separated,
    cross_check_trace,
    frozen_closure_traces,
    frozen_fixture_rows,
    is_fixed_point,
    mutate_drop_removal_bookkeeping,
    mutate_duplicate_live,
    nonempty_invariant_upper_bound,
    removed_count,
    strict_progress_or_fixed_point,
    strict_removal_upper_bound,
    total_removed_across,
    unique_live,
)

_FIXTURES = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "slm_training"
    / "resources"
    / "formal"
    / "closure_stabilization_fixtures.v1.json"
)


def test_lattice_bounds_match_lean_fixtures() -> None:
    assert strict_removal_upper_bound([2, 3]) == 5
    assert nonempty_invariant_upper_bound([2, 3]) == 3
    assert assignment_search_bound([2, 3]) == 6
    assert closure_complexity_separated([1, 1, 1])
    assert strict_removal_upper_bound([1, 1, 1]) == 3
    assert assignment_search_bound([1, 1, 1]) == 1


def test_strict_progress_or_fixed_point() -> None:
    live = [0, 1, 2]
    empty: list[QueryResultV1] = []
    assert is_fixed_point(live, empty)
    assert strict_progress_or_fixed_point(live, empty)
    rem = [QueryResultV1(0, 1, "unsupported", True)]
    assert not is_fixed_point(live, rem)
    assert strict_progress_or_fixed_point(live, rem)
    assert removed_count(live, rem) == 1
    assert 1 not in close_pass_results(live, rem)


def test_total_removed_across_le_live() -> None:
    live = [0, 1, 2, 3]
    batches = [
        [QueryResultV1(0, 1, "unsupported", True)],
        [QueryResultV1(0, 3, "unsupported", True)],
    ]
    assert total_removed_across(live, batches) == 2
    assert total_removed_across(live, batches) <= len(live)


def test_batch_coverage_holds_for_encoding() -> None:
    live = [0, 1]
    results = [QueryResultV1(0, 1, "unsupported", True)]
    assert batch_covers_removable(live, results)


def test_bound_ast_strict_removals_registered() -> None:
    assert BOUND_AST_STRICT_REMOVALS == BOUND_CLOSURE_STRICT_REMOVALS
    assert_registered_bound_ast_id(BOUND_CLOSURE_STRICT_REMOVALS)
    assert_registered_bound_ast_id(BOUND_CLOSURE_LIVE_UPPER)
    result = evaluate_bound(BOUND_CLOSURE_STRICT_REMOVALS, {"domain_sizes": [2, 3]})
    assert result.value == "5"
    live = evaluate_bound(BOUND_CLOSURE_LIVE_UPPER, {"live_count": 7})
    assert live.value == "7"


def test_frozen_fixture_file_parity() -> None:
    payload = json.loads(_FIXTURES.read_text(encoding="utf-8"))
    assert payload["schema"] == "closure_stabilization_fixtures/v1"
    live_rows = {row["id"]: row for row in frozen_fixture_rows()}
    for expected in payload["cases"]:
        got = live_rows[expected["id"]]
        for key, value in expected.items():
            if key == "id":
                continue
            assert got[key] == value, (expected["id"], key, got[key], value)


def test_cross_check_all_frozen_traces() -> None:
    for trace in frozen_closure_traces():
        check = cross_check_trace(trace)
        assert check["within_strict_bound"] is True
        assert check["batch_coverage"] is True
        assert check["strict_progress_or_fixed_point"] is True


def test_mutation_duplicate_live_breaks_uniqueness() -> None:
    live = [0, 1, 2]
    assert unique_live(live)
    mutated = mutate_duplicate_live(live)
    assert not unique_live(mutated)
    assert len(mutated) != len(set(mutated))


def test_mutation_nonfixed_pass_no_progress_detected() -> None:
    live = [0, 1, 2]
    results = [QueryResultV1(0, 1, "unsupported", True)]
    assert not is_fixed_point(live, results)
    fake = mutate_drop_removal_bookkeeping(live, results)
    assert fake == live
    assert close_pass_results(live, results) != fake
    assert removed_count(live, results) >= 1


def test_evidence_refuses_maximality_claim() -> None:
    evidence = build_stabilization_evidence(
        domain_sizes=[2, 3],
        total_removed=2,
        unique=True,
        coverage=True,
        nonempty_invariant=True,
    )
    payload = evidence.to_dict()
    assert payload["maximality_claimed"] is False
    assert payload["wall_clock_claim"] is False
    assert "no_maximality_without_proof" in payload["preconditions"]
