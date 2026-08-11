"""KERN-03 / SLM-521: finite completion-search bound Lean↔Python parity tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.formal.complete_domain import (
    build_complete_domain,
    from_legacy_coverage_flag,
)
from slm_training.formal.finite_search_bounds import (
    BOUND_AST_ID,
    COST_MODEL_NAME,
    UNIT_COST_MODEL,
    U64_MAX,
    SearchTraceV1,
    build_finite_search_bound,
    build_finite_search_bound_from_domains,
    coarse_node_bound,
    complete_assignment_count,
    cross_check_enumeration,
    export_four_axis_bound_evidence,
    frozen_fixture_rows,
    prefix_tree_le_coarse,
    prefix_tree_node_bound,
    proved_singleton_domains_fixture,
    pruning_memo_cannot_increase_bound,
    respects_upper_bounds,
    sizes_from_proved_domains,
)

_FIXTURES = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "slm_training"
    / "resources"
    / "formal"
    / "finite_search_bounds_fixtures.v1.json"
)


def test_assignment_and_prefix_formulas() -> None:
    assert complete_assignment_count([]) == 1
    assert prefix_tree_node_bound([]) == 1
    assert complete_assignment_count([2, 3]) == 6
    assert prefix_tree_node_bound([2, 3]) == 9
    assert coarse_node_bound([2, 3]) == 1 + 2 * 6
    assert prefix_tree_le_coarse([2, 3])


def test_zero_domain_and_empty_hole() -> None:
    assert complete_assignment_count([2, 0, 4]) == 0
    assert prefix_tree_node_bound([2, 0, 4]) == 3  # 1 + 2 + 0 + 0
    check = cross_check_enumeration([[0, 1], [], [0, 1, 2, 3]])
    assert check["assignments_match"] is True
    assert check["prefix_nodes_match"] is True
    assert check["predicted_assignments"] == 0


def test_overflow_u64_status() -> None:
    evidence = build_finite_search_bound([1 << 32, 1 << 32, 4])
    assert evidence.assignment_count == 1 << 66
    assert evidence.assignment_count > U64_MAX
    assert evidence.u64_overflow == "overflow_u64"
    small = build_finite_search_bound([2, 3])
    assert small.u64_overflow == "fits_u64"


def test_rejects_unproved_domains() -> None:
    forged = from_legacy_coverage_flag("s", [1, 2], coverage_complete=True)
    with pytest.raises(ValueError, match="proved-complete"):
        sizes_from_proved_domains([forged])


def test_proved_domain_sizes_feed_bounds() -> None:
    evidence = proved_singleton_domains_fixture()
    assert evidence.domain_sizes == (1, 1)
    assert evidence.assignment_count == 1
    assert evidence.prefix_tree_nodes == 3  # 1 + 1 + 1
    domains = (
        build_complete_domain("s1", [7], [7]),
        build_complete_domain("s1", [8, 9], [8, 9]),
    )
    bound = build_finite_search_bound_from_domains(domains)
    assert bound.domain_sizes == (1, 2)
    assert bound.assignment_count == 2


def test_pruning_memo_cannot_increase_bound() -> None:
    sizes = (2, 3)
    full = SearchTraceV1(expansions=9, verifications=6)
    pruned = SearchTraceV1(expansions=4, verifications=2)
    assert respects_upper_bounds(sizes, full)
    assert pruning_memo_cannot_increase_bound(sizes, full, pruned)
    increased = SearchTraceV1(expansions=10, verifications=6)
    assert not pruning_memo_cannot_increase_bound(sizes, full, increased)


def test_cost_model_is_named_query_model_not_wall_clock() -> None:
    evidence = build_finite_search_bound([2, 3], cost_model=UNIT_COST_MODEL)
    payload = evidence.to_dict()
    assert payload["cost_model"]["name"] == COST_MODEL_NAME
    assert payload["cost_model"]["wall_clock"] is False
    assert payload["wall_clock_claim"] is False
    assert evidence.total_query_bound == 9 + 6


def test_four_axis_export() -> None:
    evidence = build_finite_search_bound([2, 3])
    axes = export_four_axis_bound_evidence(evidence)
    assert axes.resource_bounds.status == "proved_axis"
    assert axes.resource_bounds.bound_ast_id == BOUND_AST_ID
    assert axes.resource_bounds.theorem_ref.endswith(
        "prefix_tree_le_one_plus_holes_mul_assignments"
    )
    assert axes.computability.status == "proved_axis"
    assert axes.implementation_refinement.status == "empirical_remainder"
    assert "wall-clock" in axes.resource_bounds.notes.lower() or (
        "wall-clock" in axes.implementation_refinement.notes.lower()
    )


def test_frozen_fixture_file_and_exhaustive_cross_check() -> None:
    payload = json.loads(_FIXTURES.read_text(encoding="utf-8"))
    assert payload["schema"] == "finite_search_bounds_fixtures/v1"
    assert payload["cost_model"] == COST_MODEL_NAME
    rows = {row["id"]: row for row in payload["cases"]}
    live = {row["id"]: row for row in frozen_fixture_rows()}
    for fid, expected in rows.items():
        got = live[fid]
        assert got["domain_sizes"] == expected["domain_sizes"]
        assert got["predicted_assignments"] == expected["predicted_assignments"]
        assert got["predicted_prefix_nodes"] == expected["predicted_prefix_nodes"]
        assert got["u64_overflow"] == expected["u64_overflow"]
        if expected.get("enumerate", True):
            assert got["assignments_match"] is True
            assert got["prefix_nodes_match"] is True
