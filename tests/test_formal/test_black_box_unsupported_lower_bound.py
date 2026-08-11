"""KERN-04 / SLM-522: black-box UNSUPPORTED query lower-bound Lean↔Python parity."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.formal.black_box_unsupported_lower_bound import (
    BOUND_AST_ID,
    COST_MODEL_NAME,
    ESCAPE_NOTES,
    EscapeClass,
    black_box_unsupported_query_lower_bound,
    build_black_box_lower_bound,
    early_stop_distinguishes,
    escape_leaves_black_box,
    exhaustive_partial_query_unsound,
    export_four_axis_lower_bound_evidence,
    frozen_fixture_rows,
    model_from_domain_sizes,
    one_valid,
    oracles_agree_on,
    all_invalid,
    sound_unsupported_claim,
    sound_unsupported_requires_full_coverage,
    worst_case_query_lower_bound,
)

_FIXTURES = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "slm_training"
    / "resources"
    / "formal"
    / "black_box_unsupported_fixtures.v1.json"
)


def test_model_from_domain_sizes_matches_kern03_P() -> None:
    model = model_from_domain_sizes([2, 3])
    assert model.assignment_count == 6
    assert worst_case_query_lower_bound(model) == 6
    assert model_from_domain_sizes([]).assignment_count == 1


def test_early_stop_distinguisher() -> None:
    agree, not_sound = early_stop_distinguishes(3, 2, [0, 1])
    assert agree is True
    assert not_sound is True
    assert oracles_agree_on(all_invalid, one_valid(2), [0, 1])
    assert sound_unsupported_claim(3, all_invalid)
    assert not sound_unsupported_claim(3, one_valid(2))


def test_full_coverage_required() -> None:
    assert black_box_unsupported_query_lower_bound(3, [0, 1, 2])
    assert not black_box_unsupported_query_lower_bound(3, [0, 1])
    assert sound_unsupported_requires_full_coverage(3, [0, 1, 2]) is True
    assert sound_unsupported_requires_full_coverage(3, [0, 1]) is False


def test_exhaustive_small_P() -> None:
    for p in range(0, 5):
        result = exhaustive_partial_query_unsound(p)
        assert result["ok"] is True
        assert result["lower_bound"] == p


def test_escapes_leave_model() -> None:
    for escape in EscapeClass:
        assert escape_leaves_black_box(escape) is True
        assert ESCAPE_NOTES[escape]


def test_four_axis_export() -> None:
    evidence = build_black_box_lower_bound([2, 3])
    axes = export_four_axis_lower_bound_evidence(evidence)
    assert axes.resource_bounds.status == "proved_axis"
    assert axes.resource_bounds.bound_ast_id == BOUND_AST_ID
    assert "black_box_unsupported_query_lower_bound" in (
        axes.resource_bounds.theorem_ref or ""
    )
    assert axes.computability.status == "proved_axis"
    assert axes.implementation_refinement.status == "empirical_remainder"
    payload = evidence.to_dict()
    assert payload["cost_model"]["name"] == COST_MODEL_NAME
    assert payload["universal_solver_claim"] is False
    assert payload["worst_case_query_lower_bound"] == 6


def test_frozen_fixture_file() -> None:
    payload = json.loads(_FIXTURES.read_text(encoding="utf-8"))
    assert payload["schema"] == "black_box_unsupported_fixtures/v1"
    assert payload["cost_model"] == COST_MODEL_NAME
    rows = {row["id"]: row for row in payload["cases"]}
    live = {row["id"]: row for row in frozen_fixture_rows()}
    for fid, expected in rows.items():
        got = live[fid]
        assert got["P"] == expected["P"]
        assert got["lower_bound"] == expected["lower_bound"]
        if expected.get("exhaustive_ok") is not None:
            assert got["exhaustive_ok"] is True
