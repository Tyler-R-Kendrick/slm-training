"""DSH1-07 (SLM-359): PartialStateClassV1 contract — pure classification logic.

Torch-free; no tokenizer/grammar wiring. The OpenUI adapter that supplies real
``CompletionForest``/``validate_output`` facts is covered separately in
``tests/test_dsl/test_openui_partial_state.py``.
"""

from __future__ import annotations

import pytest

from slm_training.dsl.solver.partial_state import (
    PartialKind,
    PartialStateClassV1,
    classify_partial_state,
)


def test_document_kind_yields_complete_with_no_frontier() -> None:
    result = classify_partial_state(
        completed_kind="document",
        forest_has_paths=False,
        forest_coverage="none",
    )
    assert result.kind is PartialKind.COMPLETE
    assert result.start_symbol is None
    assert result.frontier == ()
    assert result.min_completion_cost == 0


def test_non_document_kind_yields_named_fragment() -> None:
    result = classify_partial_state(
        completed_kind="statement",
        forest_has_paths=False,
        forest_coverage="none",
        open_nonterminals=("NAME",),
        required_roles=("hero",),
    )
    assert result.kind is PartialKind.FRAGMENT
    assert result.start_symbol == "statement"
    assert result.frontier == ()
    assert result.open_nonterminals == ("NAME",)
    assert result.required_roles == ("hero",)
    assert result.min_completion_cost == 0


@pytest.mark.parametrize("coverage", ["complete", "partial"])
def test_forest_paths_under_certified_coverage_yield_prefix(coverage: str) -> None:
    result = classify_partial_state(
        completed_kind=None,
        forest_has_paths=True,
        forest_coverage=coverage,
        frontier=("EQUAL", "NAME"),
        singleton_positions=(0, 2, 1, 2),
        min_completion_cost=3,
    )
    assert result.kind is PartialKind.PREFIX
    assert result.frontier == ("EQUAL", "NAME")
    # Deduplicated + sorted, per the recorded-fact contract.
    assert result.singleton_positions == (0, 1, 2)
    assert result.support_coverage == coverage
    assert result.min_completion_cost == 3


def test_no_paths_yields_invalid() -> None:
    result = classify_partial_state(
        completed_kind=None, forest_has_paths=False, forest_coverage="none"
    )
    assert result.kind is PartialKind.INVALID
    assert result.start_symbol is None
    assert result.frontier == ()


def test_none_coverage_with_paths_is_invalid_not_prefix() -> None:
    # A "none"-coverage forest never certifies a continuation, regardless of
    # forest_has_paths (which should not happen together in a real forest,
    # but the contract must not silently promote it to PREFIX either way).
    result = classify_partial_state(
        completed_kind=None,
        forest_has_paths=True,
        forest_coverage="none",
        frontier=("NAME",),
        min_completion_cost=1,
    )
    assert result.kind is PartialKind.INVALID
    assert result.frontier == ()


def test_prefix_requires_min_completion_cost() -> None:
    with pytest.raises(ValueError):
        classify_partial_state(
            completed_kind=None,
            forest_has_paths=True,
            forest_coverage="complete",
            frontier=("NAME",),
            min_completion_cost=None,
        )


def test_prefix_requires_nonempty_frontier() -> None:
    with pytest.raises(ValueError):
        classify_partial_state(
            completed_kind=None,
            forest_has_paths=True,
            forest_coverage="complete",
            min_completion_cost=0,
        )


def test_cut_inside_lexical_boundary_is_refused() -> None:
    with pytest.raises(ValueError, match="splits a lexical token"):
        classify_partial_state(
            completed_kind=None,
            forest_has_paths=True,
            forest_coverage="complete",
            frontier=("NAME",),
            min_completion_cost=0,
            cut_is_lexical_boundary=False,
        )


def test_invalid_forest_coverage_value_rejected() -> None:
    with pytest.raises(ValueError):
        classify_partial_state(
            completed_kind=None, forest_has_paths=False, forest_coverage="bogus"
        )


class TestPartialStateClassV1Validation:
    def test_complete_rejects_start_symbol(self) -> None:
        with pytest.raises(ValueError):
            PartialStateClassV1(kind=PartialKind.COMPLETE, start_symbol="document")

    def test_complete_rejects_nonempty_frontier(self) -> None:
        with pytest.raises(ValueError):
            PartialStateClassV1(kind=PartialKind.COMPLETE, frontier=("NAME",))

    def test_fragment_requires_start_symbol(self) -> None:
        with pytest.raises(ValueError):
            PartialStateClassV1(kind=PartialKind.FRAGMENT)

    def test_fragment_rejects_nonempty_frontier(self) -> None:
        with pytest.raises(ValueError):
            PartialStateClassV1(
                kind=PartialKind.FRAGMENT, start_symbol="expression", frontier=("NAME",)
            )

    def test_prefix_requires_frontier_and_cost(self) -> None:
        with pytest.raises(ValueError):
            PartialStateClassV1(
                kind=PartialKind.PREFIX, support_coverage="complete", min_completion_cost=0
            )

    def test_prefix_rejects_none_coverage(self) -> None:
        with pytest.raises(ValueError):
            PartialStateClassV1(
                kind=PartialKind.PREFIX,
                frontier=("NAME",),
                support_coverage="none",
                min_completion_cost=0,
            )

    def test_invalid_rejects_start_symbol_and_frontier(self) -> None:
        with pytest.raises(ValueError):
            PartialStateClassV1(kind=PartialKind.INVALID, start_symbol="statement")
        with pytest.raises(ValueError):
            PartialStateClassV1(kind=PartialKind.INVALID, frontier=("NAME",))

    def test_negative_singleton_position_rejected(self) -> None:
        with pytest.raises(ValueError):
            PartialStateClassV1(
                kind=PartialKind.PREFIX,
                frontier=("NAME",),
                support_coverage="complete",
                min_completion_cost=0,
                singleton_positions=(-1,),
            )

    def test_negative_min_completion_cost_rejected(self) -> None:
        with pytest.raises(ValueError):
            PartialStateClassV1(
                kind=PartialKind.PREFIX,
                frontier=("NAME",),
                support_coverage="complete",
                min_completion_cost=-1,
            )

    def test_to_dict_from_dict_round_trip(self) -> None:
        original = PartialStateClassV1(
            kind=PartialKind.PREFIX,
            frontier=("NAME", "EQUAL"),
            open_nonterminals=("NAME", "EQUAL"),
            required_roles=("hero",),
            singleton_positions=(0, 1, 2),
            support_coverage="complete",
            min_completion_cost=2,
        )
        restored = PartialStateClassV1.from_dict(original.to_dict())
        assert restored == original

    def test_from_dict_rejects_unknown_and_missing_fields(self) -> None:
        payload = PartialStateClassV1(kind=PartialKind.COMPLETE, min_completion_cost=0).to_dict()
        payload["bogus"] = True
        with pytest.raises(ValueError):
            PartialStateClassV1.from_dict(payload)
        del payload["bogus"]
        del payload["kind"]
        with pytest.raises(ValueError):
            PartialStateClassV1.from_dict(payload)
