"""SRP-002 (SLM-453): typed symbolic-expression IR, codec, and budget tests.

Extends test_symbolic_regression_pack.py (SRP-001's problem contract and
untyped dict-based reference evaluator) with coverage for the new typed
AST/serializer/parser this issue adds -- not a duplicate of those tests.
"""

from __future__ import annotations

import random

import pytest

from tests.casefiles import case_values

from slm_training.dsl.symbolic_expr_ir import (
    BINARY_OPERATORS,
    UNARY_OPERATORS,
    CoefficientHole,
    OpApply,
    VarRef,
    collect_symbols,
    expr_depth_and_node_count,
    parse_expr,
    parse_problem_expr,
    serialize_expr,
    validate_expr_budget,
)
from slm_training.dsl.symbolic_regression_pack import (
    ALLOWED_OPERATORS,
    SymbolicRegressionProblemV1,
)


def _problem(**overrides: object) -> SymbolicRegressionProblemV1:
    fields = dict(
        variables=("x", "y"),
        table=((1.0, 2.0), (3.0, 4.0)),
        targets=(3.0, 7.0),
        allowed_operators=frozenset({"+", "*"}),
        coefficient_hole_policy="fixed_constants_only",
        complexity_budget=10,
        numeric_domain_policy="reject_nonfinite",
        train_indices=(0, 1),
        validation_indices=(),
    )
    fields.update(overrides)
    return SymbolicRegressionProblemV1(**fields)


# ---------------------------------------------------------------------------
# Deterministic round trip
# ---------------------------------------------------------------------------


def test_leaf_nodes_round_trip() -> None:
    assert serialize_expr(VarRef(0)) == "v0"
    assert parse_expr("v0", num_variables=1) == VarRef(0)
    assert serialize_expr(CoefficientHole(3)) == "c3"
    assert parse_expr("c3", num_variables=1) == CoefficientHole(3)


def test_operator_application_round_trips() -> None:
    node = OpApply("+", (VarRef(0), OpApply("neg", (CoefficientHole(0),))))
    text = serialize_expr(node)
    assert text == "(+ v0 (neg c0))"
    assert parse_expr(text, num_variables=1) == node


def _random_tree(rng: random.Random, *, num_variables: int, depth: int) -> object:
    if depth <= 0 or rng.random() < 0.4:
        if rng.random() < 0.5:
            return VarRef(rng.randrange(num_variables))
        return CoefficientHole(rng.randrange(4))
    op = rng.choice(sorted(ALLOWED_OPERATORS))
    arity = 2 if op in BINARY_OPERATORS else 1
    args = tuple(
        _random_tree(rng, num_variables=num_variables, depth=depth - 1)
        for _ in range(arity)
    )
    return OpApply(op, args)


def test_property_random_trees_round_trip_deterministically() -> None:
    rng = random.Random(1234)
    for _ in range(200):
        tree = _random_tree(rng, num_variables=3, depth=5)
        text = serialize_expr(tree)
        assert parse_expr(text, num_variables=3, max_depth=64, max_nodes=1024) == tree
        # Serializing twice is byte-identical -- deterministic, not just equal.
        assert serialize_expr(tree) == text


# ---------------------------------------------------------------------------
# Operator arity and unknown-operator rejection
# ---------------------------------------------------------------------------


def test_op_apply_rejects_wrong_arity() -> None:
    with pytest.raises(ValueError, match="requires 2 argument"):
        OpApply("+", (VarRef(0),))
    with pytest.raises(ValueError, match="requires 1 argument"):
        OpApply("sin", (VarRef(0), VarRef(1)))


def test_op_apply_rejects_unknown_operator() -> None:
    with pytest.raises(ValueError, match="unknown operator"):
        OpApply("matmul", (VarRef(0), VarRef(1)))


def test_all_allowed_operators_have_a_declared_arity() -> None:
    for op in BINARY_OPERATORS | UNARY_OPERATORS:
        assert op in ALLOWED_OPERATORS
    assert (BINARY_OPERATORS | UNARY_OPERATORS) == ALLOWED_OPERATORS


# ---------------------------------------------------------------------------
# Operator-policy: only the active problem's operators enter legal support
# ---------------------------------------------------------------------------


def test_parse_rejects_an_operator_outside_the_declared_allowlist() -> None:
    # "-" is in the pack's global ALLOWED_OPERATORS but not this narrower set.
    with pytest.raises(ValueError, match="not authorized for this problem"):
        parse_expr("(- v0 v1)", num_variables=2, allowed_operators=frozenset({"+"}))


def test_parse_problem_expr_uses_the_problem_contracts_own_allowlist() -> None:
    problem = _problem(allowed_operators=frozenset({"+"}))
    with pytest.raises(ValueError, match="not authorized for this problem"):
        parse_problem_expr(problem, "(* v0 v1)")
    node = parse_problem_expr(problem, "(+ v0 v1)")
    assert node == OpApply("+", (VarRef(0), VarRef(1)))


def test_parse_problem_expr_derives_variable_count_from_the_problem() -> None:
    problem = _problem(
        variables=("x", "y", "z"),
        table=((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
        allowed_operators=frozenset({"+"}),
    )
    with pytest.raises(ValueError, match="out of range"):
        parse_problem_expr(problem, "v3")
    assert parse_problem_expr(problem, "v2") == VarRef(2)


# ---------------------------------------------------------------------------
# Alpha-renaming safety: variable identity is a pure index, never a name
# ---------------------------------------------------------------------------


def test_structural_codec_is_invariant_to_variable_renaming() -> None:
    renamed = _problem(variables=("alpha", "beta"), allowed_operators=frozenset({"+"}))
    original = _problem(variables=("x", "y"), allowed_operators=frozenset({"+"}))
    text = "(+ v0 v1)"
    # The parsed structure and its re-serialization are identical regardless
    # of the underlying variable names -- names never enter the IR.
    assert parse_problem_expr(original, text) == parse_problem_expr(renamed, text)
    assert serialize_expr(parse_problem_expr(original, text)) == serialize_expr(
        parse_problem_expr(renamed, text)
    )


# ---------------------------------------------------------------------------
# Structural codec cannot smuggle free-form numbers/strings/code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "malformed",
    case_values(__file__, "test_parser_fails_closed_on_malformed_or_smuggled_input"),
)
def test_parser_fails_closed_on_malformed_or_smuggled_input(malformed: str) -> None:
    with pytest.raises(ValueError):
        parse_expr(malformed, num_variables=2)


def test_variable_index_out_of_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="out of range"):
        parse_expr("v5", num_variables=2)


def test_trailing_tokens_after_a_complete_expression_are_rejected() -> None:
    with pytest.raises(ValueError, match="trailing tokens"):
        parse_expr("v0 v1", num_variables=2)


# ---------------------------------------------------------------------------
# Depth/node budgets fail closed
# ---------------------------------------------------------------------------


def _deep_chain(depth: int) -> str:
    text = "v0"
    for _ in range(depth):
        text = f"(neg {text})"
    return text


def test_parser_rejects_expressions_exceeding_max_depth() -> None:
    with pytest.raises(ValueError, match="max_depth"):
        parse_expr(_deep_chain(50), num_variables=1, max_depth=10)
    # Within budget succeeds.
    parse_expr(_deep_chain(5), num_variables=1, max_depth=10)


def test_parser_rejects_expressions_exceeding_max_nodes() -> None:
    wide = "(+ " + " ".join("v0" for _ in range(2)) + ")"
    tree = OpApply("+", (VarRef(0), VarRef(0)))
    assert parse_expr(wide, num_variables=1, max_nodes=3) == tree
    with pytest.raises(ValueError, match="max_nodes"):
        parse_expr(_deep_chain(20), num_variables=1, max_nodes=5)


def test_validate_expr_budget_checks_directly_constructed_trees() -> None:
    tree = OpApply("neg", (OpApply("neg", (VarRef(0),)),))
    validate_expr_budget(tree, max_depth=10, max_nodes=10)
    with pytest.raises(ValueError, match="max_depth"):
        validate_expr_budget(tree, max_depth=1, max_nodes=10)
    with pytest.raises(ValueError, match="max_nodes"):
        validate_expr_budget(tree, max_depth=10, max_nodes=1)


def test_expr_depth_and_node_count_matches_manual_construction() -> None:
    leaf = VarRef(0)
    assert expr_depth_and_node_count(leaf) == (1, 1)
    tree = OpApply("+", (VarRef(0), OpApply("neg", (CoefficientHole(0),))))
    depth, count = expr_depth_and_node_count(tree)
    assert depth == 3  # +  -> neg -> c0
    assert count == 4  # +, v0, neg, c0


# ---------------------------------------------------------------------------
# Symbol table
# ---------------------------------------------------------------------------


def test_collect_symbols_returns_the_referenced_variable_and_coefficient_sets() -> None:
    tree = OpApply(
        "+",
        (
            OpApply("*", (VarRef(0), CoefficientHole(0))),
            OpApply("*", (VarRef(1), CoefficientHole(0))),
        ),
    )
    variables, coefficients = collect_symbols(tree)
    assert variables == frozenset({0, 1})
    assert coefficients == frozenset({0})  # reused hole is not double-counted
