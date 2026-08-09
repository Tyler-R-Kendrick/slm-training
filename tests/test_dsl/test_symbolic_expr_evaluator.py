"""SRP-003 (SLM-461) regression tests: vectorized NumPy evaluator over the
typed symbolic-expression IR, with explicit numerical-domain policies."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from slm_training.dsl.symbolic_expr_evaluator import (
    DEFAULT_DOMAIN_POLICY,
    NUMERIC_DOMAIN_POLICIES,
    DomainViolationV1,
    EvaluationResultV1,
    evaluate_vectorized,
)
from slm_training.dsl.symbolic_expr_ir import (
    BINARY_OPERATORS,
    UNARY_OPERATORS,
    CoefficientHole,
    OpApply,
    VarRef,
)


def _v(index: int) -> VarRef:
    return VarRef(index=index)


def _c(index: int) -> CoefficientHole:
    return CoefficientHole(index=index)


def _op(operator: str, *args) -> OpApply:
    return OpApply(operator=operator, args=tuple(args))


# ---------------------------------------------------------------------------
# Basic arithmetic and unary operators
# ---------------------------------------------------------------------------


def test_addition_subtraction_multiplication_vectorize_over_rows():
    node = _op("-", _op("*", _v(0), _v(1)), _op("+", _v(0), _v(1)))
    variables = np.array([[2.0, 3.0], [5.0, -1.0], [0.0, 0.0]])
    result = evaluate_vectorized(node, variables=variables)
    assert result.predictions == pytest.approx((2.0 * 3.0 - (2.0 + 3.0), 5.0 * -1.0 - (5.0 - 1.0), 0.0))
    assert result.valid_mask == (True, True, True)
    assert result.domain_violations == ()
    assert result.failure_class == "none"


@pytest.mark.parametrize("operator", sorted(UNARY_OPERATORS - {"log", "sqrt"}))
def test_unary_operators_match_math_reference(operator):
    reference = {"neg": lambda x: -x, "sin": math.sin, "cos": math.cos, "exp": math.exp, "abs": abs}[operator]
    node = _op(operator, _v(0))
    variables = np.array([[0.37], [-1.5], [2.0]])
    result = evaluate_vectorized(node, variables=variables)
    expected = tuple(reference(row[0]) for row in variables)
    assert result.predictions == pytest.approx(expected)
    assert result.valid_mask == (True, True, True)


def test_coefficient_hole_binds_external_value_not_carried_by_the_ast():
    node = _op("+", _v(0), _c(0))
    variables = np.array([[1.0], [2.0]])
    result = evaluate_vectorized(node, variables=variables, coefficients=np.array([10.0]))
    assert result.predictions == pytest.approx((11.0, 12.0))


# ---------------------------------------------------------------------------
# Hard domain violations -- always invalid, regardless of domain_policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain_policy", sorted(NUMERIC_DOMAIN_POLICIES))
def test_division_near_zero_is_always_a_hard_domain_violation(domain_policy):
    node = _op("/", _v(0), _v(1))
    variables = np.array([[1.0, 0.0], [4.0, 2.0]])
    result = evaluate_vectorized(node, variables=variables, domain_policy=domain_policy)
    assert result.valid_mask == (False, True)
    assert math.isnan(result.predictions[0])
    assert result.predictions[1] == pytest.approx(2.0)
    assert result.domain_violations == (
        DomainViolationV1(row_index=0, operator="/", reason="division_near_zero"),
    )
    assert result.failure_class == "partial"


@pytest.mark.parametrize("domain_policy", sorted(NUMERIC_DOMAIN_POLICIES))
def test_log_nonpositive_is_always_a_hard_domain_violation(domain_policy):
    node = _op("log", _v(0))
    variables = np.array([[-1.0], [0.0], [math.e]])
    result = evaluate_vectorized(node, variables=variables, domain_policy=domain_policy)
    assert result.valid_mask == (False, False, True)
    assert math.isnan(result.predictions[0])
    assert math.isnan(result.predictions[1])
    assert result.predictions[2] == pytest.approx(1.0)
    reasons = {v.reason for v in result.domain_violations}
    assert reasons == {"log_nonpositive_domain"}


@pytest.mark.parametrize("domain_policy", sorted(NUMERIC_DOMAIN_POLICIES))
def test_sqrt_negative_is_always_a_hard_domain_violation(domain_policy):
    node = _op("sqrt", _v(0))
    variables = np.array([[-4.0], [9.0]])
    result = evaluate_vectorized(node, variables=variables, domain_policy=domain_policy)
    assert result.valid_mask == (False, True)
    assert math.isnan(result.predictions[0])
    assert result.predictions[1] == pytest.approx(3.0)
    assert result.domain_violations[0].reason == "sqrt_negative_domain"


# ---------------------------------------------------------------------------
# Soft non-finite results -- governed by domain_policy
# ---------------------------------------------------------------------------


def test_exp_overflow_invalidates_row_under_default_reject_policy():
    node = _op("exp", _v(0))
    variables = np.array([[1000.0], [0.0]])
    result = evaluate_vectorized(node, variables=variables)
    assert result.domain_policy == DEFAULT_DOMAIN_POLICY == "reject_nonfinite"
    assert result.valid_mask == (False, True)
    assert math.isinf(result.predictions[0])
    assert result.domain_violations == (
        DomainViolationV1(row_index=0, operator="__result__", reason="nonfinite_result"),
    )
    assert result.failure_class == "partial"


def test_exp_overflow_stays_valid_under_allow_nonfinite_but_is_never_hidden():
    node = _op("exp", _v(0))
    variables = np.array([[1000.0], [0.0]])
    result = evaluate_vectorized(node, variables=variables, domain_policy="allow_nonfinite")
    assert result.valid_mask == (True, True)
    # The row is "valid" (caller opted in), but the value is still the raw
    # non-finite result -- never silently coerced into a finite "good" score.
    assert math.isinf(result.predictions[0])
    assert result.domain_violations == (
        DomainViolationV1(row_index=0, operator="__result__", reason="nonfinite_result"),
    )


def test_unsupported_domain_policy_raises():
    node = _v(0)
    with pytest.raises(ValueError, match="unsupported numeric_domain_policy"):
        evaluate_vectorized(node, variables=np.array([[1.0]]), domain_policy="ignore_everything")


# ---------------------------------------------------------------------------
# Structural fail-closed behavior
# ---------------------------------------------------------------------------


def test_variable_index_out_of_range_raises():
    node = _v(5)
    with pytest.raises(ValueError, match="out of range"):
        evaluate_vectorized(node, variables=np.array([[1.0, 2.0]]))


def test_missing_coefficient_binding_raises():
    node = _c(0)
    with pytest.raises(ValueError, match="no coefficients were bound"):
        evaluate_vectorized(node, variables=np.array([[1.0]]))


def test_coefficient_array_too_short_raises():
    node = _op("+", _c(0), _c(1))
    with pytest.raises(ValueError, match="missing coefficient binding"):
        evaluate_vectorized(node, variables=np.array([[1.0]]), coefficients=np.array([1.0]))


def test_variables_must_be_2d():
    node = _v(0)
    with pytest.raises(ValueError, match="2-D array"):
        evaluate_vectorized(node, variables=np.array([1.0, 2.0]))


def test_variables_must_be_numeric_dtype():
    node = _v(0)
    with pytest.raises(TypeError, match="numeric dtype"):
        evaluate_vectorized(node, variables=np.array([["not-a-number"]]))


def test_coefficients_must_be_1d():
    node = _c(0)
    with pytest.raises(ValueError, match="1-D array"):
        evaluate_vectorized(node, variables=np.array([[1.0]]), coefficients=np.array([[1.0]]))


def test_coefficients_must_be_numeric_dtype():
    node = _c(0)
    with pytest.raises(TypeError, match="numeric dtype"):
        evaluate_vectorized(node, variables=np.array([[1.0]]), coefficients=np.array(["x"]))


def test_expression_over_budget_raises():
    node = _v(0)
    for _ in range(40):  # exceeds DEFAULT_MAX_DEPTH=32
        node = _op("neg", node)
    with pytest.raises(ValueError, match="max_depth"):
        evaluate_vectorized(node, variables=np.array([[1.0]]))


def test_hand_assembled_node_with_unsupported_operator_fails_closed():
    """Bypassing OpApply.__post_init__'s own arity/name validation (e.g. via
    object.__new__) must still fail closed at evaluation time -- no
    eval/getattr dispatch exists for it to silently succeed through."""
    node = object.__new__(OpApply)
    object.__setattr__(node, "operator", "__import__")
    object.__setattr__(node, "args", (_v(0),))
    with pytest.raises(ValueError, match="unsupported operator"):
        evaluate_vectorized(node, variables=np.array([[1.0]]))


def test_unsupported_node_type_fails_closed():
    with pytest.raises(TypeError, match="unsupported symbolic expression node type"):
        evaluate_vectorized("not-a-node", variables=np.array([[1.0]]))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Determinism and differential correctness
# ---------------------------------------------------------------------------


def test_same_ast_and_data_produce_bit_identical_predictions():
    node = _op("/", _op("sin", _v(0)), _op("+", _v(1), _c(0)))
    variables = np.array([[0.5, 1.5], [2.0, -0.5]])
    coefficients = np.array([3.0])
    first = evaluate_vectorized(node, variables=variables, coefficients=coefficients)
    second = evaluate_vectorized(node, variables=variables, coefficients=coefficients)
    assert first == second


def test_differential_against_python_math_reference():
    node = _op("-", _op("*", _op("sqrt", _op("abs", _v(0))), _c(0)), _op("cos", _v(1)))
    rng = random.Random(2024)
    rows = [(rng.uniform(-5, 5), rng.uniform(-5, 5)) for _ in range(20)]
    variables = np.array(rows)
    coefficients = np.array([2.5])
    result = evaluate_vectorized(node, variables=variables, coefficients=coefficients)
    expected = tuple(math.sqrt(abs(x0)) * 2.5 - math.cos(x1) for x0, x1 in rows)
    assert result.predictions == pytest.approx(expected)
    assert result.valid_mask == tuple(True for _ in rows)
    assert result.failure_class == "none"


def test_random_trees_round_trip_through_evaluation_deterministically():
    rng = random.Random(99)
    leaves = [lambda: _v(rng.randrange(2)), lambda: _c(rng.randrange(2))]

    def random_tree(depth: int):
        if depth <= 0 or rng.random() < 0.35:
            return rng.choice(leaves)()
        operator = rng.choice(sorted(BINARY_OPERATORS | UNARY_OPERATORS))
        arity = 2 if operator in BINARY_OPERATORS else 1
        return _op(operator, *(random_tree(depth - 1) for _ in range(arity)))

    variables = np.array([[1.3, -0.7], [0.2, 4.0]])
    coefficients = np.array([1.0, -2.0])
    for _ in range(25):
        node = random_tree(4)
        first = evaluate_vectorized(node, variables=variables, coefficients=coefficients)
        second = evaluate_vectorized(node, variables=variables, coefficients=coefficients)
        assert np.array_equal(first.predictions, second.predictions, equal_nan=True)
        assert first.valid_mask == second.valid_mask
        assert first.domain_violations == second.domain_violations
        assert first.failure_class == second.failure_class
        assert len(first.predictions) == 2
        assert first.failure_class in ("none", "partial", "total")


# ---------------------------------------------------------------------------
# Typed evidence contract
# ---------------------------------------------------------------------------


def test_failure_class_total_when_every_row_invalid():
    node = _op("log", _v(0))
    variables = np.array([[-1.0], [-2.0]])
    result = evaluate_vectorized(node, variables=variables)
    assert result.failure_class == "total"
    assert result.valid_mask == (False, False)


def test_evaluation_result_round_trips_to_dict():
    node = _op("+", _v(0), _v(0))
    result = evaluate_vectorized(node, variables=np.array([[1.0]]))
    payload = result.to_dict()
    assert payload["predictions"] == [2.0]
    assert payload["valid_mask"] == [True]
    assert payload["domain_violations"] == []
    assert payload["failure_class"] == "none"
    assert payload["domain_policy"] == "reject_nonfinite"


def test_domain_violation_rejects_unknown_reason():
    with pytest.raises(ValueError, match="unsupported domain violation reason"):
        DomainViolationV1(row_index=0, operator="/", reason="made_up_reason")


def test_evaluation_result_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        EvaluationResultV1(
            predictions=(1.0, 2.0),
            valid_mask=(True,),
            domain_violations=(),
            failure_class="none",
            domain_policy="reject_nonfinite",
        )
