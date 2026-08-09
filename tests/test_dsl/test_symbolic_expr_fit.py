"""SRP-004 (SLM-465) regression tests: affine coefficient-dependence analysis
and deterministic least-squares fitting over the typed symbolic-expression IR."""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

from slm_training.dsl.symbolic_expr_fit import (
    LeastSquaresFitV1,
    NonAffineError,
    classify_affine_dependence,
    fit_integrity_ok,
    fit_least_squares,
)
from slm_training.dsl.symbolic_expr_ir import BINARY_OPERATORS, CoefficientHole, OpApply, VarRef


def _v(index: int) -> VarRef:
    return VarRef(index=index)


def _c(index: int) -> CoefficientHole:
    return CoefficientHole(index=index)


def _op(operator: str, *args) -> OpApply:
    return OpApply(operator=operator, args=tuple(args))


# ---------------------------------------------------------------------------
# Structural affine classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "node",
    [
        _c(0),
        _op("+", _op("*", _c(0), _v(0)), _c(1)),
        _op("*", _c(0), _op("*", _v(0), _v(1))),
        _op("/", _c(0), _v(0)),
        _op("-", _c(0), _op("*", _c(1), _v(0))),
        _op("neg", _op("*", _c(0), _v(0))),
        _op("+", _op("*", _c(0), _v(0)), _op("*", _c(0), _v(1))),  # reused coefficient, still affine
        _op("sin", _v(0)),  # transcendental of a coefficient-free argument is fine
    ],
)
def test_classify_affine_cases(node):
    result = classify_affine_dependence(node)
    assert result.status == "affine"
    assert result.schema_version == "v1"


@pytest.mark.parametrize(
    "node,expected_reason_fragment",
    [
        (_op("*", _c(0), _c(1)), "product of two coefficient-dependent"),
        (_op("sin", _c(0)), "coefficient hole inside 'sin'"),
        (_op("log", _op("*", _c(0), _v(0))), "coefficient hole inside 'log'"),
        (_op("/", _v(0), _c(0)), "division's denominator"),
        (_op("/", _c(0), _c(1)), "division's denominator"),
        (_op("abs", _c(0)), "coefficient hole inside 'abs'"),
    ],
)
def test_classify_non_affine_cases_with_reason(node, expected_reason_fragment):
    result = classify_affine_dependence(node)
    assert result.status == "non_affine"
    assert expected_reason_fragment in result.reason


def test_classify_reports_referenced_coefficient_indices_sorted():
    node = _op("+", _c(2), _c(0))
    result = classify_affine_dependence(node)
    assert result.coefficient_indices == (0, 2)


def test_classify_purely_variable_expression_is_trivially_affine_with_no_coefficients():
    node = _op("+", _v(0), _v(1))
    result = classify_affine_dependence(node)
    assert result.status == "affine"
    assert result.coefficient_indices == ()


def test_classify_unsupported_node_type_fails_closed():
    with pytest.raises(TypeError, match="unsupported symbolic expression node type"):
        classify_affine_dependence("not-a-node")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Anti-false-affine property test: whenever classify says "affine", the
# expression must actually behave affinely in its coefficients (verified via
# the finite-difference / vanishing-second-difference test at 3 coefficient
# scalings), over many random ASTs of arbitrary (not pre-constrained) shape.
# ---------------------------------------------------------------------------


def test_no_false_affine_over_random_trees():
    from slm_training.dsl.symbolic_expr_evaluator import evaluate_vectorized
    from slm_training.dsl.symbolic_expr_ir import UNARY_OPERATORS

    rng = random.Random(2026)

    def random_tree(depth: int):
        if depth <= 0 or rng.random() < 0.35:
            return rng.choice([lambda: _v(rng.randrange(2)), lambda: _c(rng.randrange(2))])()
        operator = rng.choice(sorted(BINARY_OPERATORS | UNARY_OPERATORS))
        arity = 2 if operator in BINARY_OPERATORS else 1
        return _op(operator, *(random_tree(depth - 1) for _ in range(arity)))

    variables = np.array([[1.3, -0.7], [0.4, 2.1], [-2.0, 0.5]])
    checked_affine = 0
    for _ in range(60):
        node = random_tree(4)
        result = classify_affine_dependence(node)
        if result.status != "affine" or not result.coefficient_indices:
            continue
        checked_affine += 1
        n_coeffs = max(result.coefficient_indices) + 1

        def predict(scale: float) -> np.ndarray:
            coefficients = np.full(n_coeffs, scale)
            return np.array(
                evaluate_vectorized(
                    node, variables=variables, coefficients=coefficients, domain_policy="allow_nonfinite"
                ).predictions
            )

        f0, f1, f2 = predict(0.0), predict(1.0), predict(2.0)
        # A true affine function of the coefficient scale has a vanishing
        # second difference: f(2s) - 2*f(s) + f(0) == 0.
        second_difference = f2 - 2 * f1 + f0
        finite = np.isfinite(second_difference)
        assert np.allclose(second_difference[finite], 0.0, atol=1e-6), (node, second_difference)
    assert checked_affine > 0  # sanity: the property test actually exercised affine cases


# ---------------------------------------------------------------------------
# Least-squares fit: closed-form example
# ---------------------------------------------------------------------------


def test_fit_recovers_known_affine_coefficients_exactly():
    node = _op("+", _op("*", _c(0), _v(0)), _c(1))  # c0*v0 + c1
    rng = random.Random(5)
    v0 = np.array([rng.uniform(-10, 10) for _ in range(10)])
    variables = v0.reshape(-1, 1)
    targets = 2.0 * v0 + 3.0
    fit = fit_least_squares(node, variables=variables, targets=targets, train_indices=np.arange(10))
    assert fit.coefficient_indices == (0, 1)
    assert fit.fitted_coefficients == pytest.approx((2.0, 3.0), abs=1e-8)
    assert fit.rank == 2
    assert not fit.is_rank_deficient
    assert fit.train_residual_l2 == pytest.approx(0.0, abs=1e-8)
    assert fit.solver_identity == "numpy.linalg.lstsq"


def test_fit_matches_direct_numpy_reference_solve():
    node = _op("+", _op("*", _c(0), _v(0)), _op("*", _c(1), _v(1)))
    rng = random.Random(9)
    rows = [(rng.uniform(-5, 5), rng.uniform(-5, 5)) for _ in range(12)]
    variables = np.array(rows)
    true_c = np.array([1.7, -0.4])
    targets = variables @ true_c + rng.gauss(0, 0.0)  # exact, no noise
    fit = fit_least_squares(node, variables=variables, targets=targets, train_indices=np.arange(12))

    reference_design = variables  # design matrix is exactly [v0, v1] here
    reference_coefficients, *_ = np.linalg.lstsq(reference_design, targets, rcond=None)
    assert fit.fitted_coefficients == pytest.approx(tuple(reference_coefficients), abs=1e-8)


def test_fit_raises_non_affine_error_and_does_not_fit():
    node = _op("sin", _c(0))
    variables = np.array([[1.0], [2.0]])
    with pytest.raises(NonAffineError):
        fit_least_squares(node, variables=variables, targets=np.array([1.0, 2.0]), train_indices=np.array([0, 1]))


def test_fit_raises_when_no_coefficient_holes_present():
    node = _op("+", _v(0), _v(1))
    variables = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="no coefficient hole"):
        fit_least_squares(node, variables=variables, targets=np.array([3.0]), train_indices=np.array([0]))


def test_fit_rejects_bad_shapes_and_out_of_range_train_indices():
    node = _op("*", _c(0), _v(0))
    variables = np.array([[1.0], [2.0]])
    with pytest.raises(ValueError, match="targets must be shape"):
        fit_least_squares(node, variables=variables, targets=np.array([1.0]), train_indices=np.array([0, 1]))
    with pytest.raises(ValueError, match="train_indices out of range"):
        fit_least_squares(
            node, variables=variables, targets=np.array([1.0, 2.0]), train_indices=np.array([0, 5])
        )
    with pytest.raises(ValueError, match="non-empty"):
        fit_least_squares(
            node, variables=variables, targets=np.array([1.0, 2.0]), train_indices=np.array([], dtype=int)
        )


def test_rank_deficient_design_is_flagged():
    # c0 and c1 both multiply the same variable -- their columns are identical.
    node = _op("+", _op("*", _c(0), _v(0)), _op("*", _c(1), _v(0)))
    variables = np.array([[1.0], [2.0], [3.0], [4.0]])
    targets = 5.0 * variables[:, 0]
    fit = fit_least_squares(node, variables=variables, targets=targets, train_indices=np.arange(4))
    assert fit.is_rank_deficient
    assert fit.rank == 1


# ---------------------------------------------------------------------------
# Train/holdout leakage
# ---------------------------------------------------------------------------


def test_holdout_rows_never_influence_the_fit():
    node = _op("+", _op("*", _c(0), _v(0)), _c(1))
    v0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    variables = v0.reshape(-1, 1)
    targets = 2.0 * v0 + 3.0
    train_indices = np.array([0, 1, 2, 3])

    fit_a = fit_least_squares(node, variables=variables, targets=targets, train_indices=train_indices)

    perturbed_targets = targets.copy()
    perturbed_targets[4:] += 1000.0  # corrupt only held-out rows (indices 4, 5)
    fit_b = fit_least_squares(node, variables=variables, targets=perturbed_targets, train_indices=train_indices)

    assert fit_a.fitted_coefficients == fit_b.fitted_coefficients
    assert fit_a.fit_digest == fit_b.fit_digest


# ---------------------------------------------------------------------------
# Determinism and tamper-evidence
# ---------------------------------------------------------------------------


def test_fit_is_deterministic():
    node = _op("*", _c(0), _v(0))
    variables = np.array([[1.0], [2.0], [3.0]])
    targets = np.array([2.0, 4.0, 6.0])
    fit_a = fit_least_squares(node, variables=variables, targets=targets, train_indices=np.arange(3))
    fit_b = fit_least_squares(node, variables=variables, targets=targets, train_indices=np.arange(3))
    assert fit_a == fit_b


def test_fit_round_trips_through_dict():
    node = _op("*", _c(0), _v(0))
    variables = np.array([[1.0], [2.0], [3.0]])
    targets = np.array([2.0, 4.0, 6.0])
    fit = fit_least_squares(node, variables=variables, targets=targets, train_indices=np.arange(3))
    restored = LeastSquaresFitV1.from_dict(fit.to_dict())
    assert restored == fit


def test_fit_rejects_unsupported_schema_version():
    node = _op("*", _c(0), _v(0))
    variables = np.array([[1.0], [2.0], [3.0]])
    targets = np.array([2.0, 4.0, 6.0])
    fit = fit_least_squares(node, variables=variables, targets=targets, train_indices=np.arange(3))
    payload = {**fit.to_dict(), "schema_version": "v99"}
    with pytest.raises(ValueError, match="unsupported least-squares fit schema"):
        LeastSquaresFitV1.from_dict(payload)


@pytest.mark.parametrize("field_name", ["solver_identity", "coefficient_indices", "fitted_coefficients"])
def test_fit_tamper_on_any_field_is_detected(field_name):
    node = _op("*", _c(0), _v(0))
    variables = np.array([[1.0], [2.0], [3.0]])
    targets = np.array([2.0, 4.0, 6.0])
    fit = fit_least_squares(node, variables=variables, targets=targets, train_indices=np.arange(3))
    if field_name == "solver_identity":
        tampered = replace(fit, solver_identity="scipy.optimize.curve_fit")
    elif field_name == "coefficient_indices":
        tampered = replace(fit, coefficient_indices=(99,))
    else:
        tampered = replace(fit, fitted_coefficients=(999.0,))
    assert not fit_integrity_ok(tampered)
    with pytest.raises(ValueError, match="digest mismatch"):
        LeastSquaresFitV1.from_dict(tampered.to_dict())
