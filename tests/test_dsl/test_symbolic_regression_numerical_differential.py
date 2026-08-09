"""SRP-006 (SLM-470): numerical differential tests across the symbolic-
regression pack's evaluator, fitting, and canonicalization modules.

Extends the existing per-module suites (``test_symbolic_expr_ir.py``,
``test_symbolic_expr_evaluator.py``, ``test_symbolic_expr_fit.py``,
``test_symbolic_expr_canonicalize.py``) rather than duplicating them: those
already cover per-operator correctness, one-off differential cases, and
hand-picked before/after examples. This file fills the specific gaps in
SLM-470's own test matrix -- a property-generalized vectorized-vs-scalar
differential, least-squares recovery against known ground truth (not just
one case), canonicalization numerical equivalence over generated trees and a
broad numeric domain, canonicalization's permutation invariance, and the
untested ``is_ill_conditioned`` diagnostic. No `hypothesis` dependency (none
exists in this repo); mirrors the existing ``random.Random(<seed>)`` +
fixed-iteration-count pattern from the sibling suites. On a property-test
failure, the assertion message includes the failing tree's ``serialize_expr``
form and the seed/iteration -- the natural minimal reproduction for this
IR, since no separate shrinking infrastructure exists anywhere in the repo.
"""

from __future__ import annotations

import math
import random

import numpy as np

from slm_training.dsl.symbolic_expr_canonicalize import canonicalize_expr
from slm_training.dsl.symbolic_expr_evaluator import evaluate_vectorized
from slm_training.dsl.symbolic_expr_fit import (
    classify_affine_dependence,
    fit_least_squares,
)
from slm_training.dsl.symbolic_expr_ir import (
    BINARY_OPERATORS,
    CoefficientHole,
    OpApply,
    SymbolicExprNode,
    VarRef,
    serialize_expr,
)
from slm_training.dsl.symbolic_regression_pack import ALLOWED_OPERATORS

_SEED = 20260809


def _random_tree(
    rng: random.Random, *, num_variables: int, num_coefficients: int, depth: int
) -> SymbolicExprNode:
    if depth <= 0 or rng.random() < 0.4:
        if rng.random() < 0.5:
            return VarRef(rng.randrange(num_variables))
        return CoefficientHole(rng.randrange(num_coefficients))
    op = rng.choice(sorted(ALLOWED_OPERATORS))
    arity = 2 if op in BINARY_OPERATORS else 1
    args = tuple(
        _random_tree(
            rng,
            num_variables=num_variables,
            num_coefficients=num_coefficients,
            depth=depth - 1,
        )
        for _ in range(arity)
    )
    return OpApply(op, args)


def _random_coefficient_free_tree(
    rng: random.Random, *, num_variables: int, depth: int
) -> SymbolicExprNode:
    """A tree with no ``CoefficientHole`` -- a valid affine basis term."""
    if depth <= 0 or rng.random() < 0.5:
        return VarRef(rng.randrange(num_variables))
    op = rng.choice(("+", "-", "neg", "abs"))
    arity = 2 if op in ("+", "-") else 1
    args = tuple(
        _random_coefficient_free_tree(rng, num_variables=num_variables, depth=depth - 1)
        for _ in range(arity)
    )
    return OpApply(op, args)


def _random_affine_tree(
    rng: random.Random, *, num_variables: int, num_coefficients: int, depth: int
) -> SymbolicExprNode:
    """``sum(c_i * g_i(vars))`` -- affine in every coefficient hole by
    construction (never a product of two coefficient-dependent subtrees,
    never a coefficient inside a transcendental or division denominator)."""
    terms = [
        OpApply(
            "*",
            (
                CoefficientHole(i),
                _random_coefficient_free_tree(
                    rng, num_variables=num_variables, depth=depth
                ),
            ),
        )
        for i in range(num_coefficients)
    ]
    node: SymbolicExprNode = terms[0]
    for term in terms[1:]:
        node = OpApply("+", (node, term))
    return node


# ---------------------------------------------------------------------------
# Vectorized evaluator vs. a from-scratch scalar reference
# ---------------------------------------------------------------------------


def _scalar_evaluate(
    node: SymbolicExprNode,
    variables_row: tuple[float, ...],
    coefficients: tuple[float, ...],
) -> tuple[float, bool]:
    """Pure-Python, per-row evaluator over the typed IR -- independent of
    ``symbolic_expr_evaluator``'s NumPy dispatch. Returns ``(value,
    hard_domain_violation)``; a hard violation is always ``NaN`` and always
    invalid, mirroring the vectorized evaluator's own contract exactly."""
    if isinstance(node, VarRef):
        return float(variables_row[node.index]), False
    if isinstance(node, CoefficientHole):
        return float(coefficients[node.index]), False
    assert isinstance(node, OpApply)
    if node.operator in {"+", "-", "*", "/"}:
        left, left_hard = _scalar_evaluate(node.args[0], variables_row, coefficients)
        right, right_hard = _scalar_evaluate(node.args[1], variables_row, coefficients)
        if node.operator == "+":
            return left + right, left_hard or right_hard
        if node.operator == "-":
            return left - right, left_hard or right_hard
        if node.operator == "*":
            return left * right, left_hard or right_hard
        # "/"
        if abs(right) < 1e-12:
            return math.nan, True
        return left / right, left_hard or right_hard
    (arg,) = node.args
    value, hard = _scalar_evaluate(arg, variables_row, coefficients)
    if node.operator == "neg":
        return -value, hard
    if node.operator == "abs":
        return abs(value), hard
    if node.operator == "sin":
        return math.sin(value) if math.isfinite(value) else math.nan, hard
    if node.operator == "cos":
        return math.cos(value) if math.isfinite(value) else math.nan, hard
    if node.operator == "exp":
        try:
            return math.exp(value) if math.isfinite(value) else math.nan, hard
        except OverflowError:
            return math.inf, hard
    if node.operator == "log":
        if not math.isfinite(value) or value <= 0:
            return math.nan, True
        return math.log(value), hard
    if node.operator == "sqrt":
        if not math.isfinite(value) or value < 0:
            return math.nan, True
        return math.sqrt(value), hard
    raise AssertionError(f"unhandled operator in scalar reference: {node.operator!r}")


def test_vectorized_matches_scalar_reference_over_random_trees_and_domains() -> None:
    rng = random.Random(_SEED)
    domain_edges = (-1e6, -10.0, -1.0, -1e-13, 0.0, 1e-13, 1.0, 10.0, 1e6)
    for i in range(150):
        tree = _random_tree(rng, num_variables=3, num_coefficients=2, depth=4)
        # Half the rows are broad-random, half deliberately probe domain
        # edges (near-zero denominators/log-sqrt arguments, overflow-prone
        # magnitudes) -- a purely random domain rarely hits these exactly.
        rows = [
            tuple(rng.choice(domain_edges) for _ in range(3))
            if rng.random() < 0.5
            else tuple(rng.uniform(-5, 5) for _ in range(3))
            for _ in range(20)
        ]
        coefficients = tuple(rng.uniform(-5, 5) for _ in range(2))
        variables = np.array(rows, dtype=np.float64)
        for policy in ("reject_nonfinite", "allow_nonfinite"):
            result = evaluate_vectorized(
                tree,
                variables=variables,
                coefficients=np.array(coefficients),
                domain_policy=policy,
            )
            for row_index, row in enumerate(rows):
                expected_value, expected_hard = _scalar_evaluate(
                    tree, row, coefficients
                )
                actual_value = result.predictions[row_index]
                if math.isnan(expected_value):
                    assert math.isnan(actual_value), (
                        i,
                        policy,
                        row_index,
                        serialize_expr(tree),
                        row,
                        expected_value,
                        actual_value,
                    )
                else:
                    assert math.isclose(
                        actual_value, expected_value, rel_tol=1e-9, abs_tol=1e-9
                    ) or (math.isinf(actual_value) and math.isinf(expected_value)), (
                        i,
                        policy,
                        row_index,
                        serialize_expr(tree),
                        row,
                        expected_value,
                        actual_value,
                    )
                if expected_hard:
                    assert result.valid_mask[row_index] is False, (
                        i,
                        policy,
                        row_index,
                        serialize_expr(tree),
                        row,
                    )
                elif policy == "reject_nonfinite" and not math.isfinite(expected_value):
                    assert result.valid_mask[row_index] is False
                elif policy == "reject_nonfinite":
                    assert result.valid_mask[row_index] is True


# ---------------------------------------------------------------------------
# Least-squares fit vs. known ground-truth coefficients
# ---------------------------------------------------------------------------


def test_least_squares_recovers_known_coefficients_over_random_affine_trees() -> None:
    rng = random.Random(_SEED)
    max_residual = 0.0
    exact_recovery_checks = 0
    for i in range(80):
        num_variables = rng.randrange(1, 3)
        num_coefficients = rng.randrange(1, 3)
        tree = _random_affine_tree(
            rng, num_variables=num_variables, num_coefficients=num_coefficients, depth=2
        )
        # Affine-by-construction must never be misclassified -- the false-
        # negative property the existing anti-false-affine test doesn't
        # cover (that one only proves classified-affine trees behave
        # affinely, not that a truly affine tree is always classified so).
        classification = classify_affine_dependence(tree)
        assert classification.status == "affine", (
            i,
            serialize_expr(tree),
            classification,
        )

        n_rows = 40
        variables = np.array(
            [[rng.uniform(-5, 5) for _ in range(num_variables)] for _ in range(n_rows)]
        )
        true_coefficients = np.array(
            [rng.uniform(-3, 3) for _ in range(num_coefficients)]
        )
        evaluated = evaluate_vectorized(
            tree,
            variables=variables,
            coefficients=true_coefficients,
            domain_policy="reject_nonfinite",
        )
        if not all(evaluated.valid_mask):
            continue  # this iteration's random domain hit a genuine violation; skip it
        targets = np.array(evaluated.predictions)

        fit = fit_least_squares(
            tree, variables=variables, targets=targets, train_indices=np.arange(n_rows)
        )
        max_residual = max(max_residual, fit.train_residual_l2)
        # A noise-free affine system's least-squares residual is always
        # near-zero, regardless of rank -- true even when the coefficients
        # themselves aren't uniquely recoverable (collinear basis columns).
        assert fit.train_residual_l2 < 1e-6, (i, serialize_expr(tree), fit)
        if not fit.is_rank_deficient:
            recovered = np.array(fit.fitted_coefficients)
            expected = true_coefficients[list(fit.coefficient_indices)]
            assert np.allclose(recovered, expected, atol=1e-6), (
                i,
                serialize_expr(tree),
                recovered,
                expected,
            )
            exact_recovery_checks += 1
    # The property loop must actually exercise the full-rank recovery path,
    # not vacuously pass because every trial happened to be rank-deficient.
    assert exact_recovery_checks > 40


def test_fit_ill_conditioned_diagnostic_fires_without_rank_deficiency() -> None:
    """AC: rank-deficient and ill-conditioned are diagnosed separately.

    ``test_symbolic_expr_fit.py`` already covers exact rank deficiency
    (identical basis columns); this constructs the untested middle case --
    two *nearly* collinear columns that remain full rank but produce a
    condition number far past ``ILL_CONDITIONED_THRESHOLD``.
    """
    rng = np.random.default_rng(2026)
    n = 30
    v0 = rng.uniform(1, 10, n)
    v1 = v0 + rng.normal(0, 1e-9, n)
    variables = np.column_stack([v0, v1])
    node = OpApply(
        "+",
        (
            OpApply("*", (CoefficientHole(0), VarRef(0))),
            OpApply("*", (CoefficientHole(1), VarRef(1))),
        ),
    )
    targets = 2.0 * v0 + 3.0 * v1
    fit = fit_least_squares(
        node, variables=variables, targets=targets, train_indices=np.arange(n)
    )
    assert fit.is_rank_deficient is False
    assert fit.is_ill_conditioned is True
    assert fit.condition_number > 1e10


# ---------------------------------------------------------------------------
# Canonicalization: numerical equivalence and permutation invariance
# ---------------------------------------------------------------------------


def test_canonicalize_preserves_numerical_semantics_over_random_trees() -> None:
    rng = random.Random(_SEED)
    domain_edges = (-1e6, -10.0, -1.0, -1e-13, 0.0, 1e-13, 1.0, 10.0, 1e6)
    for i in range(120):
        tree = _random_tree(rng, num_variables=3, num_coefficients=2, depth=4)
        canonical = canonicalize_expr(tree)
        rows = [
            tuple(rng.choice(domain_edges) for _ in range(3))
            if rng.random() < 0.5
            else tuple(rng.uniform(-5, 5) for _ in range(3))
            for _ in range(15)
        ]
        variables = np.array(rows, dtype=np.float64)
        coefficients = np.array([rng.uniform(-5, 5) for _ in range(2)])
        before = evaluate_vectorized(
            tree,
            variables=variables,
            coefficients=coefficients,
            domain_policy="allow_nonfinite",
        )
        after = evaluate_vectorized(
            canonical,
            variables=variables,
            coefficients=coefficients,
            domain_policy="allow_nonfinite",
        )
        assert before.valid_mask == after.valid_mask, (
            i,
            serialize_expr(tree),
            serialize_expr(canonical),
        )
        for row_index in range(len(rows)):
            b, a = before.predictions[row_index], after.predictions[row_index]
            if math.isnan(b):
                assert math.isnan(a), (
                    i,
                    row_index,
                    serialize_expr(tree),
                    serialize_expr(canonical),
                )
            else:
                assert b == a or math.isclose(b, a, rel_tol=1e-12, abs_tol=1e-12), (
                    i,
                    row_index,
                    serialize_expr(tree),
                    serialize_expr(canonical),
                    b,
                    a,
                )


def _swap_commutative_operands(
    rng: random.Random, node: SymbolicExprNode
) -> SymbolicExprNode:
    """A structurally different but semantically identical tree: every
    commutative (``+``/``*``) node's operands are recursively, randomly
    reordered. Non-commutative structure is untouched."""
    if isinstance(node, (VarRef, CoefficientHole)):
        return node
    args = tuple(_swap_commutative_operands(rng, arg) for arg in node.args)
    if node.operator in {"+", "*"} and rng.random() < 0.5:
        args = (args[1], args[0])
    return OpApply(node.operator, args)


def test_canonicalize_is_permutation_invariant_over_random_trees() -> None:
    """AC: alpha/permutation invariance. Reordering commutative operands
    before canonicalizing must converge to the identical canonical form."""
    rng = random.Random(_SEED)
    for i in range(100):
        tree = _random_tree(rng, num_variables=3, num_coefficients=2, depth=4)
        permuted = _swap_commutative_operands(rng, tree)
        canonical_a = canonicalize_expr(tree)
        canonical_b = canonicalize_expr(permuted)
        assert serialize_expr(canonical_a) == serialize_expr(canonical_b), (
            i,
            serialize_expr(tree),
            serialize_expr(permuted),
            serialize_expr(canonical_a),
            serialize_expr(canonical_b),
        )
