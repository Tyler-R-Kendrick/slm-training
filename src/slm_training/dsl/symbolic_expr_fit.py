"""SRP-004 (SLM-465): affine coefficient-dependence analysis and deterministic
least-squares fitting over the typed symbolic-expression IR.

Separates discrete expression-structure synthesis (SRP-002's AST) from
continuous coefficient fitting. An expression is *affine in its coefficient
holes* when, viewed as a function of the coefficient values with variables
held fixed, it is exactly ``sum_i(c_i * basis_i(vars)) + offset(vars)`` for
some data-dependent per-coefficient basis vectors and a fixed offset -- never
a product of two coefficient-dependent subtrees, never a coefficient inside
a transcendental function, never a coefficient in a division's denominator.

:func:`classify_affine_dependence` is a purely structural, data-independent
check (no numeric evaluation) -- the same recursive rule set that
:func:`fit_least_squares` uses to build the design matrix, so the two can
never silently disagree. Every coefficient-free subtree is evaluated via
SRP-003's :func:`symbolic_expr_evaluator.evaluate_vectorized` rather than a
second, independently-written arithmetic dispatch, so domain-violation
semantics stay identical between analysis and fitting.

Non-affine coefficient holes are unsupported in this module by design: no
optimizer is ever silently invoked for them. Fitting reads only the caller's
``train_indices`` -- held-out/extrapolation rows can be scored afterward but
never influence the least-squares solve itself.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from slm_training.dsl.symbolic_expr_evaluator import NUMERIC_DOMAIN_POLICIES, evaluate_vectorized
from slm_training.dsl.symbolic_expr_ir import (
    CoefficientHole,
    OpApply,
    SymbolicExprNode,
    VarRef,
    collect_symbols,
    validate_expr_budget,
)

CLASSIFICATION_SCHEMA_VERSION = "v1"
FIT_SCHEMA_VERSION = "v1"
SOLVER_IDENTITY = "numpy.linalg.lstsq"
ILL_CONDITIONED_THRESHOLD = 1e10

_UNARY_TRANSCENDENTAL = frozenset({"sin", "cos", "exp", "log", "sqrt", "abs"})


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class NonAffineError(ValueError):
    """``node`` is not affine in its coefficient holes under this module's
    conservative supported grammar."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass
class _Basis:
    """``node``'s value equals ``sum(c[i] * coefficient_columns[i]) + fixed_offset``,
    row-wise, over some fixed dataset."""

    coefficient_columns: dict[int, np.ndarray] = field(default_factory=dict)
    fixed_offset: np.ndarray | None = None  # None until first assigned; always (n_rows,) after


def _is_coefficient_free(node: SymbolicExprNode) -> bool:
    return not collect_symbols(node)[1]


def _evaluate_coefficient_free(
    node: SymbolicExprNode, variables: np.ndarray, domain_policy: str
) -> np.ndarray:
    result = evaluate_vectorized(node, variables=variables, domain_policy=domain_policy)
    return np.array(result.predictions, dtype=np.float64)


def _combine_add(a: _Basis, b: _Basis) -> _Basis:
    columns: dict[int, np.ndarray] = dict(a.coefficient_columns)
    for index, column in b.coefficient_columns.items():
        columns[index] = columns[index] + column if index in columns else column
    offset = a.fixed_offset + b.fixed_offset
    return _Basis(coefficient_columns=columns, fixed_offset=offset)


def _combine_sub(a: _Basis, b: _Basis) -> _Basis:
    columns: dict[int, np.ndarray] = dict(a.coefficient_columns)
    for index, column in b.coefficient_columns.items():
        columns[index] = columns[index] - column if index in columns else -column
    offset = a.fixed_offset - b.fixed_offset
    return _Basis(coefficient_columns=columns, fixed_offset=offset)


def _combine_neg(a: _Basis) -> _Basis:
    columns = {index: -column for index, column in a.coefficient_columns.items()}
    return _Basis(coefficient_columns=columns, fixed_offset=-a.fixed_offset)


def _combine_scale(a: _Basis, factor: np.ndarray) -> _Basis:
    columns = {index: column * factor for index, column in a.coefficient_columns.items()}
    return _Basis(coefficient_columns=columns, fixed_offset=a.fixed_offset * factor)


def _combine_divide(a: _Basis, divisor: np.ndarray) -> _Basis:
    columns = {index: column / divisor for index, column in a.coefficient_columns.items()}
    return _Basis(coefficient_columns=columns, fixed_offset=a.fixed_offset / divisor)


def _affine_columns(node: SymbolicExprNode, variables: np.ndarray, domain_policy: str) -> _Basis:
    if _is_coefficient_free(node):
        return _Basis(coefficient_columns={}, fixed_offset=_evaluate_coefficient_free(node, variables, domain_policy))
    if isinstance(node, CoefficientHole):
        n_rows = variables.shape[0]
        return _Basis(coefficient_columns={node.index: np.ones(n_rows)}, fixed_offset=np.zeros(n_rows))
    if not isinstance(node, OpApply):
        raise TypeError(f"unsupported symbolic expression node type: {type(node).__name__}")

    if node.operator == "+":
        left, right = (_affine_columns(arg, variables, domain_policy) for arg in node.args)
        return _combine_add(left, right)
    if node.operator == "-":
        left, right = (_affine_columns(arg, variables, domain_policy) for arg in node.args)
        return _combine_sub(left, right)
    if node.operator == "neg":
        return _combine_neg(_affine_columns(node.args[0], variables, domain_policy))
    if node.operator == "*":
        left_node, right_node = node.args
        if _is_coefficient_free(left_node):
            factor = _evaluate_coefficient_free(left_node, variables, domain_policy)
            return _combine_scale(_affine_columns(right_node, variables, domain_policy), factor)
        if _is_coefficient_free(right_node):
            factor = _evaluate_coefficient_free(right_node, variables, domain_policy)
            return _combine_scale(_affine_columns(left_node, variables, domain_policy), factor)
        raise NonAffineError("product of two coefficient-dependent subexpressions is not affine")
    if node.operator == "/":
        numerator_node, denominator_node = node.args
        if _is_coefficient_free(denominator_node):
            divisor = _evaluate_coefficient_free(denominator_node, variables, domain_policy)
            return _combine_divide(_affine_columns(numerator_node, variables, domain_policy), divisor)
        raise NonAffineError("a coefficient hole in a division's denominator is not affine")
    if node.operator in _UNARY_TRANSCENDENTAL:
        raise NonAffineError(
            f"a coefficient hole inside '{node.operator}' is not affine (coefficient-dependent argument)"
        )
    raise NonAffineError(f"unsupported operator for affine analysis: {node.operator!r}")


@dataclass(frozen=True)
class AffineClassificationV1:
    """Structural, data-independent verdict on ``node``'s affinity in its
    coefficient holes."""

    schema_version: str
    status: str  # "affine" | "non_affine"
    coefficient_indices: tuple[int, ...]
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_affine_dependence(node: SymbolicExprNode) -> AffineClassificationV1:
    """Whether ``node`` is affine in its coefficient holes -- structural
    only, no numeric data required. Uses the identical rule set
    :func:`fit_least_squares` uses to build the design matrix (via a
    single-row dummy evaluation), so classification and fitting can never
    silently disagree."""
    if not isinstance(node, (VarRef, CoefficientHole, OpApply)):
        raise TypeError(f"unsupported symbolic expression node type: {type(node).__name__}")
    validate_expr_budget(node)
    variable_indices, coefficient_indices = collect_symbols(node)
    n_vars = (max(variable_indices) + 1) if variable_indices else 0
    dummy_variables = np.ones((1, n_vars)) if n_vars else np.zeros((1, 0))
    try:
        basis = _affine_columns(node, dummy_variables, domain_policy="allow_nonfinite")
    except NonAffineError as exc:
        return AffineClassificationV1(
            schema_version=CLASSIFICATION_SCHEMA_VERSION,
            status="non_affine",
            coefficient_indices=tuple(sorted(coefficient_indices)),
            reason=exc.reason,
        )
    return AffineClassificationV1(
        schema_version=CLASSIFICATION_SCHEMA_VERSION,
        status="affine",
        coefficient_indices=tuple(sorted(basis.coefficient_columns)),
        reason=None,
    )


def _fit_digest(fit: "LeastSquaresFitV1") -> str:
    payload = asdict(fit)
    payload.pop("fit_digest", None)
    return _digest(payload)


@dataclass(frozen=True)
class LeastSquaresFitV1:
    """Tamper-evident, replayable deterministic least-squares fit."""

    schema_version: str
    coefficient_indices: tuple[int, ...]
    fitted_coefficients: tuple[float, ...]
    rank: int
    condition_number: float
    is_rank_deficient: bool
    is_ill_conditioned: bool
    n_train_rows: int
    train_residual_l2: float
    solver_identity: str
    fit_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LeastSquaresFitV1":
        if payload.get("schema_version") != FIT_SCHEMA_VERSION:
            raise ValueError(f"unsupported least-squares fit schema: {payload.get('schema_version')!r}")
        fit = cls(
            **{
                **payload,
                "coefficient_indices": tuple(payload["coefficient_indices"]),
                "fitted_coefficients": tuple(payload["fitted_coefficients"]),
            }
        )
        if _fit_digest(fit) != fit.fit_digest:
            raise ValueError("least-squares fit digest mismatch (tampered or corrupted payload)")
        return fit


def fit_integrity_ok(fit: LeastSquaresFitV1) -> bool:
    return _fit_digest(fit) == fit.fit_digest


def fit_least_squares(
    node: SymbolicExprNode,
    *,
    variables: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    domain_policy: str = "reject_nonfinite",
) -> LeastSquaresFitV1:
    """Deterministically fit ``node``'s coefficient holes by ordinary least
    squares, using only ``variables[train_indices]``/``targets[train_indices]``.

    Raises :class:`NonAffineError` if ``node`` is not affine in its
    coefficient holes -- no optimizer is ever silently substituted for the
    non-affine case. Raises ``ValueError`` if ``node`` references no
    coefficient hole at all (nothing to fit).
    """
    if domain_policy not in NUMERIC_DOMAIN_POLICIES:
        raise ValueError(f"unsupported numeric_domain_policy: {domain_policy!r}")
    if not isinstance(node, (VarRef, CoefficientHole, OpApply)):
        raise TypeError(f"unsupported symbolic expression node type: {type(node).__name__}")
    validate_expr_budget(node)

    variables_arr = np.asarray(variables, dtype=np.float64)
    if variables_arr.ndim != 2:
        raise ValueError(f"variables must be a 2-D array, got shape {variables_arr.shape}")
    targets_arr = np.asarray(targets, dtype=np.float64)
    if targets_arr.shape != (variables_arr.shape[0],):
        raise ValueError(
            f"targets must be shape ({variables_arr.shape[0]},), got {targets_arr.shape}"
        )
    train_indices_arr = np.asarray(train_indices, dtype=np.intp)
    if train_indices_arr.ndim != 1 or train_indices_arr.size == 0:
        raise ValueError("train_indices must be a non-empty 1-D array")
    if np.any(train_indices_arr < 0) or np.any(train_indices_arr >= variables_arr.shape[0]):
        raise ValueError("train_indices out of range for variables")

    _, coefficient_indices = collect_symbols(node)
    if not coefficient_indices:
        raise ValueError("expression references no coefficient hole -- nothing to fit")

    basis = _affine_columns(node, variables_arr, domain_policy)  # raises NonAffineError if not affine
    ordered_indices = tuple(sorted(basis.coefficient_columns))

    design_matrix = np.column_stack([basis.coefficient_columns[i] for i in ordered_indices])
    adjusted_targets = targets_arr - basis.fixed_offset

    train_design = design_matrix[train_indices_arr]
    train_target = adjusted_targets[train_indices_arr]

    coefficients, _residuals, rank, singular_values = np.linalg.lstsq(train_design, train_target, rcond=None)

    nonzero_singular = singular_values[singular_values > 0]
    condition_number = (
        float(nonzero_singular[0] / nonzero_singular[-1]) if nonzero_singular.size else float("inf")
    )
    is_rank_deficient = bool(rank < len(ordered_indices))
    is_ill_conditioned = bool(condition_number > ILL_CONDITIONED_THRESHOLD)

    train_residual = train_design @ coefficients - train_target
    train_residual_l2 = float(np.linalg.norm(train_residual))

    fit = LeastSquaresFitV1(
        schema_version=FIT_SCHEMA_VERSION,
        coefficient_indices=ordered_indices,
        fitted_coefficients=tuple(float(c) for c in coefficients),
        rank=int(rank),
        condition_number=condition_number,
        is_rank_deficient=is_rank_deficient,
        is_ill_conditioned=is_ill_conditioned,
        n_train_rows=int(train_indices_arr.size),
        train_residual_l2=train_residual_l2,
        solver_identity=SOLVER_IDENTITY,
        fit_digest="",
    )
    return dataclasses.replace(fit, fit_digest=_fit_digest(fit))


__all__ = [
    "CLASSIFICATION_SCHEMA_VERSION",
    "FIT_SCHEMA_VERSION",
    "ILL_CONDITIONED_THRESHOLD",
    "SOLVER_IDENTITY",
    "AffineClassificationV1",
    "LeastSquaresFitV1",
    "NonAffineError",
    "classify_affine_dependence",
    "fit_integrity_ok",
    "fit_least_squares",
]
