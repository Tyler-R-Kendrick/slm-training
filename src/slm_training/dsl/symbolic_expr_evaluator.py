"""SRP-003 (SLM-461): safe vectorized NumPy evaluator for the typed symbolic
expression IR (``symbolic_expr_ir.py``), with explicit numerical-domain
policies.

Supersedes ``symbolic_regression_pack.evaluate_expression`` (the scalar,
untyped-tree reference evaluator kept only to prove the parse/evaluate round
trip) for real search/fitting use. Every operator is a fixed, hardcoded
NumPy dispatch keyed on the ``OpApply.operator`` string -- there is no
``eval``/``exec``/``getattr``/``importlib`` path anywhere in this module, so
no expression can ever run arbitrary Python.

Numerical-domain policy (``symbolic_regression_pack.NUMERIC_DOMAIN_POLICIES``,
honored here rather than duplicated):

- Division, ``log``, and ``sqrt`` have operators whose real-valued domain is
  a strict subset of the reals (division near zero, ``log``/``sqrt`` of a
  non-positive number). Applying them outside that domain is always a
  *hard domain violation*: the affected row is always invalid, and the
  produced value is always ``NaN``, regardless of ``domain_policy``.
- Any other row whose final prediction is non-finite (``NaN``/``Inf`` from
  overflow/underflow in otherwise-legal arithmetic, e.g. ``exp`` of a large
  argument) is a *soft* non-finite result. Under the default
  ``"reject_nonfinite"`` policy it also invalidates the row; under
  ``"allow_nonfinite"`` the row stays valid for callers that explicitly want
  to observe raw overflow propagation, but the non-finite value and a
  ``nonfinite_result`` violation record are still always returned -- a
  ``NaN``/``Inf`` prediction can never silently look like a good score,
  because it is never coerced to a finite value and never dropped from
  ``domain_violations``.

Structural problems (unknown/out-of-range variable index, an unbound
coefficient hole, a non-2-D or non-numeric ``variables`` array, an
expression over budget) always raise -- they are caller bugs, not data to
evaluate, and this evaluator fails closed on them rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from slm_training.dsl.symbolic_expr_ir import (
    CoefficientHole,
    OpApply,
    SymbolicExprNode,
    VarRef,
    collect_symbols,
    validate_expr_budget,
)

NUMERIC_DOMAIN_POLICIES = frozenset({"reject_nonfinite", "allow_nonfinite"})
DEFAULT_DOMAIN_POLICY = "reject_nonfinite"

_DIVISION_NEAR_ZERO_EPSILON = 1e-12

_HARD_DOMAIN_VIOLATION_REASONS = frozenset(
    {"division_near_zero", "log_nonpositive_domain", "sqrt_negative_domain"}
)
_SOFT_NONFINITE_REASON = "nonfinite_result"
_DOMAIN_VIOLATION_REASONS = _HARD_DOMAIN_VIOLATION_REASONS | {_SOFT_NONFINITE_REASON}

FAILURE_CLASSES = frozenset({"none", "partial", "total"})


@dataclass(frozen=True)
class DomainViolationV1:
    """One row's operator application falling outside its declared domain."""

    row_index: int
    operator: str
    reason: str

    def __post_init__(self) -> None:
        if self.row_index < 0:
            raise ValueError(f"row_index must be >= 0, got {self.row_index}")
        if self.reason not in _DOMAIN_VIOLATION_REASONS:
            raise ValueError(f"unsupported domain violation reason: {self.reason!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "operator": self.operator,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EvaluationResultV1:
    """Typed evaluation evidence for one expression over one dataset."""

    predictions: tuple[float, ...]
    valid_mask: tuple[bool, ...]
    domain_violations: tuple[DomainViolationV1, ...]
    failure_class: str
    domain_policy: str

    def __post_init__(self) -> None:
        if len(self.predictions) != len(self.valid_mask):
            raise ValueError("predictions and valid_mask must have the same length")
        if self.failure_class not in FAILURE_CLASSES:
            raise ValueError(f"unsupported failure_class: {self.failure_class!r}")
        if self.domain_policy not in NUMERIC_DOMAIN_POLICIES:
            raise ValueError(f"unsupported domain_policy: {self.domain_policy!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictions": list(self.predictions),
            "valid_mask": list(self.valid_mask),
            "domain_violations": [v.to_dict() for v in self.domain_violations],
            "failure_class": self.failure_class,
            "domain_policy": self.domain_policy,
        }


def _apply_operator(
    operator: str, args: list[np.ndarray], row_indices: np.ndarray
) -> tuple[np.ndarray, list[DomainViolationV1]]:
    violations: list[DomainViolationV1] = []
    with np.errstate(all="ignore"):
        if operator == "+":
            result = args[0] + args[1]
        elif operator == "-":
            result = args[0] - args[1]
        elif operator == "*":
            result = args[0] * args[1]
        elif operator == "/":
            numerator, denominator = args
            near_zero = np.abs(denominator) < _DIVISION_NEAR_ZERO_EPSILON
            safe_denominator = np.where(near_zero, 1.0, denominator)
            result = np.where(near_zero, np.nan, numerator / safe_denominator)
            violations.extend(
                DomainViolationV1(row_index=int(i), operator="/", reason="division_near_zero")
                for i in np.nonzero(near_zero)[0]
            )
        elif operator == "neg":
            result = -args[0]
        elif operator == "abs":
            result = np.abs(args[0])
        elif operator == "sin":
            result = np.sin(args[0])
        elif operator == "cos":
            result = np.cos(args[0])
        elif operator == "exp":
            result = np.exp(args[0])
        elif operator == "log":
            nonpositive = args[0] <= 0
            safe_arg = np.where(nonpositive, 1.0, args[0])
            result = np.where(nonpositive, np.nan, np.log(safe_arg))
            violations.extend(
                DomainViolationV1(row_index=int(i), operator="log", reason="log_nonpositive_domain")
                for i in np.nonzero(nonpositive)[0]
            )
        elif operator == "sqrt":
            negative = args[0] < 0
            safe_arg = np.where(negative, 0.0, args[0])
            result = np.where(negative, np.nan, np.sqrt(safe_arg))
            violations.extend(
                DomainViolationV1(row_index=int(i), operator="sqrt", reason="sqrt_negative_domain")
                for i in np.nonzero(negative)[0]
            )
        else:
            # Unreachable for any node built through symbolic_expr_ir (its
            # OpApply.__post_init__ already rejects unknown operators), but
            # a tree can be hand-assembled bypassing that path -- fail
            # closed rather than guess.
            raise ValueError(f"unsupported operator: {operator!r}")
    return result.astype(np.float64, copy=False), violations


def _evaluate_node(
    node: SymbolicExprNode,
    variables: np.ndarray,
    coefficients: np.ndarray | None,
    row_indices: np.ndarray,
) -> tuple[np.ndarray, list[DomainViolationV1]]:
    if isinstance(node, VarRef):
        return variables[:, node.index].astype(np.float64, copy=True), []
    if isinstance(node, CoefficientHole):
        assert coefficients is not None  # validated by the caller before recursion
        value = float(coefficients[node.index])
        return np.full(row_indices.shape[0], value, dtype=np.float64), []
    if isinstance(node, OpApply):
        arg_arrays: list[np.ndarray] = []
        violations: list[DomainViolationV1] = []
        for arg in node.args:
            arr, arg_violations = _evaluate_node(arg, variables, coefficients, row_indices)
            arg_arrays.append(arr)
            violations.extend(arg_violations)
        result, op_violations = _apply_operator(node.operator, arg_arrays, row_indices)
        violations.extend(op_violations)
        return result, violations
    raise TypeError(f"unsupported symbolic expression node type: {type(node).__name__}")


def evaluate_vectorized(
    node: SymbolicExprNode,
    *,
    variables: np.ndarray,
    coefficients: np.ndarray | None = None,
    domain_policy: str = DEFAULT_DOMAIN_POLICY,
) -> EvaluationResultV1:
    """Evaluate ``node`` over every row of ``variables`` at once.

    ``variables`` must be a 2-D numeric array shaped ``(n_rows, n_vars)``;
    column ``i`` binds ``VarRef(index=i)``. ``coefficients`` is a 1-D numeric
    array binding ``CoefficientHole(index=i)`` to ``coefficients[i]`` --
    required only if ``node`` references any coefficient hole.

    Raises on structural problems (bad shape/dtype, out-of-range variable
    index, an unbound coefficient hole, an over-budget expression). Never
    raises for a numeric-domain violation -- that is reported as evaluation
    evidence in the returned :class:`EvaluationResultV1` instead.
    """
    if domain_policy not in NUMERIC_DOMAIN_POLICIES:
        raise ValueError(f"unsupported numeric_domain_policy: {domain_policy!r}")
    if not isinstance(node, (VarRef, CoefficientHole, OpApply)):
        raise TypeError(f"unsupported symbolic expression node type: {type(node).__name__}")
    validate_expr_budget(node)

    variables_arr = np.asarray(variables)
    if variables_arr.ndim != 2:
        raise ValueError(
            f"variables must be a 2-D array of shape (n_rows, n_vars), got shape {variables_arr.shape}"
        )
    if not np.issubdtype(variables_arr.dtype, np.number):
        raise TypeError(f"variables must have a numeric dtype, got {variables_arr.dtype}")
    variables_arr = variables_arr.astype(np.float64, copy=False)
    n_rows, n_vars = variables_arr.shape

    variable_indices, coefficient_indices = collect_symbols(node)
    out_of_range = {i for i in variable_indices if i >= n_vars}
    if out_of_range:
        raise ValueError(
            f"variable index/indices {sorted(out_of_range)} out of range for {n_vars} column(s)"
        )

    coefficients_arr: np.ndarray | None = None
    if coefficient_indices:
        if coefficients is None:
            raise ValueError(
                f"expression references coefficient hole(s) {sorted(coefficient_indices)} "
                "but no coefficients were bound"
            )
        coefficients_arr = np.asarray(coefficients)
        if coefficients_arr.ndim != 1:
            raise ValueError(f"coefficients must be a 1-D array, got shape {coefficients_arr.shape}")
        if not np.issubdtype(coefficients_arr.dtype, np.number):
            raise TypeError(f"coefficients must have a numeric dtype, got {coefficients_arr.dtype}")
        coefficients_arr = coefficients_arr.astype(np.float64, copy=False)
        missing = {i for i in coefficient_indices if i >= coefficients_arr.shape[0]}
        if missing:
            raise ValueError(f"missing coefficient binding(s) for hole index/indices {sorted(missing)}")

    row_indices = np.arange(n_rows)
    predictions, violations = _evaluate_node(node, variables_arr, coefficients_arr, row_indices)
    if predictions.shape != (n_rows,):
        raise ValueError(f"internal shape error: predictions shape {predictions.shape} != ({n_rows},)")

    finite_mask = np.isfinite(predictions)
    hard_violation_rows = {v.row_index for v in violations if v.reason in _HARD_DOMAIN_VIOLATION_REASONS}
    soft_violations = [
        DomainViolationV1(row_index=int(row), operator="__result__", reason=_SOFT_NONFINITE_REASON)
        for row in np.nonzero(~finite_mask)[0]
        if int(row) not in hard_violation_rows
    ]
    all_violations = tuple(
        sorted(violations + soft_violations, key=lambda v: (v.row_index, v.operator, v.reason))
    )

    if domain_policy == "allow_nonfinite":
        valid_mask = tuple(row not in hard_violation_rows for row in range(n_rows))
    else:
        valid_mask = tuple(
            bool(finite_mask[row]) and row not in hard_violation_rows for row in range(n_rows)
        )

    invalid_count = valid_mask.count(False)
    if invalid_count == 0:
        failure_class = "none"
    elif invalid_count == n_rows:
        failure_class = "total"
    else:
        failure_class = "partial"

    return EvaluationResultV1(
        predictions=tuple(float(p) for p in predictions),
        valid_mask=valid_mask,
        domain_violations=all_violations,
        failure_class=failure_class,
        domain_policy=domain_policy,
    )


__all__ = [
    "DEFAULT_DOMAIN_POLICY",
    "FAILURE_CLASSES",
    "NUMERIC_DOMAIN_POLICIES",
    "DomainViolationV1",
    "EvaluationResultV1",
    "evaluate_vectorized",
]
