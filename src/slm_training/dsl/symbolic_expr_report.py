"""SRP-007 (SLM-466): validity, fit, extrapolation, complexity, and Pareto
reporting over the typed symbolic-expression IR.

A pack-local reporting layer that composes SRP-002/003/004/005 rather than
re-deriving evaluation, fitting, or canonicalization logic: complexity comes
from ``symbolic_expr_ir.expr_depth_and_node_count``, numerical validity and
predictions come from ``symbolic_expr_evaluator.evaluate_vectorized``, fitted
coefficients come from ``symbolic_expr_fit.fit_least_squares``, and canonical
identity comes from ``symbolic_expr_canonicalize.canonicalize_expr``.

Divergence note (same precedent as SRP-004/SRP-005, itself following
SGS-008): the pre-registered ``downstream_extension_map`` row for SRP-007
predicted ``new_owner_justified: false`` ("Reporting layer over
SRP-003/004/005"), i.e. adding these symbols to an existing owner's
``owner_symbols``. The record/front dataclasses below are not literally
defined in any of those sibling files, so per the ownership map's own rule
this module is registered as its own subsystem instead; the divergence is
documented in ``ownership_map.json``.

Per-expression evidence is a *vector*, never collapsed to one champion
scalar: :class:`ExpressionEvidenceV1` reports train/validation/extrapolation
loss separately (never averaged together), alongside complexity and fit
evidence, and :func:`pareto_front` extracts the non-dominated subset across
every recorded objective rather than ranking by a single weighted score.

A split's loss is declared ``math.inf`` whenever any row in that split is
numerically invalid (per ``EvaluationResultV1.failure_class``) rather than
being averaged over only the valid rows -- an expression that fits well on
a handful of valid rows while producing domain violations elsewhere must
never look better than one that is valid and merely mediocre everywhere;
that would let invalid numerical output buy a better score by omission.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from slm_training.dsl.symbolic_expr_canonicalize import canonicalize_expr
from slm_training.dsl.symbolic_expr_evaluator import (
    DEFAULT_DOMAIN_POLICY,
    FAILURE_CLASSES,
    NUMERIC_DOMAIN_POLICIES,
    evaluate_vectorized,
)
from slm_training.dsl.symbolic_expr_fit import classify_affine_dependence, fit_least_squares
from slm_training.dsl.symbolic_expr_ir import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_NODES,
    SymbolicExprNode,
    collect_symbols,
    expr_depth_and_node_count,
    serialize_expr,
    validate_expr_budget,
)

if TYPE_CHECKING:
    from slm_training.dsl.symbolic_regression_pack import SymbolicRegressionProblemV1

EVIDENCE_SCHEMA_VERSION = "v1"
PARETO_SCHEMA_VERSION = "v1"
COMPLEXITY_FORMULA = "node_count_v1"
LOSS_FUNCTION = "mean_squared_error"

SPLIT_NAMES = ("train", "validation", "extrapolation")
FIT_STATUSES = frozenset({"no_coefficients", "fitted", "non_affine"})


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class SplitEvidenceV1:
    """Loss/validity evidence for one problem split (train/validation/extrapolation)."""

    split: str
    n_rows: int
    n_valid_rows: int
    loss: float
    failure_class: str

    def __post_init__(self) -> None:
        if self.split not in SPLIT_NAMES:
            raise ValueError(f"unsupported split name: {self.split!r}")
        if self.n_rows <= 0:
            raise ValueError(f"n_rows must be positive, got {self.n_rows}")
        if not 0 <= self.n_valid_rows <= self.n_rows:
            raise ValueError(
                f"n_valid_rows must be within [0, {self.n_rows}], got {self.n_valid_rows}"
            )
        if self.failure_class not in FAILURE_CLASSES:
            raise ValueError(f"unsupported failure_class: {self.failure_class!r}")
        if self.loss != math.inf and not (math.isfinite(self.loss) and self.loss >= 0):
            raise ValueError(f"loss must be a finite non-negative float or +inf, got {self.loss}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _evidence_digest(record: "ExpressionEvidenceV1") -> str:
    payload = asdict(record)
    payload.pop("record_digest", None)
    return _digest(payload)


@dataclass(frozen=True)
class ExpressionEvidenceV1:
    """Tamper-evident, replayable multi-objective evidence for one expression.

    Deliberately reports a vector -- validity, per-split loss, complexity,
    canonical identity, and fit evidence -- rather than a single champion
    score. Any weighted scalar an experiment wants is derived from this
    record afterward; it is never computed here in place of the vector.
    """

    schema_version: str
    problem_fingerprint: str
    pack_id: str
    canonical_expr: str
    canonical_hash: str
    node_count: int
    depth: int
    coefficient_count: int
    complexity_formula: str
    loss_function: str
    domain_policy: str
    fit_status: str
    fit_rank: int | None
    fit_condition_number: float | None
    fit_is_rank_deficient: bool | None
    fit_is_ill_conditioned: bool | None
    solver_identity: str | None
    splits: tuple[SplitEvidenceV1, ...]
    max_depth_budget: int
    max_nodes_budget: int
    record_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported evidence schema: {self.schema_version!r}")
        if self.complexity_formula != COMPLEXITY_FORMULA:
            raise ValueError(f"unsupported complexity_formula: {self.complexity_formula!r}")
        if self.loss_function != LOSS_FUNCTION:
            raise ValueError(f"unsupported loss_function: {self.loss_function!r}")
        if self.domain_policy not in NUMERIC_DOMAIN_POLICIES:
            raise ValueError(f"unsupported domain_policy: {self.domain_policy!r}")
        if self.fit_status not in FIT_STATUSES:
            raise ValueError(f"unsupported fit_status: {self.fit_status!r}")
        if self.node_count <= 0 or self.depth <= 0 or self.coefficient_count < 0:
            raise ValueError("node_count/depth must be positive and coefficient_count non-negative")
        if self.max_depth_budget <= 0 or self.max_nodes_budget <= 0:
            raise ValueError("max_depth_budget/max_nodes_budget must be positive")
        if not self.splits:
            raise ValueError("splits must be non-empty")
        seen_splits = {s.split for s in self.splits}
        if len(seen_splits) != len(self.splits):
            raise ValueError("splits must not repeat a split name")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExpressionEvidenceV1":
        if payload.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported evidence schema: {payload.get('schema_version')!r}")
        splits = tuple(SplitEvidenceV1(**split) for split in payload["splits"])
        record = cls(**{**payload, "splits": splits})
        if _evidence_digest(record) != record.record_digest:
            raise ValueError("evidence record digest mismatch (tampered or corrupted payload)")
        return record


def evidence_integrity_ok(record: ExpressionEvidenceV1) -> bool:
    return _evidence_digest(record) == record.record_digest


def _score_split(
    node: SymbolicExprNode,
    split: str,
    indices: tuple[int, ...],
    variables: np.ndarray,
    targets: np.ndarray,
    coefficients: np.ndarray | None,
    domain_policy: str,
    fit_status: str,
) -> SplitEvidenceV1:
    n_rows = len(indices)
    if fit_status == "non_affine":
        # No coefficients could be fit, so nothing here is scoreable -- fail
        # closed rather than silently treat an unscored split as a good one.
        return SplitEvidenceV1(
            split=split, n_rows=n_rows, n_valid_rows=0, loss=math.inf, failure_class="total"
        )
    row_indices = np.asarray(indices, dtype=np.intp)
    result = evaluate_vectorized(
        node, variables=variables[row_indices], coefficients=coefficients, domain_policy=domain_policy
    )
    n_valid_rows = result.valid_mask.count(True)
    if result.failure_class != "none":
        loss = math.inf
    else:
        predictions = np.array(result.predictions, dtype=np.float64)
        loss = float(np.mean((predictions - targets[row_indices]) ** 2))
    return SplitEvidenceV1(
        split=split, n_rows=n_rows, n_valid_rows=n_valid_rows, loss=loss, failure_class=result.failure_class
    )


def build_evidence_record(
    node: SymbolicExprNode,
    problem: "SymbolicRegressionProblemV1",
    *,
    domain_policy: str | None = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> ExpressionEvidenceV1:
    """Build the full multi-objective evidence record for ``node`` under ``problem``.

    ``domain_policy`` defaults to ``problem.numeric_domain_policy`` -- the
    problem contract's own declared policy, not a second independent default.
    Coefficients are fit on ``problem.train_indices`` only (via
    ``fit_least_squares``); validation/extrapolation rows are scored
    afterward and never influence the fit.
    """
    validate_expr_budget(node, max_depth=max_depth, max_nodes=max_nodes)
    policy = domain_policy if domain_policy is not None else problem.numeric_domain_policy
    if policy not in NUMERIC_DOMAIN_POLICIES:
        raise ValueError(f"unsupported domain_policy: {policy!r}")

    depth, node_count = expr_depth_and_node_count(node)
    canonical_expr = serialize_expr(canonicalize_expr(node))
    canonical_hash = hashlib.sha256(canonical_expr.encode("utf-8")).hexdigest()
    _variable_indices, coefficient_indices = collect_symbols(node)
    coefficient_count = len(coefficient_indices)

    variables_arr = np.asarray(problem.table, dtype=np.float64)
    targets_arr = np.asarray(problem.targets, dtype=np.float64)

    fit_status: str
    coefficients_arr: np.ndarray | None = None
    fit_rank: int | None = None
    fit_condition_number: float | None = None
    fit_is_rank_deficient: bool | None = None
    fit_is_ill_conditioned: bool | None = None
    solver_identity: str | None = None

    if coefficient_count == 0:
        fit_status = "no_coefficients"
    elif classify_affine_dependence(node).status != "affine":
        fit_status = "non_affine"
    else:
        fit = fit_least_squares(
            node,
            variables=variables_arr,
            targets=targets_arr,
            train_indices=np.asarray(problem.train_indices, dtype=np.intp),
            domain_policy=policy,
        )
        coefficients_arr = np.zeros(max(fit.coefficient_indices) + 1, dtype=np.float64)
        for index, value in zip(fit.coefficient_indices, fit.fitted_coefficients):
            coefficients_arr[index] = value
        fit_status = "fitted"
        fit_rank = fit.rank
        fit_condition_number = fit.condition_number
        fit_is_rank_deficient = fit.is_rank_deficient
        fit_is_ill_conditioned = fit.is_ill_conditioned
        solver_identity = fit.solver_identity

    splits = tuple(
        _score_split(node, split, indices, variables_arr, targets_arr, coefficients_arr, policy, fit_status)
        for split, indices in (
            ("train", problem.train_indices),
            ("validation", problem.validation_indices),
            ("extrapolation", problem.extrapolation_indices),
        )
        if indices
    )

    record = ExpressionEvidenceV1(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        problem_fingerprint=problem.fingerprint,
        pack_id=problem.pack_id,
        canonical_expr=canonical_expr,
        canonical_hash=canonical_hash,
        node_count=node_count,
        depth=depth,
        coefficient_count=coefficient_count,
        complexity_formula=COMPLEXITY_FORMULA,
        loss_function=LOSS_FUNCTION,
        domain_policy=policy,
        fit_status=fit_status,
        fit_rank=fit_rank,
        fit_condition_number=fit_condition_number,
        fit_is_rank_deficient=fit_is_rank_deficient,
        fit_is_ill_conditioned=fit_is_ill_conditioned,
        solver_identity=solver_identity,
        splits=splits,
        max_depth_budget=max_depth,
        max_nodes_budget=max_nodes,
        record_digest="",
    )
    return dataclasses.replace(record, record_digest=_evidence_digest(record))


def _front_digest(front: "ParetoFrontV1") -> str:
    payload = asdict(front)
    payload.pop("front_digest", None)
    return _digest(payload)


@dataclass(frozen=True)
class ParetoFrontV1:
    """Deterministic Pareto-front comparison artifact over a set of
    :class:`ExpressionEvidenceV1` records from one problem/pack/budget."""

    schema_version: str
    problem_fingerprint: str
    pack_id: str
    max_depth_budget: int
    max_nodes_budget: int
    member_hashes: tuple[str, ...]
    dominated_hashes: tuple[str, ...]
    front_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != PARETO_SCHEMA_VERSION:
            raise ValueError(f"unsupported pareto front schema: {self.schema_version!r}")
        if tuple(sorted(self.member_hashes)) != self.member_hashes:
            raise ValueError("member_hashes must be sorted")
        if tuple(sorted(self.dominated_hashes)) != self.dominated_hashes:
            raise ValueError("dominated_hashes must be sorted")
        if set(self.member_hashes) & set(self.dominated_hashes):
            raise ValueError("member_hashes and dominated_hashes must be disjoint")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ParetoFrontV1":
        if payload.get("schema_version") != PARETO_SCHEMA_VERSION:
            raise ValueError(f"unsupported pareto front schema: {payload.get('schema_version')!r}")
        front = cls(
            **{
                **payload,
                "member_hashes": tuple(payload["member_hashes"]),
                "dominated_hashes": tuple(payload["dominated_hashes"]),
            }
        )
        if _front_digest(front) != front.front_digest:
            raise ValueError("pareto front digest mismatch (tampered or corrupted payload)")
        return front


def front_integrity_ok(front: ParetoFrontV1) -> bool:
    return _front_digest(front) == front.front_digest


def _objectives(record: ExpressionEvidenceV1) -> tuple[float, ...]:
    loss_by_split = {s.split: s.loss for s in record.splits}
    return tuple(loss_by_split.get(name, math.inf) for name in SPLIT_NAMES) + (float(record.node_count),)


def _dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """``a`` dominates ``b``: at least as good on every objective (lower is
    better throughout -- loss and node_count are both minimized), strictly
    better on at least one. Equal vectors dominate neither direction, so
    ties are kept on the front rather than arbitrarily broken."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def pareto_front(records: Sequence[ExpressionEvidenceV1]) -> ParetoFrontV1:
    """Deterministic non-dominated front over ``records``.

    All records must share one ``problem_fingerprint``/``pack_id``/resource
    budget -- comparing evidence across different problems or budgets would
    silently discard the identity the comparison artifact is required to
    retain. Objectives are every split's loss (train/validation/extrapolation,
    ``+inf`` where a record has no such split -- always cancels out since
    every record from one problem shares the same split availability) plus
    ``node_count``, all minimized. Duplicate canonical expressions collapse
    to one member: identical canonical form implies identical evidence.
    """
    if not records:
        raise ValueError("pareto_front requires at least one evidence record")
    fingerprints = {r.problem_fingerprint for r in records}
    pack_ids = {r.pack_id for r in records}
    budgets = {(r.max_depth_budget, r.max_nodes_budget) for r in records}
    if len(fingerprints) != 1:
        raise ValueError("pareto_front requires evidence records from exactly one problem")
    if len(pack_ids) != 1:
        raise ValueError("pareto_front requires evidence records from exactly one pack")
    if len(budgets) != 1:
        raise ValueError("pareto_front requires evidence records sharing one resource budget")

    by_hash = {r.canonical_hash: r for r in records}
    objectives_by_hash = {h: _objectives(r) for h, r in by_hash.items()}
    hashes = sorted(by_hash)

    member_hashes: list[str] = []
    dominated_hashes: list[str] = []
    for h in hashes:
        obj = objectives_by_hash[h]
        dominated = any(_dominates(objectives_by_hash[other], obj) for other in hashes if other != h)
        (dominated_hashes if dominated else member_hashes).append(h)

    max_depth_budget, max_nodes_budget = next(iter(budgets))
    front = ParetoFrontV1(
        schema_version=PARETO_SCHEMA_VERSION,
        problem_fingerprint=next(iter(fingerprints)),
        pack_id=next(iter(pack_ids)),
        max_depth_budget=max_depth_budget,
        max_nodes_budget=max_nodes_budget,
        member_hashes=tuple(sorted(member_hashes)),
        dominated_hashes=tuple(sorted(dominated_hashes)),
        front_digest="",
    )
    return dataclasses.replace(front, front_digest=_front_digest(front))


__all__ = [
    "COMPLEXITY_FORMULA",
    "DEFAULT_DOMAIN_POLICY",
    "EVIDENCE_SCHEMA_VERSION",
    "FIT_STATUSES",
    "LOSS_FUNCTION",
    "PARETO_SCHEMA_VERSION",
    "SPLIT_NAMES",
    "ExpressionEvidenceV1",
    "ParetoFrontV1",
    "SplitEvidenceV1",
    "build_evidence_record",
    "evidence_integrity_ok",
    "front_integrity_ok",
    "pareto_front",
]
