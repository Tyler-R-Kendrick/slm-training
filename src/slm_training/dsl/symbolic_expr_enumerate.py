"""SRP-009 (SLM-472): bounded canonical enumerative symbolic-regression baseline.

A transparent, model-free exact/bounded baseline: enumerate every legal
:mod:`symbolic_expr_ir` AST up to declared operator/depth/node budgets,
deduplicate by the certified :func:`symbolic_expr_canonicalize.canonicalize_expr`
identity *before* any evaluation, then score each unique candidate through
:func:`symbolic_expr_report.build_evidence_record` (which itself composes
:mod:`symbolic_expr_evaluator` and :mod:`symbolic_expr_fit`) and extract a
:func:`symbolic_expr_report.pareto_front`.

No new evaluation, fitting, canonicalization, or reporting logic is added
here -- this module only enumerates candidate structures and counts.

Divergence note (same precedent as SRP-004/SRP-005/SRP-007): the
pre-registered ``downstream_extension_map`` row for SRP-009 predicted
``valid_state_search_generic`` (the fixed-hole finite-domain CSP controller,
``dsl/solver/controller.py``) as the search backbone. That controller's
state model is a *fixed* set of holes each drawn from a finite domain,
solved via backtracking + certificate-backed closure -- it does not
naturally represent "generate a variable-shaped tree up to a depth/node
budget" without an artificial fixed-arity template encoding that would add
real complexity without a matching correctness benefit here, especially
since this issue's core acceptance criterion (tiny fixtures must match an
*independent* brute-force oracle exactly) is best served by a small,
obviously-correct, directly-auditable recursive enumerator over the
grammar that :mod:`symbolic_expr_ir` already declares (its own arity table
and depth/node budgets are the natural generative model). The divergence is
documented in ``ownership_map.json``.

Never claims a global optimum when the enumeration bound is incomplete:
``optimality_claim`` is fail-closed to ``"unknown_bound_truncated"`` any
time the declared ``max_states`` cap is hit before the search space is
naturally exhausted -- enforced structurally, not just by convention (see
:class:`EnumerativeBaselineResultV1.__post_init__`).
"""

from __future__ import annotations

import hashlib
import time
import tracemalloc
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Iterator

from slm_training.dsl.symbolic_expr_canonicalize import canonicalize_expr
from slm_training.dsl.symbolic_expr_ir import (
    CoefficientHole,
    OpApply,
    SymbolicExprNode,
    VarRef,
    _OPERATOR_ARITY,
    serialize_expr,
)
from slm_training.dsl.symbolic_expr_report import (
    ExpressionEvidenceV1,
    ParetoFrontV1,
    build_evidence_record,
    pareto_front,
)

if TYPE_CHECKING:
    from slm_training.dsl.symbolic_regression_pack import SymbolicRegressionProblemV1

ENUMERATION_SCHEMA_VERSION = "v1"
OPTIMALITY_CLAIMS = frozenset({"exhaustive_under_declared_bounds", "unknown_bound_truncated"})

__all__ = [
    "ENUMERATION_SCHEMA_VERSION",
    "OPTIMALITY_CLAIMS",
    "EnumerationCountersV1",
    "EnumerativeBaselineResultV1",
    "generate_all_expressions",
    "run_enumerative_baseline",
]


def _terminal_candidates(num_variables: int, *, allow_coefficient: bool) -> list[SymbolicExprNode]:
    terminals: list[SymbolicExprNode] = [VarRef(index=i) for i in range(num_variables)]
    if allow_coefficient:
        terminals.append(CoefficientHole(index=0))  # placeholder; renumbered canonically below
    return terminals


def _generate_exact(
    depth_budget: int,
    size: int,
    *,
    num_variables: int,
    operators: frozenset[str],
    allow_coefficient: bool,
) -> Iterator[SymbolicExprNode]:
    """Every tree with exactly ``size`` nodes and tree-depth <= ``depth_budget``."""
    if size == 1:
        yield from _terminal_candidates(num_variables, allow_coefficient=allow_coefficient)
        return
    if depth_budget < 2:
        return
    remaining = size - 1
    for operator in sorted(operators):
        arity = _OPERATOR_ARITY[operator]
        if arity == 1:
            if remaining < 1:
                continue
            for child in _generate_exact(
                depth_budget - 1,
                remaining,
                num_variables=num_variables,
                operators=operators,
                allow_coefficient=allow_coefficient,
            ):
                yield OpApply(operator=operator, args=(child,))
        elif arity == 2:
            for left_size in range(1, remaining):
                right_size = remaining - left_size
                if right_size < 1:
                    continue
                for left in _generate_exact(
                    depth_budget - 1,
                    left_size,
                    num_variables=num_variables,
                    operators=operators,
                    allow_coefficient=allow_coefficient,
                ):
                    for right in _generate_exact(
                        depth_budget - 1,
                        right_size,
                        num_variables=num_variables,
                        operators=operators,
                        allow_coefficient=allow_coefficient,
                    ):
                        yield OpApply(operator=operator, args=(left, right))
        else:
            raise ValueError(f"unsupported operator arity for enumeration: {arity}")


def generate_all_expressions(
    *,
    max_depth: int,
    max_nodes: int,
    num_variables: int,
    operators: frozenset[str],
    allow_coefficient: bool = True,
) -> Iterator[SymbolicExprNode]:
    """Every legal AST with node_count in ``[1, max_nodes]`` and depth
    ``<= max_depth`` over ``operators``/``num_variables`` -- the brute-force
    grammar enumeration this baseline scores. Each raw shape is generated
    exactly once (no combinatorial re-derivation of smaller subtrees at
    multiple sizes); semantically-equivalent-but-differently-shaped trees
    (e.g. operand order under a commutative operator) are intentionally
    still both generated here -- collapsing those is canonicalization's
    job, applied by the caller, never guessed at during generation."""
    if num_variables < 1:
        raise ValueError("num_variables must be positive")
    if max_depth < 1 or max_nodes < 1:
        raise ValueError("max_depth and max_nodes must be positive")
    for size in range(1, max_nodes + 1):
        yield from _generate_exact(
            max_depth,
            size,
            num_variables=num_variables,
            operators=operators,
            allow_coefficient=allow_coefficient,
        )


def _renumber_coefficients(node: SymbolicExprNode) -> SymbolicExprNode:
    """Assign each :class:`CoefficientHole` a fresh, sequential, left-to-right
    (pre-order) index. Raw generation reuses a single placeholder index
    (``0``) for every coefficient terminal; two distinct coefficient sites
    in one expression are two independent free coefficients, not a shared
    value, so they must not collide before canonical identity is computed."""
    counter = 0

    def walk(current: SymbolicExprNode) -> SymbolicExprNode:
        nonlocal counter
        if isinstance(current, VarRef):
            return current
        if isinstance(current, CoefficientHole):
            index = counter
            counter += 1
            return CoefficientHole(index=index)
        if isinstance(current, OpApply):
            return OpApply(operator=current.operator, args=tuple(walk(arg) for arg in current.args))
        raise TypeError(f"unsupported symbolic expression node type: {type(current).__name__}")

    return walk(node)


@dataclass(frozen=True)
class EnumerationCountersV1:
    """Resource accounting for one enumerative baseline run -- field names
    chosen to stay comparable with the repository's established
    ``compute: {forwards, verifier_calls, wall_ms}`` harness convention
    (AC: "resource accounting is compatible with later EXP-SR comparisons")."""

    states_generated: int
    unique_canonical_states: int
    duplicate_eliminations: int
    evaluation_count: int
    wall_ms: float
    peak_memory_bytes: int
    bound_exhausted: bool
    max_states_cap: int

    def __post_init__(self) -> None:
        if self.states_generated < 0 or self.unique_canonical_states < 0:
            raise ValueError("counters must be non-negative")
        if self.duplicate_eliminations < 0 or self.evaluation_count < 0:
            raise ValueError("counters must be non-negative")
        if self.unique_canonical_states + self.duplicate_eliminations != self.states_generated:
            raise ValueError(
                "unique_canonical_states + duplicate_eliminations must equal states_generated"
            )
        if self.evaluation_count != self.unique_canonical_states:
            raise ValueError("every unique canonical state is evaluated exactly once")
        if self.max_states_cap < 1:
            raise ValueError("max_states_cap must be positive")
        if self.states_generated > self.max_states_cap:
            raise ValueError("states_generated cannot exceed the declared max_states cap")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def compute(self) -> dict[str, Any]:
        """The cross-harness ``compute`` shape (no model: ``forwards=0``)."""
        return {
            "forwards": 0,
            "verifier_calls": self.evaluation_count,
            "wall_ms": self.wall_ms,
        }


@dataclass(frozen=True)
class EnumerativeBaselineResultV1:
    """Full evidence for one bounded enumerative baseline run."""

    schema_version: str
    problem_fingerprint: str
    pack_id: str
    max_depth_budget: int
    max_nodes_budget: int
    counters: EnumerationCountersV1
    optimality_claim: str
    pareto_front: ParetoFrontV1 | None
    records: tuple[ExpressionEvidenceV1, ...]

    def __post_init__(self) -> None:
        if self.schema_version != ENUMERATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported enumeration schema: {self.schema_version!r}")
        if self.optimality_claim not in OPTIMALITY_CLAIMS:
            raise ValueError(f"unsupported optimality_claim: {self.optimality_claim!r}")
        # AC: never claim a global optimum when the enumeration bound is
        # incomplete -- fail closed, not just by convention.
        if not self.counters.bound_exhausted and self.optimality_claim != "unknown_bound_truncated":
            raise ValueError(
                "a truncated enumeration must declare optimality_claim='unknown_bound_truncated'"
            )
        if self.counters.bound_exhausted and self.optimality_claim != "exhaustive_under_declared_bounds":
            raise ValueError(
                "an exhausted enumeration must declare optimality_claim='exhaustive_under_declared_bounds'"
            )
        if len(self.records) != self.counters.unique_canonical_states:
            raise ValueError("records count must equal counters.unique_canonical_states")
        if self.records and self.pareto_front is None:
            raise ValueError("a non-empty record set requires a pareto front")
        if not self.records and self.pareto_front is not None:
            raise ValueError("an empty record set cannot have a pareto front")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "problem_fingerprint": self.problem_fingerprint,
            "pack_id": self.pack_id,
            "max_depth_budget": self.max_depth_budget,
            "max_nodes_budget": self.max_nodes_budget,
            "counters": self.counters.to_dict(),
            "optimality_claim": self.optimality_claim,
            "pareto_front": self.pareto_front.to_dict() if self.pareto_front else None,
            "records": [record.to_dict() for record in self.records],
        }


def run_enumerative_baseline(
    problem: "SymbolicRegressionProblemV1",
    *,
    max_depth: int,
    max_nodes: int,
    max_states: int = 10_000,
) -> EnumerativeBaselineResultV1:
    """Enumerate, dedup, fit, and score every legal expression up to the
    declared budgets, in canonical generation order.

    Uses no target/model information during the search itself -- structural
    generation never inspects ``problem.targets``; coefficients are fit only
    on ``problem.train_indices`` inside :func:`build_evidence_record` (AC:
    "no model, target-equation oracle, or held-out target during search
    beyond declared train fitness").
    """
    if max_depth < 1 or max_nodes < 1:
        raise ValueError("max_depth and max_nodes must be positive")
    if max_states < 1:
        raise ValueError("max_states must be positive")

    start = time.perf_counter()
    tracemalloc.start()
    try:
        seen_hashes: set[str] = set()
        records: list[ExpressionEvidenceV1] = []
        states_generated = 0
        duplicate_eliminations = 0
        bound_exhausted = True

        for raw in generate_all_expressions(
            max_depth=max_depth,
            max_nodes=max_nodes,
            num_variables=len(problem.variables),
            operators=problem.allowed_operators,
        ):
            if states_generated >= max_states:
                bound_exhausted = False
                break
            states_generated += 1
            canonical = canonicalize_expr(_renumber_coefficients(raw))
            canonical_hash = hashlib.sha256(
                serialize_expr(canonical).encode("utf-8")
            ).hexdigest()
            if canonical_hash in seen_hashes:
                duplicate_eliminations += 1
                continue
            seen_hashes.add(canonical_hash)
            records.append(
                build_evidence_record(canonical, problem, max_depth=max_depth, max_nodes=max_nodes)
            )
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    wall_ms = (time.perf_counter() - start) * 1000.0

    counters = EnumerationCountersV1(
        states_generated=states_generated,
        unique_canonical_states=len(records),
        duplicate_eliminations=duplicate_eliminations,
        evaluation_count=len(records),
        wall_ms=round(wall_ms, 3),
        peak_memory_bytes=peak_memory_bytes,
        bound_exhausted=bound_exhausted,
        max_states_cap=max_states,
    )
    return EnumerativeBaselineResultV1(
        schema_version=ENUMERATION_SCHEMA_VERSION,
        problem_fingerprint=problem.fingerprint,
        pack_id=problem.pack_id,
        max_depth_budget=max_depth,
        max_nodes_budget=max_nodes,
        counters=counters,
        optimality_claim=(
            "exhaustive_under_declared_bounds" if bound_exhausted else "unknown_bound_truncated"
        ),
        pareto_front=pareto_front(records) if records else None,
        records=tuple(records),
    )
