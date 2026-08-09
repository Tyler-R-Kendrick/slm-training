"""SRP-009 (SLM-472) regression tests: bounded canonical enumerative
symbolic-regression baseline.

Drives the shipped ``symbolic_expr_enumerate`` entry points -- and, through
them, the real SRP-002/005/007 IR/canonicalization/reporting -- rather than
reimplementing any of them. The brute-force oracle in
``test_tiny_fixture_enumeration_matches_an_independent_brute_force_oracle``
is deliberately written from scratch (not calling
``generate_all_expressions``) so it is a genuinely independent check.
"""

from __future__ import annotations

import hashlib
import itertools

import pytest

from slm_training.dsl.symbolic_expr_canonicalize import canonicalize_expr
from slm_training.dsl.symbolic_expr_ir import CoefficientHole, OpApply, VarRef, serialize_expr
from slm_training.dsl.symbolic_expr_enumerate import (
    OPTIMALITY_CLAIMS,
    generate_all_expressions,
    run_enumerative_baseline,
)
from slm_training.dsl.symbolic_regression_pack import SymbolicRegressionProblemV1


def _problem(**overrides: object) -> SymbolicRegressionProblemV1:
    fields = dict(
        variables=("x",),
        table=((0.0,), (1.0,), (2.0,), (3.0,)),
        targets=(1.0, 3.0, 5.0, 7.0),  # y = 2x + 1
        allowed_operators=frozenset({"+", "*"}),
        coefficient_hole_policy="fit_free_coefficients",
        complexity_budget=10,
        numeric_domain_policy="reject_nonfinite",
        train_indices=(0, 1, 2),
        validation_indices=(3,),
    )
    fields.update(overrides)
    return SymbolicRegressionProblemV1(**fields)


def test_generate_all_expressions_rejects_invalid_budgets() -> None:
    with pytest.raises(ValueError, match="num_variables"):
        list(generate_all_expressions(max_depth=2, max_nodes=2, num_variables=0, operators=frozenset({"+"})))
    with pytest.raises(ValueError, match="max_depth and max_nodes"):
        list(generate_all_expressions(max_depth=0, max_nodes=2, num_variables=1, operators=frozenset({"+"})))


def test_tiny_fixture_enumeration_matches_an_independent_brute_force_oracle() -> None:
    """AC: small finite fixtures enumerate all canonical expressions and
    match an independent brute-force oracle.

    This oracle is written from scratch: it directly nests Python loops
    over terminal/operator choices for a 1-variable, {+, *}, max_nodes=3,
    max_depth=2 grammar and canonicalizes with the same certified
    ``canonicalize_expr`` (reusing the canonicalizer under test is
    appropriate -- it is the certified authority on equivalence; the
    *enumeration/counting* logic is what must be independent)."""
    operators = ("+", "*")

    terminals = [VarRef(index=0), CoefficientHole(index=0)]
    oracle_hashes: set[str] = set()
    # size-1 trees
    for term in terminals:
        canonical = canonicalize_expr(term)
        oracle_hashes.add(hashlib.sha256(serialize_expr(canonical).encode("utf-8")).hexdigest())
    # size-3 trees: one binary operator over two size-1 subtrees, with each
    # coefficient site independently renumbered (0, 1, ...) in appearance order.
    for operator in operators:
        for left_kind, right_kind in itertools.product(("var", "coef"), repeat=2):
            counter = itertools.count()
            left = VarRef(index=0) if left_kind == "var" else CoefficientHole(index=next(counter))
            right = VarRef(index=0) if right_kind == "var" else CoefficientHole(index=next(counter))
            tree = OpApply(operator=operator, args=(left, right))
            canonical = canonicalize_expr(tree)
            oracle_hashes.add(hashlib.sha256(serialize_expr(canonical).encode("utf-8")).hexdigest())

    problem = _problem(allowed_operators=frozenset(operators))
    result = run_enumerative_baseline(problem, max_depth=2, max_nodes=3, max_states=1_000)

    assert result.optimality_claim == "exhaustive_under_declared_bounds"
    produced_hashes = {record.canonical_hash for record in result.records}
    assert produced_hashes == oracle_hashes


def test_run_enumerative_baseline_deduplicates_before_evaluation() -> None:
    problem = _problem()
    result = run_enumerative_baseline(problem, max_depth=3, max_nodes=3, max_states=1_000)

    assert result.counters.states_generated == 10  # 2 terminals + 2 ops * 4 binary combos
    assert result.counters.unique_canonical_states == len(result.records)
    assert result.counters.duplicate_eliminations > 0
    assert (
        result.counters.unique_canonical_states + result.counters.duplicate_eliminations
        == result.counters.states_generated
    )
    # AC: dedup happens before evaluation -- evaluation_count is exactly the
    # unique count, never the raw generated count.
    assert result.counters.evaluation_count == result.counters.unique_canonical_states
    canonical_hashes = [record.canonical_hash for record in result.records]
    assert len(canonical_hashes) == len(set(canonical_hashes))


def test_canonical_dedup_never_collapses_nonequivalent_expressions() -> None:
    """AC: canonical dedup cannot discard nonequivalent expressions under
    the certified rules -- (+ v0 v0) and (* v0 v0) must never collide."""
    problem = _problem()
    result = run_enumerative_baseline(problem, max_depth=3, max_nodes=3, max_states=1_000)
    canonical_exprs = {record.canonical_expr for record in result.records}
    assert "(+ v0 v0)" in canonical_exprs
    assert "(* v0 v0)" in canonical_exprs


def test_bound_exhaustion_is_explicit_and_never_claims_optimality_when_truncated() -> None:
    """AC: bound exhaustion is explicit and propagates UNKNOWN/incomplete
    optimality -- a tiny max_states cap must truncate honestly, never
    silently pretend the enumeration was exhaustive."""
    problem = _problem()
    full = run_enumerative_baseline(problem, max_depth=3, max_nodes=3, max_states=1_000)
    assert full.optimality_claim == "exhaustive_under_declared_bounds"
    assert full.counters.bound_exhausted is True

    truncated = run_enumerative_baseline(problem, max_depth=3, max_nodes=3, max_states=3)
    assert truncated.optimality_claim == "unknown_bound_truncated"
    assert truncated.counters.bound_exhausted is False
    assert truncated.counters.states_generated == 3
    assert truncated.optimality_claim in OPTIMALITY_CLAIMS


def test_result_invariants_reject_a_mismatched_optimality_claim() -> None:
    """The dataclass itself, not just the runner, refuses to construct a
    result claiming exhaustive optimality while bound_exhausted=False."""
    from slm_training.dsl.symbolic_expr_enumerate import (
        EnumerationCountersV1,
        EnumerativeBaselineResultV1,
    )

    counters = EnumerationCountersV1(
        states_generated=3,
        unique_canonical_states=3,
        duplicate_eliminations=0,
        evaluation_count=3,
        wall_ms=1.0,
        peak_memory_bytes=100,
        bound_exhausted=False,
        max_states_cap=3,
    )
    with pytest.raises(ValueError, match="unknown_bound_truncated"):
        EnumerativeBaselineResultV1(
            schema_version="v1",
            problem_fingerprint="fp",
            pack_id="symbolic_regression",
            max_depth_budget=3,
            max_nodes_budget=3,
            counters=counters,
            optimality_claim="exhaustive_under_declared_bounds",
            pareto_front=None,
            records=(),
        )


def test_baseline_never_reads_targets_during_search_only_at_fit_and_score_time() -> None:
    """AC: no model, target-equation oracle, or held-out target during
    search beyond declared train fitness -- ``generate_all_expressions``
    (the search) takes no problem/target argument at all; only
    ``run_enumerative_baseline``'s post-generation scoring step (via
    ``build_evidence_record``) touches ``problem.targets``, and even then
    coefficients are fit on ``train_indices`` only."""
    import inspect

    signature = inspect.signature(generate_all_expressions)
    assert "problem" not in signature.parameters
    assert "targets" not in signature.parameters

    problem = _problem()
    result = run_enumerative_baseline(problem, max_depth=3, max_nodes=3, max_states=1_000)
    for record in result.records:
        if record.fit_status == "fitted":
            train_split = next(s for s in record.splits if s.split == "train")
            validation_split = next((s for s in record.splits if s.split == "validation"), None)
            assert train_split.n_rows == len(problem.train_indices)
            if validation_split is not None:
                assert validation_split.n_rows == len(problem.validation_indices)


def test_run_enumerative_baseline_is_deterministic_on_repeat() -> None:
    problem = _problem()
    first = run_enumerative_baseline(problem, max_depth=3, max_nodes=3, max_states=1_000)
    second = run_enumerative_baseline(problem, max_depth=3, max_nodes=3, max_states=1_000)

    assert first.counters.states_generated == second.counters.states_generated
    assert first.counters.unique_canonical_states == second.counters.unique_canonical_states
    assert [r.canonical_hash for r in first.records] == [r.canonical_hash for r in second.records]
    assert first.pareto_front.member_hashes == second.pareto_front.member_hashes


def test_pareto_front_and_compute_shape_are_present() -> None:
    problem = _problem()
    result = run_enumerative_baseline(problem, max_depth=3, max_nodes=3, max_states=1_000)
    assert result.pareto_front is not None
    assert set(result.pareto_front.member_hashes) <= {r.canonical_hash for r in result.records}
    assert set(result.counters.compute) == {"forwards", "verifier_calls", "wall_ms"}
    assert result.counters.compute["forwards"] == 0
