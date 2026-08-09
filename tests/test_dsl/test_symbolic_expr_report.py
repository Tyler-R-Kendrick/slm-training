"""SRP-007 (SLM-466) regression tests: validity, fit, extrapolation,
complexity, and Pareto reporting over the typed symbolic-expression IR."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from slm_training.dsl.symbolic_expr_ir import CoefficientHole, OpApply, VarRef
from slm_training.dsl.symbolic_expr_report import (
    COMPLEXITY_FORMULA,
    LOSS_FUNCTION,
    ExpressionEvidenceV1,
    ParetoFrontV1,
    SplitEvidenceV1,
    build_evidence_record,
    evidence_integrity_ok,
    front_integrity_ok,
    pareto_front,
)
from slm_training.dsl.symbolic_regression_pack import SymbolicRegressionProblemV1


def _v(index: int) -> VarRef:
    return VarRef(index=index)


def _c(index: int) -> CoefficientHole:
    return CoefficientHole(index=index)


def _op(operator: str, *args) -> OpApply:
    return OpApply(operator=operator, args=tuple(args))


def _problem(**overrides: object) -> SymbolicRegressionProblemV1:
    fields = dict(
        variables=("x",),
        table=((1.0,), (2.0,), (3.0,), (4.0,), (5.0,), (6.0,), (9.0,), (10.0,)),
        targets=(5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 121.0, 123.0),  # 2x+3, rows 6/7 offset by +100
        allowed_operators=frozenset({"+", "*"}),
        coefficient_hole_policy="fit_free_coefficients",
        complexity_budget=10,
        numeric_domain_policy="reject_nonfinite",
        train_indices=(0, 1, 2, 3),
        validation_indices=(4, 5),
        extrapolation_indices=(6, 7),
    )
    fields.update(overrides)
    return SymbolicRegressionProblemV1(**fields)


def _synthetic_record(canonical_hash: str, train_loss: float, node_count: int) -> ExpressionEvidenceV1:
    return ExpressionEvidenceV1(
        schema_version="v1",
        problem_fingerprint="fp",
        pack_id="symbolic_regression",
        canonical_expr="v0",
        canonical_hash=canonical_hash,
        node_count=node_count,
        depth=1,
        coefficient_count=0,
        complexity_formula=COMPLEXITY_FORMULA,
        loss_function=LOSS_FUNCTION,
        domain_policy="reject_nonfinite",
        fit_status="no_coefficients",
        fit_rank=None,
        fit_condition_number=None,
        fit_is_rank_deficient=None,
        fit_is_ill_conditioned=None,
        solver_identity=None,
        splits=(SplitEvidenceV1(split="train", n_rows=1, n_valid_rows=1, loss=train_loss, failure_class="none"),),
        max_depth_budget=32,
        max_nodes_budget=256,
        record_digest="",
    )


# ---------------------------------------------------------------------------
# Complexity
# ---------------------------------------------------------------------------


def test_complexity_formula_is_explicit_and_matches_node_count():
    node = _op("+", _v(0), _v(1))
    problem = SymbolicRegressionProblemV1(
        variables=("x", "y"),
        table=((1.0, 2.0),),
        targets=(3.0,),
        allowed_operators=frozenset({"+"}),
        coefficient_hole_policy="fixed_constants_only",
        complexity_budget=10,
        numeric_domain_policy="reject_nonfinite",
        train_indices=(0,),
        validation_indices=(),
    )
    record = build_evidence_record(node, problem)
    assert record.complexity_formula == COMPLEXITY_FORMULA
    assert record.node_count == 3
    assert record.depth == 2
    assert record.coefficient_count == 0
    assert record.fit_status == "no_coefficients"


# ---------------------------------------------------------------------------
# Train/validation/extrapolation partitions reported separately
# ---------------------------------------------------------------------------


def test_splits_are_reported_separately_with_independent_loss():
    node = _op("+", _op("*", _c(0), _v(0)), _c(1))  # c0*x + c1
    problem = _problem()
    record = build_evidence_record(node, problem)

    assert record.fit_status == "fitted"
    assert record.fit_rank == 2
    by_split = {s.split: s for s in record.splits}
    assert set(by_split) == {"train", "validation", "extrapolation"}

    assert by_split["train"].loss == pytest.approx(0.0, abs=1e-8)
    assert by_split["validation"].loss == pytest.approx(0.0, abs=1e-8)
    # Extrapolation rows were deliberately offset by +100 -- large but finite.
    assert by_split["extrapolation"].loss > 1000.0
    assert math.isfinite(by_split["extrapolation"].loss)


def test_holdout_and_extrapolation_rows_never_influence_the_fit():
    node = _op("+", _op("*", _c(0), _v(0)), _c(1))
    problem_a = _problem()
    problem_b = _problem(
        targets=(5.0, 7.0, 9.0, 11.0, 999.0, -999.0, 5.0, 5.0)  # corrupt only non-train rows
    )

    record_a = build_evidence_record(node, problem_a)
    record_b = build_evidence_record(node, problem_b)

    assert record_a.fit_rank == record_b.fit_rank
    assert record_a.fit_condition_number == record_b.fit_condition_number
    train_a = next(s for s in record_a.splits if s.split == "train")
    train_b = next(s for s in record_b.splits if s.split == "train")
    assert train_a == train_b

    validation_a = next(s for s in record_a.splits if s.split == "validation")
    validation_b = next(s for s in record_b.splits if s.split == "validation")
    assert validation_a.loss != validation_b.loss


# ---------------------------------------------------------------------------
# Invalid numerical outputs cannot score as high quality
# ---------------------------------------------------------------------------


def test_domain_violation_forces_infinite_loss_not_a_partial_average():
    node = _op("log", _v(0))
    problem = SymbolicRegressionProblemV1(
        variables=("x",),
        table=((-1.0,), (1.0,), (1.0,)),  # row 0 is outside log's domain
        targets=(0.0, 0.0, 0.0),
        allowed_operators=frozenset({"log"}),
        coefficient_hole_policy="fixed_constants_only",
        complexity_budget=10,
        numeric_domain_policy="reject_nonfinite",
        train_indices=(0, 1, 2),
        validation_indices=(),
    )
    record = build_evidence_record(node, problem)
    train_split = next(s for s in record.splits if s.split == "train")
    assert train_split.failure_class == "partial"
    assert train_split.n_valid_rows == 2
    assert train_split.loss == math.inf  # never silently averaged over the 2 valid rows only


def test_non_affine_coefficient_expression_is_reported_not_silently_fit():
    node = _op("sin", _c(0))
    problem = SymbolicRegressionProblemV1(
        variables=("x",),
        table=((1.0,), (2.0,)),
        targets=(0.5, 0.9),
        allowed_operators=frozenset({"sin"}),
        coefficient_hole_policy="fit_free_coefficients",
        complexity_budget=10,
        numeric_domain_policy="reject_nonfinite",
        train_indices=(0, 1),
        validation_indices=(),
    )
    record = build_evidence_record(node, problem)
    assert record.fit_status == "non_affine"
    assert record.fit_rank is None
    train_split = next(s for s in record.splits if s.split == "train")
    assert train_split.loss == math.inf


# ---------------------------------------------------------------------------
# Pareto dominance/ties are deterministic
# ---------------------------------------------------------------------------


def test_pareto_front_known_dominated_and_nondominated_fixtures():
    a = _synthetic_record("hash-a", train_loss=0.0, node_count=5)
    b = _synthetic_record("hash-b", train_loss=1.0, node_count=3)  # tradeoff vs a: non-dominated
    c = _synthetic_record("hash-c", train_loss=2.0, node_count=5)  # dominated by a on both axes
    d = _synthetic_record("hash-d", train_loss=0.0, node_count=5)  # exact tie with a

    front = pareto_front([a, b, c, d])
    assert front.member_hashes == ("hash-a", "hash-b", "hash-d")
    assert front.dominated_hashes == ("hash-c",)


def test_pareto_front_requires_a_single_problem_pack_and_budget():
    a = _synthetic_record("hash-a", train_loss=0.0, node_count=1)
    b = replace(_synthetic_record("hash-b", train_loss=0.0, node_count=1), problem_fingerprint="other-fp")
    with pytest.raises(ValueError, match="exactly one problem"):
        pareto_front([a, b])


def test_pareto_front_retains_resource_budgets_and_problem_pack_identity():
    node = _op("+", _op("*", _c(0), _v(0)), _c(1))
    problem = _problem()
    record = build_evidence_record(node, problem, max_depth=16, max_nodes=64)
    front = pareto_front([record])
    assert front.problem_fingerprint == problem.fingerprint
    assert front.pack_id == problem.pack_id
    assert front.max_depth_budget == 16
    assert front.max_nodes_budget == 64
    assert front.member_hashes == (record.canonical_hash,)


# ---------------------------------------------------------------------------
# Determinism, round-trip, and tamper-evidence
# ---------------------------------------------------------------------------


def test_evidence_record_is_deterministic():
    node = _op("+", _op("*", _c(0), _v(0)), _c(1))
    problem = _problem()
    record_a = build_evidence_record(node, problem)
    record_b = build_evidence_record(node, problem)
    assert record_a == record_b


def test_evidence_record_round_trips_through_dict():
    node = _op("+", _op("*", _c(0), _v(0)), _c(1))
    problem = _problem()
    record = build_evidence_record(node, problem)
    restored = ExpressionEvidenceV1.from_dict(record.to_dict())
    assert restored == record


@pytest.mark.parametrize("field_name", ["node_count", "canonical_hash", "fit_status"])
def test_evidence_record_tamper_on_any_field_is_detected(field_name):
    node = _op("+", _op("*", _c(0), _v(0)), _c(1))
    problem = _problem()
    record = build_evidence_record(node, problem)
    if field_name == "node_count":
        tampered = replace(record, node_count=999)
    elif field_name == "canonical_hash":
        tampered = replace(record, canonical_hash="tampered")
    else:
        tampered = replace(record, fit_status="no_coefficients")
    assert not evidence_integrity_ok(tampered)
    with pytest.raises(ValueError, match="digest mismatch"):
        ExpressionEvidenceV1.from_dict(tampered.to_dict())


def test_pareto_front_round_trips_and_detects_tamper():
    node = _op("+", _op("*", _c(0), _v(0)), _c(1))
    problem = _problem()
    record = build_evidence_record(node, problem)
    front = pareto_front([record])

    restored = ParetoFrontV1.from_dict(front.to_dict())
    assert restored == front

    tampered = replace(front, max_depth_budget=999)
    assert not front_integrity_ok(tampered)
    with pytest.raises(ValueError, match="digest mismatch"):
        ParetoFrontV1.from_dict(tampered.to_dict())
