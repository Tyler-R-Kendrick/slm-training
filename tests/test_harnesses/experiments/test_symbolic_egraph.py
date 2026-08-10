"""RSP-004 (SLM-479) regression tests: the e-graph/equivalence-region
mechanism itself (EXP-SR-8)."""

from __future__ import annotations

import numpy as np

from slm_training.dsl.symbolic_expr_canonicalize import canonicalize_expr
from slm_training.dsl.symbolic_expr_ir import CoefficientHole, OpApply, VarRef, serialize_expr
from slm_training.harnesses.experiments.symbolic_egraph import (
    EGraph,
    saturate,
    verify_rewrite_numerically,
)


def _v(index: int) -> VarRef:
    return VarRef(index=index)


def _c(index: int) -> CoefficientHole:
    return CoefficientHole(index=index)


def _op(operator: str, *args) -> OpApply:
    return OpApply(operator=operator, args=tuple(args))


# ---------------------------------------------------------------------------
# Union-find / hashcons mechanics
# ---------------------------------------------------------------------------


def test_add_hashconses_structurally_identical_nodes_to_one_class():
    egraph = EGraph()
    a = egraph.add(_op("+", _v(0), _v(1)))
    b = egraph.add(_op("+", _v(0), _v(1)))
    assert egraph.find(a) == egraph.find(b)
    assert egraph.n_enodes() == 3  # v0, v1, one "+" node -- not duplicated


def test_distinct_expressions_start_in_distinct_classes():
    egraph = EGraph()
    a = egraph.add(_v(0))
    b = egraph.add(_v(1))
    assert egraph.find(a) != egraph.find(b)


def test_union_is_transitive_and_deterministic_root_tie_break():
    egraph = EGraph()
    a = egraph.add(_v(0))
    b = egraph.add(_v(1))
    c = egraph.add(_v(2))
    from slm_training.dsl.symbolic_expr_canonicalize import build_rewrite_certificate
    from slm_training.dsl.symbolic_expr_canonicalize import DOUBLE_NEGATION_ELIMINATION

    fake_cert = build_rewrite_certificate(
        DOUBLE_NEGATION_ELIMINATION, _v(0), _v(1), preconditions=("test",)
    )
    egraph.union(a, b, fake_cert)
    egraph.union(b, c, fake_cert)
    assert egraph.find(a) == egraph.find(b) == egraph.find(c)


# ---------------------------------------------------------------------------
# Saturation finds every known equivalence certified by SRP-005's rules
# ---------------------------------------------------------------------------


def test_saturation_discovers_commutative_equivalence():
    egraph = EGraph()
    left = egraph.add(_op("+", _v(0), _v(1)))
    right = egraph.add(_op("+", _v(1), _v(0)))
    assert egraph.find(left) != egraph.find(right)  # not yet merged
    rounds, applications = saturate(egraph)
    assert egraph.find(left) == egraph.find(right)
    assert rounds >= 1
    assert applications


def test_saturation_discovers_double_negation_equivalence():
    egraph = EGraph()
    plain = egraph.add(_v(0))
    double_neg = egraph.add(_op("neg", _op("neg", _v(0))))
    saturate(egraph)
    assert egraph.find(plain) == egraph.find(double_neg)


def test_saturation_discovers_double_abs_equivalence():
    egraph = EGraph()
    single = egraph.add(_op("abs", _v(0)))
    double = egraph.add(_op("abs", _op("abs", _v(0))))
    saturate(egraph)
    assert egraph.find(single) == egraph.find(double)


def test_saturation_handles_compound_equivalence_across_multiple_rules():
    # neg(neg(v0+v1)) == v1+v0 requires both double-negation elimination and
    # commutative sort -- exercises more than one saturation round.
    egraph = EGraph()
    compound = egraph.add(_op("neg", _op("neg", _op("+", _v(0), _v(1)))))
    simple = egraph.add(_op("+", _v(1), _v(0)))
    rounds, _ = saturate(egraph)
    assert egraph.find(compound) == egraph.find(simple)
    assert rounds >= 2


def test_saturation_never_merges_genuinely_distinct_expressions():
    egraph = EGraph()
    addition = egraph.add(_op("+", _v(0), _v(1)))
    multiplication = egraph.add(_op("*", _v(0), _v(1)))
    var0 = egraph.add(_v(0))
    var1 = egraph.add(_v(1))
    saturate(egraph)
    assert egraph.find(addition) != egraph.find(multiplication)
    assert egraph.find(var0) != egraph.find(var1)


def test_saturation_terminates_within_the_round_budget_on_a_fixed_point():
    egraph = EGraph()
    egraph.add(_op("+", _v(0), _v(1)))
    rounds, _ = saturate(egraph, max_rounds=64)
    assert rounds < 64  # converged before exhausting the budget


# ---------------------------------------------------------------------------
# Extraction is deterministic and picks the minimal representative
# ---------------------------------------------------------------------------


def test_extraction_picks_the_canonicalize_expr_minimal_form():
    egraph = EGraph()
    root = egraph.add(_op("neg", _op("neg", _op("+", _v(1), _v(0)))))
    saturate(egraph)
    extracted = egraph.extract(root)
    assert extracted == canonicalize_expr(_op("neg", _op("neg", _op("+", _v(1), _v(0)))))


def test_extraction_is_deterministic_regardless_of_which_member_was_added_first():
    egraph_a = EGraph()
    left_first = egraph_a.add(_op("+", _v(0), _v(1)))
    egraph_a.add(_op("+", _v(1), _v(0)))
    saturate(egraph_a)

    egraph_b = EGraph()
    egraph_b.add(_op("+", _v(1), _v(0)))
    right_first = egraph_b.add(_op("+", _v(0), _v(1)))
    saturate(egraph_b)

    assert serialize_expr(egraph_a.extract(left_first)) == serialize_expr(egraph_b.extract(right_first))


# ---------------------------------------------------------------------------
# Independent numerical re-verification (the EXP-SR-8 falsifier)
# ---------------------------------------------------------------------------


def test_verify_rewrite_numerically_passes_a_genuine_certified_rewrite():
    egraph = EGraph()
    egraph.add(_op("+", _v(0), _v(1)))
    egraph.add(_op("+", _v(1), _v(0)))
    _rounds, applications = saturate(egraph)
    assert applications

    variables = np.array([[1.0, 2.0], [3.0, -4.0], [0.0, 0.0]])
    for application in applications:
        verdict = verify_rewrite_numerically(
            application.before, application.after, application.certificate, variables=variables
        )
        assert verdict.passed
        assert verdict.is_certified
        assert verdict.is_numerically_equivalent


def test_verify_rewrite_numerically_catches_a_tampered_certificate():
    from dataclasses import replace

    egraph = EGraph()
    egraph.add(_op("+", _v(0), _v(1)))
    egraph.add(_op("+", _v(1), _v(0)))
    _rounds, applications = saturate(egraph)
    application = applications[0]
    tampered = replace(application.certificate, rule_version="v999")

    variables = np.array([[1.0, 2.0]])
    verdict = verify_rewrite_numerically(application.before, application.after, tampered, variables=variables)
    assert not verdict.is_certified
    assert not verdict.passed


def test_verify_rewrite_numerically_catches_a_non_equivalent_pair():
    egraph = EGraph()
    egraph.add(_op("+", _v(0), _v(1)))
    egraph.add(_op("+", _v(1), _v(0)))
    _rounds, applications = saturate(egraph)
    application = applications[0]

    variables = np.array([[1.0, 2.0], [5.0, -1.0]])
    # Swap in a genuinely different expression as the "after" side.
    verdict = verify_rewrite_numerically(
        application.before, _op("*", _v(0), _v(1)), application.certificate, variables=variables
    )
    assert not verdict.is_numerically_equivalent
    assert not verdict.passed


def test_verify_rewrite_numerically_is_nan_aware():
    # Both sides hit the same hard domain violation at row 0 -- a matching
    # NaN on both sides is equivalence, not a spurious mismatch.
    node = _op("/", _v(0), _v(1))
    from slm_training.dsl.symbolic_expr_canonicalize import (
        ADD_COMMUTATIVE_SORT,
        build_rewrite_certificate,
    )

    fake_cert = build_rewrite_certificate(ADD_COMMUTATIVE_SORT, node, node, preconditions=("identity",))
    variables = np.array([[1.0, 0.0], [2.0, 4.0]])
    verdict = verify_rewrite_numerically(node, node, fake_cert, variables=variables, domain_policy="allow_nonfinite")
    assert verdict.is_numerically_equivalent
