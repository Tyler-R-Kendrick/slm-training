"""SRP-005 (SLM-462) regression tests: canonicalization and the tamper-evident
rewrite-certificate schema over the typed symbolic-expression IR."""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

from tests.casefiles import case_values

from slm_training.dsl.symbolic_expr_canonicalize import (
    ADD_COMMUTATIVE_SORT,
    DOUBLE_NEGATION_ELIMINATION,
    MUL_COMMUTATIVE_SORT,
    REWRITE_RULES,
    RewriteCertificateV1,
    RewriteRuleV1,
    RewriteSideConditionError,
    apply_rewrite_rule,
    canonicalize_expr,
    canonicalize_with_certificates,
    certificate_integrity_ok,
    is_canonical,
)
from slm_training.dsl.symbolic_expr_evaluator import evaluate_vectorized
from slm_training.dsl.symbolic_expr_ir import BINARY_OPERATORS, UNARY_OPERATORS, OpApply, VarRef


def _v(index: int) -> VarRef:
    return VarRef(index=index)


def _op(operator: str, *args) -> OpApply:
    return OpApply(operator=operator, args=tuple(args))


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------


def test_rewrite_rules_are_declared_once_each_with_valid_versions():
    assert len({r.rule_id for r in REWRITE_RULES}) == len(REWRITE_RULES)
    for rule in REWRITE_RULES:
        assert rule.rule_version == "v1"


def test_rule_version_must_match_vN_pattern():
    with pytest.raises(ValueError, match="rule_version"):
        RewriteRuleV1(
            rule_id="x", rule_version="1", operator="+", commutes=True, idempotent=False, description="d"
        )
    with pytest.raises(ValueError, match="rule_version"):
        RewriteRuleV1(
            rule_id="x", rule_version="v0", operator="+", commutes=True, idempotent=False, description="d"
        )


# ---------------------------------------------------------------------------
# Individual rule semantics
# ---------------------------------------------------------------------------


def test_commutative_add_sort_orders_by_canonical_serialization():
    node = _op("+", _v(1), _v(0))
    canonical = canonicalize_expr(node)
    assert canonical == _op("+", _v(0), _v(1))


def test_commutative_mul_sort_orders_by_canonical_serialization():
    node = _op("*", _v(2), _v(0))
    canonical = canonicalize_expr(node)
    assert canonical == _op("*", _v(0), _v(2))


def test_already_sorted_commutative_node_is_unchanged():
    node = _op("+", _v(0), _v(1))
    canonical, certs = canonicalize_with_certificates(node)
    assert canonical == node
    assert certs == ()


def test_double_negation_elimination():
    node = _op("neg", _op("neg", _v(0)))
    assert canonicalize_expr(node) == _v(0)


def test_quadruple_negation_collapses_to_original():
    node = _op("neg", _op("neg", _op("neg", _op("neg", _v(0)))))
    assert canonicalize_expr(node) == _v(0)


def test_triple_negation_collapses_to_single_negation():
    node = _op("neg", _op("neg", _op("neg", _v(0))))
    assert canonicalize_expr(node) == _op("neg", _v(0))


def test_double_abs_elimination():
    node = _op("abs", _op("abs", _v(0)))
    assert canonicalize_expr(node) == _op("abs", _v(0))


def test_rewrites_apply_inside_nested_subtrees():
    node = _op("sin", _op("neg", _op("neg", _v(0))))
    assert canonicalize_expr(node) == _op("sin", _v(0))


# ---------------------------------------------------------------------------
# Side-condition failure (apply_rewrite_rule is the single-rule entry point)
# ---------------------------------------------------------------------------


def test_apply_rewrite_rule_raises_on_structural_mismatch_wrong_operator():
    node = _op("-", _v(0), _v(1))
    with pytest.raises(RewriteSideConditionError, match="commutative_operand_sort_add"):
        apply_rewrite_rule(ADD_COMMUTATIVE_SORT, node)


def test_apply_rewrite_rule_raises_when_already_canonical():
    node = _op("+", _v(0), _v(1))
    with pytest.raises(RewriteSideConditionError):
        apply_rewrite_rule(ADD_COMMUTATIVE_SORT, node)


def test_apply_rewrite_rule_raises_on_non_matching_leaf():
    with pytest.raises(RewriteSideConditionError, match="double_negation_elimination"):
        apply_rewrite_rule(DOUBLE_NEGATION_ELIMINATION, _v(0))


def test_apply_rewrite_rule_rejects_unknown_rule():
    fake_rule = RewriteRuleV1(
        rule_id="not_registered",
        rule_version="v1",
        operator="+",
        commutes=True,
        idempotent=False,
        description="not in REWRITE_RULES",
    )
    with pytest.raises(ValueError, match="unknown rewrite rule"):
        apply_rewrite_rule(fake_rule, _op("+", _v(0), _v(1)))


def test_apply_rewrite_rule_succeeds_and_returns_certificate():
    node = _op("*", _v(1), _v(0))
    result, cert = apply_rewrite_rule(MUL_COMMUTATIVE_SORT, node)
    assert result == _op("*", _v(0), _v(1))
    assert cert.rule_id == "commutative_operand_sort_mul"
    assert cert.preconditions == ("operator == '*'",)
    assert certificate_integrity_ok(cert)


# ---------------------------------------------------------------------------
# Determinism / idempotence
# ---------------------------------------------------------------------------


def test_canonicalize_is_idempotent():
    node = _op("+", _op("neg", _op("neg", _v(1))), _v(0))
    once = canonicalize_expr(node)
    twice = canonicalize_expr(once)
    assert once == twice
    assert is_canonical(once)


def test_canonicalizing_twice_produces_no_further_certificates():
    node = _op("+", _v(1), _v(0))
    canonical, _first_certs = canonicalize_with_certificates(node)
    _canonical_again, second_certs = canonicalize_with_certificates(canonical)
    assert second_certs == ()


def test_random_trees_canonicalize_deterministically_and_idempotently():
    rng = random.Random(7)

    def random_tree(depth: int):
        if depth <= 0 or rng.random() < 0.4:
            return _v(rng.randrange(3))
        operator = rng.choice(sorted(BINARY_OPERATORS | UNARY_OPERATORS))
        arity = 2 if operator in BINARY_OPERATORS else 1
        return _op(operator, *(random_tree(depth - 1) for _ in range(arity)))

    for _ in range(30):
        node = random_tree(4)
        first = canonicalize_expr(node)
        second = canonicalize_expr(node)
        assert first == second
        assert canonicalize_expr(first) == first


# ---------------------------------------------------------------------------
# Certificate tamper-evidence and replay
# ---------------------------------------------------------------------------


def test_certificate_round_trips_through_dict():
    node = _op("+", _v(1), _v(0))
    _canonical, certs = canonicalize_with_certificates(node)
    assert len(certs) == 1
    payload = certs[0].to_dict()
    restored = RewriteCertificateV1.from_dict(payload)
    assert restored == certs[0]


def test_certificate_rejects_unsupported_schema_version():
    node = _op("+", _v(1), _v(0))
    _canonical, certs = canonicalize_with_certificates(node)
    payload = {**certs[0].to_dict(), "schema_version": "v99"}
    with pytest.raises(ValueError, match="unsupported rewrite certificate schema"):
        RewriteCertificateV1.from_dict(payload)


@pytest.mark.parametrize(
    "field",
    case_values(__file__, "test_certificate_tamper_on_any_field_is_detected"),
)
def test_certificate_tamper_on_any_field_is_detected(field):
    node = _op("+", _v(1), _v(0))
    _canonical, certs = canonicalize_with_certificates(node)
    cert = certs[0]
    tampered = replace(cert, **{field: cert.__getattribute__(field) + "_tampered"})
    assert not certificate_integrity_ok(tampered)
    with pytest.raises(ValueError, match="digest mismatch"):
        RewriteCertificateV1.from_dict(tampered.to_dict())


def test_certificate_chain_is_replayable_through_recorded_hashes():
    node = _op("+", _op("neg", _op("neg", _v(1))), _v(0))
    canonical, certs = canonicalize_with_certificates(node)
    assert len(certs) >= 1
    for cert in certs:
        assert certificate_integrity_ok(cert)
    # The final certificate's output hash matches the returned canonical form.
    from slm_training.dsl.symbolic_expr_ir import serialize_expr
    import hashlib

    assert certs[-1].output_canonical_hash == hashlib.sha256(
        serialize_expr(canonical).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Structural fail-closed behavior
# ---------------------------------------------------------------------------


def test_unsupported_node_type_fails_closed():
    with pytest.raises(TypeError, match="unsupported symbolic expression node type"):
        canonicalize_expr("not-a-node")  # type: ignore[arg-type]


def test_expression_over_budget_raises():
    node = _v(0)
    for _ in range(40):
        node = _op("neg", node)
    with pytest.raises(ValueError, match="max_depth"):
        canonicalize_expr(node)


# ---------------------------------------------------------------------------
# Semantic differential tests: canonicalization must never change the
# expression's evaluated numerical semantics.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "node",
    [
        _op("+", _v(1), _v(0)),
        _op("*", _v(1), _v(0)),
        _op("neg", _op("neg", _v(0))),
        _op("abs", _op("abs", _v(0))),
        _op("-", _op("+", _v(1), _v(0)), _op("neg", _op("neg", _v(1)))),
        _op("sin", _op("*", _v(1), _v(0))),
    ],
)
def test_canonicalization_preserves_evaluated_predictions_bit_for_bit(node):
    rng = random.Random(11)
    rows = [(rng.uniform(-5, 5), rng.uniform(-5, 5)) for _ in range(15)]
    variables = np.array(rows)
    canonical = canonicalize_expr(node)

    before = evaluate_vectorized(node, variables=variables, domain_policy="allow_nonfinite")
    after = evaluate_vectorized(canonical, variables=variables, domain_policy="allow_nonfinite")

    assert np.array_equal(before.predictions, after.predictions, equal_nan=True)
    assert before.valid_mask == after.valid_mask
