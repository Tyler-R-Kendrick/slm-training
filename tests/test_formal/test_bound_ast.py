"""EVID-04 / SLM-525: safe symbolic bound AST + exact evaluator."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from slm_training.formal import bound_ast as ba
from slm_training.formal.bound_ast import (
    BOUND_AST_SCHEMA,
    BOUND_CLOSURE_LIVE_UPPER,
    BOUND_FINITE_SEARCH_PREFIX_TREE,
    BOUND_PLACEHOLDER_PENDING,
    REGISTERED_BOUND_AST_IDS,
    Add,
    BoundAstDocumentV1,
    BoundAstError,
    BoundVarDeclV1,
    Ceil,
    ConstInt,
    ConstRat,
    Div,
    Floor,
    ProdOver,
    SumOver,
    Var,
    assert_registered_bound_ast_id,
    evaluate,
    evaluate_bound,
    evaluate_predicate,
    load_bound_ast_registry,
    load_parity_fixtures,
    pretty,
    resolve_bound_ast,
    round_trip,
)
from slm_training.formal.finite_search_bounds import (
    BOUND_AST_ID,
    build_finite_search_bound,
)

_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY = _ROOT / "src/slm_training/resources/formal/bound_ast_registry.v1.json"


def test_registry_ids_cover_evid03_placeholders() -> None:
    assert BOUND_AST_ID == BOUND_FINITE_SEARCH_PREFIX_TREE
    assert BOUND_PLACEHOLDER_PENDING in REGISTERED_BOUND_AST_IDS
    registry = load_bound_ast_registry()
    assert set(registry) == set(REGISTERED_BOUND_AST_IDS)
    for bound_id in REGISTERED_BOUND_AST_IDS:
        assert_registered_bound_ast_id(bound_id)


def test_prefix_tree_matches_kern03_total_query_bound() -> None:
    evidence = build_finite_search_bound([2, 3])
    result = evaluate_bound(
        BOUND_FINITE_SEARCH_PREFIX_TREE,
        {"domain_sizes": [2, 3], "c_exp": 1, "c_ver": 1},
    )
    assert result.is_integer
    assert int(result.fraction) == evidence.total_query_bound == 15
    assert result.u64_overflow == "fits_u64"


def test_canonical_round_trip_and_mutation_changes_digest() -> None:
    doc = resolve_bound_ast(BOUND_FINITE_SEARCH_PREFIX_TREE)
    assert doc.schema_version == BOUND_AST_SCHEMA
    again = round_trip(doc)
    assert again.digest() == doc.digest()
    mutated = doc.model_copy(
        update={"expression": Add(args=(ConstInt(value=1), ConstInt(value=1)))}
    )
    assert mutated.digest() != doc.digest()


def test_rejects_unknown_variable_div_zero_codelike_and_unsupported() -> None:
    variables = (
        BoundVarDeclV1(
            name="x",
            kind="scalar",
            domain="nat",
            unit="query",
            meaning="scalar workload query count",
        ),
    )
    with pytest.raises((BoundAstError, ValidationError), match="unknown variable"):
        BoundAstDocumentV1(
            bound_ast_id="bound.test.unknown.v1",
            meaning="document used only for rejection tests in unit suite",
            variables=variables,
            expression=Var(name="y"),
        )
    with pytest.raises(BoundAstError, match="division by zero"):
        evaluate(
            Div(numerator=ConstInt(value=1), denominator=ConstInt(value=0)),
            variables=(),
            env={},
        )
    with pytest.raises((BoundAstError, ValidationError), match="code-like"):
        BoundVarDeclV1(
            name="x",
            kind="scalar",
            domain="nat",
            unit="query",
            meaning="calls eval on attacker input",
        )
    with pytest.raises(BoundAstError, match="not a registered"):
        assert_registered_bound_ast_id("bound.not.registered.v1")


def test_exact_rational_floor_ceil_no_float() -> None:
    half = ConstRat(numerator=1, denominator=2)
    assert evaluate(Floor(arg=half), variables=(), env={}).value == "0"
    assert evaluate(Ceil(arg=half), variables=(), env={}).value == "1"
    neg = ConstRat(numerator=-5, denominator=2)
    assert evaluate(Floor(arg=neg), variables=(), env={}).value == "-3"
    assert evaluate(Ceil(arg=neg), variables=(), env={}).value == "-2"


def test_sum_prod_over_and_pretty() -> None:
    variables = (
        BoundVarDeclV1(
            name="xs",
            kind="finite_sequence",
            domain="nat",
            unit="cardinality",
            meaning="finite declared input sizes",
        ),
    )
    expr = Add(
        args=(
            SumOver(index="d", sequence="xs", body=Var(name="d")),
            ProdOver(index="d", sequence="xs", body=Var(name="d")),
        )
    )
    result = evaluate(expr, variables=variables, env={"xs": [2, 3]})
    assert result.value == "11"
    text = pretty(expr)
    assert "Σ" in text and "Π" in text


def test_pending_placeholder_not_evaluable() -> None:
    doc = resolve_bound_ast(BOUND_PLACEHOLDER_PENDING)
    assert doc.evaluable is False
    with pytest.raises(BoundAstError, match="not evaluable"):
        evaluate_bound(BOUND_PLACEHOLDER_PENDING, {})


def test_closure_live_upper() -> None:
    result = evaluate_bound(BOUND_CLOSURE_LIVE_UPPER, {"live_count": 4})
    assert result.value == "4"


def test_parity_fixtures_against_registry_and_kern03() -> None:
    fixtures = load_parity_fixtures()
    adapter = TypeAdapter(ba.BoundExpr)
    for case in fixtures["cases"]:
        env = case["env"]
        if case.get("bound_ast_id"):
            doc = resolve_bound_ast(case["bound_ast_id"])
            if not doc.evaluable:
                continue
            result = evaluate_bound(case["bound_ast_id"], env)
            assert result.value == case["expected_value"], case["id"]
            if case.get("expected_predicate") is not None and doc.predicate is not None:
                assert (
                    evaluate_predicate(doc.predicate, variables=doc.variables, env=env)
                    is case["expected_predicate"]
                )
            field = case.get("kern03_field")
            if field and "domain_sizes" in env:
                if env.get("c_exp", 1) == 1 and env.get("c_ver", 1) == 1:
                    evidence = build_finite_search_bound(env["domain_sizes"])
                    assert int(result.fraction) == getattr(evidence, field)
        else:
            expr = adapter.validate_python(case["inline_expression"])
            result = evaluate(expr, variables=(), env=env)
            assert result.value == case["expected_value"], case["id"]


def test_registry_file_digests_match() -> None:
    raw = json.loads(_REGISTRY.read_text(encoding="utf-8"))
    for item in raw["entries"]:
        expected = item["digest_sha256"]
        payload = deepcopy(item)
        payload.pop("digest_sha256")
        doc = BoundAstDocumentV1.model_validate(payload)
        assert doc.digest() == expected


def test_no_eval_exec_in_module_source() -> None:
    source = (_ROOT / "src/slm_training/formal/bound_ast.py").read_text(encoding="utf-8")
    assert "eval(" not in source
    assert "exec(" not in source
    assert "__import__(" not in source
