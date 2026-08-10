"""SRP-001 (SLM-441): symbolic-regression DslPack registry, problem contract,
and the safe minimal parse/evaluate round trip.

Fits the canonical DslPack slot contract (F1/#290) — registration/isolation,
malformed-problem rejection, serialization/hash determinism, and a minimal
parse/evaluate round trip that never calls ``eval``/``exec``.
"""

from __future__ import annotations

import pytest

from tests.casefiles import case_values

from slm_training.dsl.lang_core import ParseError
from slm_training.dsl.pack import DslPack, get_pack, list_packs
from slm_training.dsl.symbolic_regression_pack import (
    ALLOWED_OPERATORS,
    SYMBOLIC_REGRESSION_PROBLEM_SCHEMA,
    SymbolicRegressionProblemV1,
    evaluate_expression,
    evaluate_source,
)

ROOT_SOURCE = "a = 2 * x\nroot = a + sin(y) - 1\n"


def _problem(**overrides: object) -> SymbolicRegressionProblemV1:
    fields = dict(
        variables=("x", "y"),
        table=((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)),
        targets=(3.0, 7.0, 11.0),
        allowed_operators=frozenset({"+", "*"}),
        coefficient_hole_policy="fixed_constants_only",
        complexity_budget=10,
        numeric_domain_policy="reject_nonfinite",
        train_indices=(0, 1),
        validation_indices=(2,),
    )
    fields.update(overrides)
    return SymbolicRegressionProblemV1(**fields)


# ---------------------------------------------------------------------------
# Registry resolution and opt-in isolation
# ---------------------------------------------------------------------------


def test_pack_registers_and_is_opt_in_only(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "symbolic_regression" in list_packs()
    pack = get_pack("symbolic_regression")
    assert isinstance(pack, DslPack)
    assert pack.pack_id == "symbolic_regression"
    assert pack.reward_label == "well_formed_not_behavioral"
    assert pack.require("oracle") is not None
    assert pack.require("canonicalize") is not None

    # Merely registering never changes the default pack.
    monkeypatch.delenv("SLM_GRAMMAR_DSL", raising=False)
    import slm_training.models.grammar as grammar

    monkeypatch.setattr(grammar, "_ACTIVE_DSL", None)
    assert get_pack().pack_id == "openui"
    assert get_pack("default").pack_id == "openui"
    assert get_pack("auto").pack_id == "openui"


def test_pack_reachable_only_through_explicit_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLM_GRAMMAR_DSL", "symbolic_regression")
    import slm_training.models.grammar as grammar

    monkeypatch.setattr(grammar, "_ACTIVE_DSL", None)
    assert get_pack().pack_id == "symbolic_regression"


def test_no_content_placeholders_in_this_domain() -> None:
    policy = get_pack("symbolic_regression").placeholder_policy
    assert policy.is_placeholder("x") is False
    assert policy.extract(ROOT_SOURCE) == []
    assert policy.slot_contract(ROOT_SOURCE) == ()
    assert policy.slot_contract(ROOT_SOURCE, declared=("a",)) == ("a",)


# ---------------------------------------------------------------------------
# Problem contract: valid construction, determinism, malformed rejection
# ---------------------------------------------------------------------------


def test_problem_contract_is_versioned_and_hashable() -> None:
    problem = _problem()
    assert problem.schema == SYMBOLIC_REGRESSION_PROBLEM_SCHEMA
    assert problem.pack_id == "symbolic_regression"
    digest = problem.fingerprint
    assert isinstance(digest, str) and len(digest) == 64
    assert digest == _problem().fingerprint  # deterministic, same content


def test_problem_contract_fingerprint_changes_with_content() -> None:
    baseline = _problem().fingerprint
    assert _problem(seed=1).fingerprint != baseline
    assert _problem(complexity_budget=99).fingerprint != baseline


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"allowed_operators": frozenset({"eval"})}, "unsupported operator"),
        ({"table": ((1.0,), (2.0,))}, "does not match"),
        ({"targets": (1.0,)}, "one value per table row"),
        ({"complexity_budget": 0}, "positive"),
        ({"numeric_domain_policy": "clip_and_pray"}, "numeric_domain_policy"),
        ({"coefficient_hole_policy": "anything_goes"}, "coefficient_hole_policy"),
        ({"variables": ()}, "at least one input variable"),
        ({"variables": ("x", "x")}, "unique"),
        ({"variables": ("2x", "y")}, "invalid variable name"),
        ({"train_indices": (99,)}, "out-of-range"),
        ({"train_indices": (0,), "validation_indices": (0,)}, "more than one split"),
        ({"train_indices": ()}, "train_indices must be non-empty"),
        (
            {"dimensional_constraints": {"z": "meters"}},
            "unknown variable",
        ),
    ],
)
def test_problem_contract_rejects_malformed_input(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _problem(**overrides)


def test_problem_contract_rejects_operators_outside_the_pack_allowlist() -> None:
    with pytest.raises(ValueError, match="unsupported operator"):
        _problem(allowed_operators=frozenset({"+", "matmul"}))
    assert "matmul" not in ALLOWED_OPERATORS


# ---------------------------------------------------------------------------
# Minimal, safe parse/evaluate round trip
# ---------------------------------------------------------------------------


def test_canonicalize_round_trips_a_valid_program() -> None:
    pack = get_pack("symbolic_regression")
    canonical = pack.canonicalize(ROOT_SOURCE)
    assert canonical == pack.canonicalize(canonical)  # idempotent


def test_oracle_accepts_allow_listed_expression_and_rejects_unknown_operator() -> None:
    pack = get_pack("symbolic_regression")
    program = pack.oracle(ROOT_SOURCE)
    assert program.root is not None

    with pytest.raises(ParseError, match="unsupported operator"):
        pack.oracle("root = tan(x)\n")  # parses fine; `tan` is not allow-listed


def test_oracle_requires_a_root_binding() -> None:
    pack = get_pack("symbolic_regression")
    with pytest.raises(ParseError, match="must bind `root`"):
        pack.oracle("a = 1\n")


def test_evaluate_source_computes_the_expected_value() -> None:
    # a = 2*x; root = a + sin(y) - 1, at x=3, y=0 -> 2*3 + sin(0) - 1 = 5
    assert evaluate_source(ROOT_SOURCE, {"x": 3.0, "y": 0.0}) == pytest.approx(5.0)


def test_evaluate_expression_rejects_division_by_zero() -> None:
    node = {"k": "BinOp", "op": "/", "left": 1.0, "right": 0.0}
    with pytest.raises(ValueError, match="division by zero"):
        evaluate_expression(node, {})


def test_evaluate_expression_rejects_undefined_variable() -> None:
    node = {"type": "ref", "name": "z"}
    with pytest.raises(ValueError, match="undefined variable"):
        evaluate_expression(node, {"x": 1.0})


def test_evaluate_expression_rejects_unsupported_function() -> None:
    node = {"type": "call", "name": "system", "args": [1.0]}
    with pytest.raises(ValueError, match="unsupported function"):
        evaluate_expression(node, {})


def test_evaluate_expression_rejects_non_finite_result() -> None:
    node = {"k": "BinOp", "op": "/", "left": 1.0, "right": 1e-320}
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_expression(node, {})


def test_evaluate_expression_rejects_out_of_domain_function_input() -> None:
    node = {"type": "call", "name": "log", "args": [0.0]}
    with pytest.raises(ValueError, match="outside its domain"):
        evaluate_expression(node, {})


@pytest.mark.parametrize(
    "malicious_source",
    case_values(__file__, "test_core_execution_never_evaluates_arbitrary_python"),
)
def test_core_execution_never_evaluates_arbitrary_python(malicious_source: str) -> None:
    """Core parse/oracle path only ever walks a fixed dispatch table — it
    must reject anything resembling code execution, never run it."""
    pack = get_pack("symbolic_regression")
    with pytest.raises((ParseError, ValueError)):
        pack.oracle(malicious_source)
