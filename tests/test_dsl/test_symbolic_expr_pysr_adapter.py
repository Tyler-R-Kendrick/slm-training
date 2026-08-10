"""SRP-011 (SLM-475) regression tests: optional PySR/SRBench adapter with
an isolated environment manifest.

Every test here uses a mocked/golden ``raw_output_provider`` -- none
imports or requires a real PySR/Julia install, matching the issue's own
"Core repo tests use mocked/golden adapter output" scope line.
"""

from __future__ import annotations

import pytest

from slm_training.dsl.symbolic_expr_pysr_adapter import (
    ADAPTER_SCHEMA_VERSION,
    DEFAULT_PYSR_MANIFEST,
    DEFAULT_SRBENCH_MANIFEST,
    AdapterEnvironmentManifestV1,
    AdapterRunResultV1,
    build_adapter_run_input,
    parse_external_expression,
    run_adapter,
)
from slm_training.dsl.symbolic_expr_ir import CoefficientHole, OpApply, VarRef
from slm_training.dsl.symbolic_regression_pack import ALLOWED_OPERATORS, SymbolicRegressionProblemV1


def _problem(**overrides) -> SymbolicRegressionProblemV1:
    defaults = dict(
        variables=("x0", "x1"),
        table=((1.0, 2.0), (2.0, 3.0), (3.0, 1.0), (4.0, 5.0)),
        targets=(3.0, 5.0, 4.0, 9.0),  # x0 + x1, exactly
        allowed_operators=ALLOWED_OPERATORS,
        coefficient_hole_policy="fit_free_coefficients",
        complexity_budget=256,
        numeric_domain_policy="reject_nonfinite",
        train_indices=(0, 1, 2),
        validation_indices=(3,),
    )
    defaults.update(overrides)
    return SymbolicRegressionProblemV1(**defaults)


# ---------------------------------------------------------------------------
# Environment manifest
# ---------------------------------------------------------------------------


def test_manifest_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="tool_name"):
        AdapterEnvironmentManifestV1(
            schema_version="v1", tool_name="julia_direct", tool_version="1", python_version="3.11",
            seed=0, cpu_budget=1, memory_budget_mb=1, time_budget_s=1.0, iteration_budget=1,
        )
    with pytest.raises(ValueError, match="cpu_budget"):
        AdapterEnvironmentManifestV1(
            schema_version="v1", tool_name="pysr", tool_version="1", python_version="3.11",
            seed=0, cpu_budget=0, memory_budget_mb=1, time_budget_s=1.0, iteration_budget=1,
        )
    with pytest.raises(ValueError, match="time_budget_s"):
        AdapterEnvironmentManifestV1(
            schema_version="v1", tool_name="pysr", tool_version="1", python_version="3.11",
            seed=0, cpu_budget=1, memory_budget_mb=1, time_budget_s=0.0, iteration_budget=1,
        )


def test_manifest_fingerprint_is_stable_and_content_derived():
    a = DEFAULT_PYSR_MANIFEST
    b = AdapterEnvironmentManifestV1(**a.to_dict())
    assert a.fingerprint() == b.fingerprint()
    c = AdapterEnvironmentManifestV1(**{**a.to_dict(), "seed": 999})
    assert a.fingerprint() != c.fingerprint()


def test_manifest_round_trips_through_dict():
    restored = AdapterEnvironmentManifestV1.from_dict(DEFAULT_SRBENCH_MANIFEST.to_dict())
    assert restored == DEFAULT_SRBENCH_MANIFEST


# ---------------------------------------------------------------------------
# Problem -> external input translation
# ---------------------------------------------------------------------------


def test_build_adapter_run_input_carries_no_training_data():
    problem = _problem()
    adapter_input = build_adapter_run_input(problem, DEFAULT_PYSR_MANIFEST)
    payload = adapter_input.to_dict()
    assert payload["variable_names"] == ("x0", "x1")
    assert set(payload["binary_operators"]) == {"+", "-", "*", "/"}
    assert "sin" in payload["unary_operators"] and "sqrt" in payload["unary_operators"]
    assert payload["n_train_rows"] == 3
    assert payload["manifest_fingerprint"] == DEFAULT_PYSR_MANIFEST.fingerprint()
    # No raw table/targets values anywhere in the translated input.
    assert "table" not in payload and "targets" not in payload


def test_build_adapter_run_input_rejects_variable_name_colliding_with_a_function():
    problem = _problem(variables=("sin", "x1"))
    with pytest.raises(ValueError, match="collides with a supported function"):
        build_adapter_run_input(problem, DEFAULT_PYSR_MANIFEST)


# ---------------------------------------------------------------------------
# External output -> typed IR translation (golden cases)
# ---------------------------------------------------------------------------


def test_parse_simple_variable_sum():
    problem = _problem()
    result = parse_external_expression("x0 + x1", problem=problem)
    assert result.status == "ok"
    assert result.node == OpApply(operator="+", args=(VarRef(index=0), VarRef(index=1)))
    assert result.coefficients == ()


def test_parse_expression_with_fitted_constant_and_unary_function():
    problem = _problem()
    result = parse_external_expression("2.5*x0 - sin(x1)", problem=problem)
    assert result.status == "ok"
    assert result.coefficients == (2.5,)
    assert result.node == OpApply(
        operator="-",
        args=(
            OpApply(operator="*", args=(CoefficientHole(index=0), VarRef(index=0))),
            OpApply(operator="sin", args=(VarRef(index=1),)),
        ),
    )


def test_parse_multiple_numeric_literals_get_distinct_coefficient_holes():
    problem = _problem()
    result = parse_external_expression("1.5*x0 + 2.5", problem=problem)
    assert result.status == "ok"
    assert result.coefficients == (1.5, 2.5)
    assert result.node == OpApply(
        operator="+",
        args=(
            OpApply(operator="*", args=(CoefficientHole(index=0), VarRef(index=0))),
            CoefficientHole(index=1),
        ),
    )


def test_parse_unary_minus_precedence_matches_standard_convention():
    problem = _problem()
    result = parse_external_expression("-2*x0", problem=problem)
    assert result.status == "ok"
    # -2*x0 == (-2)*x0, not -(2*x0) in serialized form -- both are
    # mathematically identical here, but the parse must not raise.
    assert result.node == OpApply(
        operator="*", args=(OpApply(operator="neg", args=(CoefficientHole(index=0),)), VarRef(index=0))
    )


@pytest.mark.parametrize(
    "text,unsupported",
    [
        ("x0^2 + x1", ("^",)),
        ("square(x0)", ("square",)),
        ("x0 % x1", ("%",)),
        ("x0 + unknown_var", ("unknown_var",)),
    ],
)
def test_unsupported_operators_are_reported_not_approximated(text, unsupported):
    problem = _problem()
    result = parse_external_expression(text, problem=problem)
    assert result.status == "operator_mismatch"
    assert result.unsupported_operators == unsupported
    assert result.node is None


@pytest.mark.parametrize("text", ["x0 + (x1", "x0 +", "(x0 + x1))"])
def test_malformed_syntax_is_a_parse_error_not_a_crash(text):
    problem = _problem()
    result = parse_external_expression(text, problem=problem)
    assert result.status == "parse_error"
    assert result.error_detail
    assert result.node is None


def test_oversized_external_expression_fails_closed_on_budget():
    problem = _problem()
    text = "x0" + " + x0" * 400  # far past DEFAULT_MAX_NODES
    result = parse_external_expression(text, problem=problem)
    assert result.status == "parse_error"


# ---------------------------------------------------------------------------
# run_adapter: rescoring through the pack's own evaluator (not the tool's
# self-reported loss), and typed non-"ok" outcomes on failure.
# ---------------------------------------------------------------------------


def test_run_adapter_success_rescores_through_the_packs_own_evaluator():
    problem = _problem()

    def mock_pysr(_adapter_input):
        return "x0 + x1"

    result = run_adapter(problem, DEFAULT_PYSR_MANIFEST, raw_output_provider=mock_pysr)
    assert isinstance(result, AdapterRunResultV1)
    assert result.schema_version == ADAPTER_SCHEMA_VERSION
    assert result.status == "ok"
    assert result.evidence is not None
    # x0 + x1 exactly matches every constructed target -> zero loss on both splits.
    losses = {split["split"]: split["loss"] for split in result.evidence["splits"]}
    assert losses["train"] == pytest.approx(0.0)
    assert losses["validation"] == pytest.approx(0.0)
    assert result.node_sexpr == "(+ v0 v1)"


def test_run_adapter_external_failure_is_external_blocked_not_a_loss():
    problem = _problem()

    def mock_crash(_adapter_input):
        raise RuntimeError("julia process crashed")

    result = run_adapter(problem, DEFAULT_PYSR_MANIFEST, raw_output_provider=mock_crash)
    assert result.status == "external_blocked"
    assert result.evidence is None
    assert "julia process crashed" in result.error_detail


def test_run_adapter_timeout_is_reported_as_timeout():
    problem = _problem()

    def mock_timeout(_adapter_input):
        raise TimeoutError("exceeded time_budget_s")

    result = run_adapter(problem, DEFAULT_PYSR_MANIFEST, raw_output_provider=mock_timeout)
    assert result.status == "timeout"
    assert result.evidence is None


def test_run_adapter_unsupported_operator_is_reported_not_approximated():
    problem = _problem()

    def mock_power(_adapter_input):
        return "x0^2 + x1"

    result = run_adapter(problem, DEFAULT_PYSR_MANIFEST, raw_output_provider=mock_power)
    assert result.status == "operator_mismatch"
    assert result.unsupported_operators == ("^",)
    assert result.evidence is None
    # The raw external output is still preserved for audit even though it
    # could not be represented.
    assert result.raw_output == "x0^2 + x1"


def test_run_adapter_malformed_output_is_parse_error():
    problem = _problem()

    def mock_garbage(_adapter_input):
        return "x0 +"

    result = run_adapter(problem, DEFAULT_PYSR_MANIFEST, raw_output_provider=mock_garbage)
    assert result.status == "parse_error"
    assert result.evidence is None


def test_adapter_run_result_forbids_evidence_on_non_ok_status():
    with pytest.raises(ValueError, match="must not carry evidence"):
        AdapterRunResultV1(
            schema_version=ADAPTER_SCHEMA_VERSION,
            problem_fingerprint="x",
            pack_id="symbolic_regression",
            manifest=DEFAULT_PYSR_MANIFEST.to_dict(),
            status="external_blocked",
            raw_output=None,
            node_sexpr=None,
            coefficients=(),
            unsupported_operators=(),
            error_detail="boom",
            evidence={"fake": "evidence"},
            wall_ms=0.0,
        )


def test_adapter_run_result_requires_evidence_on_ok_status():
    with pytest.raises(ValueError, match="requires evidence"):
        AdapterRunResultV1(
            schema_version=ADAPTER_SCHEMA_VERSION,
            problem_fingerprint="x",
            pack_id="symbolic_regression",
            manifest=DEFAULT_PYSR_MANIFEST.to_dict(),
            status="ok",
            raw_output="x0",
            node_sexpr="v0",
            coefficients=(),
            unsupported_operators=(),
            error_detail=None,
            evidence=None,
            wall_ms=0.0,
        )


def test_repeated_runs_with_identical_mocked_output_are_deterministic():
    problem = _problem()

    def mock_pysr(_adapter_input):
        return "x0 + x1"

    first = run_adapter(problem, DEFAULT_PYSR_MANIFEST, raw_output_provider=mock_pysr)
    second = run_adapter(problem, DEFAULT_PYSR_MANIFEST, raw_output_provider=mock_pysr)
    assert first.evidence == second.evidence
    assert first.node_sexpr == second.node_sexpr
    assert first.coefficients == second.coefficients
