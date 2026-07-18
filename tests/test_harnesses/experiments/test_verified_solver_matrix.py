"""VSS4-02 verified-solver matrix: schema, matched rows, fixture, and hard gates (SLM-75).

Torch-free. The fixture path consumes the committed VSS4-01 benchmark
(``solver_bench``) so R0/R1 correctness metrics have independent ground truth. Every
fail-closed gate is exercised with an injected violation, and ``not_applicable``
correctness fields are never coerced to zero.
"""

from __future__ import annotations

import json

import pytest

from slm_training.harnesses.experiments.verified_solver_matrix import (
    MATRIX_SET,
    CapsuleMetrics,
    EnergyMetrics,
    ExactSearchMetrics,
    QualityMetrics,
    SolverProofMetrics,
    SurfaceMetrics,
    TopologyMetrics,
    VerifiedSolverRow,
    describe_matrix,
    evaluate_hard_gates,
    render_markdown,
    run_fixture_matrix,
)

_METRIC_GROUPS = (
    SolverProofMetrics,
    ExactSearchMetrics,
    CapsuleMetrics,
    TopologyMetrics,
    EnergyMetrics,
    SurfaceMetrics,
    QualityMetrics,
)


def _row(**overrides) -> VerifiedSolverRow:
    """A minimal resolved row; override one metric group to inject a violation."""
    base = dict(
        row_id="X",
        description="test row",
        control_row_id="R0",
        variable="test",
        config={},
        capability_status="run",
        blocked_reason=None,
    )
    base.update(overrides)
    return VerifiedSolverRow(**base)


def test_schema_zero_default_and_json_scalar():
    """Every metric group is default-constructible and serializes to JSON scalars."""
    for cls in _METRIC_GROUPS:
        d = cls().to_dict()
        assert d, cls.__name__
        for k, v in d.items():
            assert isinstance(v, (int, float, bool, str, type(None))), f"{cls.__name__}.{k}"
    # Correctness false-support defaults to not_applicable (None), never 0.
    assert SolverProofMetrics().false_unsupported_count is None
    assert SolverProofMetrics().false_unsupported_rate is None


def test_row_serialization_is_deterministic_and_backward_compatible():
    a = _row().to_dict()
    b = _row().to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    # Backward compatible: all seven metric groups are present as nested dicts.
    for group in ("solver", "exact_search", "capsule", "topology", "energy", "surface", "quality"):
        assert group in a and isinstance(a[group], dict)


def test_all_matched_rows_resolve_with_single_variable_deltas():
    rows = describe_matrix().rows
    ids = [r.row_id for r in rows]
    assert ids == ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]
    by_id = {r.row_id: r for r in rows}
    # R0 is the root control; every other row names its control and its one variable.
    assert by_id["R0"].control_row_id is None
    for rid in ("R1", "R2", "R3", "R4", "R5", "R6"):
        assert by_id[rid].control_row_id in by_id
        assert by_id[rid].variable and by_id[rid].variable != "baseline"


def test_describe_loads_no_model_and_marks_rows():
    """--describe resolves configs without a benchmark/model run; all rows either
    run-free control specs or not_run with a reason."""
    report = describe_matrix()
    assert report.mode == "describe"
    assert report.matrix_set == MATRIX_SET
    for r in report.rows:
        assert r.capability_status in ("run", "not_run", "blocked")
        if r.capability_status != "run":
            assert r.blocked_reason


def test_fixture_r1_closed_benchmark_correctness_is_exact():
    """R1 exact solver over the VSS4-01 closed benchmark: 1 supported, 2 certified
    unsat, 1 unknown, zero false prunes / unknown-preservation / replay failures."""
    report = run_fixture_matrix()
    r1 = next(r for r in report.rows if r.row_id == "R1")
    assert r1.capability_status == "run"
    s = r1.solver
    assert s.enabled is True
    assert (s.status_solved, s.status_certified_unsat, s.status_unknown) == (1, 2, 1)
    assert s.false_unsupported_count == 0  # measured, not None: closed benchmark
    assert s.unknown_preservation_violations == 0
    assert s.certificate_replay_failures == 0
    assert s.certificates_emitted == 4 == s.certificates_replayed
    # Real exact-search work counters were recorded from the oracle.
    assert r1.exact_search.solver_verifier_calls > 0
    assert r1.exact_search.certified_removals == 2


def test_fixture_r0_control_reports_not_applicable_not_zero():
    """R0 has the solver off, so false-support fields are not_applicable (None),
    never a fabricated zero, and R0 cannot trip a correctness gate."""
    report = run_fixture_matrix()
    r0 = next(r for r in report.rows if r.row_id == "R0")
    assert r0.solver.enabled is False
    assert r0.solver.false_unsupported_count is None
    assert r0.solver.false_unsupported_rate is None
    statuses = {g.gate: g.status for g in evaluate_hard_gates(r0)}
    assert statuses["false_unsupported_count"] == "not_applicable"


def test_model_backed_rows_are_not_run_not_silently_downgraded():
    """Rows needing a model/energy/surface head are marked not_run with a reason in
    fixture mode rather than silently substituting a weaker config under the row id."""
    report = run_fixture_matrix()
    for rid in ("R2", "R3", "R4", "R5", "R6"):
        row = next(r for r in report.rows if r.row_id == rid)
        assert row.capability_status == "not_run"
        assert row.blocked_reason
        # No fabricated correctness claim on an unrun row.
        assert row.solver.false_unsupported_count is None


@pytest.mark.parametrize(
    "gate, overrides",
    [
        ("false_unsupported_count", {"solver": SolverProofMetrics(enabled=True, false_unsupported_count=1)}),
        ("unknown_preservation_violations", {"solver": SolverProofMetrics(unknown_preservation_violations=1)}),
        ("certificate_replay_failures", {"solver": SolverProofMetrics(certificate_replay_failures=1)}),
        ("solved_without_final_verifier", {"solver": SolverProofMetrics(solved_without_final_verifier=1)}),
        ("certified_unsat_with_incomplete_proof", {"solver": SolverProofMetrics(certified_unsat_with_incomplete_proof=1)}),
        ("candidate_set_parity_failures", {"energy": EnergyMetrics(candidate_set_parity_failures=1)}),
        ("semantic_ir_mutation_violations", {"surface": SurfaceMetrics(semantic_ir_mutation_violations=1)}),
        ("structured_or_observable_slots_routed_to_ar", {"surface": SurfaceMetrics(structured_slots_routed_to_ar=1)}),
    ],
)
def test_hard_gates_fail_closed_on_each_injected_violation(gate, overrides):
    """Every fail-closed gate flips to ``fail`` on its injected nonzero violation."""
    row = _row(**overrides)
    results = {g.gate: g for g in evaluate_hard_gates(row)}
    assert results[gate].status == "fail", gate
    assert results[gate].observed == 1


def test_na_correctness_field_is_not_applicable_not_a_pass_of_a_real_violation():
    """A None (not measured) correctness field is not_applicable, distinct from a
    measured 0 pass; it is never coerced to zero."""
    na_row = _row(solver=SolverProofMetrics(enabled=False, false_unsupported_count=None))
    zero_row = _row(solver=SolverProofMetrics(enabled=True, false_unsupported_count=0))
    na = {g.gate: g.status for g in evaluate_hard_gates(na_row)}
    zero = {g.gate: g.status for g in evaluate_hard_gates(zero_row)}
    assert na["false_unsupported_count"] == "not_applicable"
    assert zero["false_unsupported_count"] == "pass"


def test_fixture_matrix_passes_all_gates_and_renders_consistent_evidence():
    report = run_fixture_matrix()
    assert report.passed is True
    assert report.gate_failures == ()
    # JSON and Markdown are both derivable and internally consistent.
    payload = json.loads(report.to_json())
    assert payload["passed"] is True
    assert payload["gate_failure_count"] == 0
    assert len(payload["rows"]) == 7
    md = render_markdown(report)
    assert "VSS4-02 verified-solver matrix" in md
    assert "**PASS**" in md
    assert "not_run" in md  # frontier rows honestly marked


def test_run_id_is_stable_across_identical_runs():
    assert describe_matrix().run_id == describe_matrix().run_id
    assert run_fixture_matrix().run_id == run_fixture_matrix().run_id


def test_blocked_row_requires_reason():
    with pytest.raises(ValueError):
        _row(capability_status="not_run", blocked_reason=None)
    with pytest.raises(ValueError):
        _row(capability_status="bogus")
