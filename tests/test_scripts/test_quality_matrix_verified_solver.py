"""VSS4-02 (SLM-75): verified-solver matrix set — schema, rows, gates, fixture.

Torch-free: the fixture path drives the VSS4-01 benchmark on CPU and the gate
tests inject faults directly into the metric schema.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from scripts.run_quality_matrix import main
from slm_training.harnesses.model_build import verified_solver_matrix as vsm
from slm_training.harnesses.model_build.ship_gates import DEFAULT_SHIP_GATES
from slm_training.harnesses.solver_bench import CaseResult, SuiteReport, run_reference_suite


# --------------------------------------------------------------------------- #
# Schema: zero-default + backward compatibility.
# --------------------------------------------------------------------------- #


def test_row_metrics_zero_default_serializes_all_groups() -> None:
    payload = vsm.RowMetrics().to_dict()
    assert set(payload) == {
        "solver",
        "exact_search_work",
        "capsule",
        "topology",
        "energy",
        "surface",
        "quality",
    }
    # Correctness false-support fields default to not_applicable, never zero.
    assert payload["solver"]["false_unsupported_count"] is None
    assert payload["solver"]["unknown_preservation_violations"] is None
    # Preserved semantic-quality metrics are present and default to None.
    assert payload["quality"]["meaningful_program_rate"] is None
    # The whole schema round-trips through JSON unchanged.
    assert json.loads(json.dumps(payload)) == payload


def test_schema_version_and_matrix_set_are_stable() -> None:
    assert vsm.SCHEMA_VERSION == "verified_scope_solver_matrix_v1"
    assert vsm.MATRIX_SET == "verified-solver"
    # The eight fail-closed gate names are stable.
    assert vsm.HARD_GATES == (
        "false_unsupported_count",
        "unknown_preservation_violations",
        "certificate_replay_failures",
        "solved_without_final_verifier",
        "certified_unsat_with_incomplete_proof",
        "candidate_set_parity_failures",
        "surface.semantic_ir_mutation_violations",
        "structured_or_observable_slots_routed_to_ar",
    )


# --------------------------------------------------------------------------- #
# Closed benchmark computes false-support/unknown-preservation correctly.
# --------------------------------------------------------------------------- #


def test_closed_benchmark_row_computes_false_support_from_ground_truth() -> None:
    report = run_reference_suite()
    solver = vsm.solver_metrics_from_suite(report)
    # The committed fixture is clean: no false prune, no unknown removal.
    assert solver.false_unsupported_count == 0
    assert solver.unknown_preservation_violations == 0
    assert solver.certificate_replay_failures == 0
    # Support verdicts map onto the solve-status vocabulary (a/b/c/d fixture).
    assert solver.status_counts == {
        "solved": 1,
        "certified_unsat": 2,
        "unknown": 1,
        "budget_exhausted": 0,
    }
    assert solver.certificates_emitted == 4
    assert solver.certificates_replayed == 4


def test_false_prune_in_suite_propagates_to_solver_metric() -> None:
    # A synthetic case where the oracle certified unsupported but ground truth
    # keeps an accepted terminal live: a false certified prune.
    bad = CaseResult(
        case_id="synthetic-false-prune",
        family="finite-domain",
        oracle_verdict="unsupported",
        ground_truth_verdict="supported",
        expected_verdict="supported",
        certificate_replays=True,
        false_unsupported=True,
        unknown_preservation_violation=False,
        agrees=False,
    )
    report = SuiteReport(results=(bad,), manifest_digest="synthetic")
    solver = vsm.solver_metrics_from_suite(report)
    assert solver.false_unsupported_count == 1
    assert solver.false_unsupported_rate == 1.0


def test_non_closed_row_reports_false_support_as_not_applicable() -> None:
    result = vsm.run_fixture_row(vsm.row_by_id()["R0"])
    assert result.status == "ran"
    solver = result.metrics.solver.to_dict()
    assert solver["false_unsupported_count"] is None
    assert solver["unknown_preservation_violations"] is None
    assert result.gate is not None
    # The ground-truth gates are excluded, not counted as passing zeros.
    assert "false_unsupported_count" in result.gate.not_applicable


# --------------------------------------------------------------------------- #
# Matched rows resolve to the intended single-variable delta.
# --------------------------------------------------------------------------- #


def test_matched_rows_registered_with_declared_controls() -> None:
    rows = vsm.verified_solver_rows()
    assert [r.row_id for r in rows] == ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]
    controls = {r.row_id: r.control for r in rows}
    assert controls == {
        "R0": None,
        "R1": "R0",
        "R2": "R1",
        "R3": "R2",
        "R4": "R3",
        "R5": "R3",
        "R6": "R5",
    }


def test_each_matched_pair_differs_by_one_declared_variable() -> None:
    by_id = vsm.row_by_id()
    # R1 vs R0: only exact closure / solver turns on.
    r0, r1 = by_id["R0"], by_id["R1"]
    assert (r0.solver_enabled, r1.solver_enabled) == (False, True)
    assert r1.single_variable == "exact_closure=on"
    # R2 vs R1: only the ranker changes.
    r1, r2 = by_id["R1"], by_id["R2"]
    assert r1.ranker == "deterministic" and r2.ranker == "model"
    diff = {
        k
        for k in vars(r1)
        if getattr(r1, k) != getattr(r2, k)
    }
    # Identity/bookkeeping fields plus the single config lever (ranker).
    assert diff == {
        "row_id",
        "run_id",
        "description",
        "control",
        "single_variable",
        "ranker",
        "required_capabilities",
        "tags",
    }
    # R4 vs R3: only the ranker changes (model -> energy).
    r3, r4 = by_id["R3"], by_id["R4"]
    assert r3.ranker == "model" and r4.ranker == "energy"
    # R6 vs R5: only the realizer changes (deterministic -> ar).
    r5, r6 = by_id["R5"], by_id["R6"]
    assert r5.realizer == "deterministic" and r6.realizer == "ar"


# --------------------------------------------------------------------------- #
# Deterministic config/hash serialization.
# --------------------------------------------------------------------------- #


def test_config_hash_is_deterministic_and_row_sensitive() -> None:
    rows = vsm.verified_solver_rows()
    first = {r.row_id: vsm.config_hash(r) for r in rows}
    second = {r.row_id: vsm.config_hash(r) for r in rows}
    assert first == second
    # Distinct rows hash distinctly.
    assert len(set(first.values())) == len(first)
    # Flipping a single lever changes the hash.
    mutated = replace(rows[1], ranker="energy")
    assert vsm.config_hash(mutated) != first["R1"]


# --------------------------------------------------------------------------- #
# Missing capability marks the row blocked, never silently reconfigured.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("row_id", ["R2", "R3", "R4", "R5", "R6"])
def test_missing_checkpoint_marks_row_blocked_with_reason(row_id: str) -> None:
    result = vsm.run_fixture_row(vsm.row_by_id()[row_id])
    assert result.status == "blocked"
    assert result.blocked_reason
    assert "requires" in result.blocked_reason
    # A blocked row runs no solver and asserts no gate pass.
    assert result.gate is None


# --------------------------------------------------------------------------- #
# Hard gates fail closed on every injected fault.
# --------------------------------------------------------------------------- #


def _closed_solver(**overrides: object) -> vsm.SolverCorrectness:
    base = dict(
        enabled=True,
        false_unsupported_count=0,
        unknown_preservation_violations=0,
        certificate_replay_failures=0,
        solved_without_final_verifier=0,
        certified_unsat_with_incomplete_proof=0,
    )
    base.update(overrides)
    return vsm.SolverCorrectness(**base)  # type: ignore[arg-type]


def test_hard_gates_pass_on_clean_closed_metrics() -> None:
    metrics = vsm.RowMetrics(solver=_closed_solver())
    gate = vsm.evaluate_verified_solver_gates(
        metrics, closed_benchmark=True, solver_enabled=True, surface_active=False
    )
    assert gate.passed


@pytest.mark.parametrize(
    "fault,builder",
    [
        (
            "false_unsupported_count",
            lambda: vsm.RowMetrics(solver=_closed_solver(false_unsupported_count=1)),
        ),
        (
            "unknown_preservation_violations",
            lambda: vsm.RowMetrics(
                solver=_closed_solver(unknown_preservation_violations=1)
            ),
        ),
        (
            "certificate_replay_failures",
            lambda: vsm.RowMetrics(solver=_closed_solver(certificate_replay_failures=1)),
        ),
        (
            "solved_without_final_verifier",
            lambda: vsm.RowMetrics(
                solver=_closed_solver(solved_without_final_verifier=1)
            ),
        ),
        (
            "certified_unsat_with_incomplete_proof",
            lambda: vsm.RowMetrics(
                solver=_closed_solver(certified_unsat_with_incomplete_proof=1)
            ),
        ),
    ],
)
def test_solver_hard_gates_fail_on_injected_fault(fault, builder) -> None:
    gate = vsm.evaluate_verified_solver_gates(
        builder(), closed_benchmark=True, solver_enabled=True, surface_active=False
    )
    assert not gate.passed
    assert any(f.startswith(fault) for f in gate.failures)


def test_candidate_set_parity_failure_fails_gate() -> None:
    metrics = vsm.RowMetrics(
        solver=_closed_solver(),
        energy=vsm.EnergyMetrics(candidate_set_parity_failures=1),
    )
    gate = vsm.evaluate_verified_solver_gates(
        metrics, closed_benchmark=True, solver_enabled=True, surface_active=False
    )
    assert not gate.passed
    assert any("candidate_set_parity_failures" in f for f in gate.failures)


def test_semantic_ir_mutation_fails_gate() -> None:
    metrics = vsm.RowMetrics(
        solver=_closed_solver(),
        surface=vsm.SurfaceMetrics(semantic_ir_mutation_violations=1),
    )
    gate = vsm.evaluate_verified_solver_gates(
        metrics, closed_benchmark=True, solver_enabled=True, surface_active=True
    )
    assert not gate.passed
    assert any("semantic_ir_mutation_violations" in f for f in gate.failures)


def test_structured_slot_routed_to_ar_fails_gate() -> None:
    metrics = vsm.RowMetrics(
        solver=_closed_solver(),
        surface=vsm.SurfaceMetrics(structured_or_observable_slots_routed_to_ar=1),
    )
    gate = vsm.evaluate_verified_solver_gates(
        metrics, closed_benchmark=True, solver_enabled=True, surface_active=True
    )
    assert not gate.passed
    assert any("structured_or_observable_slots_routed_to_ar" in f for f in gate.failures)


def test_closed_row_missing_measurement_fails_closed() -> None:
    # A closed benchmark that never measured false-support fails, not passes.
    metrics = vsm.RowMetrics(solver=vsm.SolverCorrectness(enabled=True))
    gate = vsm.evaluate_verified_solver_gates(
        metrics, closed_benchmark=True, solver_enabled=True, surface_active=False
    )
    assert not gate.passed
    assert any("false_unsupported_count:unmeasured" in f for f in gate.failures)


# --------------------------------------------------------------------------- #
# Existing ship gates remain present and unchanged.
# --------------------------------------------------------------------------- #


def test_existing_ship_gates_are_retained() -> None:
    # The verified-solver work must not weaken or remove the OpenUI ship gates.
    assert "smoke" in DEFAULT_SHIP_GATES
    assert DEFAULT_SHIP_GATES["smoke"]["meaningful_program_rate"] == 0.66
    for suite in ("smoke", "held_out", "adversarial", "ood", "rico_held"):
        assert suite in DEFAULT_SHIP_GATES


# --------------------------------------------------------------------------- #
# never averages not_applicable into zero.
# --------------------------------------------------------------------------- #


def test_matched_delta_skips_not_applicable_fields() -> None:
    control = vsm.RowMetrics(solver=vsm.SolverCorrectness(enabled=False))
    row = vsm.RowMetrics(solver=_closed_solver())
    delta = vsm.matched_delta(control, row)
    # false_unsupported_count is None in the control, so it is not in the delta.
    assert "solver.false_unsupported_count" not in delta
    # A measured-on-both field is present.
    assert "solver.certificates_emitted" in delta


# --------------------------------------------------------------------------- #
# CLI integration: describe writes/loads nothing; fixture JSON/MD agree.
# --------------------------------------------------------------------------- #


def test_describe_writes_nothing_and_loads_nothing(tmp_path, capsys) -> None:
    run_root = tmp_path / "runs"
    rc = main(
        [
            "--matrix-set",
            "verified-solver",
            "--describe",
            "--run-root",
            str(run_root),
        ]
    )
    assert rc == 0
    # No file was written anywhere under the run root.
    assert not run_root.exists() or not any(run_root.iterdir())
    described = json.loads(capsys.readouterr().out)
    assert [row["id"] for row in described] == [
        "R0",
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6",
    ]
    # Fixture-runnable rows expect CPU; checkpoint rows expect GPU.
    by_id = {row["id"]: row for row in described}
    assert by_id["R0"]["hardware_expectation"] == "cpu"
    assert by_id["R2"]["hardware_expectation"] == "gpu"


def test_fixture_matrix_writes_matching_json_and_markdown(tmp_path, capsys) -> None:
    run_root = tmp_path / "runs"
    rc = main(["--matrix-set", "verified-solver", "--run-root", str(run_root)])
    assert rc == 0
    payload = json.loads(
        (run_root / "verified_solver_matrix_results.json").read_text()
    )
    markdown = (run_root / "verified_solver_matrix_results.md").read_text()
    assert payload["matrix_set"] == "verified-solver"
    assert payload["rows_ran"] == ["R0", "R1"]
    assert payload["hard_gates_pass"] is True
    # The Markdown carries the same per-row config hashes as the JSON.
    for entry in payload["results"]:
        assert entry["config_hash"] in markdown
    # No docs mirror is written for a non-default run root.
    assert "quality-experiment-matrix" in payload["matrix"]


def test_frontier_mode_blocks_all_rows_when_artifacts_absent(tmp_path) -> None:
    run_root = tmp_path / "runs"
    rc = main(
        [
            "--matrix-set",
            "verified-solver",
            "--mode",
            "frontier",
            "--run-root",
            str(run_root),
        ]
    )
    assert rc == 0
    payload = json.loads(
        (run_root / "verified_solver_matrix_results.json").read_text()
    )
    # Frontier mode resolves real artifacts; none exist here, so every row is
    # blocked with a clear reason rather than silently downgraded.
    assert payload["rows_ran"] == []
    assert payload["recipe"]["mode"] == "frontier"
    for entry in payload["results"]:
        assert entry["status"] == "blocked"
        assert entry["blocked_reason"]


def test_row_filter_selects_subset(tmp_path) -> None:
    run_root = tmp_path / "runs"
    rc = main(
        [
            "--matrix-set",
            "verified-solver",
            "--row",
            "R0,R1",
            "--run-root",
            str(run_root),
        ]
    )
    assert rc == 0
    payload = json.loads(
        (run_root / "verified_solver_matrix_results.json").read_text()
    )
    assert [entry["id"] for entry in payload["results"]] == ["R0", "R1"]


def test_unknown_row_filter_is_rejected(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--matrix-set",
                "verified-solver",
                "--row",
                "R9",
                "--run-root",
                str(tmp_path),
            ]
        )


# --------------------------------------------------------------------------- #
# Historical E-matrix readers remain compatible (runner not broken).
# --------------------------------------------------------------------------- #


def test_existing_quality_matrix_list_still_works(capsys) -> None:
    rc = main(["--matrix", "legacy", "--list"])
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert isinstance(rows, list) and rows
    assert {"id", "run_id", "description"} <= set(rows[0])
