"""Tests for SLM-223 (NCS0-03) SemanticFloorGateV1 fixture harness."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm223_semantic_floor_gate import (
    DEFAULT_FLOOR_THRESHOLD,
    MATRIX_SET,
    MATRIX_VERSION,
    SemanticFloorGateReport,
    run_semantic_floor_gate_fixture,
)


def test_fixture_generates_rows_and_disposition() -> None:
    report = run_semantic_floor_gate_fixture(synthetic_runs=4)
    assert report.matrix_set == MATRIX_SET
    assert report.matrix_version == MATRIX_VERSION
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert report.floor_threshold == DEFAULT_FLOOR_THRESHOLD
    assert report.n_runs == 4
    assert report.n_families == 2
    assert report.rows
    assert report.disposition in {"signal_predictive", "no_signal", "inconclusive"}
    assert report.atlas_hash
    assert report.gate_hash


def test_all_rows_have_leave_one_family_out_fold() -> None:
    report = run_semantic_floor_gate_fixture(synthetic_runs=4)
    for row in report.rows:
        assert row.fold.startswith("held_out_")
        assert row.floor_label is not None


def test_permutation_null_is_evaluated() -> None:
    report = run_semantic_floor_gate_fixture(synthetic_runs=4)
    assert report.permutation_null["status"] == "evaluated"
    assert report.permutation_null["draws"] > 0
    assert 0.0 <= report.permutation_null["mean"] <= 1.0


def test_report_round_trip() -> None:
    report = run_semantic_floor_gate_fixture(synthetic_runs=4)
    recovered = SemanticFloorGateReport.from_dict(report.to_dict())
    assert recovered.n_runs == report.n_runs
    assert recovered.gate_hash == report.gate_hash
    assert len(recovered.rows) == len(report.rows)
    assert recovered.disposition == report.disposition


def test_gate_hash_is_deterministic() -> None:
    a = run_semantic_floor_gate_fixture(synthetic_runs=4)
    b = run_semantic_floor_gate_fixture(synthetic_runs=4)
    assert a.gate_hash == b.gate_hash
    assert a.disposition == b.disposition


def test_single_family_is_inconclusive() -> None:
    # 1 synthetic run means only one family is present; LOFO cannot hold anything out.
    report = run_semantic_floor_gate_fixture(synthetic_runs=1)
    assert report.n_families == 1
    assert report.disposition == "inconclusive"
    for row in report.rows:
        assert row.fold == "insufficient_families"
        assert row.gate_flag is None


def test_n_families_default_preserves_two_families() -> None:
    # Backward compatibility: omitting n_families still yields exactly 2
    # families, matching the original hardcoded behavior.
    report = run_semantic_floor_gate_fixture(synthetic_runs=8)
    assert report.n_families == 2


def test_n_families_parameter_controls_family_count() -> None:
    report = run_semantic_floor_gate_fixture(synthetic_runs=8, n_families=4)
    assert report.n_families == 4
    assert {row.family for row in report.rows} == {
        "family_0",
        "family_1",
        "family_2",
        "family_3",
    }


def test_floor_threshold_changes_labels() -> None:
    low = run_semantic_floor_gate_fixture(synthetic_runs=4, floor_threshold=0.0)
    high = run_semantic_floor_gate_fixture(synthetic_runs=4, floor_threshold=2.0)
    # floor_threshold=0.0: parse_rate is clipped to >= 0.0, so nothing is below it.
    assert all(row.floor_label is False for row in low.rows)
    # floor_threshold=2.0: parse_rate is clipped to <= 1.0, so everything is below it.
    assert all(row.floor_label is True for row in high.rows)
