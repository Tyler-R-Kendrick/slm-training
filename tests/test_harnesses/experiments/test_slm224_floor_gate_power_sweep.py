"""Tests for SLM-224 (NCS0-04) SemanticFloorGateV1 power-sweep fixture harness."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm224_floor_gate_power_sweep import (
    DEFAULT_SWEEP_GRID,
    MATRIX_SET,
    MATRIX_VERSION,
    SIGNAL_MARGIN,
    PowerSweepReport,
    run_power_sweep_fixture,
)


def test_sweep_generates_one_point_per_grid_value() -> None:
    grid = (4, 8, 16)
    report = run_power_sweep_fixture(sweep_grid=grid)
    assert report.matrix_set == MATRIX_SET
    assert report.matrix_version == MATRIX_VERSION
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert report.sweep_grid == grid
    assert len(report.points) == len(grid)
    assert [p.synthetic_runs for p in report.points] == list(grid)


def test_all_points_stay_at_two_synthetic_families() -> None:
    # The SLM-215 synthetic generator always splits run_idx % 2 -> 2 families,
    # regardless of synthetic_runs; this sweep only grows runs-per-family.
    report = run_power_sweep_fixture(sweep_grid=(4, 16, 64))
    for point in report.points:
        assert point.n_families == 2
        assert point.n_runs == point.synthetic_runs


def test_margin_and_disposition_are_consistent_per_point() -> None:
    report = run_power_sweep_fixture(sweep_grid=(4, 32))
    for point in report.points:
        if point.margin is not None:
            if point.margin >= SIGNAL_MARGIN:
                assert point.disposition == "signal_predictive"
            else:
                assert point.disposition in {"no_signal", "inconclusive"}


def test_sweep_disposition_is_power_limited_or_genuinely_no_signal() -> None:
    report = run_power_sweep_fixture(sweep_grid=DEFAULT_SWEEP_GRID)
    assert report.disposition in {
        "power_limited",
        "genuinely_no_signal_in_range",
        "inconclusive",
    }


def test_report_round_trip() -> None:
    report = run_power_sweep_fixture(sweep_grid=(4, 8))
    recovered = PowerSweepReport.from_dict(report.to_dict())
    assert recovered.sweep_grid == report.sweep_grid
    assert recovered.sweep_hash == report.sweep_hash
    assert len(recovered.points) == len(report.points)
    assert recovered.disposition == report.disposition


def test_sweep_hash_is_deterministic() -> None:
    a = run_power_sweep_fixture(sweep_grid=(4, 8, 16))
    b = run_power_sweep_fixture(sweep_grid=(4, 8, 16))
    assert a.sweep_hash == b.sweep_hash
    assert a.disposition == b.disposition
    assert [p.gate_hash for p in a.points] == [p.gate_hash for p in b.points]


def test_reruns_slm223_gate_pipeline_exactly_at_default_point() -> None:
    # synthetic_runs=4 is SLM-223's own default fixture point; the sweep must
    # reproduce SLM-223's committed no_signal result at that grid point.
    report = run_power_sweep_fixture(sweep_grid=(4,))
    point = report.points[0]
    assert point.n_runs == 4
    assert point.n_families == 2
    assert point.real_balanced_accuracy == 0.5
