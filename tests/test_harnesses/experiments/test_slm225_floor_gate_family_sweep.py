"""Tests for SLM-225 (NCS0-05) SemanticFloorGateV1 family-count sweep harness."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm225_floor_gate_family_sweep import (
    DEFAULT_RUNS_PER_FAMILY,
    DEFAULT_SWEEP_GRID,
    MATRIX_SET,
    MATRIX_VERSION,
    SIGNAL_MARGIN,
    FamilySweepReport,
    run_family_sweep_fixture,
)


def test_sweep_generates_one_point_per_grid_value() -> None:
    grid = (2, 4, 8)
    report = run_family_sweep_fixture(sweep_grid=grid)
    assert report.matrix_set == MATRIX_SET
    assert report.matrix_version == MATRIX_VERSION
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert report.sweep_grid == grid
    assert len(report.points) == len(grid)
    assert [p.n_families_requested for p in report.points] == list(grid)


def test_family_count_actually_varies() -> None:
    # Unlike SLM-224's sweep (fixed at 2 families), n_families here must vary
    # with the grid, and runs-per-family stays fixed.
    report = run_family_sweep_fixture(sweep_grid=(2, 4, 8))
    for point in report.points:
        assert point.n_families == point.n_families_requested
        assert point.runs_per_family == DEFAULT_RUNS_PER_FAMILY
        assert point.synthetic_runs == point.n_families_requested * DEFAULT_RUNS_PER_FAMILY
        assert point.n_runs == point.synthetic_runs


def test_runs_per_family_flag_is_plumbed() -> None:
    report = run_family_sweep_fixture(sweep_grid=(2, 4), runs_per_family=6)
    for point in report.points:
        assert point.runs_per_family == 6
        assert point.synthetic_runs == point.n_families_requested * 6


def test_margin_and_disposition_are_consistent_per_point() -> None:
    report = run_family_sweep_fixture(sweep_grid=(2, 8))
    for point in report.points:
        if point.margin is not None:
            if point.margin >= SIGNAL_MARGIN:
                assert point.disposition == "signal_predictive"
            else:
                assert point.disposition in {"no_signal", "inconclusive"}


def test_sweep_disposition_is_family_count_limited_or_genuinely_no_signal() -> None:
    report = run_family_sweep_fixture(sweep_grid=DEFAULT_SWEEP_GRID)
    assert report.disposition in {
        "family_count_limited",
        "genuinely_no_signal_in_range",
        "inconclusive",
    }


def test_report_round_trip() -> None:
    report = run_family_sweep_fixture(sweep_grid=(2, 4))
    recovered = FamilySweepReport.from_dict(report.to_dict())
    assert recovered.sweep_grid == report.sweep_grid
    assert recovered.sweep_hash == report.sweep_hash
    assert len(recovered.points) == len(report.points)
    assert recovered.disposition == report.disposition


def test_sweep_hash_is_deterministic() -> None:
    a = run_family_sweep_fixture(sweep_grid=(2, 4, 8))
    b = run_family_sweep_fixture(sweep_grid=(2, 4, 8))
    assert a.sweep_hash == b.sweep_hash
    assert a.disposition == b.disposition
    assert [p.gate_hash for p in a.points] == [p.gate_hash for p in b.points]


def test_two_families_point_matches_slm223_default_family_count() -> None:
    # n_families=2 with runs_per_family=4 reproduces SLM-223's own default
    # fixture shape (synthetic_runs=8, 2 families) -- a sanity anchor point.
    report = run_family_sweep_fixture(sweep_grid=(2,))
    point = report.points[0]
    assert point.n_families == 2
    assert point.synthetic_runs == 8
