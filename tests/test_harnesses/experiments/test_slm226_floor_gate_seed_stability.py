"""Tests for SLM-226 (NCS0-06) SemanticFloorGateV1 permutation-null seed-stability sweep harness."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm226_floor_gate_seed_stability import (
    DEFAULT_RUNS_PER_FAMILY,
    DEFAULT_SEEDS,
    DEFAULT_SWEEP_GRID,
    MATRIX_SET,
    MATRIX_VERSION,
    SIGNAL_MARGIN,
    SeedStabilityReport,
    run_seed_stability_fixture,
)


def test_sweep_generates_one_point_per_grid_value() -> None:
    grid = (2, 4)
    report = run_seed_stability_fixture(sweep_grid=grid, seeds=(11, 3))
    assert report.matrix_set == MATRIX_SET
    assert report.matrix_version == MATRIX_VERSION
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert report.sweep_grid == grid
    assert len(report.points) == len(grid)
    assert [p.n_families for p in report.points] == list(grid)


def test_each_point_evaluates_every_seed() -> None:
    seeds = (11, 3, 7)
    report = run_seed_stability_fixture(sweep_grid=(2, 4), seeds=seeds)
    for point in report.points:
        assert point.seeds == seeds
        assert len(point.margins) == len(seeds)
        assert len(point.gate_hashes) == len(seeds)


def test_real_balanced_accuracy_is_seed_invariant() -> None:
    # permutation_seed only affects the null baseline, not the real LOFO
    # accuracy, so real_balanced_accuracy must be identical regardless of
    # which permutation-null seeds were swept.
    report_a = run_seed_stability_fixture(sweep_grid=(4,), seeds=(11,))
    report_b = run_seed_stability_fixture(sweep_grid=(4,), seeds=(3, 7, 19))
    assert (
        report_a.points[0].real_balanced_accuracy == report_b.points[0].real_balanced_accuracy
    )


def test_default_seed_first_entry_reproduces_slm225() -> None:
    # DEFAULT_SEEDS starts with 11, SLM-223/224/225's hardcoded permutation
    # seed, so the first margin at n_families=4 must match SLM-225's committed
    # 0.188 result.
    report = run_seed_stability_fixture(sweep_grid=(4,), seeds=DEFAULT_SEEDS)
    point = report.points[0]
    assert point.seeds[0] == 11
    assert abs(point.margins[0] - 0.1875) < 1e-9


def test_stability_labels_are_consistent_with_margins() -> None:
    report = run_seed_stability_fixture(sweep_grid=(2, 4, 8), seeds=(11, 3, 7))
    for point in report.points:
        crossing = sum(1 for m in point.margins if m >= SIGNAL_MARGIN)
        assert point.seeds_crossing_margin == crossing
        if crossing == 0:
            assert point.stability == "stable_no_signal"
        elif crossing == len(point.margins):
            assert point.stability == "stable_signal"
        else:
            assert point.stability == "seed_sensitive"


def test_sweep_disposition_is_one_of_expected_values() -> None:
    report = run_seed_stability_fixture(sweep_grid=DEFAULT_SWEEP_GRID, seeds=DEFAULT_SEEDS)
    assert report.disposition in {
        "permutation_noise_explains_dip",
        "dip_stable_under_permutation_resampling",
        "inconclusive",
    }


def test_runs_per_family_flag_is_plumbed() -> None:
    report = run_seed_stability_fixture(sweep_grid=(2, 4), seeds=(11,), runs_per_family=6)
    for point in report.points:
        assert point.runs_per_family == 6
        assert point.synthetic_runs == point.n_families * 6


def test_report_round_trip() -> None:
    report = run_seed_stability_fixture(sweep_grid=(2, 4), seeds=(11, 3))
    recovered = SeedStabilityReport.from_dict(report.to_dict())
    assert recovered.sweep_grid == report.sweep_grid
    assert recovered.sweep_hash == report.sweep_hash
    assert len(recovered.points) == len(report.points)
    assert recovered.disposition == report.disposition


def test_sweep_hash_is_deterministic() -> None:
    a = run_seed_stability_fixture(sweep_grid=(2, 4), seeds=(11, 3))
    b = run_seed_stability_fixture(sweep_grid=(2, 4), seeds=(11, 3))
    assert a.sweep_hash == b.sweep_hash
    assert a.disposition == b.disposition
    assert [p.gate_hashes for p in a.points] == [p.gate_hashes for p in b.points]


def test_default_runs_per_family_matches_slm225() -> None:
    assert DEFAULT_RUNS_PER_FAMILY == 4
