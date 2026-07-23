"""Tests for SLM-215 (NCS0-02) SpectralAtlasV1 fixture harness."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.harnesses.experiments.slm215_spectral_atlas import (
    MATRIX_SET,
    MATRIX_VERSION,
    SpectralAtlasReport,
    run_spectral_atlas_fixture,
)


def test_fixture_generates_rows_and_signal() -> None:
    report = run_spectral_atlas_fixture(synthetic_runs=4)
    assert report.matrix_set == MATRIX_SET
    assert report.matrix_version == MATRIX_VERSION
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert report.n_rows > 0
    assert report.n_runs == 4
    assert report.n_families == 2
    assert report.signal["status"] == "evaluated"
    assert "spearman_alpha_z_vs_parse" in report.signal
    assert report.atlas_hash
    assert report.floor_gate_hash
    assert report.floor_gate_verdict == "inconclusive"


def test_report_round_trip() -> None:
    report = run_spectral_atlas_fixture(synthetic_runs=3)
    recovered = SpectralAtlasReport.from_dict(report.to_dict())
    assert recovered.n_rows == report.n_rows
    assert recovered.atlas_hash == report.atlas_hash
    assert len(recovered.rows) == len(report.rows)


def test_atlas_hash_is_deterministic() -> None:
    a = run_spectral_atlas_fixture(synthetic_runs=4)
    b = run_spectral_atlas_fixture(synthetic_runs=4)
    assert a.atlas_hash == b.atlas_hash


def test_custom_floor_gate_path_is_recorded() -> None:
    gate_path = Path(__file__).resolve().parents[3] / "docs/design/semantic-floor-gate-v1.json"
    report = run_spectral_atlas_fixture(
        synthetic_runs=2,
        floor_gate_path=gate_path,
    )
    assert report.floor_gate_ref == gate_path.as_posix()
    assert report.floor_gate_hash == json.loads(gate_path.read_text())["gate_hash"]


def test_role_summaries_present() -> None:
    report = run_spectral_atlas_fixture(synthetic_runs=4)
    assert report.role_summaries
    for role, summary in report.role_summaries.items():
        assert summary["n_matrices"] > 0


def test_collect_from_existing_slm214_report(tmp_path) -> None:
    from slm_training.harnesses.experiments.slm214_spectral_snapshot import (
        run_spectral_snapshot_fixture,
    )

    spectral_report = run_spectral_snapshot_fixture(null_draws=5)
    spectral_report.to_json(tmp_path / "slm214_spectral_report.json")
    atlas = run_spectral_atlas_fixture(reports_dir=tmp_path, synthetic_runs=4)
    assert any(r.source_reports == (str(tmp_path / "slm214_spectral_report.json"),) for r in [atlas])
    assert atlas.n_rows > 0
