"""Tests for SLM-183 (PQR) powered cluster-aware confirmation protocol harness."""

from __future__ import annotations

import json
import math

from slm_training.harnesses.experiments.slm183_power_protocol import (
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    PowerProtocolReport,
    analyze_existing_iter,
    build_default_manifest,
    render_markdown,
    run_variance_fixture,
)


def test_build_default_manifest() -> None:
    manifest = build_default_manifest(seeds=(0, 1, 2))
    assert manifest.suite_role
    assert len(manifest.seeds) == 3
    assert manifest.target_cluster_id
    assert manifest.mde > 0.0


def test_manifest_round_trip() -> None:
    manifest = build_default_manifest()
    reconstructed = type(manifest).from_dict(manifest.to_dict())
    assert reconstructed == manifest


def test_run_variance_fixture_produces_report() -> None:
    report = run_variance_fixture(
        n_targets=10, paths_per_target=2, n_seeds=3, run_id="test"
    )
    assert report.matrix_set == MATRIX_SET
    assert report.matrix_version == MATRIX_VERSION
    assert report.experiment_id == EXPERIMENT_ID
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert len(report.cells) == 10 * 3
    assert report.mde_curve
    assert report.version_stamp


def test_report_round_trip() -> None:
    report = run_variance_fixture(
        n_targets=8, paths_per_target=2, n_seeds=2, run_id="roundtrip"
    )
    reconstructed = PowerProtocolReport.from_dict(report.to_dict())
    assert reconstructed.matrix_set == MATRIX_SET
    assert reconstructed.experiment_id == EXPERIMENT_ID
    assert reconstructed.status == "fixture"
    assert len(reconstructed.cells) == len(report.cells)


def test_report_version_stamp() -> None:
    report = run_variance_fixture(
        n_targets=8, paths_per_target=2, n_seeds=2, run_id="stamp"
    )
    assert report.version_stamp["stamp_schema"] == "version_stamp/v1"
    components = report.version_stamp.get("components", {})
    assert "harness.experiments" in components
    assert "harness.experiments.slm183_power_protocol" in components
    assert "evals.power_protocol" in components


def test_render_markdown_contains_claim_class() -> None:
    report = run_variance_fixture(
        n_targets=8, paths_per_target=2, n_seeds=2, run_id="md"
    )
    md = render_markdown(report)
    assert "wiring / fixture only" in md
    assert "No-go for promotion" in md
    assert report.run_id in md


def test_analyze_existing_iter_with_records(tmp_path) -> None:
    records = [
        {"example_id": "ex1", "seed": 0, "pass": True},
        {"example_id": "ex2", "seed": 0, "pass": False},
        {"example_id": "ex3", "seed": 1, "pass": True},
        {"example_id": "ex4", "seed": 1, "pass": True},
    ]
    path = tmp_path / "iter.json"
    path.write_text(json.dumps({"records": records}))
    analysis = analyze_existing_iter(path)
    assert analysis["n_records"] == 4
    assert analysis["n_successes"] == 3
    assert math.isclose(analysis["success_rate"], 0.75, abs_tol=1e-9)
    assert "wilson_interval" in analysis
    assert "by_seed" in analysis
    assert len(analysis["by_seed"]) == 2


def test_analyze_existing_iter_top_level_list(tmp_path) -> None:
    records = [
        {"example_id": "ex1", "seed": 0, "target_score": 1.0},
        {"example_id": "ex2", "seed": 0, "target_score": 0.0},
    ]
    path = tmp_path / "iter.json"
    path.write_text(json.dumps(records))
    analysis = analyze_existing_iter(path)
    assert analysis["n_records"] == 2
    assert analysis["n_successes"] == 1
