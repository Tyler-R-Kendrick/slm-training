"""Tests for SLM-176 (P14) action-shortlist rerank fixture harness."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm176_action_shortlist_rerank import (
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    ShortlistReport,
    ShortlistScenario,
    build_cells,
    run_fixture_campaign,
    validate_manifest,
)


def test_build_cells_produces_scenarios() -> None:
    cells = build_cells(seeds=(0, 1))
    assert len(cells) > 0
    for cell in cells:
        assert cell.scenario_id
        assert cell.legal_set_size > 0
        assert cell.k >= 0


def test_validate_manifest_accepts_valid_cells() -> None:
    cells = build_cells(seeds=(0,))
    assert validate_manifest(cells) == []


def test_validate_manifest_rejects_duplicate_scenario_id() -> None:
    cells = build_cells(seeds=(0,))
    duplicated = cells + (cells[0],)
    errors = validate_manifest(duplicated)
    assert any("duplicate scenario_id" in e for e in errors)


def test_validate_manifest_rejects_bad_k() -> None:
    cells = build_cells(seeds=(0,))
    bad = ShortlistScenario(
        scenario_id="bad", legal_set_size=8, k=-1, seed=0, query_hint="x"
    )
    errors = validate_manifest(cells + (bad,))
    assert any("k must be non-negative" in e for e in errors)


def test_fixture_campaign_produces_report() -> None:
    report = run_fixture_campaign(seeds=(0,), d_model=32)
    assert report.matrix_set == MATRIX_SET
    assert report.matrix_version == MATRIX_VERSION
    assert report.experiment_id == EXPERIMENT_ID
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert len(report.rows) == len(report.cells)


def test_report_round_trip() -> None:
    report = run_fixture_campaign(seeds=(0,), d_model=32)
    reconstructed = ShortlistReport.from_dict(report.to_dict())
    assert reconstructed.matrix_set == MATRIX_SET
    assert reconstructed.matrix_version == MATRIX_VERSION
    assert reconstructed.experiment_id == EXPERIMENT_ID
    assert reconstructed.status == "fixture"
    assert reconstructed.claim_class == "wiring"
    assert len(reconstructed.rows) == len(report.rows)


def test_report_version_stamp() -> None:
    report = run_fixture_campaign(seeds=(0,), d_model=32)
    assert report.version_stamp["stamp_schema"] == "version_stamp/v1"
    components = report.version_stamp.get("components", {})
    assert "harness.experiments" in components
    assert "harness.experiments.slm176_action_shortlist_rerank" in components
    assert "dsl.action_shortlist" in components
    assert "dsl.action_descriptions" in components
