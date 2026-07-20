"""Tests for SLM-172 (SDE2-05) render-equivalence fixture harness."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm172_render_equivalence import (
    ARM_NAMES,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    RenderEquivalenceArm,
    RenderEquivalenceReport,
    build_cells,
    resolve_disposition,
    run_fixture_campaign,
    validate_manifest,
)


def test_build_cells_produces_all_arms_per_seed() -> None:
    cells = build_cells(seeds=(0, 1, 2))
    assert len(cells) == len(ARM_NAMES) * 3
    per_seed = {}
    for cell in cells:
        per_seed.setdefault(cell.seed, set()).add(cell.arm_id)
    assert len(per_seed) == 3
    for seed, ids in per_seed.items():
        assert len(ids) == len(ARM_NAMES), f"seed {seed} has {len(ids)} cells"


def test_cells_cover_all_arm_names() -> None:
    cells = build_cells(seeds=(0,))
    seen = {c.arm_name for c in cells}
    assert seen == set(ARM_NAMES)


def test_validate_manifest_accepts_valid_cells() -> None:
    cells = build_cells(seeds=(0,))
    assert validate_manifest(cells) == []


def test_validate_manifest_rejects_duplicate_arm_id() -> None:
    cells = build_cells(seeds=(0,))
    duplicated = cells + (cells[0],)
    errors = validate_manifest(duplicated)
    assert any("duplicate arm_id" in e for e in errors)


def test_validate_manifest_rejects_invalid_arm_name() -> None:
    cells = build_cells(seeds=(0,))
    bad = RenderEquivalenceArm(arm_id="bad", arm_name="invalid", seed=0)
    errors = validate_manifest(cells + (bad,))
    assert any("invalid arm_name" in e for e in errors)


def test_fixture_campaign_produces_report() -> None:
    report = run_fixture_campaign(seeds=(0,))
    assert report.matrix_set == MATRIX_SET
    assert report.matrix_version == MATRIX_VERSION
    assert report.experiment_id == EXPERIMENT_ID
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert len(report.rows) == len(ARM_NAMES)


def test_exact_arms_are_equivalent() -> None:
    report = run_fixture_campaign(seeds=(0,))
    for row in report.rows:
        if row.arm_name in {"canonical_exact", "alpha_renamed", "style_only_change"}:
            assert row.equivalent is True, f"{row.arm_name} should be equivalent"


def test_corrupted_arms_are_not_equivalent() -> None:
    report = run_fixture_campaign(seeds=(0,))
    for row in report.rows:
        if row.arm_name in {
            "topology_corruption",
            "binding_corruption",
            "component_substitution",
            "metric_gaming_minimal_valid",
        }:
            assert row.equivalent is False, f"{row.arm_name} should not be equivalent"


def test_report_round_trip() -> None:
    report = run_fixture_campaign(seeds=(0,))
    reconstructed = RenderEquivalenceReport.from_dict(report.to_dict())
    assert reconstructed.matrix_set == MATRIX_SET
    assert reconstructed.matrix_version == MATRIX_VERSION
    assert reconstructed.experiment_id == EXPERIMENT_ID
    assert reconstructed.status == "fixture"
    assert reconstructed.claim_class == "wiring"
    assert len(reconstructed.rows) == len(report.rows)


def test_report_version_stamp() -> None:
    report = run_fixture_campaign(seeds=(0,))
    assert report.version_stamp["stamp_schema"] == "version_stamp/v1"
    components = report.version_stamp.get("components", {})
    assert "harness.experiments" in components
    assert "harness.experiments.slm172_render_equivalence" in components
    assert "evals.render_equivalence" in components


def test_resolve_disposition_calibrated() -> None:
    means = {
        "canonical_exact": {"equivalent": 1.0},
        "alpha_renamed": {"equivalent": 1.0},
        "style_only_change": {"equivalent": 1.0},
        "topology_corruption": {"equivalent": 0.0},
        "binding_corruption": {"equivalent": 0.0},
        "component_substitution": {"equivalent": 0.0},
        "metric_gaming_minimal_valid": {"equivalent": 0.0},
    }
    disposition, _ = resolve_disposition(means)
    assert disposition == "calibrated"


def test_resolve_disposition_leak() -> None:
    means = {
        "canonical_exact": {"equivalent": 1.0},
        "alpha_renamed": {"equivalent": 1.0},
        "style_only_change": {"equivalent": 1.0},
        "topology_corruption": {"equivalent": 1.0},
        "binding_corruption": {"equivalent": 0.0},
        "component_substitution": {"equivalent": 0.0},
        "metric_gaming_minimal_valid": {"equivalent": 0.0},
    }
    disposition, _ = resolve_disposition(means)
    assert disposition == "semantic_leak"


def test_resolve_disposition_unreliable() -> None:
    means = {
        "canonical_exact": {"equivalent": 1.0},
        "alpha_renamed": {"equivalent": 0.5},
        "style_only_change": {"equivalent": 1.0},
    }
    disposition, _ = resolve_disposition(means)
    assert disposition == "canonical_signature_unreliable"
