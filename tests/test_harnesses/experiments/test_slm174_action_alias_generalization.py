"""Tests for SLM-174 (SDE2-07) action-alias generalization fixture harness."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm174_action_alias_generalization import (
    ARM_NAMES,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    AliasArm,
    AliasReport,
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
    bad = AliasArm(arm_id="bad", arm_name="invalid", seed=0)
    errors = validate_manifest(cells + (bad,))
    assert any("invalid arm_name" in e for e in errors)


def test_fixture_campaign_produces_report() -> None:
    report = run_fixture_campaign(seeds=(0,), d_model=32)
    assert report.matrix_set == MATRIX_SET
    assert report.matrix_version == MATRIX_VERSION
    assert report.experiment_id == EXPERIMENT_ID
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert len(report.rows) == len(ARM_NAMES)


def test_fixture_campaign_no_alias_leakage() -> None:
    report = run_fixture_campaign(seeds=(0,), d_model=32)
    for row in report.rows:
        if row.arm_name in {
            "fixed_alias_description_without_name",
            "multiple_alias_shuffled_descriptions",
            "alias_signature_only",
        }:
            assert row.leakage_findings == (), f"{row.arm_name} leaked: {row.leakage_findings}"


def test_report_round_trip() -> None:
    report = run_fixture_campaign(seeds=(0,), d_model=32)
    reconstructed = AliasReport.from_dict(report.to_dict())
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
    assert "harness.experiments.slm174_action_alias_generalization" in components
    assert "dsl.action_descriptions" in components


def test_resolve_disposition_baseline_unreliable() -> None:
    means = {"canonical_name_plus_description": {"family_purity": 0.1}}
    disposition, _ = resolve_disposition(means)
    assert disposition == "baseline_unreliable"


def test_resolve_disposition_alias_generalization_wired() -> None:
    means = {
        "canonical_name_plus_description": {"family_purity": 1.0},
        "canonical_name_description_without_name": {"family_purity": 0.9},
        "fixed_alias_description_without_name": {"family_purity": 0.85},
        "multiple_alias_shuffled_descriptions": {"family_purity": 0.3},
        "alias_signature_only": {"family_purity": 0.2},
        "multiple_alias_augmentation_held_out": {"held_out_transfer_score": 0.2},
        "canonical_evaluated_under_unseen_alias": {"canonical_unseen_alias_score": 0.2},
    }
    disposition, _ = resolve_disposition(means)
    assert disposition == "alias_generalization_wired"
