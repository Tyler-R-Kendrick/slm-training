"""Tests for SLM-170 (SDE2-03) exposure-targeted rare-action sampling fixture harness."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm170_exposure_targeted_rare_action import (
    ARM_NAMES,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    RareActionArm,
    RareActionReport,
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
    bad = RareActionArm(
        arm_id="bad",
        arm_name="invalid",
        policy="exposure_targeted",
        seed=0,
        total_decision_budget=64,
        per_root_cap=4,
        per_template_cap=4,
        max_importance_weight=10.0,
    )
    errors = validate_manifest(cells + (bad,))
    assert any("invalid arm_name" in e for e in errors)


def test_validate_manifest_rejects_non_positive_budget() -> None:
    cells = build_cells(seeds=(0,))
    bad = RareActionArm(
        arm_id="bad",
        arm_name="current",
        policy="with_replacement",
        seed=0,
        total_decision_budget=0,
        per_root_cap=4,
        per_template_cap=4,
        max_importance_weight=10.0,
    )
    errors = validate_manifest(cells + (bad,))
    assert any("total_decision_budget" in e for e in errors)


def test_fixture_campaign_produces_report() -> None:
    report = run_fixture_campaign(seeds=(0,))
    assert report.matrix_set == MATRIX_SET
    assert report.matrix_version == MATRIX_VERSION
    assert report.experiment_id == EXPERIMENT_ID
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert len(report.rows) == len(ARM_NAMES)


def test_exposure_targeted_arms_increase_rare_action_recall() -> None:
    report = run_fixture_campaign(seeds=(0, 1))
    current = report.arm_means["current"]["rare_action_recall"]
    floor = report.arm_means["minimum_exposure_floor"]["rare_action_recall"]
    assert floor >= current


def test_report_round_trip() -> None:
    report = run_fixture_campaign(seeds=(0,))
    reconstructed = RareActionReport.from_dict(report.to_dict())
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
    assert "harness.experiments.slm170_exposure_targeted_rare_action" in components


def test_resolve_disposition_useful_or_inconclusive() -> None:
    report = run_fixture_campaign(seeds=(0, 1, 2))
    assert report.disposition in {
        "useful_rare_action_exposure",
        "modest_rare_action_lift",
        "no_exposure_lift",
        "inconclusive",
    }


def test_resolve_disposition_no_lift() -> None:
    means = {
        "current": {"rare_action_recall": 0.50, "rare_to_common_ratio": 1.0},
        "e396_balanced": {"rare_action_recall": 0.51, "rare_to_common_ratio": 1.02},
        "sqrt_inverse_frequency": {
            "rare_action_recall": 0.52,
            "rare_to_common_ratio": 1.03,
        },
        "minimum_exposure_floor": {
            "rare_action_recall": 0.51,
            "rare_to_common_ratio": 1.01,
        },
        "root_template_balanced_control": {
            "rare_action_recall": 0.50,
            "rare_to_common_ratio": 1.00,
        },
    }
    disposition, _ = resolve_disposition(means)
    assert disposition == "no_exposure_lift"


def test_resolve_disposition_useful_exposure() -> None:
    means = {
        "current": {"rare_action_recall": 0.25, "rare_to_common_ratio": 0.5},
        "e396_balanced": {"rare_action_recall": 0.75, "rare_to_common_ratio": 2.0},
        "sqrt_inverse_frequency": {
            "rare_action_recall": 0.70,
            "rare_to_common_ratio": 1.8,
        },
        "minimum_exposure_floor": {
            "rare_action_recall": 0.80,
            "rare_to_common_ratio": 2.1,
        },
        "root_template_balanced_control": {
            "rare_action_recall": 0.72,
            "rare_to_common_ratio": 1.9,
        },
    }
    disposition, _ = resolve_disposition(means)
    assert disposition == "useful_rare_action_exposure"
