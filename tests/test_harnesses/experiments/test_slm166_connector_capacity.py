"""Tests for SLM-166 (SDE1-04) connector capacity fixture harness."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm166_connector_capacity import (
    ARM_NAMES,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    ConnectorArm,
    ConnectorReport,
    build_cells,
    resolve_disposition,
    run_fixture_campaign,
    validate_manifest,
)


def test_build_cells_produces_seven_arms_per_seed() -> None:
    cells = build_cells(seeds=(0, 1, 2))
    assert len(cells) == 21
    per_seed = {}
    for cell in cells:
        per_seed.setdefault(cell.seed, set()).add(cell.arm_id)
    assert len(per_seed) == 3
    for seed, ids in per_seed.items():
        assert len(ids) == 7, f"seed {seed} has {len(ids)} cells"


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


def test_validate_manifest_rejects_invalid_connector_type() -> None:
    cells = build_cells(seeds=(0,))
    bad = ConnectorArm(
        arm_id="bad",
        arm_name="linear",
        connector_type="invalid",
        train_scope="connector_only",
        seed=0,
        d_model=64,
        connector_hidden_dim=256,
        connector_rank=32,
        connector_n_queries=4,
        connector_freeze_encoder=True,
        target_decisions=1000,
    )
    errors = validate_manifest(cells + (bad,))
    assert any("invalid connector_type" in e for e in errors)


def test_validate_manifest_rejects_invalid_train_scope() -> None:
    cells = build_cells(seeds=(0,))
    bad = ConnectorArm(
        arm_id="bad",
        arm_name="linear",
        connector_type="linear",
        train_scope="invalid",
        seed=0,
        d_model=64,
        connector_hidden_dim=256,
        connector_rank=32,
        connector_n_queries=4,
        connector_freeze_encoder=True,
        target_decisions=1000,
    )
    errors = validate_manifest(cells + (bad,))
    assert any("invalid train_scope" in e for e in errors)


def test_shuffled_context_near_baseline() -> None:
    report = run_fixture_campaign(seeds=(0, 1, 2))
    means = report.arm_means
    current = means["current"]["rare_component_recall"]
    shuffled = means["linear_shuffled_context"]["rare_component_recall"]
    linear = means["linear"]["rare_component_recall"]
    assert abs(shuffled - current) < 0.05
    assert linear - shuffled > 0.03


def test_capacity_ordering_on_rare_recall() -> None:
    report = run_fixture_campaign(seeds=(0, 1, 2))
    means = report.arm_means
    linear = means["linear"]["rare_component_recall"]
    low_rank = means["low_rank"]["rare_component_recall"]
    cross = means["cross_attention"]["rare_component_recall"]
    control = means["local_target"]["rare_component_recall"]
    assert linear < low_rank < cross < control


def test_disposition_decoder_adaptation_needed() -> None:
    report = run_fixture_campaign(seeds=(0, 1, 2))
    assert report.disposition == "decoder_adaptation_needed"
    assert "cross-attention" in report.disposition_rationale.lower()
    assert "small-model control" in report.disposition_rationale.lower()


def test_resolve_disposition_data_limited() -> None:
    means = {
        "current": {"rare_component_recall": 0.35},
        "linear": {"rare_component_recall": 0.36},
        "low_rank": {"rare_component_recall": 0.36},
        "cross_attention": {"rare_component_recall": 0.36},
        "local_target": {"rare_component_recall": 0.36},
    }
    disposition, _ = resolve_disposition(means)
    assert disposition == "data_or_objective_limited"


def test_resolve_disposition_linear_sufficient() -> None:
    means = {
        "current": {"rare_component_recall": 0.35},
        "linear": {"rare_component_recall": 0.55},
        "low_rank": {"rare_component_recall": 0.56},
        "cross_attention": {"rare_component_recall": 0.56},
        "local_target": {"rare_component_recall": 0.57},
    }
    disposition, _ = resolve_disposition(means)
    assert disposition == "linear_sufficient"


def test_report_round_trip() -> None:
    report = run_fixture_campaign(seeds=(0,))
    reconstructed = ConnectorReport.from_dict(report.to_dict())
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
    assert "harness.experiments.slm166_connector_capacity" in components
    assert "model.twotower" in components
