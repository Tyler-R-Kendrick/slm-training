"""Tests for SLM-164 (SDE1-02) confusion-targeted legal-sibling margin fixture harness."""

from __future__ import annotations

import pytest

from slm_training.harnesses.experiments.slm164_targeted_margin import (
    MATRIX_SET,
    MATRIX_VERSION,
    EXPERIMENT_ID,
    ContrastFamily,
    build_default_families,
    build_manifest,
    build_synthetic_manifest,
    compute_targeted_margin_loss,
    run_fixture_campaign,
    validate_manifest,
)


@pytest.fixture
def arms() -> tuple:
    return build_manifest()


def test_contrast_family_exposes_all_families() -> None:
    families = build_default_families()
    expected = list(ContrastFamily().to_dict().values())
    assert families == expected
    assert "empty_vs_child" in families
    assert "stack_vs_card" in families
    assert "rare_component_substitution" in families
    assert "binder_arity" in families
    assert "slot_pointer" in families
    assert "same_type_different_role" in families


def test_build_synthetic_manifest_covers_all_families() -> None:
    manifest = build_synthetic_manifest(seed=0)
    assert len(manifest.rows) >= 20
    families = {row.family for row in manifest.rows}
    assert families == set(build_default_families())


def test_build_manifest_has_required_arms(arms: tuple) -> None:
    sources = {arm.source for arm in arms}
    assert "none" in sources
    assert "uniform" in sources
    assert "targeted_hardest" in sources
    assert "targeted_weighted" in sources
    assert "shuffled" in sources


def test_build_manifest_arms_non_promotable(arms: tuple) -> None:
    for arm in arms:
        assert not arm.promotable


def test_validate_manifest_passes_for_default(arms: tuple) -> None:
    assert validate_manifest(arms) == []


def test_validate_manifest_rejects_duplicate_arm_ids(arms: tuple) -> None:
    duplicated = arms + (arms[0],)
    errors = validate_manifest(duplicated)
    assert any("duplicate arm_id" in e for e in errors)


def test_validate_manifest_rejects_promotable(arms: tuple) -> None:
    from slm_training.harnesses.experiments.slm164_targeted_margin import (
        TargetedMarginArm,
    )

    bad = TargetedMarginArm(
        arm_id="Z_bad",
        source="uniform",
        name="bad",
        description="bad",
        promotable=True,
    )
    errors = validate_manifest(arms + (bad,))
    assert any("promotable" in e for e in errors)


def test_compute_targeted_margin_loss_none_is_zero() -> None:
    scores = {"a": 1.0, "b": 2.0, "c": 0.5}
    loss, violation = compute_targeted_margin_loss(scores, "a", ("b", "c"), 1.0, "none")
    assert loss == pytest.approx(0.0)
    assert violation is False


def test_compute_targeted_margin_loss_uniform_violation() -> None:
    scores = {"expected": 0.0, "bad": 1.5}
    loss, violation = compute_targeted_margin_loss(
        scores, "expected", ("bad",), 1.0, "uniform"
    )
    assert loss == pytest.approx(2.5)
    assert violation is True


def test_compute_targeted_margin_loss_no_violation() -> None:
    scores = {"expected": 2.0, "bad": 0.5}
    loss, violation = compute_targeted_margin_loss(
        scores, "expected", ("bad",), 1.0, "uniform"
    )
    assert loss == pytest.approx(0.0)
    assert violation is False


def test_run_fixture_campaign_produces_rows(arms: tuple) -> None:
    report = run_fixture_campaign(
        arms=arms,
        run_id="test_slm164",
        seeds=(0,),
        margin=1.0,
    )
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert report.matrix_set == MATRIX_SET
    assert report.matrix_version == MATRIX_VERSION
    assert report.experiment_id == EXPERIMENT_ID
    assert report.rows
    assert report.manifest.rows
    assert report.version_stamp
    assert report.version_stamp["stamp_schema"] == "version_stamp/v1"
    assert "harness.experiments" in report.version_stamp.get("components", {})
    assert "harness.experiments.slm164_targeted_margin" in report.version_stamp.get(
        "components", {}
    )


def test_run_fixture_campaign_none_arm_has_zero_metrics(arms: tuple) -> None:
    report = run_fixture_campaign(
        arms=arms,
        run_id="test_slm164_none",
        seeds=(0,),
        margin=1.0,
    )
    none_rows = [r for r in report.rows if r.source == "none"]
    assert none_rows
    for row in none_rows:
        assert row.mean_margin_loss == pytest.approx(0.0)
        assert row.violation_rate == pytest.approx(0.0)


def test_run_fixture_campaign_targeted_arms_non_zero(arms: tuple) -> None:
    report = run_fixture_campaign(
        arms=arms,
        run_id="test_slm164_targeted",
        seeds=(0,),
        margin=1.0,
    )
    for source in ("uniform", "targeted_hardest", "targeted_weighted"):
        row = next(r for r in report.rows if r.source == source)
        assert row.mean_margin_loss > 0.0
        assert row.violation_rate > 0.0


def test_run_fixture_campaign_shuffled_differs(arms: tuple) -> None:
    report = run_fixture_campaign(
        arms=arms,
        run_id="test_slm164_shuffled",
        seeds=(0,),
        margin=1.0,
    )
    targeted = next(r for r in report.rows if r.source == "targeted_weighted")
    shuffled = next(r for r in report.rows if r.source == "shuffled")
    assert targeted.family_violation_rates != shuffled.family_violation_rates
