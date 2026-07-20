"""Tests for SLM-163 (SDE1-01) schema-description action-embedding fixture harness."""

from __future__ import annotations

import pytest

from slm_training.harnesses.experiments.slm163_schema_action_embedding import (
    MATRIX_SET,
    MATRIX_VERSION,
    EXPERIMENT_ID,
    build_manifest,
    run_fixture_campaign,
    validate_manifest,
)


@pytest.fixture
def arms() -> tuple:
    return build_manifest()


def test_build_manifest_has_required_arms(arms: tuple) -> None:
    sources = {arm.source for arm in arms}
    assert "none" in sources
    assert "current_stub" in sources
    assert "schema_description" in sources
    assert "expanded_description" in sources
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
    from slm_training.harnesses.experiments.slm163_schema_action_embedding import InitArm

    bad = InitArm(
        arm_id="Z_bad",
        source="schema_description",
        name="bad",
        description="bad",
        promotable=True,
    )
    errors = validate_manifest(arms + (bad,))
    assert any("promotable" in e for e in errors)


def test_run_fixture_campaign_produces_rows(arms: tuple) -> None:
    report = run_fixture_campaign(
        arms=arms,
        run_id="test_slm163",
        seeds=(0,),
        d_model=32,
    )
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert report.matrix_set == MATRIX_SET
    assert report.matrix_version == MATRIX_VERSION
    assert report.experiment_id == EXPERIMENT_ID
    assert report.rows
    assert report.version_stamp
    assert report.version_stamp["stamp_schema"] == "version_stamp/v1"
    assert "harness.experiments" in report.version_stamp.get("components", {})
    assert "harness.experiments.slm163_schema_action_embedding" in report.version_stamp.get("components", {})


def test_run_fixture_campaign_none_arm_has_zero_metrics(arms: tuple) -> None:
    report = run_fixture_campaign(
        arms=arms,
        run_id="test_slm163_none",
        seeds=(0,),
        d_model=32,
    )
    none_rows = [r for r in report.rows if r.source == "none"]
    assert none_rows
    for row in none_rows:
        assert row.n_actions == 0
        assert row.coverage_fraction == pytest.approx(0.0)
        assert row.mean_nearest_cosine == pytest.approx(0.0)


def test_run_fixture_campaign_schema_and_stub_differ(arms: tuple) -> None:
    report = run_fixture_campaign(
        arms=arms,
        run_id="test_slm163_diff",
        seeds=(0,),
        d_model=32,
    )
    schema = next(r for r in report.rows if r.source == "schema_description")
    stub = next(r for r in report.rows if r.source == "current_stub")
    # The two descriptions differ, so at least one metric should differ.
    assert (
        schema.mean_nearest_cosine != stub.mean_nearest_cosine
        or schema.sibling_separation != stub.sibling_separation
        or schema.rare_common_centroid_distance != stub.rare_common_centroid_distance
    )


def test_run_fixture_campaign_deterministic(arms: tuple) -> None:
    report1 = run_fixture_campaign(
        arms=arms,
        run_id="test_slm163_det1",
        seeds=(0,),
        d_model=32,
    )
    report2 = run_fixture_campaign(
        arms=arms,
        run_id="test_slm163_det2",
        seeds=(0,),
        d_model=32,
    )
    rows1 = {(r.arm_id, r.seed): r for r in report1.rows}
    rows2 = {(r.arm_id, r.seed): r for r in report2.rows}
    assert set(rows1.keys()) == set(rows2.keys())
    for key in rows1:
        assert rows1[key].mean_nearest_cosine == pytest.approx(rows2[key].mean_nearest_cosine, abs=1e-9)
        assert rows1[key].sibling_separation == pytest.approx(rows2[key].sibling_separation, abs=1e-9)
