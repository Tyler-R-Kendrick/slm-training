"""Tests for SLM-157 (SPV3-04) flow-consistency fixture harness."""

from __future__ import annotations

import pytest

from slm_training.dsl.parser import validate
from slm_training.harnesses.experiments.slm157_flow_consistency import (
    CommonConfig,
    FlowArm,
    FlowManifest,
    build_manifest,
    run_fixture_campaign,
    validate_manifest,
)
from slm_training.models.tree_edit_diffusion import (
    ACTION_STOP,
    Edit,
    TreeEditSpace,
    parse_statements,
)


@pytest.fixture
def manifest() -> FlowManifest:
    return build_manifest()


def test_build_manifest_has_required_arms(manifest: FlowManifest) -> None:
    arm_ids = {arm.arm_id for arm in manifest.arms}
    assert "A_teacher_long_x22" in arm_ids
    assert "B_direct_trajectory_imitation" in arm_ids
    assert "C_consistency_student_x22" in arm_ids
    assert "D_consistency_student_coarse" in arm_ids
    assert "E_discrete_flow_rate" in arm_ids
    assert "F_random_path_control" in arm_ids
    assert "G_ar_x22_hybrid_placeholder" in arm_ids
    assert "H_oracle_boundary" in arm_ids


def test_build_manifest_oracle_is_diagnostic(manifest: FlowManifest) -> None:
    oracle = next(arm for arm in manifest.arms if arm.arm_id == "H_oracle_boundary")
    assert oracle.diagnostic
    assert not oracle.promotable


def test_build_manifest_all_non_oracle_non_promotable(manifest: FlowManifest) -> None:
    for arm in manifest.arms:
        if arm.arm_id != "H_oracle_boundary":
            assert not arm.promotable
            assert not arm.diagnostic


def test_validate_manifest_passes_for_default(manifest: FlowManifest) -> None:
    assert validate_manifest(manifest) == []


def test_validate_manifest_rejects_duplicate_arm_ids(manifest: FlowManifest) -> None:
    arms = list(manifest.arms)
    duplicated = arms + [arms[0]]
    bad = FlowManifest(arms=tuple(duplicated))
    errors = validate_manifest(bad)
    assert any("duplicate arm_id" in e for e in errors)


def test_validate_manifest_rejects_promotable_blocked(manifest: FlowManifest) -> None:
    arms = list(manifest.arms)
    blocked = FlowArm(
        **{
            **arms[0].__dict__,
            "blocked": True,
            "promotable": True,
            "blocker": "test blocker",
        }
    )
    new_arms = [blocked if arm.arm_id == blocked.arm_id else arm for arm in arms]
    bad = FlowManifest(arms=tuple(new_arms))
    errors = validate_manifest(bad)
    assert any(blocked.arm_id in e for e in errors)


def test_validate_manifest_rejects_promotable_diagnostic(manifest: FlowManifest) -> None:
    arms = list(manifest.arms)
    diagnostic = FlowArm(
        **{
            **arms[0].__dict__,
            "diagnostic": True,
            "promotable": True,
        }
    )
    new_arms = [diagnostic if arm.arm_id == diagnostic.arm_id else arm for arm in arms]
    bad = FlowManifest(arms=tuple(new_arms))
    errors = validate_manifest(bad)
    assert any(diagnostic.arm_id in e for e in errors)


def test_common_config_roundtrip() -> None:
    cfg = CommonConfig()
    data = cfg.to_dict()
    restored = CommonConfig.from_dict(data)
    assert restored.seeds == cfg.seeds
    assert restored.steps_list == cfg.steps_list
    assert restored.n_records == cfg.n_records


def test_manifest_to_dict_roundtrip(manifest: FlowManifest) -> None:
    data = manifest.to_dict()
    restored = FlowManifest.from_dict(data)
    assert restored.matrix_set == manifest.matrix_set
    assert {a.arm_id for a in restored.arms} == {a.arm_id for a in manifest.arms}
    assert restored.common_config.seeds == manifest.common_config.seeds


def test_run_fixture_campaign_produces_rows(manifest: FlowManifest) -> None:
    cfg = CommonConfig(n_records=2, seeds=(0,), steps_list=(4,))
    report = run_fixture_campaign(
        manifest=FlowManifest(common_config=cfg, arms=manifest.arms),
        run_id="test_slm157",
    )
    assert report.status == "fixture"
    assert report.rows
    assert report.version_stamp
    assert report.version_stamp["stamp_schema"] == "version_stamp/v1"
    assert "harness.experiments" in report.version_stamp.get("components", {})
    assert "harness.experiments.slm157_flow_consistency" in report.version_stamp.get("components", {})


def test_run_fixture_campaign_deterministic(manifest: FlowManifest) -> None:
    cfg = CommonConfig(n_records=2, seeds=(0,), steps_list=(4,))
    report1 = run_fixture_campaign(
        manifest=FlowManifest(common_config=cfg, arms=manifest.arms),
        run_id="test_determinism_1",
    )
    report2 = run_fixture_campaign(
        manifest=FlowManifest(common_config=cfg, arms=manifest.arms),
        run_id="test_determinism_2",
    )
    rows1 = {(r.arm_id, r.seed, r.steps): r for r in report1.rows}
    rows2 = {(r.arm_id, r.seed, r.steps): r for r in report2.rows}
    assert set(rows1.keys()) == set(rows2.keys())
    for key in rows1:
        assert rows1[key].target_reach_rate == pytest.approx(rows2[key].target_reach_rate, abs=1e-9)
        assert rows1[key].mean_remaining_distance == pytest.approx(
            rows2[key].mean_remaining_distance, abs=1e-9
        )


def test_run_fixture_campaign_all_materialized_states_valid(manifest: FlowManifest) -> None:
    cfg = CommonConfig(n_records=2, seeds=(0,), steps_list=(4,))
    report = run_fixture_campaign(
        manifest=FlowManifest(common_config=cfg, arms=manifest.arms),
        run_id="test_validity",
    )
    space = TreeEditSpace()
    for row in report.rows:
        assert row.state_validity == pytest.approx(1.0, abs=1e-9)
        assert row.transition_validity == pytest.approx(1.0, abs=1e-9)
        for record in row.records:
            for program in (record.source, record.target, record.final_state):
                assert parse_statements(program) is not None
                validate(program)
            # Final state must be hard-valid through TreeEditSpace as well.
            final_statements = parse_statements(record.final_state)
            assert final_statements is not None
            assert space.apply(final_statements, Edit(ACTION_STOP), []) is not None
