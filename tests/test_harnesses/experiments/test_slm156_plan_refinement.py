"""Tests for SLM-156 (SPV3-03) shared recursive plan-refinement fixture harness."""

from __future__ import annotations

import pytest

from slm_training.harnesses.experiments.slm156_plan_refinement import (
    CommonConfig,
    RefinementArm,
    RefinementArmKind,
    RefinementManifest,
    build_manifest,
    run_fixture_campaign,
    validate_manifest,
)


@pytest.fixture
def manifest() -> RefinementManifest:
    return build_manifest()


def test_build_manifest_has_required_arms(manifest: RefinementManifest) -> None:
    arm_ids = {arm.arm_id for arm in manifest.arms}
    assert "A_one_pass" in arm_ids
    assert "B_deeper_non_shared" in arm_ids
    assert "C_shared_fixed_2" in arm_ids
    assert "D_shared_fixed_4" in arm_ids
    assert "E_shared_adaptive" in arm_ids
    assert "H_gold_oracle" in arm_ids


def test_validate_manifest_passes_for_default(manifest: RefinementManifest) -> None:
    assert validate_manifest(manifest) == []


def test_validate_manifest_rejects_duplicate_arm_ids(
    manifest: RefinementManifest,
) -> None:
    arms = list(manifest.arms)
    duplicated = arms + [arms[0]]
    bad = RefinementManifest(arms=tuple(duplicated))
    errors = validate_manifest(bad)
    assert any("duplicate arm_id" in e for e in errors)


def test_validate_manifest_rejects_promotable_gold_oracle(
    manifest: RefinementManifest,
) -> None:
    arms = list(manifest.arms)
    for i, arm in enumerate(arms):
        if arm.uses_gold:
            bad = RefinementArm(
                **{
                    **arm.__dict__,
                    "promotable": True,
                    "diagnostic": False,
                }
            )
            new_arms = arms.copy()
            new_arms[i] = bad
            bad_manifest = RefinementManifest(arms=tuple(new_arms))
            errors = validate_manifest(bad_manifest)
            assert any(arm.arm_id in e for e in errors)
            break
    else:
        pytest.skip("no gold arm to mutate")


def test_validate_manifest_rejects_promotable_diagnostic(
    manifest: RefinementManifest,
) -> None:
    arms = list(manifest.arms)
    for i, arm in enumerate(arms):
        if arm.diagnostic:
            bad = RefinementArm(
                **{
                    **arm.__dict__,
                    "promotable": True,
                }
            )
            new_arms = arms.copy()
            new_arms[i] = bad
            bad_manifest = RefinementManifest(arms=tuple(new_arms))
            errors = validate_manifest(bad_manifest)
            assert any(arm.arm_id in e for e in errors)
            break
    else:
        pytest.skip("no diagnostic arm to mutate")


def test_validate_manifest_enforces_adaptive_kind(
    manifest: RefinementManifest,
) -> None:
    arms = list(manifest.arms)
    one_pass = next(arm for arm in arms if arm.kind is RefinementArmKind.ONE_PASS)
    bad = RefinementArm(**{**one_pass.__dict__, "adaptive": True})
    new_arms = [bad if arm.arm_id == bad.arm_id else arm for arm in arms]
    bad_manifest = RefinementManifest(arms=tuple(new_arms))
    errors = validate_manifest(bad_manifest)
    assert any(bad.arm_id in e for e in errors)


def test_common_config_state_dim_is_derived() -> None:
    cfg = CommonConfig(num_archetypes=4, num_roles=8)
    assert cfg.state_dim == 4 + 8 + 1


def test_manifest_to_dict_roundtrip(manifest: RefinementManifest) -> None:
    data = manifest.to_dict()
    assert data["common_config"]["state_dim"] == manifest.common_config.state_dim
    restored = RefinementManifest(
        matrix_set=data.get("matrix_set", ""),
        matrix_version=data.get("matrix_version", ""),
        experiment_id=data.get("experiment_id", ""),
        hypothesis=data.get("hypothesis", ""),
        falsifier=data.get("falsifier", ""),
        common_config=CommonConfig.from_dict(data.get("common_config", {})),
        arms=tuple(RefinementArm.from_dict(a) for a in data.get("arms", [])),
        claim_class=data.get("claim_class", "wiring"),
        status=data.get("status", "not_run"),
    )
    assert restored.matrix_set == manifest.matrix_set
    assert {a.arm_id for a in restored.arms} == {a.arm_id for a in manifest.arms}
    assert restored.common_config.state_dim == manifest.common_config.state_dim


def test_run_fixture_campaign_produces_rows() -> None:
    pytest.importorskip("torch")
    cfg = CommonConfig(n_train=32, n_eval=8, epochs=5, seeds=(0,))
    report = run_fixture_campaign(
        manifest=RefinementManifest(common_config=cfg, arms=build_manifest().arms),
        run_id="test_slm156",
    )
    assert report.status == "fixture"
    assert report.rows
    promotable_rows = [r for r in report.rows if r.promotable]
    assert promotable_rows
    assert all(r.n_records > 0 for r in report.rows)
    assert report.version_stamp


def test_gold_and_diagnostic_arms_non_promotable(manifest: RefinementManifest) -> None:
    for arm in manifest.arms:
        if arm.uses_gold or arm.diagnostic:
            assert not arm.promotable


def test_shared_arms_use_tied_cell_parameters() -> None:
    """Fixed-depth and adaptive shared arms must share the same cell object."""
    pytest.importorskip("torch")
    from slm_training.harnesses.experiments.slm156_plan_refinement import (
        PlanRefinementModel,
    )

    cfg = CommonConfig(n_train=16, n_eval=4, epochs=2, seeds=(0,))
    manifest = RefinementManifest(common_config=cfg, arms=build_manifest().arms)
    report = run_fixture_campaign(manifest=manifest, run_id="test_shared")
    shared_rows = [
        r
        for r in report.rows
        if r.kind
        in {
            RefinementArmKind.SHARED_FIXED,
            RefinementArmKind.SHARED_ADAPTIVE,
        }
    ]
    assert shared_rows

    fixed_ids = {
        r.arm_id
        for r in shared_rows
        if r.kind is RefinementArmKind.SHARED_FIXED
    }
    adaptive_ids = {
        r.arm_id
        for r in shared_rows
        if r.kind is RefinementArmKind.SHARED_ADAPTIVE
    }
    assert fixed_ids
    assert adaptive_ids

    # A model built with the same cell shares parameters across depths.
    cell = PlanRefinementModel(cfg.state_dim, max_depth=cfg.max_depth).cell
    shallow = PlanRefinementModel(cfg.state_dim, cell=cell, max_depth=2)
    deep = PlanRefinementModel(cfg.state_dim, cell=cell, max_depth=4)
    assert shallow.cell is deep.cell
    for p1, p2 in zip(shallow.parameters(), deep.parameters()):
        assert p1 is p2


def test_trace_envelope_present() -> None:
    pytest.importorskip("torch")
    cfg = CommonConfig(n_train=16, n_eval=4, epochs=2, seeds=(0,))
    manifest = RefinementManifest(common_config=cfg, arms=build_manifest().arms)
    report = run_fixture_campaign(manifest=manifest, run_id="test_trace")
    for row in report.rows:
        assert row.n_records > 0
        assert row.mean_forwards >= 0
        assert row.mean_depth >= 0


def test_deterministic_replay() -> None:
    pytest.importorskip("torch")
    cfg = CommonConfig(n_train=16, n_eval=4, epochs=2, seeds=(0,))
    manifest = RefinementManifest(common_config=cfg, arms=build_manifest().arms)
    report1 = run_fixture_campaign(manifest=manifest, run_id="replay1")
    report2 = run_fixture_campaign(manifest=manifest, run_id="replay2")
    rows1 = {f"{r.arm_id}-{r.seed}": r for r in report1.rows}
    rows2 = {f"{r.arm_id}-{r.seed}": r for r in report2.rows}
    assert set(rows1) == set(rows2)
    for key in rows1:
        assert rows1[key].mean_plan_score == pytest.approx(
            rows2[key].mean_plan_score, abs=1e-6
        )
        assert rows1[key].mean_forwards == pytest.approx(
            rows2[key].mean_forwards, abs=1e-6
        )
        assert rows1[key].mean_depth == pytest.approx(
            rows2[key].mean_depth, abs=1e-6
        )
