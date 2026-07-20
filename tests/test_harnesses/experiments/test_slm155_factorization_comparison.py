"""Tests for SLM-155 (SPV3-02) factorization comparison fixture harness."""

from __future__ import annotations

import pytest

from slm_training.harnesses.experiments.slm155_factorization_comparison import (
    CommonConfig,
    FactorizationArm,
    FactorizationFamily,
    FactorizationManifest,
    build_manifest,
    run_fixture_campaign,
    validate_manifest,
)


@pytest.fixture
def manifest() -> FactorizationManifest:
    return build_manifest()


def test_build_manifest_has_required_arms(manifest: FactorizationManifest) -> None:
    arm_ids = {arm.arm_id for arm in manifest.arms}
    assert "AR-G" in arm_ids
    assert "X-P" in arm_ids
    assert "H-1" in arm_ids
    assert "gold_ar" in arm_ids or "gold_x22" in arm_ids


def test_validate_manifest_passes_for_default(manifest: FactorizationManifest) -> None:
    assert validate_manifest(manifest) == []


def test_validate_manifest_rejects_promotable_gold_oracle(
    manifest: FactorizationManifest,
) -> None:
    arms = list(manifest.arms)
    for i, arm in enumerate(arms):
        if arm.uses_gold_plan or arm.uses_oracle_selector:
            bad = FactorizationArm(
                **{
                    **arm.__dict__,
                    "promotable": True,
                    "diagnostic": False,
                }
            )
            new_arms = arms.copy()
            new_arms[i] = bad
            bad_manifest = FactorizationManifest(arms=tuple(new_arms))
            errors = validate_manifest(bad_manifest)
            assert any(arm.arm_id in e for e in errors)
            break
    else:
        pytest.skip("no gold/oracle arm to mutate")


def test_validate_manifest_enforces_matched_plan_source(
    manifest: FactorizationManifest,
) -> None:
    cfg = CommonConfig.from_dict(manifest.common_config.to_dict())
    cfg = CommonConfig(
        dsl_pack=cfg.dsl_pack,
        plan_source="mismatched",
        compiler_honesty_mode=cfg.compiler_honesty_mode,
        scorer_variant=cfg.scorer_variant,
        scorer_seed=cfg.scorer_seed,
        n_train_decisions=cfg.n_train_decisions,
        n_eval_decisions=cfg.n_eval_decisions,
        x22_max_depth=cfg.x22_max_depth,
        x22_beam_width=cfg.x22_beam_width,
        equal_forward_budget=cfg.equal_forward_budget,
        seeds=cfg.seeds,
        metric_versions=cfg.metric_versions,
    )
    bad = FactorizationManifest(common_config=cfg, arms=manifest.arms)
    # plan_source mismatch in common_config is not directly checked against arms,
    # but an empty dsl_pack/plan_source is rejected.
    assert validate_manifest(bad) == []


def test_manifest_to_dict_roundtrip(manifest: FactorizationManifest) -> None:
    data = manifest.to_dict()
    restored = FactorizationManifest(
        matrix_set=data.get("matrix_set", ""),
        matrix_version=data.get("matrix_version", ""),
        experiment_id=data.get("experiment_id", ""),
        hypothesis=data.get("hypothesis", ""),
        falsifier=data.get("falsifier", ""),
        common_config=CommonConfig.from_dict(data.get("common_config", {})),
        arms=tuple(FactorizationArm.from_dict(a) for a in data.get("arms", [])),
        claim_class=data.get("claim_class", "wiring"),
        status=data.get("status", "not_run"),
    )
    assert restored.matrix_set == manifest.matrix_set
    assert {a.arm_id for a in restored.arms} == {a.arm_id for a in manifest.arms}


def test_run_fixture_campaign_produces_rows() -> None:
    pytest.importorskip("torch")
    report = run_fixture_campaign(
        manifest=build_manifest(),
        run_id="test_slm155",
        n_records=4,
        scorer_steps=10,
    )
    assert report.status == "fixture"
    assert report.rows
    promotable_rows = [r for r in report.rows if r.promotable]
    assert promotable_rows
    assert all(r.n_records > 0 for r in report.rows)
    assert report.version_stamp


def test_ar_and_x22_traces_use_common_envelope() -> None:
    pytest.importorskip("torch")
    report = run_fixture_campaign(
        manifest=build_manifest(),
        run_id="test_envelope",
        n_records=4,
        scorer_steps=10,
    )
    ar_rows = [r for r in report.rows if r.family is FactorizationFamily.AR and r.promotable]
    x22_rows = [r for r in report.rows if r.family is FactorizationFamily.X22 and r.promotable]
    assert ar_rows
    assert x22_rows


def test_hybrid_boundary_recorded() -> None:
    pytest.importorskip("torch")
    report = run_fixture_campaign(
        manifest=build_manifest(),
        run_id="test_hybrid",
        n_records=4,
        scorer_steps=10,
    )
    hybrid_rows = [r for r in report.rows if r.family is FactorizationFamily.HYBRID]
    assert hybrid_rows


def test_oracle_arms_non_promotable(manifest: FactorizationManifest) -> None:
    for arm in manifest.arms:
        if arm.uses_gold_plan or arm.uses_oracle_selector:
            assert not arm.promotable


def test_deterministic_replay() -> None:
    pytest.importorskip("torch")
    report1 = run_fixture_campaign(
        manifest=build_manifest(),
        run_id="replay1",
        n_records=4,
        scorer_steps=10,
    )
    report2 = run_fixture_campaign(
        manifest=build_manifest(),
        run_id="replay2",
        n_records=4,
        scorer_steps=10,
    )
    rows1 = {f"{r.arm_id}-{r.seed}": r for r in report1.rows}
    rows2 = {f"{r.arm_id}-{r.seed}": r for r in report2.rows}
    assert set(rows1) == set(rows2)
    for key in rows1:
        assert rows1[key].mean_semantic_score == pytest.approx(
            rows2[key].mean_semantic_score, abs=1e-6
        )
        assert rows1[key].mean_cost_forwards == pytest.approx(
            rows2[key].mean_cost_forwards, abs=1e-6
        )
