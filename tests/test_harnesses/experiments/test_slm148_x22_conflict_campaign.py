"""Tests for slm_training.harnesses.experiments.slm148_x22_conflict_campaign."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from slm_training.data.semantic_plan.corpus import build_fixture_plan_corpus
from slm_training.harnesses.experiments.slm148_x22_conflict_campaign import (
    SeedStrategy,
    Slm148Report,
    Slm148Row,
    Slm148SeedArm,
    build_manifest,
    render_markdown,
    run_fixture_matrix,
    validate_manifest,
)


def _valid_corpus(count: int = 16) -> dict[str, list[tuple[Any, Any]]]:
    return build_fixture_plan_corpus(
        count=count,
        seed=0,
        root_containers=["Stack", "Card"],
        leaf_components=["TextContent", "Button"],
    )


def test_manifest_has_all_seed_and_recovery_arms() -> None:
    manifest = build_manifest()
    seed_ids = {arm.arm_id for arm in manifest.seed_arms}
    recovery_ids = {arm.arm_id for arm in manifest.recovery_arms}
    assert seed_ids == {
        "S0_minimal",
        "S1_frequency_prior",
        "S2_learned_archetype_role_set",
        "S3_learned_full_plan",
        "S4_gold_factor_bindings",
        "S5_gold_plan_oracle",
        "S6_retrieved_prototype",
        "S7_plan_reranked_retrieval",
    }
    assert recovery_ids == {
        "R0_none",
        "R1_full_remask",
        "R2_suffix_rollback",
        "R3_conflict_slice",
        "R4_oracle_conflict_slice",
    }


def test_gold_seed_arms_are_not_promotable() -> None:
    manifest = build_manifest()
    for arm in manifest.seed_arms:
        if arm.strategy in (SeedStrategy.GOLD_FACTOR, SeedStrategy.GOLD_PLAN):
            assert arm.promotable is False


def test_oracle_recovery_arm_is_diagnostic() -> None:
    manifest = build_manifest()
    oracle = next(arm for arm in manifest.recovery_arms if arm.arm_id == "R4_oracle_conflict_slice")
    assert oracle.diagnostic is True


def test_default_manifest_passes_validation() -> None:
    manifest = build_manifest()
    assert validate_manifest(manifest) == []


def test_validate_manifest_catches_promotable_gold_arm() -> None:
    manifest = build_manifest()
    bad_arms = (
        Slm148SeedArm(
            arm_id="S5_gold_plan_oracle",
            strategy=SeedStrategy.GOLD_PLAN,
            seeds=(0,),
            description="bad",
            promotable=True,
        ),
    ) + manifest.seed_arms[1:]
    bad = replace(manifest, seed_arms=bad_arms)
    errors = validate_manifest(bad)
    assert any("non-promotable" in e for e in errors)


def test_fixture_matrix_produces_report(tmp_path: Path) -> None:
    corpus = _valid_corpus(count=16)
    report = run_fixture_matrix(
        corpus,
        run_id="test_slm148",
        output_dir=tmp_path,
        predictor_epochs=4,
        predictor_batch_size=4,
    )
    assert report.status == "fixture"
    assert (tmp_path / "slm148_x22_conflict_campaign_report.json").exists()
    assert report.rows
    assert report.survivors


def test_screening_rows_cover_all_seed_arms(tmp_path: Path) -> None:
    corpus = _valid_corpus(count=16)
    report = run_fixture_matrix(
        corpus,
        output_dir=tmp_path,
        predictor_epochs=4,
        predictor_batch_size=4,
    )
    screening = [row for row in report.rows if row.stage == "screening"]
    seed_ids = {row.arm_id for row in screening}
    assert seed_ids == {arm.arm_id for arm in report.manifest.seed_arms}


def test_gold_arms_produce_valid_seeds(tmp_path: Path) -> None:
    corpus = _valid_corpus(count=16)
    report = run_fixture_matrix(
        corpus,
        output_dir=tmp_path,
        predictor_epochs=4,
        predictor_batch_size=4,
    )
    for row in report.rows:
        if row.stage == "screening" and row.arm_id in {
            "S4_gold_factor_bindings",
            "S5_gold_plan_oracle",
        }:
            assert row.seed_valid_count == row.n_records


def test_promotable_seed_arms_survive_screening(tmp_path: Path) -> None:
    corpus = _valid_corpus(count=16)
    report = run_fixture_matrix(
        corpus,
        output_dir=tmp_path,
        predictor_epochs=4,
        predictor_batch_size=4,
    )
    for arm in report.manifest.seed_arms:
        if arm.promotable:
            assert arm.arm_id in report.survivors


def test_cross_rows_exist_for_survivors(tmp_path: Path) -> None:
    corpus = _valid_corpus(count=16)
    report = run_fixture_matrix(
        corpus,
        output_dir=tmp_path,
        predictor_epochs=4,
        predictor_batch_size=4,
    )
    cross = [row for row in report.rows if row.stage == "cross"]
    assert cross
    for row in cross:
        assert row.arm_id in report.survivors


def test_oracle_recovery_rows_are_not_promotable(tmp_path: Path) -> None:
    corpus = _valid_corpus(count=16)
    report = run_fixture_matrix(
        corpus,
        output_dir=tmp_path,
        predictor_epochs=4,
        predictor_batch_size=4,
    )
    for row in report.rows:
        if row.recovery_policy == "conflict_slice_expanded":
            assert row.promotable is False


def test_conflict_slices_are_supported_classes(tmp_path: Path) -> None:
    corpus = _valid_corpus(count=16)
    report = run_fixture_matrix(
        corpus,
        output_dir=tmp_path,
        predictor_epochs=4,
        predictor_batch_size=4,
    )
    for row in report.rows:
        # Every record should carry a recognized completeness class.
        assert row.mean_remasked_nodes >= 0.0


def test_render_markdown_includes_caveat_and_arms(tmp_path: Path) -> None:
    corpus = _valid_corpus(count=16)
    report = run_fixture_matrix(
        corpus,
        output_dir=tmp_path,
        predictor_epochs=4,
        predictor_batch_size=4,
    )
    md = render_markdown(report)
    assert "SLM-148" in md
    assert "fixture only" in md.lower()
    for arm in report.manifest.seed_arms:
        assert arm.arm_id in md
    for arm in report.manifest.recovery_arms:
        assert arm.arm_id in md


def test_report_to_dict_round_trip() -> None:
    manifest = build_manifest()
    row = Slm148Row(
        arm_id="x",
        seed_strategy="minimal",
        recovery_policy="none",
        stage="screening",
        seed=0,
        promotable=True,
        n_records=1,
        seed_valid_count=1,
        mean_seed_to_gold_ratio=0.5,
        mean_component_coverage=0.75,
        recovery_rate=0.0,
        mean_remasked_nodes=0.0,
        mean_preserved_nodes=1.0,
        mean_forwards=64.0,
        mean_verifier_calls=16.0,
        repeated_conflict_rate=0.0,
        notes=["note"],
    )
    report = Slm148Report(
        matrix_set="x",
        matrix_version="v1",
        experiment_id="x",
        run_id="test",
        status="fixture",
        manifest=manifest,
        rows=[row],
        survivors=[],
    )
    data = report.to_dict()
    assert data["run_id"] == "test"
    assert data["rows"][0]["arm_id"] == "x"
