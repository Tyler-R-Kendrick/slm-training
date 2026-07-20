"""Tests for slm_training.harnesses.experiments.slm147_x22_retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from slm_training.data.semantic_plan.corpus import build_fixture_plan_corpus
from slm_training.harnesses.experiments.slm147_x22_retrieval import (
    RetrievalStrategy,
    Slm147Arm,
    Slm147Manifest,
    Slm147Report,
    Slm147Row,
    build_manifest,
    build_prototype_index,
    render_markdown,
    run_fixture_matrix,
)


def _valid_corpus(count: int = 16) -> dict[str, list[tuple[Any, Any]]]:
    return build_fixture_plan_corpus(
        count=count,
        seed=0,
        root_containers=["Stack", "Card"],
        leaf_components=["TextContent", "Button"],
    )


def test_manifest_has_all_eight_arms() -> None:
    manifest = build_manifest()
    ids = {arm.arm_id for arm in manifest.arms}
    expected = {
        "A_minimal_seed",
        "B_random_prototype",
        "C_prompt_similarity",
        "D_ast_sketch",
        "E_semantic_plan",
        "F_hybrid",
        "G_oracle_nearest",
        "H_retrieval_as_context",
    }
    assert ids == expected


def test_oracle_arm_is_not_promotable() -> None:
    manifest = build_manifest()
    oracle = next(arm for arm in manifest.arms if arm.arm_id == "G_oracle_nearest")
    assert oracle.promotable is False
    assert oracle.strategy is RetrievalStrategy.ORACLE_NEAREST


def test_retrieval_as_context_arm_is_promotable() -> None:
    manifest = build_manifest()
    control = next(
        arm for arm in manifest.arms if arm.arm_id == "H_retrieval_as_context"
    )
    assert control.promotable is True


def test_manifest_dataclass_round_trip() -> None:
    manifest = build_manifest()
    data = manifest.to_dict()
    restored = Slm147Manifest(
        matrix_set=data["matrix_set"],
        matrix_version=data["matrix_version"],
        experiment_id=data["experiment_id"],
        hypothesis=data["hypothesis"],
        falsifier=data["falsifier"],
        arms=tuple(Slm147Arm(**a) for a in data["arms"]),
        claim_class=data["claim_class"],
        status=data["status"],
    )
    assert restored.matrix_version == manifest.matrix_version
    assert len(restored.arms) == len(manifest.arms)


def test_prototype_index_is_train_only() -> None:
    corpus = _valid_corpus(count=16)
    index = build_prototype_index(corpus)
    assert index.manifest["split"] == "train"
    assert index.manifest["n_entries"] == len(corpus["train"])
    assert all(entry.record.split == "train" for entry in index.entries)


def test_fixture_matrix_produces_report(tmp_path: Path) -> None:
    corpus = _valid_corpus(count=16)
    report = run_fixture_matrix(
        corpus,
        run_id="test_slm147",
        output_dir=tmp_path,
    )
    assert report.status == "fixture"
    assert (tmp_path / "slm147_x22_retrieval_report.json").exists()
    ids = {(row.arm_id, row.seed) for row in report.rows}
    expected = {
        (arm.arm_id, seed)
        for arm in report.manifest.arms
        for seed in arm.seeds
    }
    assert ids == expected


def test_all_arms_produce_valid_seeds(tmp_path: Path) -> None:
    corpus = _valid_corpus(count=16)
    report = run_fixture_matrix(corpus, output_dir=tmp_path)
    for row in report.rows:
        assert row.seed_valid_count == row.n_records, f"{row.arm_id} seed {row.seed} invalid"


def test_oracle_arm_row_is_not_promotable(tmp_path: Path) -> None:
    corpus = _valid_corpus(count=16)
    report = run_fixture_matrix(corpus, output_dir=tmp_path)
    oracle_rows = [row for row in report.rows if row.arm_id == "G_oracle_nearest"]
    assert oracle_rows
    for row in oracle_rows:
        assert row.promotable is False


def test_leakage_pass_count_equals_n_records(tmp_path: Path) -> None:
    corpus = _valid_corpus(count=16)
    report = run_fixture_matrix(corpus, output_dir=tmp_path)
    for row in report.rows:
        assert row.leakage_pass_count == row.n_records, f"{row.arm_id} seed {row.seed} leaked"


def test_retrieval_as_context_matches_minimal_seed(tmp_path: Path) -> None:
    corpus = _valid_corpus(count=16)
    report = run_fixture_matrix(corpus, output_dir=tmp_path)
    minimal = {
        (row.seed, row.mean_seed_to_gold_ratio)
        for row in report.rows
        if row.arm_id == "A_minimal_seed"
    }
    context = {
        (row.seed, row.mean_seed_to_gold_ratio)
        for row in report.rows
        if row.arm_id == "H_retrieval_as_context"
    }
    assert minimal == context


def test_render_markdown_includes_caveat_and_arms() -> None:
    corpus = _valid_corpus(count=16)
    report = run_fixture_matrix(corpus)
    md = render_markdown(report)
    assert "SLM-147" in md
    assert "fixture only" in md.lower()
    for arm in report.manifest.arms:
        assert arm.arm_id in md


def test_report_to_dict_round_trip() -> None:
    manifest = build_manifest()
    row = Slm147Row(
        arm_id="x",
        strategy="minimal",
        seed=0,
        status="fixture",
        promotable=True,
        n_records=1,
        seed_valid_count=1,
        mean_seed_to_gold_ratio=0.5,
        mean_component_coverage=0.75,
        leakage_pass_count=1,
        adaptation_valid_count=0,
        mean_retrieval_score=None,
        notes=["note"],
    )
    report = Slm147Report(
        matrix_set="x",
        matrix_version="v1",
        experiment_id="x",
        run_id="test",
        status="fixture",
        manifest=manifest,
        rows=[row],
        index_manifest={},
    )
    data = report.to_dict()
    assert data["run_id"] == "test"
    assert data["rows"][0]["arm_id"] == "x"
