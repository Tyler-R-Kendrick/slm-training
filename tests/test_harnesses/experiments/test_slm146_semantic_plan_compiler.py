"""Tests for slm_training.harnesses.experiments.slm146_semantic_plan_compiler."""

from __future__ import annotations

from pathlib import Path
from typing import Any


from slm_training.data.semantic_plan.corpus import build_fixture_plan_corpus


def _valid_corpus(count: int = 16) -> dict[str, list[tuple[Any, Any]]]:
    return build_fixture_plan_corpus(
        count=count,
        seed=0,
        root_containers=["Stack", "Card"],
        leaf_components=["TextContent", "Button"],
    )
from slm_training.harnesses.experiments.slm146_semantic_plan_compiler import (
    Slm146Arm,
    Slm146Manifest,
    Slm146Report,
    Slm146Row,
    build_manifest,
    render_markdown,
    run_fixture_matrix,
)


def test_manifest_has_all_six_arms() -> None:
    manifest = build_manifest()
    ids = {arm.arm_id for arm in manifest.arms}
    expected = {
        "A_baseline",
        "B_gold_seed",
        "C_gold_seed_soft",
        "D_baseline_soft",
        "E_certified_restrictions",
        "F_unsafe_predicted_hard",
    }
    assert ids == expected


def test_unsafe_arm_is_not_promotable() -> None:
    manifest = build_manifest()
    unsafe = next(arm for arm in manifest.arms if arm.arm_id == "F_unsafe_predicted_hard")
    assert unsafe.promotable is False


def test_manifest_dataclass_round_trip() -> None:
    manifest = build_manifest()
    data = manifest.to_dict()
    restored = Slm146Manifest(
        matrix_set=data["matrix_set"],
        matrix_version=data["matrix_version"],
        hypothesis=data["hypothesis"],
        falsifier=data["falsifier"],
        arms=[Slm146Arm(**a) for a in data["arms"]],
        claim_class=data["claim_class"],
        status=data["status"],
    )
    assert restored.matrix_version == manifest.matrix_version
    assert len(restored.arms) == len(manifest.arms)


def test_fixture_matrix_produces_report(tmp_path: Path) -> None:
    corpus = _valid_corpus(count=16)
    report = run_fixture_matrix(
        corpus,
        run_id="test_slm146",
        output_dir=tmp_path,
    )
    assert report.status == "fixture"
    ids = {row.arm_id for row in report.rows}
    assert ids == {arm.arm_id for arm in report.manifest.arms}
    assert (tmp_path / "slm146_semantic_plan_compiler_report.json").exists()


def test_promotable_arms_have_zero_false_hard_prunes(tmp_path: Path) -> None:
    corpus = _valid_corpus(count=16)
    report = run_fixture_matrix(corpus, output_dir=tmp_path)
    for row in report.rows:
        if row.promotable:
            assert row.total_false_hard_prunes == 0, f"{row.arm_id} has false prunes"


def test_unsafe_arm_has_hard_removals_and_is_not_promotable(tmp_path: Path) -> None:
    corpus = _valid_corpus(count=16)
    report = run_fixture_matrix(corpus, output_dir=tmp_path)
    unsafe = next(row for row in report.rows if row.arm_id == "F_unsafe_predicted_hard")
    assert unsafe.promotable is False
    assert unsafe.total_hard_removals > 0


def test_gold_seed_arm_has_valid_seeds(tmp_path: Path) -> None:
    corpus = _valid_corpus(count=16)
    report = run_fixture_matrix(corpus, output_dir=tmp_path)
    gold = next(row for row in report.rows if row.arm_id == "B_gold_seed")
    assert gold.seed_ok_count == gold.n_records
    assert gold.seed_valid_count == gold.n_records


def test_render_markdown_includes_caveat_and_arms() -> None:
    corpus = _valid_corpus(count=16)
    report = run_fixture_matrix(corpus)
    md = render_markdown(report)
    assert "SLM-146" in md
    assert "fixture only" in md.lower()
    for arm in report.manifest.arms:
        assert arm.arm_id in md


def test_report_to_dict_round_trip() -> None:
    manifest = build_manifest()
    row = Slm146Row(
        arm_id="x",
        status="fixture",
        promotable=True,
        n_records=1,
        seed_ok_count=1,
        seed_valid_count=1,
        mean_seed_to_gold_ratio=0.5,
        mean_role_coverage=1.0,
        mean_topology_coverage=1.0,
        mean_binding_coverage=1.0,
        total_soft_features=2,
        total_hard_removals=0,
        total_false_hard_prunes=0,
    )
    report = Slm146Report(
        matrix_set="x",
        matrix_version="v1",
        run_id="test",
        status="fixture",
        manifest=manifest,
        rows=[row],
    )
    data = report.to_dict()
    assert data["run_id"] == "test"
    assert data["rows"][0]["arm_id"] == "x"


def test_baseline_arm_has_no_soft_features(tmp_path: Path) -> None:
    corpus = _valid_corpus(count=16)
    report = run_fixture_matrix(corpus, output_dir=tmp_path)
    baseline = next(row for row in report.rows if row.arm_id == "A_baseline")
    assert baseline.total_soft_features == 0
    assert baseline.total_hard_removals == 0
