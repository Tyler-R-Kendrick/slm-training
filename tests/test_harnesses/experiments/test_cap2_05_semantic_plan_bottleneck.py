"""Regression tests for the CAP2-05 program-scale SemanticPlanV1 bottleneck harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.casefiles import case_values

torch = pytest.importorskip("torch")

from slm_training.harnesses.experiments.cap2_bottleneck import (
    build_matrix as build_cap2_02_matrix,
)
from slm_training.harnesses.experiments.cap2_semantic_plan_bottleneck import (
    build_matrix,
    evaluate_program_factor_arm,
    load_corpus_manifest,
    load_fixture_plans,
    run_matrix,
)


def test_build_matrix_includes_all_codec_families() -> None:
    arms = build_matrix()
    codecs = {a.codec for a in arms}
    assert codecs == {"uniform_scalar", "fsq", "lfq", "vq", "continuous"}


def test_arms_filter_narrows_matrix() -> None:
    arms = build_matrix(arms_filter=("plan_fsq_2_3_3_4_5",))
    assert [a.arm_id for a in arms] == ["plan_fsq_2_3_3_4_5"]


def test_load_fixture_plans_is_deterministic() -> None:
    first = load_fixture_plans(count=12, seed=0)
    second = load_fixture_plans(count=12, seed=0)
    assert len(first) == len(second) > 0
    assert [p.to_dict() for p in first] == [p.to_dict() for p in second]


@pytest.mark.parametrize(
    "arm_id",
    case_values(
        __file__,
        "test_program_factor_arm_reconstructs_small_fixture_and_passes_no_bypass",
    ),
)
def test_program_factor_arm_reconstructs_small_fixture_and_passes_no_bypass(
    arm_id: str,
) -> None:
    plans = load_fixture_plans(count=12, seed=0)
    arms = {a.arm_id: a for a in build_matrix(arms_filter=(arm_id,))}
    arm = arms[arm_id]
    # The fixture train split is tiny (9 records) and every arm's capacity
    # comfortably exceeds it, so the default per-arm step counts (already
    # tuned for CAP2-02's M=41 oracle-state matrix) converge reliably here too.
    result = evaluate_program_factor_arm(arm, plans)
    assert result.exact_reconstruction_rate == 1.0
    assert result.no_bypass_ok is True
    assert result.leakage is False
    assert result.nominal_bits > 0.0
    assert result.serialized_bytes_per_example >= 0.0
    assert set(result.factor_distortion) == {
        "archetype",
        "role_set",
        "component_family",
        "cardinality",
        "symbol_role",
        "topology",
        "bindings",
    }


def test_run_matrix_fixture_mode_produces_versioned_report() -> None:
    report = run_matrix(
        mode="fixture",
        fixture_count=12,
        seeds=(0,),
        arms_filter=("plan_fsq_2_3_3_4_5",),
    )
    assert report.version == "cap2-05-v1"
    assert report.mode == "fixture"
    assert len(report.arms) == 1
    assert not any(a.leakage for a in report.arms)
    assert all(a.no_bypass_ok for a in report.arms)


def test_fixture_mode_rejects_corpus_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fixture mode"):
        run_matrix(mode="fixture", corpus_path=tmp_path / "manifest.jsonl")


@pytest.mark.parametrize("mode", ["small_corpus", "full_corpus"])
def test_small_and_full_corpus_modes_require_manifest(mode: str) -> None:
    with pytest.raises(ValueError, match="requires corpus_path"):
        run_matrix(mode=mode)


def test_small_corpus_mode_loads_manifest(tmp_path: Path) -> None:
    plans = load_fixture_plans(count=8, seed=0)
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(p.to_dict()) for p in plans),
        encoding="utf-8",
    )
    report = run_matrix(
        mode="small_corpus",
        corpus_path=manifest_path,
        arms_filter=("plan_fsq_2_3_3_4_5",),
    )
    assert report.mode == "small_corpus"
    assert report.num_records == len(plans)


def test_load_corpus_manifest_rejects_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no records"):
        load_corpus_manifest(empty)


def test_existing_cap2_bottleneck_matrix_is_unchanged() -> None:
    """CAP2-05 must not perturb the existing CAP2-01/02 oracle-state matrix."""
    arms = build_cap2_02_matrix(41)
    ids = {a.arm_id for a in arms}
    assert {
        "fsq_2_3_3_4_5",
        "lfq_d6",
        "vq_64_d8",
        "continuous_d6",
        "uniform_b2d6",
    } <= ids
