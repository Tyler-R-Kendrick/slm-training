"""Tests for the SLM-261 memorization probe harness."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.harnesses.experiments.slm261_memorization_probe import (
    MATRIX_VERSION,
    build_corruption_suite,
    render_markdown,
    run_memorization_probe_fixture,
    select_corpus_fixture,
    validate_manifest,
)


@pytest.fixture
def smoke_corpus(tmp_path: Path) -> Path:
    src = Path("src/slm_training/resources/data/eval/remediated/suites/smoke/records.jsonl")
    dst = tmp_path / "records.jsonl"
    dst.write_bytes(src.read_bytes())
    return dst


def test_select_corpus_fixture_returns_requested_count(smoke_corpus: Path) -> None:
    selected, negative = select_corpus_fixture(smoke_corpus, n_records=2, negative_n=1, seed=7)
    assert len(selected) == 2
    assert len(negative) == 1
    assert all(isinstance(r, ExampleRecord) for r in selected)
    assert not {r.id for r in selected} & {r.id for r in negative}


def test_select_corpus_fixture_is_deterministic(smoke_corpus: Path) -> None:
    a, _ = select_corpus_fixture(smoke_corpus, n_records=2, seed=5)
    b, _ = select_corpus_fixture(smoke_corpus, n_records=2, seed=5)
    assert [r.id for r in a] == [r.id for r in b]


def test_build_corruption_suite_is_stable(smoke_corpus: Path) -> None:
    records = load_jsonl(smoke_corpus)[:2]
    suite_a = build_corruption_suite(records, suite_id="test_suite", seed=9)
    suite_b = build_corruption_suite(records, suite_id="test_suite", seed=9)
    assert suite_a.source_corpus_sha256 == suite_b.source_corpus_sha256
    assert suite_a.suite_id == suite_b.suite_id
    assert suite_a.conditions == suite_b.conditions
    assert suite_a.cases == suite_b.cases
    assert len(suite_a.cases) == len(records) * len(suite_a.conditions)
    for case in suite_a.cases:
        assert case.row_sha256
        assert len(case.target_ids) == len(case.noisy_ids) == len(case.predict_mask)


def test_run_fixture_produces_valid_manifest(smoke_corpus: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "probe"
    manifest = run_memorization_probe_fixture(
        corpus_path=smoke_corpus,
        output_dir=output_dir,
        arms=("M1_current_recipe",),
        seeds=(0,),
        n_records=2,
        steps=1,
        lr=3e-4,
        fast=True,
        version_components=("harness.experiments",),
    )
    assert manifest.matrix_version == MATRIX_VERSION
    assert manifest.experiment_id == "slm261-memorization-probe"
    assert len(manifest.arms) == 1
    assert validate_manifest(manifest) == []
    assert (output_dir / "report.json").exists()
    assert (output_dir / "corruption_suite.json").exists()
    arm = manifest.arms[0]
    assert arm.arm_name == "M1_current_recipe"
    assert arm.trainable_parameter_count is not None
    assert arm.wall_seconds < 180.0


def test_render_markdown_contains_key_sections(smoke_corpus: Path, tmp_path: Path) -> None:
    manifest = run_memorization_probe_fixture(
        corpus_path=smoke_corpus,
        output_dir=tmp_path / "probe",
        arms=("M1_current_recipe",),
        seeds=(0,),
        n_records=2,
        steps=1,
        fast=True,
        version_components=(),
    )
    md = render_markdown(manifest)
    assert "# SLM-261" in md
    assert manifest.disposition in md
    assert "M1_current_recipe" in md


def test_cli_describe() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_memorization_probe", "--describe"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "SLM-261" in result.stdout
    assert "M0_principal_only" in result.stdout
