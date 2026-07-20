"""Regression tests for the semantic-contrast corpus builder (SPV2-01)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.data.progspec.generate import GeneratorConfig, ProgramGenerator
from slm_training.data.semantic_contrast import (
    ContrastFamily,
    SemanticContrastBuilder,
    generate_transforms,
)
from slm_training.data.semantic_plan.extract import OpenUISemanticPlanExtractor
from slm_training.data.verify import VerificationContext, verify_record
from slm_training.dsl.pack import get_pack
from slm_training.dsl.schema import ExampleRecord


@pytest.fixture
def pack():
    return get_pack("openui")


@pytest.fixture
def sample_program():
    generator = ProgramGenerator(
        GeneratorConfig(
            max_depth=2,
            max_width=3,
            components=("Stack", "TextContent", "Button"),
            split="train",
        ),
        seed=7,
    )
    result = generator.generate(1)
    assert result.programs
    return result.programs[0]


@pytest.fixture
def sample_plan(sample_program, pack):
    return OpenUISemanticPlanExtractor().extract(sample_program, pack)


def _out(tmp_path: Path, dataset_id: str) -> Path:
    return tmp_path / "eval" / dataset_id


def test_generate_transforms_includes_positive_and_families(sample_plan):
    candidates = generate_transforms(sample_plan)
    assert any(c.family is ContrastFamily.POSITIVE for c in candidates)
    families = {c.family for c in candidates if c.family is not ContrastFamily.POSITIVE}
    assert ContrastFamily.CONTENT in families
    assert ContrastFamily.BINDING in families
    assert ContrastFamily.CONTRACT in families


def test_transform_candidates_are_rebuildable_plans(sample_plan):
    for candidate in generate_transforms(sample_plan):
        assert isinstance(candidate.plan, type(sample_plan))
        assert candidate.transform_id
        assert candidate.family.value
        assert candidate.severity.value


def test_builder_produces_required_artifacts(tmp_path):
    dataset_id = "semantic_contrast_test_v1"
    builder = SemanticContrastBuilder(
        output_root=tmp_path,
        dataset_id=dataset_id,
        seed=3,
        source_count=4,
        splits=("train", "held_out"),
        split_weights=(0.75, 0.25),
    )
    summary = builder.build()
    out = _out(tmp_path, dataset_id)
    assert (out / "pairs.jsonl").is_file()
    assert (out / "records.jsonl").is_file()
    assert (out / "rejected.jsonl").is_file()
    assert (out / "summary.json").is_file()
    assert (out / "manifest.json").is_file()
    assert summary["dataset_id"] == dataset_id
    assert summary["seed"] == 3
    assert summary["source_count"] == 4
    # Version stamp present and includes our component.
    stamp = summary["version_stamp"]
    assert stamp["stamp_schema"] == "version_stamp/v1"
    assert "data.semantic_contrast" in stamp["components"]


def test_positive_passes_negative_fails_and_surface_stays_valid(tmp_path):
    dataset_id = "semantic_contrast_test_v2"
    builder = SemanticContrastBuilder(
        output_root=tmp_path,
        dataset_id=dataset_id,
        seed=5,
        source_count=4,
        splits=("train",),
        split_weights=(1.0,),
    )
    builder.build()
    out = _out(tmp_path, dataset_id)
    pairs = [json.loads(line) for line in (out / "pairs.jsonl").read_text().splitlines()]
    assert pairs
    positives = [p["positive"] for p in pairs]
    negatives = [p["negative"] for p in pairs if p["family"] != "positive"]
    assert all(p["meaningful_report"]["verdict"] for p in positives)
    assert all(not n["meaningful_report"]["verdict"] for n in negatives)
    # Every record surface must pass the verifier.
    for rec in positives + negatives:
        record = ExampleRecord.from_dict(rec["record"])
        report = verify_record(record, VerificationContext(source_kind="program"))
        assert report.ok, f"{rec['record']['id']} failed verifier"


def test_split_isolation(tmp_path):
    dataset_id = "semantic_contrast_test_v3"
    builder = SemanticContrastBuilder(
        output_root=tmp_path,
        dataset_id=dataset_id,
        seed=11,
        source_count=8,
        splits=("train", "held_out"),
        split_weights=(0.5, 0.5),
    )
    builder.build()
    out = _out(tmp_path, dataset_id)
    records = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
    train_ids = {
        r["source_program_id"]
        for r in records
        if r["meta"]["split"] == "train"
    }
    test_ids = {
        r["source_program_id"]
        for r in records
        if r["meta"]["split"] == "held_out"
    }
    assert train_ids
    assert test_ids
    assert not train_ids.intersection(test_ids)


def test_scoreboard_reports_all_families(tmp_path):
    dataset_id = "semantic_contrast_test_v4"
    builder = SemanticContrastBuilder(
        output_root=tmp_path,
        dataset_id=dataset_id,
        seed=13,
        source_count=4,
        splits=("train",),
        split_weights=(1.0,),
    )
    summary = builder.build()
    families = {m["family"] for m in summary["scoreboard"]}
    assert "positive" in families
    assert any(f in families for f in ("content", "binding", "contract"))
    for metric in summary["scoreboard"]:
        assert "n_total" in metric
        assert "verifier_pass_rate" in metric
        assert "false_negative_rate" in metric
