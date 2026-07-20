"""Tests for slm_training.harnesses.experiments.slm144_plan_predictor_matrix."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from slm_training.harnesses.experiments.slm144_plan_predictor_matrix import (
    Slm144Arm,
    Slm144Manifest,
    Slm144Report,
    Slm144Row,
    build_slm144_manifest,
    render_markdown,
    run_fixture_matrix,
)
from slm_training.models.semantic_plan_predictor import (
    PlanTrainingExample,
    build_role_set_target,
)


def _examples(
    n: int,
    *,
    num_roles: int = 4,
    num_archetypes: int = 2,
    input_dim: int = 4,
    max_len: int = 3,
) -> list[PlanTrainingExample]:
    examples: list[PlanTrainingExample] = []
    for i in range(n):
        label = i % num_archetypes
        role_ids = [f"r{i % num_roles}"]
        mask = build_role_set_target(role_ids, {f"r{j}": j for j in range(num_roles)}, num_roles)
        serial = torch.full((max_len,), -1, dtype=torch.long)
        serial[0] = i % num_roles
        examples.append(
            PlanTrainingExample(
                example_id=f"ex{i}",
                input_features=torch.randn(input_dim),
                archetype_label=label,
                role_set_mask=mask,
                serialized_roles=serial,
            )
        )
    return examples


def test_manifest_has_all_arms() -> None:
    manifest = build_slm144_manifest()
    ids = {arm.arm_id for arm in manifest.arms}
    expected = {
        "baseline_none",
        "frequency_prior",
        "serialized_inventory",
        "set_matching",
        "gold_archetype",
        "gold_role_set",
        "gold_both",
    }
    assert ids == expected


def test_manifest_dataclass_round_trip() -> None:
    manifest = build_slm144_manifest()
    data = manifest.to_dict()
    restored = Slm144Manifest(
        matrix_set=data["matrix_set"],
        matrix_version=data["matrix_version"],
        hypothesis=data["hypothesis"],
        arms=[Slm144Arm(**a) for a in data["arms"]],
    )
    assert restored.matrix_version == manifest.matrix_version
    assert len(restored.arms) == len(manifest.arms)


def test_report_to_dict_round_trip() -> None:
    manifest = build_slm144_manifest()
    row = Slm144Row(arm_id="x", status="fixture", archetype_accuracy=0.5)
    report = Slm144Report(
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


def test_fixture_matrix_produces_report(tmp_path: Path) -> None:
    train = _examples(24)
    val = _examples(8)
    report = run_fixture_matrix(
        train,
        val,
        run_id="test_slm144",
        output_dir=tmp_path,
        epochs=10,
        batch_size=4,
    )
    assert report.status == "fixture"
    ids = {row.arm_id for row in report.rows}
    assert ids == {arm.arm_id for arm in report.manifest.arms}
    assert (tmp_path / "slm144_plan_predictor_report.json").exists()


def test_gold_both_arm_has_perfect_metrics() -> None:
    train = _examples(16)
    val = _examples(8)
    report = run_fixture_matrix(
        train,
        val,
        run_id="test_slm144_gold",
        epochs=4,
        batch_size=4,
    )
    gold_row = next(row for row in report.rows if row.arm_id == "gold_both")
    assert gold_row.archetype_accuracy == pytest.approx(1.0)
    assert gold_row.role_f1 == pytest.approx(1.0)


def test_render_markdown_includes_caveat_and_arms() -> None:
    train = _examples(16)
    val = _examples(8)
    report = run_fixture_matrix(
        train,
        val,
        run_id="test_slm144_md",
        epochs=4,
        batch_size=4,
    )
    md = render_markdown(report)
    assert "SLM-144" in md
    assert "fixture only" in md.lower()
    for arm in report.manifest.arms:
        assert arm.arm_id in md


def test_report_json_round_trip(tmp_path: Path) -> None:
    train = _examples(16)
    val = _examples(8)
    run_fixture_matrix(
        train,
        val,
        run_id="rt",
        output_dir=tmp_path,
        epochs=4,
        batch_size=4,
    )
    payload = json.loads((tmp_path / "slm144_plan_predictor_report.json").read_text())
    assert payload["run_id"] == "rt"
    assert payload["matrix_version"] == "spv1-01-v1"
    assert "version_stamp" in payload
