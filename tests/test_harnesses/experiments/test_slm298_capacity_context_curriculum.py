"""Focused SLM-298 factorial contract tests."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.dsl.schema import ExampleRecord, OutputTarget
from slm_training.harnesses.experiments.slm298_capacity_context_curriculum import (
    CURRICULA,
    build_campaign,
    complexity_curriculum_records,
    load_protocol,
    write_train_view,
)
from slm_training.harnesses.model_build.data import load_train_records


def _train_dir(tmp_path: Path) -> Path:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    records = [
        ExampleRecord(
            id=f"train-{index}",
            prompt=f"record {index}",
            openui=f'root = TextContent(":label{index}")',
            placeholders=[f":label{index}"],
        )
        for index in range(3)
    ]
    (train_dir / "records.jsonl").write_text(
        "".join(json.dumps(record.to_dict()) + "\n" for record in records), encoding="utf-8"
    )
    (train_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    return train_dir


def test_protocol_has_matched_24_cell_three_seed_factorial(tmp_path: Path) -> None:
    protocol = load_protocol(train_dir=_train_dir(tmp_path))
    assert len(protocol.cells) == 24
    assert {cell.seed for cell in protocol.cells} == {0, 1, 2}
    assert protocol.train_record_count == 3
    campaign = build_campaign(protocol)
    assert campaign.locked_eval_manifest_sha256 == protocol.locked_eval_manifest_sha256
    assert len(campaign.arms) == 8


def test_complexity_curriculum_is_ast_reference_based_not_token_length(tmp_path: Path) -> None:
    records = load_train_records(_train_dir(tmp_path))
    records[0].target_kind = "statement"
    records[0].target_category = "root_binding"
    records[0].accepted_outputs = [
        OutputTarget('root = TextContent(":alternate")', kind="statement")
    ]
    staged = complexity_curriculum_records(records)
    assert len(staged) == len(records)
    assert {record.meta["curriculum"] for record in staged} == {"A", "B", "C"}
    assert [record.id for record in staged] == [record.id for record in complexity_curriculum_records(records)]
    preserved = next(record for record in staged if record.id == "train-0")
    assert preserved.target_kind == "statement"
    assert preserved.target_category == "root_binding"
    assert preserved.accepted_outputs == records[0].accepted_outputs


def test_train_views_bind_the_source_manifest(tmp_path: Path) -> None:
    train_dir = _train_dir(tmp_path)
    protocol = load_protocol(train_dir=train_dir)
    for curriculum in CURRICULA:
        view = write_train_view(
            protocol,
            train_dir=train_dir,
            output_dir=tmp_path,
            curriculum=curriculum,
        )
        assert len(load_train_records(view)) == protocol.train_record_count
        assert (view / "manifest.json").is_file()
