"""Harness prompts fail before shared model-data construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.dsl.harness_dsl import (
    HarnessOperation,
    HarnessPayloadKind,
    HarnessTaskV1,
    parse_harness_task,
    serialize_harness_task,
)
from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.model_build.data import load_train_records


def _record() -> ExampleRecord:
    task = HarnessTaskV1(
        operation=HarnessOperation.IDENTITY,
        pack_id="openui",
        payload_kind=HarnessPayloadKind.DOCUMENT,
        grammar_category="document",
        payload="root = Separator()",
    )
    prompt = serialize_harness_task(task)
    parsed = parse_harness_task(prompt)
    return ExampleRecord(
        id="harness-row",
        prompt=prompt,
        openui="root = Separator()",
        meta={
            "harness_dsl": {
                "schema": parsed.schema,
                "grammar_fingerprint": parsed.grammar_fingerprint,
                "operation": parsed.operation.value,
                "pack_id": parsed.pack_id,
                "payload_kind": parsed.payload_kind.value,
                "grammar_category": parsed.grammar_category,
            }
        },
    )


def test_valid_harness_prompt_reaches_shared_loader(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    write_jsonl(train_dir / "records.jsonl", [_record()])

    assert load_train_records(train_dir)[0].id == "harness-row"


def test_malformed_harness_prompt_fails_before_model_input(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    record = _record()
    record.prompt = record.prompt.replace("OP IDENTITY", "OP EXPLAIN")
    write_jsonl(train_dir / "records.jsonl", [record])

    with pytest.raises(ValueError, match="Harness"):
        load_train_records(train_dir)
