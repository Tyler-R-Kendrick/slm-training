"""Dataset helpers that consume harness artifacts only."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.data.store import DataStore
from slm_training.dsl.language_contract import assert_symbol_only_output
from slm_training.dsl.harness_dsl import (
    HARNESS_SCHEMA,
    is_harness_prompt,
    parse_harness_task,
)


def _load_symbol_only_records(path: Path) -> list[ExampleRecord]:
    """Load records only after every completion target clears contract v2."""
    records = load_jsonl(path)
    for record in records:
        try:
            harness_meta = record.meta.get("harness_dsl")
            if harness_meta is not None or is_harness_prompt(record.prompt):
                if not isinstance(harness_meta, dict):
                    raise ValueError(
                        "symbolic Harness prompt lacks harness_dsl metadata"
                    )
                task = parse_harness_task(record.prompt)
                expected = {
                    "schema": HARNESS_SCHEMA,
                    "grammar_fingerprint": task.grammar_fingerprint,
                    "operation": task.operation.value,
                    "pack_id": task.pack_id,
                    "payload_kind": task.payload_kind.value,
                    "grammar_category": task.grammar_category,
                }
                if harness_meta != expected:
                    raise ValueError("Harness prompt metadata mismatch")
            assert_symbol_only_output(
                record.openui,
                output_kind=record.target_kind,
            )
        except ValueError as exc:
            raise ValueError(
                f"{path}: record {record.id!r} violates the symbol-only output "
                f"contract: {exc}"
            ) from exc
    return records


def load_train_records(train_dir: Path) -> list[ExampleRecord]:
    train_dir = DataStore().resolve_path("train", train_dir)
    records_path = train_dir / "records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"missing train records: {records_path}")
    return _load_symbol_only_records(records_path)


def load_suite_records(test_dir: Path, suite: str) -> list[ExampleRecord]:
    test_dir = DataStore().resolve_path("eval", test_dir)
    manifest_path = test_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        suites = manifest.get("suites") or {}
        if suite in suites:
            declared = Path(suites[suite])
            if declared.is_file():
                return _load_symbol_only_records(declared)
    # Fallback to conventional path
    path = test_dir / "suites" / suite / "records.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing suite {suite!r} under {test_dir}")
    return _load_symbol_only_records(path)


def batched(items: list[ExampleRecord], batch_size: int) -> list[list[ExampleRecord]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
