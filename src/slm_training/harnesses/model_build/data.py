"""Dataset helpers that consume harness artifacts only."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.dsl.schema import ExampleRecord, load_jsonl


def load_train_records(train_dir: Path) -> list[ExampleRecord]:
    records_path = train_dir / "records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"missing train records: {records_path}")
    return load_jsonl(records_path)


def load_suite_records(test_dir: Path, suite: str) -> list[ExampleRecord]:
    manifest_path = test_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        suites = manifest.get("suites") or {}
        if suite in suites:
            return load_jsonl(suites[suite])
    # Fallback to conventional path
    path = test_dir / "suites" / suite / "records.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing suite {suite!r} under {test_dir}")
    return load_jsonl(path)


def batched(items: list[ExampleRecord], batch_size: int) -> list[list[ExampleRecord]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
