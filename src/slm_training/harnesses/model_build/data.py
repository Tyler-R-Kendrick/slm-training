"""Dataset helpers that consume harness artifacts only."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.data.contract import (
    assert_canonical_template_markers,
    assert_no_template_semantic_labels,
)
from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.data.store import DataStore
from slm_training.dsl.language_contract import (
    OUTPUT_CONTRACT_VERSION,
    assert_symbol_only_output,
)
from slm_training.dsl.analysis.templatize import assert_role_safe_output


def _load_symbol_only_records(path: Path) -> list[ExampleRecord]:
    """Load records only after every input and target clears the active contract."""
    records = load_jsonl(path)
    for record in records:
        try:
            assert_no_template_semantic_labels(record.prompt, record.design_md)
            assert_canonical_template_markers(record)
            assert_symbol_only_output(
                record.openui,
                output_kind=record.target_kind,
            )
            assert_role_safe_output(
                record.openui,
                output_kind=record.target_kind,
            )
            for target in record.accepted_outputs:
                assert_symbol_only_output(target.text, output_kind=target.kind)
                assert_role_safe_output(target.text, output_kind=target.kind)
        except ValueError as exc:
            raise ValueError(
                f"{path}: record {record.id!r} violates the symbol-only output "
                f"contract v{OUTPUT_CONTRACT_VERSION}: {exc}"
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
