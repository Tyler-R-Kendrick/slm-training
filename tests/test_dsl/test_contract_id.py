"""OpenUI language-contract identity and record stamping."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.dsl.contract_id import compute_contract_id, contract_manifest
from slm_training.dsl.schema import ExampleRecord, load_jsonl


def test_contract_id_is_deterministic_and_tool_schema_sensitive() -> None:
    base = compute_contract_id()
    assert base == compute_contract_id()
    assert base.startswith("openui-v0.5-")
    changed = compute_contract_id(
        tool_schema=[{"name": "get_items", "inputSchema": {"type": "object"}}]
    )
    assert changed != base
    manifest = contract_manifest()
    assert manifest["lang_spec_version"] == "0.5"
    assert manifest["parser"]["package_version"] == "0.2.9"


def test_example_records_are_stamped_and_old_fixtures_migrate() -> None:
    record = ExampleRecord(id="x", prompt="p", openui="root = Stack([])")
    assert record.contract_id == compute_contract_id()
    assert record.to_dict()["contract_id"] == record.contract_id

    old = json.loads(
        Path("fixtures/train_seeds.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert "contract_id" not in old
    migrated = ExampleRecord.from_dict(old)
    assert migrated.contract_id == compute_contract_id(
        tool_schema=migrated.meta.get("tool_schema") or []
    )
    assert all(record.contract_id for record in load_jsonl("fixtures/train_seeds.jsonl"))
