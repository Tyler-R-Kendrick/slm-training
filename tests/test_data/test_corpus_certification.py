from __future__ import annotations

import json

import pytest

from slm_training.data.verify.corpus import certify_snapshots, write_certified_snapshot
from slm_training.dsl.schema import ExampleRecord, load_jsonl


def _record(record_id: str, source: str = "fixture") -> ExampleRecord:
    return ExampleRecord(
        id=record_id,
        prompt="Return a stack.",
        openui='root = Stack([TextContent(":item")])',
        source=source,
        placeholders=[":item"],
    )


def test_certification_preserves_duplicate_lineage_and_core_admission(tmp_path):
    rows = certify_snapshots({"a": [_record("one")], "b": [_record("two")]})
    assert len(rows) == 2
    assert rows[1].duplicate
    assert len(rows[0].membership()["gates"]) == 13

    write_certified_snapshot(
        tmp_path / "snapshot",
        dataset_id="openui_verified_v1",
        rows=rows,
        source_fingerprints={"a": "a", "b": "b"},
    )
    assert len(load_jsonl(tmp_path / "snapshot" / "records.jsonl")) == 1
    assert len((tmp_path / "snapshot" / "all.jsonl").read_text().splitlines()) == 2
    assert json.loads((tmp_path / "snapshot" / "manifest.json").read_text())["immutable"] is True
    with pytest.raises(FileExistsError, match="immutable"):
        write_certified_snapshot(
            tmp_path / "snapshot",
            dataset_id="openui_verified_v1",
            rows=rows,
            source_fingerprints={"a": "a", "b": "b"},
        )
