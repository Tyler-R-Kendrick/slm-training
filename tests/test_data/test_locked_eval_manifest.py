from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.data.locked_eval_manifest import build_locked_manifest, write_locked_manifest
from slm_training.dsl.schema import load_jsonl


def test_locked_manifest_is_partitioned_and_immutable(tmp_path: Path) -> None:
    candidates = load_jsonl("src/slm_training/resources/test_seeds.jsonl")[:4]
    manifest = build_locked_manifest(
        candidates,
        source_records=[],
        min_locked_records=1,
        partition_size=1,
    )
    path = tmp_path / "locked.json"

    digest = write_locked_manifest(path, manifest)
    assert digest == manifest.sha256
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["manifest_sha256"] == digest
    assert payload["metadata"]["human_audit_is_optional_not_a_promotion_gate"] is True
    assert {row["partition"] for row in payload["rows"]} == {
        "dev", "locked_test", "agentv_calibration", "human_audit"
    }
    assert write_locked_manifest(path, manifest) == digest

    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable manifest"):
        write_locked_manifest(path, manifest)
