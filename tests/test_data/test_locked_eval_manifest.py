from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.data.locked_eval_manifest import build_locked_manifest, write_locked_manifest
from slm_training.data.locked_eval_manifest import complexity_stratum, record_complexity
from slm_training.data.locked_eval_manifest import measure_stratified_legal_entropy
from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.models.tokenizer import tokenize_text


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


def test_stratified_legal_entropy_uses_exact_compiler_sets(tmp_path: Path) -> None:
    candidates = load_jsonl("src/slm_training/resources/test_seeds.jsonl")[:4]
    manifest = build_locked_manifest(
        candidates, source_records=[], min_locked_records=1, partition_size=1
    )
    report = measure_stratified_legal_entropy(manifest.payload())

    assert report["authority"] == "gold_compiler_decisions"
    assert report["records"]
    assert all(row["max_legal_action_count"] >= 2 for row in report["records"])


def test_record_complexity_uses_ast_and_references_not_lexical_length() -> None:
    short = ExampleRecord(
        id="short",
        prompt="short",
        openui='root = TextContent(":x")',
        placeholders=[":x"],
    )
    long = ExampleRecord(
        id="long",
        prompt="long",
        openui=(
            'root = TextContent(":this.is.a.much.longer.placeholder.reference.'
            'with.many.lexical.fragments")'
        ),
        placeholders=[":this.is.a.much.longer.placeholder.reference.with.many.lexical.fragments"],
    )

    assert len(tokenize_text(long.openui)) > len(tokenize_text(short.openui))
    assert record_complexity(long) == record_complexity(short)
    assert complexity_stratum(long) == complexity_stratum(short)
