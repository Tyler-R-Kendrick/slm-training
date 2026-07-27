from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.data.locked_eval_manifest import build_locked_manifest, write_locked_manifest
from slm_training.data.locked_eval_manifest import complexity_stratum, record_complexity
from slm_training.data.locked_eval_manifest import measure_stratified_legal_entropy
from slm_training.data.locked_eval_manifest import (
    canonical_manifest_path,
    load_locked_manifest_payload,
    verify_locked_manifest_digest,
)
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


def test_canonical_manifest_path_is_the_committed_locked_manifest() -> None:
    path = canonical_manifest_path()

    assert path.is_file()
    assert path.as_posix().endswith(
        "resources/data/eval/manifests/abstract_planning_locked_v1.jsonl"
    )
    payload = load_locked_manifest_payload(path)
    assert verify_locked_manifest_digest(path, payload["manifest_sha256"])


def test_verify_locked_manifest_digest_rejects_tampered_or_wrong_digest(
    tmp_path: Path,
) -> None:
    candidates = load_jsonl("src/slm_training/resources/test_seeds.jsonl")[:4]
    manifest = build_locked_manifest(
        candidates, source_records=[], min_locked_records=1, partition_size=1
    )
    path = tmp_path / "locked.json"
    digest = write_locked_manifest(path, manifest)

    assert verify_locked_manifest_digest(path, digest) is True
    assert verify_locked_manifest_digest(path, "0" * 64) is False
    assert verify_locked_manifest_digest(tmp_path / "missing.json", digest) is False

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"] = list(payload["rows"])[:-1] if payload["rows"] else payload["rows"]
    payload["metadata"] = {**payload["metadata"], "tampered": True}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_locked_manifest_digest(path, digest) is False
    with pytest.raises(ValueError, match="does not match its declared manifest_sha256"):
        load_locked_manifest_payload(path)


def test_load_locked_manifest_payload_rejects_unsupported_schema(tmp_path: Path) -> None:
    path = tmp_path / "not_a_manifest.json"
    path.write_text(json.dumps({"schema": "SomethingElseV1"}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported schema"):
        load_locked_manifest_payload(path)
