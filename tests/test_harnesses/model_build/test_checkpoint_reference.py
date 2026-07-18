"""Canonical checkpoint-reference schema: provenance, gating, serialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.harnesses.model_build.checkpoint_reference import (
    UNKNOWN,
    CheckpointReferenceV1,
    FileArtifact,
    build_references,
    reference_filename,
    reference_manifest,
    sha256_file,
    write_reference_sidecars,
)

_FULL_FRONTIER = dict(
    run_id="r1",
    claim_class="frontier",
    checkpoint_role="last",
    checkpoint_filename="last.pt",
    size_bytes=10,
    sha256="a" * 64,
    remote_uri="hf://buckets/TKendrick/OpenUI/checkpoints/r1/last.pt",
    bucket_id="TKendrick/OpenUI",
    training_source_commit="c" * 40,
    evaluation_source_commit="d" * 40,
    model_config_hash="m",
    tokenizer_hash="t",
    output_codec_hash="o",
    corpus_manifest_hash="cm",
    data_version="v1",
    verification_timestamp="2026-07-17T00:00:00Z",
    verifier_version="checkpoint_reference_verifier/v1",
)


def test_rejects_unknown_claim_class_and_bad_sha() -> None:
    with pytest.raises(ValueError, match="claim_class"):
        CheckpointReferenceV1(
            run_id="r", claim_class="bogus", checkpoint_role="last", checkpoint_filename="last.pt"
        )
    with pytest.raises(ValueError, match="sha256"):
        CheckpointReferenceV1(
            run_id="r",
            claim_class="fixture",
            checkpoint_role="last",
            checkpoint_filename="last.pt",
            sha256="tooshort",
        )


def test_fixture_and_diagnostic_are_publishable_local_only() -> None:
    for claim in ("fixture", "diagnostic"):
        ref = CheckpointReferenceV1(
            run_id="r", claim_class=claim, checkpoint_role="last", checkpoint_filename="last.pt"
        )
        assert ref.blocking_reasons() == ()
        assert ref.is_publishable is True
        ref.require_publishable()  # does not raise


def test_frontier_without_provenance_is_blocked() -> None:
    ref = CheckpointReferenceV1(
        run_id="r", claim_class="frontier", checkpoint_role="last", checkpoint_filename="last.pt"
    )
    reasons = dict(ref.blocking_reasons())
    assert "remote_uri" in reasons and "sha256" in reasons
    assert "verification_timestamp" in reasons
    assert ref.is_publishable is False
    with pytest.raises(ValueError, match="cannot be published as 'frontier'"):
        ref.require_publishable()


def test_full_frontier_is_publishable() -> None:
    ref = CheckpointReferenceV1(**_FULL_FRONTIER)
    assert ref.blocking_reasons() == ()
    assert ref.is_publishable is True
    ref.require_publishable()


def test_missing_verification_alone_blocks_frontier() -> None:
    data = dict(_FULL_FRONTIER)
    data["verification_timestamp"] = None
    data["verifier_version"] = None
    ref = CheckpointReferenceV1(**data)
    assert not ref.is_publishable
    assert "verification_timestamp" in dict(ref.blocking_reasons())


def test_serialization_is_deterministic_and_roundtrips(tmp_path: Path) -> None:
    ref = CheckpointReferenceV1(**_FULL_FRONTIER)
    # Stable content hash + canonical form.
    assert ref.sha == CheckpointReferenceV1(**_FULL_FRONTIER).sha
    assert ref.canonical_json() == CheckpointReferenceV1(**_FULL_FRONTIER).canonical_json()
    # dict round-trip.
    assert CheckpointReferenceV1.from_dict(ref.to_dict()) == ref
    # file round-trip.
    path = ref.write_json(tmp_path / "last.pt.ref.json")
    assert CheckpointReferenceV1.load_json(path) == ref
    # Byte-stable across two writes.
    again = CheckpointReferenceV1.load_json(path).write_json(tmp_path / "again.json")
    assert path.read_text() == again.read_text()


def test_from_dict_rejects_foreign_schema_version() -> None:
    data = CheckpointReferenceV1(**_FULL_FRONTIER).to_dict()
    data["schema_version"] = "checkpoint_reference/v2"
    with pytest.raises(ValueError, match="schema_version"):
        CheckpointReferenceV1.from_dict(data)


def test_file_artifact_and_for_local_file(tmp_path: Path) -> None:
    blob = tmp_path / "last.pt"
    blob.write_bytes(b"weights")
    art = FileArtifact.from_dict(
        {"name": "last.pt", "size_bytes": 7, "sha256": sha256_file(blob)}
    )
    assert art.sha256 == sha256_file(blob)
    ref = CheckpointReferenceV1.for_local_file(
        blob, run_id="r", claim_class="diagnostic", checkpoint_role="last"
    )
    assert ref.size_bytes == 7
    assert ref.sha256 == sha256_file(blob)
    # Nothing else was invented.
    assert ref.remote_uri == UNKNOWN


def test_build_references_attaches_companions_and_manifest(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "last.pt").write_bytes(b"w")
    (stage / "last.tokenizer.json").write_text("{}", encoding="utf-8")
    (stage / "best_ship_score.pt").write_bytes(b"b")
    (stage / "train_summary.json").write_text("{}", encoding="utf-8")

    refs = build_references(
        staged_dir=stage,
        checkpoint_names=("last.pt", "best_ship_score.pt"),
        run_id="demo",
        remote_uri="hf://buckets/TKendrick/OpenUI/checkpoints/demo",
        bucket_id="TKendrick/OpenUI",
        claim_class="diagnostic",
        sync_timestamp="2026-07-17T00:00:00Z",
    )
    by_role = {r.checkpoint_role: r for r in refs}
    assert set(by_role) == {"last", "best_ship_score"}
    last = by_role["last"]
    companions = {c.name for c in last.companion_files}
    # Stem-matched + shared companion, but not the other checkpoint or sidecars.
    assert "last.tokenizer.json" in companions
    assert "train_summary.json" in companions
    assert "best_ship_score.pt" not in companions
    assert last.remote_uri.endswith("/checkpoints/demo/last.pt")

    manifest = reference_manifest(
        refs,
        run_id="demo",
        remote_uri="hf://buckets/TKendrick/OpenUI/checkpoints/demo",
        bucket_id="TKendrick/OpenUI",
        claim_class="diagnostic",
        sync_timestamp="2026-07-17T00:00:00Z",
    )
    written = write_reference_sidecars(stage, refs, manifest=manifest)
    assert reference_filename("last.pt") in written
    assert "checkpoint_references.json" in written
    reloaded = json.loads((stage / "checkpoint_references.json").read_text())
    assert len(reloaded["references"]) == 2
