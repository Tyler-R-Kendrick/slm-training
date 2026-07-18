"""Fail-closed checkpoint-reference audit."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.harnesses.model_build.checkpoint_reference import (
    CheckpointReferenceV1,
    sha256_file,
)
from scripts.verify_checkpoint_references import build_report

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


def _frontier(**kw: object) -> dict:
    data = dict(_FULL_FRONTIER)
    data.update(kw)
    return CheckpointReferenceV1(**data).to_dict()


def _write(root: Path, files: dict[str, object]) -> None:
    design = root / "docs" / "design"
    design.mkdir(parents=True, exist_ok=True)
    for name, obj in files.items():
        (design / name).write_text(json.dumps(obj), encoding="utf-8")


def test_empty_repo_passes(tmp_path: Path) -> None:
    report = build_report(root=tmp_path)
    assert report["pass"] is True
    assert report["reference_count"] == 0


def test_verified_frontier_passes(tmp_path: Path) -> None:
    _write(tmp_path, {"a.json": _frontier()})
    report = build_report(root=tmp_path)
    assert report["pass"] is True
    assert report["reference_count"] == 1


def test_under_provenanced_frontier_fails(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {"b.json": _frontier(verification_timestamp=None, verifier_version=None)},
    )
    report = build_report(root=tmp_path)
    assert report["pass"] is False
    assert any("verification_timestamp" in err for err in report["errors"])


def test_fixture_local_only_is_allowed(tmp_path: Path) -> None:
    fixture = CheckpointReferenceV1(
        run_id="fx",
        claim_class="fixture",
        checkpoint_role="last",
        checkpoint_filename="last.pt",
    ).to_dict()
    _write(tmp_path, {"c.json": fixture})
    report = build_report(root=tmp_path)
    assert report["pass"] is True


def test_duplicate_run_role_different_sha_fails(tmp_path: Path) -> None:
    _write(tmp_path, {"d.json": _frontier(), "e.json": _frontier(sha256="b" * 64)})
    report = build_report(root=tmp_path)
    assert report["pass"] is False
    assert report["cross_reference_errors"]


def test_conflicting_training_commit_fails(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "f.json": _frontier(),
            "g.json": _frontier(sha256="a" * 64, training_source_commit="e" * 40),
        },
    )
    report = build_report(root=tmp_path)
    assert report["pass"] is False
    assert any("training_source_commit" in err for err in report["errors"])


def test_local_hash_mismatch_fails_even_for_fixture(tmp_path: Path) -> None:
    blob = tmp_path / "artifact.pt"
    blob.write_bytes(b"real-bytes")
    ref = CheckpointReferenceV1(
        run_id="fx",
        claim_class="fixture",
        checkpoint_role="last",
        checkpoint_filename="artifact.pt",
        sha256="0" * 64,  # deliberately wrong
        metadata=(("local_path", str(blob)),),
    ).to_dict()
    _write(tmp_path, {"h.json": ref})
    report = build_report(root=tmp_path)
    assert report["pass"] is False
    assert any("SHA-256 mismatch" in err for err in report["errors"])


def test_frontier_resolves_and_verifies_tracked_local(tmp_path: Path) -> None:
    """A durable reference whose artifact is a tracked file is byte-verified."""
    tracked = tmp_path / "src/slm_training/resources/checkpoints/demo"
    tracked.mkdir(parents=True)
    blob = tracked / "last.pt"
    blob.write_bytes(b"weights")
    ref = _frontier(sha256=sha256_file(blob))
    _write(tmp_path, {"i.json": ref})
    report = build_report(root=tmp_path)
    assert report["pass"] is True
    check = report["checks"][0]
    assert check["resolution"]["status"] == "verified_local"


def test_manifest_and_embedded_references_are_discovered(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "checkpoint_reference/v1",
        "run_id": "r1",
        "references": [_frontier()],
    }
    wrapper = {"experiment": "E999", "checkpoint_reference": _frontier(run_id="r2")}
    _write(tmp_path, {"manifest.json": manifest, "result.json": wrapper})
    report = build_report(root=tmp_path)
    assert report["reference_count"] == 2
    assert report["pass"] is True
