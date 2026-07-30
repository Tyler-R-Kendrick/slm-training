"""SLM-291: content-addressed evidence bundles and preregistered claim bars."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.casefiles import case_values

from slm_training.harness_core.evidence_bundle import (
    BAR_SCHEMA_VERSION,
    SCHEMA_VERSION,
    ArtifactRef,
    EvidenceBundleV1,
    LocalEvidenceStore,
    PreregisteredClaimBarV1,
    bundle_from_dict,
    claim_bar_from_dict,
    require_valid_claim_bar,
    validate_claim_bar,
    verify_evidence_bundle,
)


def _file_ref(
    root: Path, name: str, data: bytes, *, uri: str | None = None
) -> ArtifactRef:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return ArtifactRef(
        name=name,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        uri=uri if uri is not None else name,
    )


def _valid_bundle(tmp_path: Path, **overrides) -> EvidenceBundleV1:
    checkpoint = _file_ref(
        tmp_path,
        "ckpt/last.pt",
        b"weights",
        uri="hf://buckets/TKendrick/OpenUI/checkpoints/run1/last.pt",
    )
    suites = {
        "smoke": _file_ref(tmp_path, "suites/smoke.jsonl", b"smoke-rows"),
    }
    kwargs = {
        "run_id": "run1",
        "claim_class": "ship_candidate",
        "source_commit": "a" * 40,
        "source_dirty": False,
        "config_sha256": "c" * 64,
        "seeds": (0, 1, 2),
        "environment_digest": "e" * 64,
        "corpus_sha256": "d" * 64,
        "suite_hashes": suites,
        "checkpoint": checkpoint,
        "evaluator_version": "evals.meaningful_program/v9",
        "gate_version": "gates.ship/v3",
    }
    kwargs.update(overrides)
    return EvidenceBundleV1(**kwargs)


def test_valid_bundle_verifies(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    assert bundle.is_publishable
    assert verify_evidence_bundle(bundle, artifact_root=tmp_path, store=None) == []


def test_missing_checkpoint_fails(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    (tmp_path / "ckpt/last.pt").unlink()
    local = ArtifactRef(
        name="last.pt",
        sha256=bundle.checkpoint.sha256,
        size_bytes=bundle.checkpoint.size_bytes,
        uri="outputs/runs/run1/checkpoints/last.pt",
    )
    broken = EvidenceBundleV1(
        **{**bundle.__dict__, "checkpoint": local, "claim_class": "diagnostic"}
    )
    failures = verify_evidence_bundle(broken, artifact_root=tmp_path, store=None)
    assert any("missing" in reason for reason in failures)


def test_altered_checkpoint_bytes_fail(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    (tmp_path / "ckpt/last.pt").write_bytes(b"tampered")
    local = ArtifactRef(
        name="last.pt",
        sha256=bundle.checkpoint.sha256,
        size_bytes=bundle.checkpoint.size_bytes,
        uri="ckpt/last.pt",
    )
    broken = EvidenceBundleV1(**{**bundle.__dict__, "checkpoint": local})
    failures = verify_evidence_bundle(broken, artifact_root=tmp_path, store=None)
    assert any("altered" in reason for reason in failures)


def test_altered_suite_bytes_fail(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    (tmp_path / "suites/smoke.jsonl").write_bytes(b"tampered-rows")
    failures = verify_evidence_bundle(bundle, artifact_root=tmp_path, store=None)
    assert any("suite 'smoke'" in reason and "altered" in reason for reason in failures)


def test_dirty_source_tree_blocks_non_fixture(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path, source_dirty=True)
    reasons = bundle.blocking_reasons()
    assert any("clean source tree" in reason for reason in reasons)


def test_fixture_claim_may_reference_local_paths(tmp_path: Path) -> None:
    local = ArtifactRef(
        name="last.pt",
        sha256=hashlib.sha256(b"weights").hexdigest(),
        size_bytes=7,
        uri="/tmp/fixture-run/last.pt",
    )
    bundle = _valid_bundle(tmp_path, claim_class="fixture", checkpoint=local)
    assert bundle.blocking_reasons() == []


def test_production_claim_pointing_at_tmp_fails(tmp_path: Path) -> None:
    local = ArtifactRef(
        name="last.pt",
        sha256=hashlib.sha256(b"weights").hexdigest(),
        size_bytes=7,
        uri="/tmp/slm-training-overnight/outputs/runs/x/checkpoints/last.pt",
    )
    bundle = _valid_bundle(tmp_path, checkpoint=local)
    reasons = bundle.blocking_reasons()
    assert any("durable URI" in reason for reason in reasons)
    with pytest.raises(ValueError, match="not publishable"):
        bundle.require_publishable()


def test_unpinned_evaluator_blocks(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path, evaluator_version="UNKNOWN")
    assert any("evaluator" in reason for reason in bundle.blocking_reasons())


def test_bundle_roundtrip_and_content_address(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    restored = bundle_from_dict(json.loads(json.dumps(bundle.to_dict())))
    assert restored == bundle
    assert restored.sha == bundle.sha
    assert len(bundle.sha) == 64
    with pytest.raises(ValueError, match="not an"):
        bundle_from_dict({"schema": "other/v1"})


def test_local_store_roundtrip_and_corruption(tmp_path: Path) -> None:
    store = LocalEvidenceStore(tmp_path / "cas")
    digest = store.put(b"payload")
    assert store.exists(digest)
    assert store.get(digest) == b"payload"
    assert store.uri(digest).endswith(digest)
    assert store.put(b"payload") == digest  # idempotent
    # Corrupt the stored object: reads must fail closed.
    (tmp_path / "cas" / "objects" / digest[:2] / digest).write_bytes(b"evil")
    with pytest.raises(ValueError, match="digest mismatch"):
        store.get(digest)


def test_store_addressed_artifact_verifies(tmp_path: Path) -> None:
    store = LocalEvidenceStore(tmp_path / "cas")
    digest = store.put(b"suite-rows")
    ref = ArtifactRef(name="smoke", sha256=digest, size_bytes=10, uri=store.uri(digest))
    bundle = _valid_bundle(tmp_path, suite_hashes={"smoke": ref})
    assert verify_evidence_bundle(bundle, artifact_root=tmp_path, store=store) == []
    missing = ArtifactRef(
        name="smoke", sha256="f" * 64, size_bytes=1, uri=store.uri("f" * 64)
    )
    broken = _valid_bundle(tmp_path, suite_hashes={"smoke": missing})
    failures = verify_evidence_bundle(broken, artifact_root=tmp_path, store=store)
    assert any("missing from the evidence store" in reason for reason in failures)


def test_claim_bar_schema_and_validation() -> None:
    bar = PreregisteredClaimBarV1(
        bar_id="lar1-01",
        target_metrics={"meaningful_program_rate": 0.5},
        min_sample_size=20,
        seeds=(0, 1, 2),
        required_lower_bound=0.4,
        stop_rule="stop after 3 seeds or budget exhaustion",
        entry_gate="floor_escaped",
        falsifier="confidence lower bound below 0.4 on the primary suite",
    )
    assert validate_claim_bar(bar) == []
    require_valid_claim_bar(bar)
    restored = claim_bar_from_dict(json.loads(json.dumps(bar.to_dict())))
    assert restored == bar
    assert restored.to_dict()["schema"] == BAR_SCHEMA_VERSION
    assert len(bar.sha) == 64


@pytest.mark.parametrize(
    "overrides,fragment",
    case_values(__file__, "test_claim_bar_fail_closed"),
)
def test_claim_bar_fail_closed(overrides, fragment) -> None:
    base = {
        "bar_id": "b",
        "target_metrics": {"m": 0.5},
        "min_sample_size": 20,
        "seeds": (0,),
        "stop_rule": "s",
        "entry_gate": "g",
        "falsifier": "f",
    }
    bar = PreregisteredClaimBarV1(**{**base, **overrides})
    reasons = validate_claim_bar(bar)
    assert any(fragment in reason for reason in reasons)
    with pytest.raises(ValueError, match="invalid"):
        require_valid_claim_bar(bar)


def test_serialized_schema_marker(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    assert bundle.to_dict()["schema"] == SCHEMA_VERSION
