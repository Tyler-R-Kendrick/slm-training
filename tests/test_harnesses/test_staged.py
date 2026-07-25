import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.staged import (
    UNKNOWN,
    Capability,
    Difficulty,
    EvaluationSource,
    EvidenceIdentityV1,
    EvidenceStatus,
    FoundationArtifactV1,
    FoundationClaimStatus,
    FoundationClaimV1,
    FoundationEvidenceClass,
    FrozenFoundationIdentityV1,
    FOUNDATION_CLAIMS,
    StagedHarnessBaselineV1,
    StagedHarnessFoundationDispositionV1,
    SupervisionSource,
)

SHA = "a" * 64
REPO_ROOT = Path(__file__).resolve().parents[2]


def _baseline() -> StagedHarnessBaselineV1:
    return StagedHarnessBaselineV1(
        repo_commit="b" * 40,
        repo_dirty=False,
        quality_matrix_frontier="DSH3-17",
        output_contract_generation="output_contract/v2",
        checkpoint_generation="model.twotower/v227",
        run_class="fixture_demo",
        artifacts=(
            EvidenceIdentityV1(
                path="docs/design/quality-experiment-matrix.md",
                identity="quality-matrix-frontier",
                sha256=SHA,
                status=EvidenceStatus.VERIFIED,
            ),
        ),
    )


def test_axes_are_complete_orthogonal_and_do_not_reuse_ladder_levels() -> None:
    baseline = _baseline()

    assert baseline.capabilities == tuple(Capability)
    assert baseline.supervision_sources == tuple(SupervisionSource)
    assert baseline.evaluation_sources == tuple(EvaluationSource)
    assert baseline.difficulties == tuple(Difficulty)
    assert {item.value for item in Capability} == {
        "CAP0_GRAMMAR",
        "CAP1_SEMANTICS",
        "CAP2_TRANSFORM",
    }
    assert all(not item.value.startswith("L") for item in Difficulty)


def test_baseline_serialization_and_hash_are_deterministic() -> None:
    first = _baseline()
    second = _baseline()

    assert first.to_dict() == second.to_dict()
    assert StagedHarnessBaselineV1.from_dict(first.to_dict()) == first
    assert json.loads(first.to_json()) == first.to_dict()
    assert first.sha == second.sha
    first.require_reusable()


def test_missing_or_invalid_evidence_never_becomes_zero() -> None:
    unknown = EvidenceIdentityV1(
        path="missing.json",
        identity=UNKNOWN,
        status=EvidenceStatus.UNKNOWN,
        reason="not produced",
    )
    baseline = replace(
        _baseline(),
        repo_commit=UNKNOWN,
        repo_dirty=None,
        quality_matrix_frontier=UNKNOWN,
        artifacts=(unknown,),
    )

    reasons = baseline.blocking_reasons()
    assert reasons
    assert "0" not in baseline.to_dict().values()
    with pytest.raises(ValueError, match="not reusable"):
        baseline.require_reusable()


def test_verified_evidence_requires_a_real_digest() -> None:
    with pytest.raises(ValueError, match="requires a sha256"):
        EvidenceIdentityV1(
            path="artifact.json",
            identity="artifact",
            status=EvidenceStatus.VERIFIED,
        )


def test_unknown_schema_and_run_class_fail_closed() -> None:
    with pytest.raises(ValueError, match="schema mismatch"):
        replace(_baseline(), schema_version="staged_harness_baseline/v2")
    with pytest.raises(ValueError, match="run_class"):
        replace(_baseline(), run_class="production")


def test_committed_baseline_and_reuse_map_hashes_match_their_sources() -> None:
    payload = json.loads(
        (
            REPO_ROOT / "docs/design/dsh0-01-staged-harness-baseline-20260723.json"
        ).read_text(encoding="utf-8")
    )

    assert content_sha(payload["baseline"]) == payload["baseline_sha"]
    assert (
        StagedHarnessBaselineV1.from_dict(payload["baseline"]).sha
        == payload["baseline_sha"]
    )
    rows = [*payload["baseline"]["artifacts"], *payload["reuse_map"]]
    for row in rows:
        if row["sha256"] == UNKNOWN:
            continue
        actual = hashlib.sha256(
            (REPO_ROOT / row.get("path", row.get("owner"))).read_bytes()
        ).hexdigest()
        assert actual == row["sha256"]


def _foundation(tmp_path: Path) -> StagedHarnessFoundationDispositionV1:
    artifact_path = tmp_path / "evidence.json"
    artifact_path.write_text("{}\n", encoding="utf-8")
    artifact = FoundationArtifactV1(
        path=artifact_path.relative_to(tmp_path).as_posix(),
        sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
    )
    claims = tuple(
        FoundationClaimV1(
            name=name,
            status=FoundationClaimStatus.SUPPORTED,
            evidence_class=FoundationEvidenceClass.CONTRACT_FIXTURE,
            artifacts=(artifact,),
            commands=(f"pytest -q {name}",),
        )
        for name in FOUNDATION_CLAIMS
    )
    return StagedHarnessFoundationDispositionV1(
        source_commit="b" * 40,
        claims=claims,
        frozen_identities=(
            FrozenFoundationIdentityV1(
                name="cap0.plan",
                version="v1",
                identity_sha256=SHA,
                source=artifact,
            ),
        ),
        next_work_item="SLM-353",
        version_stamp={
            "stamp_schema": "version_stamp/v1",
            "components": {"harness.staged": "v2"},
        },
    )


def test_foundation_unknown_or_invalid_claims_block_cap0(tmp_path: Path) -> None:
    disposition = _foundation(tmp_path)
    blocked = replace(
        disposition.claims[0],
        status=FoundationClaimStatus.UNKNOWN,
        blockers=("evidence missing",),
    )
    disposition = replace(disposition, claims=(blocked, *disposition.claims[1:]))

    assert not disposition.is_supported
    with pytest.raises(ValueError, match="CAP0 foundation is blocked"):
        disposition.require_supported(tmp_path)


def test_foundation_requires_exact_claims_and_current_artifacts(
    tmp_path: Path,
) -> None:
    disposition = _foundation(tmp_path)
    disposition.require_supported(tmp_path)
    with pytest.raises(ValueError, match="every claim exactly once"):
        replace(disposition, claims=disposition.claims[:-1])

    (tmp_path / "evidence.json").write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        disposition.require_supported(tmp_path)


def test_foundation_identity_and_decision_are_immutable(tmp_path: Path) -> None:
    disposition = _foundation(tmp_path)
    row = disposition.to_dict()

    assert StagedHarnessFoundationDispositionV1.from_dict(row) == disposition
    row["next_work_item"] = "SLM-999"
    with pytest.raises(ValueError, match="sha256 does not match"):
        StagedHarnessFoundationDispositionV1.from_dict(row)
    row = disposition.to_dict()
    row["waived"] = True
    with pytest.raises(ValueError, match="keys must be exact"):
        StagedHarnessFoundationDispositionV1.from_dict(row)


def test_committed_foundation_disposition_is_supported_and_current() -> None:
    path = REPO_ROOT / "docs/design/dsh0-08-foundation-disposition-20260723.json"
    disposition = StagedHarnessFoundationDispositionV1.load(path)

    disposition.require_supported(REPO_ROOT)
    assert disposition.is_supported
    assert disposition.next_work_item == "SLM-353"
