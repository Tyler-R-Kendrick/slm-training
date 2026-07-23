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
    StagedHarnessBaselineV1,
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
            REPO_ROOT
            / "docs/design/dsh0-01-staged-harness-baseline-20260723.json"
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
