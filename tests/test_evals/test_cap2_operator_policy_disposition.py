"""SLM-408 claim-separation regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.evals.cap2_operator_policy_disposition import (
    RebasedCap2Claim,
    RebasedCap2ClaimV1,
    RebasedCap2OperatorPolicyDispositionV1,
    RebasedCap2Verdict,
    build_rebased_cap2_operator_policy_disposition,
)


ROOT = Path(__file__).resolve().parents[2]


def _report(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _build() -> RebasedCap2OperatorPolicyDispositionV1:
    return build_rebased_cap2_operator_policy_disposition(
        cap2_fixture=_report("docs/design/dsh3-28-cap2-operator-v2-20260726/report.json"),
        typed_policy=_report("docs/design/dsh3-28-typed-operator-policy-20260726-cap2-v2-fiveheads/report.json"),
        coverage=_report("docs/design/dsh3-29-operator-ecoc-coverage-20260726-local/report.json"),
        negative_ablation=_report("docs/design/dsh3-30-collapse-negative-ablation-20260726-local/report.json"),
        failure_prediction=_report("docs/design/dsh3-31-operator-failure-prediction-20260726-local/report.json"),
        systems=_report("docs/design/dsh3-32-operator-serving-work-20260726-local/report.json"),
        permutation=_report("docs/design/dsh3-27-operator-permutation-consistency.json"),
        version_stamp={"stamp_schema": "version_stamp/v1"},
    )


def test_rebased_ledger_retains_negative_and_unavailable_evidence() -> None:
    disposition = _build()
    claims = {item.claim: item for item in disposition.claims}

    assert len(claims) == 10
    assert claims[RebasedCap2Claim.HELD_OUT_SEMANTICS].verdict is RebasedCap2Verdict.NEGATIVE
    assert claims[RebasedCap2Claim.SYSTEMS_EFFICIENCY].verdict is RebasedCap2Verdict.UNAVAILABLE
    assert disposition.dsh5_may_start is False
    assert not disposition.allowed_heads


def test_dsh5_cannot_open_without_supported_held_out_semantics() -> None:
    disposition = _build()
    claims = tuple(
        RebasedCap2ClaimV1(
            claim.claim,
            RebasedCap2Verdict.NEGATIVE if claim.claim is RebasedCap2Claim.HELD_OUT_SEMANTICS else claim.verdict,
            claim.reason,
            claim.evidence_ids,
        )
        for claim in disposition.claims
    )
    with pytest.raises(ValueError, match="DSH5 cannot start"):
        RebasedCap2OperatorPolicyDispositionV1(
            evidence=disposition.evidence,
            claims=claims,
            dsh5_may_start=True,
            allowed_heads=("local_flat",),
            allowed_objectives=("coverage-aware DEFER",),
            allowed_actions=("primitive",),
            dsh5_reason="invalid",
            historical_disposition=disposition.historical_disposition,
            version_stamp=disposition.version_stamp,
        )
