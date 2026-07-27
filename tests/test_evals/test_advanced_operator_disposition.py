"""SLM-420 (DSH5-12) claim-separation regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.evals.advanced_operator_disposition import (
    AdvancedOperatorClaim,
    AdvancedOperatorClaimV1,
    AdvancedOperatorDimension,
    AdvancedOperatorDispositionV1,
    AdvancedOperatorVerdict,
    build_advanced_operator_disposition,
    component_version_stamps,
)


ROOT = Path(__file__).resolve().parents[2]


def _report(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _build() -> AdvancedOperatorDispositionV1:
    return build_advanced_operator_disposition(
        dsh3_33_report=_report("docs/design/dsh3-33-rebased-cap2-disposition-20260726-local/report.json"),
        dsh5_03_report=_report("docs/design/dsh5-03-bulk-operator-crossover-20260726-local/report.json"),
        dsh5_09_report=_report("docs/design/dsh5-09-adaptive-router-20260726-local/report.json"),
        version_stamp={"stamp_schema": "version_stamp/v1"},
        component_version_stamps=component_version_stamps(),
    )


def test_disposition_covers_all_eleven_claims_exactly_once() -> None:
    disposition = _build()
    claims = {item.claim for item in disposition.claims}
    assert claims == set(AdvancedOperatorClaim)
    assert len(disposition.claims) == 11


def test_supported_and_unavailable_verdicts_retained() -> None:
    disposition = _build()
    assert disposition.claim(AdvancedOperatorClaim.SELECTOR_CORRECTNESS).verdict is AdvancedOperatorVerdict.SUPPORTED
    assert disposition.claim(AdvancedOperatorClaim.BULK_ATOMICITY).verdict is AdvancedOperatorVerdict.SUPPORTED
    assert disposition.claim(AdvancedOperatorClaim.TRANSACTION_CONTRACTS_EXECUTION).verdict is AdvancedOperatorVerdict.SUPPORTED
    assert disposition.claim(AdvancedOperatorClaim.SEQUENCE_MERGE).verdict is AdvancedOperatorVerdict.SUPPORTED
    assert disposition.claim(AdvancedOperatorClaim.CROSSOVER_WORK).verdict is AdvancedOperatorVerdict.UNAVAILABLE
    assert disposition.claim(AdvancedOperatorClaim.ADAPTIVE_ROUTING).verdict is AdvancedOperatorVerdict.UNAVAILABLE
    assert disposition.claim(AdvancedOperatorClaim.PARAMETERIZED_TEMPLATES).verdict is AdvancedOperatorVerdict.UNAVAILABLE
    assert disposition.claim(AdvancedOperatorClaim.SYSTEMS_EFFICIENCY).verdict is AdvancedOperatorVerdict.UNAVAILABLE
    assert disposition.claim(AdvancedOperatorClaim.SET_VALUED_SELECTION).verdict is AdvancedOperatorVerdict.UNRUN_CONDITIONAL
    assert disposition.claim(AdvancedOperatorClaim.EVENT_MEMORY).verdict is AdvancedOperatorVerdict.UNRUN_CONDITIONAL


def test_control_plane_learning_axis_is_not_conflated_with_execution_axis() -> None:
    disposition = _build()
    claim = disposition.claim(AdvancedOperatorClaim.CONTROL_PLANE_EXECUTION_LEARNING)
    assert claim.verdict is AdvancedOperatorVerdict.SUPPORTED
    assert claim.dimension_verdicts[AdvancedOperatorDimension.RUNTIME_CORRECTNESS] is AdvancedOperatorVerdict.SUPPORTED
    assert claim.dimension_verdicts[AdvancedOperatorDimension.LEARNED_SEMANTIC_BENEFIT] is AdvancedOperatorVerdict.UNRUN_CONDITIONAL


def test_headline_verdict_must_be_one_of_its_own_dimension_verdicts() -> None:
    disposition = _build()
    claim = disposition.claim(AdvancedOperatorClaim.SELECTOR_CORRECTNESS)
    with pytest.raises(ValueError, match="must equal one of the claim's own"):
        AdvancedOperatorClaimV1(
            claim=claim.claim,
            verdict=AdvancedOperatorVerdict.NEGATIVE,
            reason=claim.reason,
            dimension_verdicts=claim.dimension_verdicts,
            dimension_reasons=claim.dimension_reasons,
            evidence_ids=claim.evidence_ids,
        )


def test_claim_must_address_every_dimension() -> None:
    disposition = _build()
    claim = disposition.claim(AdvancedOperatorClaim.SELECTOR_CORRECTNESS)
    incomplete = {
        AdvancedOperatorDimension.RUNTIME_CORRECTNESS: AdvancedOperatorVerdict.SUPPORTED,
    }
    with pytest.raises(ValueError, match="a claim must state a verdict for every dimension"):
        AdvancedOperatorClaimV1(
            claim=claim.claim,
            verdict=AdvancedOperatorVerdict.SUPPORTED,
            reason=claim.reason,
            dimension_verdicts=incomplete,
            dimension_reasons=claim.dimension_reasons,
            evidence_ids=claim.evidence_ids,
        )


def test_inherited_policy_inventory_is_empty_and_matches_slm_408() -> None:
    disposition = _build()
    inherited = disposition.inherited_policy
    assert inherited.dsh5_may_start is False
    assert inherited.allowed_heads == ()
    assert inherited.allowed_objectives == ()
    assert inherited.allowed_actions == ()
    assert "DSH3-33" in inherited.source_disposition


def test_rejected_inherited_disposition_cannot_carry_a_policy_inventory() -> None:
    from slm_training.evals.advanced_operator_disposition import InheritedPolicyInventoryV1

    with pytest.raises(ValueError, match="cannot carry an allowed learned-policy inventory"):
        InheritedPolicyInventoryV1(
            source_disposition="SLM-408 (DSH3-33)",
            source_artifact_path="docs/design/dsh3-33-rebased-cap2-disposition-20260726-local/report.json",
            dsh5_may_start=False,
            allowed_heads=("local_flat",),
            allowed_objectives=(),
            allowed_actions=(),
            reason="invalid",
        )


def test_retention_covers_cap0_cap1_cap2_and_cap2_is_retained_not_advanced() -> None:
    disposition = _build()
    retention = {item.capability: item for item in disposition.retention}
    assert set(retention) == {"CAP0", "CAP1", "CAP2"}
    assert retention["CAP2"].verdict is AdvancedOperatorVerdict.SUPPORTED
    assert "retained unchanged" in retention["CAP2"].reason


def test_advanced_path_is_never_enabled_by_default() -> None:
    disposition = _build()
    assert disposition.advanced_path_enabled_by_default is False
    with pytest.raises(ValueError, match="no advanced path may be enabled by default"):
        AdvancedOperatorDispositionV1(
            evidence=disposition.evidence,
            claims=disposition.claims,
            inherited_policy=disposition.inherited_policy,
            retention=disposition.retention,
            advanced_path_enabled_by_default=True,
            recommendation=disposition.recommendation,
            recommendation_reason=disposition.recommendation_reason,
            historical_disposition=disposition.historical_disposition,
            version_stamp=disposition.version_stamp,
        )


def test_recommendation_is_one_of_the_allowed_five_and_matches_evidence() -> None:
    disposition = _build()
    assert disposition.recommendation == "retain_as_compiler_utility"


def test_round_trip_to_dict_preserves_claim_count_and_dimension_breakdown() -> None:
    disposition = _build()
    payload = disposition.to_dict()
    assert len(payload["claims"]) == 11
    for claim in payload["claims"]:
        assert set(claim["dimension_verdicts"]) == {d.value for d in AdvancedOperatorDimension}
