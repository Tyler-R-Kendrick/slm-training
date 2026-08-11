"""EVID-03 / SLM-519: four-axis ledger on FormalPreflightV1 (torch-free)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from slm_training.autoresearch.formal import (
    attach_four_axis_ledger,
    formal_preflight_payload,
    formal_preflight_sha256,
    ledger_from_four_axis_analysis,
    migrate_formal_preflight_v1,
)
from slm_training.autoresearch.schemas import (
    BOUND_AST_ID_PLACEHOLDERS,
    FormalPreflightFourAxisLedgerV1,
)
from slm_training.formal.finite_search_bounds import (
    BOUND_AST_ID,
    build_finite_search_bound,
    export_four_axis_bound_evidence,
)
from slm_training.harnesses.reasoning.revmath.schemas import (
    AxisCertificateRefV1,
    RevmathFourAxisAnalysisV1,
)

ROOT = Path(__file__).resolve().parents[2]
SFF_PREFLIGHT = (
    ROOT
    / "src/slm_training/resources/experiments/semantic_factor_frontier"
    / "formal_preflights"
    / "sff_factor-membership-roundtrip.json"
)


def _axis(
    name: str,
    status: str = "not_claimed",
    *,
    bound_ast_id: str | None = None,
    notes: str = "",
) -> AxisCertificateRefV1:
    kwargs: dict = {"axis": name, "status": status, "notes": notes}
    if status in ("proved_axis", "refuted_axis"):
        kwargs["certificate_sha256"] = "ab" * 32
        kwargs["theorem_ref"] = f"Thm.{name}"
    if name == "resource_bounds" and status in ("proved_axis", "refuted_axis"):
        kwargs["bound_ast_id"] = bound_ast_id or "bound.placeholder.pending_evid04.v1"
    return AxisCertificateRefV1(**kwargs)


def _analysis(**status_overrides: str) -> RevmathFourAxisAnalysisV1:
    return RevmathFourAxisAnalysisV1(
        assumption_strength=_axis(
            "assumption_strength",
            status_overrides.get("assumption_strength", "not_claimed"),
        ),
        computability=_axis(
            "computability",
            status_overrides.get("computability", "unknown_axis"),
        ),
        resource_bounds=_axis(
            "resource_bounds",
            status_overrides.get("resource_bounds", "not_claimed"),
        ),
        implementation_refinement=_axis(
            "implementation_refinement",
            status_overrides.get("implementation_refinement", "empirical_remainder"),
            notes="runtime wall-clock remains empirical",
        ),
    )


def test_historical_sff_preflight_migrates_without_inventing_ledger() -> None:
    raw = json.loads(SFF_PREFLIGHT.read_text(encoding="utf-8"))
    assert "four_axis_ledger" not in raw
    preflight = migrate_formal_preflight_v1(raw)
    assert preflight.four_axis_ledger is None
    assert "four_axis_ledger" not in formal_preflight_payload(preflight)
    # Digest stable vs hashing the historical payload directly.
    from slm_training.lineage.records import canonical_json

    historical = hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest()
    assert formal_preflight_sha256(preflight) == historical


def test_wall_clock_and_neural_quality_cannot_be_theorem_claims() -> None:
    with pytest.raises(ValueError, match="wall_clock_latency"):
        FormalPreflightFourAxisLedgerV1(
            analysis=_analysis(resource_bounds="proved_axis"),
            theorem_claim_kinds=("wall_clock_latency",),
            empirical_remainder_claim_ids=("wall_clock_latency",),
        )
    with pytest.raises(ValueError, match="neural_quality"):
        FormalPreflightFourAxisLedgerV1(
            analysis=_analysis(resource_bounds="proved_axis"),
            theorem_claim_kinds=("neural_quality",),
        )


def test_migrate_rejects_legacy_overclaim_side_channel() -> None:
    raw = json.loads(SFF_PREFLIGHT.read_text(encoding="utf-8"))
    raw["theorem_claim_kinds"] = ["wall_clock_latency"]
    with pytest.raises(ValueError, match="overclaim rejected"):
        migrate_formal_preflight_v1(raw)


def test_rm_subsystem_requires_explicit_interpretation() -> None:
    with pytest.raises(ValueError, match="explicit_reversal or explicit_interpretation"):
        FormalPreflightFourAxisLedgerV1(
            analysis=_analysis(assumption_strength="proved_axis"),
            rm_subsystem_id="RCA0",
            rm_interpretation_status="uninterpreted",
        )
    with pytest.raises(ValueError, match="explicit_reversal or explicit_interpretation"):
        FormalPreflightFourAxisLedgerV1(
            analysis=_analysis(assumption_strength="proved_axis"),
            rm_subsystem_id="WKL0",
            rm_interpretation_status="not_applicable",
        )
    ok = FormalPreflightFourAxisLedgerV1(
        analysis=_analysis(assumption_strength="proved_axis"),
        rm_subsystem_id="RCA0",
        rm_interpretation_status="explicit_interpretation",
    )
    assert ok.rm_subsystem_id == "RCA0"


def test_resource_bounds_require_bound_ast_id_placeholder() -> None:
    with pytest.raises(ValueError, match="bound_ast_id"):
        AxisCertificateRefV1(
            axis="resource_bounds",
            status="proved_axis",
            certificate_sha256="cd" * 32,
            theorem_ref="Thm.bound",
        )
    assert "bound.finite_search.prefix_tree.v1" in BOUND_AST_ID_PLACEHOLDERS
    assert "bound.placeholder.pending_evid04.v1" in BOUND_AST_ID_PLACEHOLDERS


def test_kern03_bound_evidence_attaches_to_preflight_ledger() -> None:
    raw = json.loads(SFF_PREFLIGHT.read_text(encoding="utf-8"))
    preflight = migrate_formal_preflight_v1(raw)
    evidence = build_finite_search_bound((2, 3))
    analysis = export_four_axis_bound_evidence(evidence)
    assert analysis.resource_bounds.bound_ast_id == BOUND_AST_ID
    ledger = ledger_from_four_axis_analysis(
        analysis,
        computability_classification="finite_decidable",
        resource_cost_model_id="ExpansionVerificationQueryModel",
        empirical_remainder_claim_ids=("wall_clock_latency", "neural_quality"),
        theorem_claim_kinds=("abstract_work_bound",),
    )
    attached = attach_four_axis_ledger(preflight, ledger)
    assert attached.four_axis_ledger is not None
    assert (
        attached.four_axis_ledger.analysis.resource_bounds.bound_ast_id == BOUND_AST_ID
    )
    assert attached.four_axis_ledger.computability_classification == "finite_decidable"
    assert "wall_clock_latency" in attached.four_axis_ledger.empirical_remainder_claim_ids
    # Attachment widens digest; historical digest remains on analysis link.
    assert attached.four_axis_ledger.analysis.formal_preflight_sha256 == (
        formal_preflight_sha256(preflight)
    )
    assert formal_preflight_sha256(attached) != formal_preflight_sha256(preflight)


def test_computability_label_requires_claimed_axis() -> None:
    with pytest.raises(ValueError, match="computability_classification"):
        FormalPreflightFourAxisLedgerV1(
            analysis=_analysis(computability="not_claimed"),
            computability_classification="bounded_search",
        )
