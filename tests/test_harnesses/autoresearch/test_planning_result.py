"""Contract tests for the AP-037 / SLM-338 abstract-planning result object."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from slm_training.autoresearch.experiment_campaign import (
    ExperimentCampaignV1,
    campaign_manifest_sha256,
)
from slm_training.harnesses.autoresearch.planning_result import (
    SCHEMA_ID,
    AbstractPlanningResultV1,
    LatencyBreakdownV1,
    MetricPathV1,
    PlanControlV1,
    publication_blockers,
)
from slm_training.harness_core.evidence_bundle import ArtifactRef

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "planning_result"
    / "ap037_fixture.json"
)
HEX_64 = "b" * 64


def _fixture_payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _result(**updates) -> AbstractPlanningResultV1:
    result = AbstractPlanningResultV1.from_dict(_fixture_payload())
    return replace(result, **updates)


def test_fixture_roundtrips_and_content_sha_is_deterministic() -> None:
    result = AbstractPlanningResultV1.from_dict(_fixture_payload())
    assert result.schema == SCHEMA_ID
    again = AbstractPlanningResultV1.from_dict(result.to_dict())
    assert again == result
    assert result.content_sha256() == again.content_sha256()
    assert len(result.content_sha256()) == 64


def test_fixture_manifest_hash_matches_locked_campaign() -> None:
    payload = _fixture_payload()
    assert payload["campaign_manifest_sha256"] == (
        "00601c6d1c08dbe7ca22a3402f2c9e93d9178d9634018e0c6db29baef3215f41"
    )
    # The digest function is the canonical campaign lock hash from SLM-337.
    manifest = ExperimentCampaignV1.model_validate(
        json.loads(FIXTURE.with_name("ap037_campaign.json").read_text())
    )
    assert campaign_manifest_sha256(manifest) == payload["campaign_manifest_sha256"]


def test_valid_diagnostic_result_has_no_blockers() -> None:
    assert publication_blockers(_result()) == []


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("campaign_manifest_sha256", "", "missing_campaign_manifest_sha256"),
        ("campaign_manifest_sha256", "not-a-sha", "missing_campaign_manifest_sha256"),
        ("source_commit", "UNKNOWN", "missing_provenance:source_commit"),
        ("source_commit", "", "missing_provenance:source_commit"),
        ("data_snapshot_sha256", "UNKNOWN", "missing_provenance:data_snapshot_sha256"),
    ],
)
def test_missing_provenance_and_manifest_block(
    field: str, value: str, reason: str
) -> None:
    blockers = publication_blockers(_result(**{field: value}))
    assert reason in blockers


def test_missing_total_latency_blocks() -> None:
    latency = replace(_result().latency, total_seconds=None)
    assert "missing_total_latency" in publication_blockers(_result(latency=latency))
    latency = replace(_result().latency, total_seconds=0.0)
    assert "missing_total_latency" in publication_blockers(_result(latency=latency))


def test_promotion_classes_require_controls_paths_and_checkpoint() -> None:
    promotion = _result(claim_class="promotion_candidate")
    assert publication_blockers(promotion) == []

    no_controls = replace(promotion, controls=())
    assert "missing_plan_controls" in publication_blockers(no_controls)

    raw_only = replace(promotion, metrics=(promotion.metric_path("raw"),))
    blockers = publication_blockers(raw_only)
    assert "missing_metric_paths:constrained,repaired" in blockers

    no_checkpoint = replace(promotion, checkpoint=None)
    assert "missing_provenance:checkpoint" in publication_blockers(no_checkpoint)


def test_diagnostic_class_does_not_require_controls_or_full_paths() -> None:
    diagnostic = _result(
        claim_class="diagnostic",
        controls=(),
        checkpoint=None,
        metrics=(MetricPathV1(path="raw", n=4, parse_rate=0.5),),
    )
    assert publication_blockers(diagnostic) == []


def test_ship_gate_class_is_treated_as_promotion() -> None:
    ship = _result(claim_class="ship_gate", controls=())
    assert "missing_plan_controls" in publication_blockers(ship)


def test_unknown_claim_class_blocks() -> None:
    blockers = publication_blockers(_result(claim_class="bogus"))
    assert "unknown_claim_class:bogus" in blockers


def test_negative_control_kinds_satisfy_controls() -> None:
    promotion = _result(
        claim_class="promotion_candidate",
        controls=(PlanControlV1(kind="shuffled", n=8),),
    )
    assert "missing_plan_controls" not in publication_blockers(promotion)
    oracle_only = _result(
        claim_class="promotion_candidate",
        controls=(PlanControlV1(kind="oracle", n=8),),
    )
    assert "missing_plan_controls" in publication_blockers(oracle_only)


def test_latency_default_is_blocked_not_silent() -> None:
    result = _result(latency=LatencyBreakdownV1())
    assert "missing_total_latency" in publication_blockers(result)


def test_checkpoint_artifact_ref_roundtrip() -> None:
    ref = ArtifactRef(name="ckpt.pt", sha256=HEX_64, size_bytes=1, uri="outputs/x.pt")
    result = _result(checkpoint=ref)
    assert AbstractPlanningResultV1.from_dict(result.to_dict()).checkpoint == ref
