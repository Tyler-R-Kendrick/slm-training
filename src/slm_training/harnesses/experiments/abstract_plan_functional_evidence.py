"""Fail-closed functional-evidence contract for AbstractPlan interventions.

This owner evaluates a supplied shared-model runner; it never creates a second
trainer or treats raw decode, an oracle, or a fixture as a ship result.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
)
from slm_training.autoresearch.schemas import CampaignBudget
from slm_training.evals.power_protocol import cluster_bootstrap_ci
from slm_training.versioning import build_version_stamp

LOCKED_MANIFEST = Path(
    "src/slm_training/resources/data/eval/manifests/abstract_planning_locked_v1.jsonl"
)
ARMS = (
    "no_plan",
    "empty",
    "random_norm_matched",
    "shuffle_between_examples",
    "within_plan_permutation",
    "prefix_truncation",
    "length_matched_verbal",
    "gold_semantic_plan",
    "learned_abstract_plan",
)
PATHS = ("raw", "constrained", "repaired")
PRIMARY_METRICS = ("binding_aware_meaningful_v2", "binder_reference_f1")
REQUIRED_METRICS = PRIMARY_METRICS + ("canonical_ast_match",)
GATES = tuple(f"G{index}" for index in range(13))
_DESTRUCTIVE = ("empty", "random_norm_matched", "shuffle_between_examples")


class FunctionalVerdict(str, Enum):
    FUNCTIONAL = "functional"
    IGNORED_OR_COLLAPSED = "ignored_or_collapsed"
    UNAVAILABLE = "unavailable"


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class AbstractPlanProtocolV1:
    manifest_sha256: str
    record_ids: tuple[str, ...]
    seed: int = 313

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "abstract_plan_functional_evidence/v1",
            "locked_eval_manifest_sha256": self.manifest_sha256,
            "record_ids": list(self.record_ids),
            "arms": list(ARMS),
            "paths": list(PATHS),
            "seed": self.seed,
            "raw_diagnostic_only": True,
        }
        return {**payload, "protocol_sha256": _sha(payload)}


def build_campaign(protocol: AbstractPlanProtocolV1) -> ExperimentCampaignV1:
    """Lock SLM-313 before a runner observes evaluation outcomes."""
    stamp = build_version_stamp(
        "harness.experiments",
        "evals.power_protocol",
        "data.locked_eval_manifest",
    )
    protocol_sha = protocol.to_dict()["protocol_sha256"]
    return ExperimentCampaignV1(
        campaign_id="slm313-abstract-plan-functional-evidence",
        experiment_id="slm313-ap022",
        hypothesis="A learned AbstractPlan causally improves constrained semantic outcomes over matched destructive controls.",
        decision="Classify the plan as functional, ignored/collapsed, or unavailable.",
        endpoints=(
            CampaignEndpointV1(
                endpoint_id="meaning_v2",
                metric="binding_aware_meaningful_v2",
                role="primary",
                direction="increase",
                minimum_effect=0.0,
            ),
            CampaignEndpointV1(
                endpoint_id="binder_f1",
                metric="binder_reference_f1",
                role="secondary",
                direction="increase",
                minimum_effect=0.0,
            ),
        ),
        arms=tuple(
            CampaignArmV1(
                arm_id=arm,
                role="candidate" if arm == "learned_abstract_plan" else "control",
                config_sha256=_sha({"protocol": protocol_sha, "arm": arm}),
            )
            for arm in ARMS
        ),
        seeds=(protocol.seed,),
        budget=CampaignBudget(max_experiments=len(ARMS), max_wall_minutes=3.0),
        stopping_rules=(
            "Stop and report unavailable when no trained learned-plan checkpoint is supplied.",
            "Raw decode is diagnostic-only and cannot determine promotion.",
        ),
        controls=tuple(
            CampaignControlV1(
                control_id=arm,
                description=f"Matched AbstractPlan intervention: {arm}.",
                kind="negative" if arm in _DESTRUCTIVE else "quality",
            )
            for arm in ARMS
            if arm != "learned_abstract_plan"
        ),
        negative_controls=_DESTRUCTIVE,
        multiplicity_families=(
            MultiplicityFamilyV1(
                family_id="plan_function",
                hypothesis_ids=("meaning_v2", "binder_f1"),
                alpha=0.05,
            ),
        ),
        promotion_gates=(
            CampaignGateV1(
                gate_id="functional_plan",
                endpoint_id="meaning_v2",
                operator="gt",
                threshold=0.0,
            ),
        ),
        rollback_gates=(
            CampaignGateV1(
                gate_id="collapsed_plan",
                endpoint_id="meaning_v2",
                operator="le",
                threshold=0.0,
            ),
        ),
        artifact_requirements=(
            ArtifactRequirementV1(kind="version_stamp"),
            ArtifactRequirementV1(kind="paired_examples"),
            ArtifactRequirementV1(kind="agentevals"),
            ArtifactRequirementV1(kind="agentv"),
        ),
        claim_class="diagnostic",
        source_commit=str(stamp["code_commit"]),
        source_dirty=bool(stamp["code_dirty"]),
        author="slm-training",
        locked_eval_manifest_sha256=protocol.manifest_sha256,
        created_at="2026-07-26T00:00:00Z",
    )


@dataclass(frozen=True)
class PlanEvidenceRow:
    record_id: str
    group_id: str
    arm: str
    path: str
    plan_tokens: tuple[int, ...] | None
    total_tokens: int
    latency_ms: float
    metrics: Mapping[str, float]
    gates: Mapping[str, bool]
    result_ref: str
    plan_length: int
    binder_count: int
    reference_diameter: int
    factor: str
    promotion_eligible: bool = False
    pair_manifest_sha256: str | None = None

    def validate(self) -> None:
        if self.arm not in ARMS or self.path not in PATHS:
            raise ValueError("unknown abstract-plan arm or decode path")
        if self.total_tokens < 0 or self.latency_ms < 0:
            raise ValueError("token count and latency must be non-negative")
        if set(REQUIRED_METRICS) - set(self.metrics):
            raise ValueError("missing required functional metric")
        if any(not math.isfinite(float(self.metrics[key])) for key in REQUIRED_METRICS):
            raise ValueError("functional metrics must be finite")
        if set(GATES) != set(self.gates):
            raise ValueError("every row must retain the complete G0-G12 vector")
        if self.arm == "no_plan" and self.plan_tokens is not None:
            raise ValueError("no_plan must not carry plan tokens")
        if self.arm != "no_plan" and self.plan_tokens is None:
            raise ValueError("plan intervention arm must retain its plan tokens")
        if self.plan_length < 0 or self.binder_count < 0 or self.reference_diameter < 0:
            raise ValueError("complexity strata must be non-negative")
        if not self.factor:
            raise ValueError("factor stratum is required")
        if self.path == "raw" and self.promotion_eligible:
            raise ValueError("raw decode is diagnostic-only and cannot be promoted")
        if self.arm == "shuffle_between_examples" and (
            not self.pair_manifest_sha256 or len(self.pair_manifest_sha256) != 64
        ):
            raise ValueError("shuffle-between-examples requires a pair manifest SHA-256")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {**asdict(self), "plan_tokens": list(self.plan_tokens) if self.plan_tokens is not None else None}


def load_locked_protocol(path: Path = LOCKED_MANIFEST) -> AbstractPlanProtocolV1:
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = _sha({"schema": payload.get("schema"), "rows": payload.get("rows"), "metadata": payload.get("metadata")})
    if payload.get("schema") != "LockedPromotionManifestV1" or digest != payload.get("manifest_sha256"):
        raise ValueError("locked promotion manifest digest mismatch")
    ids = tuple(str(row["record"]["id"]) for row in payload["rows"] if row.get("partition") == "locked_test")
    if not ids:
        raise ValueError("functional evidence requires locked_test records")
    return AbstractPlanProtocolV1(digest, ids)


def validate_rows(protocol: AbstractPlanProtocolV1, rows: Iterable[PlanEvidenceRow]) -> list[PlanEvidenceRow]:
    collected = list(rows)
    expected = {(record_id, arm, path) for record_id in protocol.record_ids for arm in ARMS for path in PATHS}
    observed = {(row.record_id, row.arm, row.path) for row in collected}
    if observed != expected or len(collected) != len(expected):
        raise ValueError("functional evidence requires every arm/path on every locked record")
    for row in collected:
        row.validate()
    return collected


def _paired_ci(rows: list[PlanEvidenceRow], control: str, metric: str) -> dict[str, Any]:
    learned = {(row.record_id, row.path): row for row in rows if row.arm == "learned_abstract_plan" and row.path == "constrained"}
    baseline = {(row.record_id, row.path): row for row in rows if row.arm == control and row.path == "constrained"}
    if set(learned) != set(baseline):
        raise ValueError("paired functional controls must cover the exact learned record set")
    ids = sorted(learned)
    deltas = [float(learned[key].metrics[metric]) - float(baseline[key].metrics[metric]) for key in ids]
    return cluster_bootstrap_ci(deltas, [key[0] for key in ids], lambda values: float(values.mean()), seed=313)


def summarize_functional_evidence(
    protocol: AbstractPlanProtocolV1, rows: Iterable[PlanEvidenceRow], *, learned_available: bool
) -> dict[str, Any]:
    materialized = validate_rows(protocol, rows)
    if not learned_available:
        return {
            "verdict": FunctionalVerdict.UNAVAILABLE.value,
            "reason": "learned abstract-plan checkpoint/connector evidence is unavailable; no proxy claim is permitted",
            "protocol": protocol.to_dict(),
            "ship_eligible": False,
        }
    intervals = {
        control: {metric: _paired_ci(materialized, control, metric) for metric in PRIMARY_METRICS}
        for control in _DESTRUCTIVE
    }
    positive = all(float(intervals[control]["binding_aware_meaningful_v2"]["low"]) > 0.0 for control in _DESTRUCTIVE)
    destructive_drop = any(float(intervals[control]["binding_aware_meaningful_v2"]["low"]) > 0.0 for control in _DESTRUCTIVE)
    verdict = FunctionalVerdict.FUNCTIONAL if positive and destructive_drop else FunctionalVerdict.IGNORED_OR_COLLAPSED
    return {
        "verdict": verdict.value,
        "protocol": protocol.to_dict(),
        "paired_intervals": intervals,
        "ship_eligible": False,
        "raw_diagnostic_only": True,
        "meaningful_parse_primary": "binding_aware_meaningful_v2",
    }
