"""Decision contracts for the bounded SLM-233 recursive-depth campaign."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class RecursiveCoreVerdict(str, Enum):
    RECURSIVE_CORE_POSITIVE = "recursive_core_positive"
    DEEP_SUPERVISION_ONLY = "deep_supervision_only"
    WEIGHT_SHARING_ONLY = "weight_sharing_only"
    EXPLICIT_Z_POSITIVE = "explicit_z_positive"
    Y_ONLY_PREFERRED = "y_only_preferred"
    NO_RECURSIVE_GAIN = "no_recursive_gain"
    ARCHITECTURE_NOT_IDENTIFIABLE = "architecture_not_identifiable"
    UNSTABLE = "unstable"
    INCONCLUSIVE = "inconclusive"


def stable_hash(value: Any) -> str:
    """Return a deterministic SHA-256 over JSON-compatible evidence."""
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class RecursiveFairnessManifestV1:
    """Frozen initialization, exposure, accounting, and evaluator contract."""

    common_tensor_hashes: Mapping[str, str]
    common_tensor_hashes_match: bool
    architecture_seed_namespaces: Mapping[str, int]
    stacked_to_shared_layer_mapping: Mapping[str, str]
    accounting_by_arm: Mapping[str, Mapping[str, int | float]]
    optimizer_contract: Mapping[str, Any]
    corpus_hash: str
    data_order_hash: str
    corruption_schedule_hash: str
    exposure_hash: str
    checkpoint_eval_schedule_hash: str
    decode_evaluator_gate_hashes: Mapping[str, str]
    hardware_runtime_budget: Mapping[str, Any]
    schema: str = "RecursiveFairnessManifestV1"

    def validate(self) -> None:
        if not self.common_tensor_hashes or not self.common_tensor_hashes_match:
            raise ValueError("common initialization tensors must match")
        if not self.architecture_seed_namespaces:
            raise ValueError("architecture seed namespaces are required")
        if not self.accounting_by_arm:
            raise ValueError("per-arm accounting is required")
        required_hashes = (
            self.corpus_hash,
            self.data_order_hash,
            self.corruption_schedule_hash,
            self.exposure_hash,
            self.checkpoint_eval_schedule_hash,
        )
        if any(len(value) != 64 for value in required_hashes):
            raise ValueError("fairness hashes must be SHA-256 hex digests")
        if not self.decode_evaluator_gate_hashes:
            raise ValueError("gate/evaluator hashes are required")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["manifest_hash"] = stable_hash(result)
        return result


@dataclass(frozen=True)
class RecursiveCoreGateV2:
    verdict: str
    gate_refs: Mapping[str, Mapping[str, str]]
    matched_matrix_ref: str
    fairness_manifest_hash: str
    primary_effect_sizes: Mapping[str, Any]
    equivalence_margins: Mapping[str, float]
    cost_frontier: Sequence[Mapping[str, Any]]
    allowed_downstream_work: tuple[str, ...]
    blocked_claims: tuple[str, ...]
    checkpoint_refs: tuple[str, ...]
    rationale: str
    schema: str = "RecursiveCoreGateV2"

    def validate(self) -> None:
        if self.verdict not in {item.value for item in RecursiveCoreVerdict}:
            raise ValueError(f"unsupported RecursiveCoreGateV2 verdict: {self.verdict}")
        if not self.gate_refs or not self.fairness_manifest_hash:
            raise ValueError("gate and fairness references are required")
        if not self.blocked_claims:
            raise ValueError("blocked claims must be explicit")
        if self.verdict == RecursiveCoreVerdict.ARCHITECTURE_NOT_IDENTIFIABLE:
            forbidden = {"rsc3", "rsc4", "checkpoint_promotion", "ship"}
            if forbidden.intersection(self.allowed_downstream_work):
                raise ValueError(
                    "unidentifiable evidence cannot authorize RSC3/RSC4/promotion/ship"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def classify_recursive_core_gate(
    *,
    floor_verdict: str,
    observability_verdict: str,
    dynamics_verdict: str,
    z_verdict: str,
    matrix_complete: bool,
    controls_matched: bool,
    all_finite: bool,
    semantic_outcomes_available: bool,
    fairness_manifest_hash: str,
    gate_refs: Mapping[str, Mapping[str, str]],
    matched_matrix_ref: str,
    primary_effect_sizes: Mapping[str, Any],
    equivalence_margins: Mapping[str, float],
    cost_frontier: Sequence[Mapping[str, Any]],
    checkpoint_refs: Sequence[str] = (),
    positive_classification: str | None = None,
) -> RecursiveCoreGateV2:
    """Apply gate precedence before interpreting descriptive proxy outcomes.

    A non-escaped semantic floor dominates proxy loss movement. This is the
    central SLM-233 repair: unidentifiable signal is neither a positive result
    nor evidence that recurrence itself failed.
    """
    common_blocked = (
        "semantic_architecture_efficacy",
        "explicit_z_mechanism",
        "rsc3",
        "rsc4",
        "checkpoint_promotion",
        "production_default_change",
        "ship",
    )
    kwargs = {
        "gate_refs": gate_refs,
        "matched_matrix_ref": matched_matrix_ref,
        "fairness_manifest_hash": fairness_manifest_hash,
        "primary_effect_sizes": primary_effect_sizes,
        "equivalence_margins": equivalence_margins,
        "cost_frontier": cost_frontier,
        "checkpoint_refs": tuple(checkpoint_refs),
    }
    if not matrix_complete or not controls_matched:
        return RecursiveCoreGateV2(
            verdict=RecursiveCoreVerdict.INCONCLUSIVE.value,
            allowed_downstream_work=("repair_campaign_evidence",),
            blocked_claims=common_blocked,
            rationale="matched campaign coverage or fairness controls are incomplete",
            **kwargs,
        )
    if not all_finite:
        return RecursiveCoreGateV2(
            verdict=RecursiveCoreVerdict.UNSTABLE.value,
            allowed_downstream_work=("architecture_stability_repair",),
            blocked_claims=common_blocked,
            rationale="one or more matched campaign cells were non-finite",
            **kwargs,
        )
    if floor_verdict != "floor_escaped" or not semantic_outcomes_available:
        return RecursiveCoreGateV2(
            verdict=RecursiveCoreVerdict.ARCHITECTURE_NOT_IDENTIFIABLE.value,
            allowed_downstream_work=(
                "bounded_proxy_controls",
                "architecture_repair_without_semantic_claim",
            ),
            blocked_claims=common_blocked,
            rationale=(
                "the semantic floor did not escape, so finite matched proxy "
                "movement cannot identify a semantic architecture effect"
            ),
            **kwargs,
        )
    if observability_verdict in {"unstable", "oscillatory"} or dynamics_verdict in {
        "dead_increment",
        "expansive_unstable",
        "overcontractive",
    }:
        return RecursiveCoreGateV2(
            verdict=RecursiveCoreVerdict.UNSTABLE.value,
            allowed_downstream_work=("architecture_stability_repair",),
            blocked_claims=common_blocked,
            rationale="repaired observability/dynamics gates block semantic depth claims",
            **kwargs,
        )
    supported = {item.value for item in RecursiveCoreVerdict} - {
        RecursiveCoreVerdict.ARCHITECTURE_NOT_IDENTIFIABLE.value,
        RecursiveCoreVerdict.UNSTABLE.value,
        RecursiveCoreVerdict.INCONCLUSIVE.value,
    }
    if positive_classification not in supported:
        return RecursiveCoreGateV2(
            verdict=RecursiveCoreVerdict.INCONCLUSIVE.value,
            allowed_downstream_work=("replicate_semantic_campaign",),
            blocked_claims=common_blocked,
            rationale="semantic evidence exists but no preregistered effect class cleared",
            **kwargs,
        )
    blocked = list(common_blocked)
    allowed = ["replicate_semantic_campaign"]
    if positive_classification == RecursiveCoreVerdict.EXPLICIT_Z_POSITIVE.value:
        if z_verdict not in {"causal_use", "explicit_z_positive"}:
            return RecursiveCoreGateV2(
                verdict=RecursiveCoreVerdict.INCONCLUSIVE.value,
                allowed_downstream_work=tuple(allowed),
                blocked_claims=tuple(blocked),
                rationale="explicit-z outcome lacks a causal z-use gate",
                **kwargs,
            )
    return RecursiveCoreGateV2(
        verdict=positive_classification,
        allowed_downstream_work=tuple(allowed),
        blocked_claims=tuple(blocked),
        rationale="a preregistered semantic effect class cleared all repaired gates",
        **kwargs,
    )
