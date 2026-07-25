"""Typed terminal disposition for CAP2 operator capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class Cap2Capability(str, Enum):
    SYMBOLIC_TRANSFORM = "symbolic_transform"
    NL_TRANSFORM = "nl_transform"
    DISCRETE_TOKEN_ACTION = "discrete_token_action"
    HIERARCHICAL_HEAD = "hierarchical_head"
    TOPOLOGY_APPLICATION = "topology_application"
    BOUNDED_MERGE = "bounded_merge"
    EFFICIENCY = "efficiency"


class Cap2CapabilityVerdict(str, Enum):
    CONTRACT_ONLY = "contract_only"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    UNRUN_CONDITIONAL = "unrun_conditional"
    SUPPORTED = "supported"


@dataclass(frozen=True)
class CapabilityEvidenceV1:
    evidence_id: str
    evidence_class: str
    code_identity: str
    data_identity: Mapping[str, Any]
    checkpoint_identity: str | None
    suite_identity: Mapping[str, Any]
    config_identity: Mapping[str, Any]
    hardware_identity: Mapping[str, Any]
    result_identity: Mapping[str, Any]
    schema: str = "cap2_capability_evidence/v1"

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.code_identity:
            raise ValueError("CAP2 evidence requires stable identity")
        for name in (
            "data_identity",
            "suite_identity",
            "config_identity",
            "hardware_identity",
            "result_identity",
        ):
            if not getattr(self, name):
                raise ValueError(f"CAP2 evidence requires {name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_id": self.evidence_id,
            "evidence_class": self.evidence_class,
            "code_identity": self.code_identity,
            "data_identity": dict(self.data_identity),
            "checkpoint_identity": self.checkpoint_identity,
            "suite_identity": dict(self.suite_identity),
            "config_identity": dict(self.config_identity),
            "hardware_identity": dict(self.hardware_identity),
            "result_identity": dict(self.result_identity),
        }


@dataclass(frozen=True)
class CapabilityDispositionV1:
    capability: Cap2Capability
    verdict: Cap2CapabilityVerdict
    reason: str
    evidence_ids: tuple[str, ...] = ()
    implemented_benefit: bool = False
    schema: str = "cap2_capability_verdict/v1"

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("CAP2 capability verdict requires a reason")
        if self.verdict is Cap2CapabilityVerdict.SUPPORTED:
            if not self.implemented_benefit or not self.evidence_ids:
                raise ValueError("supported capability requires implemented evidence")
        elif self.implemented_benefit:
            raise ValueError("unsupported capability cannot claim an implemented benefit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "capability": self.capability.value,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "evidence_ids": list(self.evidence_ids),
            "implemented_benefit": self.implemented_benefit,
        }


@dataclass(frozen=True)
class Cap2CapabilityDispositionV1:
    evidence: tuple[CapabilityEvidenceV1, ...]
    capabilities: tuple[CapabilityDispositionV1, ...]
    cert_cap2_issued: bool
    cert_cap2_reason: str
    dsh4_action_distillation_open: bool
    version_stamp: Mapping[str, Any]
    schema: str = "cap2_capability_disposition/v1"

    def __post_init__(self) -> None:
        expected = set(Cap2Capability)
        actual = {item.capability for item in self.capabilities}
        if actual != expected or len(actual) != len(self.capabilities):
            raise ValueError("CAP2 disposition must cover each capability exactly once")
        evidence_ids = {item.evidence_id for item in self.evidence}
        if len(evidence_ids) != len(self.evidence):
            raise ValueError("CAP2 evidence IDs must be unique")
        if any(
            evidence_id not in evidence_ids
            for item in self.capabilities
            for evidence_id in item.evidence_ids
        ):
            raise ValueError("CAP2 verdict references unavailable evidence")
        supported = [
            item
            for item in self.capabilities
            if item.verdict is Cap2CapabilityVerdict.SUPPORTED
        ]
        if self.cert_cap2_issued and not supported:
            raise ValueError("CERT_CAP2 requires a supported learned capability")
        if self.dsh4_action_distillation_open != self.cert_cap2_issued:
            raise ValueError("DSH4 action distillation must follow CERT_CAP2")
        if not self.cert_cap2_reason:
            raise ValueError("CAP2 certificate disposition requires a reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence": [item.to_dict() for item in self.evidence],
            "capabilities": [item.to_dict() for item in self.capabilities],
            "cert_cap2": {
                "issued": self.cert_cap2_issued,
                "reason": self.cert_cap2_reason,
            },
            "dsh4_action_distillation": {
                "open": self.dsh4_action_distillation_open,
                "reason": self.cert_cap2_reason,
            },
            "version_stamp": dict(self.version_stamp),
        }


def build_cap2_disposition(
    *,
    cap2_report: Mapping[str, Any],
    token_report: Mapping[str, Any],
    version_stamp: Mapping[str, Any],
) -> Cap2CapabilityDispositionV1:
    if cap2_report.get("schema") != "cap2_operator_fixture_report/v1":
        raise ValueError("unsupported frozen CAP2 report")
    if token_report.get("schema") != "reserved_operator_baseline_report/v1":
        raise ValueError("unsupported token-baseline report")
    cap2_suite = cap2_report["suite"]
    token_result = token_report["result"]
    if cap2_report["policy_scores"]["oracle"]["gate_pass"] is not True:
        raise ValueError("frozen CAP2 oracle contract did not pass")
    if token_result["verdict"] != "reject" or token_result["accepted"] is not False:
        raise ValueError("token evidence does not support CERT_CAP2 rejection")
    if token_result["acceptance"]["zero_false_legal_admissions"] is not True:
        raise ValueError("token evidence has a legal-admission regression")

    frozen_evidence = CapabilityEvidenceV1(
        evidence_id="SLM-381.cap2_operator_v1",
        evidence_class="fixture_contract",
        code_identity=cap2_report["version_stamp"]["code_commit"],
        data_identity={
            "source_records_fingerprint": cap2_suite["source_records_fingerprint"],
            "operator_corpus_fingerprint": cap2_suite[
                "operator_corpus_fingerprint"
            ],
        },
        checkpoint_identity=None,
        suite_identity={
            "suite_version": cap2_suite["suite_version"],
            "suite_hash": cap2_suite["suite_hash"],
            "suite_n": len(cap2_suite["cases"]),
        },
        config_identity={
            "thresholds": cap2_suite["thresholds"],
            "matrix_set": cap2_report["run"]["matrix_set"],
        },
        hardware_identity={
            "device": cap2_report["run"]["device"],
            "exact_hardware": None,
            "efficiency_claim": False,
        },
        result_identity={
            "oracle_gate_pass": True,
            "oracle_exact": cap2_report["policy_scores"]["oracle"][
                "case_successes"
            ],
            "agentv": cap2_report["agentv"]["summary"],
            "claim": "evaluation_contract_only",
        },
    )
    token_evidence = CapabilityEvidenceV1(
        evidence_id="SLM-382.E803",
        evidence_class="bounded_matched_negative",
        code_identity=token_report["version_stamp"]["code_commit"],
        data_identity={
            "train": token_report["corpora"]["train"]["content_fingerprint"],
            "held_out": token_report["corpora"]["held_out"][
                "content_fingerprint"
            ],
        },
        checkpoint_identity=None,
        suite_identity={
            "suite": "CAP2 held-out operator decisions",
            "held_out_n": token_result["held_out_decision_n"],
        },
        config_identity={
            "seeds": token_result["seeds"],
            "steps_per_arm": token_result["steps_per_arm"],
            "learning_rate": token_result["learning_rate"],
            "parameter_count": token_result["arms"]["RESULT_AST_ONLY"][0][
                "parameter_count"
            ],
        },
        hardware_identity={
            "device": token_report["run"]["device"],
            "backend": token_report["run"]["backend"],
            "exact_hardware": None,
            "efficiency_claim": False,
        },
        result_identity={
            "verdict": token_result["verdict"],
            "acceptance": token_result["acceptance"],
            "mean_result_ast_accuracy": token_result[
                "mean_result_ast_accuracy"
            ],
            "agentv": token_report["agentv"]["summary"],
        },
    )
    capabilities = (
        CapabilityDispositionV1(
            Cap2Capability.SYMBOLIC_TRANSFORM,
            Cap2CapabilityVerdict.CONTRACT_ONLY,
            "Exact symbolic data/eval replay is frozen, but no learned model capability passed.",
            (frozen_evidence.evidence_id,),
        ),
        CapabilityDispositionV1(
            Cap2Capability.NL_TRANSFORM,
            Cap2CapabilityVerdict.UNAVAILABLE,
            "CERT_CAP1 is unavailable, so no NL CAP2 row or retention claim exists.",
        ),
        CapabilityDispositionV1(
            Cap2Capability.DISCRETE_TOKEN_ACTION,
            Cap2CapabilityVerdict.REJECTED,
            "E803 did not improve held-out CAP2 behavior or correct-over-wrong changes.",
            (token_evidence.evidence_id,),
        ),
        CapabilityDispositionV1(
            Cap2Capability.HIERARCHICAL_HEAD,
            Cap2CapabilityVerdict.UNRUN_CONDITIONAL,
            "SLM-383 stayed closed because its token-baseline prerequisite failed.",
        ),
        CapabilityDispositionV1(
            Cap2Capability.TOPOLOGY_APPLICATION,
            Cap2CapabilityVerdict.UNRUN_CONDITIONAL,
            "SLM-384 stayed closed because no accepted hierarchical-head arm exists.",
        ),
        CapabilityDispositionV1(
            Cap2Capability.BOUNDED_MERGE,
            Cap2CapabilityVerdict.CONTRACT_ONLY,
            "Merge replay/conflict behavior is fixture-frozen, not a learned benefit.",
            (frozen_evidence.evidence_id,),
        ),
        CapabilityDispositionV1(
            Cap2Capability.EFFICIENCY,
            Cap2CapabilityVerdict.UNAVAILABLE,
            "No exact hardware identity or matched-quality systems experiment exists.",
            (token_evidence.evidence_id,),
        ),
    )
    return Cap2CapabilityDispositionV1(
        evidence=(frozen_evidence, token_evidence),
        capabilities=capabilities,
        cert_cap2_issued=False,
        cert_cap2_reason=(
            "No action representation causally improved held-out CAP2 behavior; "
            "CAP1 retention is unavailable."
        ),
        dsh4_action_distillation_open=False,
        version_stamp=version_stamp,
    )


__all__ = [
    "Cap2Capability",
    "Cap2CapabilityDispositionV1",
    "Cap2CapabilityVerdict",
    "CapabilityDispositionV1",
    "CapabilityEvidenceV1",
    "build_cap2_disposition",
]
