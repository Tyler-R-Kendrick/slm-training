"""Lossless first-failure traces built from the canonical OpenUI evaluators.

This module is deliberately an adapter: ``binding_aware_meaningful_v2`` and the
G0--G12 verifier stack remain the scoring authorities.  It only gives their
ordered evidence a stable, versioned diagnostic vocabulary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from slm_training.data.contract import GenerationRequest
from slm_training.data.verify.stack import (
    Gate,
    GateStatus,
    VerificationContext,
    verify_record,
)
from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.meaningful_program import (
    CheckStatus,
    SemanticMeaningReportV2,
    binding_aware_meaningful_v2,
)

TAXONOMY_VERSION = "semantic_failure_taxonomy/v1"


class SemanticFailureFamily(str, Enum):
    LEXICAL_OR_PARSE = "lexical_or_parse"
    SCHEMA_OR_VALUE_ROLE = "schema_or_value_role"
    REFERENCE_OR_BINDING = "reference_or_binding"
    DATAFLOW_OR_EFFECT = "dataflow_or_effect"
    CANONICAL_OR_ROUNDTRIP = "canonical_or_roundtrip"
    PROMPT_CONTRACT_INVENTORY = "prompt_contract_inventory"
    COMPONENT_OR_ROLE_SELECTION = "component_or_role_selection"
    TOPOLOGY_OR_CARDINALITY = "topology_or_cardinality"
    PLACEHOLDER_IDENTITY_OR_FIDELITY = "placeholder_identity_or_fidelity"
    RUNTIME_OR_BEHAVIOR = "runtime_or_behavior"
    GROUNDING_OR_PROMPT_COMPATIBILITY = "grounding_or_prompt_compatibility"
    TRIVIAL_EMPTY_OR_MINIMAL_SHELL = "trivial_empty_or_minimal_shell"
    DUPLICATE_FILLER_OR_METRIC_GAMING = "duplicate_filler_or_metric_gaming"
    SEARCH_COVERAGE_OR_CANDIDATE_ABSENCE = "search_coverage_or_candidate_absence"
    LOCAL_SCORING_REGRET = "local_scoring_regret"
    SEARCH_OR_PRUNING_REGRET = "search_or_pruning_regret"
    GLOBAL_RANKING_REGRET = "global_ranking_regret"
    PLAN_PREDICTION_REGRET = "plan_prediction_regret"
    ENVIRONMENT_OR_EVALUATOR_FAILURE = "environment_or_evaluator_failure"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SemanticFailureTaxonomyV1:
    version: str = TAXONOMY_VERSION
    families: tuple[str, ...] = tuple(item.value for item in SemanticFailureFamily)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CHECK_FAMILIES = {
    "official_parse": SemanticFailureFamily.LEXICAL_OR_PARSE,
    "canonical_roundtrip": SemanticFailureFamily.CANONICAL_OR_ROUNDTRIP,
    "symbol_only_output": SemanticFailureFamily.PLACEHOLDER_IDENTITY_OR_FIDELITY,
    "prompt_relevant_semantic_content": SemanticFailureFamily.COMPONENT_OR_ROLE_SELECTION,
    "required_inventory_coverage": SemanticFailureFamily.PROMPT_CONTRACT_INVENTORY,
    "binding_correctness": SemanticFailureFamily.REFERENCE_OR_BINDING,
    "schema_value_role_correctness": SemanticFailureFamily.SCHEMA_OR_VALUE_ROLE,
    "whole_program_verifier": SemanticFailureFamily.LEXICAL_OR_PARSE,
    "anti_gaming": SemanticFailureFamily.DUPLICATE_FILLER_OR_METRIC_GAMING,
}


def family_for_reason(reason: str) -> SemanticFailureFamily:
    """Map a native reason code without discarding it; unknowns stay visible."""
    if reason.startswith(("official_parse", "verifier_g0", "verifier_g1", "verifier_input")):
        return SemanticFailureFamily.LEXICAL_OR_PARSE
    if reason.startswith(("canonical_roundtrip", "verifier_g8")):
        return SemanticFailureFamily.CANONICAL_OR_ROUNDTRIP
    if reason.startswith(("schema_value_role_mismatch", "free_form_output_string")):
        return SemanticFailureFamily.SCHEMA_OR_VALUE_ROLE
    if reason in {
        "binding_analysis_unavailable", "duplicate_binding", "unresolved_binding", "unreachable_binding"
    }:
        return SemanticFailureFamily.REFERENCE_OR_BINDING
    if reason in {"no_nontrivial_content"}:
        return SemanticFailureFamily.TRIVIAL_EMPTY_OR_MINIMAL_SHELL
    if reason in {"prompt_component_missing"}:
        return SemanticFailureFamily.COMPONENT_OR_ROLE_SELECTION
    if reason in {
        "prompt_contract_unknown", "required_placeholder_missing", "required_component_missing",
        "required_inventory_unknown",
    }:
        return SemanticFailureFamily.PROMPT_CONTRACT_INVENTORY
    if reason in {
        "unexpected_placeholder_identity", "duplicate_placeholder_identity",
        "placeholder_semantic_role_mismatch",
    }:
        return SemanticFailureFamily.PLACEHOLDER_IDENTITY_OR_FIDELITY
    if reason in {
        "gaming_analysis_unavailable", "duplicate_subtree_spam", "placeholder_spam",
        "low_diversity_filler", "mechanical_inventory_coverage",
    }:
        return SemanticFailureFamily.DUPLICATE_FILLER_OR_METRIC_GAMING
    if reason.startswith(("prediction_", "generation_", "test_record_", "stored_prediction_")):
        return SemanticFailureFamily.ENVIRONMENT_OR_EVALUATOR_FAILURE
    return SemanticFailureFamily.UNKNOWN


def _gate_family(gate: Gate) -> SemanticFailureFamily:
    return {
        Gate.LEXICAL: SemanticFailureFamily.LEXICAL_OR_PARSE,
        Gate.GRAMMAR: SemanticFailureFamily.LEXICAL_OR_PARSE,
        Gate.SCHEMA: SemanticFailureFamily.SCHEMA_OR_VALUE_ROLE,
        Gate.REFERENCES: SemanticFailureFamily.REFERENCE_OR_BINDING,
        Gate.DATAFLOW: SemanticFailureFamily.DATAFLOW_OR_EFFECT,
        Gate.RUNTIME: SemanticFailureFamily.RUNTIME_OR_BEHAVIOR,
        Gate.BEHAVIOR: SemanticFailureFamily.RUNTIME_OR_BEHAVIOR,
        Gate.GROUNDING: SemanticFailureFamily.GROUNDING_OR_PROMPT_COMPATIBILITY,
        Gate.CANONICAL: SemanticFailureFamily.CANONICAL_OR_ROUNDTRIP,
        Gate.PATCH: SemanticFailureFamily.SEARCH_COVERAGE_OR_CANDIDATE_ABSENCE,
        Gate.PROVENANCE: SemanticFailureFamily.ENVIRONMENT_OR_EVALUATOR_FAILURE,
        Gate.INDEPENDENT_JUDGE: SemanticFailureFamily.ENVIRONMENT_OR_EVALUATOR_FAILURE,
        Gate.HUMAN_AUDIT: SemanticFailureFamily.ENVIRONMENT_OR_EVALUATOR_FAILURE,
    }[gate]


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class SemanticFailureTraceV1:
    generation_id: str
    checkpoint_sha256: str | None
    suite: str | None
    raw_program_sha256: str
    canonical_program_sha256: str | None
    first_failed_gate: str | None
    first_failure_family: str | None
    all_failure_families: tuple[str, ...]
    reason_codes: tuple[str, ...]
    unmapped_reason_codes: tuple[str, ...]
    gate_outcomes: tuple[dict[str, str], ...]
    semantic_checks: tuple[dict[str, Any], ...]
    regret_evidence: dict[str, str]
    human_label: bool | None
    agentv_label: bool | None
    taxonomy_version: str
    evaluator_version: str
    evaluator_hash: str
    trace_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SemanticFailureTraceV1:
        return cls(
            **{
                **payload,
                "all_failure_families": tuple(payload["all_failure_families"]),
                "reason_codes": tuple(payload["reason_codes"]),
                "unmapped_reason_codes": tuple(payload["unmapped_reason_codes"]),
                "gate_outcomes": tuple(payload["gate_outcomes"]),
                "semantic_checks": tuple(payload["semantic_checks"]),
            }
        )


def trace_semantic_failure(
    prediction: str,
    record: ExampleRecord,
    *,
    request: GenerationRequest | None = None,
    generation_id: str | None = None,
    checkpoint_sha256: str | None = None,
    suite: str | None = None,
    semantic_report: SemanticMeaningReportV2 | None = None,
    context: VerificationContext | None = None,
    labels: dict[str, bool] | None = None,
    exact_regret: dict[str, str] | None = None,
) -> SemanticFailureTraceV1:
    """Build one deterministic trace.  Optional labels/oracles never affect failure."""
    report = semantic_report or binding_aware_meaningful_v2(
        prediction, record=record, request=request
    )
    candidate = ExampleRecord.from_dict({**record.to_dict(), "openui": prediction})
    verification = verify_record(candidate, context)
    gate_outcomes = tuple(result.to_dict() for result in verification.results)
    first_gate = next(
        (result.gate for result in verification.results if result.status is GateStatus.FAIL), None
    )
    families: list[SemanticFailureFamily] = []
    for check in report.checks:
        if check.status in {CheckStatus.FAIL, CheckStatus.UNKNOWN}:
            families.extend(family_for_reason(reason) for reason in check.reason_codes)
            if not check.reason_codes:
                families.append(_CHECK_FAMILIES.get(check.name, SemanticFailureFamily.UNKNOWN))
    if first_gate is not None:
        families.append(_gate_family(first_gate))
    ordered = tuple(dict.fromkeys(family.value for family in families))
    canonical = next(
        (check for check in report.checks if check.name == "canonical_roundtrip"), None
    )
    canonical_hash = _digest(prediction.strip()) if canonical and canonical.status is CheckStatus.PASS else None
    regret = {
        family.value: "NOT_APPLICABLE"
        for family in (
            SemanticFailureFamily.LOCAL_SCORING_REGRET,
            SemanticFailureFamily.SEARCH_OR_PRUNING_REGRET,
            SemanticFailureFamily.GLOBAL_RANKING_REGRET,
            SemanticFailureFamily.PLAN_PREDICTION_REGRET,
        )
    }
    for key, value in (exact_regret or {}).items():
        if key in regret and value:
            regret[key] = str(value)
            if key not in ordered:
                ordered += (key,)
    unmapped = tuple(reason for reason in report.reason_codes if family_for_reason(reason) is SemanticFailureFamily.UNKNOWN)
    semantic_checks = tuple(
        {
            "name": check.name,
            "status": check.status.value,
            "reason_codes": list(check.reason_codes),
            "evidence": [asdict(item) for item in check.evidence],
        }
        for check in report.checks
    )
    fingerprint_payload = {
        "generation_id": generation_id or record.id,
        "checkpoint_sha256": checkpoint_sha256,
        "suite": suite,
        "raw_program_sha256": _digest(prediction),
        "canonical_program_sha256": canonical_hash,
        "first_failed_gate": first_gate.value if first_gate else None,
        "families": ordered,
        "reason_codes": report.reason_codes,
        "taxonomy_version": TAXONOMY_VERSION,
        "evaluator_version": report.metric_version,
        "evaluator_hash": report.metric_implementation_hash,
    }
    labels = labels or {}
    return SemanticFailureTraceV1(
        generation_id=generation_id or record.id,
        checkpoint_sha256=checkpoint_sha256,
        suite=suite,
        raw_program_sha256=_digest(prediction),
        canonical_program_sha256=canonical_hash,
        first_failed_gate=first_gate.value if first_gate else None,
        first_failure_family=ordered[0] if ordered else None,
        all_failure_families=ordered,
        reason_codes=report.reason_codes,
        unmapped_reason_codes=unmapped,
        gate_outcomes=gate_outcomes,
        semantic_checks=semantic_checks,
        regret_evidence=regret,
        human_label=labels.get("human"),
        agentv_label=labels.get("agentv"),
        taxonomy_version=TAXONOMY_VERSION,
        evaluator_version=report.metric_version,
        evaluator_hash=report.metric_implementation_hash,
        trace_fingerprint=_digest(fingerprint_payload),
    )
