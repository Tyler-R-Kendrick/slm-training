"""Lossless first-failure traces built from the canonical OpenUI evaluators.

This module is deliberately an adapter: ``binding_aware_meaningful_v2`` and the
G0--G12 verifier stack remain the scoring authorities.  It only gives their
ordered evidence a stable, versioned diagnostic vocabulary.

``VerifierWitnessV1`` extends that adapter: it re-localizes an existing
``SemanticFailureTraceV1`` into typed, tamper-evident evidence records
(gate/reason/AST-path/span, completeness, redaction-safe detail) without
scoring anything itself -- it is a lossless superset of the trace, never a
competing legality authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
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


WITNESS_SCHEMA_VERSION = "verifier_witness/v1"

# Gates whose pass/fail is not a deterministic compiler-hard decision (judge
# or human evidence); their localization completeness is downgraded from
# EXACT to HEURISTIC even when a result is present.
_HEURISTIC_GATES = frozenset({Gate.INDEPENDENT_JUDGE, Gate.HUMAN_AUDIT})

_SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{10,}"),
    re.compile(r"sk-[A-Za-z0-9]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)=([^\s,;]+)"),
    re.compile(r"\b[0-9a-f]{48,}\b"),
)
_MAX_DETAIL_CHARS = 240


def _scrub_detail(text: str) -> str:
    """Strip secret-shaped substrings and cap length before text enters a witness."""
    if not text:
        return text
    scrubbed = text
    for pattern in _SECRET_PATTERNS:
        scrubbed = pattern.sub("[REDACTED]", scrubbed)
    if len(scrubbed) > _MAX_DETAIL_CHARS:
        scrubbed = f"{scrubbed[:_MAX_DETAIL_CHARS]}...[truncated:{_digest(text)[:12]}]"
    return scrubbed


def _gate_completeness(gate: Gate, status: str) -> str:
    if status == GateStatus.SKIP.value:
        return "UNKNOWN"
    return "HEURISTIC" if gate in _HEURISTIC_GATES else "EXACT"


# VCE-003: the closed set every legitimate producer in this module assigns
# (see _gate_completeness and the semantic_check branches below). from_dict
# fails closed on anything outside it rather than trusting an untrusted
# payload to self-report a stronger completeness tier than it earned.
_LOCALIZATION_COMPLETENESS_CLASSES = frozenset({"EXACT", "HEURISTIC", "UNKNOWN"})


@dataclass(frozen=True)
class VerifierLocalizationV1:
    """One typed, redaction-safe evidence unit inside a ``VerifierWitnessV1``.

    ``completeness_class`` is one of ``EXACT`` (deterministic authority gave a
    concrete localized fact), ``HEURISTIC`` (a judge/human authority gave a
    fact that is not independently replayable), or ``UNKNOWN`` (no authority
    localized this evidence point -- stays explicit rather than omitted).
    """

    source: str
    gate: str | None
    reason_code: str | None
    ast_path: str | None
    source_span: tuple[int, int] | None
    expected: str | None
    observed: str | None
    completeness_class: str
    certificate_id: str | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VerifierLocalizationV1:
        completeness = payload.get("completeness_class")
        if completeness not in _LOCALIZATION_COMPLETENESS_CLASSES:
            raise ValueError(
                f"unsupported verifier localization completeness_class: {completeness!r}"
            )
        span = payload.get("source_span")
        return cls(**{**payload, "source_span": tuple(span) if span is not None else None})


def _witness_digest(witness: VerifierWitnessV1) -> str:
    payload = asdict(witness)
    payload.pop("witness_digest", None)
    return _digest(payload)


@dataclass(frozen=True)
class VerifierWitnessV1:
    """Deterministic, tamper-evident typed localization over a failure trace.

    Purely diagnostic: it never overrides G0--G12 pass/fail or any semantic
    check outcome, it only re-expresses their existing evidence as typed,
    replayable localization records.
    """

    schema_version: str
    source_trace_fingerprint: str
    taxonomy_version: str
    evaluator_version: str
    evaluator_hash: str
    first_failed_gate: str | None
    localizations: tuple[VerifierLocalizationV1, ...]
    unresolved_localizations: tuple[str, ...]
    witness_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VerifierWitnessV1:
        if payload.get("schema_version") != WITNESS_SCHEMA_VERSION:
            raise ValueError(f"unsupported verifier witness schema: {payload.get('schema_version')!r}")
        witness = cls(
            **{
                **payload,
                "localizations": tuple(
                    VerifierLocalizationV1.from_dict(item) for item in payload["localizations"]
                ),
                "unresolved_localizations": tuple(payload["unresolved_localizations"]),
            }
        )
        if _witness_digest(witness) != witness.witness_digest:
            raise ValueError("verifier witness digest mismatch (tampered or corrupted payload)")
        return witness


def witness_integrity_ok(witness: VerifierWitnessV1) -> bool:
    """Recompute the witness digest and compare; used by replay/tamper checks."""
    return _witness_digest(witness) == witness.witness_digest


def build_verifier_witness(trace: SemanticFailureTraceV1) -> VerifierWitnessV1:
    """Losslessly re-localize an existing trace's evidence into typed witnesses.

    Deterministic: identical traces always yield an identical witness payload
    (including ``witness_digest``), because every field is derived only from
    ``trace``'s own already-deterministic fields.
    """
    localizations: list[VerifierLocalizationV1] = []
    unresolved: list[str] = []

    for outcome in trace.gate_outcomes:
        gate = Gate(outcome["gate"])
        completeness = _gate_completeness(gate, outcome["status"])
        if completeness == "UNKNOWN":
            unresolved.append(f"gate:{outcome['gate']}")
        localizations.append(
            VerifierLocalizationV1(
                source="gate",
                gate=outcome["gate"],
                reason_code=None,
                ast_path=None,
                source_span=None,
                expected="pass",
                observed=outcome["status"],
                completeness_class=completeness,
                certificate_id=None,
                detail=_scrub_detail(outcome.get("detail", "")),
            )
        )

    for check in trace.semantic_checks:
        reason_code = ",".join(check["reason_codes"]) if check["reason_codes"] else None
        evidence_items = check["evidence"]
        if not evidence_items:
            if check["status"] != "PASS":
                unresolved.append(f"semantic_check:{check['name']}")
                localizations.append(
                    VerifierLocalizationV1(
                        source="semantic_check",
                        gate=None,
                        reason_code=reason_code,
                        ast_path=None,
                        source_span=None,
                        expected=None,
                        observed=check["status"],
                        completeness_class="UNKNOWN",
                        certificate_id=None,
                        detail=_scrub_detail(reason_code or ""),
                    )
                )
            continue
        for item in evidence_items:
            span = item.get("source_span")
            ast_path = item.get("ast_path") or None
            if not ast_path:
                unresolved.append(f"semantic_check:{check['name']}:unlocalized")
            localizations.append(
                VerifierLocalizationV1(
                    source="semantic_check",
                    gate=None,
                    reason_code=reason_code,
                    ast_path=ast_path,
                    source_span=tuple(span) if span is not None else None,
                    expected=None,
                    observed=check["status"],
                    completeness_class="EXACT" if ast_path else "UNKNOWN",
                    certificate_id=None,
                    detail=_scrub_detail(item.get("detail", "")),
                )
            )

    witness = VerifierWitnessV1(
        schema_version=WITNESS_SCHEMA_VERSION,
        source_trace_fingerprint=trace.trace_fingerprint,
        taxonomy_version=trace.taxonomy_version,
        evaluator_version=trace.evaluator_version,
        evaluator_hash=trace.evaluator_hash,
        first_failed_gate=trace.first_failed_gate,
        localizations=tuple(localizations),
        unresolved_localizations=tuple(dict.fromkeys(unresolved)),
        witness_digest="",
    )
    return replace(witness, witness_digest=_witness_digest(witness))
