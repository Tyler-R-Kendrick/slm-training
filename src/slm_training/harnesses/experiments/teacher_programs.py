"""Fail-closed SLM-266 teacher-program request and admission contracts.

This owner deliberately has no provider client.  It freezes a provider-neutral
generation manifest and admits *already archived* raw responses through the
existing ProgramSpec, verifier, and leakage owners.  A missing judge, human
audit, budget, or protected-split manifest is a rejection, never an inferred
pass.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from slm_training.data.leakage import (
    find_leakage,
    fingerprint_openui,
    fingerprint_pair,
)
from slm_training.data.progspec.schema import ProgramSpec, emit_record
from slm_training.data.verify import (
    Gate,
    GateStatus,
    VerificationContext,
    verify_record,
)

REQUEST_SCHEMA = "teacher_program_request/v1"
GENERATION_SCHEMA = "teacher_program_generation_manifest/v1"
ADMISSION_SCHEMA = "teacher_program_admission/v1"


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class AdmissionMode(StrEnum):
    DEEP_VERIFIED = "deep_verified"
    PARSE_ONLY = "parse_only"
    NO_CANONICAL_DEDUP = "no_canonical_dedup"


@dataclass(frozen=True)
class TeacherProgramRequestV1:
    """Source-intent request; never contains a target program or protected prompt."""

    request_id: str
    coverage_manifest_hash: str
    coverage_gap_ids: tuple[str, ...]
    allowed_components: tuple[str, ...]
    output_kind: str
    protected_exclusion_manifest_hash: str
    template_hash: str
    seed: int
    schema_version: str = REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if not all(
            (self.request_id, self.coverage_manifest_hash, self.output_kind,
             self.protected_exclusion_manifest_hash, self.template_hash)
        ):
            raise ValueError("teacher request fields must be non-empty")
        if not self.coverage_gap_ids or not self.allowed_components:
            raise ValueError("teacher request requires coverage gaps and allowed components")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "coverage_gap_ids": list(self.coverage_gap_ids), "allowed_components": list(self.allowed_components)}


@dataclass(frozen=True)
class TeacherProgramGenerationManifestV1:
    """Frozen no-spend manifest; an executor must enforce its caps separately."""

    manifest_id: str
    provider: str
    model: str
    revision: str
    request_ids: tuple[str, ...]
    max_dollars: float
    max_input_tokens: int
    max_output_tokens: int
    protected_exclusion_manifest_hash: str
    schema_version: str = GENERATION_SCHEMA

    def __post_init__(self) -> None:
        if not all((self.manifest_id, self.provider, self.model, self.revision, self.protected_exclusion_manifest_hash)):
            raise ValueError("generation manifest identity fields must be non-empty")
        if not self.request_ids:
            raise ValueError("generation manifest requires at least one request")
        if min(self.max_dollars, self.max_input_tokens, self.max_output_tokens) <= 0:
            raise ValueError("generation manifest requires positive hard budget caps")

    @property
    def manifest_hash(self) -> str:
        return _hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "request_ids": list(self.request_ids)}


@dataclass(frozen=True)
class TeacherProgramCandidate:
    """One immutable raw response, with exactly one declared program payload."""

    candidate_id: str
    request_id: str
    response_id: str
    prompt: str
    program_payloads: tuple[str, ...]
    provider: str
    model: str
    revision: str
    generator_family: str
    judge_family: str | None
    coverage_gap_ids: tuple[str, ...]
    program_family_id: str
    lineage_id: str
    split_group_id: str
    provenance_complete: bool
    independent_judge_passed: bool | None = None
    human_audit_passed: bool | None = None
    require_runtime: bool = False
    require_behavior: bool = False
    required_facts: tuple[str, ...] = ()
    forbidden_facts: tuple[str, ...] = ()
    facts: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdmissionResultV1:
    mode: AdmissionMode
    accepted: tuple[dict[str, Any], ...]
    rejected: tuple[dict[str, Any], ...]
    manifest_hash: str
    schema_version: str = ADMISSION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "accepted": list(self.accepted),
            "rejected": list(self.rejected),
            "accepted_n": len(self.accepted),
            "rejected_n": len(self.rejected),
            "manifest_hash": self.manifest_hash,
        }


def _required_gates(candidate: TeacherProgramCandidate, mode: AdmissionMode) -> frozenset[Gate]:
    if mode is AdmissionMode.PARSE_ONLY:
        return frozenset({Gate.LEXICAL, Gate.GRAMMAR})
    gates = {
        Gate.LEXICAL, Gate.GRAMMAR, Gate.SCHEMA, Gate.REFERENCES, Gate.DATAFLOW,
        Gate.GROUNDING, Gate.CANONICAL, Gate.PROVENANCE,
        Gate.INDEPENDENT_JUDGE,
    }
    if candidate.require_runtime:
        gates.add(Gate.RUNTIME)
    if candidate.require_behavior:
        gates.add(Gate.BEHAVIOR)
    return frozenset(gates)


def _reject(candidate: TeacherProgramCandidate, reason: str, **detail: Any) -> dict[str, Any]:
    return {"candidate_id": candidate.candidate_id, "reason": reason, **detail}


def _admit_one(
    candidate: TeacherProgramCandidate,
    *,
    mode: AdmissionMode,
    protected_fingerprints: Mapping[str, set[str]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if len(candidate.program_payloads) != 1:
        return None, _reject(candidate, "program_payload_count", count=len(candidate.program_payloads))
    if mode is not AdmissionMode.PARSE_ONLY and candidate.generator_family == candidate.judge_family:
        return None, _reject(candidate, "generator_judge_not_independent")
    try:
        spec = ProgramSpec.from_openui(
            id=candidate.candidate_id,
            openui=candidate.program_payloads[0],
            facts=dict(candidate.facts),
            program_family_id=candidate.program_family_id,
            lineage_id=candidate.lineage_id,
            split_group_id=candidate.split_group_id,
            provenance={
                "teacher_provider": candidate.provider,
                "teacher_model": candidate.model,
                "teacher_revision": candidate.revision,
                "request_id": candidate.request_id,
                "response_id": candidate.response_id,
                "coverage_gap_ids": list(candidate.coverage_gap_ids),
            },
        )
    except (ValueError, TypeError) as exc:
        return None, _reject(candidate, "program_parse", detail=str(exc))
    record = emit_record(
        spec,
        prompt=candidate.prompt,
        task="generation",
        source="teacher",
        tier="Bronze",
        determinacy="teacher_generated",
        meta={
            "source_kind": "teacher",
            "provenance_complete": candidate.provenance_complete,
            "independent_judge_passed": candidate.independent_judge_passed,
            "human_audit_passed": candidate.human_audit_passed,
            "require_runtime": candidate.require_runtime,
            "require_behavior": candidate.require_behavior,
            "required_facts": list(candidate.required_facts),
            "forbidden_facts": list(candidate.forbidden_facts),
        },
    )
    leakage = find_leakage(record, dict(protected_fingerprints))
    if leakage:
        return None, _reject(candidate, "protected_split_leakage", leakage=leakage)
    report = verify_record(
        record,
        VerificationContext(
            source_kind="teacher",
            provenance_complete=candidate.provenance_complete,
            independent_judge_passed=candidate.independent_judge_passed,
            human_audit_passed=candidate.human_audit_passed,
            require_runtime=candidate.require_runtime,
            require_behavior=candidate.require_behavior,
            required_facts=candidate.required_facts,
            forbidden_facts=candidate.forbidden_facts,
        ),
    )
    statuses = {result.gate: result.status for result in report.results}
    missing = sorted(
        gate.value for gate in _required_gates(candidate, mode)
        if statuses[gate] is not GateStatus.PASS
    )
    if missing:
        return None, _reject(candidate, "required_gate_not_pass", gates=missing, verification=report.to_dict())
    admission_tier = (
        None
        if mode is AdmissionMode.PARSE_ONLY
        else "Gold" if report.tier.value == "Gold" else "Silver"
    )
    return {
        "candidate_id": candidate.candidate_id,
        "record": record.to_dict(),
        "canonical_root_hash": fingerprint_openui(record.openui),
        "pair_hash": fingerprint_pair(record.prompt, record.openui),
        "verification": report.to_dict(),
        "admission_tier": admission_tier,
    }, None


def admit_teacher_programs(
    candidates: Iterable[TeacherProgramCandidate],
    *,
    mode: AdmissionMode | str,
    protected_fingerprints: Mapping[str, set[str]],
) -> AdmissionResultV1:
    """Admit one declared control axis from the identical raw candidate pool.

    Deep mode keeps one deterministic root representative; no-canonical-dedup
    keeps distinct records and only removes exact prompt/program byte pairs.
    """
    mode = AdmissionMode(mode)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        row, rejection = _admit_one(candidate, mode=mode, protected_fingerprints=protected_fingerprints)
        if row is None:
            rejected.append(rejection or _reject(candidate, "unknown"))
        else:
            accepted.append(row)
    seen: set[str] = set()
    materialized: list[dict[str, Any]] = []
    dedup_key = "canonical_root_hash" if mode is AdmissionMode.DEEP_VERIFIED else "pair_hash"
    for row in accepted:
        key = str(row[dedup_key])
        if key in seen:
            rejected.append({"candidate_id": row["candidate_id"], "reason": f"duplicate_{dedup_key}"})
        else:
            seen.add(key)
            materialized.append(row)
    payload = {
        "mode": mode.value,
        "accepted": materialized,
        "rejected": rejected,
    }
    return AdmissionResultV1(mode, tuple(materialized), tuple(rejected), _hash(payload))


__all__ = [
    "ADMISSION_SCHEMA", "GENERATION_SCHEMA", "REQUEST_SCHEMA", "AdmissionMode",
    "AdmissionResultV1", "TeacherProgramCandidate", "TeacherProgramGenerationManifestV1",
    "TeacherProgramRequestV1", "admit_teacher_programs",
]
