"""Goal-verifier profiles and redacted terminal-evidence contracts (PGS-B01 / SLM-497).

``GoalVerifierProfileV1`` pins the semantics of every goal-support query: which
constraints, gates, evaluators, and metric identities are in scope, under which
closed mode, and with what authority. ``GoalTerminalEvidenceV1`` is the
request-local, privacy-bounded evidence record that binds verifier outcomes back
to a profile digest without persisting raw prompts, OpenUI source, witness text,
secrets, logs, timestamps, process ids, or model scores.

This module defines contracts only — no OpenUI parsing, gate execution,
meaningful evaluator execution, support search, obstruction, closure, or decode
wiring.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from slm_training.data.progspec.goal_constraints import (
    COMPILED_GOAL_CONSTRAINT_SET_SCHEMA_VERSION,
    CompiledGoalConstraintSetV1,
    GoalConstraintEvaluationV1,
    canonical_json_bytes,
)
from slm_training.data.progspec.synthesis_problem import PackIdentityV1

__all__ = [
    "GOAL_VERIFIER_PROFILE_SCHEMA_VERSION",
    "GOAL_TERMINAL_EVIDENCE_SCHEMA_VERSION",
    "GOAL_SUPPORT_IMPLEMENTATION_VERSION",
    "GoalVerifierMode",
    "GoalProfileAuthorityTier",
    "GoalTerminalStatus",
    "MetricIdentityV1",
    "EvaluatorIdentityV1",
    "GoalVerifierProfileV1",
    "GoalGateResultV1",
    "GoalEvaluatorResultV1",
    "GoalFailureAtomV1",
    "GoalUnknownAtomV1",
    "GoalTerminalEvidenceV1",
    "profile_mode_authority_table",
    "terminal_status_decision_table",
    "profile_digest_inputs",
    "compute_pack_identity_digest",
    "validate_profile_against_constraint_set",
    "redact_bounded_string",
    "redact_terminal_evidence_dict",
]

GOAL_VERIFIER_PROFILE_SCHEMA_VERSION = "goal_verifier_profile/v1"
GOAL_TERMINAL_EVIDENCE_SCHEMA_VERSION = "goal_terminal_evidence/v1"
GOAL_SUPPORT_IMPLEMENTATION_VERSION = "goal_support/v1"

GoalVerifierMode = Literal["production_exact", "evaluation_oracle", "advisory_diagnostic"]

GoalProfileAuthorityTier = Literal[
    "compiler-hard",
    "verifier-hard",
    "evaluation-only",
    "advisory-learned",
    "oracle-diagnostic",
]

GoalTerminalStatus = Literal["ACCEPT", "REJECT", "UNAVAILABLE"]

GoalAtomCompleteness = Literal["EXACT", "HEURISTIC", "UNKNOWN"]

GoalAtomSourceKind = Literal[
    "constraint",
    "gate",
    "evaluator",
    "structural",
]

_HARD_AUTHORITIES: frozenset[str] = frozenset({"compiler-hard", "verifier-hard"})
_ADVISORY_AUTHORITIES: frozenset[str] = frozenset({"advisory-learned", "oracle-diagnostic"})
_EVALUATION_AUTHORITIES: frozenset[str] = frozenset({"evaluation-only"})

_MODE_ALLOWED_AUTHORITY: dict[str, frozenset[str]] = {
    "production_exact": _HARD_AUTHORITIES,
    "evaluation_oracle": _EVALUATION_AUTHORITIES,
    "advisory_diagnostic": _ADVISORY_AUTHORITIES,
}

# Gates whose pass/fail is not a deterministic exact decision (mirrors
# evals/semantic_failure.py heuristic gate set — referenced by identity only).
_HEURISTIC_GATE_IDS: frozenset[str] = frozenset({"G11", "G12", "independent_judge", "human_audit"})

_FORBIDDEN_EVIDENCE_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "raw_prompt",
        "openui_source",
        "witness_text",
        "log",
        "logs",
        "timestamp",
        "timestamps",
        "pid",
        "process_id",
        "model_score",
        "model_scores",
        "opaque_literal",
        "opaque_literals",
    }
)

_SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{10,}"),
    re.compile(r"sk-[A-Za-z0-9]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)=([^\s,;]+)"),
    re.compile(r"\b[0-9a-f]{48,}\b"),
)
_MAX_BOUNDED_FIELD_CHARS = 240


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def redact_bounded_string(text: str) -> str:
    """Strip secret-shaped substrings and bound length for terminal evidence."""
    if not text:
        return text
    if len(text) == 64 and all(char in "0123456789abcdef" for char in text):
        return text
    if "...[truncated:" in text:
        return text
    scrubbed = text
    for pattern in _SECRET_PATTERNS:
        scrubbed = pattern.sub("[REDACTED]", scrubbed)
    if len(scrubbed) > _MAX_BOUNDED_FIELD_CHARS:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        scrubbed = f"{scrubbed[:_MAX_BOUNDED_FIELD_CHARS]}...[truncated:{digest}]"
    return scrubbed


def compute_pack_identity_digest(pack_identity: PackIdentityV1) -> str:
    return hashlib.sha256(canonical_json_bytes(pack_identity.model_dump(mode="json"))).hexdigest()


def profile_mode_authority_table() -> dict[str, Any]:
    """Closed profile-mode → authority/partition/pruning matrix."""
    return {
        "production_exact": {
            "allowed_authority_tiers": sorted(_HARD_AUTHORITIES),
            "constraint_partition": "hard_constraint_ids",
            "constraint_completeness_required": "EXACT",
            "gate_policy": "deterministic_exact_only",
            "heuristic_gates_forbidden": sorted(_HEURISTIC_GATE_IDS),
            "may_authorize_pruning": True,
            "unresolved_hard_ambiguity_forbidden": True,
        },
        "evaluation_oracle": {
            "allowed_authority_tiers": sorted(_EVALUATION_AUTHORITIES),
            "constraint_partition": "evaluation_constraint_ids",
            "constraint_completeness_required": None,
            "gate_policy": "frozen_eval_oracle",
            "heuristic_gates_forbidden": [],
            "may_authorize_pruning": False,
            "unresolved_hard_ambiguity_forbidden": False,
        },
        "advisory_diagnostic": {
            "allowed_authority_tiers": sorted(_ADVISORY_AUTHORITIES),
            "constraint_partition": "advisory_constraint_ids",
            "constraint_completeness_required": None,
            "gate_policy": "advisory_scoring",
            "heuristic_gates_forbidden": [],
            "may_authorize_pruning": False,
            "unresolved_hard_ambiguity_forbidden": False,
        },
    }


def terminal_status_decision_table() -> dict[str, Any]:
    """Exact ACCEPT / REJECT / UNAVAILABLE semantics (not VerificationReport.ok)."""
    return {
        "ACCEPT": {
            "meaning": "Every required hard/exact obligation under the profile passed with exact authority.",
            "overall_when": "All required gates/evaluators/constraints resolved ACCEPT with EXACT sources.",
            "pruning_authority": "Only when profile.mode=production_exact and sources are exact-hard.",
        },
        "REJECT": {
            "meaning": "At least one required obligation failed with exact, replayable authority.",
            "overall_when": "Any required gate/evaluator/constraint is REJECT with EXACT completeness.",
            "pruning_authority": "production_exact only; never from advisory/evaluation modes.",
        },
        "UNAVAILABLE": {
            "meaning": "Required evidence missing, skipped, heuristic, advisory-only, or version-mismatched.",
            "overall_when": "Any required check is UNAVAILABLE or lacks exact authority.",
            "pruning_authority": False,
        },
    }


def profile_digest_inputs() -> tuple[str, ...]:
    """Identity fields covered by GoalVerifierProfileV1.digest (deterministic order)."""
    return (
        "schema_version",
        "profile_id",
        "mode",
        "problem_digest",
        "constraint_set_digest",
        "pack_identity",
        "pack_identity_digest",
        "required_constraint_ids",
        "required_gates",
        "required_evaluators",
        "metric_identities",
        "implementation_version",
        "authority_tier",
    )


class MetricIdentityV1(_StrictModel):
    metric_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    implementation_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$|^$")


class EvaluatorIdentityV1(_StrictModel):
    evaluator_id: str = Field(min_length=1)
    version: str | None = Field(default=None, min_length=1)
    implementation_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$|^$")

    @model_validator(mode="after")
    def _require_version_or_hash(self) -> "EvaluatorIdentityV1":
        if not self.version and not self.implementation_hash:
            raise ValueError("evaluator identity requires version or implementation_hash")
        return self


class GoalVerifierProfileV1(_StrictModel):
    """Digest-bound profile that pins goal-support query semantics."""

    schema_version: str = Field(
        default=GOAL_VERIFIER_PROFILE_SCHEMA_VERSION,
        pattern=r"^goal_verifier_profile/v1$",
    )
    profile_id: str = Field(min_length=1)
    mode: GoalVerifierMode
    problem_digest: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    constraint_set_digest: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    pack_identity: PackIdentityV1
    pack_identity_digest: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    required_constraint_ids: tuple[str, ...] = ()
    required_gates: tuple[str, ...] = ()
    required_evaluators: tuple[EvaluatorIdentityV1, ...] = ()
    metric_identities: tuple[MetricIdentityV1, ...] = ()
    implementation_version: str = Field(
        default=GOAL_SUPPORT_IMPLEMENTATION_VERSION, min_length=1
    )
    authority_tier: GoalProfileAuthorityTier
    digest: str = Field(default="", pattern=r"^[0-9a-f]{64}$|^$")

    @model_validator(mode="after")
    def _check_integrity(self) -> "GoalVerifierProfileV1":
        expected_pack = compute_pack_identity_digest(self.pack_identity)
        if self.pack_identity_digest != expected_pack:
            raise ValueError("pack_identity_digest does not match pack_identity payload")

        allowed = _MODE_ALLOWED_AUTHORITY[self.mode]
        if self.authority_tier not in allowed:
            raise ValueError(
                f"authority_tier {self.authority_tier!r} incompatible with mode {self.mode!r}; "
                f"expected one of {sorted(allowed)}"
            )

        for name, values in (
            ("required_constraint_ids", self.required_constraint_ids),
            ("required_gates", self.required_gates),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")

        evaluator_ids = [item.evaluator_id for item in self.required_evaluators]
        if len(evaluator_ids) != len(set(evaluator_ids)):
            raise ValueError("required_evaluators evaluator_id values must be unique")

        metric_ids = [item.metric_id for item in self.metric_identities]
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("metric_identities metric_id values must be unique")

        if self.mode == "production_exact":
            heuristic = sorted(set(self.required_gates) & _HEURISTIC_GATE_IDS)
            if heuristic:
                raise ValueError(
                    "production_exact forbids heuristic gate identities: "
                    f"{heuristic}"
                )

        if self.digest:
            expected = self.compute_digest()
            if self.digest != expected:
                raise ValueError("digest does not match canonical payload")

        return self

    def _canonical_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("digest", None)
        payload["required_constraint_ids"] = sorted(payload["required_constraint_ids"])
        payload["required_gates"] = sorted(payload["required_gates"])
        payload["required_evaluators"] = sorted(
            payload["required_evaluators"], key=lambda item: item["evaluator_id"]
        )
        payload["metric_identities"] = sorted(
            payload["metric_identities"], key=lambda item: item["metric_id"]
        )
        return payload

    def compute_digest(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self._canonical_payload())).hexdigest()

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        if not include_digest:
            payload.pop("digest", None)
        elif not payload.get("digest"):
            payload["digest"] = self.compute_digest()
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GoalVerifierProfileV1":
        if str(value.get("schema_version")) != GOAL_VERIFIER_PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported GoalVerifierProfileV1 version")
        return cls.model_validate(value)


def validate_profile_against_constraint_set(
    profile: GoalVerifierProfileV1,
    constraint_set: CompiledGoalConstraintSetV1,
) -> None:
    """Fail closed when profile identities are stale or mode/partition laws break."""
    if constraint_set.schema_version != COMPILED_GOAL_CONSTRAINT_SET_SCHEMA_VERSION:
        raise ValueError("unsupported CompiledGoalConstraintSetV1 version")

    if profile.problem_digest != constraint_set.problem_digest:
        raise ValueError("profile problem_digest is stale vs constraint set")
    if profile.constraint_set_digest != constraint_set.digest:
        raise ValueError("profile constraint_set_digest is stale vs constraint set")
    if profile.pack_identity_digest != constraint_set.pack_identity_digest:
        raise ValueError("profile pack_identity_digest is stale vs constraint set")

    known_ids = {constraint.constraint_id for constraint in constraint_set.constraints}
    unknown = set(profile.required_constraint_ids) - known_ids
    if unknown:
        raise ValueError(
            f"required_constraint_ids reference unknown constraints: {sorted(unknown)}"
        )

    authority_by_id = {
        constraint.constraint_id: constraint for constraint in constraint_set.constraints
    }

    if profile.mode == "production_exact":
        allowed_partition = set(constraint_set.hard_constraint_ids)
        wrong_partition = sorted(
            cid for cid in profile.required_constraint_ids if cid not in allowed_partition
        )
        if wrong_partition:
            raise ValueError(
                "production_exact required constraints must be hard-eligible: "
                f"{wrong_partition}"
            )
        for constraint_id in profile.required_constraint_ids:
            constraint = authority_by_id[constraint_id]
            if constraint.completeness != "EXACT":
                raise ValueError(
                    f"production_exact requires EXACT completeness for {constraint_id!r}"
                )
        for group in constraint_set.ambiguity_groups:
            members = set(group.member_constraint_ids)
            if members & set(profile.required_constraint_ids):
                raise ValueError(
                    "production_exact forbids unresolved hard ambiguity among "
                    f"required constraints (group {group.group_id!r})"
                )
    elif profile.mode == "evaluation_oracle":
        allowed_partition = set(constraint_set.evaluation_constraint_ids)
        wrong_partition = sorted(
            cid for cid in profile.required_constraint_ids if cid not in allowed_partition
        )
        if wrong_partition:
            raise ValueError(
                "evaluation_oracle required constraints must be evaluation-only: "
                f"{wrong_partition}"
            )
    else:
        allowed_partition = set(constraint_set.advisory_constraint_ids)
        wrong_partition = sorted(
            cid for cid in profile.required_constraint_ids if cid not in allowed_partition
        )
        if wrong_partition:
            raise ValueError(
                "advisory_diagnostic required constraints must be advisory: "
                f"{wrong_partition}"
            )


class GoalGateResultV1(_StrictModel):
    gate_id: str = Field(min_length=1)
    status: GoalTerminalStatus
    reason_code: str = Field(default="")

    @field_validator("reason_code")
    @classmethod
    def _redact_reason(cls, value: str) -> str:
        return redact_bounded_string(value)


class GoalEvaluatorResultV1(_StrictModel):
    evaluator_id: str = Field(min_length=1)
    status: GoalTerminalStatus
    reason_code: str = Field(default="")

    @field_validator("reason_code")
    @classmethod
    def _redact_reason(cls, value: str) -> str:
        return redact_bounded_string(value)


class GoalFailureAtomV1(_StrictModel):
    atom_id: str = Field(min_length=1)
    reason_code: str = Field(min_length=1)
    source_kind: GoalAtomSourceKind
    completeness_class: GoalAtomCompleteness
    source_identity: str | None = None
    implementation_version: str | None = Field(default=None, max_length=64)

    @field_validator("reason_code", "source_identity")
    @classmethod
    def _redact_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return redact_bounded_string(value)


class GoalUnknownAtomV1(_StrictModel):
    atom_id: str = Field(min_length=1)
    reason_code: str = Field(min_length=1)
    source_kind: GoalAtomSourceKind
    source_identity: str | None = None

    @field_validator("reason_code", "source_identity")
    @classmethod
    def _redact_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return redact_bounded_string(value)


def _exact_atom_allowed(atom: GoalFailureAtomV1, *, implementation_version: str) -> bool:
    if atom.completeness_class != "EXACT":
        return False
    if atom.implementation_version not in (None, implementation_version):
        return False
    if atom.source_kind == "constraint":
        return True
    if atom.source_kind == "gate":
        return atom.source_identity not in _HEURISTIC_GATE_IDS
    if atom.source_kind == "evaluator":
        return atom.implementation_version == implementation_version
    return atom.source_kind == "structural"


class GoalTerminalEvidenceV1(_StrictModel):
    """Redacted, digest-bound terminal evidence for one goal-support query."""

    schema_version: str = Field(
        default=GOAL_TERMINAL_EVIDENCE_SCHEMA_VERSION,
        pattern=r"^goal_terminal_evidence/v1$",
    )
    profile_digest: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    program_digest: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    canonical_program_digest: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    program_spec_digest: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    semantic_plan_digest: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    constraint_evaluations: tuple[GoalConstraintEvaluationV1, ...] = ()
    required_gate_results: tuple[GoalGateResultV1, ...] = ()
    required_evaluator_results: tuple[GoalEvaluatorResultV1, ...] = ()
    mandatory_failure_atoms: tuple[GoalFailureAtomV1, ...] = ()
    mandatory_unknown_atoms: tuple[GoalUnknownAtomV1, ...] = ()
    structural_status: GoalTerminalStatus
    overall_status: GoalTerminalStatus
    evidence_digest: str = Field(default="", pattern=r"^[0-9a-f]{64}$|^$")

    @model_validator(mode="after")
    def _check_integrity(self) -> "GoalTerminalEvidenceV1":
        for name, values in (
            ("constraint_evaluations", [item.constraint_id for item in self.constraint_evaluations]),
            ("required_gate_results", [item.gate_id for item in self.required_gate_results]),
            (
                "required_evaluator_results",
                [item.evaluator_id for item in self.required_evaluator_results],
            ),
            ("mandatory_failure_atoms", [item.atom_id for item in self.mandatory_failure_atoms]),
            ("mandatory_unknown_atoms", [item.atom_id for item in self.mandatory_unknown_atoms]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} identifiers must be unique")

        for atom in self.mandatory_failure_atoms:
            if not _exact_atom_allowed(
                atom, implementation_version=GOAL_SUPPORT_IMPLEMENTATION_VERSION
            ):
                raise ValueError(
                    "mandatory_failure_atoms cannot claim EXACT authority when source is "
                    "unknown, skipped, heuristic, advisory, or version-mismatched "
                    f"(atom_id={atom.atom_id!r})"
                )

        if self.evidence_digest:
            expected = self.compute_digest()
            if self.evidence_digest != expected:
                raise ValueError("evidence_digest does not match canonical payload")

        return self

    def _canonical_payload(self) -> dict[str, Any]:
        return _canonical_terminal_payload(self.model_dump(mode="json"))

    def compute_digest(self) -> str:
        return _terminal_evidence_digest(self._canonical_payload())

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = redact_terminal_evidence_dict(self.model_dump(mode="json"))
        payload.pop("evidence_digest", None)
        if include_digest:
            payload["evidence_digest"] = _terminal_evidence_digest(payload)
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GoalTerminalEvidenceV1":
        if str(value.get("schema_version")) != GOAL_TERMINAL_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported GoalTerminalEvidenceV1 version")
        forbidden = _FORBIDDEN_EVIDENCE_FIELD_NAMES & set(value)
        if forbidden:
            raise ValueError(f"forbidden terminal-evidence fields present: {sorted(forbidden)}")
        return cls.model_validate(value)


def redact_terminal_evidence_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact string leaves and reject forbidden persistence keys."""

    def _walk(value: Any, *, path: str) -> Any:
        if isinstance(value, dict):
            for key in value:
                if key in _FORBIDDEN_EVIDENCE_FIELD_NAMES:
                    raise ValueError(f"forbidden terminal-evidence field at {path}.{key}")
            return {key: _walk(item, path=f"{path}.{key}") for key, item in value.items()}
        if isinstance(value, list):
            return [_walk(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, str):
            return redact_bounded_string(value)
        return value

    return _walk(payload, path="evidence")


def _canonical_terminal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    canonical = dict(payload)
    canonical.pop("evidence_digest", None)
    canonical["constraint_evaluations"] = sorted(
        canonical["constraint_evaluations"], key=lambda item: item["constraint_id"]
    )
    canonical["required_gate_results"] = sorted(
        canonical["required_gate_results"], key=lambda item: item["gate_id"]
    )
    canonical["required_evaluator_results"] = sorted(
        canonical["required_evaluator_results"], key=lambda item: item["evaluator_id"]
    )
    canonical["mandatory_failure_atoms"] = sorted(
        canonical["mandatory_failure_atoms"], key=lambda item: item["atom_id"]
    )
    canonical["mandatory_unknown_atoms"] = sorted(
        canonical["mandatory_unknown_atoms"], key=lambda item: item["atom_id"]
    )
    return canonical


def _terminal_evidence_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(_canonical_terminal_payload(payload))
    ).hexdigest()
