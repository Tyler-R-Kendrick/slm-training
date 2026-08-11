"""Goal-verifier profiles, terminal evidence, and goal-aware support (PGS-B01/C01/C02).

``GoalVerifierProfileV1`` pins the semantics of every goal-support query: which
constraints, gates, evaluators, and metric identities are in scope, under which
closed mode, and with what authority. ``GoalTerminalEvidenceV1`` is the
request-local, privacy-bounded evidence record that binds verifier outcomes back
to a profile digest without persisting raw prompts, OpenUI source, witness text,
secrets, logs, timestamps, process ids, or model scores.

``GoalSupportProvider`` (PGS-C01 / SLM-499) is a thin goal-aware wrapper over the
existing VSS enumerative oracle and certificate replay. The canonical search proof
remains :class:`~slm_training.dsl.solver.support.SupportCertificate`; property-
specific semantics travel in digest-bound :class:`GoalSupportResultV1` sidecars
with a compiler-hard ``exact_goal_closure`` authority guard.

``DomainObstructionCoreV1`` (PGS-C02 / SLM-500) records deterministic bounded
domain-relative hitting sets over exact mandatory failure atoms when legal
emission preconditions hold.

``GoalActionEvidenceV1`` and ``GoalDomainAdequacyReportV1`` (PGS-C03 / SLM-501)
classify one exact legal set into supported/unsupported/unknown/unobserved
partitions and diagnose domain adequacy vs selection regret.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from slm_training.dsl.solver.closure import ClosureResult, QueryOrder, exact_closure
from slm_training.dsl.solver.state import FiniteDomainState, SupportVerdict
from slm_training.dsl.solver.support import (
    EnumerativeSupportOracle,
    ProblemExpander,
    ReplayResult,
    SearchCounters,
    SupportCertificate,
    SupportQuery,
    SupportResult,
    Verifier,
    replay_support_certificate,
)

from slm_training.data.progspec.goal_constraints import (
    COMPILED_GOAL_CONSTRAINT_SET_SCHEMA_VERSION,
    CompiledGoalConstraintSetV1,
    GoalConstraintEvaluationV1,
    canonical_json_bytes,
)
from slm_training.data.progspec.synthesis_problem import PackIdentityV1
from slm_training.data.semantic_plan.requirements_compile import (
    COMPILER_VERSION,
    evaluator_digest,
)
from slm_training.data.verify.stack import gate_stack_implementation_digest

GOAL_VERIFIER_PROFILE_SCHEMA_VERSION = "goal_verifier_profile/v1"
GOAL_TERMINAL_EVIDENCE_SCHEMA_VERSION = "goal_terminal_evidence/v1"
GOAL_SUPPORT_RESULT_SCHEMA_VERSION = "goal_support_result/v1"
GOAL_SUPPORT_IMPLEMENTATION_VERSION = (
    f"goal_support/v1:g-{gate_stack_implementation_digest()[:16]}:"
    f"c-{evaluator_digest()[:16]}"
)

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
_PRODUCTION_EXACT_GATE_IDS: frozenset[str] = frozenset(
    {"G0", "G1", "G2", "G3", "G4", "G8"}
)

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

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """Copy through validation so mode/authority cannot bypass validators."""
        payload = self.model_dump(mode="python", round_trip=True)
        payload.update(dict(update or {}))
        return type(self).model_validate(payload)


def redact_bounded_string(text: str) -> str:
    """Strip secret-shaped substrings and bound length for terminal evidence."""
    if not text:
        return text
    if len(text) == 64 and all(char in "0123456789abcdef" for char in text):
        return text
    if text.startswith("[REDACTED:") and text.endswith("]"):
        return text
    scrubbed = text
    for pattern in _SECRET_PATTERNS:
        scrubbed = pattern.sub("[REDACTED]", scrubbed)
    if len(text) > _MAX_BOUNDED_FIELD_CHARS:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        scrubbed = f"[REDACTED:{digest}]"
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
        if any(not item.implementation_hash for item in self.required_evaluators):
            raise ValueError("required evaluators must bind an implementation_hash")

        metric_ids = [item.metric_id for item in self.metric_identities]
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("metric_identities metric_id values must be unique")
        if any(not item.implementation_hash for item in self.metric_identities):
            raise ValueError("metric identities must bind an implementation_hash")

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

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        copied = super().model_copy(update=update, deep=deep)
        if copied.mode != self.mode or copied.authority_tier != self.authority_tier:
            raise ValueError("copy cannot change goal-verifier mode or authority")
        return copied


def validate_profile_against_constraint_set(
    profile: GoalVerifierProfileV1,
    constraint_set: CompiledGoalConstraintSetV1,
) -> None:
    """Fail closed when profile identities are stale or mode/partition laws break."""
    if constraint_set.schema_version != COMPILED_GOAL_CONSTRAINT_SET_SCHEMA_VERSION:
        raise ValueError("unsupported CompiledGoalConstraintSetV1 version")
    if constraint_set.compiler_version != COMPILER_VERSION:
        raise ValueError("constraint compiler identity is stale")
    if not constraint_set.digest or constraint_set.compute_digest() != constraint_set.digest:
        raise ValueError("constraint set digest is stale or its payload was mutated")
    if not profile.digest or profile.compute_digest() != profile.digest:
        raise ValueError("goal-verifier profile digest is stale or tampered")
    if profile.implementation_version != GOAL_SUPPORT_IMPLEMENTATION_VERSION:
        raise ValueError("goal-verifier implementation identity is stale")

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
        unsafe_gates = sorted(
            set(profile.required_gates) - _PRODUCTION_EXACT_GATE_IDS
        )
        if unsafe_gates:
            raise ValueError(
                "production_exact requires context-free exact gates only: "
                f"{unsafe_gates}"
            )
        missing_identity = [
            name
            for name in (
                "contract_version",
                "grammar_sha256",
                "tokenizer_id",
                "canonicalizer_id",
                "artifact_schema",
                "artifact_sha256",
            )
            if getattr(profile.pack_identity, name) is None
        ]
        if missing_identity:
            raise ValueError(
                "production_exact requires complete pack identity fields: "
                f"{missing_identity}"
            )
        if profile.required_evaluators or profile.metric_identities:
            raise ValueError(
                "production_exact has no certified exact evaluator registry; "
                "required evaluators and metrics are diagnostic-only"
            )
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
            if not constraint.may_prune:
                raise ValueError(
                    f"production_exact requires explicit may_prune authority for {constraint_id!r}"
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


def exact_atom_allowed_for_obstruction(atom: GoalFailureAtomV1) -> bool:
    """Public guard for exact mandatory atoms eligible in obstruction cores."""
    return _exact_atom_allowed(
        atom, implementation_version=GOAL_SUPPORT_IMPLEMENTATION_VERSION
    )


def exact_mandatory_failure_atom_ids(
    record: GoalTerminalEvidenceV1,
) -> tuple[str, ...]:
    """Exact mandatory failure atom ids eligible for obstruction cores."""
    if record.mandatory_unknown_atoms:
        return ()
    return tuple(
        sorted(
            atom.atom_id
            for atom in record.mandatory_failure_atoms
            if exact_atom_allowed_for_obstruction(atom)
        )
    )


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

        has_reject = (
            self.structural_status == "REJECT"
            or bool(self.mandatory_failure_atoms)
            or any(item.status == "FAIL" for item in self.constraint_evaluations)
            or any(item.status == "REJECT" for item in self.required_gate_results)
            or any(item.status == "REJECT" for item in self.required_evaluator_results)
        )
        has_unknown = (
            self.structural_status == "UNAVAILABLE"
            or bool(self.mandatory_unknown_atoms)
            or any(
                item.status in {"UNKNOWN", "SKIPPED", "NOT_APPLICABLE"}
                for item in self.constraint_evaluations
            )
            or any(item.status == "UNAVAILABLE" for item in self.required_gate_results)
            or any(
                item.status == "UNAVAILABLE" for item in self.required_evaluator_results
            )
        )
        expected = "REJECT" if has_reject else "UNAVAILABLE" if has_unknown else "ACCEPT"
        if self.overall_status != expected:
            raise ValueError(
                f"overall_status {self.overall_status!r} conflicts with terminal evidence; "
                f"expected {expected!r}"
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


# --------------------------------------------------------------------------- #
# Goal-aware support provider (PGS-C01 / SLM-499)
# --------------------------------------------------------------------------- #


class GoalVerifierFactory(Protocol):
    """Deterministic factory: every ``check``/``replay`` gets a fresh verifier."""

    def __call__(self) -> Verifier: ...


def goal_support_backend_version(profile: GoalVerifierProfileV1) -> str:
    """Profile-sensitive backend id for exact-closure cache keys."""
    digest = profile.digest or profile.compute_digest()
    return f"goal-support/{GOAL_SUPPORT_IMPLEMENTATION_VERSION}/{digest}"


def exact_goal_closure_authority_table() -> dict[str, Any]:
    """Which profile modes/tiers ``exact_goal_closure`` accepts or rejects."""
    return {
        "accepted": {
            "mode": "production_exact",
            "authority_tiers": sorted(_HARD_AUTHORITIES),
        },
        "rejected_modes": sorted(
            mode for mode in _MODE_ALLOWED_AUTHORITY if mode != "production_exact"
        ),
        "rejected_authority_tiers": sorted(
            tier for tier in (
                *_EVALUATION_AUTHORITIES,
                *_ADVISORY_AUTHORITIES,
            )
        ),
    }


class GoalSupportResultV1(_StrictModel):
    """Digest-bound sidecar keyed by the canonical base certificate digest."""

    schema_version: str = Field(
        default=GOAL_SUPPORT_RESULT_SCHEMA_VERSION,
        pattern=r"^goal_support_result/v1$",
    )
    base_certificate_digest: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    base_verdict: str = Field(min_length=1)
    profile_digest: str = Field(min_length=1, pattern=r"^[0-9a-f]{64}$")
    authority_tier: GoalProfileAuthorityTier
    terminal_evidence_digests: tuple[str, ...] = ()
    witness_terminal_evidence_digest: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$|^$"
    )
    obstruction_core_digest: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$|^$"
    )
    digest: str = Field(default="", pattern=r"^[0-9a-f]{64}$|^$")

    @model_validator(mode="after")
    def _check_integrity(self) -> "GoalSupportResultV1":
        if self.base_verdict not in {item.value for item in SupportVerdict}:
            raise ValueError(f"unknown base_verdict {self.base_verdict!r}")
        if self.digest:
            expected = self.compute_digest()
            if self.digest != expected:
                raise ValueError("digest does not match canonical payload")
        return self

    def _canonical_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("digest", None)
        payload["terminal_evidence_digests"] = sorted(payload["terminal_evidence_digests"])
        return payload

    def compute_digest(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self._canonical_payload())
        ).hexdigest()

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        if not include_digest:
            payload.pop("digest", None)
        elif not payload.get("digest"):
            payload["digest"] = self.compute_digest()
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GoalSupportResultV1":
        if str(value.get("schema_version")) != GOAL_SUPPORT_RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported GoalSupportResultV1 version")
        return cls.model_validate(value)


def _terminal_records(
    verifier: Verifier, *, profile_digest: str
) -> tuple[GoalTerminalEvidenceV1, ...]:
    records = getattr(verifier, "terminal_evidence", None)
    if records is None:
        trace = getattr(verifier, "last_trace", None)
        records = getattr(trace, "records", None) if trace is not None else None
    if records is None:
        raise ValueError("goal verifier must expose query-local terminal evidence")
    by_key: dict[tuple[str, str], GoalTerminalEvidenceV1] = {}
    for raw in records:
        if not isinstance(raw, GoalTerminalEvidenceV1):
            raise ValueError("goal verifier emitted invalid terminal evidence")
        record = GoalTerminalEvidenceV1.from_dict(raw.to_dict())
        if record.profile_digest != profile_digest:
            raise ValueError("terminal evidence profile digest mismatch")
        key = (record.profile_digest, record.canonical_program_digest)
        previous = by_key.get(key)
        if previous is not None:
            def semantics(item: GoalTerminalEvidenceV1) -> dict[str, Any]:
                payload = item.to_dict()
                payload.pop("program_digest", None)
                payload.pop("evidence_digest", None)
                return payload

            if semantics(previous) != semantics(record):
                raise ValueError("conflicting duplicate terminal evidence")
            if record.compute_digest() >= previous.compute_digest():
                continue
        by_key[key] = record
    return tuple(by_key[key] for key in sorted(by_key))


def _record_digest(record: GoalTerminalEvidenceV1) -> str:
    return record.evidence_digest or record.compute_digest()


def _sum_counters(*values: SearchCounters) -> SearchCounters:
    return SearchCounters(
        nodes=sum(item.nodes for item in values),
        tokens=sum(item.tokens for item in values),
        depth=max((item.depth for item in values), default=0),
        backtracks=sum(item.backtracks for item in values),
        verifier_calls=sum(item.verifier_calls for item in values),
    )


def _build_goal_support_sidecar(
    certificate: SupportCertificate,
    *,
    profile: GoalVerifierProfileV1,
    records: tuple[GoalTerminalEvidenceV1, ...],
    state: FiniteDomainState,
    replay_ok: bool,
) -> GoalSupportResultV1:
    profile_digest = profile.digest or profile.compute_digest()
    if certificate.verdict is SupportVerdict.SUPPORTED:
        accepted = tuple(
            item
            for item in records
            if item.overall_status == "ACCEPT"
            and item.program_digest == certificate.witness_digest
        )
        if len(accepted) != 1:
            raise ValueError(
                "SUPPORTED certificate must bind one matching terminal evidence"
            )
        witness_digest = _record_digest(accepted[0])
    else:
        witness_digest = None
    if certificate.verdict is SupportVerdict.UNKNOWN and any(
        item.overall_status == "ACCEPT" for item in records
    ):
        raise ValueError("UNKNOWN goal result cannot contain accepted terminal evidence")
    if certificate.verdict is SupportVerdict.UNSUPPORTED and any(
        item.overall_status != "REJECT" for item in records
    ):
        raise ValueError("UNSUPPORTED goal result can contain only rejected evidence")
    obstruction_core_digest: str | None = None
    from slm_training.dsl.solver import goal_support_obstruction as _obstruction

    core = _obstruction.compute_domain_obstruction_core(
        certificate=certificate,
        terminal_records=records,
        profile=profile,
        state=state,
        replay_ok=replay_ok,
    )
    if core is not None:
        obstruction_core_digest = core.core_digest or core.compute_digest()
    payload = {
        "base_certificate_digest": certificate.digest,
        "base_verdict": certificate.verdict.value,
        "profile_digest": profile_digest,
        "authority_tier": profile.authority_tier,
        "terminal_evidence_digests": tuple(_record_digest(item) for item in records),
        "witness_terminal_evidence_digest": witness_digest,
        "obstruction_core_digest": obstruction_core_digest,
    }
    sidecar = GoalSupportResultV1(**payload)
    return GoalSupportResultV1.from_dict(sidecar.to_dict(include_digest=True))


class GoalSupportProvider:
    """Goal-aware :class:`SupportProvider` over the enumerative oracle."""

    def __init__(
        self,
        expander: ProblemExpander,
        profile: GoalVerifierProfileV1,
        verifier_factory: GoalVerifierFactory,
        *,
        constraint_set: CompiledGoalConstraintSetV1,
    ) -> None:
        validate_profile_against_constraint_set(profile, constraint_set)
        self._expander = expander
        self._profile = profile
        self._constraint_set = constraint_set
        self._verifier_factory = verifier_factory
        self._profile_digest = profile.digest or profile.compute_digest()
        self._verifier_instances: list[Verifier] = []
        self._sidecars: dict[str, GoalSupportResultV1] = {}
        self._obstruction_cores: dict[str, Any] = {}

    @property
    def expander(self) -> ProblemExpander:
        return self._expander

    @property
    def profile(self) -> GoalVerifierProfileV1:
        return self._profile

    @property
    def profile_digest(self) -> str:
        return self._profile_digest

    @property
    def backend_version(self) -> str:
        return goal_support_backend_version(self._profile)

    def _fresh_verifier(self) -> Verifier:
        verifier = self._verifier_factory()
        if any(verifier is previous for previous in self._verifier_instances):
            raise ValueError("goal verifier factory reused mutable query state")
        self._verifier_instances.append(verifier)
        return verifier

    def validate_identities(self, state: FiniteDomainState) -> None:
        if (
            state.problem_id != self._expander.problem_id
            or state.pack_id != self._expander.pack_id
            or state.constraint_version != self._expander.constraint_version
            or state.bounds != self._expander.bounds
        ):
            raise ValueError("goal support state identity does not match expander")

    def check_with_sidecar(
        self, state: FiniteDomainState, query: SupportQuery
    ) -> tuple[SupportResult, GoalSupportResultV1]:
        self.validate_identities(state)
        verifier = self._fresh_verifier()
        result = EnumerativeSupportOracle(self._expander, verifier).check(state, query)
        replay = replay_support_certificate(
            result.certificate,
            state=state,
            expander=self._expander,
            verifier=self._fresh_verifier(),
        )
        result = replace(
            result,
            counters=_sum_counters(result.counters, replay.counters),
        )
        records = _terminal_records(verifier, profile_digest=self._profile_digest)
        sidecar = _build_goal_support_sidecar(
            result.certificate,
            profile=self._profile,
            records=records,
            state=state,
            replay_ok=replay.ok,
        )
        if sidecar.base_certificate_digest != result.certificate.digest:
            raise ValueError("sidecar base_certificate_digest drift")
        if sidecar.base_verdict != result.verdict.value:
            raise ValueError("sidecar base_verdict drift")
        previous = self._sidecars.get(result.certificate.digest)
        if previous is not None and previous.digest != sidecar.digest:
            raise ValueError("same base certificate produced conflicting goal sidecars")
        self._sidecars[result.certificate.digest] = sidecar
        if sidecar.obstruction_core_digest is not None:
            from slm_training.dsl.solver import goal_support_obstruction as _obstruction

            core = _obstruction.compute_domain_obstruction_core(
                certificate=result.certificate,
                terminal_records=records,
                profile=self._profile,
                state=state,
                replay_ok=replay.ok,
            )
            if core is None or (core.core_digest or core.compute_digest()) != (
                sidecar.obstruction_core_digest
            ):
                raise ValueError("obstruction core changed while binding sidecar")
            self._obstruction_cores[result.certificate.digest] = core
        return result, sidecar

    def check(self, state: FiniteDomainState, query: SupportQuery) -> SupportResult:
        result, _sidecar = self.check_with_sidecar(state, query)
        return result

    def replay(
        self, certificate: SupportCertificate, *, state: FiniteDomainState
    ) -> ReplayResult:
        try:
            self.validate_identities(state)
        except ValueError as exc:
            return ReplayResult(False, certificate.verdict, (str(exc),))
        sidecar = self._sidecars.get(certificate.digest)
        if sidecar is None:
            return ReplayResult(
                False, certificate.verdict, ("goal sidecar missing",)
            )
        return replay_goal_support_result(
            certificate, sidecar, state=state, provider=self
        )

    def goal_result(self, base_certificate_digest: str) -> GoalSupportResultV1:
        try:
            return self._sidecars[base_certificate_digest]
        except KeyError as exc:
            raise LookupError("no goal sidecar for base certificate") from exc

    def obstruction_core(self, base_certificate_digest: str) -> Any | None:
        return self._obstruction_cores.get(base_certificate_digest)


def replay_goal_support_result(
    certificate: SupportCertificate,
    sidecar: GoalSupportResultV1,
    *,
    state: FiniteDomainState,
    provider: GoalSupportProvider,
) -> ReplayResult:
    """Replay base certificate plus digest-bound goal sidecar.

    Replay failures are reported as violations; the certificate verdict is never
    upgraded to ``UNSUPPORTED`` when replay fails.
    """
    violations: list[str] = []

    if sidecar.base_certificate_digest != certificate.digest:
        violations.append("sidecar base_certificate_digest mismatch")
    if sidecar.base_verdict != certificate.verdict.value:
        violations.append("sidecar base_verdict mismatch")
    if sidecar.profile_digest != provider.profile_digest:
        violations.append("stale or cross-profile sidecar profile_digest")
    if sidecar.authority_tier != provider.profile.authority_tier:
        violations.append("sidecar authority_tier mismatch")

    expected_sidecar_digest = sidecar.compute_digest()
    if sidecar.digest and sidecar.digest != expected_sidecar_digest:
        violations.append("tampered sidecar digest")

    try:
        provider.validate_identities(state)
    except ValueError as exc:
        violations.append(str(exc))
        return ReplayResult(
            ok=False,
            verdict=certificate.verdict,
            violations=tuple(violations),
        )

    replay_verifier = provider._fresh_verifier()
    base_replay = replay_support_certificate(
        certificate,
        state=state,
        expander=provider.expander,
        verifier=replay_verifier,
    )
    violations.extend(base_replay.violations)

    try:
        records = _terminal_records(
            replay_verifier, profile_digest=provider.profile_digest
        )
        rederived = _build_goal_support_sidecar(
            certificate,
            profile=provider.profile,
            records=records,
            state=state,
            replay_ok=base_replay.ok,
        )
        if rederived.digest != sidecar.digest:
            violations.append("goal support sidecar changed on replay")
    except ValueError as exc:
        violations.append(f"goal sidecar replay failed: {exc}")

    # Preserve UNKNOWN as honest non-pruning evidence; never fabricate UNSUPPORTED.
    verdict = certificate.verdict
    if verdict is SupportVerdict.UNKNOWN:
        pass
    elif not base_replay.ok and verdict is SupportVerdict.UNSUPPORTED:
        pass

    return ReplayResult(
        ok=not violations,
        verdict=verdict,
        violations=tuple(violations),
        counters=base_replay.counters,
    )


def exact_goal_closure(
    state: FiniteDomainState,
    provider: GoalSupportProvider,
    *,
    query_order: QueryOrder | None = None,
    cache: dict[str, SupportResult] | None = None,
    certificate_store: dict[str, SupportCertificate] | None = None,
    max_queries: int | None = None,
) -> ClosureResult:
    """Authority guard over ``exact_closure`` for compiler/verifier-hard profiles."""
    if not isinstance(provider, GoalSupportProvider):
        raise ValueError("exact_goal_closure requires GoalSupportProvider")
    profile = provider.profile
    if profile.mode != "production_exact":
        raise ValueError(
            f"exact_goal_closure rejects non-production profile mode {profile.mode!r}"
        )
    if profile.authority_tier not in _HARD_AUTHORITIES:
        raise ValueError(
            f"exact_goal_closure rejects non-hard authority tier {profile.authority_tier!r}"
        )
    provider.validate_identities(state)

    kwargs: dict[str, Any] = {}
    if query_order is not None:
        kwargs["query_order"] = query_order
    if cache is not None:
        kwargs["cache"] = cache
    if certificate_store is not None:
        kwargs["certificate_store"] = certificate_store
    if max_queries is not None:
        kwargs["max_queries"] = max_queries
    return exact_closure(state, provider, **kwargs)


from slm_training.dsl.solver import goal_support_obstruction as _obstruction  # noqa: E402
from slm_training.dsl.solver import goal_support_domain_adequacy as _adequacy  # noqa: E402

DOMAIN_OBSTRUCTION_CLAIM = _obstruction.DOMAIN_OBSTRUCTION_CLAIM
DOMAIN_OBSTRUCTION_CORE_SCHEMA_VERSION = _obstruction.DOMAIN_OBSTRUCTION_CORE_SCHEMA_VERSION
MIN_CARDINALITY_EXACT_ATOM_THRESHOLD = _obstruction.MIN_CARDINALITY_EXACT_ATOM_THRESHOLD
OBSTRUCTION_ALGORITHM_ID = _obstruction.OBSTRUCTION_ALGORITHM_ID
OBSTRUCTION_ALGORITHM_VERSION = _obstruction.OBSTRUCTION_ALGORITHM_VERSION
SUBSET_MINIMAL_DELETION_ATOM_THRESHOLD = _obstruction.SUBSET_MINIMAL_DELETION_ATOM_THRESHOLD
DomainObstructionCoreV1 = _obstruction.DomainObstructionCoreV1
ObstructionCoreMode = _obstruction.ObstructionCoreMode
ObstructionCoreStatsV1 = _obstruction.ObstructionCoreStatsV1
TerminalFailureSetV1 = _obstruction.TerminalFailureSetV1
compute_domain_obstruction_core = _obstruction.compute_domain_obstruction_core
obstruction_core_algorithm_table = _obstruction.obstruction_core_algorithm_table
obstruction_core_emission_allowed = _obstruction.obstruction_core_emission_allowed
replay_obstruction_core = _obstruction.replay_obstruction_core
terminal_failure_set_from_evidence = _obstruction.terminal_failure_set_from_evidence
validate_domain_obstruction_core = _obstruction.validate_domain_obstruction_core

GOAL_ACTION_EVIDENCE_SCHEMA_VERSION = _adequacy.GOAL_ACTION_EVIDENCE_SCHEMA_VERSION
GOAL_DOMAIN_ADEQUACY_REPORT_SCHEMA_VERSION = (
    _adequacy.GOAL_DOMAIN_ADEQUACY_REPORT_SCHEMA_VERSION
)
DOMAIN_ADEQUACY_CLASSIFIER_ID = _adequacy.DOMAIN_ADEQUACY_CLASSIFIER_ID
DOMAIN_ADEQUACY_CLASSIFIER_VERSION = _adequacy.DOMAIN_ADEQUACY_CLASSIFIER_VERSION
GoalActionEvidenceV1 = _adequacy.GoalActionEvidenceV1
GoalActionWorkCountersV1 = _adequacy.GoalActionWorkCountersV1
GoalDomainActionPartitionsV1 = _adequacy.GoalDomainActionPartitionsV1
GoalDomainAdequacyReportV1 = _adequacy.GoalDomainAdequacyReportV1
GoalDomainAdequacyStatsV1 = _adequacy.GoalDomainAdequacyStatsV1
AggregateObstructionSummaryV1 = _adequacy.AggregateObstructionSummaryV1
DomainAdequacyClassification = _adequacy.DomainAdequacyClassification
GoalActionPartition = _adequacy.GoalActionPartition
action_id_from_value = _adequacy.action_id_from_value
legal_set_fingerprint = _adequacy.legal_set_fingerprint
bounds_digest_from_state = _adequacy.bounds_digest_from_state
domain_adequacy_cap_policy_table = _adequacy.domain_adequacy_cap_policy_table
domain_adequacy_classification_table = _adequacy.domain_adequacy_classification_table
analyze_goal_domain = _adequacy.analyze_goal_domain

__all__ = [
    "GOAL_VERIFIER_PROFILE_SCHEMA_VERSION",
    "GOAL_TERMINAL_EVIDENCE_SCHEMA_VERSION",
    "GOAL_SUPPORT_RESULT_SCHEMA_VERSION",
    "GOAL_SUPPORT_IMPLEMENTATION_VERSION",
    "GOAL_ACTION_EVIDENCE_SCHEMA_VERSION",
    "GOAL_DOMAIN_ADEQUACY_REPORT_SCHEMA_VERSION",
    "DOMAIN_ADEQUACY_CLASSIFIER_ID",
    "DOMAIN_ADEQUACY_CLASSIFIER_VERSION",
    "DOMAIN_OBSTRUCTION_CORE_SCHEMA_VERSION",
    "DOMAIN_OBSTRUCTION_CLAIM",
    "OBSTRUCTION_ALGORITHM_ID",
    "OBSTRUCTION_ALGORITHM_VERSION",
    "MIN_CARDINALITY_EXACT_ATOM_THRESHOLD",
    "SUBSET_MINIMAL_DELETION_ATOM_THRESHOLD",
    "GoalVerifierFactory",
    "GoalVerifierMode",
    "GoalProfileAuthorityTier",
    "GoalTerminalStatus",
    "ObstructionCoreMode",
    "MetricIdentityV1",
    "EvaluatorIdentityV1",
    "GoalVerifierProfileV1",
    "GoalGateResultV1",
    "GoalEvaluatorResultV1",
    "GoalFailureAtomV1",
    "GoalUnknownAtomV1",
    "GoalTerminalEvidenceV1",
    "GoalSupportResultV1",
    "GoalSupportProvider",
    "GoalActionEvidenceV1",
    "GoalActionWorkCountersV1",
    "GoalDomainActionPartitionsV1",
    "GoalDomainAdequacyReportV1",
    "GoalDomainAdequacyStatsV1",
    "AggregateObstructionSummaryV1",
    "DomainAdequacyClassification",
    "GoalActionPartition",
    "DomainObstructionCoreV1",
    "TerminalFailureSetV1",
    "ObstructionCoreStatsV1",
    "profile_mode_authority_table",
    "terminal_status_decision_table",
    "exact_goal_closure_authority_table",
    "obstruction_core_algorithm_table",
    "profile_digest_inputs",
    "goal_support_backend_version",
    "compute_pack_identity_digest",
    "validate_profile_against_constraint_set",
    "redact_bounded_string",
    "redact_terminal_evidence_dict",
    "exact_atom_allowed_for_obstruction",
    "exact_mandatory_failure_atom_ids",
    "terminal_failure_set_from_evidence",
    "obstruction_core_emission_allowed",
    "compute_domain_obstruction_core",
    "validate_domain_obstruction_core",
    "replay_obstruction_core",
    "action_id_from_value",
    "legal_set_fingerprint",
    "bounds_digest_from_state",
    "domain_adequacy_cap_policy_table",
    "domain_adequacy_classification_table",
    "analyze_goal_domain",
    "replay_goal_support_result",
    "exact_goal_closure",
]
