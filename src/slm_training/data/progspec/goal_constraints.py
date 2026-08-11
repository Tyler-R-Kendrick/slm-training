"""Compiled goal-constraint contract (PGS-A01 / SLM-494).

``PromptSemanticRequirementsV1`` and ``VerifiedSynthesisProblemV1`` carry
request-side, advisory evidence. ``SemanticPlanV1`` carries program-side plan
structure. This module is the compiled, pack-neutral *goal constraint* output
contract: deterministic constraints a candidate must satisfy, each labeled
with authority, completeness, and optional prune permission under the
non-escalation law.

Design invariants (see SLM-494 acceptance criteria):

* ``may_prune=True`` only when authority is ``compiler-hard`` or
  ``verifier-hard``, completeness is ``EXACT``, and ``source_kind`` is an
  independently exact source (structured ``generation_request``,
  ``pack_contract``, or ``verification_requirement`` — fail-closed).
* No constructor, loader, migration, copy, merge, or projection may increase
  authority, completeness, or pruning power.
* Identical constraints merge to the weaker authority/completeness; same-id
  conflicts fail closed.
* ``CompiledGoalConstraintSetV1.digest`` is deterministic, excludes itself,
  and is verified on load.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "GOAL_CONSTRAINT_SCHEMA_VERSION",
    "COMPILED_GOAL_CONSTRAINT_SET_SCHEMA_VERSION",
    "GOAL_CONSTRAINT_EVALUATION_SCHEMA_VERSION",
    "GoalConstraintKind",
    "GoalConstraintAuthority",
    "GoalConstraintCompleteness",
    "GoalConstraintStatus",
    "GoalConstraintSourceKind",
    "GoalConstraintV1",
    "GoalConstraintAmbiguityGroup",
    "CompiledGoalConstraintSetV1",
    "GoalConstraintEvidenceV1",
    "GoalConstraintEvaluationV1",
    "INDEPENDENTLY_EXACT_SOURCE_KINDS",
    "combine_authority",
    "combine_completeness",
    "combine_may_prune",
    "merge_goal_constraints",
    "authority_matrix",
    "canonical_json_bytes",
    "validate_json_parameters",
]

GOAL_CONSTRAINT_SCHEMA_VERSION = "goal_constraint/v1"
COMPILED_GOAL_CONSTRAINT_SET_SCHEMA_VERSION = "compiled_goal_constraint_set/v1"
GOAL_CONSTRAINT_EVALUATION_SCHEMA_VERSION = "goal_constraint_evaluation/v1"

GoalConstraintKind = Literal[
    "slot_present",
    "slot_inventory_exact",
    "runtime_symbol_accounted",
    "output_kind_equals",
    "output_category_equals",
    "component_count_at_least",
    "component_absent",
    "binding_resolved",
    "binding_role_compatible",
    "topology_relation_present",
    "required_gate_passes",
]

GoalConstraintAuthority = Literal[
    "compiler-hard",
    "verifier-hard",
    "advisory-learned",
    "oracle-diagnostic",
    "evaluation-only",
]

GoalConstraintCompleteness = Literal["EXACT", "SOUND_OVERAPPROX", "HEURISTIC", "UNKNOWN"]

GoalConstraintStatus = Literal["PASS", "FAIL", "UNKNOWN", "NOT_APPLICABLE", "SKIPPED"]

GoalConstraintSourceKind = Literal[
    "generation_request",
    "pack_contract",
    "verification_requirement",
    "prompt_requirement",
    "evaluation_fixture",
    "oracle_diagnostic",
]

INDEPENDENTLY_EXACT_SOURCE_KINDS: frozenset[str] = frozenset(
    {"generation_request", "pack_contract", "verification_requirement"}
)

_AUTHORITY_RANK: dict[str, int] = {
    "evaluation-only": 0,
    "oracle-diagnostic": 1,
    "advisory-learned": 2,
    "verifier-hard": 3,
    "compiler-hard": 4,
}

_COMPLETENESS_RANK: dict[str, int] = {
    "UNKNOWN": 0,
    "HEURISTIC": 1,
    "SOUND_OVERAPPROX": 2,
    "EXACT": 3,
}

_HARD_AUTHORITIES: frozenset[str] = frozenset({"compiler-hard", "verifier-hard"})
_ADVISORY_AUTHORITIES: frozenset[str] = frozenset({"advisory-learned", "oracle-diagnostic"})
_EVALUATION_AUTHORITIES: frozenset[str] = frozenset({"evaluation-only"})


def combine_authority(a: str, b: str) -> str:
    """Return the weaker (lower-ranked) of two goal-constraint authorities."""
    return a if _AUTHORITY_RANK[a] <= _AUTHORITY_RANK[b] else b


def combine_completeness(a: str, b: str) -> str:
    """Return the weaker (lower-ranked) of two completeness labels."""
    return a if _COMPLETENESS_RANK[a] <= _COMPLETENESS_RANK[b] else b


def combine_may_prune(a: bool, b: bool) -> bool:
    """Merge prune permission without escalation — both sources must allow prune."""
    return bool(a and b)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def validate_json_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Reject non-JSON-serializable values and non-finite floats."""
    try:
        encoded = json.dumps(parameters, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"parameters must be JSON-compatible: {exc}") from exc
    loaded = json.loads(encoded)
    if not isinstance(loaded, dict):
        raise ValueError("parameters must be a JSON object")
    _reject_nonfinite(loaded)
    return loaded


def _reject_nonfinite(value: Any, *, path: str = "parameters") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains non-finite number")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, path=f"{path}[{index}]")


def _may_prune_allowed(
    *,
    authority_tier: str,
    completeness: str,
    source_kind: str,
    may_prune: bool,
) -> bool:
    if not may_prune:
        return True
    if authority_tier not in _HARD_AUTHORITIES:
        return False
    if completeness != "EXACT":
        return False
    return source_kind in INDEPENDENTLY_EXACT_SOURCE_KINDS


def authority_matrix() -> dict[str, Any]:
    """Machine-readable authority/completeness/prune matrix for downstream tooling."""
    return {
        "authority_tiers": dict(_AUTHORITY_RANK),
        "completeness_tiers": dict(_COMPLETENESS_RANK),
        "partitions": {
            "hard_constraint_ids": sorted(_HARD_AUTHORITIES),
            "advisory_constraint_ids": sorted(_ADVISORY_AUTHORITIES),
            "evaluation_constraint_ids": sorted(_EVALUATION_AUTHORITIES),
        },
        "independently_exact_source_kinds": sorted(INDEPENDENTLY_EXACT_SOURCE_KINDS),
        "may_prune_requires": {
            "authority_tier": sorted(_HARD_AUTHORITIES),
            "completeness": ["EXACT"],
            "source_kind": sorted(INDEPENDENTLY_EXACT_SOURCE_KINDS),
        },
    }


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        """Copy through validation; Pydantic's unchecked update can launder authority."""
        payload = self.model_dump(mode="python", round_trip=True)
        payload.update(dict(update or {}))
        return type(self).model_validate(payload)


class GoalConstraintV1(_StrictModel):
    """One compiled goal constraint with explicit authority and completeness."""

    schema_version: str = Field(
        default=GOAL_CONSTRAINT_SCHEMA_VERSION, pattern=r"^goal_constraint/v1$"
    )
    constraint_id: str = Field(min_length=1)
    kind: GoalConstraintKind
    parameters: dict[str, Any] = Field(default_factory=dict)
    source_kind: GoalConstraintSourceKind
    source_id: str = Field(min_length=1)
    source_digest: str = Field(min_length=1)
    authority_tier: GoalConstraintAuthority
    completeness: GoalConstraintCompleteness
    may_prune: bool = False

    @field_validator("parameters")
    @classmethod
    def _validate_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        return validate_json_parameters(value)

    @model_validator(mode="after")
    def _check_may_prune_law(self) -> "GoalConstraintV1":
        if not _may_prune_allowed(
            authority_tier=self.authority_tier,
            completeness=self.completeness,
            source_kind=self.source_kind,
            may_prune=self.may_prune,
        ):
            raise ValueError(
                "may_prune=True requires compiler-hard or verifier-hard authority, "
                "EXACT completeness, and an independently exact source_kind "
                f"({sorted(INDEPENDENTLY_EXACT_SOURCE_KINDS)})"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        copied = super().model_copy(update=update, deep=deep)
        if _AUTHORITY_RANK[copied.authority_tier] > _AUTHORITY_RANK[self.authority_tier]:
            raise ValueError("copy cannot increase goal-constraint authority")
        if _COMPLETENESS_RANK[copied.completeness] > _COMPLETENESS_RANK[self.completeness]:
            raise ValueError("copy cannot increase goal-constraint completeness")
        if copied.may_prune and not self.may_prune:
            raise ValueError("copy cannot add goal-constraint pruning authority")
        return copied

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GoalConstraintV1":
        if str(value.get("schema_version")) != GOAL_CONSTRAINT_SCHEMA_VERSION:
            raise ValueError("unsupported GoalConstraintV1 version")
        return cls.model_validate(value)

    def merged_with(self, other: "GoalConstraintV1") -> "GoalConstraintV1":
        """Merge two constraints with the same id without escalating authority."""
        if self.constraint_id != other.constraint_id:
            raise ValueError("constraint_id must match to merge two GoalConstraintV1 values")
        identity_fields = (
            "kind",
            "parameters",
            "source_kind",
            "source_id",
            "source_digest",
        )
        for field_name in identity_fields:
            if getattr(self, field_name) != getattr(other, field_name):
                raise ValueError(
                    f"conflicting GoalConstraintV1 values for {self.constraint_id!r} "
                    f"on field {field_name!r}"
                )
        return self.model_copy(
            update={
                "authority_tier": combine_authority(self.authority_tier, other.authority_tier),
                "completeness": combine_completeness(self.completeness, other.completeness),
                "may_prune": combine_may_prune(self.may_prune, other.may_prune),
            }
        )


def merge_goal_constraints(
    left: GoalConstraintV1, right: GoalConstraintV1
) -> GoalConstraintV1:
    """Public alias for :meth:`GoalConstraintV1.merged_with`."""
    return left.merged_with(right)


class GoalConstraintAmbiguityGroup(_StrictModel):
    """Mutually exclusive compiled constraints that must not be silently resolved."""

    group_id: str = Field(min_length=1)
    member_constraint_ids: tuple[str, ...] = Field(min_length=2)
    note: str | None = None


class CompiledGoalConstraintSetV1(_StrictModel):
    """Versioned compiled goal-constraint bundle with deterministic digest."""

    schema_version: str = Field(
        default=COMPILED_GOAL_CONSTRAINT_SET_SCHEMA_VERSION,
        pattern=r"^compiled_goal_constraint_set/v1$",
    )
    problem_digest: str = Field(min_length=1)
    request_digest: str = Field(min_length=1)
    pack_identity_digest: str = Field(min_length=1)
    compiler_version: str = Field(min_length=1)
    constraints: tuple[GoalConstraintV1, ...] = ()
    hard_constraint_ids: tuple[str, ...] = ()
    advisory_constraint_ids: tuple[str, ...] = ()
    evaluation_constraint_ids: tuple[str, ...] = ()
    uncompiled_source_ids: tuple[str, ...] = ()
    ambiguity_groups: tuple[GoalConstraintAmbiguityGroup, ...] = ()
    digest: str = Field(default="", pattern=r"^[0-9a-f]{64}$|^$")

    @model_validator(mode="after")
    def _check_integrity(self) -> "CompiledGoalConstraintSetV1":
        constraint_ids = [constraint.constraint_id for constraint in self.constraints]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("constraint_id values must be unique")

        known_ids = set(constraint_ids)
        authority_by_id = {
            constraint.constraint_id: constraint.authority_tier for constraint in self.constraints
        }

        for partition_name, partition_ids, allowed in (
            ("hard_constraint_ids", self.hard_constraint_ids, _HARD_AUTHORITIES),
            (
                "advisory_constraint_ids",
                self.advisory_constraint_ids,
                _ADVISORY_AUTHORITIES,
            ),
            (
                "evaluation_constraint_ids",
                self.evaluation_constraint_ids,
                _EVALUATION_AUTHORITIES,
            ),
        ):
            if len(partition_ids) != len(set(partition_ids)):
                raise ValueError(f"{partition_name} must be unique")
            unknown = set(partition_ids) - known_ids
            if unknown:
                raise ValueError(
                    f"{partition_name} references unknown constraint ids: {sorted(unknown)}"
                )
            for constraint_id in partition_ids:
                tier = authority_by_id[constraint_id]
                if tier not in allowed:
                    raise ValueError(
                        f"{partition_name} contains {constraint_id!r} with authority "
                        f"{tier!r}, expected one of {sorted(allowed)}"
                    )

        partitioned = (
            set(self.hard_constraint_ids)
            | set(self.advisory_constraint_ids)
            | set(self.evaluation_constraint_ids)
        )
        if partitioned != known_ids:
            missing = sorted(known_ids - partitioned)
            extra = sorted(partitioned - known_ids)
            raise ValueError(
                "constraint partitions must cover every constraint exactly once "
                f"(missing={missing}, extra={extra})"
            )

        group_ids = [group.group_id for group in self.ambiguity_groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("ambiguity group_id values must be unique")
        for group in self.ambiguity_groups:
            unknown_members = set(group.member_constraint_ids) - known_ids
            if unknown_members:
                raise ValueError(
                    "ambiguity group references unknown constraint ids: "
                    f"{sorted(unknown_members)}"
                )

        if self.digest:
            expected = self.compute_digest()
            if self.digest != expected:
                raise ValueError("digest does not match canonical payload")

        return self

    def _canonical_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("digest", None)
        payload["constraints"] = sorted(
            payload["constraints"], key=lambda item: item["constraint_id"]
        )
        payload["hard_constraint_ids"] = sorted(payload["hard_constraint_ids"])
        payload["advisory_constraint_ids"] = sorted(payload["advisory_constraint_ids"])
        payload["evaluation_constraint_ids"] = sorted(payload["evaluation_constraint_ids"])
        payload["uncompiled_source_ids"] = sorted(payload["uncompiled_source_ids"])
        payload["ambiguity_groups"] = sorted(
            payload["ambiguity_groups"], key=lambda item: item["group_id"]
        )
        for group in payload["ambiguity_groups"]:
            group["member_constraint_ids"] = sorted(group["member_constraint_ids"])
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
    def from_dict(cls, value: dict[str, Any]) -> "CompiledGoalConstraintSetV1":
        if str(value.get("schema_version")) != COMPILED_GOAL_CONSTRAINT_SET_SCHEMA_VERSION:
            raise ValueError("unsupported CompiledGoalConstraintSetV1 version")
        return cls.model_validate(value)


class GoalConstraintEvidenceV1(_StrictModel):
    """Safe, digest-bound evidence pointer — never raw prompt/program text."""

    location: str = Field(min_length=1)
    digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class GoalConstraintEvaluationV1(_StrictModel):
    """Typed per-constraint evaluation suitable for PGS-A03 / goal-verifier handoff."""

    schema_version: str = Field(
        default=GOAL_CONSTRAINT_EVALUATION_SCHEMA_VERSION,
        pattern=r"^goal_constraint_evaluation/v1$",
    )
    constraint_id: str = Field(min_length=1)
    status: GoalConstraintStatus
    reason_code: str = ""
    evidence: tuple[GoalConstraintEvidenceV1, ...] = ()
    evaluator_id: str = ""
    evaluator_digest: str = Field(default="", pattern=r"^[0-9a-f]{64}$|^$")
    completeness_achieved: GoalConstraintCompleteness = "UNKNOWN"
    authority_tier: GoalConstraintAuthority | None = None
    may_prune: bool | None = None
    detail: str | None = None

    @field_validator("reason_code")
    @classmethod
    def _validate_reason_code(cls, value: str) -> str:
        if not value:
            return value
        if len(value) > 64:
            raise ValueError("reason_code must be at most 64 characters")
        allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_.-")
        if not set(value) <= allowed:
            raise ValueError("reason_code must use lowercase alnum/._- only")
        return value

    @field_validator("detail")
    @classmethod
    def _validate_detail(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if len(value) > 280:
            raise ValueError("detail must be at most 280 characters")
        lowered = value.lower()
        if ":slot_" in lowered or "prompt" in lowered or "root =" in lowered:
            raise ValueError("detail must not contain raw prompt/program content")
        return value

    @model_validator(mode="after")
    def _check_authority_preservation(self) -> "GoalConstraintEvaluationV1":
        if self.authority_tier is None:
            return self
        if self.may_prune is None:
            raise ValueError("may_prune must be set when authority_tier is set")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GoalConstraintEvaluationV1":
        if str(value.get("schema_version")) != GOAL_CONSTRAINT_EVALUATION_SCHEMA_VERSION:
            raise ValueError("unsupported GoalConstraintEvaluationV1 version")
        return cls.model_validate(value)
