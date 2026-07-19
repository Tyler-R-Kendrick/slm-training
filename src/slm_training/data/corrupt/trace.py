"""Versioned corruption trace schema for semantic severity curriculum.

SLM-120 EFS3-02: a corruption trace records the exact provenance of one
corrupted training example so that severity (S0–S4) and semantic class can be
accounted for independently of the model representation or policy that produced
it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SeverityLevel(str, Enum):
    """Representation-independent corruption severity."""

    S0_CLEAN = "S0_clean"
    S1_NEAR_SOLVED_1 = "S1_near_solved_1"
    S2_NEAR_SOLVED_2 = "S2_near_solved_2"
    S3_MEDIUM = "S3_medium"
    S4_HEAVY = "S4_heavy"


class SemanticClass(str, Enum):
    """Semantic class of a single corruption operation."""

    STRUCTURE = "structure"
    COMPONENT = "component"
    BINDING = "binding"
    PLACEHOLDER = "placeholder"
    SCHEMA_VALUE = "schema_value"
    INVENTORY = "inventory"
    ORDERING = "ordering"
    SURFACE_ONLY = "surface_only"


@dataclass(frozen=True)
class CorruptionOperation:
    """One applied corruption operation."""

    operator: str
    operator_family: str
    semantic_class: SemanticClass
    ast_path: tuple[str | int, ...] = ()
    source_span: tuple[int, int] = (0, 0)
    depends_on: tuple[str, ...] = ()
    surface_only: bool = False
    equivalent_rewrite: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["semantic_class"] = self.semantic_class.value
        return data


@dataclass(frozen=True)
class CorruptionTraceV2:
    """Immutable provenance record for a corrupted training example.

    A trace is representation-independent: it can be produced from grammar
    diffusion, tree-edit diffusion, token-level diffusion, or the formal
    corruption oracle. It must be serializable and hash-stable so that curricula
    can be reproduced and audited.
    """

    trace_schema_version: str = "corruption_trace/v2"
    source_program_hash: str = ""
    prompt_hash: str = ""
    contract_hash: str = ""
    representation: str = "unknown"
    model_family: str = "unknown"
    severity: SeverityLevel = SeverityLevel.S0_CLEAN
    operations: tuple[CorruptionOperation, ...] = ()
    rng_seed: int | None = None
    policy_version: str = ""
    corrupted_valid: bool = False
    repairable: bool | None = None
    inverse_target_hash: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_schema_version": self.trace_schema_version,
            "source_program_hash": self.source_program_hash,
            "prompt_hash": self.prompt_hash,
            "contract_hash": self.contract_hash,
            "representation": self.representation,
            "model_family": self.model_family,
            "severity": self.severity.value,
            "operations": [op.to_dict() for op in self.operations],
            "semantic_operation_count": self.semantic_operation_count,
            "rng_seed": self.rng_seed,
            "policy_version": self.policy_version,
            "corrupted_valid": self.corrupted_valid,
            "repairable": self.repairable,
            "inverse_target_hash": self.inverse_target_hash,
            "meta": dict(self.meta),
        }

    @property
    def semantic_operation_count(self) -> int:
        """Count operations that change semantics, excluding surface-only."""
        return sum(
            1
            for op in self.operations
            if not op.surface_only and not op.equivalent_rewrite
        )

    def validate(self) -> list[str]:
        """Fail-closed validation for a trace."""
        errors: list[str] = []
        if not self.trace_schema_version:
            errors.append("trace_schema_version is required")
        if not self.source_program_hash:
            errors.append("source_program_hash is required")
        if self.severity == SeverityLevel.S0_CLEAN and self.operations:
            errors.append("S0_clean trace must not contain operations")
        if self.severity != SeverityLevel.S0_CLEAN and not self.operations:
            errors.append("non-clean trace must contain at least one operation")
        for op in self.operations:
            if op.equivalent_rewrite and self.severity != SeverityLevel.S0_CLEAN:
                errors.append(
                    "equivalent rewrite must be labeled S0_clean, not a corruption"
                )
        return errors
