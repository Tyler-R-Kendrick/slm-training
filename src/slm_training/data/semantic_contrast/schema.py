"""Versioned schema for the semantic-contrast corpus (SPV2-01)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from slm_training.dsl.schema import ExampleRecord


class ContrastFamily(str, Enum):
    """High-level corruption taxonomy families."""

    CONTENT = "content"
    TOPOLOGY = "topology"
    BINDING = "binding"
    CONTRACT = "contract"
    POSITIVE = "positive"
    UNKNOWN = "unknown"


class ContrastSeverity(str, Enum):
    """Semantic impact severity of a contrast transformation."""

    BENIGN = "benign"
    MODERATE = "moderate"
    SEVERE = "severe"


class ContrastRole(str, Enum):
    """Role of one side of a contrast record."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass(frozen=True)
class SemanticContrastRecord:
    """One side of a contrast pair, typed and versioned."""

    record: ExampleRecord
    role: ContrastRole
    family: ContrastFamily
    transform_id: str
    transform_description: str
    severity: ContrastSeverity
    source_program_id: str
    source_plan: dict[str, Any]
    verifier_ok: bool
    verifier_tier: str | None
    meaningful_report: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "role": self.role.value,
            "family": self.family.value,
            "transform_id": self.transform_id,
            "transform_description": self.transform_description,
            "severity": self.severity.value,
            "source_program_id": self.source_program_id,
            "source_plan": self.source_plan,
            "verifier_ok": self.verifier_ok,
            "verifier_tier": self.verifier_tier,
            "meaningful_report": self.meaningful_report,
            "meta": self.meta,
        }


@dataclass(frozen=True)
class ContrastPair:
    """A positive/negative pair derived from one source ProgramSpec."""

    pair_id: str
    positive: SemanticContrastRecord
    negative: SemanticContrastRecord
    family: ContrastFamily
    transform_id: str
    source_program_id: str
    admitted: bool
    admission_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "positive": self.positive.to_dict(),
            "negative": self.negative.to_dict(),
            "family": self.family.value,
            "transform_id": self.transform_id,
            "source_program_id": self.source_program_id,
            "admitted": self.admitted,
            "admission_reason": self.admission_reason,
        }


@dataclass(frozen=True)
class ContrastList:
    """A list-style contrast sample (one positive + N negatives)."""

    list_id: str
    positive: SemanticContrastRecord
    negatives: tuple[SemanticContrastRecord, ...]
    source_program_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "list_id": self.list_id,
            "positive": self.positive.to_dict(),
            "negatives": [n.to_dict() for n in self.negatives],
            "source_program_id": self.source_program_id,
        }


@dataclass(frozen=True)
class FamilyMetrics:
    """Baseline scoreboard for one corruption family."""

    family: str
    n_total: int
    n_admitted: int
    verifier_pass_rate: float
    meaningful_pass_rate: float
    false_negative_rate: float
    mean_reason_count: float
    top_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "n_total": self.n_total,
            "n_admitted": self.n_admitted,
            "verifier_pass_rate": self.verifier_pass_rate,
            "meaningful_pass_rate": self.meaningful_pass_rate,
            "false_negative_rate": self.false_negative_rate,
            "mean_reason_count": self.mean_reason_count,
            "top_reasons": list(self.top_reasons),
        }


CorpusSplit = Literal["train", "held_out", "ood"]


__all__ = [
    "ContrastFamily",
    "ContrastSeverity",
    "ContrastRole",
    "SemanticContrastRecord",
    "ContrastPair",
    "ContrastList",
    "FamilyMetrics",
    "CorpusSplit",
]
