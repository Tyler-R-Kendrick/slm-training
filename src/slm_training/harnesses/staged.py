"""Versioned contracts for the staged DSL capability curriculum.

Capability, supervision source, evaluation source, and difficulty are separate
axes.  In particular, distillation and trace collection are processes; neither
is evidence that a model acquired a higher capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from slm_training.harness_core.lineage.records import canonical_json, content_sha
from slm_training.harness_core.record_schema import RUN_CLASSES

SCHEMA_VERSION = "staged_harness_baseline/v1"
UNKNOWN = "UNKNOWN"


class Capability(str, Enum):
    CAP0_GRAMMAR = "CAP0_GRAMMAR"
    CAP1_SEMANTICS = "CAP1_SEMANTICS"
    CAP2_TRANSFORM = "CAP2_TRANSFORM"


class SupervisionSource(str, Enum):
    SUP_COMPILER = "SUP_COMPILER"
    SUP_PARAPHRASE = "SUP_PARAPHRASE"
    SUP_DISTILL = "SUP_DISTILL"


class EvaluationSource(str, Enum):
    EVAL_STATIC = "EVAL_STATIC"
    EVAL_PROPERTY = "EVAL_PROPERTY"
    EVAL_TRACE = "EVAL_TRACE"


class Difficulty(str, Enum):
    """Task complexity, deliberately independent of capability."""

    ATOMIC = "atomic"
    COMPOSITIONAL = "compositional"
    CONTEXTUAL = "contextual"
    ADVERSARIAL = "adversarial"


class EvidenceStatus(str, Enum):
    VERIFIED = "verified"
    UNKNOWN = "unknown"
    INVALID = "invalid"


@dataclass(frozen=True)
class EvidenceIdentityV1:
    """One cited artifact identity; missing evidence stays explicit."""

    path: str
    identity: str
    sha256: str = UNKNOWN
    status: EvidenceStatus = EvidenceStatus.UNKNOWN
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("evidence path is required")
        if not self.identity:
            raise ValueError("evidence identity is required")
        if self.sha256 != UNKNOWN and (
            len(self.sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.sha256)
        ):
            raise ValueError("sha256 must be lowercase hexadecimal or UNKNOWN")
        if self.status is EvidenceStatus.VERIFIED and self.sha256 == UNKNOWN:
            raise ValueError("verified evidence requires a sha256")
        if self.status is not EvidenceStatus.VERIFIED and not self.reason:
            raise ValueError("unknown or invalid evidence requires a reason")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "path": self.path,
            "identity": self.identity,
            "sha256": self.sha256,
            "status": self.status.value,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvidenceIdentityV1:
        return cls(
            path=str(value["path"]),
            identity=str(value["identity"]),
            sha256=str(value.get("sha256", UNKNOWN)),
            status=EvidenceStatus(value.get("status", EvidenceStatus.UNKNOWN.value)),
            reason=None if value.get("reason") is None else str(value["reason"]),
        )


@dataclass(frozen=True)
class StagedHarnessBaselineV1:
    """Pinned identity and vocabulary for staged-harness follow-on work."""

    repo_commit: str
    repo_dirty: bool | None
    quality_matrix_frontier: str
    output_contract_generation: str
    checkpoint_generation: str
    run_class: str
    artifacts: tuple[EvidenceIdentityV1, ...]
    capabilities: tuple[Capability, ...] = tuple(Capability)
    supervision_sources: tuple[SupervisionSource, ...] = tuple(SupervisionSource)
    evaluation_sources: tuple[EvaluationSource, ...] = tuple(EvaluationSource)
    difficulties: tuple[Difficulty, ...] = tuple(Difficulty)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema mismatch: expected {SCHEMA_VERSION!r}, "
                f"got {self.schema_version!r}"
            )
        if self.run_class not in RUN_CLASSES:
            raise ValueError(
                f"run_class must be one of {RUN_CLASSES}, got {self.run_class!r}"
            )
        for name in (
            "repo_commit",
            "quality_matrix_frontier",
            "output_contract_generation",
            "checkpoint_generation",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        paths = [artifact.path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("artifact paths must be unique")

    def blocking_reasons(self) -> tuple[str, ...]:
        """Return reasons this baseline cannot authorize follow-on work."""

        reasons: list[str] = []
        if self.repo_commit == UNKNOWN:
            reasons.append("repo_commit is unknown")
        if self.repo_dirty is not False:
            reasons.append("repo cleanliness is unknown or dirty")
        for name in (
            "quality_matrix_frontier",
            "output_contract_generation",
            "checkpoint_generation",
        ):
            if getattr(self, name) == UNKNOWN:
                reasons.append(f"{name} is unknown")
        for artifact in self.artifacts:
            if artifact.status is not EvidenceStatus.VERIFIED:
                reasons.append(
                    f"{artifact.path} is {artifact.status.value}: {artifact.reason}"
                )
        return tuple(reasons)

    def require_reusable(self) -> None:
        reasons = self.blocking_reasons()
        if reasons:
            raise ValueError("baseline is not reusable: " + "; ".join(reasons))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repo_commit": self.repo_commit,
            "repo_dirty": self.repo_dirty,
            "quality_matrix_frontier": self.quality_matrix_frontier,
            "output_contract_generation": self.output_contract_generation,
            "checkpoint_generation": self.checkpoint_generation,
            "run_class": self.run_class,
            "capabilities": [value.value for value in self.capabilities],
            "supervision_sources": [
                value.value for value in self.supervision_sources
            ],
            "evaluation_sources": [
                value.value for value in self.evaluation_sources
            ],
            "difficulties": [value.value for value in self.difficulties],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StagedHarnessBaselineV1:
        return cls(
            schema_version=str(value["schema_version"]),
            repo_commit=str(value["repo_commit"]),
            repo_dirty=value.get("repo_dirty"),
            quality_matrix_frontier=str(value["quality_matrix_frontier"]),
            output_contract_generation=str(value["output_contract_generation"]),
            checkpoint_generation=str(value["checkpoint_generation"]),
            run_class=str(value["run_class"]),
            capabilities=tuple(Capability(item) for item in value["capabilities"]),
            supervision_sources=tuple(
                SupervisionSource(item) for item in value["supervision_sources"]
            ),
            evaluation_sources=tuple(
                EvaluationSource(item) for item in value["evaluation_sources"]
            ),
            difficulties=tuple(Difficulty(item) for item in value["difficulties"]),
            artifacts=tuple(
                EvidenceIdentityV1.from_dict(item) for item in value["artifacts"]
            ),
        )

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def sha(self) -> str:
        return content_sha(self.to_dict())
