"""Versioned contracts for the staged DSL capability curriculum.

Capability, supervision source, evaluation source, and difficulty are separate
axes.  In particular, distillation and trace collection are processes; neither
is evidence that a model acquired a higher capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from slm_training.harness_core.checkpoint_reference import sha256_file
from slm_training.harness_core.lineage.records import canonical_json, content_sha
from slm_training.harness_core.record_schema import RUN_CLASSES

SCHEMA_VERSION = "staged_harness_baseline/v1"
UNKNOWN = "UNKNOWN"
FOUNDATION_CLAIMS = (
    "symbolic_surface_exactness",
    "plan_state_machine_enforcement",
    "artifact_identity",
    "root_family_split_isolation",
    "materialization_no_plan_parity",
    "certificate_lever_fail_closed",
)


def _require_exact_keys(
    value: dict[str, Any], expected: set[str], context: str
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{context} keys must be exact: "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


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


class FoundationClaimStatus(str, Enum):
    SUPPORTED = "supported"
    REJECTED = "rejected"
    UNKNOWN = "unknown"
    INVALID = "invalid"


class FoundationEvidenceClass(str, Enum):
    CONTRACT_FIXTURE = "contract_fixture"
    POWERED = "powered"


@dataclass(frozen=True)
class FoundationArtifactV1:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if (
            not self.path
            or Path(self.path).is_absolute()
            or ".." in Path(self.path).parts
        ):
            raise ValueError("foundation artifact path must be repository-relative")
        if len(self.sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.sha256
        ):
            raise ValueError("foundation artifact sha256 must be lowercase hexadecimal")

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FoundationArtifactV1:
        _require_exact_keys(value, {"path", "sha256"}, "foundation artifact")
        return cls(path=str(value["path"]), sha256=str(value["sha256"]))


@dataclass(frozen=True)
class FoundationClaimV1:
    name: str
    status: FoundationClaimStatus
    evidence_class: FoundationEvidenceClass
    artifacts: tuple[FoundationArtifactV1, ...]
    commands: tuple[str, ...]
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.name not in FOUNDATION_CLAIMS:
            raise ValueError(f"unknown foundation claim {self.name!r}")
        if not self.artifacts or not self.commands:
            raise ValueError(f"{self.name} requires artifacts and exact commands")
        if any(not command.strip() for command in self.commands):
            raise ValueError(f"{self.name} commands must be non-empty")
        paths = [artifact.path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError(f"{self.name} artifact paths must be unique")
        if self.status is FoundationClaimStatus.SUPPORTED and self.blockers:
            raise ValueError("supported foundation claims cannot carry blockers")
        if self.status is not FoundationClaimStatus.SUPPORTED and not self.blockers:
            raise ValueError("non-supported foundation claims require blockers")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "evidence_class": self.evidence_class.value,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "commands": list(self.commands),
            "blockers": list(self.blockers),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FoundationClaimV1:
        _require_exact_keys(
            value,
            {
                "name",
                "status",
                "evidence_class",
                "artifacts",
                "commands",
                "blockers",
            },
            "foundation claim",
        )
        return cls(
            name=str(value["name"]),
            status=FoundationClaimStatus(value["status"]),
            evidence_class=FoundationEvidenceClass(value["evidence_class"]),
            artifacts=tuple(
                FoundationArtifactV1.from_dict(item) for item in value["artifacts"]
            ),
            commands=tuple(str(item) for item in value["commands"]),
            blockers=tuple(str(item) for item in value.get("blockers", ())),
        )


@dataclass(frozen=True)
class FrozenFoundationIdentityV1:
    name: str
    version: str
    identity_sha256: str
    source: FoundationArtifactV1

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("frozen identity name and version are required")
        if len(self.identity_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.identity_sha256
        ):
            raise ValueError("frozen identity sha256 must be lowercase hexadecimal")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "identity_sha256": self.identity_sha256,
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FrozenFoundationIdentityV1:
        _require_exact_keys(
            value,
            {"name", "version", "identity_sha256", "source"},
            "frozen foundation identity",
        )
        return cls(
            name=str(value["name"]),
            version=str(value["version"]),
            identity_sha256=str(value["identity_sha256"]),
            source=FoundationArtifactV1.from_dict(value["source"]),
        )


@dataclass(frozen=True)
class StagedHarnessFoundationDispositionV1:
    source_commit: str
    claims: tuple[FoundationClaimV1, ...]
    frozen_identities: tuple[FrozenFoundationIdentityV1, ...]
    next_work_item: str
    version_stamp: dict[str, Any]
    run_class: str = "fixture_demo"
    schema_version: str = "staged_harness_foundation_disposition/v1"

    def __post_init__(self) -> None:
        if self.schema_version != "staged_harness_foundation_disposition/v1":
            raise ValueError("unsupported foundation disposition schema")
        if len(self.source_commit) != 40 or any(
            char not in "0123456789abcdef" for char in self.source_commit
        ):
            raise ValueError("source_commit must be a full lowercase Git SHA")
        if not self.next_work_item:
            raise ValueError("source_commit and next_work_item are required")
        if self.version_stamp.get(
            "stamp_schema"
        ) != "version_stamp/v1" or not isinstance(
            self.version_stamp.get("components"), dict
        ):
            raise ValueError("foundation disposition requires a version_stamp/v1")
        if self.run_class != "fixture_demo":
            raise ValueError("foundation disposition is contract-fixture evidence")
        names = tuple(claim.name for claim in self.claims)
        if len(names) != len(set(names)) or set(names) != set(FOUNDATION_CLAIMS):
            raise ValueError(
                "foundation disposition must cover every claim exactly once"
            )
        frozen_names = [identity.name for identity in self.frozen_identities]
        if len(frozen_names) != len(set(frozen_names)):
            raise ValueError("frozen foundation identity names must be unique")
        if self.is_supported and not self.frozen_identities:
            raise ValueError(
                "supported foundation disposition requires frozen identities"
            )

    @property
    def is_supported(self) -> bool:
        return all(
            claim.status is FoundationClaimStatus.SUPPORTED for claim in self.claims
        )

    def blocking_reasons(self, repo_root: Path | None = None) -> tuple[str, ...]:
        reasons = [
            f"{claim.name}: {claim.status.value}: {', '.join(claim.blockers)}"
            for claim in self.claims
            if claim.status is not FoundationClaimStatus.SUPPORTED
        ]
        if repo_root is not None:
            artifacts = {
                (artifact.path, artifact.sha256)
                for claim in self.claims
                for artifact in claim.artifacts
            }
            artifacts.update(
                (identity.source.path, identity.source.sha256)
                for identity in self.frozen_identities
            )
            for path, expected in sorted(artifacts):
                candidate = repo_root / path
                if not candidate.is_file():
                    reasons.append(f"{path}: missing")
                elif sha256_file(candidate) != expected:
                    reasons.append(f"{path}: sha256 mismatch")
        return tuple(reasons)

    def require_supported(self, repo_root: Path | None = None) -> None:
        reasons = self.blocking_reasons(repo_root)
        if reasons:
            raise ValueError("CAP0 foundation is blocked: " + "; ".join(reasons))

    def to_dict(self, *, include_sha: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "source_commit": self.source_commit,
            "run_class": self.run_class,
            "decision": "supported" if self.is_supported else "blocked",
            "claims": [claim.to_dict() for claim in self.claims],
            "frozen_identities": [
                identity.to_dict() for identity in self.frozen_identities
            ],
            "next_work_item": self.next_work_item,
            "version_stamp": self.version_stamp,
        }
        if include_sha:
            value["disposition_sha256"] = self.sha
        return value

    @property
    def sha(self) -> str:
        return content_sha(self.to_dict(include_sha=False))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> StagedHarnessFoundationDispositionV1:
        _require_exact_keys(
            value,
            {
                "schema_version",
                "source_commit",
                "run_class",
                "decision",
                "claims",
                "frozen_identities",
                "next_work_item",
                "version_stamp",
                "disposition_sha256",
            },
            "foundation disposition",
        )
        disposition = cls(
            schema_version=str(value["schema_version"]),
            source_commit=str(value["source_commit"]),
            run_class=str(value["run_class"]),
            claims=tuple(FoundationClaimV1.from_dict(item) for item in value["claims"]),
            frozen_identities=tuple(
                FrozenFoundationIdentityV1.from_dict(item)
                for item in value["frozen_identities"]
            ),
            next_work_item=str(value["next_work_item"]),
            version_stamp=dict(value["version_stamp"]),
        )
        if value.get("decision") != (
            "supported" if disposition.is_supported else "blocked"
        ):
            raise ValueError("foundation decision does not match claim statuses")
        if value.get("disposition_sha256") != disposition.sha:
            raise ValueError("foundation disposition sha256 does not match its body")
        return disposition

    @classmethod
    def load(cls, path: Path) -> StagedHarnessFoundationDispositionV1:
        import json

        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("foundation disposition must be a JSON object")
        return cls.from_dict(value)


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
            "supervision_sources": [value.value for value in self.supervision_sources],
            "evaluation_sources": [value.value for value in self.evaluation_sources],
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
