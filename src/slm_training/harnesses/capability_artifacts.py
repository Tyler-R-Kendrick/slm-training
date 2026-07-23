"""Content-addressed staged-capability artifacts and publication checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from slm_training.harness_core.lineage.records import canonical_json, content_sha
from slm_training.harnesses.staged import Capability

SCHEMA_VERSION = "capability_artifacts/v1"


class TeacherTraceMode(str, Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"


@dataclass(frozen=True)
class ComplexityVectorV1:
    ast_nodes: int
    max_depth: int
    decision_count: int
    marker_count: int

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.to_dict().values()):
            raise ValueError("complexity values must be non-negative")

    def to_dict(self) -> dict[str, int]:
        return {
            "ast_nodes": self.ast_nodes,
            "max_depth": self.max_depth,
            "decision_count": self.decision_count,
            "marker_count": self.marker_count,
        }


@dataclass(frozen=True)
class ProcessProvenanceV1:
    process_id: str
    process_version: str
    config_sha256: str
    code_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.process_id, "process_id")
        _require_text(self.process_version, "process_version")
        _require_digest(self.config_sha256, "config_sha256")
        _require_digest(self.code_sha256, "code_sha256")

    def to_dict(self) -> dict[str, str]:
        return {
            "process_id": self.process_id,
            "process_version": self.process_version,
            "config_sha256": self.config_sha256,
            "code_sha256": self.code_sha256,
        }


@dataclass(frozen=True)
class LLMProvenanceV1:
    provider: str
    model: str
    prompt_sha256: str
    response_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.provider, "LLM provider")
        _require_text(self.model, "LLM model")
        _require_digest(self.prompt_sha256, "prompt_sha256")
        _require_digest(self.response_sha256, "response_sha256")

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_sha256": self.prompt_sha256,
            "response_sha256": self.response_sha256,
        }


@dataclass(frozen=True)
class TeacherTraceV1:
    mode: TeacherTraceMode
    teacher_id: str
    teacher_version: str
    trace_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.teacher_id, "teacher_id")
        _require_text(self.teacher_version, "teacher_version")
        _require_digest(self.trace_sha256, "trace_sha256")

    def to_dict(self) -> dict[str, str]:
        return {
            "mode": self.mode.value,
            "teacher_id": self.teacher_id,
            "teacher_version": self.teacher_version,
            "trace_sha256": self.trace_sha256,
        }


@dataclass(frozen=True)
class CompilerCoverageV1:
    required_paths: tuple[str, ...]
    covered_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_unique_text(self.required_paths, "required compiler paths")
        _require_unique_text(self.covered_paths, "covered compiler paths")
        unknown = set(self.covered_paths) - set(self.required_paths)
        if unknown:
            raise ValueError(
                f"covered compiler paths are not required: {sorted(unknown)}"
            )

    @property
    def complete(self) -> bool:
        return set(self.covered_paths) == set(self.required_paths)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "required_paths": sorted(self.required_paths),
            "covered_paths": sorted(self.covered_paths),
        }


@dataclass(frozen=True)
class AnswerArtifactV1:
    family_id: str
    split_id: str
    parent_ids: tuple[str, ...]
    canonical_ast_sha256: str
    surface_sha256: str
    marker_table_id: str
    grammar_start: str
    category: str
    complexity: ComplexityVectorV1
    equivalent_answer_ids: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        _require_text(self.family_id, "family_id")
        _require_text(self.split_id, "split_id")
        _require_digests(self.parent_ids, "parent_ids")
        _require_digest(self.canonical_ast_sha256, "canonical_ast_sha256")
        _require_digest(self.surface_sha256, "surface_sha256")
        _require_digest(self.marker_table_id, "marker_table_id")
        _require_text(self.grammar_start, "grammar_start")
        _require_text(self.category, "category")
        _require_digests(self.equivalent_answer_ids, "equivalent_answer_ids")

    @property
    def semantic_id(self) -> str:
        """Canonical meaning identity; surface and run lineage are metadata."""

        return content_sha(
            {
                "schema_version": self.schema_version,
                "canonical_ast_sha256": self.canonical_ast_sha256,
                "marker_table_id": self.marker_table_id,
                "grammar_start": self.grammar_start,
                "category": self.category,
                "complexity": self.complexity.to_dict(),
            }
        )

    def source_ids(self) -> tuple[str, ...]:
        return (*self.parent_ids, *self.equivalent_answer_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "answer",
            "schema_version": self.schema_version,
            "semantic_id": self.semantic_id,
            "family_id": self.family_id,
            "split_id": self.split_id,
            "parent_ids": sorted(self.parent_ids),
            "canonical_ast_sha256": self.canonical_ast_sha256,
            "surface_sha256": self.surface_sha256,
            "marker_table_id": self.marker_table_id,
            "grammar_start": self.grammar_start,
            "category": self.category,
            "complexity": self.complexity.to_dict(),
            "equivalent_answer_ids": sorted(self.equivalent_answer_ids),
        }


@dataclass(frozen=True)
class QuestionArtifactV1:
    family_id: str
    split_id: str
    parent_ids: tuple[str, ...]
    question_sha256: str
    marker_table_id: str
    grammar_start: str
    category: str
    complexity: ComplexityVectorV1
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        _require_text(self.family_id, "family_id")
        _require_text(self.split_id, "split_id")
        _require_digests(self.parent_ids, "parent_ids")
        _require_digest(self.question_sha256, "question_sha256")
        _require_digest(self.marker_table_id, "marker_table_id")
        _require_text(self.grammar_start, "grammar_start")
        _require_text(self.category, "category")

    @property
    def semantic_id(self) -> str:
        return content_sha(
            {
                "schema_version": self.schema_version,
                "question_sha256": self.question_sha256,
                "marker_table_id": self.marker_table_id,
                "grammar_start": self.grammar_start,
                "category": self.category,
                "complexity": self.complexity.to_dict(),
            }
        )

    def source_ids(self) -> tuple[str, ...]:
        return self.parent_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "question",
            "schema_version": self.schema_version,
            "semantic_id": self.semantic_id,
            "family_id": self.family_id,
            "split_id": self.split_id,
            "parent_ids": sorted(self.parent_ids),
            "question_sha256": self.question_sha256,
            "marker_table_id": self.marker_table_id,
            "grammar_start": self.grammar_start,
            "category": self.category,
            "complexity": self.complexity.to_dict(),
        }


@dataclass(frozen=True)
class QAPairArtifactV1:
    question_id: str
    accepted_answer_ids: tuple[str, ...]
    canonical_preference_answer_id: str | None
    equivalence_relation_sha256: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        _require_digest(self.question_id, "question_id")
        _require_digests(self.accepted_answer_ids, "accepted_answer_ids")
        if not self.accepted_answer_ids:
            raise ValueError("accepted_answer_ids must not be empty")
        if (
            self.canonical_preference_answer_id is not None
            and self.canonical_preference_answer_id not in self.accepted_answer_ids
        ):
            raise ValueError("canonical preference must belong to the accepted set")
        _require_digest(self.equivalence_relation_sha256, "equivalence_relation_sha256")

    @property
    def accepted_set_id(self) -> str:
        """Accepted semantics stay stable when canonical preference changes."""

        return content_sha(
            {
                "schema_version": self.schema_version,
                "question_id": self.question_id,
                "accepted_answer_ids": sorted(self.accepted_answer_ids),
                "equivalence_relation_sha256": self.equivalence_relation_sha256,
            }
        )

    @property
    def semantic_id(self) -> str:
        return self.accepted_set_id

    def source_ids(self) -> tuple[str, ...]:
        return (self.question_id, *self.accepted_answer_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "qa_pair",
            "schema_version": self.schema_version,
            "semantic_id": self.semantic_id,
            "question_id": self.question_id,
            "accepted_answer_ids": sorted(self.accepted_answer_ids),
            "accepted_set_id": self.accepted_set_id,
            "canonical_preference_answer_id": self.canonical_preference_answer_id,
            "equivalence_relation_sha256": self.equivalence_relation_sha256,
        }


@dataclass(frozen=True)
class DerivationActivityV1:
    invocation_id: str
    created_at: str
    source_ids: tuple[str, ...]
    output_ids: tuple[str, ...]
    process: ProcessProvenanceV1
    seed: int
    llm: LLMProvenanceV1 | None = None
    teacher_trace: TeacherTraceV1 | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        _require_text(self.invocation_id, "invocation_id")
        _require_text(self.created_at, "created_at")
        _require_digests(self.source_ids, "source_ids")
        _require_digests(self.output_ids, "output_ids")
        if not self.source_ids or not self.output_ids:
            raise ValueError("derivation source_ids and output_ids must not be empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    @property
    def activity_id(self) -> str:
        return content_sha(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "artifact_type": "derivation",
            "schema_version": self.schema_version,
            "invocation_id": self.invocation_id,
            "created_at": self.created_at,
            "source_ids": sorted(self.source_ids),
            "output_ids": sorted(self.output_ids),
            "process": self.process.to_dict(),
            "seed": self.seed,
            "llm": None if self.llm is None else self.llm.to_dict(),
            "teacher_trace": (
                None if self.teacher_trace is None else self.teacher_trace.to_dict()
            ),
        }
        if include_id:
            value["activity_id"] = self.activity_id
        return value


@dataclass(frozen=True)
class ValidationReportV1:
    invocation_id: str
    created_at: str
    source_ids: tuple[str, ...]
    process: ProcessProvenanceV1
    accepted: bool
    rejection_codes: tuple[str, ...]
    compiler_coverage: CompilerCoverageV1
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        _require_text(self.invocation_id, "invocation_id")
        _require_text(self.created_at, "created_at")
        _require_digests(self.source_ids, "source_ids")
        if not self.source_ids:
            raise ValueError("validation source_ids must not be empty")
        _require_unique_text(self.rejection_codes, "rejection_codes", allow_empty=True)
        if self.accepted and self.rejection_codes:
            raise ValueError("accepted validation cannot carry rejection codes")
        if self.accepted and not self.compiler_coverage.complete:
            raise ValueError("accepted validation requires complete compiler coverage")
        if not self.accepted and not self.rejection_codes:
            raise ValueError("rejected validation requires rejection codes")

    @property
    def report_id(self) -> str:
        return content_sha(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "artifact_type": "validation",
            "schema_version": self.schema_version,
            "invocation_id": self.invocation_id,
            "created_at": self.created_at,
            "source_ids": sorted(self.source_ids),
            "process": self.process.to_dict(),
            "accepted": self.accepted,
            "rejection_codes": sorted(self.rejection_codes),
            "compiler_coverage": self.compiler_coverage.to_dict(),
        }
        if include_id:
            value["report_id"] = self.report_id
        return value


@dataclass(frozen=True)
class CapabilityCertificateV1:
    capability: Capability
    plan_id: str
    plan_sha256: str
    qa_pair_ids: tuple[str, ...]
    validation_report_ids: tuple[str, ...]
    gate_process: ProcessProvenanceV1
    passed: bool
    rejection_codes: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        _require_text(self.plan_id, "plan_id")
        _require_digest(self.plan_sha256, "plan_sha256")
        _require_digests(self.qa_pair_ids, "qa_pair_ids")
        _require_digests(self.validation_report_ids, "validation_report_ids")
        if not self.qa_pair_ids or not self.validation_report_ids:
            raise ValueError("certificate evidence IDs must not be empty")
        _require_unique_text(self.rejection_codes, "rejection_codes", allow_empty=True)
        if self.passed and self.rejection_codes:
            raise ValueError("passed certificate cannot carry rejection codes")
        if not self.passed and not self.rejection_codes:
            raise ValueError("failed certificate requires rejection codes")

    @property
    def certificate_id(self) -> str:
        return content_sha(self.to_dict(include_id=False))

    def source_ids(self) -> tuple[str, ...]:
        return (*self.qa_pair_ids, *self.validation_report_ids)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "artifact_type": "certificate",
            "schema_version": self.schema_version,
            "capability": self.capability.value,
            "plan_id": self.plan_id,
            "plan_sha256": self.plan_sha256,
            "qa_pair_ids": sorted(self.qa_pair_ids),
            "validation_report_ids": sorted(self.validation_report_ids),
            "gate_process": self.gate_process.to_dict(),
            "passed": self.passed,
            "rejection_codes": sorted(self.rejection_codes),
        }
        if include_id:
            value["certificate_id"] = self.certificate_id
        return value


ArtifactV1 = (
    AnswerArtifactV1
    | QuestionArtifactV1
    | QAPairArtifactV1
    | DerivationActivityV1
    | ValidationReportV1
    | CapabilityCertificateV1
)


def artifact_id(artifact: ArtifactV1) -> str:
    for name in ("semantic_id", "activity_id", "report_id", "certificate_id"):
        value = getattr(artifact, name, None)
        if value is not None:
            return str(value)
    raise TypeError(f"unsupported artifact {type(artifact).__name__}")


def require_publishable(
    artifacts: Iterable[ArtifactV1], *, external_ids: Iterable[str] = ()
) -> None:
    """Fail unless every lineage edge resolves inside the batch or known store."""

    rows = tuple(artifacts)
    known = set(external_ids)
    _require_digests(tuple(known), "external_ids")
    for row in rows:
        identity = artifact_id(row)
        if identity in known:
            raise ValueError(f"duplicate artifact identity {identity}")
        known.add(identity)
    for row in rows:
        source_value = getattr(row, "source_ids")
        sources = tuple(source_value() if callable(source_value) else source_value)
        if isinstance(row, DerivationActivityV1):
            sources = (*sources, *row.output_ids)
        missing = sorted(set(sources) - known)
        if missing:
            raise ValueError(
                f"{type(row).__name__} has unresolved source IDs: {missing}"
            )


def record_json(artifact: ArtifactV1) -> str:
    return canonical_json(artifact.to_dict())


def artifact_from_json(value: str) -> ArtifactV1:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("artifact JSON must contain an object")
    return artifact_from_dict(payload)


def artifact_from_dict(value: Mapping[str, Any]) -> ArtifactV1:
    """Strict loader: unknown fields, stale schemas, and identity drift reject."""

    artifact_type = value.get("artifact_type")
    if artifact_type == "answer":
        artifact: ArtifactV1 = AnswerArtifactV1(
            family_id=_string(value, "family_id"),
            split_id=_string(value, "split_id"),
            parent_ids=_strings(value, "parent_ids"),
            canonical_ast_sha256=_string(value, "canonical_ast_sha256"),
            surface_sha256=_string(value, "surface_sha256"),
            marker_table_id=_string(value, "marker_table_id"),
            grammar_start=_string(value, "grammar_start"),
            category=_string(value, "category"),
            complexity=_complexity(value),
            equivalent_answer_ids=_strings(value, "equivalent_answer_ids"),
            schema_version=_string(value, "schema_version"),
        )
    elif artifact_type == "question":
        artifact = QuestionArtifactV1(
            family_id=_string(value, "family_id"),
            split_id=_string(value, "split_id"),
            parent_ids=_strings(value, "parent_ids"),
            question_sha256=_string(value, "question_sha256"),
            marker_table_id=_string(value, "marker_table_id"),
            grammar_start=_string(value, "grammar_start"),
            category=_string(value, "category"),
            complexity=_complexity(value),
            schema_version=_string(value, "schema_version"),
        )
    elif artifact_type == "qa_pair":
        preference = value.get("canonical_preference_answer_id")
        if preference is not None and not isinstance(preference, str):
            raise ValueError("canonical_preference_answer_id must be a string or null")
        artifact = QAPairArtifactV1(
            question_id=_string(value, "question_id"),
            accepted_answer_ids=_strings(value, "accepted_answer_ids"),
            canonical_preference_answer_id=preference,
            equivalence_relation_sha256=_string(value, "equivalence_relation_sha256"),
            schema_version=_string(value, "schema_version"),
        )
    elif artifact_type == "derivation":
        llm_value = value.get("llm")
        trace_value = value.get("teacher_trace")
        artifact = DerivationActivityV1(
            invocation_id=_string(value, "invocation_id"),
            created_at=_string(value, "created_at"),
            source_ids=_strings(value, "source_ids"),
            output_ids=_strings(value, "output_ids"),
            process=_process(value, "process"),
            seed=_integer(value, "seed"),
            llm=None if llm_value is None else _llm(_mapping(llm_value, "llm")),
            teacher_trace=(
                None
                if trace_value is None
                else _teacher_trace(_mapping(trace_value, "teacher_trace"))
            ),
            schema_version=_string(value, "schema_version"),
        )
    elif artifact_type == "validation":
        artifact = ValidationReportV1(
            invocation_id=_string(value, "invocation_id"),
            created_at=_string(value, "created_at"),
            source_ids=_strings(value, "source_ids"),
            process=_process(value, "process"),
            accepted=_boolean(value, "accepted"),
            rejection_codes=_strings(value, "rejection_codes"),
            compiler_coverage=_coverage(value),
            schema_version=_string(value, "schema_version"),
        )
    elif artifact_type == "certificate":
        artifact = CapabilityCertificateV1(
            capability=Capability(_string(value, "capability")),
            plan_id=_string(value, "plan_id"),
            plan_sha256=_string(value, "plan_sha256"),
            qa_pair_ids=_strings(value, "qa_pair_ids"),
            validation_report_ids=_strings(value, "validation_report_ids"),
            gate_process=_process(value, "gate_process"),
            passed=_boolean(value, "passed"),
            rejection_codes=_strings(value, "rejection_codes"),
            schema_version=_string(value, "schema_version"),
        )
    else:
        raise ValueError(f"unknown artifact_type {artifact_type!r}")
    if artifact.to_dict() != dict(value):
        raise ValueError("artifact fields or recorded identity are not canonical")
    return artifact


def _require_schema(value: str) -> None:
    if value != SCHEMA_VERSION:
        raise ValueError(f"schema mismatch: expected {SCHEMA_VERSION!r}, got {value!r}")


def _require_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")


def _require_digest(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")


def _require_digests(values: tuple[str, ...], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    for value in values:
        _require_digest(value, label)


def _require_unique_text(
    values: tuple[str, ...], label: str, *, allow_empty: bool = False
) -> None:
    if not allow_empty and not values:
        raise ValueError(f"{label} must not be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    for value in values:
        _require_text(value, label)


def _string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise ValueError(f"{key} must be a string")
    return item


def _strings(value: Mapping[str, Any], key: str) -> tuple[str, ...]:
    item = value.get(key)
    if not isinstance(item, list) or any(not isinstance(row, str) for row in item):
        raise ValueError(f"{key} must be a string array")
    return tuple(item)


def _integer(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise ValueError(f"{key} must be an integer")
    return item


def _boolean(value: Mapping[str, Any], key: str) -> bool:
    item = value.get(key)
    if not isinstance(item, bool):
        raise ValueError(f"{key} must be a boolean")
    return item


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _complexity(value: Mapping[str, Any]) -> ComplexityVectorV1:
    row = _mapping(value.get("complexity"), "complexity")
    return ComplexityVectorV1(
        ast_nodes=_integer(row, "ast_nodes"),
        max_depth=_integer(row, "max_depth"),
        decision_count=_integer(row, "decision_count"),
        marker_count=_integer(row, "marker_count"),
    )


def _process(value: Mapping[str, Any], key: str) -> ProcessProvenanceV1:
    row = _mapping(value.get(key), key)
    return ProcessProvenanceV1(
        process_id=_string(row, "process_id"),
        process_version=_string(row, "process_version"),
        config_sha256=_string(row, "config_sha256"),
        code_sha256=_string(row, "code_sha256"),
    )


def _llm(value: Mapping[str, Any]) -> LLMProvenanceV1:
    return LLMProvenanceV1(
        provider=_string(value, "provider"),
        model=_string(value, "model"),
        prompt_sha256=_string(value, "prompt_sha256"),
        response_sha256=_string(value, "response_sha256"),
    )


def _teacher_trace(value: Mapping[str, Any]) -> TeacherTraceV1:
    return TeacherTraceV1(
        mode=TeacherTraceMode(_string(value, "mode")),
        teacher_id=_string(value, "teacher_id"),
        teacher_version=_string(value, "teacher_version"),
        trace_sha256=_string(value, "trace_sha256"),
    )


def _coverage(value: Mapping[str, Any]) -> CompilerCoverageV1:
    row = _mapping(value.get("compiler_coverage"), "compiler_coverage")
    return CompilerCoverageV1(
        required_paths=_strings(row, "required_paths"),
        covered_paths=_strings(row, "covered_paths"),
    )
