"""Immutable records for the two production model lineages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

Track = Literal["twotower", "causal_lm"]
LifecycleState = Literal[
    "running", "screened", "validated", "champion", "deployed", "rejected"
]
Initialization = Literal["scratch", "parent", "eval_only", "process", "legacy"]


def canonical_json(value: Any) -> str:
    """Return the stable JSON representation used by every lineage hash."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_sha(value: Any) -> str:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DataSnapshot:
    snapshot_id: str
    sources: tuple[str, ...]
    records_sha: str
    record_count: int
    target_token_count: int
    created_at: str
    annotations_sha: str | None = None
    replay_fraction: float = 0.10
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha(self) -> str:
        return content_sha(self)


@dataclass(frozen=True)
class EvaluationReport:
    report_id: str
    run_id: str
    eval_snapshot_sha: str
    created_at: str
    ship_gates_pass: bool
    weighted_nll: float | None
    category_nll: Mapping[str, float] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    suite_sizes: Mapping[str, int] = field(default_factory=dict)
    seed: int = 0
    token_rung: float = 1.0
    artifact_size_bytes: int | None = None
    warm_p95_seconds: float | None = None
    hardware: Mapping[str, Any] = field(default_factory=dict)
    comparisons: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha(self) -> str:
        return content_sha(self)


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    track: Track
    parent_ids: tuple[str, ...]
    base_model_id: str
    base_model_revision: str
    architecture_sha: str
    tokenizer_sha: str
    parameter_shapes_sha: str
    data_snapshot_sha: str
    eval_snapshot_sha: str
    recipe_sha: str
    code_sha: str
    seed: int
    hardware: Mapping[str, Any]
    artifact_uris: tuple[str, ...]
    metrics: Mapping[str, float]
    lifecycle_state: LifecycleState
    initialization: Initialization
    recipe: Mapping[str, Any]
    created_at: str
    legacy_kind: Literal["legacy_evidence", "hardware_smoke"] | None = None
    trace_id: str | None = None

    def __post_init__(self) -> None:
        if self.initialization in {"parent", "process"} and len(self.parent_ids) != 1:
            raise ValueError(f"{self.initialization} runs require exactly one parent")
        if self.initialization == "eval_only" and not self.parent_ids:
            raise ValueError("eval_only runs require a source parent")
        if self.track not in {"twotower", "causal_lm"}:
            raise ValueError(f"unknown track {self.track!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha(self) -> str:
        return content_sha(self)

    @property
    def compatibility_sha(self) -> str:
        return content_sha(
            {
                "track": self.track,
                "base_model_id": self.base_model_id,
                "base_model_revision": self.base_model_revision,
                "architecture_sha": self.architecture_sha,
                "tokenizer_sha": self.tokenizer_sha,
                "parameter_shapes_sha": self.parameter_shapes_sha,
            }
        )


@dataclass(frozen=True)
class MergeManifest:
    merge_id: str
    track: Track
    parent_id: str
    child_ids: tuple[str, ...]
    method: Literal["average", "ties"]
    compatibility_sha: str
    output_uri: str
    created_at: str
    density: float = 0.2
    evaluation_report_sha: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha(self) -> str:
        return content_sha(self)


@dataclass(frozen=True)
class ChampionPointer:
    pointer_id: str
    track: Track
    run_id: str
    artifact_uri: str
    manifest_sha: str
    evaluation_report_sha: str
    created_at: str
    previous_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha(self) -> str:
        return content_sha(self)
