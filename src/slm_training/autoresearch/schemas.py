"""Strict schemas for reproducible autoresearch campaigns and decisions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CampaignBudget(StrictModel):
    max_experiments: int = Field(default=12, ge=1, le=1000)
    max_gpu_hours: float = Field(default=0.0, ge=0)
    max_wall_minutes: int = Field(default=240, ge=1)


DEFAULT_ALLOWED_KNOBS = frozenset(
    {
        "batch_size",
        "context_backend",
        "data_source",
        "derive_from",
        "lr",
        "max_records_per_parent",
        "min_quality_score",
        "mixture_weights",
        "seed",
        "steps",
        "synthesizer",
    }
)


class CampaignSpec(StrictModel):
    campaign_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    objective: str = Field(min_length=8)
    primary_metric: str = Field(min_length=1)
    track: Literal["twotower", "causal_lm"] = "twotower"
    researcher_mode: Literal["agent", "openai", "fixture"] = "agent"
    evidence_roots: tuple[str, ...] = ("outputs",)
    allowed_knobs: frozenset[str] = DEFAULT_ALLOWED_KNOBS
    budget: CampaignBudget = Field(default_factory=CampaignBudget)
    created_at: str = Field(default_factory=utc_now)
    notes: str = ""


class ResearchSource(StrictModel):
    source_id: str
    kind: Literal[
        "repo_lineage",
        "hf_daily_paper",
        "hf_paper_search",
        "web",
        "prior_run",
        "telemetry",
        "feedback",
        "data_snapshot",
        "agent",
    ]
    title: str
    uri: str
    retrieved_at: str = Field(default_factory=utc_now)
    published_at: str | None = None
    sha256: str | None = None
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(StrictModel):
    path: str
    kind: str
    sha256: str
    size_bytes: int = Field(ge=0)
    summary: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)


class EvidenceSnapshot(StrictModel):
    snapshot_id: str
    created_at: str = Field(default_factory=utc_now)
    roots: tuple[str, ...]
    items: tuple[EvidenceItem, ...]
    source_counts: dict[str, int] = Field(default_factory=dict)
    prior_campaign_ids: tuple[str, ...] = ()


class ExperimentKnobs(StrictModel):
    data_source: Literal[
        "rico",
        "fixture",
        "existing",
        "both",
        "awwwards",
        "rico+awwwards",
        "programspec",
        "language_contract",
        "deconstruct",
        "render",
        "integrated",
        "all",
    ] | None = None
    derive_from: str | None = None
    synthesizer: Literal[
        "quality", "template", "layout", "frontier", "none", "noop", "off"
    ] | None = None
    max_records_per_parent: int | None = Field(default=None, ge=1, le=100)
    min_quality_score: float | None = Field(default=None, ge=0, le=1)
    mixture_weights: dict[str, float] | None = None
    steps: int | None = Field(default=None, ge=1, le=100_000)
    batch_size: int | None = Field(default=None, ge=1, le=1024)
    lr: float | None = Field(default=None, gt=0, le=1)
    seed: int | None = Field(default=None, ge=0)
    context_backend: Literal["scratch", "hf"] | None = None

    @model_validator(mode="after")
    def validate_mixture(self) -> ExperimentKnobs:
        if self.mixture_weights is not None:
            if not self.mixture_weights or any(v <= 0 for v in self.mixture_weights.values()):
                raise ValueError("mixture weights must be non-empty and positive")
        if self.data_source == "existing" and not self.derive_from:
            raise ValueError("data_source=existing requires derive_from")
        return self


class ExperimentSpec(StrictModel):
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    campaign_id: str
    hypothesis: str = Field(min_length=12)
    rationale: str = Field(min_length=12)
    expected_effect: str = Field(min_length=8)
    falsification_criteria: tuple[str, ...] = Field(min_length=1)
    stop_conditions: tuple[str, ...] = Field(min_length=1)
    citations: tuple[str, ...] = Field(min_length=1)
    knobs: ExperimentKnobs
    parent_experiment_id: str | None = None
    requires_rl: bool = False
    rl_readiness_report: str | None = None
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_change(self) -> ExperimentSpec:
        if not self.knobs.model_dump(exclude_none=True):
            raise ValueError("experiment must change at least one allowlisted knob")
        if self.requires_rl and not self.rl_readiness_report:
            raise ValueError("RL experiments require an approved readiness report")
        return self


class ExperimentOutcome(StrictModel):
    experiment_id: str
    campaign_id: str
    status: Literal["planned", "running", "completed", "failed", "stopped"]
    metrics: dict[str, float] = Field(default_factory=dict)
    data_metrics: dict[str, float] = Field(default_factory=dict)
    artifact_uris: tuple[str, ...] = ()
    telemetry_uris: tuple[str, ...] = ()
    command: tuple[str, ...] = ()
    exit_code: int | None = None
    error: str | None = None
    stage_telemetry: tuple[dict[str, Any], ...] = ()
    started_at: str | None = None
    finished_at: str | None = None


class Diagnosis(StrictModel):
    experiment_id: str
    target: Literal["data", "researcher", "model", "infrastructure", "none"]
    confidence: float = Field(ge=0, le=1)
    evidence: tuple[str, ...]
    recommended_actions: tuple[str, ...]
    created_at: str = Field(default_factory=utc_now)


class RLReadinessReport(StrictModel):
    report_id: str
    evaluation_sha256: str
    frozen_snapshot: bool
    required_suites: tuple[str, ...]
    suite_sizes: dict[str, int]
    ship_gates_pass: bool
    agentv_pass: bool
    reward_sample_count: int = Field(ge=0)
    reward_variance: float = Field(ge=0)
    approved: bool
    failures: tuple[str, ...] = ()
    created_at: str = Field(default_factory=utc_now)


class ResearcherBenchmarkReport(StrictModel):
    benchmark_id: str
    researcher_id: str
    cases: int = Field(ge=1)
    grounded_rate: float = Field(ge=0, le=1)
    valid_spec_rate: float = Field(ge=0, le=1)
    novel_rate: float = Field(ge=0, le=1)
    actionable_rate: float = Field(ge=0, le=1)
    pass_threshold: float = Field(ge=0, le=1)
    passed: bool
    human_approved: bool = False
    promotable: bool = False
    agentv: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)
