"""Strict schemas for reproducible autoresearch campaigns and decisions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from slm_training.levers import MAX_RUN_MINUTES

HarnessFamily = Literal[
    "autoresearch",
    "annotations",
    "distill",
    "experiments",
    "model_build",
    "preference",
    "quality",
    "rl",
    "test_data",
    "train_data",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CampaignBudget(StrictModel):
    max_experiments: int = Field(default=12, ge=1, le=1000)
    max_gpu_hours: float = Field(default=0.0, ge=0)
    max_wall_minutes: float = Field(
        default=float(MAX_RUN_MINUTES), gt=0, le=float(MAX_RUN_MINUTES)
    )


DEFAULT_ALLOWED_KNOBS = frozenset(
    {
        "batch_size",
        "allow_unconstrained_fallback",
        "asap_decode",
        "bind_encoding",
        "decode_min_content",
        "denoiser_backend",
        "mask_pattern",
        "context_backend",
        "compiler_alignment_loss_weight",
        "compiler_alignment_margin",
        "compiler_alignment_stratified",
        "compiler_alignment_semantic_exhaustive",
        "component_inventory_loss_weight",
        "component_inventory_decode_weight",
        "component_plan_loss_weight",
        "component_plan_decode_weight",
        "component_edge_loss_weight",
        "component_edge_alignment_loss_weight",
        "component_edge_decode_weight",
        "binder_component_plan_loss_weight",
        "binder_component_plan_decode_weight",
        "binder_topology_loss_weight",
        "binder_topology_decode_weight",
        "binder_arity_loss_weight",
        "binder_arity_decode_weight",
        "compiler_decode_mode",
        "compiler_search_mode",
        "compiler_search_trigger",
        "compiler_search_width",
        "compiler_search_noise",
        "compiler_search_stagnation_patience",
        "compiler_search_backtrack_limit",
        "data_source",
        "design_md_context",
        "eval_version",
        "derive_from",
        "lr",
        "local_files_only",
        "max_records_per_parent",
        "min_quality_score",
        "mixture_weights",
        "mixture_sampling_policy",
        "mixture_exposure_target_profile",
        "mixture_total_decision_budget",
        "mixture_per_root_cap",
        "mixture_per_template_cap",
        "mixture_max_importance_weight",
        "output_tokenizer",
        "seed",
        "schema_in_context",
        "slot_contract_in_context",
        "sync_checkpoints",
        "steps",
        "synthesizer",
        "train_version",
        "topology_actions",
        "topology_bounded_buffer",
        "topology_critic_decode",
        "topology_accept_threshold",
        "topology_contract_threshold",
        "topology_global_sync_interval",
        "topology_heterogeneous_noise",
        "topology_max_active",
        "topology_max_arity",
        "topology_max_depth",
        "topology_max_nodes",
        "topology_max_phases",
        "topology_structural_embeddings",
        "scope_contracts",
        "scope_independent_noise",
        "scope_local_oracle",
        "scope_contract_negatives",
        "runtime_symbol_features",
        "symbol_slot_augmentation",
        "semantic_candidate_masks",
        "constraint_graph_mode",
        "grammar_completion_bounds",
        "grammar_equivalence_cache",
        "grammar_active_symbol_bitsets",
        "compact_active_canvas",
        "action_embedding_init",
        "action_embedding_train",
        "action_alias_mode",
        "action_alias_manifest",
        "action_description_name_mode",
    }
)


class CampaignSpec(StrictModel):
    campaign_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    objective: str = Field(min_length=8)
    primary_metric: str = Field(min_length=1)
    track: Literal["twotower", "grammar_diffusion", "causal_lm"] = "twotower"
    researcher_mode: str = Field(
        default="agent", pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$"
    )
    min_hypotheses: int = Field(default=5, ge=5, le=100)
    evidence_roots: tuple[str, ...] = ("outputs",)
    allowed_knobs: frozenset[str] = DEFAULT_ALLOWED_KNOBS
    budget: CampaignBudget = Field(default_factory=CampaignBudget)
    loop_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
    )
    cycle_index: int | None = Field(default=None, ge=1)
    predecessor_campaign_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
    )
    upstream_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    integration_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    created_at: str = Field(default_factory=utc_now)
    notes: str = ""

    @model_validator(mode="after")
    def validate_loop_lineage(self) -> CampaignSpec:
        if self.loop_id is None:
            if any(
                value is not None
                for value in (
                    self.cycle_index,
                    self.predecessor_campaign_id,
                    self.upstream_commit,
                    self.integration_commit,
                )
            ):
                raise ValueError("cycle lineage requires loop_id")
            return self
        if self.cycle_index is None:
            raise ValueError("loop_id requires cycle_index")
        if self.upstream_commit is None or self.integration_commit is None:
            raise ValueError(
                "continuous cycle requires upstream_commit and integration_commit"
            )
        if self.cycle_index == 1 and self.predecessor_campaign_id is not None:
            raise ValueError("first loop cycle cannot have a predecessor campaign")
        if self.cycle_index > 1 and self.predecessor_campaign_id is None:
            raise ValueError("later loop cycles require a predecessor campaign")
        return self


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
        "researcher",
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


class OpenDeepResearchConfig(StrictModel):
    search_api: str = "tavily"
    summarization_model: str = "openai:gpt-4.1-mini"
    research_model: str = "openai:gpt-4.1"
    compression_model: str = "openai:gpt-4.1"
    final_report_model: str = "openai:gpt-4.1"
    max_concurrent_research_units: int = Field(default=3, ge=1, le=20)
    max_researcher_iterations: int = Field(default=3, ge=1, le=20)
    max_react_tool_calls: int = Field(default=5, ge=1, le=50)


class OpenResearcherConfig(StrictModel):
    base_url: str
    model: str = "OpenResearcher/OpenResearcher-30B-A3B"
    browser_backend: Literal["local", "serper"] = "serper"
    search_url: str | None = None
    max_rounds: int = Field(default=50, ge=1, le=200)


class ResearchRequest(StrictModel):
    researcher_id: str
    upstream_repo: str
    upstream_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    campaign_id: str
    evidence_snapshot_id: str
    prompt: str = Field(min_length=1, max_length=120_000)
    config: dict[str, Any] = Field(default_factory=dict)


class ResearcherRun(StrictModel):
    researcher_id: str
    upstream_repo: str
    upstream_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["completed", "failed"]
    memo: str = Field(default="", max_length=250_000)
    sources: tuple[ResearchSource, ...] = ()
    trace: dict[str, Any] = Field(default_factory=dict)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    started_at: str
    finished_at: str

    @model_validator(mode="after")
    def validate_status(self) -> ResearcherRun:
        if self.status == "completed" and not self.memo.strip():
            raise ValueError("completed researcher run requires a memo")
        if self.status == "failed" and not self.error:
            raise ValueError("failed researcher run requires an error")
        return self


class ExperimentKnobs(StrictModel):
    allow_unconstrained_fallback: bool | None = None
    eval_version: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
    )
    train_version: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
    )
    data_source: (
        Literal[
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
        ]
        | None
    ) = None
    derive_from: str | None = None
    synthesizer: (
        Literal["quality", "template", "layout", "frontier", "none", "noop", "off"]
        | None
    ) = None
    max_records_per_parent: int | None = Field(default=None, ge=1, le=100)
    min_quality_score: float | None = Field(default=None, ge=0, le=1)
    mixture_weights: dict[str, float] | None = None
    mixture_sampling_policy: (
        Literal[
            "with_replacement",
            "capacity_aware",
            "quota_capacity_aware",
            "exposure_targeted",
        ]
        | None
    ) = None
    mixture_exposure_target_profile: str | None = None
    mixture_total_decision_budget: int | None = Field(default=None, ge=1)
    mixture_per_root_cap: int | None = Field(default=None, ge=1)
    mixture_per_template_cap: int | None = Field(default=None, ge=1)
    mixture_max_importance_weight: float | None = Field(default=None, ge=1.0)
    steps: int | None = Field(default=None, ge=1, le=100_000)
    batch_size: int | None = Field(default=None, ge=1, le=1024)
    lr: float | None = Field(default=None, gt=0, le=1)
    seed: int | None = Field(default=None, ge=0)
    context_backend: Literal["scratch", "hf"] | None = None
    output_tokenizer: Literal["compositional", "lexer"] | None = None
    compiler_alignment_loss_weight: float | None = Field(default=None, ge=0, le=10)
    compiler_alignment_margin: float | None = Field(default=None, ge=0, le=20)
    compiler_alignment_stratified: bool | None = None
    compiler_alignment_semantic_exhaustive: bool | None = None
    component_inventory_loss_weight: float | None = Field(default=None, ge=0, le=20)
    component_inventory_decode_weight: float | None = Field(default=None, ge=0, le=20)
    component_plan_loss_weight: float | None = Field(default=None, ge=0, le=20)
    component_plan_decode_weight: float | None = Field(default=None, ge=0, le=20)
    component_edge_loss_weight: float | None = Field(default=None, ge=0, le=20)
    component_edge_alignment_loss_weight: float | None = Field(
        default=None, ge=0, le=20
    )
    component_edge_decode_weight: float | None = Field(default=None, ge=0, le=20)
    binder_component_plan_loss_weight: float | None = Field(default=None, ge=0, le=20)
    binder_component_plan_decode_weight: float | None = Field(default=None, ge=0, le=20)
    binder_topology_loss_weight: float | None = Field(default=None, ge=0, le=20)
    binder_topology_decode_weight: float | None = Field(default=None, ge=0, le=20)
    binder_arity_loss_weight: float | None = Field(default=None, ge=0, le=20)
    binder_arity_decode_weight: float | None = Field(default=None, ge=0, le=20)
    compiler_decode_mode: Literal["off", "forced", "restricted", "tree"] | None = None
    compiler_search_mode: Literal["greedy", "lattice", "ptrm", "gram"] | None = None
    compiler_search_trigger: Literal["bottom", "stagnation", "always"] | None = None
    compiler_search_width: int | None = Field(default=None, ge=1, le=64)
    compiler_search_noise: float | None = Field(default=None, ge=0, le=100)
    compiler_search_stagnation_patience: int | None = Field(default=None, ge=1, le=64)
    compiler_search_backtrack_limit: int | None = Field(default=None, ge=0, le=1024)
    schema_in_context: bool | None = None
    slot_contract_in_context: bool | None = None
    design_md_context: bool | None = None
    local_files_only: bool | None = None
    sync_checkpoints: bool | None = None
    topology_actions: bool | None = None
    topology_structural_embeddings: bool | None = None
    topology_heterogeneous_noise: bool | None = None
    topology_critic_decode: bool | None = None
    topology_bounded_buffer: bool | None = None
    topology_max_nodes: int | None = Field(default=None, ge=16, le=1024)
    topology_max_active: int | None = Field(default=None, ge=1, le=256)
    topology_max_arity: int | None = Field(default=None, ge=1, le=32)
    topology_max_depth: int | None = Field(default=None, ge=2, le=128)
    topology_max_phases: int | None = Field(default=None, ge=2, le=256)
    topology_global_sync_interval: int | None = Field(default=None, ge=1, le=32)
    topology_accept_threshold: float | None = Field(default=None, ge=0, le=1)
    topology_contract_threshold: float | None = Field(default=None, ge=0, le=1)
    # DSL diffusion research program levers (G1, SLM-46): Track A decode
    # (A2 ASAp mass removal, A4 min-content floor), Track B backbone (B4)
    # and Track C representation (C1 relative binder refs).
    asap_decode: bool | None = None
    decode_min_content: int | None = Field(default=None, ge=-1, le=64)
    denoiser_backend: Literal["scratch", "hf"] | None = None
    bind_encoding: Literal["absolute", "relative"] | None = None
    mask_pattern: Literal["random", "mixed", "diffusion"] | None = None
    runtime_symbol_features: (
        Literal["none", "surface", "role_gated", "replace"] | None
    ) = None
    symbol_slot_augmentation: bool | None = None
    semantic_candidate_masks: bool | None = None
    constraint_graph_mode: Literal["off", "grammar", "hybrid"] | None = None
    grammar_completion_bounds: bool | None = None
    grammar_equivalence_cache: bool | None = None
    grammar_active_symbol_bitsets: bool | None = None
    compact_active_canvas: bool | None = None
    action_embedding_init: (
        Literal[
            "none",
            "current_stub",
            "schema_description",
            "expanded_description",
            "shuffled",
            "alias_aware_description",
            "alias_aware_signature_only",
            "alias_aware_shuffled",
            "description_without_canonical_name",
            "canonical_name_plus_description",
            "signature_only",
        ]
        | None
    ) = None
    action_embedding_train: Literal["frozen", "trainable"] | None = None
    action_alias_mode: Literal["canonical", "off", "fixed", "held_out"] | None = None
    action_alias_manifest: str | None = None
    action_description_name_mode: (
        Literal[
            "schema",
            "alias_aware_description",
            "alias_aware_signature_only",
            "alias_aware_shuffled",
            "description_without_canonical_name",
            "canonical_name_plus_description",
            "signature_only",
        ]
        | None
    ) = None
    scope_contracts: bool | None = None
    scope_independent_noise: bool | None = None
    scope_local_oracle: bool | None = None
    scope_contract_negatives: bool | None = None

    @model_validator(mode="after")
    def validate_mixture(self) -> ExperimentKnobs:
        if self.train_version and self.data_source:
            raise ValueError("choose train_version or data_source, not both")
        if self.mixture_weights is not None:
            if not self.mixture_weights or any(
                v <= 0 for v in self.mixture_weights.values()
            ):
                raise ValueError("mixture weights must be non-empty and positive")
        if self.data_source == "existing" and not self.derive_from:
            raise ValueError("data_source=existing requires derive_from")
        scope_fields = (
            self.scope_contracts,
            self.scope_independent_noise,
            self.scope_local_oracle,
            self.scope_contract_negatives,
        )
        if any(
            value is not None for value in scope_fields
        ) and self.data_source not in {
            "programspec",
            "integrated",
            "all",
        }:
            raise ValueError("scope knobs require ProgramSpec-backed data_source")
        return self


FormalProofPolicy = Literal["required", "advisory"]
FormalProofStatus = Literal["proved", "refuted", "conditional", "unknown"]
FormalEvidenceScope = Literal["universal", "bounded_instance", "conditional"]


class FormalClaimV1(StrictModel):
    """A hypothesis claim routed to one versioned formal template."""

    template_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    claim: str = Field(min_length=12)
    policy: FormalProofPolicy = "advisory"


class FormalObligationV1(StrictModel):
    """Immutable link from a campaign lock to a formal-preflight artifact."""

    obligation_id: str = Field(pattern=r"^formal-[0-9a-f]{16}$")
    template_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    policy: FormalProofPolicy
    preflight_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FormalPreflightV1(StrictModel):
    """Auditable proof or counterexample result produced before execution."""

    schema_version: Literal["FormalPreflightV1"] = "FormalPreflightV1"
    campaign_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    obligation_id: str = Field(pattern=r"^formal-[0-9a-f]{16}$")
    template_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    template_version: str = Field(pattern=r"^v[1-9][0-9]*$")
    claim: str = Field(min_length=12)
    policy: FormalProofPolicy
    status: FormalProofStatus
    evidence_scope: FormalEvidenceScope
    theorem: str = Field(min_length=1)
    proof_target: str = Field(min_length=1)
    checker_contract: str | None = None
    assumptions: tuple[str, ...] = ()
    open_assumptions: tuple[str, ...] = ()
    source_digests: dict[str, str] = Field(min_length=1)
    proof_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lean_version: str = Field(min_length=1)
    mathlib_version: str = Field(min_length=1)
    build_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    counterexample: dict[str, Any] | None = None
    duration_seconds: float = Field(ge=0)
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_evidence(self) -> FormalPreflightV1:
        invalid = {
            path: digest
            for path, digest in self.source_digests.items()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)
        }
        if invalid:
            raise ValueError(f"invalid source digests: {sorted(invalid)}")
        if self.status == "refuted" and self.counterexample is None:
            raise ValueError("refuted formal preflights require a counterexample")
        if self.status == "proved" and self.open_assumptions:
            raise ValueError("proved formal preflights cannot have open assumptions")
        if self.status == "conditional" and not self.open_assumptions:
            raise ValueError("conditional formal preflights require open assumptions")
        return self


class FormalTraceStepV1(StrictModel):
    """JSON representation of one step accepted by ``Trace.validTrace``."""

    before_removed: tuple[StrictInt, ...]
    after_removed: tuple[StrictInt, ...]
    certified: tuple[StrictInt, ...]
    before_history: tuple[str, ...]
    after_history: tuple[str, ...]

    @model_validator(mode="after")
    def unique_sets(self) -> FormalTraceStepV1:
        for label, values in (
            ("before_removed", self.before_removed),
            ("after_removed", self.after_removed),
            ("certified", self.certified),
        ):
            if any(value < 0 for value in values):
                raise ValueError(f"{label} must contain nonnegative integer ordinals")
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must contain unique candidates")
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
    formal_claims: tuple[FormalClaimV1, ...] = ()
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
        template_ids = tuple(claim.template_id for claim in self.formal_claims)
        if len(template_ids) != len(set(template_ids)):
            raise ValueError("formal claim template identifiers must be unique")
        return self


class EvidenceUse(StrictModel):
    role: Literal["research", "prior_trace", "prior_result"]
    citation: str = Field(min_length=1)
    contribution: str = Field(min_length=8)


class CategoricalNoveltyAudit(StrictModel):
    """Pre-run audit inspired by arXiv:2606.01444; never a discovery proof."""

    transition_kind: Literal["fixed_regime_search", "regime_transition_candidate"]
    old_schema_elements: tuple[str, ...] = Field(min_length=1)
    proposed_schema_elements: tuple[str, ...] = Field(min_length=1)
    transported_elements: tuple[str, ...] = Field(min_length=1)
    transport_analysis: tuple[str, ...] = Field(min_length=1)
    residual_elements: tuple[str, ...] = Field(min_length=1)
    preservation_checks: tuple[str, ...] = Field(min_length=1)
    stress_tests: tuple[str, ...] = Field(min_length=1)
    worthiness_criteria: tuple[str, ...] = Field(min_length=1)
    status: Literal["candidate"] = "candidate"

    @model_validator(mode="after")
    def validate_transition(self) -> CategoricalNoveltyAudit:
        if set(self.residual_elements) & set(self.transported_elements):
            raise ValueError("residual elements must lie outside declared transport")
        if self.transition_kind == "regime_transition_candidate" and set(
            self.proposed_schema_elements
        ) <= set(self.old_schema_elements):
            raise ValueError(
                "regime transition candidate requires a proposed schema extension"
            )
        return self


DiagnosisLane = Literal[
    "measurement_control",
    "training_method",
    "architecture",
    "lean_model",
    "assumptions",
]

PriorityArea = Literal[
    "data",
    "researcher",
    "model",
    "infrastructure",
    "evaluation",
    "promotion",
    "autoresearch",
    "annotations",
    "distill",
    "experiments",
    "model_build",
    "preference",
    "quality",
    "rl",
    "test_data",
    "train_data",
    "measurement_control",
    "training_method",
    "architecture",
    "lean_model",
    "assumptions",
]


class MetricBandBreachV1(StrictModel):
    metric_id: str = Field(min_length=1)
    authority: Literal["theorem", "assumption_backed"]
    relation: Literal["below", "above", "mixed"]


class OptimumFeedbackV1(StrictModel):
    """Cycle-level disposition from a replayed Lean metric certificate."""

    schema_version: Literal["OptimumFeedbackV1"] = "OptimumFeedbackV1"
    policy: Literal[
        "continue",
        "stop",
        "block_promotion_and_diagnose",
        "historical_only",
    ]
    breaches: tuple[MetricBandBreachV1, ...] = ()
    diagnosis_lanes: tuple[DiagnosisLane, ...] = ()
    metric_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric_certificate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_disposition(self) -> OptimumFeedbackV1:
        has_breach = bool(self.breaches)
        if has_breach != bool(self.diagnosis_lanes):
            raise ValueError("band breaches and diagnosis lanes must appear together")
        if has_breach and set(self.diagnosis_lanes) != {
            "measurement_control",
            "training_method",
            "architecture",
            "lean_model",
            "assumptions",
        }:
            raise ValueError("band breaches require the complete diagnosis matrix")
        if self.policy in {"continue", "historical_only"} and has_breach:
            raise ValueError(f"{self.policy} feedback cannot contain a breach")
        if self.policy in {"stop", "block_promotion_and_diagnose"} and not has_breach:
            raise ValueError(f"{self.policy} feedback requires a breach")
        if self.policy == "stop" and not any(
            breach.authority == "theorem" for breach in self.breaches
        ):
            raise ValueError("stop policy requires a theorem-backed breach")
        return self


class HypothesisCandidate(StrictModel):
    experiment: ExperimentSpec
    evidence_uses: tuple[EvidenceUse, ...] = Field(min_length=1)
    novelty: CategoricalNoveltyAudit
    diagnosis_lane: DiagnosisLane | None = None

    @model_validator(mode="after")
    def validate_evidence_citations(self) -> HypothesisCandidate:
        citations = set(self.experiment.citations)
        missing = {item.citation for item in self.evidence_uses} - citations
        if missing:
            raise ValueError(
                f"evidence uses must cite the experiment sources: {sorted(missing)}"
            )
        return self


class NextRunPriorityV1(StrictModel):
    """Evidence-linked steering; authority is kept distinct from speculation."""

    schema_version: Literal["NextRunPriorityV1"] = "NextRunPriorityV1"
    rank: int = Field(ge=1)
    area: PriorityArea
    hypothesis: str = Field(min_length=12)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    expected_information_gain: str = Field(min_length=8)
    authority: Literal[
        "lean_theorem",
        "lean_assumption",
        "reproduced_harness_signal",
        "observed_result",
        "speculative",
    ]
    disposition: Literal[
        "stop_campaign",
        "block_promotion",
        "repair_before_run",
        "experiment_next",
        "monitor",
    ]
    proposed_experiment_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
    )

    @model_validator(mode="after")
    def validate_authority(self) -> NextRunPriorityV1:
        if self.authority == "lean_theorem" and self.disposition != "stop_campaign":
            raise ValueError(
                "Lean theorem authority must stop the contradicted campaign"
            )
        if (
            self.disposition == "stop_campaign"
            and self.proposed_experiment_id is not None
        ):
            raise ValueError("stopped campaigns cannot nominate a training experiment")
        if (
            self.disposition == "experiment_next"
            and self.proposed_experiment_id is None
        ):
            raise ValueError("experiment_next priority requires an experiment id")
        return self


class HypothesisMatrix(StrictModel):
    matrix_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    campaign_id: str
    evidence_snapshot_id: str
    hypotheses: tuple[HypothesisCandidate, ...] = Field(min_length=5)
    recommended_experiment_id: str
    selection_rationale: str = Field(min_length=12)
    predecessor_matrix_id: str | None = None
    feedback_ids: tuple[str, ...] = ()
    next_run_priorities: tuple[NextRunPriorityV1, ...] = ()
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_distinct_hypotheses(self) -> HypothesisMatrix:
        experiments = [item.experiment for item in self.hypotheses]
        ids = [item.experiment_id for item in experiments]
        if len(ids) != len(set(ids)):
            raise ValueError("hypothesis matrix experiment ids must be unique")
        signatures = [
            json.dumps(
                experiment.knobs.model_dump(exclude_none=True, mode="json"),
                sort_keys=True,
            )
            for experiment in experiments
        ]
        if len(signatures) != len(set(signatures)):
            raise ValueError("hypothesis matrix knob signatures must be distinct")
        hypotheses = [experiment.hypothesis.casefold() for experiment in experiments]
        if len(hypotheses) != len(set(hypotheses)):
            raise ValueError("hypothesis matrix hypotheses must be distinct")
        if self.recommended_experiment_id not in ids:
            raise ValueError("recommended experiment must be a matrix member")
        if len(self.feedback_ids) != len(set(self.feedback_ids)):
            raise ValueError("hypothesis matrix feedback ids must be unique")
        if self.predecessor_matrix_id is not None and not self.feedback_ids:
            raise ValueError(
                "matrix with a predecessor_matrix_id must acknowledge feedback_ids"
            )
        if self.next_run_priorities:
            ordered = sorted(self.next_run_priorities, key=lambda item: item.rank)
            if [item.rank for item in ordered] != list(
                range(1, len(self.next_run_priorities) + 1)
            ):
                raise ValueError("next-run priority ranks must be contiguous from one")
            unknown = {
                item.proposed_experiment_id
                for item in ordered
                if item.proposed_experiment_id is not None
            } - set(ids)
            if unknown:
                raise ValueError(
                    f"next-run priorities reference unknown experiments: {sorted(unknown)}"
                )
            experiment_priorities = [
                item for item in ordered if item.disposition == "experiment_next"
            ]
            if (
                experiment_priorities
                and experiment_priorities[0].proposed_experiment_id
                != self.recommended_experiment_id
            ):
                raise ValueError(
                    "highest-ranked experiment priority must match the recommendation"
                )
        return self


class HypothesisFeedback(StrictModel):
    """Outcome evidence supplied to the next hypothesizer iteration."""

    feedback_id: str = Field(pattern=r"^feedback-[0-9a-f]{16}$")
    campaign_id: str
    matrix_id: str
    experiment_id: str
    hypothesis: str
    knob_signature: str
    outcome_status: Literal["completed", "failed", "stopped"]
    metrics: dict[str, float] = Field(default_factory=dict)
    data_metrics: dict[str, float] = Field(default_factory=dict)
    diagnosis_target: Literal[
        "data", "researcher", "model", "harness", "infrastructure", "none"
    ]
    harness_family: HarnessFamily | None = None
    diagnosis_evidence: tuple[str, ...]
    recommended_actions: tuple[str, ...]
    optimum_feedback: OptimumFeedbackV1 | None = None
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_harness_family(self) -> HypothesisFeedback:
        if self.diagnosis_target == "harness" and self.harness_family is None:
            raise ValueError("harness feedback requires harness_family")
        if self.diagnosis_target != "harness" and self.harness_family is not None:
            raise ValueError("harness_family is only valid for harness feedback")
        return self


class HarnessSignalV1(StrictModel):
    """Frozen-input evidence that a canonical harness, not a model arm, failed."""

    family: HarnessFamily
    code: str = Field(min_length=1)
    evidence_uri: str = Field(min_length=1)
    reproduced_on_frozen_input: bool
    primary: bool = False


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
    wall_time_budget_seconds: float | None = Field(default=None, gt=0)
    stage_telemetry: tuple[dict[str, Any], ...] = ()
    harness_signals: tuple[HarnessSignalV1, ...] = ()
    started_at: str | None = None
    finished_at: str | None = None
    campaign_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

    @model_validator(mode="after")
    def validate_harness_signals(self) -> ExperimentOutcome:
        if sum(signal.primary for signal in self.harness_signals) > 1:
            raise ValueError(
                "experiment outcome may name at most one primary harness signal"
            )
        return self


class Diagnosis(StrictModel):
    experiment_id: str
    target: Literal["data", "researcher", "model", "harness", "infrastructure", "none"]
    harness_family: HarnessFamily | None = None
    confidence: float = Field(ge=0, le=1)
    evidence: tuple[str, ...]
    recommended_actions: tuple[str, ...]
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_harness_family(self) -> Diagnosis:
        if self.target == "harness" and self.harness_family is None:
            raise ValueError("harness diagnosis requires harness_family")
        if self.target != "harness" and self.harness_family is not None:
            raise ValueError("harness_family is only valid for a harness diagnosis")
        return self


class RLReadinessReport(StrictModel):
    report_id: str
    evaluation_sha256: str
    frozen_snapshot: bool
    required_suites: tuple[str, ...]
    suite_sizes: dict[str, int]
    ship_gates_pass: bool
    eval_criteria_pass: bool
    reward_sample_count: int = Field(ge=0)
    reward_variance: float = Field(ge=0)
    approved: bool
    failures: tuple[str, ...] = ()
    evaluation_uri: str | None = None
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
    promotion_policy: Literal["human_review_v1", "automated_frozen_meta_gate_v1"] = (
        "human_review_v1"
    )
    human_approved: bool = False
    promotable: bool = False
    agentv: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)


class HypothesizerBenchmarkReport(StrictModel):
    benchmark_id: str
    hypothesizer_id: str
    cases: int = Field(ge=1)
    valid_matrix_rate: float = Field(ge=0, le=1)
    grounded_rate: float = Field(ge=0, le=1)
    novel_rate: float = Field(ge=0, le=1)
    actionable_rate: float = Field(ge=0, le=1)
    feedback_lineage_rate: float = Field(ge=0, le=1)
    pass_threshold: float = Field(ge=0, le=1)
    passed: bool
    promotion_policy: Literal["human_review_v1", "automated_frozen_meta_gate_v1"] = (
        "human_review_v1"
    )
    human_approved: bool = False
    promotable: bool = False
    agentv: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)
