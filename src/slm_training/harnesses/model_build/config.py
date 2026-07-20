"""Config for the model-building harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelBuildConfig:
    train_dir: Path
    test_dir: Path | None = None
    suite: str = "smoke"
    # Honesty label stamped into every eval payload (see evals.record_schema
    # RUN_CLASSES): fixture_demo | scratch_matrix | ship_eval.
    run_class: str = "scratch_matrix"
    run_root: Path = Path("outputs/runs")
    run_id: str = "latest"
    # None preserves legacy behavior; an explicit set limits checkpoint mutation.
    runtime_override_fields: frozenset[str] | None = None
    steps: int = 200
    # Cumulative training-harness deadline; all runs are hard-capped at 3 minutes.
    max_wall_minutes: float | None = 3.0
    batch_size: int = 4
    lr: float = 3e-4
    seed: int = 0
    device: str = "cpu"
    model_name: str = "twotower"  # twotower | grammar_diffusion | stub
    # TwoTower hyperparams
    d_model: int = 128
    n_heads: int = 4
    context_layers: int = 2
    denoiser_layers: int = 4
    mask_min: float = 0.15
    mask_max: float = 0.85
    gen_steps: int = 8
    # Prefer HF when available; tests/CI can pass --context-backend scratch.
    context_backend: str = "hf"  # scratch | hf
    hf_model_name: str = "HuggingFaceTB/SmolLM2-135M"
    hf_model_revision: str | None = None
    # False for scratch POC; True by default when context_backend=hf (see factory)
    freeze_context: bool = True
    local_files_only: bool = False
    # scratch | hf — B4: adapt the pretrained hf_model_name causal LM into the
    # (trainable) masked denoiser instead of the from-scratch DenoiserTower.
    denoiser_backend: str = "scratch"
    # stacked | shared_recursive — SLM-138 shared recursive denoiser tower.
    denoiser_arch: str = "stacked"
    # SLM-138: recurrence and transition-depth knobs for shared_recursive.
    recursive_steps: int = 1
    recursive_transition_layers: int = 0
    recursive_depth_supervision_weights: tuple[float, ...] = ()
    grammar_constrained: bool = True
    # Grammar / DSL backend id: openui | openui-lark | openui-langcore | toy-layout
    grammar_dsl: str = "openui"
    grammar_top_k: int = 16
    structural_bias: float = 1.25
    grammar_ltr_repair: bool = False
    # Length-safe for compositional tokenizer (fixture gold up to ~160 tokens).
    grammar_ltr_max_tokens: int = 256
    grammar_ltr_stages: tuple[int, ...] | None = None
    grammar_ltr_primary: bool = False
    grammar_finalize_validate: bool = False
    ltr_loss_weight: float = 0.5
    fidelity_loss_weight: float = 0.5
    # None = preserve checkpoint on load; factory defaults new models to True.
    design_md_in_context: bool | None = None
    # Deterministic record-level train-time omission; evaluation is unaffected.
    design_md_dropout: float = 0.0
    # Superfiltering-style difficulty evidence: after training, score every
    # train record's NLL under the final model and write record_nll.jsonl so
    # derived-data builds can weight curation by difficulty (opt-in; one
    # no-grad forward per record).
    emit_record_nll: bool = False
    design_md_budget: int = 1800
    schema_in_context: bool = False
    slot_contract_in_context: bool = False
    semantic_role_contract_in_context: bool = False
    slot_contract_constrained_decode: bool = False
    template_fill_decode: bool = False
    contract_template_fastpath: bool = False
    retrieval_k: int = 0
    best_of_n: int = 1
    use_curriculum: bool = False
    # Soft A/B/C mix (anti-leak); False restores hard stage cutovers.
    mix_curriculum: bool = True
    # P1b: optional mixture manifest (JSON) for online family-weighted sampling.
    mixture_manifest: Path | None = None
    mixture_min_quality_score: float = 0.0
    mixture_sampling_policy: str = "with_replacement"
    # SDE2-03 (SLM-170): exposure-targeted rare-action sampling knobs.
    mixture_exposure_target_profile: str | None = None
    mixture_total_decision_budget: int | None = None
    mixture_per_root_cap: int | None = None
    mixture_per_template_cap: int | None = None
    mixture_max_importance_weight: float | None = None
    # P1d: after base training, write promoted.pt from best_weighted_nll / last.
    register_promoted: bool = False
    # Stub-only
    noise_rate: float = 0.0
    # Eval-driven training: run suite eval every N steps (0 disables).
    eval_every: int = 0
    eval_suite: str = "smoke"
    # Deterministic denoising-NLL suites every N optimizer steps (0 disables).
    # Cheap teacher-forced signal — decoupled from the generated scoreboard.
    loss_eval_every: int = 0
    loss_suite_version: str = "v1"
    loss_mask_seed: int = 0
    # Stop when this many target tokens have been consumed (None = steps only).
    target_token_budget: int | None = None
    # Resume bit-exact from a full-state checkpoint (last_full_state.pt).
    resume_from: Path | None = None
    # Warm-start weights/tokenizers from a serving checkpoint while resetting
    # optimizer, RNG, step, and token counters for a new corpus or recipe.
    initialize_from: Path | None = None
    # Optional immutable parent corpus mixed into warm-start continuation batches.
    replay_train_dir: Path | None = None
    replay_fraction: float = 0.0
    # After each optimizer step, contract trainable weights toward their
    # initialize_from values (0 = off, 1 = exact retention).
    initialization_weight_retention: float = 0.0
    # Write full training state (optimizer/RNG/sampler) alongside last.pt.
    full_state_checkpoint: bool = True
    # Comma-separated suites for mid-train scoreboard (overrides single eval_suite when set).
    eval_suites: str = ""
    # Cap rico_held size during matrix / CPU evals (None = full suite).
    rico_eval_limit: int | None = None
    # Optional cap for every eval suite; diagnostic-only when explicitly set.
    eval_limit: int | None = None
    # Accelerator / throughput
    use_amp: bool = False
    use_compile: bool = False
    compile_mode: str = "default"
    grad_accum_steps: int = 1
    parallel_unmask: str = "adaptive"
    parallel_workers: int = 2
    remask_ratio: float = 0.0
    mdlm_schedule: bool = False
    mdlm_eps: float = 1e-3
    # Train-speed bundle (also set via --fast-train)
    cache_context: bool = True
    fuse_ltr_loss: bool = True
    grammar_fastpath: bool = True
    grammar_fastpath_mode: str = "hybrid"  # force | mask | hybrid
    grammar_draft_window: int = 8
    compiler_decode_mode: str = "off"  # off | forced | restricted | tree
    compiler_search_mode: str = "greedy"  # greedy | lattice | ptrm | gram
    compiler_search_trigger: str = "stagnation"  # bottom | stagnation | always
    compiler_search_width: int = 1
    compiler_search_noise: float = 0.0
    compiler_search_stagnation_patience: int = 2
    compiler_search_backtrack_limit: int = 8
    compiler_search_local_nogoods: bool = False
    # VSS1-03 certified-solver decode (disabled by default; decode-time only).
    verified_solver_decode: bool = False
    solver_max_nodes: int = 512
    solver_max_depth: int = 64
    solver_max_backtracks: int = 64
    solver_max_verifier_calls: int = 64
    solver_max_wall_ms: int = 0
    solver_unknown_policy: str = "keep_and_rank"
    solver_certificate_mode: str = "summary"
    decode_min_content: int = 0  # A4: 0 off | >0 floor | -1 auto-from-inventory
    asap_decode: bool = False  # A2: ASAp-style constraint-mass removal in MaskGIT
    fastpath_aux_weight: float = 0.0
    fastpath_gate_threshold: float = 0.5
    # V4 critic / remask levers
    honest_slot_contract: bool = False
    suffix_rollback_window: int = 0
    remask_use_gate: bool = False
    remask_use_entropy: bool = False
    remask_policy: str = "confidence"  # confidence | core | combined
    core_perturb_frac: float = 0.25
    remask_to_mask: bool = True
    slot_aware_trust_gate: bool = False
    visible_corrupt_rate: float = 0.0
    trust_gate_train: bool = False
    grammar_prefer_structural: bool = True
    grammar_trust_model: bool = False
    grammar_sample_decode: bool = False
    grammar_sample_temperature: float = 0.8
    grammar_block_decode: bool = False
    grammar_block_size: int = 32
    # Grammar-topology diffusion (format v2 production tree)
    block_size: int = 4
    production_loss_weight: float = 1.0
    slot_loss_weight: float = 0.5
    confidence_loss_weight: float = 0.25
    extendability_decode: bool = True
    # Grammar topology diffusion (X9-X15).
    topology_actions: bool = True
    topology_structural_embeddings: bool = True
    topology_heterogeneous_noise: bool = True
    topology_critic_decode: bool = True
    topology_bounded_buffer: bool = True
    topology_max_nodes: int = 256
    topology_max_active: int = 64
    topology_max_arity: int = 8
    topology_max_depth: int = 32
    topology_max_phases: int = 32
    topology_global_sync_interval: int = 4
    topology_accept_threshold: float = 0.5
    topology_contract_threshold: float = 0.25
    scope_contracts: bool = False
    scope_independent_noise: bool = False
    scope_local_oracle: bool = False
    scope_contract_negatives: bool = False
    # VSS3-03: topology finite-domain solver integration (disabled by default).
    topology_verified_solver: bool = False
    topology_capsule_solver: bool = False
    topology_solver_ranker: str = "model"  # deterministic | model | energy
    topology_solver_unknown_policy: str = "keep_and_rank"
    topology_solver_max_nodes: int = 256
    topology_solver_max_backtracks: int = 64
    topology_solver_max_verifier_calls: int = 64
    topology_solver_certificate_mode: str = "summary"
    topology_solver_local_oracle: bool = True
    topology_solver_global_verify: bool = True
    # Cycle telemetry (train/infer span JSON)
    telemetry: bool = True
    # V5: lexer-native output tokenizer + Stage-2 levers
    output_tokenizer: str = "compositional"  # compositional | lexer | choice
    use_symbol_table: bool = True
    # C1: absolute | relative (De Bruijn <BINDDEF>/<BINDREL_±k> binder channel)
    bind_encoding: str = "absolute"
    # C3: corpus-mined <MACRO_i> tokens with deterministic decode expansion.
    macro_tokens: bool = False
    # C4: False = surface binder/state identifiers (byte channel) instead of
    # the anonymized <BIND_j>/<STATE_k> pools; placeholders unaffected.
    symbol_anonymization: bool = True
    factorized_embeddings: bool = False
    mask_pattern: str = "random"  # random | mixed | diffusion
    statement_mask_prob: float = 0.35
    diffusion_policies: tuple[str, ...] = (
        "uniform",
        "contiguous",
        "statement",
        "ast_subtree",
        "reference",
        "edit_local",
        "disjoint",
        "all_mask",
        "expansion",
        "contraction",
        "reorder",
    )
    diffusion_length_buckets: tuple[int, ...] = (32, 64, 96, 128, 192, 256, 384, 512)
    diffusion_overallocate: int = 8
    diffusion_length_loss_weight: float = 0.1
    ltr_prefix_loss_weight: float = 0.0
    compiler_alignment_loss_weight: float = 0.0
    compiler_alignment_margin: float = 0.0
    compiler_alignment_stratified: bool = False
    compiler_alignment_semantic_exhaustive: bool = False
    # SLM-164: confusion-targeted legal-sibling contrast margin (default-off).
    legal_margin_mode: str = "none"
    targeted_margin_manifest: Path | None = None
    targeted_margin_value: float = 1.0
    targeted_margin_family_weights: tuple[tuple[str, float], ...] = ()
    component_inventory_loss_weight: float = 0.0
    component_inventory_decode_weight: float | None = None
    component_plan_loss_weight: float = 0.0
    component_plan_decode_weight: float | None = None
    slot_component_loss_weight: float = 0.0
    slot_component_focal_gamma: float = 0.0
    slot_component_class_balance_power: float = 0.0
    slot_component_owner_rare_threshold: int = 0
    slot_component_owner_rare_multiplier: int = 1
    slot_component_decode_weight: float | None = None
    semantic_role_decode_weight: float | None = None
    semantic_role_schema_candidates: bool | None = None
    slot_coverage_close_decode_weight: float | None = None
    schema_value_decode_weight: float | None = None
    schema_enum_close_decode_weight: float | None = None
    schema_opaque_decode_weight: float | None = None
    schema_opaque_close_decode_weight: float | None = None
    schema_role_slot_decode_weight: float | None = None
    semantic_plan_decode_weight: float | None = None
    semantic_plan_margin_decode_weight: float | None = None
    semantic_plan_seed_decode_weight: float | None = None
    semantic_plan_inline_decode_weight: float | None = None
    semantic_plan_binding_decode_weight: float | None = None
    semantic_plan_root_decode_weight: float | None = None
    semantic_plan_root_margin_decode_weight: float | None = None
    semantic_plan_repeated_array_close_margin_decode_weight: float | None = None
    semantic_plan_repeated_slot_margin_decode_weight: float | None = None
    visible_reference_decode_weight: float | None = None
    slot_component_prompt_context: bool = True
    slot_component_next_context: bool = False
    slot_component_pair_interaction: bool = False
    slot_component_lexeme_prior_weight: float = 0.0
    slot_component_span_prior_weight: float = 0.0
    slot_component_content_arity: bool = False
    component_edge_loss_weight: float = 0.0
    component_edge_alignment_loss_weight: float = 0.0
    component_edge_decode_weight: float | None = None
    binder_component_plan_loss_weight: float = 0.0
    binder_component_plan_decode_weight: float | None = None
    binder_topology_loss_weight: float = 0.0
    binder_topology_decode_weight: float | None = None
    binder_arity_loss_weight: float = 0.0
    binder_arity_decode_weight: float | None = None
    root_reference_arity_loss_weight: float = 0.0
    root_reference_arity_decode_weight: float | None = None
    root_reference_identity_loss_weight: float = 0.0
    root_reference_identity_negative_weight: float = 1.0
    root_reference_identity_strict_subset_multiplier: int = 1
    root_reference_identity_decode_weight: float | None = None
    symbol_boundary_loss_weight: float = 0.0
    remask_span: str = "token"  # token | statement
    teacher_init_embeddings: bool = False
    # SLM-163: action-embedding initialization source and trainability.
    action_embedding_init: str = "none"
    action_embedding_train: str = "frozen"
    # SLM-174 (SDE2-07): description-mediated generalization with anonymized
    # action aliases.  Default ``canonical`` preserves canonical names.
    action_alias_mode: str = "canonical"
    action_alias_manifest: Path | None = None
    action_description_name_mode: str = "schema"
    # SLM-166 (SDE1-04): semantic connector between frozen context encoder and
    # sparse grammar-action scorer.  ``none`` is identity and preserves behavior.
    semantic_connector: str = "none"  # none | linear | low_rank | cross_attention
    connector_hidden_dim: int = 256
    connector_rank: int = 32
    connector_n_queries: int = 4
    connector_freeze_encoder: bool = True
    # current | connector_only | connector_plus_action_residuals | small_model
    train_scope: str = "current"
    # SLM-168 (SDE2-01): explicit contract-index pointer head (default-off).
    pointer_mode: str = "legacy_tokens"  # legacy_tokens | dynamic_head
    pointer_candidate_source: str = "structured_contract"  # structured_contract | authored_only | inventory_in_prompt
    pointer_hidden_dim: int = 256
    pointer_heads: int = 4
    pointer_temperature: float = 1.0
    pointer_dropout: float = 0.0
    runtime_symbol_features: str = "none"  # none | surface | role_gated | replace (C2)
    symbol_slot_augmentation: bool = False
    semantic_candidate_masks: bool = False
    constraint_graph_mode: str = "off"  # off | grammar | hybrid
    grammar_completion_bounds: bool = False
    grammar_equivalence_cache: bool = False
    grammar_active_symbol_bitsets: bool = False
    compact_active_canvas: bool = True
    # Inference-speed levers (P/Q/R-series)
    grammar_incremental_state: bool = True
    grammar_verify_chosen_only: bool = False
    grammar_skip_exact_stream_probe: bool = True
    grammar_copy_probes: bool = True
    grammar_early_exit_pick: bool = True
    grammar_multitoken_accept: bool = False
    grammar_multitoken_max: int = 8
    grammar_canvas_lookahead: int = 0
    use_dynamic_quant: bool = False
    quant_format: str | None = None  # CAP3-01: disabled-by-default reference quantizer
    # CAP3-05: optional target byte budget for equal-byte ladder points.
    byte_budget: int | None = None
    generate_max_attempts: int = 3
    # Diagnostic per-record generation timeout; None/0 preserves unlimited eval.
    decode_timeout_seconds: float | None = None
    grammar_finalize_on_last_attempt_only: bool = False
    allow_unconstrained_fallback: bool = True
    # V7 speculative denoising (docs/design/speculative-denoising.md)
    stability_min_persistence: int = 0  # E70 commit gate (0=off)
    stability_jsd_weight: float = 1.0  # E70 remask score mix
    unmask_mode: str = "positions"  # positions | cluster (E71)
    cluster_attn_threshold: float = 0.08
    cluster_max_size: int = 4
    cluster_verify: bool = False  # E72 ordered cluster verification
    survival_gate: bool = False  # E73 decode-time survival scheduling
    survival_gate_train: bool = False  # E73 head-training stage
    survival_commit_threshold: float = 0.3
    speculative_successor: bool = False  # E74 successor-state cache
    speculative_fanout: int = 2
    speculative_overlap: bool = False
    # Hugging Face Bucket for durable checkpoints (full HF-context trains).
    # None → default hf://buckets/TKendrick/OpenUI when sync is enabled.
    # Empty string → disable auto bucket selection.
    checkpoint_bucket: str | None = None
    # Off by default for programmatic/tests. `scripts.train_model` enables this
    # for HF-context full trains (see resolve_sync_checkpoints / CLI flags).
    # None = auto (HF backend → on) for callers that still use the old sentinel.
    sync_checkpoints: bool | None = False
    # Plan-only sync (no upload) — for wiring tests / agents without write auth.
    checkpoint_bucket_dry_run: bool = False
    # VSS3-05: optional constrained autoregressive surface realizer.
    # All fields are default-off; deterministic realization remains the baseline.
    surface_realizer: str = "deterministic"  # deterministic | autoregressive
    surface_ar_enabled: bool = False
    surface_ar_d_model: int = 64
    surface_ar_n_layers: int = 2
    surface_ar_n_heads: int = 2
    surface_ar_max_bytes: int = 64
    surface_ar_temperature: float = 0.0
    surface_ar_top_k: int = 1
    surface_ar_fallback: str = "deterministic"
    surface_ar_verify_retry: bool = True
    # SDE3-01: content-addressed eval caching and deterministic sharding.
    eval_cache_mode: str = "off"  # off | read | read_write | refresh
    eval_cache_root: Path = field(default_factory=lambda: Path("outputs/eval_cache"))
    eval_shards: int = 1
    # LDI2-01: optional removable TwoTower low-rank adapter directory.
    adapter_spec: Path | None = None
    adapter_trainable: bool = True

    @property
    def run_dir(self) -> Path:
        return self.run_root / self.run_id

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def train_records(self) -> Path:
        return self.train_dir / "records.jsonl"
