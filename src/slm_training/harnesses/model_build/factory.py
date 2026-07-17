"""Factory for model plug-ins."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.plugin import ModelPlugin, StubModel

if TYPE_CHECKING:
    from slm_training.models.twotower import TwoTowerConfig


def _resolve_freeze_context(backend: str, requested: bool) -> bool:
    """Honor the caller's freeze choice for every context backend."""
    _ = backend
    return bool(requested)


def apply_runtime_overrides(model: Any, config: ModelBuildConfig) -> Any:
    """Apply eval/train decode + conditioning overrides onto a loaded plugin."""
    # Activate the selected grammar / DSL backend for constrained decode.
    try:
        from slm_training.models.grammar import set_active_dsl

        set_active_dsl(getattr(config, "grammar_dsl", None) or "openui")
    except Exception:  # noqa: BLE001
        pass
    cfg = getattr(model, "config", None)
    if cfg is None:
        return model
    allowed = config.runtime_override_fields
    for key in (
        "grammar_constrained",
        "grammar_top_k",
        "structural_bias",
        "grammar_ltr_repair",
        "grammar_ltr_max_tokens",
        "grammar_ltr_stages",
        "grammar_ltr_primary",
        "grammar_finalize_validate",
        "design_md_dropout",
        "design_md_budget",
        "schema_in_context",
        "slot_contract_in_context",
        "slot_contract_constrained_decode",
        "template_fill_decode",
        "contract_template_fastpath",
        "honest_slot_contract",
        "retrieval_k",
        "best_of_n",
        "fidelity_loss_weight",
        "ltr_loss_weight",
        "gen_steps",
        "parallel_unmask",
        "remask_ratio",
        "mdlm_schedule",
        "mdlm_eps",
        "cache_context",
        "fuse_ltr_loss",
        "grammar_fastpath",
        "grammar_fastpath_mode",
        "grammar_draft_window",
        "compiler_decode_mode",
        "compiler_search_mode",
        "compiler_search_trigger",
        "compiler_search_width",
        "compiler_search_noise",
        "compiler_search_stagnation_patience",
        "compiler_search_backtrack_limit",
        "compiler_search_local_nogoods",
        "decode_min_content",
        "asap_decode",
        "fastpath_aux_weight",
        "fastpath_gate_threshold",
        "suffix_rollback_window",
        "remask_use_gate",
        "remask_use_entropy",
        "remask_policy",
        "core_perturb_frac",
        "remask_to_mask",
        "slot_aware_trust_gate",
        "visible_corrupt_rate",
        "trust_gate_train",
        "grammar_prefer_structural",
        "grammar_trust_model",
        "grammar_sample_decode",
        "grammar_sample_temperature",
        "grammar_block_decode",
        "grammar_block_size",
        "use_amp",
        "use_compile",
        "compile_mode",
        "grammar_dsl",
        "gen_steps",
        "output_tokenizer",
        "use_symbol_table",
        "bind_encoding",
        "macro_tokens",
        "symbol_anonymization",
        "factorized_embeddings",
        "mask_pattern",
        "statement_mask_prob",
        "diffusion_policies",
        "diffusion_length_buckets",
        "diffusion_overallocate",
        "diffusion_length_loss_weight",
        "component_inventory_decode_weight",
        "component_plan_decode_weight",
        "slot_component_decode_weight",
        "component_edge_decode_weight",
        "binder_component_plan_decode_weight",
        "binder_topology_decode_weight",
        "binder_arity_decode_weight",
        "remask_span",
        "teacher_init_embeddings",
        "runtime_symbol_features",
        "symbol_slot_augmentation",
        "semantic_candidate_masks",
        "constraint_graph_mode",
        "grammar_completion_bounds",
        "grammar_equivalence_cache",
        "grammar_active_symbol_bitsets",
        "compact_active_canvas",
        "block_size",
        "production_loss_weight",
        "slot_loss_weight",
        "confidence_loss_weight",
        "grammar_incremental_state",
        "grammar_verify_chosen_only",
        "grammar_skip_exact_stream_probe",
        "grammar_copy_probes",
        "grammar_early_exit_pick",
        "grammar_multitoken_accept",
        "grammar_multitoken_max",
        "grammar_canvas_lookahead",
        "use_dynamic_quant",
        "generate_max_attempts",
        "grammar_finalize_on_last_attempt_only",
        "allow_unconstrained_fallback",
        "stability_min_persistence",
        "stability_jsd_weight",
        "unmask_mode",
        "cluster_attn_threshold",
        "cluster_max_size",
        "cluster_verify",
        "survival_gate",
        "survival_gate_train",
        "survival_commit_threshold",
        "speculative_successor",
        "speculative_fanout",
        "speculative_overlap",
    ):
        if allowed is not None and key not in allowed:
            continue
        if hasattr(config, key) and hasattr(cfg, key):
            value = getattr(config, key)
            if value is not None:
                setattr(cfg, key, value)
    # Preserve checkpoint DESIGN.md conditioning unless caller sets an explicit bool.
    # Eval defaults must not force-enable gold DESIGN.md on no-design-md checkpoints.
    dm = getattr(config, "design_md_in_context", None)
    if allowed is not None and "design_md_in_context" not in allowed:
        dm = None
    if dm is not None and hasattr(cfg, "design_md_in_context"):
        cfg.design_md_in_context = bool(dm)
    # Decode quality defaults often wanted at eval time.
    if getattr(config, "grammar_ltr_repair", False) and hasattr(
        cfg, "grammar_ltr_repair"
    ):
        cfg.grammar_ltr_repair = True
    if (
        str(getattr(config, "compiler_decode_mode", "off") or "off") != "off"
        and hasattr(cfg, "grammar_ltr_primary")
    ):
        cfg.grammar_ltr_primary = True
    if int(getattr(config, "best_of_n", 1) or 1) > 1 and hasattr(cfg, "best_of_n"):
        cfg.best_of_n = int(config.best_of_n)
    if bool(getattr(config, "use_dynamic_quant", False)) and hasattr(
        model, "apply_dynamic_quant"
    ):
        try:
            model.apply_dynamic_quant()
        except Exception:  # noqa: BLE001
            pass
    if bool(getattr(config, "use_compile", False)) and hasattr(model, "denoiser"):
        try:
            from slm_training.runtime.accel import maybe_compile

            model.denoiser = maybe_compile(
                model.denoiser,
                enabled=True,
                mode=str(getattr(config, "compile_mode", "default") or "default"),
            )
        except Exception:  # noqa: BLE001
            pass
    if int(getattr(config, "retrieval_k", 0) or 0) > 0 and hasattr(
        model, "skeleton_bank"
    ):
        try:
            from slm_training.harnesses.model_build.data import load_train_records
            from slm_training.harnesses.quality import build_skeleton_bank

            if config.train_dir.exists():
                model.skeleton_bank = build_skeleton_bank(
                    load_train_records(config.train_dir)
                )
        except Exception:  # noqa: BLE001
            pass
    return model


def _twotower_config_from_build(config: ModelBuildConfig) -> "TwoTowerConfig":
    from slm_training.models.twotower import TwoTowerConfig

    backend = (config.context_backend or "scratch").lower()
    freeze = _resolve_freeze_context(backend, config.freeze_context)
    ltr_stages = getattr(config, "grammar_ltr_stages", None)
    if ltr_stages is None:
        ltr_stages = (64, 128, 192, 256)
    return TwoTowerConfig(
        d_model=config.d_model,
        n_heads=config.n_heads,
        context_layers=config.context_layers,
        denoiser_layers=config.denoiser_layers,
        mask_min=config.mask_min,
        mask_max=config.mask_max,
        gen_steps=config.gen_steps,
        context_backend=backend,
        hf_model_name=config.hf_model_name,
        hf_model_revision=config.hf_model_revision,
        freeze_context=freeze,
        local_files_only=config.local_files_only,
        denoiser_backend=config.denoiser_backend,
        grammar_constrained=config.grammar_constrained,
        grammar_top_k=config.grammar_top_k,
        structural_bias=config.structural_bias,
        grammar_ltr_repair=config.grammar_ltr_repair,
        grammar_ltr_max_tokens=config.grammar_ltr_max_tokens,
        grammar_ltr_stages=tuple(ltr_stages),
        grammar_finalize_validate=getattr(config, "grammar_finalize_validate", False),
        ltr_loss_weight=getattr(config, "ltr_loss_weight", 0.5),
        fidelity_loss_weight=getattr(config, "fidelity_loss_weight", 0.0),
        grammar_ltr_primary=config.grammar_ltr_primary,
        design_md_in_context=(
            True
            if config.design_md_in_context is None
            else bool(config.design_md_in_context)
        ),
        design_md_dropout=float(getattr(config, "design_md_dropout", 0.0) or 0.0),
        design_md_budget=config.design_md_budget,
        schema_in_context=getattr(config, "schema_in_context", False),
        slot_contract_in_context=getattr(config, "slot_contract_in_context", False),
        slot_contract_constrained_decode=getattr(
            config, "slot_contract_constrained_decode", False
        ),
        template_fill_decode=getattr(config, "template_fill_decode", False),
        honest_slot_contract=getattr(config, "honest_slot_contract", False),
        retrieval_k=getattr(config, "retrieval_k", 0),
        best_of_n=getattr(config, "best_of_n", 1),
        parallel_unmask=getattr(config, "parallel_unmask", "adaptive"),
        remask_ratio=float(getattr(config, "remask_ratio", 0.0) or 0.0),
        remask_use_gate=bool(getattr(config, "remask_use_gate", False)),
        remask_use_entropy=bool(getattr(config, "remask_use_entropy", False)),
        remask_policy=str(
            getattr(config, "remask_policy", "confidence") or "confidence"
        ),
        core_perturb_frac=float(getattr(config, "core_perturb_frac", 0.25) or 0.25),
        remask_to_mask=bool(getattr(config, "remask_to_mask", True)),
        slot_aware_trust_gate=bool(getattr(config, "slot_aware_trust_gate", False)),
        mdlm_schedule=bool(getattr(config, "mdlm_schedule", False)),
        mdlm_eps=float(getattr(config, "mdlm_eps", 1e-3) or 1e-3),
        visible_corrupt_rate=float(getattr(config, "visible_corrupt_rate", 0.0) or 0.0),
        suffix_rollback_window=int(getattr(config, "suffix_rollback_window", 0) or 0),
        use_compile=getattr(config, "use_compile", False),
        compile_mode=getattr(config, "compile_mode", "default"),
        use_amp=getattr(config, "use_amp", False),
        cache_context=getattr(config, "cache_context", True),
        fuse_ltr_loss=getattr(config, "fuse_ltr_loss", True),
        grammar_fastpath=getattr(config, "grammar_fastpath", True),
        grammar_fastpath_mode=getattr(config, "grammar_fastpath_mode", "hybrid"),
        grammar_draft_window=int(getattr(config, "grammar_draft_window", 8) or 8),
        compiler_decode_mode=str(
            getattr(config, "compiler_decode_mode", "off") or "off"
        ),
        compiler_search_mode=str(getattr(config, "compiler_search_mode", "greedy") or "greedy"),
        compiler_search_trigger=str(getattr(config, "compiler_search_trigger", "stagnation") or "stagnation"),
        compiler_search_width=max(1, int(getattr(config, "compiler_search_width", 1) or 1)),
        compiler_search_noise=max(0.0, float(getattr(config, "compiler_search_noise", 0.0) or 0.0)),
        compiler_search_stagnation_patience=max(1, int(getattr(config, "compiler_search_stagnation_patience", 2) or 2)),
        compiler_search_backtrack_limit=max(0, int(getattr(config, "compiler_search_backtrack_limit", 8) or 0)),
        compiler_search_local_nogoods=bool(
            getattr(config, "compiler_search_local_nogoods", False)
        ),
        decode_min_content=max(-1, int(getattr(config, "decode_min_content", 0) or 0)),
        asap_decode=bool(getattr(config, "asap_decode", False)),
        fastpath_aux_weight=getattr(config, "fastpath_aux_weight", 0.0),
        fastpath_gate_threshold=float(
            getattr(config, "fastpath_gate_threshold", 0.5) or 0.5
        ),
        trust_gate_train=bool(getattr(config, "trust_gate_train", False)),
        grammar_prefer_structural=getattr(config, "grammar_prefer_structural", True),
        grammar_trust_model=getattr(config, "grammar_trust_model", False),
        grammar_sample_decode=getattr(config, "grammar_sample_decode", False),
        grammar_sample_temperature=getattr(config, "grammar_sample_temperature", 0.8),
        grammar_block_decode=getattr(config, "grammar_block_decode", False),
        grammar_block_size=getattr(config, "grammar_block_size", 32),
        output_tokenizer=getattr(config, "output_tokenizer", "compositional"),
        use_symbol_table=getattr(config, "use_symbol_table", True),
        bind_encoding=str(getattr(config, "bind_encoding", "absolute") or "absolute"),
        macro_tokens=bool(getattr(config, "macro_tokens", False)),
        symbol_anonymization=bool(getattr(config, "symbol_anonymization", True)),
        factorized_embeddings=getattr(config, "factorized_embeddings", False),
        mask_pattern=getattr(config, "mask_pattern", "random"),
        statement_mask_prob=float(getattr(config, "statement_mask_prob", 0.35) or 0.35),
        diffusion_policies=tuple(getattr(config, "diffusion_policies", ()) or ()),
        diffusion_length_buckets=tuple(
            getattr(config, "diffusion_length_buckets", ()) or ()
        ),
        diffusion_overallocate=int(getattr(config, "diffusion_overallocate", 8) or 8),
        diffusion_length_loss_weight=float(
            getattr(config, "diffusion_length_loss_weight", 0.1) or 0.0
        ),
        ltr_prefix_loss_weight=float(
            getattr(config, "ltr_prefix_loss_weight", 0.0) or 0.0
        ),
        compiler_alignment_loss_weight=float(
            getattr(config, "compiler_alignment_loss_weight", 0.0) or 0.0
        ),
        compiler_alignment_margin=float(
            getattr(config, "compiler_alignment_margin", 0.0) or 0.0
        ),
        compiler_alignment_stratified=bool(
            getattr(config, "compiler_alignment_stratified", False)
        ),
        compiler_alignment_semantic_exhaustive=bool(
            getattr(config, "compiler_alignment_semantic_exhaustive", False)
        ),
        component_inventory_loss_weight=float(
            getattr(config, "component_inventory_loss_weight", 0.0) or 0.0
        ),
        component_inventory_decode_weight=float(
            getattr(config, "component_inventory_decode_weight", 0.0) or 0.0
        ),
        component_plan_loss_weight=float(
            getattr(config, "component_plan_loss_weight", 0.0) or 0.0
        ),
        component_plan_decode_weight=float(
            getattr(config, "component_plan_decode_weight", 0.0) or 0.0
        ),
        component_plan_attention_pool=bool(
            getattr(config, "component_plan_attention_pool", False)
        ),
        component_plan_token_pool=bool(
            getattr(config, "component_plan_token_pool", False)
        ),
        slot_component_loss_weight=float(
            getattr(config, "slot_component_loss_weight", 0.0) or 0.0
        ),
        slot_component_decode_weight=float(
            getattr(config, "slot_component_decode_weight", 0.0) or 0.0
        ),
        component_edge_loss_weight=float(
            getattr(config, "component_edge_loss_weight", 0.0) or 0.0
        ),
        component_edge_alignment_loss_weight=float(
            getattr(config, "component_edge_alignment_loss_weight", 0.0) or 0.0
        ),
        component_edge_decode_weight=float(
            getattr(config, "component_edge_decode_weight", 0.0) or 0.0
        ),
        binder_component_plan_loss_weight=float(
            getattr(config, "binder_component_plan_loss_weight", 0.0) or 0.0
        ),
        binder_component_plan_decode_weight=float(
            getattr(config, "binder_component_plan_decode_weight", 0.0) or 0.0
        ),
        binder_topology_loss_weight=float(
            getattr(config, "binder_topology_loss_weight", 0.0) or 0.0
        ),
        binder_topology_decode_weight=float(
            getattr(config, "binder_topology_decode_weight", 0.0) or 0.0
        ),
        binder_arity_loss_weight=float(
            getattr(config, "binder_arity_loss_weight", 0.0) or 0.0
        ),
        binder_arity_decode_weight=float(
            getattr(config, "binder_arity_decode_weight", 0.0) or 0.0
        ),
        symbol_boundary_loss_weight=float(
            getattr(config, "symbol_boundary_loss_weight", 0.0) or 0.0
        ),
        remask_span=getattr(config, "remask_span", "token"),
        teacher_init_embeddings=getattr(config, "teacher_init_embeddings", False),
        runtime_symbol_features=getattr(config, "runtime_symbol_features", "none"),
        symbol_slot_augmentation=bool(
            getattr(config, "symbol_slot_augmentation", False)
        ),
        semantic_candidate_masks=bool(
            getattr(config, "semantic_candidate_masks", False)
        ),
        constraint_graph_mode=str(
            getattr(config, "constraint_graph_mode", "off") or "off"
        ),
        grammar_completion_bounds=bool(
            getattr(config, "grammar_completion_bounds", False)
        ),
        grammar_equivalence_cache=bool(
            getattr(config, "grammar_equivalence_cache", False)
        ),
        grammar_active_symbol_bitsets=bool(
            getattr(config, "grammar_active_symbol_bitsets", False)
        ),
        compact_active_canvas=bool(getattr(config, "compact_active_canvas", True)),
        grammar_incremental_state=getattr(config, "grammar_incremental_state", True),
        grammar_verify_chosen_only=getattr(config, "grammar_verify_chosen_only", False),
        grammar_skip_exact_stream_probe=getattr(
            config, "grammar_skip_exact_stream_probe", True
        ),
        grammar_copy_probes=getattr(config, "grammar_copy_probes", True),
        grammar_early_exit_pick=getattr(config, "grammar_early_exit_pick", True),
        grammar_multitoken_accept=getattr(config, "grammar_multitoken_accept", False),
        grammar_multitoken_max=int(getattr(config, "grammar_multitoken_max", 8) or 8),
        grammar_canvas_lookahead=int(
            getattr(config, "grammar_canvas_lookahead", 0) or 0
        ),
        use_dynamic_quant=bool(getattr(config, "use_dynamic_quant", False)),
        generate_max_attempts=int(getattr(config, "generate_max_attempts", 3) or 3),
        grammar_finalize_on_last_attempt_only=bool(
            getattr(config, "grammar_finalize_on_last_attempt_only", False)
        ),
        allow_unconstrained_fallback=bool(
            getattr(config, "allow_unconstrained_fallback", True)
        ),
        stability_min_persistence=int(
            getattr(config, "stability_min_persistence", 0) or 0
        ),
        stability_jsd_weight=float(getattr(config, "stability_jsd_weight", 1.0) or 1.0),
        unmask_mode=str(getattr(config, "unmask_mode", "positions") or "positions"),
        cluster_attn_threshold=float(
            getattr(config, "cluster_attn_threshold", 0.08) or 0.08
        ),
        cluster_max_size=int(getattr(config, "cluster_max_size", 4) or 4),
        cluster_verify=bool(getattr(config, "cluster_verify", False)),
        survival_gate=bool(getattr(config, "survival_gate", False)),
        survival_gate_train=bool(getattr(config, "survival_gate_train", False)),
        survival_commit_threshold=float(
            getattr(config, "survival_commit_threshold", 0.3) or 0.3
        ),
        speculative_successor=bool(getattr(config, "speculative_successor", False)),
        speculative_fanout=int(getattr(config, "speculative_fanout", 2) or 2),
        speculative_overlap=bool(getattr(config, "speculative_overlap", False)),
        seed=config.seed,
    )


def build_model(
    config: ModelBuildConfig,
    records: list[ExampleRecord],
    checkpoint: Path | None = None,
) -> Any:
    try:
        from slm_training.models.grammar import set_active_dsl

        set_active_dsl(getattr(config, "grammar_dsl", None) or "openui")
    except Exception:  # noqa: BLE001
        pass
    name = (config.model_name or "twotower").lower()
    if name == "stub":
        model: ModelPlugin = StubModel(noise_rate=config.noise_rate, seed=config.seed)
        if checkpoint and checkpoint.exists():
            model.load(checkpoint)
        return model

    if name in {"tree_edit_diffusion", "tree-edit-diffusion"}:
        # D3 (SLM-31): faithful Kapur-style all-valid-states baseline (X22).
        from slm_training.models.tree_edit_diffusion import (
            TreeEditDiffusionConfig,
            TreeEditDiffusionModel,
        )

        if checkpoint and checkpoint.exists():
            return TreeEditDiffusionModel.from_checkpoint(
                checkpoint, device=config.device
            )
        backend = (config.context_backend or "scratch").lower()
        ted_cfg = TreeEditDiffusionConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            context_layers=config.context_layers,
            denoiser_layers=config.denoiser_layers,
            context_backend=backend,
            hf_model_name=config.hf_model_name,
            freeze_context=_resolve_freeze_context(backend, config.freeze_context),
            local_files_only=config.local_files_only,
            design_md_in_context=(
                False
                if config.design_md_in_context is None
                else bool(config.design_md_in_context)
            ),
            design_md_budget=config.design_md_budget,
            slot_contract_in_context=getattr(
                config, "slot_contract_in_context", True
            ),
            seed=config.seed,
        )
        return TreeEditDiffusionModel.from_records(
            records, config=ted_cfg, device=config.device
        )

    if name in {"grammar_diffusion", "grammar-diffusion", "grammardiffusion"}:
        from slm_training.models.grammar_diffusion import (
            GrammarDiffusionConfig,
            GrammarDiffusionModel,
        )

        if checkpoint and checkpoint.exists():
            loaded = GrammarDiffusionModel.from_checkpoint(
                checkpoint, device=config.device
            )
            return apply_runtime_overrides(loaded, config)

        backend = (config.context_backend or "scratch").lower()
        freeze = _resolve_freeze_context(backend, config.freeze_context)
        gd_cfg = GrammarDiffusionConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            context_layers=config.context_layers,
            denoiser_layers=config.denoiser_layers,
            block_size=getattr(config, "block_size", 4),
            mask_min=config.mask_min,
            mask_max=config.mask_max,
            gen_steps=config.gen_steps,
            context_backend=backend,
            hf_model_name=config.hf_model_name,
            freeze_context=freeze,
            local_files_only=config.local_files_only,
            grammar_dsl=getattr(config, "grammar_dsl", "openui"),
            parallel_unmask=getattr(config, "parallel_unmask", "adaptive"),
            production_loss_weight=getattr(config, "production_loss_weight", 1.0),
            slot_loss_weight=getattr(config, "slot_loss_weight", 0.5),
            confidence_loss_weight=getattr(config, "confidence_loss_weight", 0.25),
            topology_actions=bool(getattr(config, "topology_actions", True)),
            topology_structural_embeddings=bool(
                getattr(config, "topology_structural_embeddings", True)
            ),
            topology_heterogeneous_noise=bool(
                getattr(config, "topology_heterogeneous_noise", True)
            ),
            topology_critic_decode=bool(
                getattr(config, "topology_critic_decode", True)
            ),
            topology_bounded_buffer=bool(
                getattr(config, "topology_bounded_buffer", True)
            ),
            topology_max_nodes=int(getattr(config, "topology_max_nodes", 256)),
            topology_max_active=int(getattr(config, "topology_max_active", 64)),
            topology_max_arity=int(getattr(config, "topology_max_arity", 8)),
            topology_max_depth=int(getattr(config, "topology_max_depth", 32)),
            topology_max_phases=int(getattr(config, "topology_max_phases", 32)),
            topology_global_sync_interval=int(
                getattr(config, "topology_global_sync_interval", 4)
            ),
            topology_accept_threshold=float(
                getattr(config, "topology_accept_threshold", 0.5)
            ),
            topology_contract_threshold=float(
                getattr(config, "topology_contract_threshold", 0.25)
            ),
            scope_contracts=bool(getattr(config, "scope_contracts", False)),
            scope_independent_noise=bool(
                getattr(config, "scope_independent_noise", False)
            ),
            scope_local_oracle=bool(getattr(config, "scope_local_oracle", False)),
            scope_contract_negatives=bool(
                getattr(config, "scope_contract_negatives", False)
            ),
            design_md_in_context=(
                False
                if config.design_md_in_context is None
                else bool(config.design_md_in_context)
            ),
            design_md_budget=config.design_md_budget,
            schema_in_context=getattr(config, "schema_in_context", False),
            slot_contract_in_context=getattr(config, "slot_contract_in_context", True),
            slot_contract_constrained_decode=getattr(
                config, "slot_contract_constrained_decode", True
            ),
            honest_slot_contract=bool(getattr(config, "honest_slot_contract", True)),
            seed=config.seed,
        )
        return GrammarDiffusionModel.from_records(
            records, config=gd_cfg, device=config.device
        )

    if name in {"twotower", "two_tower", "two-tower"}:
        from slm_training.models.twotower import TwoTowerModel

        if checkpoint and checkpoint.exists():
            loaded = TwoTowerModel.from_checkpoint(
                checkpoint,
                device=config.device,
                local_files_only=config.local_files_only,
            )
            return apply_runtime_overrides(loaded, config)

        tt_cfg = _twotower_config_from_build(config)
        model = TwoTowerModel.from_records(records, config=tt_cfg, device=config.device)
        if int(getattr(config, "retrieval_k", 0) or 0) > 0:
            from slm_training.harnesses.quality import build_skeleton_bank

            model.skeleton_bank = build_skeleton_bank(records)
        return model

    raise ValueError(f"unknown model_name {config.model_name!r}")
