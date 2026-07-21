"""TwoTower OpenUI model: context encoder + trainable masked denoiser."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm_training.dsl.schema import ExampleRecord
from slm_training.data.contract import RuntimeSymbol
from slm_training.harnesses.model_build.plugin import GenerationRequest
from slm_training.models.blocks import DenoiserTower
from slm_training.models.recursive_denoiser import SharedRecursiveDenoiserTower
from slm_training.models.twotower_schedule_policy import (
    validate_twotower_numeric_schedule,
)
from slm_training.models.context import (
    HFContextEncoder,
    ScratchContextEncoder,
    build_context_encoder,
    is_hf_context,
)
from slm_training.models.decode_stats import (
    DecodeStats,
    collect_decode_stats,
    get_active_stats,
    timed_ms,
)
from slm_training.models.grammar import (
    apply_structural_bias,
    filter_ids_by_stream,
    force_emit_token_id,
    make_grammar_state,
    pick_constrained_token,
    stream_check,
)
from slm_training.models.parallel_decode import (
    AsapLedger,
    StabilityTracker,
    core_instability_scores,
    perturb_known_neighbors,
    select_remask_core_indices,
    select_remask_coverage_indices,
    select_remask_indices,
    select_remask_policy_indices,
    select_remask_stability_indices,
    select_unmask_indices,
)
from slm_training.models.speculative_denoise import (
    SpeculativeStats,
    SuccessorCache,
    build_constraint_edges,
    build_dependency_clusters,
    enumerate_outcome_canvases,
    filter_by_cumulative_survival,
    order_clusters,
    survival_commit_budget,
    verify_clusters_ordered,
)
from slm_training.models.template_fill import (
    build_slot_contract_template,
    ensure_prompt_inventory,
    ensure_prompt_semantic_roles,
    inventory_from_prompt,
    prompt_semantic_plan,
    prompt_semantic_role_candidates,
    template_mask_positions,
)
from slm_training.dsl.action_descriptions import FixtureDescriptionEncoder
from slm_training.dsl.action_shortlist import (
    ActionShortlistPolicy,
    ActionShortlistTrace,
    build_query_vector,
    retrieve_then_rerank,
)
from slm_training.dsl.grammar.fastpath.gate import FastPathGate
from slm_training.models.tokenizer import OpenUITokenizer, tokenize_text

# _repair_surface_syntax runs per generated candidate; precompile its patterns.
_QUOTED_SPAN_RE = re.compile(r'("(?:\\.|[^"\\])*")')
_REPEATED_EQUALS_RE = re.compile(r"\s*=\s*=+\s*")
_DANGLING_EQUALS_RE = re.compile(r",\s*=\s*(?=[)\]])")


def _is_lexer_output(config: "TwoTowerConfig | None") -> bool:
    if config is None:
        return False
    return str(getattr(config, "output_tokenizer", "compositional") or "").lower() in {
        "lexer",
        "dsl",
        "dsl_native",
        "native",
    }


def _is_choice_output(config: "TwoTowerConfig | None") -> bool:
    if config is None:
        return False
    return str(getattr(config, "output_tokenizer", "compositional") or "").lower() in {
        "choice",
        "choices",
        "choice_codec",
    }


def _load_any_tokenizer(path: Path | str):
    """Load compositional, lexer-native, or choice-codec tokenizer from sidecar."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("kind") == "choice_codec":
        from slm_training.models.choice_tokenizer import ChoiceTokenizer

        return ChoiceTokenizer.load(path)
    if raw.get("kind") == "dsl_native" or "id_to_kind" in raw:
        from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

        return DSLNativeTokenizer.load(path)
    return OpenUITokenizer.load(path)


def format_context_text(
    prompt: str,
    design_md: str | None = None,
    *,
    budget: int = 1800,
    schema: str | None = None,
    retrieved_skeleton: str | None = None,
    slot_contract: list[str] | None = None,
    output_kind: str | None = None,
    output_category: str | None = None,
) -> str:
    """Concatenate prompt with optional schema / skeleton / slot contract / DESIGN.md."""
    prompt = (prompt or "").strip()
    parts = [prompt] if prompt else []
    if output_kind is not None:
        output_contract = output_kind
        if output_category:
            output_contract = f"{output_contract}:{output_category}"
        parts.append(f"---OUTPUT_CONTRACT---\n{output_contract}")
    if schema and schema.strip():
        parts.append(f"---SCHEMA---\n{schema.strip()[: min(600, budget)]}")
    if retrieved_skeleton and retrieved_skeleton.strip():
        parts.append(
            f"---RETRIEVED_SKELETON---\n{retrieved_skeleton.strip()[: min(400, budget)]}"
        )
    if slot_contract:
        slots = ", ".join(slot_contract)
        parts.append(f"---SLOT_CONTRACT---\n{slots[: min(800, budget)]}")
    if design_md and design_md.strip():
        dm = design_md.strip()
        if len(dm) > budget:
            dm = dm[:budget].rsplit("\n", 1)[0]
        parts.append(f"---DESIGN.md---\n{dm}")
    return "\n\n".join(parts) if parts else prompt


@dataclass
class TwoTowerConfig:
    d_model: int = 128
    n_heads: int = 4
    context_layers: int = 2
    denoiser_layers: int = 4
    max_prompt_len: int = 256
    max_target_len: int = 256
    dropout: float = 0.0
    mask_min: float = 0.15
    mask_max: float = 0.85
    gen_steps: int = 8
    # scratch | hf — ModelBuildConfig / CLI default to hf for production runs.
    context_backend: str = "scratch"
    # Default production HF tower; tests may override with a tiny model.
    hf_model_name: str = "HuggingFaceTB/SmolLM2-135M"
    hf_model_revision: str | None = None
    # True when using a pretrained HF context tower; optional for scratch.
    freeze_context: bool = False
    local_files_only: bool = False
    # scratch | hf — B4 DiffuLLaMA-style adaptation: reuse the pretrained
    # hf_model_name causal LM as a bidirectional masked denoiser backbone.
    denoiser_backend: str = "scratch"
    # stacked | shared_recursive — SLM-138 shared recursive denoiser tower.
    denoiser_arch: str = "stacked"
    # SLM-138: number of recurrences for the shared recursive denoiser.
    recursive_steps: int = 1
    # SLM-138: number of shared TransformerBlocks in the recursive transition.
    # 0 means inherit from denoiser_layers.
    recursive_transition_layers: int = 0
    # SLM-138: per-recursion CE weights for deep supervision (empty = off).
    recursive_depth_supervision_weights: tuple[float, ...] = ()
    # SLM-211: default-on weight tying between token embedding and output head.
    # Default True preserves exact prior behavior; False creates an independent
    # output projection initialized as a copy of the token embedding.
    tie_output_embedding: bool = True
    grammar_constrained: bool = True
    grammar_top_k: int = 16
    structural_bias: float = 1.25
    # Full LTR constrained repair is accurate but slow (Node stream_check per token).
    # Off by default; enable for final quality evals.
    grammar_ltr_repair: bool = False
    # Length-safe for compositional tokenizer (fixture gold up to ~160).
    grammar_ltr_max_tokens: int = 256
    # Progressive LTR canvases (short first). Cap must cover gold programs.
    grammar_ltr_stages: tuple[int, ...] = (64, 128, 192, 256)
    # Finalize LTR text with Node validate (adds ~1–2ms). Off by default —
    # eval already validates via meaningful-parse.
    grammar_finalize_validate: bool = False
    # When False, pick highest-scoring legal token (DINGO-style); when True,
    # prefer structural/component tokens among legal candidates (legacy).
    grammar_prefer_structural: bool = True
    # Trust-the-model decode: disable structural logit bias and structural reorder.
    grammar_trust_model: bool = False
    # Sample from renormalized legal distribution instead of greedy argmax.
    grammar_sample_decode: bool = False
    grammar_sample_temperature: float = 0.8
    # Semi-AR block decode: fill contiguous spans left-to-right (block diffusion).
    grammar_block_decode: bool = False
    grammar_block_size: int = 32
    # Eval / throughput: batch size for generate_batch.
    generate_batch_size: int = 16
    # When True and grammar_constrained, skip MaskGIT and decode LTR only.
    grammar_ltr_primary: bool = False
    # Mix teacher-forced next-token CE into training (helps LTR generate).
    ltr_loss_weight: float = 0.5
    # Extra weight on the first content transitions (root -> assignment).
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
    # Prompt-level multi-label component inventory derived from gold token kinds.
    component_inventory_loss_weight: float = 0.0
    # Bias only compiler-legal component candidates with the learned inventory.
    component_inventory_decode_weight: float = 0.0
    # Grammar-role plan: root class plus bound-component multiplicities.
    component_plan_loss_weight: float = 0.0
    component_plan_decode_weight: float = 0.0
    slot_component_loss_weight: float = 0.0
    slot_component_focal_gamma: float = 0.0
    slot_component_class_balance_power: float = 0.0
    slot_component_class_weights: tuple[float, ...] = ()
    slot_component_decode_weight: float = 0.0
    semantic_role_decode_weight: float = 0.0
    semantic_role_schema_candidates: bool = False
    slot_coverage_close_decode_weight: float = 0.0
    schema_value_decode_weight: float = 0.0
    schema_enum_close_decode_weight: float = 0.0
    schema_opaque_decode_weight: float = 0.0
    schema_opaque_close_decode_weight: float = 0.0
    schema_role_slot_decode_weight: float = 0.0
    semantic_plan_decode_weight: float = 0.0
    semantic_plan_margin_decode_weight: float = 0.0
    semantic_plan_seed_decode_weight: float = 0.0
    semantic_plan_inline_decode_weight: float = 0.0
    semantic_plan_binding_decode_weight: float = 0.0
    semantic_plan_root_decode_weight: float = 0.0
    semantic_plan_root_margin_decode_weight: float = 0.0
    semantic_plan_repeated_array_close_margin_decode_weight: float = 0.0
    semantic_plan_repeated_slot_margin_decode_weight: float = 0.0
    semantic_plan_typed_array_nonempty_margin_decode_weight: float = 0.0
    semantic_plan_typed_array_item_margin_decode_weight: float = 0.0
    visible_reference_decode_weight: float = 0.0
    slot_component_prompt_context: bool = True
    slot_component_next_context: bool = False
    slot_component_pair_interaction: bool = False
    slot_component_lexeme_prior_weight: float = 0.0
    slot_component_lexeme_priors: tuple[tuple[str, tuple[float, ...]], ...] = ()
    slot_component_span_prior_weight: float = 0.0
    slot_component_span_priors: tuple[tuple[str, tuple[float, ...]], ...] = ()
    slot_component_content_arity: bool = False
    component_edge_loss_weight: float = 0.0
    component_edge_alignment_loss_weight: float = 0.0
    component_edge_decode_weight: float = 0.0
    binder_component_plan_loss_weight: float = 0.0
    binder_component_plan_decode_weight: float = 0.0
    binder_topology_loss_weight: float = 0.0
    binder_topology_decode_weight: float = 0.0
    binder_arity_loss_weight: float = 0.0
    binder_arity_decode_weight: float = 0.0
    root_reference_arity_loss_weight: float = 0.0
    root_reference_arity_decode_weight: float = 0.0
    root_reference_identity_loss_weight: float = 0.0
    root_reference_identity_negative_weight: float = 1.0
    root_reference_identity_decode_weight: float = 0.0
    symbol_boundary_loss_weight: float = 0.0
    # Extra CE weight on gold placeholder token positions (fidelity signal).
    fidelity_loss_weight: float = 0.5
    design_md_in_context: bool = True
    # Static by (seed, record key) so context caching remains sound.
    design_md_dropout: float = 0.0
    design_md_budget: int = 1800
    schema_in_context: bool = False
    slot_contract_in_context: bool = False
    semantic_role_contract_in_context: bool = False
    slot_contract_constrained_decode: bool = False
    # E20: seed decode from a slot-contract skeleton (inventory-bound template).
    template_fill_decode: bool = False
    contract_template_fastpath: bool = False
    # E35: derive slot inventory from prompt/DESIGN.md (never gold.placeholders).
    honest_slot_contract: bool = False
    retrieval_k: int = 0
    best_of_n: int = 1
    seed: int = 0
    # Accelerator / SOTA decode knobs
    use_compile: bool = False
    compile_mode: str = "default"
    use_amp: bool = False
    # MaskGIT parallel unmask: topk | confidence | adaptive (mean-field-lite)
    parallel_unmask: str = "adaptive"
    # E22: remask lowest-confidence committed tokens each MaskGIT step (0=off).
    remask_ratio: float = 0.0
    # E33: combine grammar + gate + entropy into remask budget.
    remask_use_gate: bool = False
    remask_use_entropy: bool = False
    # V6 remask policy: confidence | core | combined (CoRe-lite + E33).
    remask_policy: str = "confidence"
    # Fraction of known neighbors masked for CoRe perturbation forward (E50).
    core_perturb_frac: float = 0.25
    # E51/T2M: always remask→mask (never token-edit). Kept True by design.
    remask_to_mask: bool = True
    # E52: trust-gate mining also labels placeholder/slot binding errors.
    slot_aware_trust_gate: bool = False
    # E21: MDLM-faithful continuous-time absorbing mask + 1/t CE weights.
    mdlm_schedule: bool = False
    mdlm_eps: float = 1e-3
    # E32: flip a fraction of visible (non-masked) tokens to wrong ids.
    visible_corrupt_rate: float = 0.0
    # E30: revisable LTR suffix window (0=off). Remask+redo on grammar/entropy.
    suffix_rollback_window: int = 0
    # Train-speed: cache frozen HF backbone hiddens + formatted context strings.
    cache_context: bool = True
    # Fuse LTR suffix masks into the MaskGIT canvas (one denoiser forward).
    fuse_ltr_loss: bool = True
    # Grammar fast-path (decode); aux weight applied in training_loss when >0.
    grammar_fastpath: bool = True
    grammar_fastpath_mode: str = "hybrid"  # force | mask | hybrid
    grammar_draft_window: int = 8
    # Compiler-drafted decode: off | forced | restricted | tree.
    # Decode-only; ``off`` preserves existing checkpoint behavior.
    compiler_decode_mode: str = "off"
    # Search over compiler-valid branches: greedy | lattice | ptrm | gram.
    compiler_search_mode: str = "greedy"
    compiler_search_trigger: str = "stagnation"  # bottom | stagnation | always
    compiler_search_width: int = 1
    compiler_search_noise: float = 0.0
    compiler_search_stagnation_patience: int = 2
    compiler_search_backtrack_limit: int = 8
    compiler_search_local_nogoods: bool = False
    # VSS1-03 certified-solver decode: exact closure prunes the compiler forest to
    # the certificate-checked live subset before soft ranking. Disabled by default;
    # ``False`` is byte-identical to existing decode. Deterministic budgets are
    # authoritative (wall timer is advisory only).
    verified_solver_decode: bool = False
    solver_max_nodes: int = 512
    solver_max_depth: int = 64
    solver_max_backtracks: int = 64
    solver_max_verifier_calls: int = 64
    solver_max_wall_ms: int = 0
    solver_unknown_policy: str = "keep_and_rank"
    solver_certificate_mode: str = "summary"  # none | summary | full
    # VSS3-02 learned cost-to-go energy scorer. Ranking-only: it orders the exact
    # live candidates and never alters hard membership, certificates, or UNKNOWN.
    # Disabled by default; ``solver_ranker="deterministic"`` is byte-identical to
    # existing decode. Old configs/checkpoints missing these load with defaults.
    solver_energy_head: bool = False
    solver_ranker: str = "deterministic"  # deterministic | model | energy
    solver_energy_hidden_dim: int = 64
    solver_energy_loss_weight: float = 0.0
    solver_energy_pairwise_weight: float = 0.0
    solver_energy_cost_version: str = "v1"
    solver_energy_fallback: str = "deterministic"
    # A4 minimum-content decode contract (compiler-tree decode only):
    #   0  -> off (empty layouts remain legal completions);
    #   >0 -> require at least this many components before EOS is admitted;
    #   -1 -> auto: derive the floor from the resolved slot-contract inventory.
    decode_min_content: int = 0
    # A2: ASAp-style distribution-aware constrained MaskGIT decode. Observed
    # constraint violations (admit rejects, grammar stream remasks) remove the
    # violating token's probability mass at that canvas position from the next
    # proposal, and unmask ordering uses the post-removal confidence.
    asap_decode: bool = False
    fastpath_aux_weight: float = 0.0
    fastpath_gate_threshold: float = 0.5
    # E31: train/use FastPathGate trust head for remask.
    trust_gate_train: bool = False
    # V5: output-side representation
    # compositional = legacy OpenUITokenizer v2; lexer = DSLNativeTokenizer
    # compositional | lexer | choice (B1 pure grammar-choice stream)
    output_tokenizer: str = "compositional"
    # When output_tokenizer=lexer: map placeholders to <SYM_i> (E41+).
    use_symbol_table: bool = True
    # C1: absolute (<BIND_j>) | relative (<BINDDEF>/<BINDREL_±k> De Bruijn refs).
    bind_encoding: str = "absolute"
    # C3 (SLM-27): mine a deterministic per-corpus macro table at build time
    # and encode targets with <MACRO_i> tokens (lossless decode-time splice).
    macro_tokens: bool = False
    # C4 (SLM-28): False = surface arm of the names-disappear comparison —
    # binder/state names ride the byte channel verbatim instead of the
    # <BIND_j>/<STATE_k> pools. Placeholders are unaffected by this flag.
    symbol_anonymization: bool = True
    # Stage-2: kind-factorized embeddings (E_tok + E_kind).
    factorized_embeddings: bool = False
    # Stage-2 training mask: random | mixed (statement spans ∪ random).
    mask_pattern: str = "random"
    # Probability of statement-span masking when mask_pattern=mixed.
    statement_mask_prob: float = 0.35
    # SLM-14: online structure-aware corruption + context length prediction.
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
    # Remask expansion: token | statement
    remask_span: str = "token"
    # Optional teacher-init of symbol embeddings from HF context (E45).
    teacher_init_embeddings: bool = False
    # SLM-163: action-embedding initialization source (none | current_stub |
    # schema_description | expanded_description | shuffled).
    action_embedding_init: str = "none"
    # SLM-163: how to treat action embeddings during training (frozen | trainable).
    action_embedding_train: str = "frozen"
    # SLM-174 (SDE2-07): action alias mode for description-mediated
    # generalization.  ``off`` / ``canonical`` preserves canonical names;
    # ``fixed`` uses a single deterministic alias map; ``held_out`` trains on
    # one alias map and evaluates on a fresh map.
    action_alias_mode: str = "canonical"
    # Optional path to a persisted alias manifest (JSON).  When None the alias
    # map is built deterministically from seed + pack hash.
    action_alias_manifest: Path | None = None
    # How canonical names are rendered in description sources:
    # schema | alias_aware_description | alias_aware_signature_only |
    # alias_aware_shuffled | description_without_canonical_name |
    # canonical_name_plus_description | signature_only.
    action_description_name_mode: str = "schema"
    # SLM-176 (P14): description-based retrieve-then-rerank over complete live
    # legal action sets.  ``off`` preserves canonical decode; other values are
    # wiring placeholders for future trained-controller integration.
    action_shortlist_mode: str = "off"  # off | description_retrieval
    action_shortlist_k: int = 8
    action_shortlist_min_legal_size: int = 16
    action_shortlist_score_margin: float = 0.0
    action_shortlist_fallback_policy: str = "confidence_and_coverage"
    action_shortlist_shadow_full_score: bool = False
    # SLM-166 (SDE1-04): semantic connector between frozen context encoder and
    # sparse grammar-action scorer.  ``none`` is identity; other values select a
    # standalone connector variant for future wiring.
    semantic_connector: str = "none"  # none | linear | low_rank | cross_attention
    connector_hidden_dim: int = 256
    connector_rank: int = 32
    connector_n_queries: int = 4
    connector_freeze_encoder: bool = True
    # SLM-166: trainable scope when a connector is present.
    # current | connector_only | connector_plus_action_residuals | small_model
    train_scope: str = "current"
    # SLM-168 (SDE2-01): explicit contract-index pointer head (default-off).
    # legacy_tokens preserves existing behavior; dynamic_head enables the scorer.
    pointer_mode: str = "legacy_tokens"  # legacy_tokens | dynamic_head
    pointer_candidate_source: str = "structured_contract"  # structured_contract | authored_only | inventory_in_prompt
    pointer_hidden_dim: int = 256
    pointer_heads: int = 4
    pointer_temperature: float = 1.0
    pointer_dropout: float = 0.0
    # V8 request-conditioned dynamic vocabulary; ``none`` is checkpoint-identical.
    # none | surface | role_gated | replace (C2: dynamic pseudo-embeddings —
    # symbol rows become deterministic byte-compositional vectors).
    runtime_symbol_features: str = "none"
    symbol_slot_augmentation: bool = False
    semantic_candidate_masks: bool = False
    constraint_graph_mode: str = "off"  # off | grammar | hybrid
    grammar_completion_bounds: bool = False
    grammar_equivalence_cache: bool = False
    grammar_active_symbol_bitsets: bool = False
    compact_active_canvas: bool = True
    # --- Inference-speed levers (P/Q/R-series; decode-only, no retrain) ---
    # P1: reuse one DFA engine + decoded prefix text per decode row.
    grammar_incremental_state: bool = True
    # P2: probe model argmax first; skip stream probes on exact DFA terminals.
    grammar_verify_chosen_only: bool = False
    grammar_skip_exact_stream_probe: bool = True
    # Q1: InteractiveParser.copy()-based DFA admit probes.
    grammar_copy_probes: bool = True
    # Q2: early-exit descending-logit candidate scoring in pick_constrained_token.
    grammar_early_exit_pick: bool = True
    # P3: accept a run of consecutive grammar-legal argmax tokens per forward.
    grammar_multitoken_accept: bool = False
    grammar_multitoken_max: int = 8
    # P4: run denoiser on prefix + K mask lookahead instead of full canvas.
    grammar_canvas_lookahead: int = 0  # 0 = disabled (use progressive stages)
    # P5: dynamic int8 quantization of Linear layers at eval time.
    use_dynamic_quant: bool = False
    # CAP3-01: reference low-bit quantizer format id (None = disabled).
    quant_format: str | None = None
    # P7: playground/generate attempt budget + finalize-only-on-last.
    generate_max_attempts: int = 3
    grammar_finalize_on_last_attempt_only: bool = False
    allow_unconstrained_fallback: bool = True
    # --- V7 speculative denoising (docs/design/speculative-denoising.md) ---
    # E70 LESS-lite: require argmax persistence before committing (0=off);
    # remask_policy="stability" ranks remasks by persistence + inter-step JSD.
    stability_min_persistence: int = 0
    stability_jsd_weight: float = 1.0
    # E71 DAPD/DAWN-lite: positions (classic) | cluster (attention clusters).
    unmask_mode: str = "positions"
    cluster_attn_threshold: float = 0.08
    cluster_max_size: int = 4
    # E72: verify clusters in anchor order; outcome (j, repair).
    cluster_verify: bool = False
    # E73 DSpark-lite: survival head at decode + cumulative commit budget.
    survival_gate: bool = False
    survival_gate_train: bool = False
    survival_commit_threshold: float = 0.3
    # E74 SSD-lite: batched successor-state cache over top-K verifier outcomes.
    speculative_successor: bool = False
    speculative_fanout: int = 2
    speculative_overlap: bool = False

    def __post_init__(self) -> None:
        # RSC-A06 (SLM-242): fail-closed gate for every numeric weight/schedule
        # vector this config carries. Runs on every construction, including
        # ``from_checkpoint`` (raw config dict -> TwoTowerConfig(**kwargs)) and
        # ``from_records``. ``apply_runtime_overrides`` and ``load`` mutate
        # fields after construction and re-run this explicitly (see call sites).
        validate_twotower_numeric_schedule(self)


def _pad_batch(
    seqs: list[list[int]], pad_id: int, device: str | torch.device | None = None
) -> torch.Tensor:
    max_len = max((len(s) for s in seqs), default=1)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        if s:
            out[i, : len(s)] = torch.as_tensor(s, dtype=torch.long)
    if device is not None:
        out = out.to(device)
    return out


def _truncate_with_eos(ids: list[int], max_len: int, eos_id: int) -> list[int]:
    """Truncate a target without dropping its termination token."""
    if max_len <= 0:
        return []
    if len(ids) <= max_len:
        return list(ids)
    out = list(ids[:max_len])
    out[-1] = eos_id
    return out


def _load_checkpoint_state(
    model: nn.Module, state_dict: dict[str, torch.Tensor]
) -> None:
    """Load a checkpoint while rejecting silent trainable-weight mismatches."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    config = getattr(model, "config", None)
    allowed_missing: set[str] = set()
    if (
        config is not None
        and bool(getattr(config, "freeze_context", False))
        and is_hf_context(getattr(model, "context", None))
    ):
        allowed_missing = {
            key for key in missing if key.startswith("context.backbone.")
        }
    # V4 FastPathGate is a plug-in head: older checkpoints omit it and keep the
    # randomly initialized gate until BackPlay-lite training (E31) runs.
    allowed_missing |= {key for key in missing if key.startswith("trust_gate.")}
    # V7 survival head is likewise a plug-in (trained via survival_train, E73).
    allowed_missing |= {key for key in missing if key.startswith("survival_head.")}
    allowed_missing |= {
        key for key in missing if key.startswith("root_reference_arity_head.")
    }
    allowed_missing |= {
        key for key in missing if key.startswith("root_reference_identity_head.")
    }
    # SLM-138: a shared-recursive denoiser adds z-state parameters that older
    # stacked checkpoints legitimately omit; warm-start them randomly.
    if getattr(config, "denoiser_arch", None) == "shared_recursive":
        allowed_missing |= {
            key
            for key in missing
            if key.startswith("denoiser.")
            and key.split(".")[1] in {"z_latent", "ctx_proj"}
        }
    bad_missing = sorted(set(missing) - allowed_missing)
    # V5 may have checkpointed a zero kind_lookup even when unused; the
    # non-factorized path now uses a non-persistent stub, so treat that legacy
    # key as ignorable when the live module does not require it.
    allowed_unexpected = {
        key
        for key in unexpected
        if key == "denoiser.kind_lookup"
        and getattr(getattr(model, "denoiser", None), "kind", None) is None
    }
    bad_unexpected = sorted(set(unexpected) - allowed_unexpected)
    if bad_missing or bad_unexpected:
        raise ValueError(
            "checkpoint state mismatch: "
            f"missing={bad_missing!r} unexpected={bad_unexpected!r}"
        )


def _check_output_head_tie_migration(
    model: "TwoTowerModel",
    source_config: dict[str, Any],
    *,
    allow_tie_migration: bool = False,
) -> None:
    """Detect tie_output_embedding mismatch and optionally apply a merge policy.

    Default behavior is fail-closed: loading a tied checkpoint into an untied
    model (or vice versa) raises a clear error. Passing
    ``allow_tie_migration=True`` opts in to a deterministic merge:

    * tied -> untied: the checkpoint's shared matrix is copied into both the
      token embedding and the new independent lm_head;
    * untied -> tied: the lm_head is discarded and tied to the token embedding.
    """
    target_tie = bool(getattr(model.config, "tie_output_embedding", True))
    source_tie = bool(source_config.get("tie_output_embedding", True))
    if source_tie == target_tie:
        return
    if not allow_tie_migration:
        raise ValueError(
            "checkpoint tie_output_embedding mismatch: "
            f"checkpoint={source_tie}, model={target_tie}. "
            "Pass allow_tie_migration=True to explicitly copy or tie the output head."
        )
    if not source_tie and target_tie:
        # Merge policy: keep the token embedding, discard the separate lm_head.
        model.denoiser.lm_head.weight = model.denoiser.tok.weight


class TwoTowerModel(nn.Module):
    """MaskGIT-style discrete diffusion conditioned on a prompt encoder."""

    def __init__(
        self,
        tokenizer: OpenUITokenizer,
        config: TwoTowerConfig | None = None,
        device: str | torch.device = "cpu",
        *,
        context_tokenizer: OpenUITokenizer | None = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.context_tokenizer = context_tokenizer or tokenizer
        self.config = config or TwoTowerConfig()
        if not 0.0 <= float(self.config.design_md_dropout) <= 1.0:
            raise ValueError("design_md_dropout must be between 0 and 1")
        self.output_contract_version = 1
        self.device_name = str(device)
        # Seed before module construction so a configured run is reproducible.
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        backend = (self.config.context_backend or "scratch").lower()
        freeze = self.config.freeze_context
        if backend in {"hf", "huggingface", "transformers"} and not freeze:
            # Explicit unfreeze allowed; factory typically sets freeze_context=True for HF.
            freeze = False

        self.context = build_context_encoder(
            backend=backend,
            vocab_size=self.context_tokenizer.vocab_size,
            d_model=self.config.d_model,
            n_layers=self.config.context_layers,
            n_heads=self.config.n_heads,
            max_len=self.config.max_prompt_len,
            dropout=self.config.dropout,
            freeze=freeze,
            hf_model_name=self.config.hf_model_name,
            hf_model_revision=self.config.hf_model_revision,
            local_files_only=self.config.local_files_only,
        )
        kind_ids = None
        if bool(getattr(self.config, "factorized_embeddings", False)):
            try:
                from slm_training.models.dsl_tokenizer import is_dsl_native_tokenizer

                if is_dsl_native_tokenizer(tokenizer):
                    # Map TokenKind -> small int for embedding table.
                    kind_name_to_idx = {
                        "special": 0,
                        "struct": 1,
                        "component": 2,
                        "sym": 3,
                        "bind": 4,
                        "lit": 5,
                        "byte": 6,
                        "builtin": 7,
                        "state": 8,
                    }
                    kind_ids = [
                        kind_name_to_idx.get(tokenizer.id_to_kind.get(i, "special"), 0)
                        for i in range(tokenizer.vocab_size)
                    ]
            except Exception:  # noqa: BLE001
                kind_ids = None
        denoiser_backend = str(
            getattr(self.config, "denoiser_backend", "scratch") or "scratch"
        ).lower()
        if denoiser_backend in {"hf", "huggingface", "transformers"}:
            from slm_training.models.hf_denoiser import HFDenoiserTower

            self.denoiser: nn.Module = HFDenoiserTower(
                vocab_size=tokenizer.vocab_size,
                d_model=self.config.d_model,
                max_len=self.config.max_target_len,
                hf_model_name=self.config.hf_model_name,
                hf_model_revision=self.config.hf_model_revision,
                local_files_only=self.config.local_files_only,
                kind_ids=kind_ids,
                n_kinds=max(kind_ids) + 1 if kind_ids else 0,
                tie_output_embedding=bool(self.config.tie_output_embedding),
            )
        elif denoiser_backend in {"scratch", "token", "local"}:
            denoiser_arch = str(
                getattr(self.config, "denoiser_arch", "stacked")
            ).lower()
            if denoiser_arch == "shared_recursive":
                transition_layers = int(
                    getattr(self.config, "recursive_transition_layers", 0) or 0
                )
                if transition_layers <= 0:
                    transition_layers = self.config.denoiser_layers
                self.denoiser: nn.Module = SharedRecursiveDenoiserTower(
                    vocab_size=tokenizer.vocab_size,
                    d_model=self.config.d_model,
                    n_layers=self.config.denoiser_layers,
                    n_heads=self.config.n_heads,
                    max_len=self.config.max_target_len,
                    dropout=self.config.dropout,
                    kind_ids=kind_ids,
                    n_kinds=max(kind_ids) + 1 if kind_ids else 0,
                    recursive_steps=int(
                        getattr(self.config, "recursive_steps", 1) or 1
                    ),
                    recursive_transition_layers=transition_layers,
                    tie_output_embedding=bool(self.config.tie_output_embedding),
                )
            else:
                self.denoiser = DenoiserTower(
                    vocab_size=tokenizer.vocab_size,
                    d_model=self.config.d_model,
                    n_layers=self.config.denoiser_layers,
                    n_heads=self.config.n_heads,
                    max_len=self.config.max_target_len,
                    dropout=self.config.dropout,
                    kind_ids=kind_ids,
                    n_kinds=max(kind_ids) + 1 if kind_ids else 0,
                    tie_output_embedding=bool(self.config.tie_output_embedding),
                )
        else:
            raise ValueError(f"unknown denoiser_backend {denoiser_backend!r}")

        def isolated_aux_init(factory, offset: int):
            """Initialize an auxiliary module without shifting train RNG."""
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(self.config.seed) + offset)
                return factory()

        self.length_head = (
            isolated_aux_init(
                lambda: nn.Linear(
                    self.config.d_model,
                    len(self.config.diffusion_length_buckets),
                ),
                101,
            )
            if str(getattr(self.config, "mask_pattern", "random")) == "diffusion"
            else None
        )
        inventory_enabled = (
            float(getattr(self.config, "component_inventory_loss_weight", 0.0) or 0.0)
            > 0.0
            or float(
                getattr(self.config, "component_inventory_decode_weight", 0.0) or 0.0
            )
            > 0.0
        )
        self.component_inventory_head = (
            isolated_aux_init(
                lambda: nn.Linear(self.config.d_model, tokenizer.vocab_size),
                102,
            )
            if inventory_enabled
            else None
        )
        plan_enabled = (
            float(getattr(self.config, "component_plan_loss_weight", 0.0) or 0.0) > 0.0
            or float(getattr(self.config, "component_plan_decode_weight", 0.0) or 0.0)
            > 0.0
        )
        self.component_plan_head = (
            isolated_aux_init(
                lambda: nn.Linear(self.config.d_model, 2 * tokenizer.vocab_size),
                103,
            )
            if plan_enabled
            else None
        )
        try:
            component_edge_ids = tuple(sorted(tokenizer.kind_ids("component")))
        except (AttributeError, TypeError, ValueError):
            component_edge_ids = ()
        slot_component_enabled = (
            float(getattr(self.config, "slot_component_loss_weight", 0.0) or 0.0) > 0.0
            or float(getattr(self.config, "slot_component_decode_weight", 0.0) or 0.0)
            > 0.0
        )
        self.slot_component_head = (
            isolated_aux_init(
                lambda: nn.Linear(self.config.d_model, len(component_edge_ids)),
                114,
            )
            if slot_component_enabled and component_edge_ids
            else None
        )
        edge_enabled = (
            float(getattr(self.config, "component_edge_loss_weight", 0.0) or 0.0) > 0.0
            or float(
                getattr(self.config, "component_edge_alignment_loss_weight", 0.0) or 0.0
            )
            > 0.0
            or float(getattr(self.config, "component_edge_decode_weight", 0.0) or 0.0)
            > 0.0
        )
        self.component_edge_head = (
            isolated_aux_init(
                lambda: nn.Linear(
                    self.config.d_model,
                    len(component_edge_ids) * len(component_edge_ids),
                ),
                104,
            )
            if edge_enabled and component_edge_ids
            else None
        )
        try:
            binder_plan_ids = tuple(sorted(tokenizer.kind_ids("bind")))
        except (AttributeError, TypeError, ValueError):
            binder_plan_ids = ()
        binder_plan_enabled = (
            float(getattr(self.config, "binder_component_plan_loss_weight", 0.0) or 0.0)
            > 0.0
            or float(
                getattr(self.config, "binder_component_plan_decode_weight", 0.0) or 0.0
            )
            > 0.0
        )
        self.binder_component_plan_head = (
            isolated_aux_init(
                lambda: nn.Linear(
                    self.config.d_model,
                    len(binder_plan_ids) * len(component_edge_ids),
                ),
                105,
            )
            if binder_plan_enabled and binder_plan_ids and component_edge_ids
            else None
        )
        binder_topology_enabled = (
            float(getattr(self.config, "binder_topology_loss_weight", 0.0) or 0.0) > 0.0
            or float(getattr(self.config, "binder_topology_decode_weight", 0.0) or 0.0)
            > 0.0
        )
        self.binder_topology_head = (
            isolated_aux_init(
                lambda: nn.Linear(
                    self.config.d_model,
                    len(binder_plan_ids) * len(binder_plan_ids),
                ),
                106,
            )
            if binder_topology_enabled and binder_plan_ids
            else None
        )
        binder_arity_enabled = (
            float(getattr(self.config, "binder_arity_loss_weight", 0.0) or 0.0) > 0.0
            or float(getattr(self.config, "binder_arity_decode_weight", 0.0) or 0.0)
            > 0.0
        )
        self.binder_arity_head = (
            isolated_aux_init(
                lambda: nn.Linear(
                    self.config.d_model,
                    len(binder_plan_ids) * (len(binder_plan_ids) + 1),
                ),
                107,
            )
            if binder_arity_enabled and binder_plan_ids
            else None
        )
        root_reference_arity_enabled = (
            float(getattr(self.config, "root_reference_arity_loss_weight", 0.0) or 0.0)
            > 0.0
            or float(
                getattr(self.config, "root_reference_arity_decode_weight", 0.0) or 0.0
            )
            > 0.0
        )
        self.root_reference_arity_head = (
            isolated_aux_init(
                lambda: nn.Linear(
                    self.config.d_model,
                    int(getattr(tokenizer, "ref_slots", 64)) + 1,
                ),
                108,
            )
            if root_reference_arity_enabled and _is_choice_output(self.config)
            else None
        )
        root_reference_identity_enabled = (
            float(
                getattr(self.config, "root_reference_identity_loss_weight", 0.0) or 0.0
            )
            > 0.0
            or float(
                getattr(self.config, "root_reference_identity_decode_weight", 0.0)
                or 0.0
            )
            > 0.0
        )
        self.root_reference_identity_head = (
            isolated_aux_init(
                lambda: nn.Linear(
                    self.config.d_model,
                    int(getattr(tokenizer, "ref_slots", 64)),
                ),
                110,
            )
            if root_reference_identity_enabled and _is_choice_output(self.config)
            else None
        )
        # E31 BackPlay-lite: plug-in trust head over denoiser hiddens.
        self.trust_gate = isolated_aux_init(
            lambda: FastPathGate(self.config.d_model), 108
        )
        # E73 DSpark-lite: plug-in trajectory-survival head (V7).
        self.survival_head = isolated_aux_init(
            lambda: FastPathGate(self.config.d_model), 109
        )
        # V7 decode telemetry (MaskGIT path): forwards, successor hits/misses.
        self.speculative_stats = SpeculativeStats()
        self._rng = random.Random(self.config.seed)
        self.gen_len = self.config.max_target_len
        # Optional decode trajectory recorder (distill.DecodeTraceRecorder).
        # Zero-cost when None; not part of checkpoints.
        self.trace_recorder = None
        # Optional grammar-state decision trace recorder (distill.GrammarTraceRecorder).
        # Zero-cost when None.
        self.grammar_trace_recorder = None
        # SLM-176: lightweight shortlist-trace buffer for wiring diagnostics.
        self.action_shortlist_traces: list[dict[str, object]] = []
        # SLM-176: lazily-built action catalog and deterministic fixture vectors.
        self._action_shortlist_catalog: object | None = None
        self._action_shortlist_vectors: dict[str, torch.Tensor] | None = None
        # Optional retrieval bank: list[(norm_prompt, openui, id)]
        self.skeleton_bank: list[tuple[str, str, str]] = []
        # Train-time caches (formatted context string keyed by record id).
        self._context_text_cache: dict[str, str] = {}
        self._target_ids_cache: dict[str, list[int]] = {}
        self._compiler_decision_cache: dict[
            tuple[tuple[int, ...], tuple[str, ...]], tuple[Any, ...]
        ] = {}
        self._context_token_count_cache: dict[str, int] = {}
        self._placeholder_token_ids: set[int] | None = None
        self._component_token_ids_cache: tuple[int, ...] | None = None
        self._binder_token_ids_cache: tuple[int, ...] | None = None
        self._component_edge_cache: dict[str, tuple[tuple[int, int], ...]] = {}
        self._slot_contracts: list[list[str] | None] | None = None
        self._semantic_role_candidates: list[dict[str, tuple[str, ...]]] | None = None
        self._semantic_plan_action_scores: list[dict[int, float]] | None = None
        self._semantic_plan_action_counts: list[dict[int, int]] | None = None
        self._last_generation_evidence: list[dict[str, object]] = []
        # Per-example symbol tables for lexer-native encode/decode.
        self._symbol_tables: dict[str, object] = {}
        self._current_runtime_table: object | None = None
        self.to(device)

    def _effective_structural_bias(self) -> float:
        if getattr(self.config, "grammar_trust_model", False):
            return 0.0
        return float(self.config.structural_bias or 0.0)

    def _effective_min_content(self, slot_contract: list[str] | None) -> int:
        """A4 minimum-content floor for the compiler-tree completion forest.

        ``decode_min_content`` == -1 derives the floor from the resolved
        slot-contract inventory (one component per distinct content slot, capped
        so it never demands more than the prompt implies); >0 uses the fixed
        value; 0 disables the contract.
        """
        raw = int(getattr(self.config, "decode_min_content", 0) or 0)
        if raw >= 0:
            return raw
        if not slot_contract:
            return 0
        # Distinct placeholder roots ≈ the number of content-bearing components
        # the prompt asks for; the empty layout binds none of them.
        roots = {str(slot).split(".", 1)[0] for slot in slot_contract if slot}
        return len(roots)

    def _coverage_deficit(self, ids: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
        """A3 per-position content-coverage deficit.

        Non-content (structural filler) positions score ``1 - content_fraction``
        per row, where ``content_fraction`` is the share of known positions that
        hold component/symbol/content tokens. Content positions score 0 (keep
        them). When a row is content-sparse the filler positions get a high
        deficit and are preferentially remasked to re-decode toward content;
        when content is already dense the deficit is near zero and the policy
        falls back to ordinary confidence remasking. Self-contained: reads only
        the tokenizer's compiler-derived symbol spaces, no slot contract.
        """
        content_ids: set[int] = set()
        for kind in ("component", "sym", "bind"):
            try:
                content_ids |= set(self.tokenizer.kind_ids(kind))
            except Exception:  # noqa: BLE001 - tokenizer without that kind
                continue
        deficit = torch.zeros_like(ids, dtype=torch.float32)
        if not content_ids:
            return deficit
        rows, length = ids.shape
        for b in range(rows):
            known_positions = [t for t in range(length) if bool(known[b, t].item())]
            if not known_positions:
                continue
            content_hits = sum(
                1 for t in known_positions if int(ids[b, t].item()) in content_ids
            )
            content_fraction = content_hits / len(known_positions)
            row_deficit = 1.0 - content_fraction
            for t in known_positions:
                if int(ids[b, t].item()) not in content_ids:
                    deficit[b, t] = row_deficit
        return deficit

    def _pick_kwargs(self) -> dict[str, object]:
        trust = bool(getattr(self.config, "grammar_trust_model", False))
        prefer = bool(getattr(self.config, "grammar_prefer_structural", True))
        return {
            "prefer_structural": False if trust else prefer,
            "sample": bool(getattr(self.config, "grammar_sample_decode", False)),
            "temperature": float(
                getattr(self.config, "grammar_sample_temperature", 0.8) or 0.8
            ),
            "verify_chosen_only": bool(
                getattr(self.config, "grammar_verify_chosen_only", False)
            ),
            "grammar_equivalence_cache": bool(
                getattr(self.config, "grammar_equivalence_cache", False)
            ),
            "active_dynamic_ids": (
                self._current_runtime_table.active_token_ids(self.tokenizer)
                if bool(getattr(self.config, "grammar_active_symbol_bitsets", False))
                and hasattr(self._current_runtime_table, "active_token_ids")
                else None
            ),
        }

    def _new_grammar_states(self, batch_size: int) -> list | None:
        """Allocate per-row GrammarDecodeState when P1 incremental state is on."""
        if not bool(getattr(self.config, "grammar_incremental_state", True)):
            return None
        return [
            make_grammar_state(
                verify_chosen_only=bool(
                    getattr(self.config, "grammar_verify_chosen_only", False)
                ),
                skip_exact_stream_probe=bool(
                    getattr(self.config, "grammar_skip_exact_stream_probe", True)
                ),
                use_copy_probes=bool(getattr(self.config, "grammar_copy_probes", True)),
                early_exit_pick=bool(
                    getattr(self.config, "grammar_early_exit_pick", True)
                ),
            )
            for _ in range(batch_size)
        ]

    def clear_train_caches(self) -> None:
        self._context_text_cache.clear()
        self._target_ids_cache.clear()
        self._compiler_decision_cache.clear()
        self._context_token_count_cache.clear()
        if is_hf_context(self.context) and hasattr(
            self.context, "clear_backbone_cache"
        ):
            self.context.clear_backbone_cache()  # type: ignore[union-attr]

    def trainable_parameters(self):
        """Yield trainable parameters, deduplicating shared/tied storage."""
        seen: set[int] = set()
        for p in self.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                yield p

    def take_detached_auxiliary_loss(self) -> torch.Tensor | None:
        loss = getattr(self, "_detached_auxiliary_loss", None)
        self._detached_auxiliary_loss = None
        return loss

    def optimizer_parameter_groups(self) -> list[dict[str, list[nn.Parameter]]]:
        """Keep shared-model optimizer grouping invariant across aux heads."""
        auxiliary = (
            "length_head.",
            "component_inventory_head.",
            "component_plan_head.",
            "slot_component_head.",
            "component_edge_head.",
            "binder_component_plan_head.",
            "binder_topology_head.",
            "binder_arity_head.",
            "root_reference_arity_head.",
            "root_reference_identity_head.",
            "trust_gate.",
            "survival_head.",
        )
        grouped: dict[str, list[nn.Parameter]] = {"base": []}
        seen: set[int] = set()
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            pid = id(parameter)
            if pid in seen:
                continue
            seen.add(pid)
            owner = next(
                (prefix for prefix in auxiliary if name.startswith(prefix)), "base"
            )
            grouped.setdefault(owner, []).append(parameter)
        return [
            {"params": grouped[owner]}
            for owner in ("base", *auxiliary)
            if grouped.get(owner)
        ]

    def attach_adapter(self, spec: Any) -> None:
        """Attach a removable low-rank adapter to the denoiser (adapter-only training).

        Fails closed if an adapter is already attached or the spec's base compatibility
        fingerprint does not match this model. Every non-adapter parameter is frozen, so
        ``trainable_parameters()`` yields only the adapter tensors while the parent
        weights stay untouched — ``disable_adapter()`` therefore restores the exact
        parent map. The context tower is never adapted.
        """
        from slm_training.models.adapters.twotower_adapter import (
            attach_low_rank_adapters,
        )

        if getattr(self, "_adapter_spec", None) is not None:
            raise ValueError("an adapter is already attached to this model")
        expected = getattr(spec, "base_compatibility_fingerprint", "")
        if expected and expected != self.compatibility_fingerprint():
            raise ValueError(
                "adapter base compatibility fingerprint does not match this model"
            )
        # NB: spec.base_checkpoint_sha is provenance only. The architecture fingerprint
        # above does not distinguish two checkpoints of the same shape; enforcing the
        # exact base checkpoint needs the model to carry its loaded checkpoint identity,
        # which is deferred with the checkpoint-interplay work (see the LDI2-01 memo).
        self._adapter_modules = attach_low_rank_adapters(
            self.denoiser, spec, seed=int(self.config.seed)
        )
        for name, parameter in self.named_parameters():
            parameter.requires_grad_("lora_" in name.lower())
        self._adapter_spec = spec

    def has_adapter(self) -> bool:
        return getattr(self, "_adapter_spec", None) is not None

    def enable_adapter(self) -> None:
        for wrapper in getattr(self, "_adapter_modules", {}).values():
            wrapper.enable_adapter()

    def disable_adapter(self) -> None:
        for wrapper in getattr(self, "_adapter_modules", {}).values():
            wrapper.disable_adapter()

    def adapter_parameters(self):
        for wrapper in getattr(self, "_adapter_modules", {}).values():
            yield from wrapper.adapter_parameters()

    def active_adapter_identity(self) -> str:
        """Content digest of the active adapter tensors, or "" when none are attached."""
        from slm_training.lineage.records import content_sha

        modules = getattr(self, "_adapter_modules", {})
        if not modules:
            return ""
        return content_sha(
            {
                key: [
                    parameter.detach().to(torch.float64).cpu().flatten().tolist()
                    for parameter in wrapper.adapter_parameters()
                ]
                for key, wrapper in sorted(modules.items())
            }
        )

    def merge_adapter_copy(self) -> "TwoTowerModel":
        """Return a wrapper-free copy with the adapter delta folded into the weights.

        Merge is one-way and on a **copy**: this model and its removable adapter are
        left untouched. Every ``LowRankAdapter`` in the copy's denoiser is replaced by a
        plain ``nn.Linear`` equal to the adapter-enabled map, so the merged model carries
        no active wrappers and trains as an ordinary full model.
        """
        import copy

        from slm_training.models.adapters.low_rank import LowRankAdapter

        if not self.has_adapter():
            raise ValueError("no adapter is attached to merge")
        merged = copy.deepcopy(self)

        def _fold(module: nn.Module) -> None:
            for name, child in list(module.named_children()):
                if isinstance(child, LowRankAdapter):
                    setattr(module, name, child.merged_linear())
                else:
                    _fold(child)

        _fold(merged.denoiser)
        merged._adapter_modules = {}
        merged._adapter_spec = None
        for parameter in merged.parameters():
            parameter.requires_grad_(True)
        return merged

    def save_adapter(
        self, path: Path | str, *, provenance: dict[str, Any] | None = None
    ) -> None:
        """Write the removable adapter (config + tensors + manifest) to its own directory.

        The base checkpoint is not duplicated — only the adapter tensors and the identity
        needed to fail closed on load. Requires an attached adapter.
        """
        import json

        if not self.has_adapter():
            raise ValueError("no adapter is attached to save")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        spec = self._adapter_spec
        (path / "adapter_config.json").write_text(
            json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tensors = {
            name: parameter.detach().cpu()
            for name, parameter in self.named_parameters()
            if "lora_" in name.lower()
        }
        torch.save(tensors, path / "adapter_model.pt")
        manifest = {
            "kind": "twotower_low_rank_adapter",
            "schema_version": spec.schema_version,
            "module_map": sorted(self._adapter_modules),
            "parameter_names": sorted(tensors),
            "parameter_shapes": {name: list(t.shape) for name, t in tensors.items()},
            "trainable_parameter_count": int(sum(t.numel() for t in tensors.values())),
            "adapter_bytes": int((path / "adapter_model.pt").stat().st_size),
            "base_compatibility_fingerprint": spec.base_compatibility_fingerprint,
            "base_checkpoint_sha": spec.base_checkpoint_sha,
            "tokenizer_sha": spec.tokenizer_sha,
            "provenance": provenance or {},
        }
        (path / "adapter_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def load_adapter(self, path: Path | str, *, trainable: bool = False) -> Any:
        """Attach and load a removable adapter, failing closed on an identity mismatch.

        The adapter's base fingerprint, tokenizer identity, and resolved module map must
        match this model, or loading raises before any tensor is copied.
        """
        import json

        from slm_training.models.adapters.spec import TwoTowerAdapterSpec

        path = Path(path)
        spec = TwoTowerAdapterSpec.from_dict(
            json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        )
        manifest_path = path / "adapter_manifest.json"
        if not manifest_path.exists():
            raise ValueError("adapter directory is missing adapter_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # --- Validate the whole artifact BEFORE mutating the model ---------------------
        # Tokenizer + config/manifest integrity are all checked up front so a mismatched
        # or truncated adapter never leaves the model half-attached.
        expected_tokenizer = self.artifact_identity()["tokenizer_sha"]
        if spec.tokenizer_sha and spec.tokenizer_sha != expected_tokenizer:
            raise ValueError("adapter tokenizer identity does not match this model")
        if (
            manifest.get("base_compatibility_fingerprint")
            != spec.base_compatibility_fingerprint
        ):
            raise ValueError(
                "adapter manifest disagrees with its config on base identity"
            )
        tensors = torch.load(
            path / "adapter_model.pt", map_location="cpu", weights_only=True
        )
        loaded_names = set(tensors)
        expected_names = set(manifest.get("parameter_names", []))
        if loaded_names != expected_names:
            raise ValueError(
                "adapter tensors do not match the manifest parameter set "
                f"(missing={sorted(expected_names - loaded_names)}, "
                f"unexpected={sorted(loaded_names - expected_names)})"
            )
        expected_shapes = manifest.get("parameter_shapes", {})
        for name, tensor in tensors.items():
            if list(tensor.shape) != list(expected_shapes.get(name, [])):
                raise ValueError(
                    f"adapter tensor {name!r} shape {list(tensor.shape)} does not match "
                    f"manifest shape {expected_shapes.get(name)}"
                )

        # --- Mutate only after every check above has passed ---------------------------
        # attach_adapter fails closed on a base-fingerprint mismatch before wrapping.
        self.attach_adapter(spec)
        parameters = dict(self.named_parameters())
        model_adapter_names = {name for name in parameters if "lora_" in name.lower()}
        if loaded_names != model_adapter_names:
            raise ValueError(
                "adapter tensors do not match the attached module map "
                f"(missing={sorted(model_adapter_names - loaded_names)}, "
                f"unexpected={sorted(loaded_names - model_adapter_names)})"
            )
        with torch.no_grad():
            for name, tensor in tensors.items():
                target = parameters[name]
                target.copy_(tensor.to(target.device, target.dtype))
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(bool(trainable) and "lora_" in name.lower())
        return spec

    def _count_context_tokens(self, text: str) -> int:
        """Context token count under the active backend (capped at max_prompt_len)."""
        if is_hf_context(self.context):
            try:
                encoded = self.context.tokenizer(  # type: ignore[union-attr]
                    text,
                    truncation=True,
                    max_length=self.config.max_prompt_len,
                )
                return len(encoded["input_ids"])
            except Exception:  # noqa: BLE001
                return 0
        ids = self.context_tokenizer.encode(text)
        return min(len(ids), self.config.max_prompt_len)

    def count_batch_tokens(self, batch: list[ExampleRecord]) -> tuple[int, int]:
        """Exact (prompt/context, target) token counts for token-budget accounting.

        Reuses the train-time caches populated by ``training_loss`` so the
        common path is cheap; counts reflect the tokens the model actually
        consumes (post truncation / context formatting).
        """
        cache_on = bool(getattr(self.config, "cache_context", True))
        prompt_tokens = 0
        target_tokens = 0
        for r in batch:
            key = r.id or r.prompt
            ids = self._target_ids_cache.get(key) if cache_on else None
            if ids is None:
                ids = _truncate_with_eos(
                    self._encode_openui(
                        r.openui,
                        placeholders=list(r.placeholders or []),
                        cache_key=key,
                    ),
                    self.config.max_target_len,
                    self.tokenizer.eos_id,
                )
                if cache_on:
                    self._target_ids_cache[key] = ids
            target_tokens += len(ids)
            count = self._context_token_count_cache.get(key) if cache_on else None
            if count is None:
                text = self._context_text_cache.get(key) if cache_on else None
                if text is None:
                    design_md = self._training_design_md(r.design_md, key)
                    text = self._format_one_context(
                        r.prompt,
                        design_md,
                        query_prompt=r.prompt,
                        slot_contract=self._resolve_slot_contract(
                            r.prompt, r, design_md, use_gold_design=False
                        )
                        if getattr(self.config, "slot_contract_in_context", False)
                        else None,
                        output_kind=r.target_kind,
                        output_category=r.target_category,
                    )
                    if cache_on:
                        self._context_text_cache[key] = text
                count = self._count_context_tokens(text)
                if cache_on:
                    self._context_token_count_cache[key] = count
            prompt_tokens += count
        return prompt_tokens, target_tokens

    def _encode_context(
        self,
        prompts: list[str],
        *,
        cache_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from slm_training.runtime.telemetry import timed

        stats = get_active_stats()
        with timed("context_encode"), timed_ms(stats, "context_ms"):
            if is_hf_context(self.context):
                assert isinstance(self.context, HFContextEncoder)
                self.context.cache_backbone = bool(
                    getattr(self.config, "cache_context", True)
                )
                return self.context.forward_prompts(
                    prompts,
                    max_len=self.config.max_prompt_len,
                    device=self.device_name,
                    cache_keys=cache_keys if self.config.cache_context else None,
                )
            assert isinstance(self.context, ScratchContextEncoder)
            enable_grad = (not self.config.freeze_context) and self.training
            with torch.set_grad_enabled(enable_grad):
                return self.context.forward_prompts(
                    prompts,
                    encode_fn=self.context_tokenizer.encode,
                    max_len=self.config.max_prompt_len,
                    pad_id=self.context_tokenizer.pad_id,
                    device=self.device_name,
                )

    def apply_dynamic_quant(self) -> bool:
        """P5: dynamically quantize Linear layers to int8 (CPU). Returns True on success."""
        if str(self.device_name) != "cpu":
            return False
        try:
            quantized = torch.ao.quantization.quantize_dynamic(
                self.denoiser,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            self.denoiser = quantized
            self.config.use_dynamic_quant = True
            return True
        except Exception:  # noqa: BLE001
            return False

    def apply_quant_format(self, format_id: str) -> bool:
        """CAP3-01: apply a reference low-bit quantizer format to Linear weights.

        This is a disabled-by-default reference path.  It records a conversion
        ledger but does not claim speedup or quality retention.
        """
        from slm_training.models.quantization.convert import (
            QuantizationPolicy,
            convert_twotower,
        )
        from slm_training.models.quantization.formats import (
            binary_format,
            binary_plus_mask_format,
            int4_format,
            int8_format,
            learned_four_level_zero_format,
            symmetric_four_level_format,
            ternary_format,
        )

        registry = {
            "binary": binary_format,
            "ternary": ternary_format,
            "learned4zero": learned_four_level_zero_format,
            "learned_four_level_zero": learned_four_level_zero_format,
            "symmetric_four_level": symmetric_four_level_format,
            "int4": int4_format,
            "int8": int8_format,
            "binary_plus_mask": binary_plus_mask_format,
        }
        factory = registry.get(format_id)
        if factory is None:
            return False
        try:
            fmt = factory(group_size=128)
            policy = QuantizationPolicy(default_format=fmt)
            _, records = convert_twotower(
                self, policy, fail_on_tied=False, in_place=True
            )
            self.config.quant_format = format_id
            self._quant_conversion_records = [r.as_dict() for r in records]
            return True
        except Exception:  # noqa: BLE001
            return False

    def _mask_targets(
        self, target_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return noisy_ids, predict mask, and optional per-row MDLM weights.

        When ``visible_corrupt_rate > 0`` (E32), a fraction of *visible*
        (non-mask) tokens are replaced with wrong vocab ids while remaining in
        the predict mask so the denoiser learns to revise wrong visibles.
        """
        bsz, seq = target_ids.shape
        device = target_ids.device
        frozen = target_ids.eq(self.tokenizer.pad_id) | target_ids.eq(
            self.tokenizer.bos_id
        )
        row_weights: torch.Tensor | None = None
        if bool(getattr(self.config, "mdlm_schedule", False)):
            # MDLM log-linear α(t)=1-t ⇒ mask rate t, CE weight 1/t.
            eps = float(getattr(self.config, "mdlm_eps", 1e-3) or 1e-3)
            t = torch.empty(bsz, 1, device=device).uniform_(eps, 1.0)
            rates = t
            row_weights = (1.0 / t.clamp(min=eps)).view(bsz)
        else:
            rates = torch.empty(bsz, 1, device=device).uniform_(
                self.config.mask_min, self.config.mask_max
            )
        rand = torch.rand(bsz, seq, device=device)
        noise = (rand < rates) & (~frozen)

        # Stage-2: mixed statement-span masking (M_random ∪ M_subtree).
        pattern = str(getattr(self.config, "mask_pattern", "random") or "random")
        stmt_p = float(getattr(self.config, "statement_mask_prob", 0.35) or 0.0)
        if pattern == "mixed" and stmt_p > 0.0:
            try:
                from slm_training.models.dsl_tokenizer import is_dsl_native_tokenizer

                if is_dsl_native_tokenizer(self.tokenizer):
                    for i in range(bsz):
                        if self._rng.random() > stmt_p:
                            continue
                        spans = self.tokenizer.statement_spans(target_ids[i].tolist())
                        if not spans:
                            continue
                        lo, hi = spans[self._rng.randrange(len(spans))]
                        for j in range(lo, hi):
                            if not bool(frozen[i, j]):
                                noise[i, j] = True
                else:
                    # CompositionalTokenizer has no lexer statement_spans API.
                    # Derive newline-delimited spans so mixed masking is not a
                    # silent random-mask fallback.
                    newline_ids = set(self.tokenizer.encode("\n", add_special=False))
                    if newline_ids:
                        for i in range(bsz):
                            if self._rng.random() > stmt_p:
                                continue
                            row = target_ids[i].tolist()
                            boundaries = [
                                j for j, tid in enumerate(row) if tid in newline_ids
                            ]
                            start = 0
                            spans: list[tuple[int, int]] = []
                            for boundary in boundaries + [seq]:
                                if boundary > start:
                                    spans.append((start, boundary + 1))
                                start = boundary + 1
                            if spans:
                                lo, hi = spans[self._rng.randrange(len(spans))]
                                for j in range(lo, min(hi, seq)):
                                    if not bool(frozen[i, j]):
                                        noise[i, j] = True
            except Exception:  # noqa: BLE001
                pass

        # Ensure at least one predictable token per non-empty row.
        for i in range(bsz):
            if frozen[i].all():
                continue
            if not bool(noise[i].any()):
                valid = (~frozen[i]).nonzero(as_tuple=False).view(-1)
                if valid.numel():
                    noise[i, int(valid[self._rng.randrange(valid.numel())])] = True
        noisy = target_ids.clone()
        noisy[noise] = self.tokenizer.mask_id

        # E32: corrupt a fraction of remaining visible tokens to wrong ids.
        corrupt_rate = float(getattr(self.config, "visible_corrupt_rate", 0.0) or 0.0)
        if corrupt_rate > 0.0:
            visible = (~noise) & (~frozen)
            flip = (torch.rand(bsz, seq, device=device) < corrupt_rate) & visible
            if bool(flip.any()):
                vocab = self.tokenizer.vocab_size
                # Sample wrong content tokens; reject pad/bos/eos/mask/unk/gold.
                special = {
                    self.tokenizer.pad_id,
                    self.tokenizer.bos_id,
                    self.tokenizer.eos_id,
                    self.tokenizer.mask_id,
                    self.tokenizer.unk_id,
                }

                def _bad(candidate: torch.Tensor) -> torch.Tensor:
                    bad = candidate.eq(target_ids)
                    for sid in special:
                        bad = bad | candidate.eq(sid)
                    return bad

                wrong = torch.randint(0, vocab, (bsz, seq), device=device)
                bad = _bad(wrong)
                for _ in range(6):
                    if not bool((flip & bad).any()):
                        break
                    resample = torch.randint(0, vocab, (bsz, seq), device=device)
                    wrong = torch.where(bad, resample, wrong)
                    bad = _bad(wrong)
                flip = flip & ~bad
                noisy = torch.where(flip, wrong, noisy)
                noise = noise | flip
        return noisy, noise, row_weights

    def _online_diffusion_targets(
        self,
        target_ids: torch.Tensor,
        batch: list[ExampleRecord],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample aligned online states while keeping the clean target cache intact."""
        from slm_training.data.diffusion import (
            DiffusionConfig,
            corrupt_batch,
        )

        config = DiffusionConfig(
            policies=tuple(self.config.diffusion_policies),
            mask_min=float(self.config.mask_min),
            mask_max=float(self.config.mask_max),
            overallocate=int(self.config.diffusion_overallocate),
            length_buckets=tuple(self.config.diffusion_length_buckets),
            max_length=int(self.config.max_target_len),
        )
        corruption = corrupt_batch(
            target_ids.detach().cpu().tolist(),
            self.tokenizer,
            config=config,
            rng=self._rng,
            metadata=[record.meta for record in batch],
        )
        targets = _pad_batch(
            [list(row.target_ids) for row in corruption.rows],
            self.tokenizer.pad_id,
            device=target_ids.device,
        )
        noisy = _pad_batch(
            [list(row.noisy_ids) for row in corruption.rows],
            self.tokenizer.pad_id,
            device=target_ids.device,
        )
        predict = torch.zeros_like(targets, dtype=torch.bool)
        for index, row in enumerate(corruption.rows):
            predict[index, : row.canvas_length] = torch.tensor(
                row.predict_mask,
                dtype=torch.bool,
                device=target_ids.device,
            )
        bucket_targets = torch.tensor(
            [row.length_bucket_index for row in corruption.rows],
            dtype=torch.long,
            device=target_ids.device,
        )
        return targets, noisy, predict, bucket_targets

    @staticmethod
    def _pool_context(
        context: torch.Tensor, pad_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if pad_mask is None:
            return context.mean(dim=1)
        visible = (~pad_mask).unsqueeze(-1).to(context.dtype)
        return (context * visible).sum(dim=1) / visible.sum(dim=1).clamp(min=1.0)

    def _predict_target_lengths(
        self, context: torch.Tensor, pad_mask: torch.Tensor | None
    ) -> list[int] | None:
        if self.length_head is None:
            return None
        logits = self.length_head(self._pool_context(context, pad_mask))
        buckets = tuple(int(value) for value in self.config.diffusion_length_buckets)
        return [
            max(8, min(buckets[int(index)], int(self.config.max_target_len)))
            for index in logits.argmax(dim=-1).tolist()
        ]

    def _merge_ltr_suffix_mask(
        self, target_ids: torch.Tensor, noisy: torch.Tensor, predict_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Force-mask a random suffix; return ltr-only positions for loss weighting."""
        bsz, seq = target_ids.shape
        ltr_suffix = torch.zeros_like(predict_mask)
        for i in range(bsz):
            cut = self._rng.randint(1, max(1, seq - 1))
            # Always train the first post-BOS decision. Random suffix cuts make
            # this position otherwise receive LTR supervision only rarely.
            if int(target_ids[i, 1]) != self.tokenizer.pad_id:
                ltr_suffix[i, 1] = True
                if not bool(predict_mask[i, 1].item()):
                    predict_mask[i, 1] = True
                    noisy[i, 1] = self.tokenizer.mask_id
            for j in range(cut, seq):
                if int(target_ids[i, j]) == self.tokenizer.pad_id:
                    break
                ltr_suffix[i, j] = True
                if not bool(predict_mask[i, j].item()):
                    predict_mask[i, j] = True
                    noisy[i, j] = self.tokenizer.mask_id
        return noisy, predict_mask, ltr_suffix

    def _placeholder_ids(self) -> set[int]:
        if self._placeholder_token_ids is None:
            ids: set[int] = set()
            try:
                from slm_training.models.choice_tokenizer import is_choice_tokenizer
                from slm_training.models.dsl_tokenizer import (
                    TokenKind,
                    is_dsl_native_tokenizer,
                )

                if is_dsl_native_tokenizer(self.tokenizer) or is_choice_tokenizer(
                    self.tokenizer
                ):
                    ids |= self.tokenizer.kind_ids(TokenKind.SYM)
                    self._placeholder_token_ids = ids
                    return ids
            except Exception:  # noqa: BLE001
                pass
            for tok, tid in self.tokenizer.token_to_id.items():
                if tok in {":", "."}:
                    ids.add(tid)
                elif tok and tok[0].islower() and tok.isidentifier():
                    ids.add(tid)
                elif tok.startswith('":') or (tok.startswith('"') and ":" in tok):
                    ids.add(tid)
            self._placeholder_token_ids = ids
        return self._placeholder_token_ids

    def _encode_openui(
        self,
        openui: str,
        *,
        placeholders: list[str] | None = None,
        cache_key: str | None = None,
    ) -> list[int]:
        """Encode target OpenUI, optionally via lexer-native symbol table."""
        try:
            from slm_training.models.choice_tokenizer import is_choice_tokenizer
            from slm_training.models.dsl_tokenizer import (
                SymbolTable,
                is_dsl_native_tokenizer,
            )

            if is_choice_tokenizer(self.tokenizer):
                # Choice codec: slot pointers resolve through the table's
                # placeholder inventory; cache the table so decode shares it.
                table = SymbolTable.from_placeholders(
                    placeholders, max_slots=self.tokenizer.sym_slots
                )
                ids = self.tokenizer.encode(
                    openui, table=table, placeholders=placeholders
                )
                if cache_key is not None:
                    self._symbol_tables[cache_key] = table
                return ids
            if is_dsl_native_tokenizer(self.tokenizer):
                use_sym = bool(getattr(self.config, "use_symbol_table", True))
                table = SymbolTable.from_placeholders(
                    placeholders, max_slots=self.tokenizer.sym_slots
                )
                if (
                    bool(getattr(self.config, "symbol_slot_augmentation", False))
                    and self.training
                ):
                    key_seed = sum(ord(ch) for ch in (cache_key or openui))
                    table = table.permuted(int(self.config.seed) + key_seed)
                if cache_key is not None:
                    self._symbol_tables[cache_key] = table
                return self.tokenizer.encode(
                    openui,
                    table=table,
                    use_symbol_table=use_sym,
                    placeholders=placeholders,
                    symbol_anonymization=bool(
                        getattr(self.config, "symbol_anonymization", True)
                    ),
                )
        except Exception:  # noqa: BLE001
            pass
        return self.tokenizer.encode(openui)

    def _runtime_feature_tensor(self, tables: list[object]) -> torch.Tensor | None:
        """Build per-example deltas for reserved symbol rows from existing embeddings."""
        mode = str(getattr(self.config, "runtime_symbol_features", "none") or "none")
        if mode == "none" or not tables:
            return None
        if mode not in {"surface", "role_gated", "replace"}:
            raise ValueError(f"unknown runtime_symbol_features mode {mode!r}")
        from slm_training.models.dsl_tokenizer import SymbolTable

        if not all(isinstance(table, SymbolTable) for table in tables):
            return None
        weight = self.denoiser.tok.weight
        features = weight.new_zeros(
            (len(tables), self.tokenizer.vocab_size, weight.size(1))
        )
        for row, raw_table in enumerate(tables):
            assert isinstance(raw_table, SymbolTable)
            targets: list[tuple[int, RuntimeSymbol]] = []
            for slot, surface in enumerate(raw_table.placeholders):
                symbol = raw_table.symbol_for_surface(surface) or RuntimeSymbol(
                    surface=surface, role="external_entity"
                )
                targets.append((self.tokenizer.sym_id(slot), symbol))
            for surface, slot in raw_table.binders.items():
                symbol = raw_table.symbol_for_surface(surface) or RuntimeSymbol(
                    surface=surface, role="alpha_binder"
                )
                targets.append((self.tokenizer.bind_id(slot), symbol))
            for surface, slot in raw_table.states.items():
                symbol = raw_table.symbol_for_surface(surface) or RuntimeSymbol(
                    surface=surface, role="state"
                )
                targets.append((self.tokenizer.state_id(slot), symbol))
            for token_id, symbol in targets:
                if mode == "role_gated" and symbol.role in {
                    "alpha_binder",
                    "fresh_binder",
                }:
                    continue
                text = " ".join(
                    part
                    for part in (
                        symbol.surface,
                        *symbol.namespace,
                        symbol.semantic_type,
                        symbol.scope,
                        symbol.signature,
                        symbol.description,
                    )
                    if part
                )
                byte_ids = self.tokenizer._encode_bytes(text) if text else []
                if byte_ids:
                    index = torch.tensor(byte_ids, device=weight.device)
                    composed = weight.index_select(0, index).mean(0)
                    if mode == "replace":
                        # C2 (SLM-26): dynamic pseudo-embedding — the delta
                        # cancels the learned pool row, so the symbol's tied
                        # input embedding AND output projection become the
                        # deterministic byte-compositional vector (DyVo-style;
                        # same embedding matrix rows, so weight tying and
                        # batching are untouched). Same surface → identical
                        # vector at every slot and position by construction.
                        features[row, token_id] = composed - weight[token_id]
                    else:
                        features[row, token_id] = composed
        return features

    def _set_runtime_symbol_features(self, tables: list[object]) -> torch.Tensor | None:
        features = self._runtime_feature_tensor(tables)
        self.denoiser.set_runtime_symbol_features(features)
        return features

    def _mask_inactive_dynamic_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Conservatively hide undeclared entity/state rows; binders stay writable."""
        if not bool(getattr(self.config, "semantic_candidate_masks", False)):
            return logits
        try:
            from slm_training.models.dsl_tokenizer import SymbolTable, TokenKind

            table = self._current_runtime_table
            if not isinstance(table, SymbolTable):
                return logits
            active = table.active_token_ids(self.tokenizer)
            dynamic = self.tokenizer.kind_ids(TokenKind.SYM) | self.tokenizer.kind_ids(
                TokenKind.STATE
            )
            blocked = sorted(dynamic - active)
            stats = get_active_stats()
            if stats is not None:
                stats.dynamic_mask_applications += 1
                stats.dynamic_candidates_before += len(dynamic)
                stats.dynamic_candidates_after += len(dynamic) - len(blocked)
            if blocked:
                logits = logits.clone()
                logits[..., blocked] = float("-inf")
        except Exception:  # noqa: BLE001
            pass
        return logits

    def _decode_openui(
        self,
        ids_1d: torch.Tensor | list[int],
        *,
        placeholders: list[str] | None = None,
        cache_key: str | None = None,
    ) -> str:
        token_ids = (
            ids_1d.tolist() if isinstance(ids_1d, torch.Tensor) else list(ids_1d)
        )
        if self.tokenizer.eos_id in token_ids[1:]:
            end = token_ids.index(self.tokenizer.eos_id, 1)
            token_ids = token_ids[: end + 1]
        try:
            from slm_training.models.choice_tokenizer import is_choice_tokenizer
            from slm_training.models.dsl_tokenizer import (
                SymbolTable,
                is_dsl_native_tokenizer,
            )

            if is_dsl_native_tokenizer(self.tokenizer) or is_choice_tokenizer(
                self.tokenizer
            ):
                table = None
                if cache_key and cache_key in self._symbol_tables:
                    table = self._symbol_tables[cache_key]  # type: ignore[assignment]
                if table is None:
                    table = SymbolTable.from_placeholders(
                        placeholders, max_slots=self.tokenizer.sym_slots
                    )
                return self.tokenizer.decode(token_ids, table=table).strip()
        except Exception:  # noqa: BLE001
            pass
        return self.tokenizer.decode(token_ids).strip()

    def forward(self, batch: list[ExampleRecord]) -> float:
        self.train()
        loss = self.training_loss(batch)
        return float(loss.detach().cpu())

    def training_loss(self, batch: list[ExampleRecord]) -> torch.Tensor:
        self.train()
        self.last_training_metrics = {}
        self._detached_auxiliary_loss: torch.Tensor | None = None
        cache_on = bool(getattr(self.config, "cache_context", True))
        prompts: list[str] = []
        cache_keys: list[str] = []
        targets: list[list[int]] = []
        for r in batch:
            key = r.id or r.prompt
            cache_keys.append(key)
            if cache_on and key in self._context_text_cache:
                prompts.append(self._context_text_cache[key])
            else:
                design_md = self._training_design_md(r.design_md, key)
                text = self._format_one_context(
                    r.prompt,
                    design_md,
                    query_prompt=r.prompt,
                    slot_contract=self._resolve_slot_contract(
                        r.prompt, r, design_md, use_gold_design=False
                    )
                    if getattr(self.config, "slot_contract_in_context", False)
                    else None,
                    output_kind=r.target_kind,
                    output_category=r.target_category,
                )
                if cache_on:
                    self._context_text_cache[key] = text
                prompts.append(text)
            if cache_on and key in self._target_ids_cache:
                targets.append(self._target_ids_cache[key])
            else:
                ids = _truncate_with_eos(
                    self._encode_openui(
                        r.openui,
                        placeholders=list(r.placeholders or []),
                        cache_key=key,
                    ),
                    self.config.max_target_len,
                    self.tokenizer.eos_id,
                )
                if cache_on:
                    self._target_ids_cache[key] = ids
                targets.append(ids)

        target_ids = _pad_batch(targets, self.tokenizer.pad_id, device=self.device_name)
        ctx, ctx_pad = self._encode_context(prompts, cache_keys=cache_keys)
        bucket_targets: torch.Tensor | None = None
        if str(getattr(self.config, "mask_pattern", "random")) == "diffusion":
            target_ids, noisy, predict_mask, bucket_targets = (
                self._online_diffusion_targets(target_ids, batch)
            )
            mdlm_row_w = None
        else:
            noisy, predict_mask, mdlm_row_w = self._mask_targets(target_ids)

        ltr_w = float(self.config.ltr_loss_weight or 0.0)
        fuse = bool(getattr(self.config, "fuse_ltr_loss", True))
        ltr_suffix = torch.zeros_like(predict_mask)
        if ltr_w > 0.0 and target_ids.size(1) >= 2 and fuse:
            noisy, predict_mask, ltr_suffix = self._merge_ltr_suffix_mask(
                target_ids, noisy, predict_mask
            )

        from slm_training.runtime.telemetry import timed

        depth_logits: list[torch.Tensor] | None = None
        with timed("denoiser_forward"):
            self._set_runtime_symbol_features(
                [self._symbol_tables.get(key) for key in cache_keys]
            )
            try:
                ds_weights = tuple(
                    getattr(self.config, "recursive_depth_supervision_weights", None)
                    or ()
                )
                has_recursive_outputs = hasattr(self.denoiser, "recursive_outputs")
                if ds_weights and has_recursive_outputs:
                    rec_out = self.denoiser.recursive_outputs(
                        noisy, ctx, pad_id=self.tokenizer.pad_id, ctx_pad_mask=ctx_pad
                    )
                    logits = rec_out["logits"]
                    assert isinstance(logits, torch.Tensor)
                    depth_logits = rec_out["depth_logits"]
                    assert isinstance(depth_logits, list)
                else:
                    logits = self.denoiser(
                        noisy, ctx, pad_id=self.tokenizer.pad_id, ctx_pad_mask=ctx_pad
                    )
            finally:
                # Request-local features must not outlive their batch: a later
                # forward with a different batch size (loss suites, eval)
                # would crash on the batch-dimension mismatch or silently
                # bias it. (Same defect class as PR #275's loss-suite fix —
                # cleared here at the source.)
                self.denoiser.set_runtime_symbol_features(None)
        if predict_mask.any():
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = target_ids.reshape(-1)
            ce = F.cross_entropy(flat_logits, flat_targets, reduction="none")
            weights = torch.ones_like(ce)
            if ltr_w > 0.0 and fuse and ltr_suffix.any():
                suffix_flat = ltr_suffix.reshape(-1)
                weights = weights + (ltr_w * suffix_flat.float())
                prefix_w = float(
                    getattr(self.config, "ltr_prefix_loss_weight", 0.0) or 0.0
                )
                if prefix_w > 0.0:
                    positions = torch.arange(
                        target_ids.size(1), device=target_ids.device
                    )
                    first = target_ids[:, 0].eq(self.tokenizer.bos_id)
                    content_rank = positions.unsqueeze(0) - first.unsqueeze(1).long()
                    prefix = (content_rank >= 0) & (content_rank < 3) & ltr_suffix
                    weights = weights + (prefix_w * prefix.reshape(-1).float())
            if mdlm_row_w is not None:
                # Broadcast per-row MDLM 1/t weights onto token positions.
                seq = target_ids.size(1)
                row_flat = mdlm_row_w.unsqueeze(1).expand(-1, seq).reshape(-1)
                weights = weights * row_flat
            mask_flat = predict_mask.reshape(-1)
            mask_loss = (ce * weights)[mask_flat].mean()
            # Diagnostic only: preserve per-record masked token loss without
            # changing the scalar objective or its gradient reduction.
            row_values = (ce * weights).reshape(target_ids.shape)
            row_mask = predict_mask
            row_counts = row_mask.sum(dim=1).clamp_min(1)
            self._last_example_token_losses = (
                ((row_values * row_mask).sum(dim=1) / row_counts)
                .detach()
                .cpu()
                .tolist()
            )

            # SLM-138: deep supervision over per-recursion logits.
            if depth_logits is not None and ds_weights:
                depth_losses: list[torch.Tensor] = []
                # RSC-A06/SLM-242: unreachable in an invalid state --
                # TwoTowerConfig.__post_init__ / validate_recursive_depth_supervision_weights
                # rejects any non-empty ds_weights whose length != recursive_steps
                # before construction, and recursive_outputs() always returns
                # exactly recursive_steps depth_logits, so the two lengths always
                # match here. Kept byte-identical rather than removed (non-goal:
                # no runtime behavior change beyond the validation gate).
                # schedule-guard: allow TRUNCATE reason=RSC-A06/SLM-242-guarded-at-config-time test=tests/test_models/test_twotower_schedule_policy.py::test_length_mismatch_rejected_before_reaching_training_loss
                usable = min(len(depth_logits), len(ds_weights))
                # RSC-A06/SLM-242: unreachable in an invalid state -- positive_sum_vector
                # rejects a non-empty, all-zero ds_weights at config-validation time,
                # so total_w > 0 always holds whenever this branch is entered.
                total_w = sum(ds_weights[:usable])
                # schedule-guard: allow UNGUARDED_SUM reason=RSC-A06/SLM-242-guarded-at-config-time test=tests/test_models/test_twotower_schedule_policy.py::test_all_zero_weights_rejected_before_reaching_training_loss
                if total_w > 0.0:
                    # RSC-A06/SLM-242 KNOWN BEHAVIOR DEFECT (not fixed by this
                    # change -- see docs/design/rsc-a06-numeric-schedule-validation-20260721.md
                    # "Found defects" #1): w is read only via total_w's sum() above
                    # and is never multiplied into d_loss below, so every
                    # supervised depth contributes to `normalized` unweighted;
                    # only the aggregate sum(ds_weights) changes the overall
                    # term's scale, not the per-depth ratio the config appears to
                    # configure. A validation gate cannot fix a loss-math bug and
                    # this issue's non-goals forbid changing it inline.
                    # schedule-guard: allow UNUSED_LOOP_WEIGHT reason=known-defect-tracked-separately-not-fixed-here test=tests/test_models/test_twotower_schedule_policy.py::test_per_depth_weight_ratio_is_not_applied_known_defect
                    for d, w in enumerate(ds_weights[:usable]):
                        d_logits = depth_logits[d]
                        d_flat = d_logits.reshape(-1, d_logits.size(-1))
                        d_ce = F.cross_entropy(d_flat, flat_targets, reduction="none")
                        d_loss = (d_ce * weights)[mask_flat].mean()
                        depth_losses.append(d_loss)
                        self.last_training_metrics[f"recursive_depth_loss_{d}"] = float(
                            d_loss.detach().cpu()
                        )
                    normalized = torch.stack(depth_losses).sum() / total_w
                    mask_loss = mask_loss + normalized
                    self.last_training_metrics["recursive_depth_supervision_loss"] = (
                        float(normalized.detach().cpu())
                    )
        else:
            mask_loss = logits.sum() * 0.0
            self._last_example_token_losses = [0.0] * len(batch)

        length_w = float(
            getattr(self.config, "diffusion_length_loss_weight", 0.0) or 0.0
        )
        if self.length_head is not None and bucket_targets is not None and length_w > 0:
            length_logits = self.length_head(self._pool_context(ctx, ctx_pad))
            mask_loss = mask_loss + length_w * F.cross_entropy(
                length_logits, bucket_targets
            )

        fid_w = float(getattr(self.config, "fidelity_loss_weight", 0.0) or 0.0)
        if fid_w > 0.0 and predict_mask.any():
            ph_ids: set[int] = set()
            try:
                from slm_training.models.choice_tokenizer import is_choice_tokenizer
                from slm_training.models.dsl_tokenizer import (
                    SymbolTable,
                    TokenKind,
                    is_dsl_native_tokenizer,
                )

                if is_choice_tokenizer(self.tokenizer):
                    # Choice codec always uses slot pointers for placeholders.
                    ph_ids |= self.tokenizer.kind_ids(TokenKind.SYM)
                elif is_dsl_native_tokenizer(self.tokenizer):
                    if bool(getattr(self.config, "use_symbol_table", True)):
                        for r in batch:
                            table = SymbolTable.from_placeholders(
                                list(r.placeholders or []),
                                max_slots=self.tokenizer.sym_slots,
                            )
                            for i, _ph in enumerate(table.placeholders):
                                ph_ids.add(self.tokenizer.sym_id(i))
                    else:
                        ph_ids |= self.tokenizer.kind_ids(TokenKind.BYTE)
                        ph_ids |= self.tokenizer.kind_ids(TokenKind.LIT)
            except Exception:  # noqa: BLE001
                pass
            if not ph_ids:
                for r in batch:
                    for ph in r.placeholders or []:
                        for tid in self.tokenizer.encode(f'"{ph}"', add_special=False):
                            ph_ids.add(tid)
            if not ph_ids:
                ph_ids = self._placeholder_ids()
            if ph_ids:
                ph_mask = predict_mask.clone()
                # Vectorized membership via isin.
                ph_tensor = torch.tensor(
                    sorted(ph_ids), device=target_ids.device, dtype=target_ids.dtype
                )
                ph_mask &= torch.isin(target_ids, ph_tensor)
                if ph_mask.any():
                    fid_loss = F.cross_entropy(logits[ph_mask], target_ids[ph_mask])
                    mask_loss = mask_loss + fid_w * fid_loss
                    boundary_w = float(
                        getattr(self.config, "symbol_boundary_loss_weight", 0.0) or 0.0
                    )
                    if boundary_w > 0.0:
                        boundary = ph_mask.clone()
                        boundary[:, 1:] |= ph_mask[:, :-1]
                        boundary[:, :-1] |= ph_mask[:, 1:]
                        boundary &= predict_mask
                        mask_loss = mask_loss + boundary_w * F.cross_entropy(
                            logits[boundary], target_ids[boundary]
                        )

        # Legacy second-forward LTR when fuse disabled.
        if ltr_w > 0.0 and target_ids.size(1) >= 2 and not fuse:
            bsz, seq = target_ids.shape
            ltr_noisy = target_ids.clone()
            ltr_mask = torch.zeros_like(target_ids, dtype=torch.bool)
            for i in range(bsz):
                first_content = (
                    1 if int(target_ids[i, 0]) == self.tokenizer.bos_id else 0
                )
                cut = self._rng.randint(first_content, max(first_content, seq - 1))
                ltr_noisy[i, cut:] = self.tokenizer.mask_id
                for j in range(cut, seq):
                    if int(target_ids[i, j]) == self.tokenizer.pad_id:
                        break
                    ltr_mask[i, j] = True
            ltr_logits = self.denoiser(
                ltr_noisy, ctx, pad_id=self.tokenizer.pad_id, ctx_pad_mask=ctx_pad
            )
            if ltr_mask.any():
                ltr_loss = F.cross_entropy(ltr_logits[ltr_mask], target_ids[ltr_mask])
            else:
                ltr_loss = mask_loss * 0.0
            mask_loss = mask_loss + ltr_w * ltr_loss

        alignment_w = float(
            getattr(self.config, "compiler_alignment_loss_weight", 0.0) or 0.0
        )
        aligned_rows = 0
        if alignment_w > 0.0:
            from slm_training.dsl.grammar.fastpath.compiler_draft import (
                gold_compiler_decisions,
            )

            stratified = bool(
                getattr(self.config, "compiler_alignment_stratified", False)
            )
            semantic_exhaustive = bool(
                getattr(self.config, "compiler_alignment_semantic_exhaustive", False)
            )
            aligned_canvases: list[torch.Tensor] = []
            aligned_targets: list[int] = []
            aligned_positions: list[int] = []
            aligned_context_rows: list[int] = []
            aligned_kinds: list[str] = []
            aligned_candidate_ids: list[tuple[int, ...]] = []
            kind_rows: dict[str, int] = {}
            for row, record in enumerate(batch):
                target_key = tuple(
                    int(token_id) for token_id in target_ids[row].tolist()
                )
                contract_key = tuple(record.placeholders or ())
                key = (target_key, contract_key)
                decisions = self._compiler_decision_cache.get(key)
                if decisions is None:
                    decisions = gold_compiler_decisions(
                        self.tokenizer,
                        target_key,
                        slot_contract=list(contract_key),
                    )
                    self._compiler_decision_cache[key] = decisions
                if not decisions:
                    continue
                if stratified:
                    by_kind: dict[str, list[Any]] = {}
                    for decision in decisions:
                        if not (semantic_exhaustive and decision.is_semantic_role):
                            by_kind.setdefault(decision.kind, []).append(decision)
                    selected = (
                        [
                            decision
                            for decision in decisions
                            if decision.is_semantic_role
                        ]
                        if semantic_exhaustive
                        else []
                    ) + [
                        choices[self._rng.randrange(len(choices))]
                        for choices in by_kind.values()
                    ]
                else:
                    selected = [decisions[self._rng.randrange(len(decisions))]]
                for decision in selected:
                    cut = int(decision.position)
                    canvas = target_ids[row].clone()
                    canvas[cut:] = self.tokenizer.mask_id
                    canvas[target_ids[row].eq(self.tokenizer.pad_id)] = (
                        self.tokenizer.pad_id
                    )
                    aligned_canvases.append(canvas)
                    aligned_targets.append(int(target_ids[row, cut]))
                    aligned_positions.append(cut)
                    aligned_context_rows.append(row)
                    aligned_kinds.append(decision.kind)
                    aligned_candidate_ids.append(decision.candidate_ids)
                    kind_rows[decision.kind] = kind_rows.get(decision.kind, 0) + 1
            aligned_rows = len(aligned_canvases)
            if aligned_canvases:
                row_index = torch.tensor(
                    aligned_context_rows, device=ctx.device, dtype=torch.long
                )
                aligned_noisy = torch.stack(aligned_canvases)
                aligned_logits = self.denoiser(
                    aligned_noisy,
                    ctx.index_select(0, row_index),
                    pad_id=self.tokenizer.pad_id,
                    ctx_pad_mask=(
                        ctx_pad.index_select(0, row_index)
                        if ctx_pad is not None
                        else None
                    ),
                )
                margin = float(
                    getattr(self.config, "compiler_alignment_margin", 0.0) or 0.0
                )
                cross_entropy_losses: list[torch.Tensor] = []
                margin_losses: list[torch.Tensor] = []
                for index, (position, target, candidates) in enumerate(
                    zip(
                        aligned_positions,
                        aligned_targets,
                        aligned_candidate_ids,
                        strict=True,
                    )
                ):
                    candidate_logits = aligned_logits[index, position][
                        torch.tensor(
                            candidates,
                            device=aligned_logits.device,
                            dtype=torch.long,
                        )
                    ]
                    gold_index = candidates.index(int(target))
                    cross_entropy_losses.append(
                        F.cross_entropy(
                            candidate_logits.unsqueeze(0),
                            torch.tensor(
                                [gold_index],
                                device=aligned_logits.device,
                                dtype=torch.long,
                            ),
                        )
                    )
                    alternatives = torch.cat(
                        (
                            candidate_logits[:gold_index],
                            candidate_logits[gold_index + 1 :],
                        )
                    )
                    margin_losses.append(
                        F.relu(
                            margin - candidate_logits[gold_index] + alternatives.max()
                        )
                        if margin > 0.0
                        else candidate_logits[gold_index] * 0.0
                    )
                cross_entropy_tensor = torch.stack(cross_entropy_losses)
                margin_tensor = torch.stack(margin_losses)
                alignment_losses = cross_entropy_tensor + margin_tensor
                alignment_loss = alignment_losses.mean()
                mask_loss = mask_loss + alignment_w * alignment_loss
                kind_losses = {
                    kind: float(
                        alignment_losses[
                            torch.as_tensor(
                                [item == kind for item in aligned_kinds],
                                device=alignment_losses.device,
                            )
                        ]
                        .mean()
                        .detach()
                        .cpu()
                    )
                    for kind in kind_rows
                }
            else:
                kind_losses = {}
            self.last_training_metrics = {
                "compiler_alignment_rows": aligned_rows,
                "compiler_alignment_loss": (
                    float(alignment_loss.detach().cpu()) if aligned_canvases else 0.0
                ),
                "compiler_alignment_candidate_count_mean": (
                    sum(map(len, aligned_candidate_ids)) / aligned_rows
                    if aligned_rows
                    else 0.0
                ),
                "compiler_alignment_candidate_count_max": max(
                    map(len, aligned_candidate_ids), default=0
                ),
                "compiler_alignment_cross_entropy": (
                    float(cross_entropy_tensor.mean().detach().cpu())
                    if aligned_canvases
                    else 0.0
                ),
                "compiler_alignment_margin_loss": (
                    float(margin_tensor.mean().detach().cpu())
                    if aligned_canvases
                    else 0.0
                ),
                "compiler_alignment_margin_violation_rate": (
                    float(margin_tensor.gt(0).float().mean().detach().cpu())
                    if aligned_canvases
                    else 0.0
                ),
                **{
                    f"compiler_alignment_{kind}_rows": count
                    for kind, count in sorted(kind_rows.items())
                },
                **{
                    f"compiler_alignment_{kind}_loss": loss
                    for kind, loss in sorted(kind_losses.items())
                },
            }

        inventory_w = float(
            getattr(self.config, "component_inventory_loss_weight", 0.0) or 0.0
        )
        if inventory_w > 0.0 and self.component_inventory_head is not None:
            component_ids = self._component_inventory_token_ids()
            if component_ids:
                index = torch.as_tensor(
                    component_ids, device=target_ids.device, dtype=torch.long
                )
                inventory_logits = self.component_inventory_head(
                    self._pool_context(ctx, ctx_pad)
                ).index_select(1, index)
                inventory_targets = torch.stack(
                    [
                        index[:, None]
                        .eq(row[None, :])
                        .any(dim=1)
                        .to(inventory_logits.dtype)
                        for row in target_ids
                    ]
                )
                positive_count = inventory_targets.sum(dim=1).clamp_min(1.0)
                negative_count = (1.0 - inventory_targets).sum(dim=1).clamp_min(1.0)
                raw = F.binary_cross_entropy_with_logits(
                    inventory_logits, inventory_targets, reduction="none"
                )
                positive_loss = (raw * inventory_targets).sum(dim=1) / positive_count
                negative_loss = (raw * (1.0 - inventory_targets)).sum(
                    dim=1
                ) / negative_count
                inventory_loss = (positive_loss + negative_loss).mean()
                mask_loss = mask_loss + inventory_w * inventory_loss

                recalls: list[torch.Tensor] = []
                for row, count in zip(
                    inventory_targets, positive_count.to(torch.long), strict=True
                ):
                    top = inventory_logits[len(recalls)].topk(int(count.item())).indices
                    recalls.append(row.index_select(0, top).sum() / count)
                positive_scores = (inventory_logits * inventory_targets).sum(
                    dim=1
                ) / positive_count
                negative_scores = (inventory_logits * (1.0 - inventory_targets)).sum(
                    dim=1
                ) / negative_count
                self.last_training_metrics.update(
                    {
                        "component_inventory_loss": float(
                            inventory_loss.detach().cpu()
                        ),
                        "component_inventory_topk_recall": float(
                            torch.stack(recalls).mean().detach().cpu()
                        ),
                        "component_inventory_score_margin": float(
                            (positive_scores - negative_scores).mean().detach().cpu()
                        ),
                        "component_inventory_positive_count_mean": float(
                            positive_count.float().mean().detach().cpu()
                        ),
                    }
                )

        plan_w = float(getattr(self.config, "component_plan_loss_weight", 0.0) or 0.0)
        if plan_w > 0.0 and self.component_plan_head is not None:
            component_ids = self._component_inventory_token_ids()
            if component_ids:
                from slm_training.dsl.grammar.fastpath.compiler_draft import (
                    gold_compiler_decisions,
                )

                component_index = {
                    token_id: i for i, token_id in enumerate(component_ids)
                }
                root_targets: list[int] = []
                bound_targets = torch.zeros(
                    len(batch), len(component_ids), device=ctx.device
                )
                for row, record in enumerate(batch):
                    target_key = tuple(
                        int(token_id) for token_id in target_ids[row].tolist()
                    )
                    contract_key = tuple(record.placeholders or ())
                    cache_key = (target_key, contract_key)
                    decisions = self._compiler_decision_cache.get(cache_key)
                    if decisions is None:
                        decisions = gold_compiler_decisions(
                            self.tokenizer,
                            target_key,
                            slot_contract=list(contract_key),
                        )
                        self._compiler_decision_cache[cache_key] = decisions
                    root_target = -1
                    for decision in decisions:
                        token_id = int(target_ids[row, int(decision.position)])
                        index = component_index.get(token_id)
                        if index is None:
                            continue
                        if decision.kind == "component_root":
                            root_target = index
                        elif decision.kind == "component_bound":
                            bound_targets[row, index] += 1.0
                    root_targets.append(root_target)

                index = torch.as_tensor(component_ids, device=ctx.device)
                plan_logits = self.component_plan_head(
                    self._pool_context(ctx, ctx_pad)
                ).view(len(batch), 2, self.tokenizer.vocab_size)
                root_logits = plan_logits[:, 0].index_select(1, index)
                bound_logits = plan_logits[:, 1].index_select(1, index)
                root_tensor = torch.as_tensor(root_targets, device=ctx.device)
                root_mask = root_tensor.ge(0)
                root_loss = (
                    F.cross_entropy(root_logits[root_mask], root_tensor[root_mask])
                    if root_mask.any()
                    else root_logits.sum() * 0.0
                )
                bound_rates = F.softplus(bound_logits)
                bound_raw = F.poisson_nll_loss(
                    bound_rates,
                    bound_targets,
                    log_input=False,
                    full=True,
                    reduction="none",
                )
                bound_positive = bound_targets.gt(0)
                bound_negative = ~bound_positive
                positive_loss = (
                    bound_raw[bound_positive].mean()
                    if bound_positive.any()
                    else bound_raw.sum() * 0.0
                )
                negative_loss = bound_raw[bound_negative].mean()
                bound_loss = positive_loss + negative_loss
                plan_loss = root_loss + bound_loss
                mask_loss = mask_loss + plan_w * plan_loss
                root_accuracy = (
                    root_logits[root_mask]
                    .argmax(dim=1)
                    .eq(root_tensor[root_mask])
                    .float()
                    .mean()
                    if root_mask.any()
                    else root_logits.new_zeros(())
                )
                bound_recalls: list[torch.Tensor] = []
                for logits_row, target_row in zip(
                    bound_logits, bound_targets, strict=True
                ):
                    positive_count = int(target_row.gt(0).sum().item())
                    if positive_count:
                        top = logits_row.topk(positive_count).indices
                        bound_recalls.append(target_row.gt(0)[top].float().mean())
                bound_recall = (
                    torch.stack(bound_recalls).mean()
                    if bound_recalls
                    else bound_logits.new_zeros(())
                )
                self.last_training_metrics.update(
                    {
                        "component_plan_loss": float(plan_loss.detach().cpu()),
                        "component_plan_root_loss": float(root_loss.detach().cpu()),
                        "component_plan_bound_loss": float(bound_loss.detach().cpu()),
                        "component_plan_root_accuracy": float(
                            root_accuracy.detach().cpu()
                        ),
                        "component_plan_bound_topk_recall": float(
                            bound_recall.detach().cpu()
                        ),
                        "component_plan_bound_count_mae": float(
                            (bound_rates - bound_targets).abs().mean().detach().cpu()
                        ),
                    }
                )

        slot_component_w = float(
            getattr(self.config, "slot_component_loss_weight", 0.0) or 0.0
        )
        if slot_component_w > 0.0 and self.slot_component_head is not None:
            component_index = self._component_name_index()
            slots: list[str] = []
            next_slots: list[str | None] = []
            slot_rows: list[int] = []
            slot_targets: list[int] = []
            for row, record in enumerate(batch):
                owners = self._slot_component_owners(record.openui)
                slot_texts = self._slot_component_texts(list(record.placeholders))
                for slot_index, (slot, slot_text) in enumerate(
                    zip(record.placeholders, slot_texts, strict=True)
                ):
                    target = component_index.get(owners.get(slot, ""))
                    if target is None:
                        continue
                    slots.append(slot_text)
                    next_slots.append(
                        record.placeholders[slot_index + 1]
                        if slot_index + 1 < len(record.placeholders)
                        else None
                    )
                    slot_rows.append(row)
                    slot_targets.append(target)
            if slots:
                rows_tensor = torch.as_tensor(slot_rows, device=ctx.device)
                targets_tensor = torch.as_tensor(slot_targets, device=ctx.device)
                slot_logits = self._slot_component_logits(
                    slots,
                    ctx,
                    ctx_pad,
                    rows_tensor,
                    next_slots=(
                        next_slots
                        if self.config.slot_component_pair_interaction
                        else None
                    ),
                )
                slot_raw = F.cross_entropy(
                    slot_logits,
                    targets_tensor,
                    weight=(
                        torch.as_tensor(
                            self.config.slot_component_class_weights,
                            dtype=slot_logits.dtype,
                            device=slot_logits.device,
                        )
                        if self.config.slot_component_class_weights
                        else None
                    ),
                    reduction="none",
                )
                focal_gamma = float(
                    getattr(self.config, "slot_component_focal_gamma", 0.0) or 0.0
                )
                slot_loss = (
                    ((1.0 - (-slot_raw).exp()).pow(focal_gamma) * slot_raw).mean()
                    if focal_gamma > 0.0
                    else slot_raw.mean()
                )
                mask_loss = mask_loss + slot_component_w * slot_loss
                target_counts = torch.bincount(targets_tensor)
                self.last_training_metrics.update(
                    {
                        "slot_component_loss": float(slot_loss.detach().cpu()),
                        "slot_component_accuracy": float(
                            slot_logits.argmax(dim=1)
                            .eq(targets_tensor)
                            .float()
                            .mean()
                            .detach()
                            .cpu()
                        ),
                        "slot_component_majority_baseline": float(
                            target_counts.max().float().div(len(slots)).cpu()
                        ),
                        "slot_component_rows": len(slots),
                    }
                )

        edge_w = float(getattr(self.config, "component_edge_loss_weight", 0.0) or 0.0)
        if edge_w > 0.0 and self.component_edge_head is not None:
            from slm_training.dsl.grammar.fastpath.compiler_draft import (
                semantic_component_edges,
            )
            from slm_training.dsl.parser import parse

            component_ids = self._component_inventory_token_ids()
            component_index = {
                token_id: index for index, token_id in enumerate(component_ids)
            }
            edge_targets = torch.zeros(
                len(batch),
                len(component_ids),
                len(component_ids),
                device=ctx.device,
            )
            for row, record in enumerate(batch):
                edges = self._component_edge_cache.get(record.openui)
                if edges is None:
                    edges = semantic_component_edges(
                        parse(record.openui).root, self.tokenizer
                    )
                    self._component_edge_cache[record.openui] = edges
                for parent_id, child_id in edges:
                    parent = component_index.get(parent_id)
                    child = component_index.get(child_id)
                    if parent is not None and child is not None:
                        edge_targets[row, parent, child] = 1.0

            edge_logits = self.component_edge_head(
                self._pool_context(ctx, ctx_pad)
            ).view_as(edge_targets)
            raw_edge_loss = F.binary_cross_entropy_with_logits(
                edge_logits, edge_targets, reduction="none"
            )
            positive = edge_targets.bool()
            negative = ~positive
            positive_loss = (
                raw_edge_loss[positive].mean()
                if positive.any()
                else raw_edge_loss.sum() * 0.0
            )
            negative_loss = raw_edge_loss[negative].mean()
            edge_loss = positive_loss + negative_loss
            mask_loss = mask_loss + edge_w * edge_loss
            recalls: list[torch.Tensor] = []
            flat_logits = edge_logits.flatten(1)
            flat_targets = edge_targets.flatten(1)
            for logits_row, target_row in zip(flat_logits, flat_targets, strict=True):
                count = int(target_row.sum().item())
                if count:
                    top = logits_row.topk(count).indices
                    recalls.append(target_row.index_select(0, top).mean())
            edge_recall = (
                torch.stack(recalls).mean() if recalls else edge_logits.new_zeros(())
            )
            self.last_training_metrics.update(
                {
                    "component_edge_loss": float(edge_loss.detach().cpu()),
                    "component_edge_topk_recall": float(edge_recall.detach().cpu()),
                    "component_edge_positive_count_mean": float(
                        edge_targets.sum(dim=(1, 2)).mean().detach().cpu()
                    ),
                }
            )

        binder_arity_w = float(
            getattr(self.config, "binder_arity_loss_weight", 0.0) or 0.0
        )
        if binder_arity_w > 0.0 and self.binder_arity_head is not None:
            from slm_training.dsl.grammar.fastpath.compiler_draft import (
                binder_reference_arities,
            )

            binder_ids = self._binder_component_token_ids()
            binder_index = {
                token_id: index for index, token_id in enumerate(binder_ids)
            }
            arity_logits = self.binder_arity_head(
                self._pool_context(ctx, ctx_pad).detach()
            ).view(len(batch), len(binder_ids), len(binder_ids) + 1)
            arity_losses: list[torch.Tensor] = []
            arity_hits: list[torch.Tensor] = []
            for row in range(len(batch)):
                target_key = tuple(
                    int(token_id) for token_id in target_ids[row].tolist()
                )
                for binder_id, count in binder_reference_arities(
                    self.tokenizer, target_key
                ):
                    binder = binder_index.get(binder_id)
                    if binder is None:
                        continue
                    target = min(int(count), len(binder_ids))
                    scores = arity_logits[row, binder]
                    target_tensor = torch.as_tensor([target], device=ctx.device)
                    arity_losses.append(F.cross_entropy(scores[None, :], target_tensor))
                    arity_hits.append(scores.argmax().eq(target_tensor[0]).float())
            arity_loss = (
                torch.stack(arity_losses).mean()
                if arity_losses
                else arity_logits.sum() * 0.0
            )
            self._detached_auxiliary_loss = binder_arity_w * arity_loss
            self.last_training_metrics.update(
                {
                    "binder_arity_loss": float(arity_loss.detach().cpu()),
                    "binder_arity_accuracy": float(
                        torch.stack(arity_hits).mean().detach().cpu()
                        if arity_hits
                        else 0.0
                    ),
                    "binder_arity_rows": len(arity_losses),
                }
            )

        root_arity_w = float(
            getattr(self.config, "root_reference_arity_loss_weight", 0.0) or 0.0
        )
        if root_arity_w > 0.0 and self.root_reference_arity_head is not None:
            from slm_training.models.choice_tokenizer import (
                structural_root_reference_arity_target,
            )

            root_logits = self.root_reference_arity_head(
                self._pool_context(ctx, ctx_pad).detach()
            )
            root_losses: list[torch.Tensor] = []
            root_hits: list[torch.Tensor] = []
            root_class_counts: list[int] = []
            for row, record in enumerate(batch):
                target_and_bound = structural_root_reference_arity_target(
                    self.tokenizer,
                    target_ids[row].tolist(),
                    slot_count=len(record.placeholders or ()),
                )
                if target_and_bound is None:
                    continue
                target, section_count = target_and_bound
                target = min(int(target), root_logits.size(1) - 1)
                valid_max = min(
                    max(target, int(section_count)), root_logits.size(1) - 1
                )
                bounded_logits = root_logits[row : row + 1, : valid_max + 1]
                target_tensor = torch.as_tensor([target], device=ctx.device)
                root_losses.append(F.cross_entropy(bounded_logits, target_tensor))
                root_hits.append(
                    bounded_logits[0].argmax().eq(target_tensor[0]).float()
                )
                root_class_counts.append(valid_max + 1)
            root_loss = (
                torch.stack(root_losses).mean()
                if root_losses
                else root_logits.sum() * 0.0
            )
            detached = root_arity_w * root_loss
            if self._detached_auxiliary_loss is not None:
                detached = detached + self._detached_auxiliary_loss
            self._detached_auxiliary_loss = detached
            self.last_training_metrics.update(
                {
                    "root_reference_arity_loss": float(root_loss.detach().cpu()),
                    "root_reference_arity_accuracy": float(
                        torch.stack(root_hits).mean().detach().cpu()
                        if root_hits
                        else 0.0
                    ),
                    "root_reference_arity_rows": len(root_losses),
                    "root_reference_arity_classes_mean": (
                        sum(root_class_counts) / len(root_class_counts)
                        if root_class_counts
                        else 0.0
                    ),
                }
            )

        root_identity_w = float(
            getattr(self.config, "root_reference_identity_loss_weight", 0.0) or 0.0
        )
        root_identity_negative_w = float(
            getattr(self.config, "root_reference_identity_negative_weight", 1.0)
        )
        if root_identity_negative_w < 0.0:
            raise ValueError("root_reference_identity_negative_weight must be >= 0")
        if root_identity_w > 0.0 and self.root_reference_identity_head is not None:
            from slm_training.models.choice_tokenizer import (
                structural_root_reference_identity_target,
            )

            identity_logits = self.root_reference_identity_head(
                self._pool_context(ctx, ctx_pad).detach()
            )
            identity_losses: list[torch.Tensor] = []
            identity_exact_hits: list[torch.Tensor] = []
            identity_positive_recalls: list[torch.Tensor] = []
            identity_negative_accuracies: list[torch.Tensor] = []
            identity_class_counts: list[int] = []
            for row, record in enumerate(batch):
                target_and_bound = structural_root_reference_identity_target(
                    self.tokenizer,
                    target_ids[row].tolist(),
                    slot_count=len(record.placeholders or ()),
                )
                if target_and_bound is None:
                    continue
                references, section_count = target_and_bound
                valid_count = min(int(section_count), identity_logits.size(1))
                if valid_count <= 0:
                    continue
                target = identity_logits.new_zeros(valid_count)
                for reference in references:
                    if 0 <= reference < valid_count:
                        target[reference] = 1.0
                scores = identity_logits[row, :valid_count]
                prediction = scores >= 0.0
                element_loss = F.binary_cross_entropy_with_logits(
                    scores, target, reduction="none"
                )
                element_weights = torch.where(
                    target.bool(),
                    element_loss.new_ones(()),
                    element_loss.new_tensor(root_identity_negative_w),
                )
                identity_losses.append(
                    (element_loss * element_weights).sum()
                    / element_weights.sum().clamp_min(1.0)
                )
                identity_exact_hits.append(prediction.eq(target.bool()).all().float())
                positives = target.sum()
                identity_positive_recalls.append(
                    (prediction & target.bool()).float().sum()
                    / positives.clamp_min(1.0)
                )
                negatives = target.eq(0.0)
                if negatives.any():
                    identity_negative_accuracies.append(
                        ((~prediction) & negatives).float().sum()
                        / negatives.float().sum()
                    )
                identity_class_counts.append(valid_count)
            identity_loss = (
                torch.stack(identity_losses).mean()
                if identity_losses
                else identity_logits.sum() * 0.0
            )
            detached = root_identity_w * identity_loss
            if self._detached_auxiliary_loss is not None:
                detached = detached + self._detached_auxiliary_loss
            self._detached_auxiliary_loss = detached
            self.last_training_metrics.update(
                {
                    "root_reference_identity_loss": float(identity_loss.detach().cpu()),
                    "root_reference_identity_exact_accuracy": float(
                        torch.stack(identity_exact_hits).mean().detach().cpu()
                        if identity_exact_hits
                        else 0.0
                    ),
                    "root_reference_identity_positive_recall": float(
                        torch.stack(identity_positive_recalls).mean().detach().cpu()
                        if identity_positive_recalls
                        else 0.0
                    ),
                    "root_reference_identity_negative_accuracy": float(
                        torch.stack(identity_negative_accuracies).mean().detach().cpu()
                        if identity_negative_accuracies
                        else 0.0
                    ),
                    "root_reference_identity_negative_rows": len(
                        identity_negative_accuracies
                    ),
                    "root_reference_identity_rows": len(identity_losses),
                    "root_reference_identity_classes_mean": (
                        sum(identity_class_counts) / len(identity_class_counts)
                        if identity_class_counts
                        else 0.0
                    ),
                }
            )

        edge_alignment_w = float(
            getattr(self.config, "component_edge_alignment_loss_weight", 0.0) or 0.0
        )
        if edge_alignment_w > 0.0 and self.component_edge_head is not None:
            from slm_training.dsl.grammar.fastpath.compiler_draft import (
                active_parent_component_ids,
                gold_compiler_decisions,
            )

            component_ids = self._component_inventory_token_ids()
            component_index = {
                token_id: index for index, token_id in enumerate(component_ids)
            }
            edge_logits = self.component_edge_head(
                self._pool_context(ctx, ctx_pad)
            ).view(len(batch), len(component_ids), len(component_ids))
            alignment_losses: list[torch.Tensor] = []
            alignment_hits: list[torch.Tensor] = []
            candidate_counts: list[int] = []
            unknown_parent_rows = 0
            for row, record in enumerate(batch):
                target_key = tuple(
                    int(token_id) for token_id in target_ids[row].tolist()
                )
                contract_key = tuple(record.placeholders or ())
                cache_key = (target_key, contract_key)
                decisions = self._compiler_decision_cache.get(cache_key)
                if decisions is None:
                    decisions = gold_compiler_decisions(
                        self.tokenizer,
                        target_key,
                        slot_contract=list(contract_key),
                    )
                    self._compiler_decision_cache[cache_key] = decisions
                for decision in decisions:
                    if decision.kind != "component_bound":
                        continue
                    position = int(decision.position)
                    child = component_index.get(int(target_ids[row, position]))
                    parents = active_parent_component_ids(
                        self.tokenizer, list(target_key[:position])
                    )
                    candidates = [
                        component_index[token_id]
                        for token_id in decision.candidate_ids
                        if token_id in component_index
                    ]
                    if child is None or child not in candidates:
                        continue
                    parent_indices = [
                        component_index[token_id]
                        for token_id in parents
                        if token_id in component_index
                    ]
                    if not parent_indices:
                        unknown_parent_rows += 1
                        continue
                    candidate_tensor = torch.as_tensor(candidates, device=ctx.device)
                    parent_tensor = torch.as_tensor(parent_indices, device=ctx.device)
                    scores = (
                        edge_logits[row]
                        .index_select(0, parent_tensor)
                        .mean(dim=0)
                        .index_select(0, candidate_tensor)
                    )
                    target = torch.as_tensor(
                        [candidates.index(child)], device=ctx.device
                    )
                    alignment_losses.append(F.cross_entropy(scores[None, :], target))
                    alignment_hits.append(scores.argmax().eq(target[0]).float())
                    candidate_counts.append(len(candidates))
            edge_alignment_loss = (
                torch.stack(alignment_losses).mean()
                if alignment_losses
                else edge_logits.sum() * 0.0
            )
            mask_loss = mask_loss + edge_alignment_w * edge_alignment_loss
            self.last_training_metrics.update(
                {
                    "component_edge_alignment_loss": float(
                        edge_alignment_loss.detach().cpu()
                    ),
                    "component_edge_alignment_accuracy": float(
                        torch.stack(alignment_hits).mean().detach().cpu()
                        if alignment_hits
                        else 0.0
                    ),
                    "component_edge_alignment_rows": len(alignment_losses),
                    "component_edge_alignment_unknown_parent_rows": unknown_parent_rows,
                    "component_edge_alignment_candidate_count_mean": (
                        sum(candidate_counts) / len(candidate_counts)
                        if candidate_counts
                        else 0.0
                    ),
                }
            )

        binder_plan_w = float(
            getattr(self.config, "binder_component_plan_loss_weight", 0.0) or 0.0
        )
        if binder_plan_w > 0.0 and self.binder_component_plan_head is not None:
            from slm_training.dsl.grammar.fastpath.compiler_draft import (
                active_declaration_binder_id,
                gold_compiler_decisions,
            )

            binder_ids = self._binder_component_token_ids()
            component_ids = self._component_inventory_token_ids()
            binder_index = {
                token_id: index for index, token_id in enumerate(binder_ids)
            }
            component_index = {
                token_id: index for index, token_id in enumerate(component_ids)
            }
            plan_logits = self.binder_component_plan_head(
                self._pool_context(ctx, ctx_pad)
            ).view(len(batch), len(binder_ids), len(component_ids))
            plan_losses: list[torch.Tensor] = []
            plan_hits: list[torch.Tensor] = []
            candidate_counts: list[int] = []
            for row, record in enumerate(batch):
                target_key = tuple(
                    int(token_id) for token_id in target_ids[row].tolist()
                )
                contract_key = tuple(record.placeholders or ())
                cache_key = (target_key, contract_key)
                decisions = self._compiler_decision_cache.get(cache_key)
                if decisions is None:
                    decisions = gold_compiler_decisions(
                        self.tokenizer,
                        target_key,
                        slot_contract=list(contract_key),
                    )
                    self._compiler_decision_cache[cache_key] = decisions
                for decision in decisions:
                    if decision.kind != "component_bound":
                        continue
                    position = int(decision.position)
                    binder_id = active_declaration_binder_id(
                        self.tokenizer, list(target_key[:position])
                    )
                    binder = binder_index.get(binder_id)
                    child = component_index.get(int(target_ids[row, position]))
                    candidates = [
                        component_index[token_id]
                        for token_id in decision.candidate_ids
                        if token_id in component_index
                    ]
                    if binder is None or child is None or child not in candidates:
                        continue
                    candidate_tensor = torch.as_tensor(candidates, device=ctx.device)
                    scores = plan_logits[row, binder].index_select(0, candidate_tensor)
                    target = torch.as_tensor(
                        [candidates.index(child)], device=ctx.device
                    )
                    plan_losses.append(F.cross_entropy(scores[None, :], target))
                    plan_hits.append(scores.argmax().eq(target[0]).float())
                    candidate_counts.append(len(candidates))
            binder_plan_loss = (
                torch.stack(plan_losses).mean()
                if plan_losses
                else plan_logits.sum() * 0.0
            )
            mask_loss = mask_loss + binder_plan_w * binder_plan_loss
            self.last_training_metrics.update(
                {
                    "binder_component_plan_loss": float(
                        binder_plan_loss.detach().cpu()
                    ),
                    "binder_component_plan_accuracy": float(
                        torch.stack(plan_hits).mean().detach().cpu()
                        if plan_hits
                        else 0.0
                    ),
                    "binder_component_plan_rows": len(plan_losses),
                    "binder_component_plan_candidate_count_mean": (
                        sum(candidate_counts) / len(candidate_counts)
                        if candidate_counts
                        else 0.0
                    ),
                }
            )

        binder_topology_w = float(
            getattr(self.config, "binder_topology_loss_weight", 0.0) or 0.0
        )
        if binder_topology_w > 0.0 and self.binder_topology_head is not None:
            from slm_training.dsl.grammar.fastpath.compiler_draft import (
                active_declaration_binder_id,
                gold_compiler_decisions,
            )

            binder_ids = self._binder_component_token_ids()
            binder_index = {
                token_id: index for index, token_id in enumerate(binder_ids)
            }
            topology_logits = self.binder_topology_head(
                # Keep this auxiliary planner from rewriting the shared prompt
                # representation before its own legal-decision signal is stable.
                self._pool_context(ctx, ctx_pad).detach()
            ).view(len(batch), len(binder_ids), len(binder_ids))
            topology_losses: list[torch.Tensor] = []
            topology_hits: list[torch.Tensor] = []
            topology_candidate_counts: list[int] = []
            for row, record in enumerate(batch):
                target_key = tuple(
                    int(token_id) for token_id in target_ids[row].tolist()
                )
                contract_key = tuple(record.placeholders or ())
                cache_key = (target_key, contract_key)
                decisions = self._compiler_decision_cache.get(cache_key)
                if decisions is None:
                    decisions = gold_compiler_decisions(
                        self.tokenizer,
                        target_key,
                        slot_contract=list(contract_key),
                    )
                    self._compiler_decision_cache[cache_key] = decisions
                for decision in decisions:
                    if not decision.kind.startswith("bind_reference"):
                        continue
                    position = int(decision.position)
                    parent_id = active_declaration_binder_id(
                        self.tokenizer, list(target_key[:position])
                    )
                    parent = binder_index.get(parent_id)
                    child = binder_index.get(int(target_ids[row, position]))
                    candidates = [
                        binder_index[token_id]
                        for token_id in decision.candidate_ids
                        if token_id in binder_index
                    ]
                    if parent is None or child is None or child not in candidates:
                        continue
                    candidate_tensor = torch.as_tensor(candidates, device=ctx.device)
                    scores = topology_logits[row, parent].index_select(
                        0, candidate_tensor
                    )
                    target = torch.as_tensor(
                        [candidates.index(child)], device=ctx.device
                    )
                    topology_losses.append(F.cross_entropy(scores[None, :], target))
                    topology_hits.append(scores.argmax().eq(target[0]).float())
                    topology_candidate_counts.append(len(candidates))
            topology_loss = (
                torch.stack(topology_losses).mean()
                if topology_losses
                else topology_logits.sum() * 0.0
            )
            mask_loss = mask_loss + binder_topology_w * topology_loss
            self.last_training_metrics.update(
                {
                    "binder_topology_loss": float(topology_loss.detach().cpu()),
                    "binder_topology_accuracy": float(
                        torch.stack(topology_hits).mean().detach().cpu()
                        if topology_hits
                        else 0.0
                    ),
                    "binder_topology_rows": len(topology_losses),
                    "binder_topology_candidate_count_mean": (
                        sum(topology_candidate_counts) / len(topology_candidate_counts)
                        if topology_candidate_counts
                        else 0.0
                    ),
                }
            )

        aux_w = float(getattr(self.config, "fastpath_aux_weight", 0.0) or 0.0)
        if aux_w > 0.0 and getattr(self.config, "grammar_fastpath", False):
            # Keep this span visible in train telemetry: a silently skipped
            # structural objective cannot be evaluated or tuned honestly.
            with timed("fastpath_aux_loss"):
                from slm_training.dsl.grammar.fastpath.losses import force_align_loss

                aux = force_align_loss(
                    logits, target_ids, self.tokenizer, pad_id=self.tokenizer.pad_id
                )
                mask_loss = mask_loss + aux_w * aux

        return mask_loss

    def _training_design_md(self, design_md: str | None, key: str) -> str | None:
        """Deterministically omit DESIGN.md for a configured share of records."""
        rate = float(getattr(self.config, "design_md_dropout", 0.0) or 0.0)
        if not design_md or rate <= 0.0:
            return design_md
        if rate >= 1.0:
            return None
        seed = int(getattr(self.config, "seed", 0))
        digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
        sample = int.from_bytes(digest[:8], "big") / float(1 << 64)
        return None if sample < rate else design_md

    def _format_one_context(
        self,
        prompt: str,
        design_md: str | None,
        *,
        query_prompt: str | None = None,
        slot_contract: list[str] | None = None,
        schema: str | None = None,
        output_kind: str | None = None,
        output_category: str | None = None,
    ) -> str:
        if schema is None and getattr(self.config, "schema_in_context", False):
            from slm_training.harnesses.quality import compact_schema_snippet

            schema = compact_schema_snippet(
                budget=min(600, self.config.design_md_budget)
            )
        skeleton = None
        k = int(getattr(self.config, "retrieval_k", 0) or 0)
        if k > 0 and self.skeleton_bank:
            from slm_training.harnesses.quality import (
                format_retrieved_skeleton,
                nearest_skeletons,
            )

            hits = nearest_skeletons(self.skeleton_bank, query_prompt or prompt, k=k)
            if hits:
                skeleton = format_retrieved_skeleton(hits[0].openui)
        dm = design_md if self.config.design_md_in_context else None
        contract = (
            slot_contract
            if getattr(self.config, "slot_contract_in_context", False)
            else None
        )
        return format_context_text(
            prompt,
            dm,
            budget=self.config.design_md_budget,
            schema=schema,
            retrieved_skeleton=skeleton,
            slot_contract=contract,
            output_kind=output_kind,
            output_category=output_category,
        )

    def _decode_ids(self, ids_1d: torch.Tensor) -> str:
        placeholders = None
        if self._slot_contracts:
            # Best-effort: use first non-empty contract in the active batch.
            for c in self._slot_contracts:
                if c:
                    placeholders = list(c)
                    break
        return self._decode_openui(ids_1d, placeholders=placeholders)

    @staticmethod
    def _repair_surface_syntax(text: str) -> str:
        """Repair local token-boundary artifacts without inventing layout content."""
        parts = _QUOTED_SPAN_RE.split(text)
        for index in range(0, len(parts), 2):
            parts[index] = _REPEATED_EQUALS_RE.sub(" = ", parts[index])
            parts[index] = _DANGLING_EQUALS_RE.sub("", parts[index])
        return "".join(parts)

    @staticmethod
    def _canonical_valid_openui(text: str) -> str | None:
        """Return serialized OpenUI if parseable and non-trivial; else None."""
        try:
            from slm_training.dsl.parser import validate
        except Exception:  # noqa: BLE001
            return None
        try:
            program = validate(text)
        except Exception:  # noqa: BLE001
            return None
        ser = (program.serialized or text).strip()
        compact = ser.replace(" ", "")
        if "Stack([])" in compact or "Stack([]," in compact:
            return None
        if "Card([])" in compact:
            return None
        if "root=" not in compact and "root =" not in ser:
            return None
        return ser

    def _minimal_valid_openui(self) -> str | None:
        """Deterministic vocab-backed valid program when model decode cannot certify."""
        candidates = [
            'root = Button(":cta.label")\n',
            'root = TextContent(":hero.title")\n',
            'root = Stack([cta])\ncta = Button(":cta.label")\n',
            'root = Card([title])\ntitle = TextContent(":hero.title")\n',
        ]
        for raw in candidates:
            ser = self._canonical_valid_openui(raw)
            if ser is None:
                continue
            ids = self.tokenizer.encode(ser, add_special=False)
            if not ids or self.tokenizer.unk_id in ids:
                continue
            return ser
        return None

    def _ltr_repair_from_bos(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        length: int,
    ) -> str:
        """Speculative LTR decode from BOS with force-emit + constrained picks."""
        device = self.device_name
        repair_len = min(length, max(8, int(self.config.grammar_ltr_max_tokens)))
        repaired = torch.full(
            (1, repair_len),
            self.tokenizer.mask_id,
            dtype=torch.long,
            device=device,
        )
        repaired[0, 0] = self.tokenizer.bos_id
        unknown_r = repaired.eq(self.tokenizer.mask_id)
        repaired = self._constrained_ltr_repair(repaired, unknown_r, ctx, ctx_pad)
        return self._decode_ids(repaired[0])

    def _ensure_valid_openui(
        self,
        text: str,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        length: int,
        *,
        attempts: int | None = None,
        slot_contract: list[str] | None = None,
    ) -> str:
        """
        Repair toward a valid OpenUI string when grammar-constrained.

        Uses DFA-constrained LTR repair (pseudo speculative decoding). When
        ``grammar_finalize_validate`` is set (playground / hard certify), never
        returns invalid OpenUI — falls back to a minimal certified program or
        raises. When finalize is off (default eval), returns the best repaired
        text so parse_rate reflects real decode quality.

        E20: when ``template_fill_decode`` and a slot contract are set, prefer the
        inventory-bound skeleton over a broken model sample.

        R5: ``attempts`` defaults to ``generate_max_attempts``; callers that
        already ran a BOS LTR repair may pass ``attempts=0`` to skip a redundant
        redo and fall through to finalize/minimal.
        """
        if attempts is None:
            attempts = int(getattr(self.config, "generate_max_attempts", 3) or 3)
        ser = self._canonical_valid_openui(text)
        if ser is not None:
            return ser
        # E20: inventory-bound skeleton is a valid, fidelity-aligned fallback.
        # Prefer it over multi-attempt LTR repair (O(T) Node checks per attempt).
        if bool(getattr(self.config, "template_fill_decode", False)) and slot_contract:
            templ = build_slot_contract_template(slot_contract)
            ser = self._canonical_valid_openui(templ)
            if ser is not None:
                active = get_active_stats()
                if active is not None:
                    active.template_fallback_count += 1
                return ser
        if (
            not self.config.grammar_ltr_repair
            and not self.config.grammar_finalize_validate
        ):
            return text
        last = text
        for _ in range(max(0, int(attempts))):
            last = self._ltr_repair_from_bos(ctx, ctx_pad, length)
            ser = self._canonical_valid_openui(last)
            if ser is not None:
                return ser
        if self.config.grammar_finalize_validate:
            fallback = self._minimal_valid_openui()
            if fallback is not None:
                return fallback
            raise RuntimeError(
                "grammar_finalize_validate: model could not produce a complete valid OpenUI program"
            )
        return last

    def _constrained_ltr_repair(
        self,
        ids: torch.Tensor,
        unknown: torch.Tensor,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        *,
        slot_contract: list[str] | None = None,
    ) -> torch.Tensor:
        """Fill remaining masks left-to-right with streaming-parser filtering.

        R4: honor ``grammar_multitoken_accept`` + ``grammar_canvas_lookahead``
        so repair/BOS certify share the same forward budget as greedy LTR.

        When ``trace_recorder`` is set (promotion/distill path from main), record
        per-forward commits the same way as the pre-R4 repair loop.
        """
        use_contract = bool(
            getattr(self.config, "slot_contract_constrained_decode", False)
        )
        contract = slot_contract if use_contract else None
        length = ids.size(1)
        use_fast = bool(getattr(self.config, "grammar_fastpath", True))
        states = self._new_grammar_states(1)
        st = states[0] if states is not None else None
        pick_kw = self._pick_kwargs()
        multitoken = bool(getattr(self.config, "grammar_multitoken_accept", False))
        multitoken_max = max(
            1, int(getattr(self.config, "grammar_multitoken_max", 8) or 8)
        )
        lookahead = int(getattr(self.config, "grammar_canvas_lookahead", 0) or 0)
        bias = self._effective_structural_bias()
        tok = self.tokenizer
        stats = get_active_stats()
        rec = getattr(self, "trace_recorder", None)
        grec = getattr(self, "grammar_trace_recorder", None)
        repair_commits: list[dict] = []

        def _record_commit(
            pos: int,
            token_id: int,
            logits_1d: torch.Tensor,
            *,
            forced: bool,
            prefix: list[int] | None = None,
        ) -> None:
            if rec is None and grec is None:
                return
            if rec is not None:
                log_probs = F.log_softmax(logits_1d.float(), dim=-1)
                repair_commits.append(
                    {
                        "t": pos,
                        "id": int(token_id),
                        "lp": float(log_probs[int(token_id)].item()),
                        "forced": forced,
                        "phase": "ltr_repair",
                    }
                )
            if grec is not None and prefix is not None:
                try:
                    from slm_training.harnesses.distill.grammar_trace import (
                        legal_action_ids_from_state,
                        state_fingerprint,
                    )

                    legal_cov = legal_action_ids_from_state(tok, st, prefix)
                    if legal_cov is not None:
                        legal_ids, coverage = legal_cov
                        token_str = tok.id_to_token.get(
                            int(token_id), str(int(token_id))
                        )
                        grec.record(
                            state_fingerprint=state_fingerprint(
                                prefix_ids=prefix,
                                legal_action_ids=legal_ids,
                                coverage=coverage,
                            ),
                            state_signature_version="1",
                            legal_action_ids=legal_ids,
                            compiler_coverage=coverage,
                            selected_action_id=token_str,
                            logits_or_energies=logits_1d.detach().tolist()
                            if grec.capture_logits
                            else None,
                            convention="logit",
                            scope_signature="",
                            expected_type=None,
                            template_signature=None,
                        )
                except (
                    AttributeError,
                    ImportError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as exc:
                    warnings.warn(
                        f"grammar trace record skipped ({type(exc).__name__}: {exc}) "
                        f"at decision {len(repair_commits)}",
                        stacklevel=2,
                    )

        t = 0
        while t < length:
            # Contract/template decode may leave only a few masked slots in a
            # long padded canvas. Once the final slot is committed, stop the
            # LTR scan instead of spending one denoiser forward per trailing
            # known/pad position.
            if not bool(unknown.any().item()):
                break
            if not bool(unknown[0, t].item()):
                if st is not None and len(st.prefix_ids) == t:
                    st.advance_token(tok, int(ids[0, t].item()))
                t += 1
                continue
            prefix = ids[0, :t].tolist()
            if st is not None:
                st.sync_ids(tok, prefix)
            forced = force_emit_token_id(tok, prefix, state=st) if use_fast else None
            # P4: truncate canvas to prefix + lookahead for the forward.
            if lookahead > 0:
                end = min(length, t + lookahead)
                fwd_ids = ids[:, :end]
            else:
                end = length
                fwd_ids = ids
            logits = self._denoiser_forward(fwd_ids, ctx, ctx_pad)
            if bias:
                logits = apply_structural_bias(logits, tok, bias=bias)
            local_t = min(t, end - 1)
            row = logits[0, local_t].clone()
            row[tok.mask_id] = -1e9
            row[tok.pad_id] = -1e9
            choice = pick_constrained_token(
                row,
                tok,
                prefix,
                top_k=self.config.grammar_top_k,
                forced_token_id=forced,
                slot_contract=contract,
                state=st,
                **pick_kw,
            )
            # Commit-boundary invariant: a bare root binding must continue with
            # assignment, never newline. This protects the LTR path from a
            # stale/broad picker admission that would make the next state
            # irrecoverable.
            if choice is not None and tok.id_to_token.get(int(choice), "") == "NL":
                current = self._decode_ids(prefix)
                if current.rstrip().strip() == "root" and "=" not in current:
                    assign_id = tok.token_to_id.get("=")
                    if assign_id is not None:
                        choice = int(assign_id)
            if choice is None:
                # Recovery for a persistent-engine divergence: rebuild the
                # incremental DFA at the exact decoded prefix and retry the
                # same logits before declaring a grammar dead end.
                if st is not None:
                    current = self._decode_ids(prefix)
                    try:
                        st.engine.set_prefix(current)
                        st.prefix_text = current
                        st.prefix_ids = list(prefix)
                        st.clear_position_memo()
                        choice = pick_constrained_token(
                            row,
                            tok,
                            prefix,
                            top_k=self.config.grammar_top_k,
                            slot_contract=contract,
                            state=st,
                            **pick_kw,
                        )
                    except Exception:  # noqa: BLE001
                        choice = None
            if choice is None:
                # No legal continuation — pad out and stop rather than emit garbage.
                if stats is not None:
                    stats.constrained_dead_ends += 1
                    stats.constrained_dead_end_last_position = int(t)
                    stats.constrained_dead_end_candidate_count += max(
                        0, int(stats.constrained_last_legal_candidates)
                    )
                    stats.constrained_dead_end_traces.append(
                        {
                            "position": int(t),
                            "prefix_text": self._decode_ids(ids[0, :t].tolist()),
                            "prefix_tokens": [
                                tok.id_to_token.get(int(token_id), "")
                                for token_id in ids[0, :t].tolist()
                            ],
                            "top_tokens": [
                                {
                                    "token": tok.id_to_token.get(int(token_id), ""),
                                    "logit": float(row[int(token_id)].item()),
                                }
                                for token_id in torch.topk(
                                    row, k=min(8, int(row.numel()))
                                ).indices.tolist()
                            ],
                        }
                    )
                    if forced is not None:
                        stats.constrained_dead_end_forced_rank = int(
                            1 + (row > row[int(forced)]).sum().item()
                        )
                ids[0, t:] = tok.pad_id
                unknown[0, t:] = False
                if rec is not None:
                    rec.event("repair_dead_end", position=t)
                break
            if stats is not None and len(stats.constrained_selection_traces) < 64:
                chosen_token = tok.id_to_token.get(int(choice), "")
                argmax_id = int(row.argmax().item())
                stats.constrained_selection_traces.append(
                    {
                        "position": int(t),
                        "prefix_text": self._decode_ids(ids[0, :t].tolist()),
                        "chosen_token": chosen_token,
                        "chosen_id": int(choice),
                        "model_argmax": tok.id_to_token.get(argmax_id, ""),
                        "model_argmax_id": argmax_id,
                        "legal_candidates": int(
                            stats.constrained_last_legal_candidates
                        ),
                        "forced": bool(forced is not None),
                        "phase": "ltr_repair",
                    }
                )
            ids[0, t] = choice
            unknown[0, t] = False
            if stats is not None and tok.id_to_token.get(int(choice), "") == "NL":
                stats.newline_commit_traces.append(
                    {
                        "position": int(t),
                        "prefix_text": self._decode_ids(ids[0, :t].tolist()),
                        "prefix_tokens": [
                            tok.id_to_token.get(int(token_id), "")
                            for token_id in ids[0, :t].tolist()
                        ],
                        "phase": "ltr_repair",
                    }
                )
            if st is not None:
                st.advance_token(tok, int(choice))
            if stats is not None:
                stats.tokens_emitted += 1
            _record_commit(
                t,
                int(choice),
                logits[0, local_t],
                forced=forced is not None,
                prefix=prefix,
            )
            if choice == tok.eos_id:
                if t + 1 < length:
                    ids[0, t + 1 :] = tok.pad_id
                    unknown[0, t + 1 :] = False
                break

            advance = 1
            # P3: greedily accept consecutive unknown positions from the same
            # forward without re-running the denoiser.
            if multitoken:
                max_run = min(multitoken_max, end - t - 1)
                for step in range(1, max_run + 1):
                    pos = t + step
                    if pos >= logits.size(1) or not bool(unknown[0, pos].item()):
                        break
                    logits_pos = logits[0, pos].clone()
                    logits_pos[tok.mask_id] = -1e9
                    logits_pos[tok.pad_id] = -1e9
                    nxt = pick_constrained_token(
                        logits_pos,
                        tok,
                        ids[0, :pos].tolist(),
                        top_k=self.config.grammar_top_k,
                        slot_contract=contract,
                        state=st,
                        **pick_kw,
                    )
                    if nxt is None:
                        break
                    ids[0, pos] = nxt
                    unknown[0, pos] = False
                    if st is not None:
                        st.advance_token(tok, int(nxt))
                    if stats is not None:
                        stats.tokens_emitted += 1
                        stats.accepted_run_tokens += 1
                    _record_commit(
                        pos,
                        int(nxt),
                        logits[0, pos],
                        forced=False,
                        prefix=ids[0, :pos].tolist(),
                    )
                    advance = step + 1
                    if nxt == tok.eos_id:
                        if pos + 1 < length:
                            ids[0, pos + 1 :] = tok.pad_id
                            unknown[0, pos + 1 :] = False
                        break
            t += advance
        if rec is not None and repair_commits:
            rec.step(
                "ltr_repair",
                canvas=ids[0].tolist(),
                unknown=unknown[0].tolist(),
                commits=repair_commits,
            )
        return ids

    def _ltr_canvases(self, length: int) -> list[int]:
        lookahead = int(getattr(self.config, "grammar_canvas_lookahead", 0) or 0)
        if lookahead > 0:
            # P4: single growing canvas is replaced by prefix+K windows inside
            # the decode loop; still expose the target length as the final stage.
            return [length]
        if getattr(self.config, "grammar_block_decode", False):
            block = max(8, int(getattr(self.config, "grammar_block_size", 32) or 32))
            stages: list[int] = []
            end = min(block, length)
            while True:
                if end not in stages:
                    stages.append(end)
                if end >= length:
                    break
                end = min(end + block, length)
            return stages if stages else [length]
        raw_stages = self.config.grammar_ltr_stages or (64, 128, 192, 256)
        stages = [s for s in raw_stages if 1 < s <= length]
        if not stages or stages[-1] != length:
            stages = [*stages, length] if stages else [length]
        seen: set[int] = set()
        canvases: list[int] = []
        for s in stages:
            if s not in seen:
                seen.add(s)
                canvases.append(s)
        return canvases

    def _denoiser_forward(
        self,
        ids: torch.Tensor,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
    ) -> torch.Tensor:
        """Run denoiser and accumulate decode stats / optional trace forwards."""
        stats = get_active_stats()
        with timed_ms(stats, "denoiser_ms"):
            logits = self.denoiser(
                ids, ctx, pad_id=self.tokenizer.pad_id, ctx_pad_mask=ctx_pad
            )
        if stats is not None:
            stats.forwards_count += 1
            stats.full_projections += 1
            stats.canvas_tokens += int(ids.size(1))
        rec = getattr(self, "trace_recorder", None)
        if rec is not None:
            rec.forward()
        return logits

    def _denoiser_hidden(
        self,
        ids: torch.Tensor,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
    ) -> torch.Tensor:
        """Run only the transformer backbone for restricted candidate scoring."""
        stats = get_active_stats()
        with timed_ms(stats, "backbone_ms"):
            hidden = self.denoiser.encode(
                ids, ctx, pad_id=self.tokenizer.pad_id, ctx_pad_mask=ctx_pad
            )
        if stats is not None:
            stats.forwards_count += 1
            stats.canvas_tokens += int(ids.size(1))
        rec = getattr(self, "trace_recorder", None)
        if rec is not None:
            rec.forward()
        return hidden

    def _compiler_canvas(self, prefix: list[int], length: int) -> torch.Tensor:
        canvas = torch.full(
            (1, length),
            self.tokenizer.mask_id,
            dtype=torch.long,
            device=self.device_name,
        )
        used = min(length, len(prefix))
        if used:
            canvas[0, :used] = torch.as_tensor(
                prefix[:used], dtype=torch.long, device=self.device_name
            )
        return canvas

    def _project_candidates(
        self, hidden: torch.Tensor, candidate_ids: tuple[int, ...]
    ) -> torch.Tensor:
        stats = get_active_stats()
        index = torch.as_tensor(candidate_ids, dtype=torch.long, device=hidden.device)
        with timed_ms(stats, "projection_ms"):
            scores = self.denoiser.project(hidden, index)
        if stats is not None:
            stats.restricted_projections += 1
        bias = self._effective_structural_bias()
        if bias:
            from slm_training.models.grammar import structural_token_ids

            structural = structural_token_ids(self.tokenizer)
            boost = torch.as_tensor(
                [bias if tid in structural else 0.0 for tid in candidate_ids],
                dtype=scores.dtype,
                device=scores.device,
            )
            scores = scores + boost
        return scores

    def _action_key_for_token_id(self, token_id: int) -> str | None:
        """Map a token id to an action-catalog key.

        Component tokens render as ``+Card``; builtin tokens as ``*Run``;
        structural tokens keep their literal token text (e.g. ``r=``).
        """
        text = self.tokenizer.id_to_token.get(int(token_id))
        if text is None:
            return None
        return text

    def _token_id_for_action_key(self, action_key: str) -> int | None:
        """Inverse of ``_action_key_for_token_id`` when the mapping is unique."""
        return self.tokenizer.token_to_id.get(str(action_key))

    def _action_shortlist_policy(self) -> ActionShortlistPolicy:
        """Build an ActionShortlistPolicy from the current config."""
        cfg = self.config
        return ActionShortlistPolicy(
            mode=str(getattr(cfg, "action_shortlist_mode", "off") or "off"),
            k=int(getattr(cfg, "action_shortlist_k", 8) or 8),
            min_legal_size=int(
                getattr(cfg, "action_shortlist_min_legal_size", 16) or 16
            ),
            score_margin=float(
                getattr(cfg, "action_shortlist_score_margin", 0.0) or 0.0
            ),
            fallback_policy=str(
                getattr(
                    cfg, "action_shortlist_fallback_policy", "confidence_and_coverage"
                )
                or "confidence_and_coverage"
            ),
            shadow_full_score=bool(
                getattr(cfg, "action_shortlist_shadow_full_score", False)
            ),
        )

    def _ensure_action_shortlist_vectors(self) -> dict[str, torch.Tensor]:
        """Lazily build and cache deterministic fixture vectors for the catalog."""
        if self._action_shortlist_vectors is not None:
            return self._action_shortlist_vectors
        from slm_training.dsl.action_descriptions import ActionDescriptionCatalog

        catalog = ActionDescriptionCatalog.build()
        self._action_shortlist_catalog = catalog
        self._action_shortlist_vectors = catalog.fixture_vectors(
            self.config.d_model,
            source="schema_description",
            name_mode=str(
                getattr(self.config, "action_description_name_mode", "schema")
                or "schema"
            ),
        )
        return self._action_shortlist_vectors

    def _maybe_apply_action_shortlist(
        self,
        paths: tuple,
        prefix: list[int],
    ) -> tuple[tuple, ActionShortlistTrace | None]:
        """Filter ``paths`` by description retrieval when configured.

        When ``action_shortlist_mode`` is ``off`` or the legal set is too small,
        this returns ``paths`` unchanged and ``None``.  Otherwise it reduces the
        candidate ids passed to ``_project_candidates`` to the retrieval
        shortlist.  The reranker itself (``_project_candidates``) is unchanged;
        only the ids it sees are reduced.
        """
        policy = self._action_shortlist_policy()
        if policy.mode == "off":
            return paths, None

        if len(paths) < policy.min_legal_size:
            return paths, None

        legal_action_ids: list[str] = []
        path_by_action: dict[str, list] = {}
        for path in paths:
            if not path.token_ids:
                continue
            action_key = self._action_key_for_token_id(int(path.token_ids[0]))
            if action_key is None:
                continue
            legal_action_ids.append(action_key)
            path_by_action.setdefault(action_key, []).append(path)

        if len(legal_action_ids) < policy.min_legal_size:
            return paths, None

        action_vectors = self._ensure_action_shortlist_vectors()
        state_context = self.tokenizer.decode(prefix)
        query_vector = build_query_vector(
            state_context,
            self._action_shortlist_catalog,  # type: ignore[arg-type]
            FixtureDescriptionEncoder(self.config.d_model),
        )

        shortlist, scores, fallback_reason = retrieve_then_rerank(
            tuple(legal_action_ids),
            query_vector,
            action_vectors,
            policy,
        )

        shadow_full_selected_id: str | None = None
        if policy.shadow_full_score and not fallback_reason:
            # Diagnostic: which action would the full-set reranker pick next?
            # We do not have the model score here, so we record the top
            # retrieval-score id as a cheap shadow comparator.
            if scores:
                shadow_full_selected_id = max(scores, key=scores.get)  # type: ignore[arg-type]

        trace = ActionShortlistTrace(
            legal_action_ids=tuple(legal_action_ids),
            shortlist_ids=shortlist,
            retrieval_scores=scores,
            fallback_reason=fallback_reason,
            shadow_full_selected_id=shadow_full_selected_id,
        )

        if fallback_reason is not None:
            self._record_action_shortlist_trace(trace)
            return paths, trace

        filtered: list = []
        for action_key in shortlist:
            filtered.extend(path_by_action.get(action_key, []))

        if not filtered:
            self._record_action_shortlist_trace(trace)
            return paths, trace

        # Preserve original order where possible.
        kept_first_tokens = {int(p.token_ids[0]) for p in filtered}
        ordered = [p for p in paths if int(p.token_ids[0]) in kept_first_tokens]
        self._record_action_shortlist_trace(trace)
        return tuple(ordered), trace

    def _record_action_shortlist_trace(self, trace: ActionShortlistTrace) -> None:
        """Emit the shortlist trace to the best available sink."""
        payload = trace.to_dict()
        grec = getattr(self, "grammar_trace_recorder", None)
        if grec is not None and hasattr(grec, "record"):
            try:
                grec.record(
                    state_fingerprint=f"slm176:{len(trace.legal_action_ids)}:{hash(tuple(trace.shortlist_ids))}",
                    legal_action_ids=list(trace.legal_action_ids),
                    selected_action_id=trace.shortlist_ids[0]
                    if trace.shortlist_ids
                    else None,
                    logits_or_energies=[
                        trace.retrieval_scores.get(aid, 0.0)
                        for aid in trace.legal_action_ids
                    ],
                    convention="energy",
                    completion_support_size_exact=len(trace.shortlist_ids),
                )
            except Exception:  # noqa: BLE001
                pass
        stats = get_active_stats()
        if stats is not None:
            if not hasattr(stats, "action_shortlist_traces"):
                stats.action_shortlist_traces = []
            stats.action_shortlist_traces.append(payload)
        self.action_shortlist_traces.append(payload)

    def _component_inventory_token_ids(self) -> tuple[int, ...]:
        if self._component_token_ids_cache is not None:
            return self._component_token_ids_cache
        try:
            from slm_training.models.dsl_tokenizer import TokenKind

            ids = tuple(
                sorted(int(i) for i in self.tokenizer.kind_ids(TokenKind.COMPONENT))
            )
        except Exception:  # noqa: BLE001
            ids = ()
        self._component_token_ids_cache = ids
        return ids

    @staticmethod
    def _slot_component_owners(source: str) -> dict[str, str]:
        from slm_training.dsl.lang_core import parse

        owners: dict[str, str] = {}

        def walk(value: object, component: str | None = None) -> None:
            if isinstance(value, dict):
                owner = (
                    str(value["typeName"])
                    if value.get("type") == "element" and value.get("typeName")
                    else component
                )
                for child in value.values():
                    walk(child, owner)
            elif isinstance(value, list):
                for child in value:
                    walk(child, component)
            elif (
                isinstance(value, str)
                and value.startswith(":")
                and component is not None
            ):
                owners.setdefault(value, component)

        walk(parse(source).root)
        return owners

    def _component_name_index(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for index, token_id in enumerate(self._component_inventory_token_ids()):
            token = str(self.tokenizer.id_to_token.get(token_id, ""))
            for prefix in ("COMP:", "+"):
                if token.startswith(prefix):
                    token = token[len(prefix) :]
                    break
            if token:
                result[token] = index
        return result

    def _slot_component_logits(
        self,
        slots: list[str],
        context: torch.Tensor,
        pad_mask: torch.Tensor | None,
        context_rows: torch.Tensor,
        next_slots: list[str | None] | None = None,
    ) -> torch.Tensor:
        assert self.slot_component_head is not None
        slot_context, slot_pad = self._encode_context(slots)
        slot_pooled = self._pool_context(slot_context, slot_pad)
        if (
            bool(getattr(self.config, "slot_component_pair_interaction", False))
            and next_slots is not None
        ):
            next_context, next_pad = self._encode_context(
                [slot or "<no-next-slot>" for slot in next_slots]
            )
            next_pooled = self._pool_context(next_context, next_pad)
            present = torch.as_tensor(
                [slot is not None for slot in next_slots],
                dtype=slot_pooled.dtype,
                device=slot_pooled.device,
            ).unsqueeze(1)
            slot_pooled = slot_pooled + slot_pooled * next_pooled * present
        if bool(getattr(self.config, "slot_component_prompt_context", True)):
            slot_pooled = slot_pooled + self._pool_context(
                context, pad_mask
            ).index_select(0, context_rows)
        logits = self.slot_component_head(slot_pooled)
        weight = float(
            getattr(self.config, "slot_component_lexeme_prior_weight", 0.0) or 0.0
        )
        priors = getattr(self.config, "slot_component_lexeme_priors", ()) or ()
        if weight > 0.0 and priors:
            lookup = {
                str(token): torch.as_tensor(
                    scores, dtype=logits.dtype, device=logits.device
                )
                for token, scores in priors
            }
            bias = torch.zeros_like(logits)
            for row, slot in enumerate(slots):
                for token in set(tokenize_text(slot)):
                    scores = lookup.get(token)
                    if scores is not None:
                        bias[row] += scores
            logits = logits + weight * bias
        return logits

    def _slot_component_texts(self, slots: list[str]) -> list[str]:
        if not bool(getattr(self.config, "slot_component_next_context", False)):
            return list(slots)
        return [
            f"{slot}\n{slots[index + 1]}" if index + 1 < len(slots) else slot
            for index, slot in enumerate(slots)
        ]

    @staticmethod
    def _slot_role_token(slot: str) -> str:
        return next(
            (
                token
                for token in reversed(tokenize_text(slot))
                if any(char.isalnum() for char in token)
            ),
            "",
        )

    def _binder_component_token_ids(self) -> tuple[int, ...]:
        if self._binder_token_ids_cache is not None:
            return self._binder_token_ids_cache
        try:
            ids = tuple(sorted(int(i) for i in self.tokenizer.kind_ids("bind")))
        except Exception:  # noqa: BLE001
            ids = ()
        self._binder_token_ids_cache = ids
        return ids

    def _component_inventory_bias(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor | None,
        candidate_ids: tuple[int, ...],
    ) -> torch.Tensor | None:
        weight = float(
            getattr(self.config, "component_inventory_decode_weight", 0.0) or 0.0
        )
        if weight <= 0.0 or self.component_inventory_head is None:
            return None
        component_ids = set(self._component_inventory_token_ids())
        if not component_ids.intersection(candidate_ids):
            return None
        inventory = self.component_inventory_head(self._pool_context(ctx, ctx_pad))[0]
        bias = inventory.new_zeros(len(candidate_ids))
        component_positions = [
            (position, token_id)
            for position, token_id in enumerate(candidate_ids)
            if token_id in component_ids
        ]
        if component_positions:
            positions, token_ids = zip(*component_positions, strict=True)
            bias[list(positions)] = weight * inventory[list(token_ids)]
        return bias

    def _component_plan_bias(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor | None,
        prefix: list[int],
        candidate_ids: tuple[int, ...],
        candidate_kinds: tuple[str, ...],
    ) -> torch.Tensor | None:
        weight = float(getattr(self.config, "component_plan_decode_weight", 0.0) or 0.0)
        if weight <= 0.0 or self.component_plan_head is None:
            return None
        component_ids = set(self._component_inventory_token_ids())
        if not component_ids.intersection(candidate_ids):
            return None
        logits = self.component_plan_head(self._pool_context(ctx, ctx_pad))[0].view(
            2, self.tokenizer.vocab_size
        )
        emitted_bound: dict[int, int] = {}
        skipped_root = False
        for token_id in prefix:
            if token_id not in component_ids:
                continue
            if not skipped_root:
                skipped_root = True
                continue
            emitted_bound[token_id] = emitted_bound.get(token_id, 0) + 1
        bias = logits.new_zeros(len(candidate_ids))
        for position, (token_id, kind) in enumerate(
            zip(candidate_ids, candidate_kinds, strict=True)
        ):
            if token_id not in component_ids:
                continue
            if kind == "component_root":
                bias[position] = weight * logits[0, token_id]
            elif kind == "component_bound":
                remaining = (
                    F.softplus(logits[1, token_id]) - emitted_bound.get(token_id, 0)
                ).clamp_min(1e-4)
                bias[position] = weight * remaining.log()
            elif kind == "component_root_or_bound":
                remaining = (
                    F.softplus(logits[1, token_id]) - emitted_bound.get(token_id, 0)
                ).clamp_min(1e-4)
                bias[position] = weight * torch.logaddexp(
                    logits[0, token_id], remaining.log()
                )
        return bias

    def _slot_component_bias(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor | None,
        prefix: list[int],
        candidate_ids: tuple[int, ...],
        candidate_kinds: tuple[str, ...],
        slot_contract: list[str] | None,
        semantic_role_candidates: dict[str, tuple[str, ...]] | None = None,
    ) -> torch.Tensor | None:
        learned_weight = float(
            getattr(self.config, "slot_component_decode_weight", 0.0) or 0.0
        )
        role_weight = float(
            getattr(self.config, "semantic_role_decode_weight", 0.0) or 0.0
        )
        learned_enabled = learned_weight > 0.0 and self.slot_component_head is not None
        if (
            (not learned_enabled and role_weight <= 0.0)
            or not slot_contract
            or "component_bound" not in candidate_kinds
        ):
            return None
        remaining_slots: list[str] = []
        for index, slot in enumerate(slot_contract):
            try:
                slot_id = int(self.tokenizer.sym_id(index))
            except (AttributeError, KeyError, ValueError):
                return None
            if slot_id not in prefix:
                remaining_slots.append(str(slot))
        if not remaining_slots:
            return None
        logits = None
        if learned_enabled:
            slot_texts = self._slot_component_texts(remaining_slots)
            slot_rows = torch.zeros(
                len(remaining_slots), dtype=torch.long, device=ctx.device
            )
            logits = self._slot_component_logits(
                slot_texts,
                ctx,
                ctx_pad,
                slot_rows,
                next_slots=(
                    [
                        remaining_slots[index + 1]
                        if index + 1 < len(remaining_slots)
                        else None
                        for index in range(len(remaining_slots))
                    ]
                    if bool(
                        getattr(self.config, "slot_component_pair_interaction", False)
                    )
                    else None
                ),
            )
        component_index = {
            token_id: index
            for index, token_id in enumerate(self._component_inventory_token_ids())
        }
        span_weight = float(
            getattr(self.config, "slot_component_span_prior_weight", 0.0) or 0.0
        )
        use_content_arity = (
            bool(getattr(self.config, "slot_component_content_arity", False))
            or span_weight > 0.0
        )
        span_lookup = dict(getattr(self.config, "slot_component_span_priors", ()) or ())
        bias = ctx.new_zeros(len(candidate_ids))
        applied = False
        for position, (token_id, kind) in enumerate(
            zip(candidate_ids, candidate_kinds, strict=True)
        ):
            index = component_index.get(token_id)
            if index is None or kind != "component_bound":
                continue
            slot_content_count = getattr(self.tokenizer, "slot_content_count", None)
            required = (
                int(slot_content_count(token_id))
                if use_content_arity and callable(slot_content_count)
                else 1
            )
            if required <= 0:
                continue
            consumed = min(required, len(remaining_slots))
            matches = 0
            visible_role_available = False
            if role_weight > 0.0 and semantic_role_candidates:
                token = str(self.tokenizer.id_to_token.get(token_id, ""))
                for prefix in ("COMP:", "+"):
                    if token.startswith(prefix):
                        token = token[len(prefix) :]
                        break
                visible_role_available = bool(slot_contract) and all(
                    semantic_role_candidates.get(slot, ()) for slot in slot_contract
                )
                matches = sum(
                    token in semantic_role_candidates.get(slot, ())
                    for slot in remaining_slots[:consumed]
                )
            if logits is not None and (not visible_role_available or matches > 0):
                bias[position] = learned_weight * logits[:consumed, index].mean()
            if role_weight > 0.0 and semantic_role_candidates:
                bias[position] += role_weight * matches / consumed
            if span_weight > 0.0 and consumed == required and required > 1:
                key = "\x1f".join(
                    self._slot_role_token(slot) for slot in remaining_slots[:consumed]
                )
                scores = span_lookup.get(key)
                if scores is not None:
                    bias[position] += span_weight * float(scores[index])
            applied = logits is not None or bool(
                role_weight > 0.0 and semantic_role_candidates
            )
        return bias if applied else None

    def _semantic_plan_covered_counts(
        self,
        state: Any,
        prefix: list[int] | None,
        family_token_ids: dict[str, int],
    ) -> Counter[int]:
        """Count planned families at every emitted nesting depth."""
        if prefix is not None:
            return Counter(
                family_token_ids[family]
                for token_id in prefix
                if (token := str(self.tokenizer.id_to_token.get(token_id, ""))).startswith(
                    ("+", "COMP:")
                )
                and (family := token.removeprefix("COMP:").removeprefix("+"))
                in family_token_ids
            )
        return Counter(
            family_token_ids.get(str(expr_type).removeprefix("element:"))
            for expr_type in getattr(state, "section_types", ())
            if str(expr_type).startswith("element:")
        )

    def _semantic_plan_bias(
        self,
        row: int,
        candidate_ids: tuple[int, ...],
        candidate_kinds: tuple[str, ...],
        state: Any | None = None,
        prefix: list[int] | None = None,
        candidate_scores: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Soft-score missing component instances and distinct-slot closure."""
        weight = float(getattr(self.config, "semantic_plan_decode_weight", 0.0) or 0.0)
        seed_weight = float(
            getattr(self.config, "semantic_plan_seed_decode_weight", 0.0) or 0.0
        )
        margin = float(
            getattr(
                self.config,
                "semantic_plan_margin_decode_weight",
                0.0,
            )
            or 0.0
        )
        if self._semantic_plan_seed_active(state, candidate_kinds):
            weight += seed_weight
        if (
            (weight <= 0.0 and margin <= 0.0)
            or not self._semantic_plan_action_scores
            or row >= len(self._semantic_plan_action_scores)
        ):
            return None
        scores = self._semantic_plan_action_scores[row]
        if not scores:
            return None
        remaining_counts = (
            dict(self._semantic_plan_action_counts[row])
            if state is not None
            and self._semantic_plan_action_counts
            and row < len(self._semantic_plan_action_counts)
            else None
        )
        family_token_ids = {
            str(self.tokenizer.id_to_token[token_id])
            .removeprefix("COMP:")
            .removeprefix("+"): token_id
            for token_id in self._component_inventory_token_ids()
        }
        if remaining_counts is not None:
            for token_id, count in self._semantic_plan_covered_counts(
                state, prefix, family_token_ids
            ).items():
                if token_id in remaining_counts:
                    remaining_counts[token_id] = max(
                        0, remaining_counts[token_id] - count
                    )
        bias = torch.zeros(len(candidate_ids), device=next(self.parameters()).device)
        applied = False
        component_kinds = {
            "component_root",
            "component_bound",
            "component_root_or_bound",
        }
        component_score_ceiling = (
            max(
                float(candidate_scores[position].item())
                for position, kind in enumerate(candidate_kinds)
                if kind in component_kinds
            )
            if margin > 0.0
            and candidate_scores is not None
            and any(kind in component_kinds for kind in candidate_kinds)
            else None
        )
        for position, (token_id, kind) in enumerate(
            zip(candidate_ids, candidate_kinds, strict=True)
        ):
            score = scores.get(token_id, 0.0)
            still_required = (
                remaining_counts is None or remaining_counts.get(token_id, 0) > 0
            )
            if kind in component_kinds and score > 0.0 and still_required:
                bias[position] = weight * score
                if component_score_ceiling is not None:
                    margin_bias = (
                        component_score_ceiling
                        + margin
                        - float(candidate_scores[position].item())
                    )
                    bias[position] = max(
                        float(bias[position].item()),
                        margin_bias,
                    )
                applied = True
        frames = getattr(state, "frames", ())
        if remaining_counts is not None and prefix and frames:
            frame = frames[-1]
            family = str(getattr(frame, "expr_type", "")).removeprefix("element:")
            family_token_id = family_token_ids.get(family)
            family_score = scores.get(family_token_id, 0.0)
            close_id = self.tokenizer.token_to_id.get(str(getattr(frame, "close", "")))
            if (
                getattr(frame, "kind", None) == "component"
                and family_token_id is not None
                and family_score > 0.0
                and remaining_counts.get(family_token_id, 0)
                > (0 if prefix is not None else 1)
                and close_id in candidate_ids
            ):
                open_position = max(
                    (
                        position
                        for position, token_id in enumerate(prefix)
                        if token_id == family_token_id
                    ),
                    default=-1,
                )
                has_visible_slot = any(
                    str(self.tokenizer.id_to_token.get(token_id, "")).startswith("@")
                    for token_id in prefix[open_position + 1 :]
                )
                if has_visible_slot:
                    bias[candidate_ids.index(close_id)] = weight * family_score
                    applied = True
        return bias if applied else None

    @staticmethod
    def _semantic_plan_seed_active(
        state: Any | None,
        candidate_kinds: tuple[str, ...],
    ) -> bool:
        return bool(
            state is not None
            and not tuple(getattr(state, "section_types", ()))
            and any(
                kind in {"component_root", "component_root_or_bound"}
                for kind in candidate_kinds
            )
        )

    def _record_semantic_plan_seed_trace(
        self,
        stats: DecodeStats,
        *,
        row: int,
        position: int,
        state: Any,
        candidate_ids: tuple[int, ...],
        candidate_kinds: tuple[str, ...],
        scores_before: torch.Tensor,
        plan_bias: torch.Tensor,
        scores_after: torch.Tensor,
    ) -> dict[str, object] | None:
        """Record a bounded first-family score decomposition."""
        seed_weight = float(
            getattr(self.config, "semantic_plan_seed_decode_weight", 0.0) or 0.0
        )
        if (
            seed_weight <= 0.0
            or len(stats.constrained_selection_traces) >= 64
            or not self._semantic_plan_seed_active(state, candidate_kinds)
        ):
            return None
        before = int(scores_before.argmax().item())
        after = int(scores_after.argmax().item())
        ranked = torch.topk(
            scores_after,
            k=min(8, int(scores_after.numel())),
        ).indices.tolist()
        trace: dict[str, object] = {
            "phase": "semantic_plan_seed",
            "row": int(row),
            "position": int(position),
            "before_token": str(
                self.tokenizer.id_to_token.get(candidate_ids[before], "")
            ),
            "chosen_token": str(
                self.tokenizer.id_to_token.get(candidate_ids[after], "")
            ),
            "choice_changed": before != after,
            "seed_weight": seed_weight,
            "semantic_plan_decode_weight": float(
                getattr(self.config, "semantic_plan_decode_weight", 0.0) or 0.0
            ),
            "top_candidates": [
                {
                    "token": str(
                        self.tokenizer.id_to_token.get(candidate_ids[index], "")
                    ),
                    "kind": candidate_kinds[index],
                    "score_before": round(float(scores_before[index].item()), 6),
                    "plan_bias": round(float(plan_bias[index].item()), 6),
                    "score_after": round(float(scores_after[index].item()), 6),
                }
                for index in ranked
            ],
        }
        stats.constrained_selection_traces.append(trace)
        return trace

    def _record_semantic_plan_missing_family_trace(
        self,
        stats: DecodeStats,
        *,
        row: int,
        position: int,
        state: Any,
        candidate_ids: tuple[int, ...],
        candidate_kinds: tuple[str, ...],
        scores_before: torch.Tensor,
        plan_bias: torch.Tensor,
        scores_after: torch.Tensor,
    ) -> dict[str, object] | None:
        """Record bounded score evidence for a planned family after the first."""
        section_types = tuple(getattr(state, "section_types", ()))
        component_kinds = {
            "component_root",
            "component_bound",
            "component_root_or_bound",
        }
        if (
            not section_types
            or len(stats.constrained_selection_traces) >= 64
            or not any(
                kind in component_kinds and float(plan_bias[index].item()) > 0.0
                for index, kind in enumerate(candidate_kinds)
            )
        ):
            return None
        remaining = (
            dict(self._semantic_plan_action_counts[row])
            if self._semantic_plan_action_counts
            and row < len(self._semantic_plan_action_counts)
            else {}
        )
        family_token_ids = {
            str(self.tokenizer.id_to_token[token_id])
            .removeprefix("COMP:")
            .removeprefix("+"): token_id
            for token_id in self._component_inventory_token_ids()
        }
        for expr_type in section_types:
            token_id = family_token_ids.get(str(expr_type).removeprefix("element:"))
            if token_id in remaining:
                remaining[token_id] = max(0, remaining[token_id] - 1)
        before = int(scores_before.argmax().item())
        after = int(scores_after.argmax().item())
        ranked = torch.topk(
            scores_after,
            k=min(8, int(scores_after.numel())),
        ).indices.tolist()
        trace: dict[str, object] = {
            "phase": "semantic_plan_missing_family",
            "row": int(row),
            "position": int(position),
            "emitted_families": [
                str(expr_type).removeprefix("element:") for expr_type in section_types
            ],
            "remaining_planned_families": {
                str(self.tokenizer.id_to_token.get(token_id, ""))
                .removeprefix("COMP:")
                .removeprefix("+"): int(count)
                for token_id, count in remaining.items()
                if count > 0
            },
            "before_token": str(
                self.tokenizer.id_to_token.get(candidate_ids[before], "")
            ),
            "chosen_token": str(
                self.tokenizer.id_to_token.get(candidate_ids[after], "")
            ),
            "choice_changed": before != after,
            "semantic_plan_decode_weight": float(
                getattr(self.config, "semantic_plan_decode_weight", 0.0) or 0.0
            ),
            "semantic_plan_margin_decode_weight": float(
                getattr(
                    self.config,
                    "semantic_plan_margin_decode_weight",
                    0.0,
                )
                or 0.0
            ),
            "planned_candidates": [
                {
                    "token": str(
                        self.tokenizer.id_to_token.get(candidate_ids[index], "")
                    ),
                    "kind": candidate_kinds[index],
                    "score_before": round(float(scores_before[index].item()), 6),
                    "plan_bias": round(float(plan_bias[index].item()), 6),
                    "score_after": round(float(scores_after[index].item()), 6),
                }
                for index in range(len(candidate_ids))
                if candidate_kinds[index] in component_kinds
                and float(plan_bias[index].item()) > 0.0
            ],
            "top_candidates": [
                {
                    "token": str(
                        self.tokenizer.id_to_token.get(candidate_ids[index], "")
                    ),
                    "kind": candidate_kinds[index],
                    "score_before": round(float(scores_before[index].item()), 6),
                    "plan_bias": round(float(plan_bias[index].item()), 6),
                    "score_after": round(float(scores_after[index].item()), 6),
                }
                for index in ranked
            ],
        }
        stats.constrained_selection_traces.append(trace)
        return trace

    def _finalize_semantic_plan_trace(
        self,
        trace: dict[str, object] | None,
        *,
        candidate_ids: tuple[int, ...],
        scores: torch.Tensor,
    ) -> None:
        """Attach the actual final choice and aggregate post-plan score delta."""
        if trace is None:
            return
        final = int(scores.argmax().item())
        final_token = str(self.tokenizer.id_to_token.get(candidate_ids[final], ""))
        trace["final_token"] = final_token
        trace["changed_after_plan"] = final_token != trace["chosen_token"]
        candidates = trace.get("top_candidates")
        score_by_token = {
            str(self.tokenizer.id_to_token.get(token_id, "")): round(
                float(scores[index].item()), 6
            )
            for index, token_id in enumerate(candidate_ids)
        }
        for candidate_group in (
            trace.get("planned_candidates"),
            candidates,
        ):
            if not isinstance(candidate_group, list):
                continue
            for candidate in candidate_group:
                if not isinstance(candidate, dict):
                    continue
                token = str(candidate.get("token", ""))
                final_score = score_by_token.get(token)
                if final_score is None:
                    continue
                candidate["post_plan_bias"] = round(
                    final_score - float(candidate["score_after"]), 6
                )
                candidate["final_score"] = final_score

    def _record_semantic_plan_root_trace(
        self,
        stats: DecodeStats,
        *,
        row: int,
        position: int,
        state: Any,
        candidate_ids: tuple[int, ...],
        scores_before: torch.Tensor,
        root_bias: torch.Tensor,
        scores_after: torch.Tensor,
    ) -> dict[str, object] | None:
        """Record bounded score evidence for a verified plan-root token."""
        if len(stats.constrained_selection_traces) >= 64:
            return None
        targeted = [
            index
            for index in range(len(candidate_ids))
            if float(root_bias[index].item()) > 0.0
        ]
        if not targeted:
            return None
        before = int(scores_before.argmax().item())
        after = int(scores_after.argmax().item())
        ranked = torch.topk(
            scores_after,
            k=min(8, int(scores_after.numel())),
        ).indices.tolist()
        trace: dict[str, object] = {
            "phase": "semantic_plan_root",
            "row": int(row),
            "position": int(position),
            "emitted_families": [
                str(expr_type).removeprefix("element:")
                for expr_type in getattr(state, "section_types", ())
            ],
            "before_token": str(
                self.tokenizer.id_to_token.get(candidate_ids[before], "")
            ),
            "chosen_token": str(
                self.tokenizer.id_to_token.get(candidate_ids[after], "")
            ),
            "choice_changed": before != after,
            "semantic_plan_root_decode_weight": float(
                getattr(self.config, "semantic_plan_root_decode_weight", 0.0) or 0.0
            ),
            "semantic_plan_root_margin_decode_weight": float(
                getattr(
                    self.config,
                    "semantic_plan_root_margin_decode_weight",
                    0.0,
                )
                or 0.0
            ),
            "planned_candidates": [
                {
                    "token": str(
                        self.tokenizer.id_to_token.get(candidate_ids[index], "")
                    ),
                    "score_before": round(float(scores_before[index].item()), 6),
                    "plan_bias": round(float(root_bias[index].item()), 6),
                    "score_after": round(float(scores_after[index].item()), 6),
                }
                for index in targeted
            ],
            "top_candidates": [
                {
                    "token": str(
                        self.tokenizer.id_to_token.get(candidate_ids[index], "")
                    ),
                    "score_before": round(float(scores_before[index].item()), 6),
                    "plan_bias": round(float(root_bias[index].item()), 6),
                    "score_after": round(float(scores_after[index].item()), 6),
                }
                for index in ranked
            ],
            **self._choice_phase_evidence(state),
        }
        stats.constrained_selection_traces.append(trace)
        return trace

    def _schema_value_bias(
        self,
        state: Any,
        candidate_ids: tuple[int, ...],
        scores: torch.Tensor,
    ) -> torch.Tensor | None:
        """Discourage visible placeholders in enum-valued component arguments."""
        weight = float(getattr(self.config, "schema_value_decode_weight", 0.0) or 0.0)
        frames = list(getattr(state, "frames", ()))
        if weight <= 0.0 or not frames:
            return None
        frame = frames[-1]
        schemas = tuple(getattr(frame, "schemas", ()))
        index = int(getattr(frame, "arg_index", -1))
        if (
            getattr(frame, "kind", None) != "component"
            or index < 0
            or index >= len(schemas)
            or not self._schema_contains_enum(schemas[index])
        ):
            return None
        from slm_training.dsl.production_codec import SLOT_PREFIX

        bias = scores.new_zeros(len(candidate_ids))
        applied = False
        for position, token_id in enumerate(candidate_ids):
            token = str(self.tokenizer.id_to_token.get(token_id, ""))
            if token.startswith(SLOT_PREFIX):
                bias[position] = -weight
                applied = True
        return bias if applied else None

    def _semantic_plan_inline_bias(
        self,
        row: int,
        prefix: list[int],
        candidate_ids: tuple[int, ...],
        candidate_kinds: tuple[str, ...],
    ) -> torch.Tensor | None:
        """Soft-score still-missing prompt families in inline component positions."""
        weight = float(
            getattr(self.config, "semantic_plan_inline_decode_weight", 0.0) or 0.0
        )
        if (
            weight <= 0.0
            or not self._semantic_plan_action_scores
            or not self._semantic_plan_action_counts
            or row >= len(self._semantic_plan_action_scores)
            or row >= len(self._semantic_plan_action_counts)
        ):
            return None
        scores = self._semantic_plan_action_scores[row]
        remaining = dict(self._semantic_plan_action_counts[row])
        for token_id in prefix:
            if token_id in remaining:
                remaining[token_id] = max(0, remaining[token_id] - 1)
        bias = torch.zeros(len(candidate_ids), device=next(self.parameters()).device)
        applied = False
        for position, (token_id, kind) in enumerate(
            zip(candidate_ids, candidate_kinds, strict=True)
        ):
            score = scores.get(token_id, 0.0)
            if kind == "component" and score > 0.0 and remaining.get(token_id, 0) > 0:
                bias[position] = weight * score
                applied = True
        return bias if applied else None

    def _schema_opaque_bias(
        self,
        state: Any,
        candidate_ids: tuple[int, ...],
        scores: torch.Tensor,
    ) -> torch.Tensor | None:
        """Discourage visible placeholders in optional unconstrained arguments."""
        weight = float(getattr(self.config, "schema_opaque_decode_weight", 0.0) or 0.0)
        frames = list(getattr(state, "frames", ()))
        if weight <= 0.0 or not frames:
            return None
        frame = frames[-1]
        schemas = tuple(getattr(frame, "schemas", ()))
        index = int(getattr(frame, "arg_index", -1))
        if (
            getattr(frame, "kind", None) != "component"
            or index < int(getattr(frame, "required_args", 0))
            or index < 0
            or index >= len(schemas)
            or schemas[index]
        ):
            return None
        from slm_training.dsl.production_codec import SLOT_PREFIX

        bias = scores.new_zeros(len(candidate_ids))
        applied = False
        for position, token_id in enumerate(candidate_ids):
            token = str(self.tokenizer.id_to_token.get(token_id, ""))
            if token.startswith(SLOT_PREFIX):
                bias[position] = -weight
                applied = True
        return bias if applied else None

    def _schema_precontent_literal_bias(
        self,
        state: Any,
        candidate_ids: tuple[int, ...],
        scores: torch.Tensor,
    ) -> torch.Tensor | None:
        """Route a pre-content operational string to the legal empty literal."""
        weight = float(getattr(self.config, "schema_opaque_decode_weight", 0.0) or 0.0)
        frames = list(getattr(state, "frames", ()))
        if weight <= 0.0 or not frames:
            return None
        frame = frames[-1]
        schemas = tuple(getattr(frame, "schemas", ()))
        index = int(getattr(frame, "arg_index", -1))
        if getattr(frame, "kind", None) != "component" or not (
            0 <= index < len(schemas)
        ):
            return None
        schema = schemas[index]
        followed_by_content = (
            schema.get("type") == "string"
            and not schema.get("x-openui-placeholder")
            and not self._schema_contains_enum(schema)
            and index + 1 < len(schemas)
            and bool(schemas[index + 1].get("x-openui-placeholder"))
        )
        if not followed_by_content:
            return None
        from slm_training.dsl.production_codec import LIT_PREFIX, SLOT_PREFIX

        slot_positions = [
            position
            for position, token_id in enumerate(candidate_ids)
            if str(self.tokenizer.id_to_token.get(token_id, "")).startswith(SLOT_PREFIX)
        ]
        literal_id = self.tokenizer.token_to_id.get(f"{LIT_PREFIX}\"\"")
        if literal_id not in candidate_ids or not slot_positions:
            return None
        literal_position = candidate_ids.index(literal_id)
        slot_max = max(float(scores[position].item()) for position in slot_positions)
        bias = scores.new_zeros(len(candidate_ids))
        bias[literal_position] = max(
            0.0,
            slot_max + weight - float(scores[literal_position].item()),
        )
        return bias

    def _schema_enum_close_bias(
        self,
        state: Any,
        candidate_ids: tuple[int, ...],
        scores: torch.Tensor,
    ) -> torch.Tensor | None:
        """Prefer closure at optional enum-valued component arguments."""
        weight = float(
            getattr(self.config, "schema_enum_close_decode_weight", 0.0) or 0.0
        )
        frames = list(getattr(state, "frames", ()))
        if weight <= 0.0 or not frames:
            return None
        frame = frames[-1]
        schemas = tuple(getattr(frame, "schemas", ()))
        index = int(getattr(frame, "arg_index", -1))
        if (
            getattr(frame, "kind", None) != "component"
            or index < int(getattr(frame, "required_args", 0))
            or index < 0
            or index >= len(schemas)
            or not self._schema_contains_enum(schemas[index])
        ):
            return None
        close_id = self.tokenizer.token_to_id.get(str(getattr(frame, "close", "")))
        if close_id not in candidate_ids:
            return None
        bias = scores.new_zeros(len(candidate_ids))
        bias[candidate_ids.index(close_id)] = weight
        return bias

    def _schema_opaque_close_bias(
        self,
        state: Any,
        candidate_ids: tuple[int, ...],
        scores: torch.Tensor,
    ) -> torch.Tensor | None:
        """Prefer closure at optional unconstrained component arguments."""
        weight = float(
            getattr(self.config, "schema_opaque_close_decode_weight", 0.0) or 0.0
        )
        frames = list(getattr(state, "frames", ()))
        if weight <= 0.0 or not frames:
            return None
        frame = frames[-1]
        schemas = tuple(getattr(frame, "schemas", ()))
        index = int(getattr(frame, "arg_index", -1))
        if (
            getattr(frame, "kind", None) != "component"
            or index < int(getattr(frame, "required_args", 0))
            or index < 0
            or index >= len(schemas)
            or schemas[index]
        ):
            return None
        close_id = self.tokenizer.token_to_id.get(str(getattr(frame, "close", "")))
        if close_id not in candidate_ids:
            return None
        bias = scores.new_zeros(len(candidate_ids))
        bias[candidate_ids.index(close_id)] = weight
        return bias

    def _schema_role_slot_bias(
        self,
        state: Any,
        candidate_ids: tuple[int, ...],
        scores: torch.Tensor,
        slot_contract: list[str] | None,
        semantic_role_candidates: dict[str, tuple[str, ...]] | None,
    ) -> torch.Tensor | None:
        """Prefer visible slots compatible with the active content property owner."""
        weight = float(
            getattr(self.config, "schema_role_slot_decode_weight", 0.0) or 0.0
        )
        frames = list(getattr(state, "frames", ()))
        if (
            weight <= 0.0
            or not frames
            or not slot_contract
            or not semantic_role_candidates
        ):
            return None
        frame = frames[-1]
        schemas = tuple(getattr(frame, "schemas", ()))
        index = int(getattr(frame, "arg_index", -1))
        active_property: str | None = None
        if getattr(frame, "kind", None) == "component":
            component = str(getattr(frame, "expr_type", "")).removeprefix(
                "element:"
            )
            accepts_slot = (
                0 <= index < len(schemas)
                and bool(schemas[index].get("x-openui-placeholder"))
            )
        elif getattr(frame, "kind", None) == "object":
            active_property = getattr(frame, "active_property", None)
            component = next(
                (
                    str(getattr(owner, "expr_type", "")).removeprefix(
                        "element:"
                    )
                    for owner in reversed(frames[:-1])
                    if getattr(owner, "kind", None) == "component"
                ),
                "",
            )
            accepts_slot = (
                active_property is not None
                and 0 <= index < len(schemas)
                and self._schema_can_reach_visible_slot(dict(schemas[index]))
            )
        else:
            return None
        if not component or not accepts_slot:
            return None
        from slm_training.data.quality import semantic_role_properties
        from slm_training.dsl.production_codec import SLOT_PREFIX

        properties_by_slot = semantic_role_properties(slot_contract)
        bias = scores.new_zeros(len(candidate_ids))
        applied = False
        for position, token_id in enumerate(candidate_ids):
            token = str(self.tokenizer.id_to_token.get(token_id, ""))
            if not token.startswith(SLOT_PREFIX):
                continue
            try:
                slot_index = int(token[len(SLOT_PREFIX) :])
                slot = slot_contract[slot_index]
            except (ValueError, IndexError):
                continue
            if (
                component in semantic_role_candidates.get(slot, ())
                and (
                    active_property is None
                    or active_property in properties_by_slot.get(slot, ())
                )
            ):
                bias[position] = weight
                applied = True
        return bias if applied else None

    def _slot_coverage_close_bias(
        self,
        state: Any,
        prefix: list[int],
        candidate_ids: tuple[int, ...],
        scores: torch.Tensor,
        slot_contract: list[str] | None,
        semantic_role_candidates: dict[str, tuple[str, ...]] | None = None,
    ) -> torch.Tensor | None:
        """Prefer coverage-compatible continuations before legal frame closure."""
        weight = float(
            getattr(self.config, "slot_coverage_close_decode_weight", 0.0) or 0.0
        )
        frames = list(getattr(state, "frames", ()))
        if weight <= 0.0 or not frames or not slot_contract:
            return None
        frame = frames[-1]
        kind = getattr(frame, "kind", None)
        if kind not in {"component", "variadic", "object"}:
            return None
        if kind == "object" and getattr(frame, "phase", None) != "key":
            return None
        close_id = self.tokenizer.token_to_id.get(str(getattr(frame, "close", "")))
        if close_id not in candidate_ids:
            return None
        try:
            missing = tuple(
                (index, slot)
                for index, slot in enumerate(slot_contract)
                if int(self.tokenizer.sym_id(index)) not in prefix
            )
        except (AttributeError, KeyError, ValueError):
            return None
        bias = scores.new_zeros(len(candidate_ids))
        if not missing:
            bias[candidate_ids.index(close_id)] = weight
            return bias

        from slm_training.data.quality import (
            semantic_role_properties,
            semantic_role_reachable_candidates,
        )
        from slm_training.dsl.production_codec import (
            LIST_OPEN,
            NAME_PREFIX,
            OBJ_OPEN,
            OPEN_PREFIX,
        )

        missing_slots_by_id = {
            int(self.tokenizer.sym_id(index)): slot for index, slot in missing
        }
        properties_by_slot = semantic_role_properties(
            [slot for _index, slot in missing]
        )
        candidate_components = {
            token[len(OPEN_PREFIX) :]
            for token_id in candidate_ids
            if (token := str(self.tokenizer.id_to_token.get(token_id, ""))).startswith(
                OPEN_PREFIX
            )
        }
        owner_component = next(
            (
                str(getattr(owner, "expr_type", "")).removeprefix("element:")
                for owner in reversed(frames)
                if getattr(owner, "kind", None) == "component"
            ),
            "",
        )
        reachable_candidates = semantic_role_reachable_candidates(
            [slot for _index, slot in missing],
            sorted(candidate_components | ({owner_component} if owner_component else set())),
        )

        def owner_matches(slot: str) -> bool:
            return bool(
                owner_component
                and (
                    (
                        semantic_role_candidates
                        and owner_component in semantic_role_candidates.get(slot, ())
                    )
                    or owner_component in reachable_candidates.get(slot, ())
                )
            )

        if kind == "component" and semantic_role_candidates and not any(
            owner_matches(slot) for _index, slot in missing
        ):
            close_position = candidate_ids.index(close_id)
            bias[close_position] = max(
                0.0,
                float(scores.max().item())
                + weight
                - float(scores[close_position].item()),
            )
            return bias

        active_schema: dict[str, Any] | None = None
        if kind == "component":
            schemas = tuple(getattr(frame, "schemas", ()))
            index = int(getattr(frame, "arg_index", -1))
            if 0 <= index < len(schemas):
                active_schema = dict(schemas[index])
        elif kind == "variadic":
            schemas = tuple(getattr(frame, "schemas", ()))
            if schemas:
                active_schema = dict(schemas[0])

        targets: list[int] = []
        direct_slot_compatible = bool(
            kind != "component"
            or (
                active_schema is not None
                and active_schema.get("x-openui-placeholder")
            )
        )
        for position, token_id in enumerate(candidate_ids):
            if token_id == close_id:
                continue
            token = str(self.tokenizer.id_to_token.get(token_id, ""))
            if token_id in missing_slots_by_id:
                slot = missing_slots_by_id[token_id]
                if direct_slot_compatible and (
                    not semantic_role_candidates or owner_matches(slot)
                ):
                    targets.append(position)
                continue
            if token.startswith(OPEN_PREFIX):
                component = token[len(OPEN_PREFIX) :]
                if any(
                    (
                        semantic_role_candidates
                        and component in semantic_role_candidates.get(slot, ())
                    )
                    or component in reachable_candidates.get(slot, ())
                    for _index, slot in missing
                ):
                    targets.append(position)
                continue
            if kind == "object" and token.startswith(NAME_PREFIX):
                property_name = token[len(NAME_PREFIX) :]
                property_names = tuple(getattr(frame, "property_names", ()))
                schemas = tuple(getattr(frame, "schemas", ()))
                if property_name not in property_names:
                    continue
                property_index = property_names.index(property_name)
                if (
                    property_index < len(schemas)
                    and self._schema_can_reach_visible_slot(
                        dict(schemas[property_index])
                    )
                    and any(
                        property_name in properties_by_slot.get(slot, ())
                        and owner_matches(slot)
                        for _index, slot in missing
                    )
                ):
                    targets.append(position)
                continue
            if (
                token in {LIST_OPEN, OBJ_OPEN}
                and active_schema is not None
                and self._schema_can_reach_visible_slot(active_schema)
                and any(owner_matches(slot) for _index, slot in missing)
            ):
                targets.append(position)
        if not targets:
            return None
        target = max(targets, key=lambda position: float(scores[position].item()))
        bias[target] = max(
            0.0,
            float(scores.max().item()) + weight - float(scores[target].item()),
        )
        return bias

    def _record_slot_coverage_close_trace(
        self,
        stats: DecodeStats,
        *,
        row: int,
        position: int,
        state: Any,
        prefix: list[int],
        candidate_ids: tuple[int, ...],
        scores_before: torch.Tensor,
        coverage_bias: torch.Tensor,
        scores_after: torch.Tensor,
        slot_contract: list[str] | None,
    ) -> dict[str, object] | None:
        """Record bounded evidence for a visible-slot closure intervention."""
        if len(stats.constrained_selection_traces) >= 64:
            return None
        targeted = [
            index
            for index in range(len(candidate_ids))
            if float(coverage_bias[index].item()) > 0.0
        ]
        if not targeted:
            return None
        frames = list(getattr(state, "frames", ()))
        frame = frames[-1] if frames else None
        close_id = self.tokenizer.token_to_id.get(
            str(getattr(frame, "close", ""))
        )
        try:
            missing_slots = [
                slot
                for index, slot in enumerate(slot_contract or ())
                if int(self.tokenizer.sym_id(index)) not in prefix
            ]
        except (AttributeError, KeyError, ValueError):
            missing_slots = []
        owner_component = next(
            (
                str(getattr(owner, "expr_type", "")).removeprefix("element:")
                for owner in reversed(frames)
                if getattr(owner, "kind", None) == "component"
            ),
            "",
        )
        before = int(scores_before.argmax().item())
        after = int(scores_after.argmax().item())
        ranked = torch.topk(
            scores_after,
            k=min(8, int(scores_after.numel())),
        ).indices.tolist()
        trace: dict[str, object] = {
            "phase": "slot_coverage_close",
            "row": int(row),
            "position": int(position),
            "mode": (
                ("covered_close" if not missing_slots else "owner_escape")
                if close_id is not None
                and close_id in {candidate_ids[index] for index in targeted}
                else "coverage_continue"
            ),
            "missing_slots": missing_slots,
            "owner_component": owner_component,
            "active_property": str(getattr(frame, "active_property", "") or ""),
            "before_token": str(
                self.tokenizer.id_to_token.get(candidate_ids[before], "")
            ),
            "chosen_token": str(
                self.tokenizer.id_to_token.get(candidate_ids[after], "")
            ),
            "choice_changed": before != after,
            "slot_coverage_close_decode_weight": float(
                getattr(self.config, "slot_coverage_close_decode_weight", 0.0)
                or 0.0
            ),
            "planned_candidates": [
                {
                    "token": str(
                        self.tokenizer.id_to_token.get(candidate_ids[index], "")
                    ),
                    "score_before": round(float(scores_before[index].item()), 6),
                    "plan_bias": round(float(coverage_bias[index].item()), 6),
                    "score_after": round(float(scores_after[index].item()), 6),
                }
                for index in targeted
            ],
            "top_candidates": [
                {
                    "token": str(
                        self.tokenizer.id_to_token.get(candidate_ids[index], "")
                    ),
                    "score_before": round(float(scores_before[index].item()), 6),
                    "plan_bias": round(float(coverage_bias[index].item()), 6),
                    "score_after": round(float(scores_after[index].item()), 6),
                }
                for index in ranked
            ],
            **self._choice_phase_evidence(state),
        }
        stats.constrained_selection_traces.append(trace)
        return trace

    def _semantic_plan_repeated_owner_id(
        self,
        row: int,
        state: Any,
    ) -> int | None:
        """Return the deepest repeated prompt-plan family in the active path."""
        return self._semantic_plan_owner_id(row, state, minimum_count=2)

    def _semantic_plan_owner_id(
        self,
        row: int,
        state: Any,
        *,
        minimum_count: int = 1,
    ) -> int | None:
        """Return the deepest authored prompt-plan family in the active path."""
        if (
            not self._semantic_plan_action_counts
            or row >= len(self._semantic_plan_action_counts)
        ):
            return None
        family_token_ids = {
            str(self.tokenizer.id_to_token.get(token_id, ""))
            .removeprefix("COMP:")
            .removeprefix("+"): token_id
            for token_id in self._component_inventory_token_ids()
        }
        for frame in reversed(getattr(state, "frames", ())):
            if frame.kind != "component":
                continue
            candidate_id = family_token_ids.get(
                str(frame.expr_type).removeprefix("element:")
            )
            if (
                self._semantic_plan_action_counts[row].get(candidate_id, 0)
                >= minimum_count
            ):
                return candidate_id
        return None

    @staticmethod
    def _schema_can_reach_visible_slot(schema: dict[str, Any]) -> bool:
        if schema.get("x-openui-placeholder") or schema.get("type") == "string":
            return True
        return any(
            TwoTowerModel._schema_can_reach_visible_slot(dict(child))
            for child in (
                *schema.get("anyOf", ()),
                *schema.get("properties", {}).values(),
                *(
                    (schema["items"],)
                    if isinstance(schema.get("items"), dict)
                    else ()
                ),
            )
            if isinstance(child, dict)
        )

    def _semantic_plan_typed_array_nonempty_bias(
        self,
        row: int,
        state: Any,
        prefix: list[int],
        candidate_ids: tuple[int, ...],
        scores: torch.Tensor,
    ) -> torch.Tensor | None:
        """Start a slot-bearing typed array for an authored plan component."""
        margin = float(
            getattr(
                self.config,
                "semantic_plan_typed_array_nonempty_margin_decode_weight",
                0.0,
            )
            or 0.0
        )
        typed_margin = float(
            getattr(
                self.config,
                "semantic_plan_typed_array_item_margin_decode_weight",
                0.0,
            )
            or 0.0
        )
        frames = list(getattr(state, "frames", ()))
        slot_contract = (
            self._slot_contracts[row]
            if self._slot_contracts and row < len(self._slot_contracts)
            else None
        )
        if (
            max(margin, typed_margin) <= 0.0
            or len(frames) < 2
            or not slot_contract
        ):
            return None
        frame = frames[-1]
        schemas = tuple(getattr(frame, "schemas", ()))
        if (
            getattr(frame, "kind", None) != "variadic"
            or getattr(frame, "expr_type", None) != "array"
            or int(getattr(frame, "item_count", 0)) != 0
            or not schemas
            or not self._schema_can_reach_visible_slot(dict(schemas[0]))
        ):
            return None
        owner_id = self._semantic_plan_owner_id(row, state)
        if owner_id is None:
            return None
        owner_frame = frames[-2]
        owner_family = str(getattr(owner_frame, "expr_type", "")).removeprefix(
            "element:"
        )
        if (
            getattr(owner_frame, "kind", None) != "component"
            or owner_id
            not in {
                self.tokenizer.token_to_id.get(f"+{owner_family}"),
                self.tokenizer.token_to_id.get(f"COMP:{owner_family}"),
                self.tokenizer.token_to_id.get(owner_family),
            }
        ):
            return None
        visible_slot_ids = {
            int(self.tokenizer.sym_id(index))
            for index in range(
                min(len(slot_contract), int(self.tokenizer.sym_slots))
            )
        }
        if not visible_slot_ids.difference(prefix):
            return None
        close_id = self.tokenizer.token_to_id.get(str(getattr(frame, "close", "")))
        if close_id not in candidate_ids:
            return None
        typed_target_id = None
        if typed_margin > 0.0:
            try:
                typed_target_id = state._minimal_schema_id(dict(schemas[0]))
            except (AttributeError, KeyError, TypeError, ValueError):
                return None
            if typed_target_id not in candidate_ids:
                return None
            targets = [candidate_ids.index(typed_target_id)]
        else:
            targets = [
                position
                for position, token_id in enumerate(candidate_ids)
                if token_id != close_id
            ]
        if not targets:
            return None
        target = max(targets, key=lambda position: float(scores[position].item()))
        bias = scores.new_zeros(len(candidate_ids))
        bias[target] = max(
            0.0,
            float(scores.max().item())
            + max(margin, typed_margin)
            - float(scores[target].item()),
        )
        return bias

    def _semantic_plan_repeated_array_close_bias(
        self,
        row: int,
        state: Any,
        candidate_ids: tuple[int, ...],
        scores: torch.Tensor,
    ) -> torch.Tensor | None:
        """Close nested arrays after one item when a repeated plan family owns them."""
        margin = float(
            getattr(
                self.config,
                "semantic_plan_repeated_array_close_margin_decode_weight",
                0.0,
            )
            or 0.0
        )
        frames = list(getattr(state, "frames", ()))
        if (
            margin <= 0.0
            or len(frames) < 2
            or row >= len(self._semantic_plan_action_counts or ())
            or frames[-1].kind != "variadic"
            or int(getattr(frames[-1], "item_count", 0)) < 1
        ):
            return None
        owner_id = self._semantic_plan_repeated_owner_id(row, state)
        if owner_id is None:
            return None
        close_id = self.tokenizer.token_to_id.get(str(frames[-1].close))
        if close_id not in candidate_ids:
            return None
        target = candidate_ids.index(close_id)
        bias = scores.new_zeros(len(candidate_ids))
        bias[target] = max(
            0.0,
            float(scores.max().item()) + margin - float(scores[target].item()),
        )
        return bias

    def _semantic_plan_repeated_slot_bias(
        self,
        row: int,
        state: Any,
        prefix: list[int],
        candidate_ids: tuple[int, ...],
        scores: torch.Tensor,
    ) -> torch.Tensor | None:
        """Floor the best unused visible slot for each repeated plan instance."""
        margin = float(
            getattr(
                self.config,
                "semantic_plan_repeated_slot_margin_decode_weight",
                0.0,
            )
            or 0.0
        )
        slot_contract = (
            self._slot_contracts[row]
            if self._slot_contracts and row < len(self._slot_contracts)
            else None
        )
        owner_id = self._semantic_plan_repeated_owner_id(row, state)
        if margin <= 0.0 or not slot_contract or owner_id is None:
            return None
        owner_position = max(
            (
                position
                for position, token_id in enumerate(prefix)
                if token_id == owner_id
            ),
            default=-1,
        )
        if owner_position < 0:
            return None
        visible_slot_ids = {
            int(self.tokenizer.sym_id(index))
            for index in range(min(len(slot_contract), int(self.tokenizer.sym_slots)))
        }
        if any(
            token_id in visible_slot_ids for token_id in prefix[owner_position + 1 :]
        ):
            return None
        used_slot_ids = visible_slot_ids.intersection(prefix[:owner_position])
        targets = [
            position
            for position, token_id in enumerate(candidate_ids)
            if token_id in visible_slot_ids and token_id not in used_slot_ids
        ]
        if not targets:
            return None
        target = max(targets, key=lambda position: float(scores[position].item()))
        bias = scores.new_zeros(len(candidate_ids))
        bias[target] = max(
            0.0,
            float(scores.max().item()) + margin - float(scores[target].item()),
        )
        return bias

    @staticmethod
    def _schema_contains_enum(schema: dict[str, Any]) -> bool:
        if schema.get("enum"):
            return True
        return any(
            TwoTowerModel._schema_contains_enum(dict(option))
            for key in ("anyOf", "oneOf")
            for option in schema.get(key, ())
            if isinstance(option, dict)
        )

    def _semantic_plan_binding_bias(
        self,
        row: int,
        state: Any,
        prefix: list[int],
        candidate_ids: tuple[int, ...],
    ) -> torch.Tensor | None:
        """Prefer legal root references backed by predicted plan families."""
        weight = float(
            getattr(self.config, "semantic_plan_binding_decode_weight", 0.0) or 0.0
        )
        frames = list(getattr(state, "frames", ()))
        structural_root_list = bool(
            getattr(state, "mode", None) == "structural"
            and frames
            and frames[-1].kind == "variadic"
            and frames[-1].expr_type == "array"
            and (
                len(frames) == 1
                or (
                    len(frames) == 2
                    and frames[-2].kind == "component"
                    and frames[-2].expr_type == "element:Stack"
                )
            )
        )
        if (
            weight <= 0.0
            or not structural_root_list
            or not self._semantic_plan_action_scores
            or row >= len(self._semantic_plan_action_scores)
        ):
            return None
        action_scores = self._semantic_plan_action_scores[row]
        if not action_scores:
            return None
        family_token_ids = {
            str(self.tokenizer.id_to_token[token_id])
            .removeprefix("COMP:")
            .removeprefix("+"): token_id
            for token_id in self._component_inventory_token_ids()
        }
        used = {
            int(token[1:])
            for token_id in prefix
            if (
                token := str(self.tokenizer.id_to_token.get(int(token_id), ""))
            ).startswith("&")
            and token[1:].isdigit()
        }
        bias = torch.zeros(
            len(candidate_ids), dtype=torch.float32, device=self.device_name
        )
        applied = False
        section_types = tuple(getattr(state, "section_types", ()))
        for position, token_id in enumerate(candidate_ids):
            token = str(self.tokenizer.id_to_token.get(int(token_id), ""))
            if not token.startswith("&") or not token[1:].isdigit():
                continue
            reference = int(token[1:])
            if reference in used or reference >= len(section_types):
                continue
            expr_type = str(section_types[reference])
            if not expr_type.startswith("element:"):
                continue
            component_id = family_token_ids.get(expr_type.removeprefix("element:"))
            confidence = action_scores.get(component_id, 0.0)
            if confidence > 0.0:
                bias[position] = weight * confidence
                applied = True
        return bias if applied else None

    def _semantic_plan_root_bias(
        self,
        row: int,
        state: Any,
        prefix: list[int] | None,
        candidate_ids: tuple[int, ...],
        candidate_scores: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Soft-follow a verifier-valid Stack closure after plan role coverage."""
        weight = float(
            getattr(self.config, "semantic_plan_root_decode_weight", 0.0) or 0.0
        )
        margin = float(
            getattr(
                self.config,
                "semantic_plan_root_margin_decode_weight",
                0.0,
            )
            or 0.0
        )
        if (
            (weight <= 0.0 and margin <= 0.0)
            or getattr(state, "mode", None) != "structural"
            or not self._semantic_plan_action_scores
            or row >= len(self._semantic_plan_action_scores)
        ):
            return None
        action_scores = self._semantic_plan_action_scores[row]
        required_ids = {
            token_id
            for token_id, confidence in action_scores.items()
            if confidence > 0.0
        }
        if not required_ids:
            return None
        family_token_ids = {
            str(self.tokenizer.id_to_token[token_id])
            .removeprefix("COMP:")
            .removeprefix("+"): token_id
            for token_id in self._component_inventory_token_ids()
        }
        section_types = tuple(getattr(state, "section_types", ()))
        required_counts = (
            self._semantic_plan_action_counts[row]
            if self._semantic_plan_action_counts
            and row < len(self._semantic_plan_action_counts)
            else {token_id: 1 for token_id in required_ids}
        )
        covered_counts = self._semantic_plan_covered_counts(
            state, prefix, family_token_ids
        )
        if any(
            covered_counts.get(token_id, 0) < required_count
            for token_id, required_count in required_counts.items()
        ):
            return None

        # Keep the unit seam for callers without a concrete prefix. Production
        # decode always supplies one and takes the verifier-gated path.
        if prefix is None:
            stack_complete = bool(
                section_types and section_types[-1] == "element:Stack"
            )
            target_id = (
                self.tokenizer.eos_id
                if stack_complete
                else self.tokenizer.token_to_id.get("+Stack")
            )
        else:
            tokens = [
                str(self.tokenizer.id_to_token.get(int(token_id), ""))
                for token_id in prefix
                if int(token_id)
                not in {
                    self.tokenizer.bos_id,
                    self.tokenizer.eos_id,
                    self.tokenizer.pad_id,
                }
            ]
            frames = tuple(getattr(state, "frames", ()))
            if not frames and section_types and section_types[-1] == "element:Stack":
                planned = tokens
                target_id = self.tokenizer.eos_id
            else:
                remaining = dict(required_counts)
                references: list[str] = []
                for index, expr_type in enumerate(section_types):
                    token_id = family_token_ids.get(
                        str(expr_type).removeprefix("element:")
                    )
                    if token_id is None or remaining.get(token_id, 0) <= 0:
                        continue
                    references.append(f"&{index}")
                    remaining[token_id] -= 1
                if not references:
                    return None
                closure = ["+Stack", "[", *references, "]", "^column", "-"]
                if frames:
                    try:
                        stack_start = len(tokens) - 1 - tokens[::-1].index("+Stack")
                    except ValueError:
                        return None
                    base = tokens[:stack_start]
                    consumed = tokens[stack_start:]
                    if consumed != closure[: len(consumed)]:
                        return None
                    planned = [*base, *closure]
                    if len(consumed) >= len(closure):
                        return None
                    target_id = self.tokenizer.token_to_id.get(closure[len(consumed)])
                else:
                    planned = [*tokens, *closure]
                    target_id = self.tokenizer.token_to_id.get(closure[0])
            try:
                from slm_training.dsl.parser import validate
                from slm_training.dsl.production_codec import decode_choices

                slot_contract = (
                    self._slot_contracts[row]
                    if self._slot_contracts and row < len(self._slot_contracts)
                    else ()
                )
                validate(decode_choices(planned, slot_contract or ()))
            except Exception:  # noqa: BLE001 - predicted plans fail closed
                return None
        if target_id is None or target_id not in candidate_ids:
            return None
        bias = torch.zeros(
            len(candidate_ids), dtype=torch.float32, device=self.device_name
        )
        target_position = candidate_ids.index(target_id)
        bias[target_position] = weight
        if margin > 0.0 and candidate_scores is not None:
            margin_bias = (
                float(candidate_scores.max().item())
                + margin
                - float(candidate_scores[target_position].item())
            )
            bias[target_position] = max(
                float(bias[target_position].item()),
                margin_bias,
            )
        return bias

    def _component_edge_bias(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor | None,
        prefix: list[int],
        candidate_ids: tuple[int, ...],
        candidate_kinds: tuple[str, ...],
    ) -> torch.Tensor | None:
        weight = float(getattr(self.config, "component_edge_decode_weight", 0.0) or 0.0)
        if weight <= 0.0 or self.component_edge_head is None:
            return None
        from slm_training.dsl.grammar.fastpath.compiler_draft import (
            active_parent_component_ids,
        )

        parents = active_parent_component_ids(self.tokenizer, prefix)
        if not parents:
            return None
        component_ids = self._component_inventory_token_ids()
        component_index = {
            token_id: index for index, token_id in enumerate(component_ids)
        }
        parent_indices = [
            component_index[token_id]
            for token_id in parents
            if token_id in component_index
        ]
        if not parent_indices:
            return None
        logits = self.component_edge_head(self._pool_context(ctx, ctx_pad))[0].view(
            len(component_ids), len(component_ids)
        )
        parent_index = torch.as_tensor(parent_indices, device=logits.device)
        child_logits = logits.index_select(0, parent_index).mean(dim=0)
        bias = logits.new_zeros(len(candidate_ids))
        applied = False
        for position, (token_id, kind) in enumerate(
            zip(candidate_ids, candidate_kinds, strict=True)
        ):
            child = component_index.get(token_id)
            if child is not None and kind == "component_bound":
                bias[position] = weight * child_logits[child]
                applied = True
        return bias if applied else None

    def _visible_reference_completeness_bias(
        self,
        state: Any,
        prefix: list[int],
        candidate_ids: tuple[int, ...],
    ) -> torch.Tensor | None:
        """Prefer unused generated element references in terminal root aggregation."""
        weight = float(
            getattr(self.config, "visible_reference_decode_weight", 0.0) or 0.0
        )
        structural_root_list = bool(
            getattr(state, "mode", None) == "structural"
            and len(state.frames) == 1
            and state.frames[-1].kind == "variadic"
            and state.frames[-1].expr_type == "array"
        )
        if (
            weight <= 0.0
            or not state.frames
            or (state.current_marker != "r=" and not structural_root_list)
        ):
            return None
        eligible = {
            index
            for index, expr_type in enumerate(state.section_types)
            if str(expr_type).startswith("element:")
        }
        used: set[int] = set()
        for token_id in prefix:
            token = str(self.tokenizer.id_to_token.get(int(token_id), ""))
            if token.startswith("&"):
                try:
                    used.add(int(token[1:]))
                except ValueError:
                    continue
        unused = eligible - used
        if not unused:
            return None
        bias = torch.zeros(
            len(candidate_ids), dtype=torch.float32, device=self.device_name
        )
        applied = False
        for position, token_id in enumerate(candidate_ids):
            token = str(self.tokenizer.id_to_token.get(int(token_id), ""))
            if not token.startswith("&"):
                continue
            try:
                reference = int(token[1:])
            except ValueError:
                continue
            if reference in unused:
                bias[position] = weight
                applied = True
        return bias if applied else None

    def _root_reference_arity_bias(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor | None,
        state: Any,
        candidate_ids: tuple[int, ...],
    ) -> torch.Tensor | None:
        """Learned continue/stop bias for a terminal structural root list."""
        weight = float(
            getattr(self.config, "root_reference_arity_decode_weight", 0.0) or 0.0
        )
        if (
            weight <= 0.0
            or self.root_reference_arity_head is None
            or getattr(state, "mode", None) != "structural"
            or len(getattr(state, "frames", ())) != 1
            or state.frames[-1].kind != "variadic"
            or state.frames[-1].expr_type != "array"
        ):
            return None
        emitted = int(getattr(state.frames[-1], "reference_count", 0))
        logits = self.root_reference_arity_head(self._pool_context(ctx, ctx_pad))[0]
        max_reference_count = min(
            len(getattr(state, "section_types", ())), logits.numel() - 1
        )
        if max_reference_count <= 0:
            return None
        logits = logits[: max_reference_count + 1]
        split = min(emitted + 1, logits.numel())
        stop_score = torch.logsumexp(logits[:split], dim=0)
        continue_score = (
            torch.logsumexp(logits[split:], dim=0)
            if split < logits.numel()
            else logits.new_tensor(-20.0)
        )
        bias = logits.new_zeros(len(candidate_ids))
        applied = False
        for position, token_id in enumerate(candidate_ids):
            token = str(self.tokenizer.id_to_token.get(int(token_id), ""))
            if token == "]":
                bias[position] = weight * stop_score
                applied = True
            elif token.startswith("&"):
                bias[position] = weight * continue_score
                applied = True
        return bias if applied else None

    def _root_reference_identity_bias(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor | None,
        state: Any,
        prefix: list[int],
        candidate_ids: tuple[int, ...],
        scores: torch.Tensor,
    ) -> torch.Tensor | None:
        """Re-rank terminal-root references without changing their best score."""
        weight = float(
            getattr(self.config, "root_reference_identity_decode_weight", 0.0) or 0.0
        )
        if (
            weight <= 0.0
            or self.root_reference_identity_head is None
            or getattr(state, "mode", None) != "structural"
            or len(getattr(state, "frames", ())) != 1
            or state.frames[-1].kind != "variadic"
            or state.frames[-1].expr_type != "array"
        ):
            return None
        valid_count = min(
            len(getattr(state, "section_types", ())),
            self.root_reference_identity_head.out_features,
        )
        if valid_count <= 0:
            return None
        logits = self.root_reference_identity_head(self._pool_context(ctx, ctx_pad))[
            0, :valid_count
        ]
        used: set[int] = set()
        for token_id in prefix:
            token = str(self.tokenizer.id_to_token.get(int(token_id), ""))
            if token.startswith("&"):
                try:
                    used.add(int(token[1:]))
                except ValueError:
                    continue
        references: list[tuple[int, int]] = []
        for position, token_id in enumerate(candidate_ids):
            token = str(self.tokenizer.id_to_token.get(int(token_id), ""))
            if not token.startswith("&"):
                continue
            try:
                reference = int(token[1:])
            except ValueError:
                continue
            if not 0 <= reference < valid_count:
                continue
            references.append((position, reference))
        unused = [
            (position, reference)
            for position, reference in references
            if reference not in used
        ]
        if not unused:
            return None
        ranked_unused = sorted(
            unused,
            key=lambda item: float(logits[item[1]].detach().cpu()),
            reverse=True,
        )
        ranked_reference_scores = sorted(
            (scores[position] for position, _ in references),
            key=lambda score: float(score.detach().cpu()),
            reverse=True,
        )
        mix = min(weight, 1.0)
        bias = logits.new_zeros(len(candidate_ids))
        for rank, (position, _) in enumerate(ranked_unused):
            bias[position] = mix * (ranked_reference_scores[rank] - scores[position])
        for position, reference in references:
            if reference in used:
                bias[position] = logits.new_tensor(-20.0) * mix
        return bias

    @staticmethod
    def _choice_phase_evidence(state: Any) -> dict[str, object]:
        """Describe the bounded generated-state phase around a choice."""
        frames = list(getattr(state, "frames", ()))
        structural_list = bool(
            getattr(state, "mode", None) == "structural"
            and frames
            and frames[-1].kind == "variadic"
            and frames[-1].expr_type == "array"
        )
        if getattr(state, "current_marker", None) == "r=":
            aggregation_scope = "v05_root"
        elif structural_list:
            aggregation_scope = (
                "structural_root_list" if len(frames) == 1 else "structural_nested_list"
            )
        else:
            aggregation_scope = "other"
        return {
            "aggregation_scope": aggregation_scope,
            "frame_depth": len(frames),
            "frame_path_truncated": len(frames) > 8,
            "frame_path": [
                {
                    "kind": str(frame.kind),
                    "expr_type": str(frame.expr_type),
                    "phase": str(frame.phase),
                    "arg_index": int(frame.arg_index),
                }
                for frame in frames[-8:]
            ],
        }

    def _binder_component_plan_bias(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor | None,
        prefix: list[int],
        candidate_ids: tuple[int, ...],
        candidate_kinds: tuple[str, ...],
    ) -> torch.Tensor | None:
        weight = float(
            getattr(self.config, "binder_component_plan_decode_weight", 0.0) or 0.0
        )
        if weight <= 0.0 or self.binder_component_plan_head is None:
            return None
        from slm_training.dsl.grammar.fastpath.compiler_draft import (
            active_declaration_binder_id,
        )

        binder_id = active_declaration_binder_id(self.tokenizer, prefix)
        binder_ids = self._binder_component_token_ids()
        try:
            binder = binder_ids.index(int(binder_id))
        except (TypeError, ValueError):
            return None
        component_ids = self._component_inventory_token_ids()
        component_index = {
            token_id: index for index, token_id in enumerate(component_ids)
        }
        logits = self.binder_component_plan_head(self._pool_context(ctx, ctx_pad))[
            0
        ].view(len(binder_ids), len(component_ids))[binder]
        bias = logits.new_zeros(len(candidate_ids))
        applied = False
        for position, (token_id, kind) in enumerate(
            zip(candidate_ids, candidate_kinds, strict=True)
        ):
            child = component_index.get(token_id)
            if child is not None and kind == "component_bound":
                bias[position] = weight * logits[child]
                applied = True
        return bias if applied else None

    def _binder_topology_bias(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor | None,
        prefix: list[int],
        candidate_ids: tuple[int, ...],
        candidate_kinds: tuple[str, ...],
    ) -> torch.Tensor | None:
        weight = float(
            getattr(self.config, "binder_topology_decode_weight", 0.0) or 0.0
        )
        if weight <= 0.0 or self.binder_topology_head is None:
            return None
        from slm_training.dsl.grammar.fastpath.compiler_draft import (
            active_declaration_binder_id,
        )

        binder_ids = self._binder_component_token_ids()
        binder_index = {token_id: index for index, token_id in enumerate(binder_ids)}
        parent = binder_index.get(active_declaration_binder_id(self.tokenizer, prefix))
        if parent is None:
            return None
        logits = self.binder_topology_head(self._pool_context(ctx, ctx_pad))[0].view(
            len(binder_ids), len(binder_ids)
        )[parent]
        bias = logits.new_zeros(len(candidate_ids))
        applied = False
        for position, (token_id, kind) in enumerate(
            zip(candidate_ids, candidate_kinds, strict=True)
        ):
            child = binder_index.get(token_id)
            if child is not None and kind.startswith("bind_reference"):
                bias[position] = weight * logits[child]
                applied = True
        return bias if applied else None

    def _binder_arity_path_bias(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor | None,
        prefix: list[int],
        paths: tuple,
    ) -> list[float] | None:
        weight = float(getattr(self.config, "binder_arity_decode_weight", 0.0) or 0.0)
        if weight <= 0.0 or self.binder_arity_head is None:
            return None
        from slm_training.dsl.grammar.fastpath.compiler_draft import (
            active_declaration_binder_id,
            active_declaration_reference_count,
        )

        binder_ids = self._binder_component_token_ids()
        binder_index = {token_id: index for index, token_id in enumerate(binder_ids)}
        parent = binder_index.get(active_declaration_binder_id(self.tokenizer, prefix))
        emitted = active_declaration_reference_count(self.tokenizer, prefix)
        if parent is None or emitted is None:
            return None
        continues = [
            any(int(token_id) in binder_index for token_id in path.token_ids)
            for path in paths
        ]
        if not any(continues) or all(continues):
            return None
        logits = self.binder_arity_head(self._pool_context(ctx, ctx_pad))[0].view(
            len(binder_ids), len(binder_ids) + 1
        )[parent]
        split = min(int(emitted) + 1, logits.numel())
        stop_score = torch.logsumexp(logits[:split], dim=0)
        continue_score = (
            torch.logsumexp(logits[split:], dim=0)
            if split < logits.numel()
            else logits.new_tensor(-1e4)
        )
        return [
            weight * float(continue_score if continues[index] else stop_score)
            for index in range(len(paths))
        ]

    def _select_compiler_path(
        self,
        prefix: list[int],
        paths: tuple,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        length: int,
        *,
        tree: bool,
        slot_contract: list[str] | None = None,
    ) -> tuple[int, ...]:
        """Rank completion paths using gathered rows of the tied LM head."""
        if len(paths) == 1:
            return tuple(paths[0].token_ids)
        stats = get_active_stats()
        recorder = getattr(self, "trace_recorder", None)
        if stats is not None:
            stats.compiler_candidates += len(paths)

        # SLM-176: optionally retrieve-then-rerank the legal action set.  When
        # disabled this is a no-op and leaves ``paths`` unchanged.
        paths, _shortlist_trace = self._maybe_apply_action_shortlist(paths, prefix)

        def record_choice(
            chosen: int,
            scores: list[float],
            phase: str,
            *,
            first_edge_scores: list[float] | None = None,
        ) -> None:
            if stats is not None and len(stats.constrained_selection_traces) < 64:
                ranked = sorted(
                    range(len(paths)), key=scores.__getitem__, reverse=True
                )[:5]
                stats.constrained_selection_traces.append(
                    {
                        "position": len(prefix),
                        "prefix_text": self.tokenizer.decode(prefix),
                        "chosen_token": self.tokenizer.id_to_token.get(
                            int(paths[chosen].token_ids[0]), ""
                        ),
                        "legal_candidates": len(paths),
                        "forced": False,
                        "phase": phase,
                        "top_candidates": [
                            {
                                "token": self.tokenizer.id_to_token.get(
                                    int(paths[index].token_ids[0]), ""
                                ),
                                "score": round(float(scores[index]), 6),
                                **(
                                    {
                                        "first_edge_score": round(
                                            float(first_edge_scores[index]), 6
                                        )
                                    }
                                    if first_edge_scores is not None
                                    else {}
                                ),
                            }
                            for index in ranked
                        ],
                    }
                )

        if not tree:
            canvas = self._compiler_canvas(prefix, length)
            hidden = self._denoiser_hidden(canvas, ctx, ctx_pad)
            candidates = tuple(int(path.token_ids[0]) for path in paths)
            scores = self._project_candidates(hidden[0, len(prefix)], candidates)
            inventory_bias = self._component_inventory_bias(ctx, ctx_pad, candidates)
            if inventory_bias is not None:
                scores = scores + inventory_bias
            plan_bias = self._component_plan_bias(
                ctx, ctx_pad, prefix, candidates, tuple(path.kind for path in paths)
            )
            if plan_bias is not None:
                before_plan = int(scores.argmax().item())
                scores = scores + plan_bias
                if stats is not None:
                    stats.component_plan_applications += 1
                    stats.component_plan_choice_changes += int(
                        int(scores.argmax().item()) != before_plan
                    )
            slot_bias = self._slot_component_bias(
                ctx,
                ctx_pad,
                prefix,
                candidates,
                tuple(path.kind for path in paths),
                slot_contract,
            )
            if slot_bias is not None:
                before_slot = int(scores.argmax().item())
                scores = scores + slot_bias
                if stats is not None:
                    stats.slot_component_applications += 1
                    stats.slot_component_choice_changes += int(
                        int(scores.argmax().item()) != before_slot
                    )
            edge_bias = self._component_edge_bias(
                ctx, ctx_pad, prefix, candidates, tuple(path.kind for path in paths)
            )
            if edge_bias is not None:
                before_edge = int(scores.argmax().item())
                scores = scores + edge_bias
                if stats is not None:
                    stats.component_edge_applications += 1
                    stats.component_edge_choice_changes += int(
                        int(scores.argmax().item()) != before_edge
                    )
            binder_bias = self._binder_component_plan_bias(
                ctx, ctx_pad, prefix, candidates, tuple(path.kind for path in paths)
            )
            if binder_bias is not None:
                before_binder = int(scores.argmax().item())
                scores = scores + binder_bias
                if stats is not None:
                    stats.binder_component_plan_applications += 1
                    stats.binder_component_plan_choice_changes += int(
                        int(scores.argmax().item()) != before_binder
                    )
            topology_bias = self._binder_topology_bias(
                ctx, ctx_pad, prefix, candidates, tuple(path.kind for path in paths)
            )
            if topology_bias is not None:
                before_topology = int(scores.argmax().item())
                scores = scores + topology_bias
                if stats is not None:
                    stats.binder_topology_applications += 1
                    stats.binder_topology_choice_changes += int(
                        int(scores.argmax().item()) != before_topology
                    )
            if bool(getattr(self.config, "grammar_sample_decode", False)):
                temp = float(
                    getattr(self.config, "grammar_sample_temperature", 0.8) or 0.8
                )
                chosen = int(torch.multinomial(F.softmax(scores / temp, dim=0), 1))
            else:
                chosen = int(scores.argmax().item())
            record_choice(
                chosen,
                [float(score) for score in scores.detach().cpu().tolist()],
                "compiler_restricted",
            )
            return tuple(paths[chosen].token_ids)

        # Prefix trie: each distinct parent canvas is encoded once, and all of
        # its children are scored by a gathered projection. Forced single-child
        # edges contribute zero after constrained renormalization.
        with timed_ms(stats, "trie_ms"):
            children: dict[tuple[int, ...], set[int]] = {}
            for path in paths:
                parent = tuple(prefix)
                for token_id in path.token_ids:
                    children.setdefault(parent, set()).add(int(token_id))
                    parent = (*parent, int(token_id))
            parents = [parent for parent in children if len(parent) < length]
            canvases = torch.cat(
                [self._compiler_canvas(list(parent), length) for parent in parents],
                dim=0,
            )
            k = len(parents)
            hidden = self._denoiser_hidden(
                canvases,
                ctx.expand(k, -1, -1),
                ctx_pad.expand(k, -1) if ctx_pad is not None else ctx_pad,
            )
            edge_scores: dict[tuple[tuple[int, ...], int], float] = {}
            first_edge_kinds = {
                int(path.token_ids[0]): str(path.kind)
                for path in paths
                if path.token_ids
            }
            for row, parent in enumerate(parents):
                candidate_ids = tuple(sorted(children[parent]))
                if len(candidate_ids) == 1:
                    edge_scores[(parent, candidate_ids[0])] = 0.0
                    continue
                scores = self._project_candidates(
                    hidden[row, len(parent)], candidate_ids
                )
                inventory_bias = self._component_inventory_bias(
                    ctx, ctx_pad, candidate_ids
                )
                if inventory_bias is not None:
                    scores = scores + inventory_bias
                if parent == tuple(prefix):
                    plan_bias = self._component_plan_bias(
                        ctx,
                        ctx_pad,
                        prefix,
                        candidate_ids,
                        tuple(
                            first_edge_kinds.get(token_id, "")
                            for token_id in candidate_ids
                        ),
                    )
                    if plan_bias is not None:
                        before_plan = int(scores.argmax().item())
                        scores = scores + plan_bias
                        if stats is not None:
                            stats.component_plan_applications += 1
                            stats.component_plan_choice_changes += int(
                                int(scores.argmax().item()) != before_plan
                            )
                    slot_bias = self._slot_component_bias(
                        ctx,
                        ctx_pad,
                        prefix,
                        candidate_ids,
                        tuple(
                            first_edge_kinds.get(token_id, "")
                            for token_id in candidate_ids
                        ),
                        slot_contract,
                    )
                    if slot_bias is not None:
                        before_slot = int(scores.argmax().item())
                        scores = scores + slot_bias
                        if stats is not None:
                            stats.slot_component_applications += 1
                            stats.slot_component_choice_changes += int(
                                int(scores.argmax().item()) != before_slot
                            )
                    edge_bias = self._component_edge_bias(
                        ctx,
                        ctx_pad,
                        prefix,
                        candidate_ids,
                        tuple(
                            first_edge_kinds.get(token_id, "")
                            for token_id in candidate_ids
                        ),
                    )
                    if edge_bias is not None:
                        before_edge = int(scores.argmax().item())
                        scores = scores + edge_bias
                        if stats is not None:
                            stats.component_edge_applications += 1
                            stats.component_edge_choice_changes += int(
                                int(scores.argmax().item()) != before_edge
                            )
                    binder_bias = self._binder_component_plan_bias(
                        ctx,
                        ctx_pad,
                        prefix,
                        candidate_ids,
                        tuple(
                            first_edge_kinds.get(token_id, "")
                            for token_id in candidate_ids
                        ),
                    )
                    if binder_bias is not None:
                        before_binder = int(scores.argmax().item())
                        scores = scores + binder_bias
                        if stats is not None:
                            stats.binder_component_plan_applications += 1
                            stats.binder_component_plan_choice_changes += int(
                                int(scores.argmax().item()) != before_binder
                            )
                    topology_bias = self._binder_topology_bias(
                        ctx,
                        ctx_pad,
                        prefix,
                        candidate_ids,
                        tuple(
                            first_edge_kinds.get(token_id, "")
                            for token_id in candidate_ids
                        ),
                    )
                    if topology_bias is not None:
                        before_topology = int(scores.argmax().item())
                        scores = scores + topology_bias
                        if stats is not None:
                            stats.binder_topology_applications += 1
                            stats.binder_topology_choice_changes += int(
                                int(scores.argmax().item()) != before_topology
                            )
                log_probs = F.log_softmax(scores, dim=0)
                for i, token_id in enumerate(candidate_ids):
                    edge_scores[(parent, token_id)] = float(log_probs[i].item())
            if stats is not None:
                stats.trie_nodes += len(parents)

        path_scores: list[float] = []
        for path in paths:
            parent = tuple(prefix)
            score = 0.0
            branches = 0
            for token_id in path.token_ids:
                score += edge_scores.get((parent, int(token_id)), 0.0)
                if len(children.get(parent, ())) > 1:
                    branches += 1
                parent = (*parent, int(token_id))
            path_scores.append(score / max(1, branches))
        arity_bias = self._binder_arity_path_bias(ctx, ctx_pad, prefix, paths)
        if arity_bias is not None:
            before_arity = max(range(len(paths)), key=path_scores.__getitem__)
            path_scores = [
                score + arity_bias[index] for index, score in enumerate(path_scores)
            ]
            if stats is not None:
                stats.binder_arity_applications += 1
                stats.binder_arity_choice_changes += int(
                    max(range(len(paths)), key=path_scores.__getitem__) != before_arity
                )
        if bool(getattr(self.config, "grammar_sample_decode", False)):
            temp = float(getattr(self.config, "grammar_sample_temperature", 0.8) or 0.8)
            probs = F.softmax(torch.tensor(path_scores) / temp, dim=0)
            chosen = int(torch.multinomial(probs, 1).item())
        else:
            chosen = max(range(len(paths)), key=path_scores.__getitem__)
        first_edge_scores = [
            edge_scores.get((tuple(prefix), int(path.token_ids[0])), 0.0)
            for path in paths
        ]
        if recorder is not None:
            parent_rows = {parent: row for row, parent in enumerate(parents)}
            commits: list[dict[str, object]] = []
            parent = tuple(prefix)
            for token_id in paths[chosen].token_ids:
                allowed = tuple(sorted(children.get(parent, ())))
                if len(allowed) > 1:
                    row = parent_rows[parent]
                    raw_scores = self.denoiser.project(hidden[row, len(parent)])
                    raw_id = int(raw_scores.argmax().item())
                    log_probs = F.log_softmax(raw_scores.float(), dim=-1)
                    commit: dict[str, object] = {
                        "t": len(parent),
                        "id": int(token_id),
                        "raw_id": raw_id,
                        "lp": float(log_probs[int(token_id)].item()),
                        "pre_canvas": self._compiler_canvas(list(parent), length)[
                            0
                        ].tolist(),
                        "phase": "compiler_tree",
                        "decision_kind": str(paths[chosen].kind),
                    }
                    if recorder.record_support:
                        commit["allowed_id_set"] = list(allowed)
                    commits.append(commit)
                parent = (*parent, int(token_id))
            if commits:
                recorder.step(
                    f"compiler_tree:{len(prefix)}",
                    canvas=self._compiler_canvas(prefix, length)[0].tolist(),
                    commits=commits,
                )
        record_choice(
            chosen,
            path_scores,
            "compiler_tree",
            first_edge_scores=first_edge_scores,
        )
        return tuple(paths[chosen].token_ids)

    def _solver_prune_forest(self, forest, prefix):
        """VSS1-03: prune the compiler forest to the certified live subset.

        Only reached when ``verified_solver_decode`` is on. Runs certificate-checked
        exact closure (the VSS0-04 oracle) over the forest and drops only candidates
        proven ``UNSUPPORTED`` with a replay-valid certificate; ``UNKNOWN`` candidates
        are kept (``keep_and_rank``) so the ordinary soft ranker still sees them. An
        unsupported tokenizer/pack fails with a clear capability error (never a
        silent weaker path).
        """
        from slm_training.dsl.language_contract import contract_id
        from slm_training.dsl.solver.closure import EnumerativeSupportProvider
        from slm_training.dsl.solver.decode import solver_prune
        from slm_training.dsl.solver.openui_support import (
            OpenUIForestExpander,
            OpenUIWellFormedVerifier,
        )
        from slm_training.dsl.solver.state import SolverBounds
        from slm_training.models.decode_stats import get_active_stats, timed_ms
        from slm_training.models.dsl_tokenizer import is_dsl_native_tokenizer

        if not is_dsl_native_tokenizer(self.tokenizer):
            raise ValueError(
                "verified_solver_decode requires a DSL-native tokenizer/pack; "
                f"{type(self.tokenizer).__name__} is unsupported"
            )
        if forest.coverage != "complete" or not forest.paths:
            return forest  # closure is authoritative only over an exhaustive set

        max_nodes = int(getattr(self.config, "solver_max_nodes", 512) or 512)
        bounds = SolverBounds(
            max_tokens=max(1, max_nodes * 64),
            max_nodes=max_nodes,
            max_depth=int(getattr(self.config, "solver_max_depth", 64) or 64),
            max_backtracks=int(getattr(self.config, "solver_max_backtracks", 64) or 64),
            max_verifier_calls=int(
                getattr(self.config, "solver_max_verifier_calls", 64) or 64
            ),
        )
        cv = contract_id()
        window = int(getattr(self.config, "grammar_draft_window", 8) or 8)
        expander = OpenUIForestExpander(
            self.tokenizer,
            prefix,
            pack_id="openui",
            constraint_version=cv,
            bounds=bounds,
            max_path_tokens=window,
        )
        provider = EnumerativeSupportProvider(expander, OpenUIWellFormedVerifier())
        policy = str(getattr(self.config, "solver_unknown_policy", "keep_and_rank"))
        root_state = expander.root_state()
        certificate_store: dict = {}
        stats = get_active_stats()
        # Solver wall time is separated from denoiser_ms/projection_ms (VSS1-04).
        with timed_ms(stats, "solver_ms"):
            pruned, result = solver_prune(
                forest,
                prefix,
                provider,
                pack_id="openui",
                constraint_version=cv,
                bounds=bounds,
                unknown_policy=policy,
                state=root_state,
                cache={},
                certificate_store=certificate_store,
            )
        if result is not None:
            self._record_solver_metrics(result, root_state, certificate_store, stats)
        return pruned

    def _record_solver_metrics(self, result, root_state, certificate_store, stats):
        """VSS1-04: fold solver work into decode stats and, when a trace recorder
        is attached, emit replayable solver-transition events + a bounded
        certificate/counter sidecar. Counters ride the existing DecodeStats
        envelope; nothing is emitted when neither stats nor a recorder is active.
        """
        from slm_training.dsl.solver.replay import (
            SOLVER_TRACE_SCHEMA_VERSION,
            closure_status,
            serialize_certificates,
            solver_events_from_closure,
            solver_trace_counters,
        )

        if stats is not None:
            counters = result.counters
            stats.solver_enabled = 1
            stats.solver_closure_passes += counters.passes
            stats.solver_support_queries += counters.support_queries
            stats.solver_support_cache_hits += counters.cache_hits
            stats.solver_supported += counters.supported
            stats.solver_unsupported += counters.unsupported
            stats.solver_unknown += counters.unknown
            stats.solver_certified_removed += counters.candidates_removed
            stats.solver_expanded_nodes += counters.expanded_nodes
            stats.solver_verifier_calls += counters.verifier_calls
            stats.solver_terminal_status = closure_status(result)

        recorder = getattr(self, "trace_recorder", None)
        if recorder is None:
            return
        mode = str(getattr(self.config, "solver_certificate_mode", "summary"))
        events = solver_events_from_closure(result, root_state, certificate_mode=mode)
        for event in events:
            kind = event["kind"]
            payload = {key: value for key, value in event.items() if key != "kind"}
            recorder.event(kind, **payload)
        recorder.record_solver(
            {
                "schema_version": SOLVER_TRACE_SCHEMA_VERSION,
                "certificate_mode": mode,
                "certificates": serialize_certificates(certificate_store, mode),
                "counters": solver_trace_counters(events),
            }
        )

    def _compiler_ltr_decode_one(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        length: int,
        *,
        mode: str,
        slot_contract: list[str] | None,
        _initial_prefix: tuple[int, ...] | None = None,
        _search_state: object | None = None,
        _trajectory_id: int = 0,
        _disable_trajectory_fork: bool = False,
    ) -> torch.Tensor:
        from slm_training.dsl.grammar.fastpath.compiler_draft import (
            build_completion_forest,
        )
        from slm_training.dsl.grammar.fastpath.lattice_search import (
            LatticeSearchState,
            StagnationTracker,
            TrajectoryCandidate,
            path_key,
            rank_forest,
            select_trajectory_candidate,
            trajectory_orders,
        )

        if mode not in {"forced", "restricted", "tree"}:
            raise ValueError(
                "compiler_decode_mode must be off, forced, restricted, or tree"
            )
        state_rows = self._new_grammar_states(1)
        state = state_rows[0] if state_rows else make_grammar_state()
        prefix = list(_initial_prefix or (int(self.tokenizer.bos_id),))
        if _initial_prefix is not None:
            state = make_grammar_state()
            for initial_token_id in prefix[1:]:
                state.advance_token(self.tokenizer, int(initial_token_id))
        stats = get_active_stats()
        search_mode = str(
            getattr(self.config, "compiler_search_mode", "greedy") or "greedy"
        ).lower()
        if search_mode not in {"greedy", "lattice", "ptrm", "gram"}:
            raise ValueError(
                "compiler_search_mode must be greedy, lattice, ptrm, or gram"
            )
        search_trigger = str(
            getattr(self.config, "compiler_search_trigger", "stagnation")
            or "stagnation"
        ).lower()
        if search_trigger not in {"bottom", "stagnation", "always"}:
            raise ValueError(
                "compiler_search_trigger must be bottom, stagnation, or always"
            )
        search = (
            _search_state
            if isinstance(_search_state, LatticeSearchState)
            else LatticeSearchState(
                backtrack_limit=max(
                    0,
                    int(
                        getattr(self.config, "compiler_search_backtrack_limit", 8) or 0
                    ),
                )
            )
        )
        stagnation = StagnationTracker(
            patience=max(
                1,
                int(
                    getattr(self.config, "compiler_search_stagnation_patience", 2) or 2
                ),
            )
        )
        after_bottom = False
        while len(prefix) < length and prefix[-1] != self.tokenizer.eos_id:
            if stats is not None and search_mode != "greedy":
                stats.compiler_lattice_recurrences += 1
            with timed_ms(stats, "compiler_ms"):
                forest = build_completion_forest(
                    self.tokenizer,
                    prefix,
                    state=state,
                    slot_contract=slot_contract,
                    max_path_tokens=int(
                        getattr(self.config, "grammar_draft_window", 8) or 8
                    ),
                    min_content=self._effective_min_content(slot_contract),
                )
            if getattr(self.config, "verified_solver_decode", False):
                # VSS1-03: certified exact closure prunes the forest to the live
                # subset before any soft ranking. Disabled by default (guard above),
                # so the default decode path is untouched.
                forest = self._solver_prune_forest(forest, prefix)
            # Partial coverage still contains individually grammar-admitted
            # paths. Tree/restricted modes must consume those paths; falling
            # back merely because the vocabulary is not exhaustive discards
            # the deterministic symbolic constraint at exactly the branch
            # points it is meant to protect. Fall back only when no legal path
            # exists at all.
            if not forest.paths:
                if search_mode != "greedy":
                    if stats is not None:
                        stats.compiler_lattice_bottoms += 1
                    restored = search.rollback(
                        local_nogoods=bool(
                            getattr(
                                self.config,
                                "compiler_search_local_nogoods",
                                False,
                            )
                        )
                    )
                    if restored is not None:
                        prefix, selected_path = restored
                        state = make_grammar_state()
                        for stable_token_id in prefix[1:]:
                            state.advance_token(self.tokenizer, stable_token_id)
                        selected = (
                            tuple(selected_path.token_ids)
                            if selected_path is not None
                            else ()
                        )
                        if stats is not None:
                            stats.compiler_lattice_rollbacks += 1
                            stats.compiler_lattice_nogoods = len(search.nogoods)
                            stats.compiler_lattice_max_rollback_depth = max(
                                stats.compiler_lattice_max_rollback_depth,
                                search.backtracks,
                            )
                        after_bottom = True
                        if selected_path is None:
                            continue
                    else:
                        selected = ()
                    if selected:
                        # ``build_completion_forest`` resynchronizes the parser
                        # from this restored stable prefix on the next loop.
                        room = length - len(prefix)
                        selected = selected[:room]
                        for token_id in selected:
                            prefix.append(int(token_id))
                            state.advance_token(self.tokenizer, int(token_id))
                            if stats is not None:
                                stats.tokens_emitted += 1
                        continue
                    if stats is not None:
                        stats.compiler_lattice_abstentions += 1
                        if search.backtracks >= search.backtrack_limit:
                            stats.compiler_lattice_budget_exhaustions += 1
                            stats.compiler_lattice_termination_reason = (
                                "backtrack_budget_exhausted"
                            )
                        else:
                            stats.compiler_lattice_termination_reason = (
                                "no_live_decision"
                            )
                if mode == "forced" and forest.paths:
                    canvas = self._compiler_canvas(prefix, length)
                    logits = self._denoiser_forward(canvas, ctx, ctx_pad)
                    choice = pick_constrained_token(
                        logits[0, len(prefix)].clone(),
                        self.tokenizer,
                        prefix,
                        top_k=self.config.grammar_top_k,
                        slot_contract=slot_contract,
                        state=state,
                        **self._pick_kwargs(),
                    )
                    token_id = int(
                        choice if choice is not None else self.tokenizer.eos_id
                    )
                    prefix.append(token_id)
                    state.advance_token(self.tokenizer, token_id)
                    if stats is not None:
                        stats.tokens_emitted += 1
                    if token_id == self.tokenizer.eos_id:
                        break
                    continue
                if stats is not None:
                    stats.compiler_fallbacks += 1
                # Tree/restricted modes must never append an unconstrained
                # suffix after the symbolic forest reaches a dead end. The
                # prefix is grammar-admitted; return it as-is and let the
                # evaluator report incompleteness instead of manufacturing
                # illegal structure with MaskGIT.
                if mode in {"tree", "restricted"}:
                    break
                if stats is not None:
                    stats.seeded_fallbacks += 1
                before_forwards = int(self.speculative_stats.denoiser_forwards)
                text = self._generate_maskgit_one(
                    ctx,
                    ctx_pad,
                    length,
                    use_grammar=False,
                    slot_contract=slot_contract,
                    seed_ids=prefix,
                )
                encoded = self._encode_openui(
                    text, placeholders=list(slot_contract or [])
                )[:length]
                fallback_forwards = max(
                    0,
                    int(self.speculative_stats.denoiser_forwards) - before_forwards,
                )
                if stats is not None:
                    stats.forwards_count += fallback_forwards
                    stats.full_projections += fallback_forwards
                    stats.canvas_tokens += fallback_forwards * int(length)
                    stats.tokens_emitted += max(0, len(encoded) - 1)
                prefix = [int(x) for x in encoded]
                break

            if mode == "forced" and len(forest.paths) > 1:
                # Forced-only mode preserves the legacy full-vocabulary choice
                # at semantic branch points while still collapsing unique spans.
                canvas = self._compiler_canvas(prefix, length)
                logits = self._denoiser_forward(canvas, ctx, ctx_pad)
                row = logits[0, len(prefix)].clone()
                choice = pick_constrained_token(
                    row,
                    self.tokenizer,
                    prefix,
                    top_k=self.config.grammar_top_k,
                    slot_contract=slot_contract,
                    state=state,
                    **self._pick_kwargs(),
                )
                selected = (
                    int(choice if choice is not None else self.tokenizer.eos_id),
                )
            else:
                selected = self._select_compiler_path(
                    prefix,
                    forest.paths,
                    ctx,
                    ctx_pad,
                    length,
                    tree=mode == "tree",
                    slot_contract=slot_contract,
                )
            if search_mode != "greedy":
                hard_signature = rank_forest(forest).signature
                ranked = rank_forest(
                    forest,
                    {tuple(selected): 1.0},
                    prefix=tuple(prefix),
                    nogoods=frozenset(search.nogoods),
                )
                is_stagnant = stagnation.observe(hard_signature, len(prefix))
                if is_stagnant and stats is not None:
                    stats.compiler_lattice_stagnation_triggers += 1
                trigger_trajectory = search_mode in {"ptrm", "gram"} and (
                    _disable_trajectory_fork
                    or search_trigger == "always"
                    or (search_trigger == "bottom" and after_bottom)
                    or (search_trigger == "stagnation" and is_stagnant)
                )
                if trigger_trajectory and len(ranked.paths) > 1:
                    width = max(
                        1,
                        int(getattr(self.config, "compiler_search_width", 1) or 1),
                    )
                    noise = max(
                        0.0,
                        float(
                            getattr(self.config, "compiler_search_noise", 0.0) or 0.0
                        ),
                    )
                    orders = trajectory_orders(
                        ranked,
                        width=1 if _disable_trajectory_fork else width,
                        noise=noise,
                        seed=(
                            int(getattr(self.config, "seed", 0) or 0)
                            + 1009 * int(_trajectory_id)
                            + len(prefix)
                        ),
                    )
                    if orders:
                        order = orders[0]
                        # RankedForest scores are a function of path_key, so a
                        # keyed lookup replaces the O(n^2) paths.index() scans.
                        score_by_key = {
                            path_key(ranked_path): ranked_score
                            for ranked_path, ranked_score in zip(
                                ranked.paths, ranked.scores, strict=True
                            )
                        }
                        if not _disable_trajectory_fork:
                            from slm_training.dsl.grammar.backends.ast_utils import (
                                ast_fingerprint,
                            )
                            from slm_training.dsl.parser import validate
                            from slm_training.dsl.placeholders import (
                                extract_placeholders,
                            )

                            candidates: list[TrajectoryCandidate[torch.Tensor]] = []
                            for trajectory_id in range(width):
                                branch_order = orders[trajectory_id % len(orders)]
                                first = branch_order[0]
                                branch_search = search.clone()
                                branch_search.choose(
                                    prefix,
                                    type(ranked)(
                                        branch_order,
                                        tuple(
                                            score_by_key[path_key(path)]
                                            for path in branch_order
                                        ),
                                        ranked.coverage,
                                    ),
                                )
                                branch_prefix = tuple(
                                    [
                                        *prefix,
                                        *first.token_ids[
                                            : max(0, length - len(prefix))
                                        ],
                                    ]
                                )
                                branch = self._compiler_ltr_decode_one(
                                    ctx,
                                    ctx_pad,
                                    length,
                                    mode=mode,
                                    slot_contract=slot_contract,
                                    _initial_prefix=branch_prefix,
                                    _search_state=branch_search,
                                    _trajectory_id=trajectory_id + 1,
                                    _disable_trajectory_fork=True,
                                )
                                text = self._decode_ids(branch)
                                canonical = self._canonical_valid_openui(
                                    self._repair_surface_syntax(text)
                                )
                                if stats is not None:
                                    stats.compiler_lattice_verifier_calls += 1
                                fingerprint = ""
                                if canonical is not None:
                                    try:
                                        fingerprint = ast_fingerprint(
                                            validate(canonical).root
                                        )
                                    except Exception:  # noqa: BLE001
                                        canonical = None
                                placeholders = set(
                                    extract_placeholders(canonical or text)
                                )
                                allowed_slots = set(slot_contract or ())
                                contract_ok = (
                                    not slot_contract or placeholders <= allowed_slots
                                )
                                candidates.append(
                                    TrajectoryCandidate(
                                        value=branch,
                                        valid=canonical is not None,
                                        contract_satisfied=contract_ok,
                                        model_score=float(
                                            score_by_key[path_key(first)]
                                        ),
                                        simplicity=sum(
                                            int(token_id)
                                            not in {
                                                int(self.tokenizer.pad_id),
                                                int(self.tokenizer.eos_id),
                                            }
                                            for token_id in branch.tolist()
                                        ),
                                        fingerprint=fingerprint,
                                    )
                                )
                            selected_candidate, unique_valid = (
                                select_trajectory_candidate(
                                    tuple(candidates),
                                    semantic_dedup=search_mode == "gram",
                                )
                            )
                            if stats is not None:
                                valid_count = sum(row.valid for row in candidates)
                                stats.compiler_lattice_trajectory_triggers += 1
                                stats.compiler_lattice_trajectories += len(candidates)
                                stats.compiler_lattice_valid_trajectories += valid_count
                                stats.compiler_lattice_unique_valid_asts += unique_valid
                                stats.compiler_lattice_unique_proposals += len(
                                    {tuple(row.value.tolist()) for row in candidates}
                                )
                                if selected_candidate is not None:
                                    stats.compiler_lattice_invalid_selected_over_valid += int(
                                        not selected_candidate.valid and valid_count > 0
                                    )
                                    stats.compiler_lattice_selector_regret += (
                                        max(row.model_score for row in candidates)
                                        - selected_candidate.model_score
                                    )
                                    stats.compiler_lattice_termination_reason = (
                                        "trajectory_valid"
                                        if selected_candidate.valid
                                        else "trajectory_abstain"
                                    )
                            if selected_candidate is not None:
                                return selected_candidate.value
                        ranked = type(ranked)(
                            order,
                            tuple(score_by_key[path_key(path)] for path in order),
                            ranked.coverage,
                        )
                        if stats is not None:
                            stats.compiler_lattice_trajectory_triggers += 1
                            if search_trigger == "always":
                                stats.compiler_lattice_always_triggers += 1
                            elif search_trigger == "bottom":
                                stats.compiler_lattice_bottom_triggers += 1
                            stats.compiler_lattice_trajectories += len(orders)
                    after_bottom = False
                decision = search.choose(prefix, ranked)
                selected = tuple(decision.token_ids) if decision else ()
                if stats is not None:
                    stats.compiler_lattice_states += 1
                    stats.compiler_lattice_candidates += len(ranked.paths)
                    stats.compiler_lattice_nogood_hits += len(forest.paths) - len(
                        ranked.paths
                    )
                    stats.compiler_lattice_last_signature = ranked.signature
                    stats.compiler_lattice_nogoods = len(search.nogoods)
            room = length - len(prefix)
            if room <= int(getattr(self.config, "grammar_draft_window", 8) or 8):
                eos_path = next(
                    (
                        path
                        for path in forest.paths
                        if path.token_ids == (int(self.tokenizer.eos_id),)
                    ),
                    None,
                )
                if eos_path is not None:
                    selected = eos_path.token_ids
            selected = selected[:room]
            if not selected:
                break
            forced_span = len(forest.paths) == 1 or len(selected) > 1
            if forced_span and stats is not None:
                stats.forced_spans += 1
                stats.forced_tokens += len(selected)
            for token_id in selected:
                prefix.append(int(token_id))
                state.advance_token(self.tokenizer, int(token_id))
                if stats is not None:
                    stats.tokens_emitted += 1
                if token_id == self.tokenizer.eos_id:
                    break
            if prefix[-1] == self.tokenizer.eos_id:
                if stats is not None and search_mode != "greedy":
                    stats.compiler_lattice_termination_reason = "solution"
                break

        if (
            stats is not None
            and search_mode != "greedy"
            and not stats.compiler_lattice_termination_reason
        ):
            stats.compiler_lattice_abstentions += 1
            stats.compiler_lattice_termination_reason = "length_or_empty_selection"

        result = torch.full(
            (length,),
            self.tokenizer.pad_id,
            dtype=torch.long,
            device=self.device_name,
        )
        used = min(length, len(prefix))
        result[:used] = torch.as_tensor(
            prefix[:used], dtype=torch.long, device=self.device_name
        )
        return result

    def _compiler_ltr_decode_batch(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        length: int,
        *,
        mode: str,
    ) -> torch.Tensor:
        rows = []
        for i in range(int(ctx.size(0))):
            contract = (
                self._slot_contracts[i]
                if self._slot_contracts and i < len(self._slot_contracts)
                else None
            )
            if not getattr(self.config, "slot_contract_constrained_decode", False):
                contract = None
            rows.append(
                self._compiler_ltr_decode_one(
                    ctx[i : i + 1],
                    ctx_pad[i : i + 1],
                    length,
                    mode=mode,
                    slot_contract=contract,
                )
            )
        return torch.stack(rows)

    def _greedy_ltr_decode(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        """Left-to-right argmax decode (batch size 1 wrapper)."""
        return self._greedy_ltr_decode_batch(ctx, ctx_pad, length)

    def _choice_ltr_decode_batch(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        length: int,
        contracts: list[list[str] | None],
    ) -> torch.Tensor:
        """Decode choice streams through their production-codec pushdown state."""
        from slm_training.models.choice_tokenizer import ChoiceDecodeState

        bsz = int(ctx.size(0))
        tok = self.tokenizer
        ids = torch.full(
            (bsz, length),
            tok.mask_id,
            dtype=torch.long,
            device=self.device_name,
        )
        ids[:, 0] = tok.bos_id
        states = [
            ChoiceDecodeState(tok, slot_count=len(contract or ()))
            for contract in contracts
        ]
        active = torch.ones(bsz, dtype=torch.bool, device=self.device_name)
        stats = get_active_stats()

        for position in range(1, length):
            rows = active.nonzero(as_tuple=False).flatten().tolist()
            if not rows:
                break
            cache_hits = tok.allowed_cache_hits
            cache_misses = tok.allowed_cache_misses
            candidates_considered = tok.candidates_considered
            vocab_candidates_avoided = tok.vocab_candidates_avoided
            completion_cache_hits = tok.completion_cache_hits
            completion_cache_misses = tok.completion_cache_misses
            allowed = {row: states[row].allowed_ids(length - position) for row in rows}
            if stats is not None:
                stats.choice_state_cache_hits += tok.allowed_cache_hits - cache_hits
                stats.choice_state_cache_misses += (
                    tok.allowed_cache_misses - cache_misses
                )
                stats.choice_candidates_considered += (
                    tok.candidates_considered - candidates_considered
                )
                stats.choice_vocab_candidates_avoided += (
                    tok.vocab_candidates_avoided - vocab_candidates_avoided
                )
                stats.choice_completion_cache_hits += (
                    tok.completion_cache_hits - completion_cache_hits
                )
                stats.choice_completion_cache_misses += (
                    tok.completion_cache_misses - completion_cache_misses
                )
            need_model = [row for row in rows if len(allowed[row]) > 1]
            logits = self._denoiser_forward(ids, ctx, ctx_pad) if need_model else None
            for row in rows:
                legal = allowed[row]
                if not legal:
                    active[row] = False
                    continue
                if len(legal) == 1:
                    choice = next(iter(legal))
                    if stats is not None:
                        stats.forced_tokens += 1
                else:
                    assert logits is not None
                    candidate_ids = tuple(sorted(legal))
                    legal_ids = torch.tensor(
                        candidate_ids, dtype=torch.long, device=logits.device
                    )
                    scores = logits[row, position].index_select(0, legal_ids)
                    inventory_bias = self._component_inventory_bias(
                        ctx[row : row + 1],
                        ctx_pad[row : row + 1],
                        candidate_ids,
                    )
                    if inventory_bias is not None:
                        scores = scores + inventory_bias
                    candidate_kinds = tuple(
                        (
                            "component_root"
                            if states[row].current_marker == "r="
                            else "component_bound"
                            if states[row].mode in {"v05", "structural"}
                            else "component_root_or_bound"
                        )
                        if tok.kind_of(token_id) == "component"
                        and not states[row].frames
                        else tok.kind_of(token_id)
                        for token_id in candidate_ids
                    )
                    plan_bias = self._component_plan_bias(
                        ctx[row : row + 1],
                        ctx_pad[row : row + 1],
                        ids[row, :position].tolist(),
                        candidate_ids,
                        candidate_kinds,
                    )
                    if plan_bias is not None:
                        before_plan = int(scores.argmax().item())
                        scores = scores + plan_bias
                        if stats is not None:
                            stats.component_plan_applications += 1
                            stats.component_plan_choice_changes += int(
                                int(scores.argmax().item()) != before_plan
                            )
                    slot_bias = self._slot_component_bias(
                        ctx[row : row + 1],
                        ctx_pad[row : row + 1],
                        ids[row, :position].tolist(),
                        candidate_ids,
                        candidate_kinds,
                        (
                            self._slot_contracts[row]
                            if self._slot_contracts and row < len(self._slot_contracts)
                            else None
                        ),
                        (
                            self._semantic_role_candidates[row]
                            if self._semantic_role_candidates
                            and row < len(self._semantic_role_candidates)
                            else None
                        ),
                    )
                    if slot_bias is not None:
                        before_slot = int(scores.argmax().item())
                        scores = scores + slot_bias
                        if stats is not None:
                            stats.slot_component_applications += 1
                            stats.slot_component_choice_changes += int(
                                int(scores.argmax().item()) != before_slot
                            )
                    schema_value_bias = self._schema_value_bias(
                        states[row], candidate_ids, scores
                    )
                    if schema_value_bias is not None:
                        scores = scores + schema_value_bias
                    schema_enum_close_bias = self._schema_enum_close_bias(
                        states[row], candidate_ids, scores
                    )
                    if schema_enum_close_bias is not None:
                        scores = scores + schema_enum_close_bias
                    schema_opaque_bias = self._schema_opaque_bias(
                        states[row], candidate_ids, scores
                    )
                    if schema_opaque_bias is not None:
                        scores = scores + schema_opaque_bias
                    schema_opaque_close_bias = self._schema_opaque_close_bias(
                        states[row], candidate_ids, scores
                    )
                    if schema_opaque_close_bias is not None:
                        scores = scores + schema_opaque_close_bias
                    schema_role_slot_bias = self._schema_role_slot_bias(
                        states[row],
                        candidate_ids,
                        scores,
                        (
                            self._slot_contracts[row]
                            if self._slot_contracts and row < len(self._slot_contracts)
                            else None
                        ),
                        (
                            self._semantic_role_candidates[row]
                            if self._semantic_role_candidates
                            and row < len(self._semantic_role_candidates)
                            else None
                        ),
                    )
                    if schema_role_slot_bias is not None:
                        scores = scores + schema_role_slot_bias
                    slot_coverage_close_bias = self._slot_coverage_close_bias(
                        states[row],
                        ids[row, :position].tolist(),
                        candidate_ids,
                        scores,
                        (
                            self._slot_contracts[row]
                            if self._slot_contracts and row < len(self._slot_contracts)
                            else None
                        ),
                        (
                            self._semantic_role_candidates[row]
                            if self._semantic_role_candidates
                            and row < len(self._semantic_role_candidates)
                            else None
                        ),
                    )
                    slot_coverage_close_trace = None
                    if slot_coverage_close_bias is not None:
                        scores_before_coverage_close = scores.clone()
                        before_coverage_close = int(scores.argmax().item())
                        scores = scores + slot_coverage_close_bias
                        if stats is not None:
                            stats.slot_coverage_close_applications += 1
                            stats.slot_coverage_close_choice_changes += int(
                                int(scores.argmax().item())
                                != before_coverage_close
                            )
                            slot_coverage_close_trace = (
                                self._record_slot_coverage_close_trace(
                                    stats,
                                    row=row,
                                    position=position,
                                    state=states[row],
                                    prefix=ids[row, :position].tolist(),
                                    candidate_ids=candidate_ids,
                                    scores_before=scores_before_coverage_close,
                                    coverage_bias=slot_coverage_close_bias,
                                    scores_after=scores,
                                    slot_contract=(
                                        self._slot_contracts[row]
                                        if self._slot_contracts
                                        and row < len(self._slot_contracts)
                                        else None
                                    ),
                                )
                            )
                    repeated_array_close_bias = (
                        self._semantic_plan_repeated_array_close_bias(
                            row,
                            states[row],
                            candidate_ids,
                            scores,
                        )
                    )
                    if repeated_array_close_bias is not None:
                        scores = scores + repeated_array_close_bias
                    typed_array_nonempty_bias = (
                        self._semantic_plan_typed_array_nonempty_bias(
                            row,
                            states[row],
                            ids[row, :position].tolist(),
                            candidate_ids,
                            scores,
                        )
                    )
                    if typed_array_nonempty_bias is not None:
                        scores = scores + typed_array_nonempty_bias
                    semantic_plan_bias = self._semantic_plan_bias(
                        row,
                        candidate_ids,
                        candidate_kinds,
                        states[row],
                        ids[row, :position].tolist(),
                        scores,
                    )
                    semantic_plan_trace = None
                    if semantic_plan_bias is not None:
                        scores_before_semantic_plan = scores.clone()
                        before_semantic_plan = int(scores.argmax().item())
                        scores = scores + semantic_plan_bias
                        if stats is not None:
                            stats.semantic_plan_applications += 1
                            stats.semantic_plan_choice_changes += int(
                                int(scores.argmax().item()) != before_semantic_plan
                            )
                            semantic_plan_trace = self._record_semantic_plan_seed_trace(
                                stats,
                                row=row,
                                position=position,
                                state=states[row],
                                candidate_ids=candidate_ids,
                                candidate_kinds=candidate_kinds,
                                scores_before=scores_before_semantic_plan,
                                plan_bias=semantic_plan_bias,
                                scores_after=scores,
                            )
                            if semantic_plan_trace is None:
                                semantic_plan_trace = (
                                    self._record_semantic_plan_missing_family_trace(
                                        stats,
                                        row=row,
                                        position=position,
                                        state=states[row],
                                        candidate_ids=candidate_ids,
                                        candidate_kinds=candidate_kinds,
                                        scores_before=scores_before_semantic_plan,
                                        plan_bias=semantic_plan_bias,
                                        scores_after=scores,
                                    )
                                )
                    semantic_plan_inline_bias = self._semantic_plan_inline_bias(
                        row,
                        ids[row, :position].tolist(),
                        candidate_ids,
                        candidate_kinds,
                    )
                    if semantic_plan_inline_bias is not None:
                        scores = scores + semantic_plan_inline_bias
                    semantic_plan_root_bias = self._semantic_plan_root_bias(
                        row,
                        states[row],
                        ids[row, :position].tolist(),
                        candidate_ids,
                        scores,
                    )
                    semantic_plan_root_trace = None
                    if semantic_plan_root_bias is not None:
                        scores_before_plan_root = scores.clone()
                        before_plan_root = int(scores.argmax().item())
                        scores = scores + semantic_plan_root_bias
                        if stats is not None:
                            stats.semantic_plan_root_applications += 1
                            stats.semantic_plan_root_choice_changes += int(
                                int(scores.argmax().item()) != before_plan_root
                            )
                            semantic_plan_root_trace = (
                                self._record_semantic_plan_root_trace(
                                    stats,
                                    row=row,
                                    position=position,
                                    state=states[row],
                                    candidate_ids=candidate_ids,
                                    scores_before=scores_before_plan_root,
                                    root_bias=semantic_plan_root_bias,
                                    scores_after=scores,
                                )
                            )
                    repeated_slot_bias = self._semantic_plan_repeated_slot_bias(
                        row,
                        states[row],
                        ids[row, :position].tolist(),
                        candidate_ids,
                        scores,
                    )
                    if repeated_slot_bias is not None:
                        scores = scores + repeated_slot_bias
                    precontent_literal_bias = self._schema_precontent_literal_bias(
                        states[row], candidate_ids, scores
                    )
                    if precontent_literal_bias is not None:
                        scores = scores + precontent_literal_bias
                    root_arity_bias = self._root_reference_arity_bias(
                        ctx[row : row + 1],
                        ctx_pad[row : row + 1],
                        states[row],
                        candidate_ids,
                    )
                    if root_arity_bias is not None:
                        before_root_arity = int(scores.argmax().item())
                        scores = scores + root_arity_bias
                        after_root_arity = int(scores.argmax().item())
                        if stats is not None:
                            stats.root_reference_arity_applications += 1
                            stats.root_reference_arity_choice_changes += int(
                                after_root_arity != before_root_arity
                            )
                            if len(stats.constrained_selection_traces) < 64:
                                stats.constrained_selection_traces.append(
                                    {
                                        "phase": "root_reference_arity",
                                        "position": position,
                                        "emitted_references": int(
                                            states[row].frames[-1].reference_count
                                        ),
                                        "before_token": str(
                                            tok.id_to_token.get(
                                                candidate_ids[before_root_arity], ""
                                            )
                                        ),
                                        "chosen_token": str(
                                            tok.id_to_token.get(
                                                candidate_ids[after_root_arity], ""
                                            )
                                        ),
                                        "choice_changed": (
                                            after_root_arity != before_root_arity
                                        ),
                                        **self._choice_phase_evidence(states[row]),
                                    }
                                )
                    root_identity_bias = self._root_reference_identity_bias(
                        ctx[row : row + 1],
                        ctx_pad[row : row + 1],
                        states[row],
                        ids[row, :position].tolist(),
                        candidate_ids,
                        scores,
                    )
                    if root_identity_bias is not None:
                        before_root_identity = int(scores.argmax().item())
                        scores = scores + root_identity_bias
                        after_root_identity = int(scores.argmax().item())
                        if stats is not None:
                            stats.root_reference_identity_applications += 1
                            stats.root_reference_identity_choice_changes += int(
                                after_root_identity != before_root_identity
                            )
                            if len(stats.constrained_selection_traces) < 64:
                                stats.constrained_selection_traces.append(
                                    {
                                        "phase": "root_reference_identity",
                                        "position": position,
                                        "before_token": str(
                                            tok.id_to_token.get(
                                                candidate_ids[before_root_identity], ""
                                            )
                                        ),
                                        "chosen_token": str(
                                            tok.id_to_token.get(
                                                candidate_ids[after_root_identity], ""
                                            )
                                        ),
                                        "choice_changed": (
                                            after_root_identity != before_root_identity
                                        ),
                                        **self._choice_phase_evidence(states[row]),
                                    }
                                )
                    # Learned identity permutes the reference score group, so apply
                    # predicted plan evidence afterward to preserve both factors.
                    semantic_plan_binding_bias = self._semantic_plan_binding_bias(
                        row,
                        states[row],
                        ids[row, :position].tolist(),
                        candidate_ids,
                    )
                    if semantic_plan_binding_bias is not None:
                        before_plan_binding = int(scores.argmax().item())
                        scores = scores + semantic_plan_binding_bias
                        if stats is not None:
                            stats.semantic_plan_binding_applications += 1
                            stats.semantic_plan_binding_choice_changes += int(
                                int(scores.argmax().item()) != before_plan_binding
                            )
                    reference_bias = self._visible_reference_completeness_bias(
                        states[row],
                        ids[row, :position].tolist(),
                        candidate_ids,
                    )
                    if reference_bias is not None:
                        before_reference = int(scores.argmax().item())
                        scores = scores + reference_bias
                        after_reference = int(scores.argmax().item())
                        if stats is not None:
                            stats.visible_reference_applications += 1
                            stats.visible_reference_choice_changes += int(
                                after_reference != before_reference
                            )
                            if len(stats.constrained_selection_traces) < 64:
                                stats.constrained_selection_traces.append(
                                    {
                                        "phase": "visible_reference_completeness",
                                        "position": position,
                                        "before_token": str(
                                            tok.id_to_token.get(
                                                candidate_ids[before_reference], ""
                                            )
                                        ),
                                        "chosen_token": str(
                                            tok.id_to_token.get(
                                                candidate_ids[after_reference], ""
                                            )
                                        ),
                                        "choice_changed": (
                                            after_reference != before_reference
                                        ),
                                        "legal_references": sorted(
                                            str(tok.id_to_token.get(token_id, ""))
                                            for token_id in candidate_ids
                                            if str(
                                                tok.id_to_token.get(token_id, "")
                                            ).startswith("&")
                                        ),
                                        **self._choice_phase_evidence(states[row]),
                                    }
                                )
                    self._finalize_semantic_plan_trace(
                        slot_coverage_close_trace,
                        candidate_ids=candidate_ids,
                        scores=scores,
                    )
                    self._finalize_semantic_plan_trace(
                        semantic_plan_trace,
                        candidate_ids=candidate_ids,
                        scores=scores,
                    )
                    self._finalize_semantic_plan_trace(
                        semantic_plan_root_trace,
                        candidate_ids=candidate_ids,
                        scores=scores,
                    )
                    best = scores.argmax()
                    choice = int(legal_ids[int(best)].item())
                ids[row, position] = choice
                if stats is not None:
                    stats.tokens_emitted += 1
                    stats.constrained_last_legal_candidates = len(legal)
                if choice == tok.eos_id:
                    active[row] = False
                    if position + 1 < length:
                        ids[row, position + 1 :] = tok.pad_id
                else:
                    assert states[row].advance_id(choice)
        return ids

    def _greedy_ltr_decode_batch(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        """
        Batched LTR argmax decode with progressive canvases.

        Finished sequences (EOS) are padded and skipped; remaining rows share
        transformer forwards — large win for eval throughput.

        P1: per-row GrammarDecodeState reuses DFA + prefix text.
        P3: multi-token accept from a single forward's logits.
        P4: optional prefix+K mask lookahead canvas truncation.
        """
        compiler_mode = str(
            getattr(self.config, "compiler_decode_mode", "off") or "off"
        ).lower()
        if compiler_mode != "off":
            return self._compiler_ltr_decode_batch(
                ctx, ctx_pad, length, mode=compiler_mode
            )
        bsz = int(ctx.size(0))
        device = self.device_name
        tok = self.tokenizer
        bias = self._effective_structural_bias()
        canvases = self._ltr_canvases(length)
        states = self._new_grammar_states(bsz)
        pick_kw = self._pick_kwargs()
        multitoken = bool(getattr(self.config, "grammar_multitoken_accept", False))
        multitoken_max = max(
            1, int(getattr(self.config, "grammar_multitoken_max", 8) or 8)
        )
        lookahead = int(getattr(self.config, "grammar_canvas_lookahead", 0) or 0)
        stats = get_active_stats()

        ids: torch.Tensor | None = None
        active = torch.ones(bsz, dtype=torch.bool, device=device)
        start_t = 1

        for canvas in canvases:
            if ids is None:
                ids = torch.full(
                    (bsz, canvas),
                    tok.mask_id,
                    dtype=torch.long,
                    device=device,
                )
                ids[:, 0] = tok.bos_id
            else:
                extra = canvas - ids.size(1)
                if extra > 0:
                    pad = torch.full(
                        (bsz, extra),
                        tok.mask_id,
                        dtype=torch.long,
                        device=device,
                    )
                    # Finished sequences should stay padded, not re-masked.
                    if (~active).any():
                        pad = pad.clone()
                        pad[~active] = tok.pad_id
                    ids = torch.cat([ids, pad], dim=1)

            use_fast = bool(getattr(self.config, "grammar_fastpath", True))
            t = start_t
            while t < canvas:
                if not bool(active.any()):
                    break
                active_idx = active.nonzero(as_tuple=False).flatten()
                forced_map: dict[int, int] = {}
                if use_fast:
                    for bi in active_idx.tolist():
                        st = states[bi] if states is not None else None
                        forced = force_emit_token_id(
                            tok, ids[bi, :t].tolist(), state=st
                        )
                        if forced is not None:
                            forced_map[bi] = forced
                need_model = active_idx.tolist()
                pred = ids[:, t].clone()
                # Optional P3: stash full-row logits for multi-token accept.
                row_logits_full: torch.Tensor | None = None

                if need_model:
                    need_t = torch.tensor(need_model, device=device, dtype=torch.long)
                    # P4: truncate canvas to prefix + lookahead for the forward.
                    if lookahead > 0:
                        end = min(canvas, t + lookahead)
                        fwd_ids = ids.index_select(0, need_t)[:, :end]
                        sub_ctx = ctx.index_select(0, need_t)
                        sub_pad = ctx_pad.index_select(0, need_t)
                        logits = self._denoiser_forward(fwd_ids, sub_ctx, sub_pad)
                        if bias:
                            logits = apply_structural_bias(logits, tok, bias=bias)
                        row = torch.full(
                            (bsz, logits.size(-1)),
                            -1e9,
                            device=device,
                            dtype=logits.dtype,
                        )
                        # Position t maps to local index t (ids already start at 0).
                        local_t = min(t, end - 1)
                        row.index_copy_(0, need_t, logits[:, local_t, :])
                        if multitoken:
                            row_logits_full = torch.full(
                                (bsz, end, logits.size(-1)),
                                -1e9,
                                device=device,
                                dtype=logits.dtype,
                            )
                            row_logits_full.index_copy_(0, need_t, logits)
                    elif need_t.numel() == bsz:
                        logits = self._denoiser_forward(ids, ctx, ctx_pad)
                        if bias:
                            logits = apply_structural_bias(logits, tok, bias=bias)
                        row = logits[:, t, :].clone()
                        if multitoken:
                            row_logits_full = logits
                    else:
                        sub_ids = ids.index_select(0, need_t)
                        sub_ctx = ctx.index_select(0, need_t)
                        sub_pad = ctx_pad.index_select(0, need_t)
                        logits = self._denoiser_forward(sub_ids, sub_ctx, sub_pad)
                        if bias:
                            logits = apply_structural_bias(logits, tok, bias=bias)
                        row = torch.full(
                            (bsz, logits.size(-1)),
                            -1e9,
                            device=device,
                            dtype=logits.dtype,
                        )
                        row.index_copy_(0, need_t, logits[:, t, :])
                        if multitoken:
                            row_logits_full = torch.full(
                                (bsz, logits.size(1), logits.size(-1)),
                                -1e9,
                                device=device,
                                dtype=logits.dtype,
                            )
                            row_logits_full.index_copy_(0, need_t, logits)
                    row = row.clone()
                    row[:, tok.mask_id] = -1e9
                    row[:, tok.pad_id] = -1e9
                    for bi in need_model:
                        contract = (
                            self._slot_contracts[bi]
                            if self._slot_contracts and bi < len(self._slot_contracts)
                            else None
                        )
                        if not getattr(
                            self.config, "slot_contract_constrained_decode", False
                        ):
                            contract = None
                        st = states[bi] if states is not None else None
                        choice = pick_constrained_token(
                            row[bi],
                            tok,
                            ids[bi, :t].tolist(),
                            top_k=self.config.grammar_top_k,
                            forced_token_id=forced_map.get(bi),
                            slot_contract=contract,
                            state=st,
                            **pick_kw,
                        )
                        if choice is None:
                            # No legal token — end sequence rather than emit garbage.
                            pred[bi] = tok.eos_id
                        else:
                            pred[bi] = choice
                            if st is not None:
                                st.advance_token(tok, int(choice))
                        if stats is not None:
                            stats.tokens_emitted += 1

                ids[:, t] = torch.where(active, pred, ids[:, t])
                hit_eos = active & pred.eq(tok.eos_id)
                if bool(hit_eos.any()) and t + 1 < canvas:
                    for b in hit_eos.nonzero(as_tuple=False).flatten().tolist():
                        ids[b, t + 1 :] = tok.pad_id
                active = active & ~pred.eq(tok.eos_id)

                # P3: greedily accept consecutive legal argmax tokens from the
                # same forward without re-running the denoiser.
                advance = 1
                if multitoken and row_logits_full is not None and bool(active.any()):
                    max_run = min(multitoken_max, canvas - t - 1)
                    for step in range(1, max_run + 1):
                        pos = t + step
                        if pos >= row_logits_full.size(1):
                            break
                        step_pred = ids[:, pos].clone()
                        any_accept = False
                        for bi in active.nonzero(as_tuple=False).flatten().tolist():
                            logits_pos = row_logits_full[bi, pos].clone()
                            logits_pos[tok.mask_id] = -1e9
                            logits_pos[tok.pad_id] = -1e9
                            contract = (
                                self._slot_contracts[bi]
                                if self._slot_contracts
                                and bi < len(self._slot_contracts)
                                else None
                            )
                            if not getattr(
                                self.config, "slot_contract_constrained_decode", False
                            ):
                                contract = None
                            st = states[bi] if states is not None else None
                            choice = pick_constrained_token(
                                logits_pos,
                                tok,
                                ids[bi, :pos].tolist(),
                                top_k=self.config.grammar_top_k,
                                slot_contract=contract,
                                state=st,
                                **pick_kw,
                            )
                            if choice is None:
                                step_pred[bi] = tok.eos_id
                            else:
                                step_pred[bi] = choice
                                any_accept = True
                                if st is not None:
                                    st.advance_token(tok, int(choice))
                                if stats is not None:
                                    stats.tokens_emitted += 1
                                    stats.accepted_run_tokens += 1
                        if not any_accept:
                            break
                        ids[:, pos] = torch.where(active, step_pred, ids[:, pos])
                        hit_eos = active & step_pred.eq(tok.eos_id)
                        if bool(hit_eos.any()) and pos + 1 < canvas:
                            for b in hit_eos.nonzero(as_tuple=False).flatten().tolist():
                                ids[b, pos + 1 :] = tok.pad_id
                        active = active & ~step_pred.eq(tok.eos_id)
                        advance = step + 1
                        if not bool(active.any()):
                            break

                # E30: suffix-rollback — revisable window behind LTR frontier.
                window = int(getattr(self.config, "suffix_rollback_window", 0) or 0)
                frontier = t + advance - 1
                if (
                    window > 0
                    and frontier >= 2
                    and (frontier % max(2, window // 2) == 0)
                ):
                    for bi in active.nonzero(as_tuple=False).flatten().tolist():
                        prefix_text = self._decode_ids(ids[bi, : frontier + 1])
                        try:
                            status = stream_check(prefix_text)
                            hard = bool(status.hard_error)
                        except Exception:  # noqa: BLE001
                            hard = True
                        ent_spike = False
                        if need_model:
                            try:
                                probs_t = F.softmax(row[bi], dim=-1)
                                ent = float(
                                    (-(probs_t * (probs_t + 1e-9).log()).sum()).item()
                                )
                                ent_spike = (
                                    ent > math.log(max(2, tok.vocab_size)) * 0.55
                                )
                            except Exception:  # noqa: BLE001
                                ent_spike = False
                        if not (hard or ent_spike):
                            continue
                        start = max(1, frontier - window + 1)
                        ids[bi, start : frontier + 1] = tok.mask_id
                        if states is not None:
                            fresh = self._new_grammar_states(1)
                            states[bi] = (
                                fresh[0]
                                if fresh is not None
                                else make_grammar_state(
                                    verify_chosen_only=bool(
                                        getattr(
                                            self.config,
                                            "grammar_verify_chosen_only",
                                            False,
                                        )
                                    ),
                                    skip_exact_stream_probe=bool(
                                        getattr(
                                            self.config,
                                            "grammar_skip_exact_stream_probe",
                                            True,
                                        )
                                    ),
                                    use_copy_probes=bool(
                                        getattr(
                                            self.config, "grammar_copy_probes", True
                                        )
                                    ),
                                    early_exit_pick=bool(
                                        getattr(
                                            self.config, "grammar_early_exit_pick", True
                                        )
                                    ),
                                )
                            )
                            states[bi].sync_ids(tok, ids[bi, :start].tolist())
                        for rt in range(start, frontier + 1):
                            st = states[bi] if states is not None else None
                            forced = (
                                force_emit_token_id(
                                    tok, ids[bi, :rt].tolist(), state=st
                                )
                                if use_fast
                                else None
                            )
                            logits_r = self._denoiser_forward(
                                ids[bi : bi + 1],
                                ctx[bi : bi + 1],
                                ctx_pad[bi : bi + 1],
                            )
                            if bias:
                                logits_r = apply_structural_bias(
                                    logits_r, tok, bias=bias
                                )
                            contract = (
                                self._slot_contracts[bi]
                                if self._slot_contracts
                                and bi < len(self._slot_contracts)
                                else None
                            )
                            if not getattr(
                                self.config, "slot_contract_constrained_decode", False
                            ):
                                contract = None
                            choice = pick_constrained_token(
                                logits_r[0, rt],
                                tok,
                                ids[bi, :rt].tolist(),
                                top_k=self.config.grammar_top_k,
                                forced_token_id=forced,
                                slot_contract=contract,
                                state=st,
                                **pick_kw,
                            )
                            if choice is None:
                                ids[bi, rt] = tok.eos_id
                                if rt + 1 < canvas:
                                    ids[bi, rt + 1 :] = tok.pad_id
                                active[bi] = False
                                break
                            ids[bi, rt] = choice
                            if st is not None:
                                st.advance_token(tok, int(choice))
                t += advance
            start_t = canvas
            if not bool(active.any()):
                break

        assert ids is not None
        return ids

    def _resolve_slot_contract(
        self,
        prompt: str,
        gold: ExampleRecord | None = None,
        design_md: str | None = None,
        *,
        use_gold_design: bool = True,
    ) -> list[str] | None:
        """Return inventory for decode/context.

        E35 honest mode: inventory comes from the user-visible prompt/DESIGN.md
        only (never ``gold.placeholders``). When the prompt lacks an explicit
        inventory, a keyword heuristic is used. Non-honest mode (legacy V3)
        falls back to gold placeholders for template fill / conditioning.
        """
        dm = design_md
        if dm is None and gold is not None and use_gold_design:
            dm = gold.design_md
        honest = bool(getattr(self.config, "honest_slot_contract", False))
        if honest:
            inv = inventory_from_prompt(prompt, dm, heuristic=True)
            return inv or None
        # Prefer visible inventory when present, else gold (legacy path).
        inv = inventory_from_prompt(prompt, dm, heuristic=False)
        if inv:
            return inv
        if gold is not None and gold.placeholders:
            return list(gold.placeholders)
        return inventory_from_prompt(prompt, dm, heuristic=True) or None

    def _context_prompts(
        self,
        prompts: list[str],
        golds: list[ExampleRecord | None] | None = None,
        design_mds: list[str | None] | None = None,
        *,
        slot_contracts: list[list[str] | None] | None = None,
        schemas: list[str | None] | None = None,
        output_kinds: list[str] | None = None,
        output_categories: list[str | None] | None = None,
    ) -> list[str]:
        out: list[str] = []
        use_contract = bool(getattr(self.config, "slot_contract_in_context", False))
        for i, prompt in enumerate(prompts):
            dm = design_mds[i] if design_mds else None
            gold = golds[i] if golds else None
            if gold is not None and dm is None:
                dm = gold.design_md  # type: ignore[union-attr]
            contract: list[str] | None = None
            schema = schemas[i] if schemas and i < len(schemas) else None
            if use_contract:
                honest = bool(getattr(self.config, "honest_slot_contract", False))
                if (
                    not honest
                    and slot_contracts
                    and i < len(slot_contracts)
                    and slot_contracts[i] is not None
                ):
                    contract = slot_contracts[i]
                else:
                    contract = self._resolve_slot_contract(prompt, gold, dm)
            out.append(
                self._format_one_context(
                    prompt,
                    dm,
                    query_prompt=prompt,
                    slot_contract=contract,
                    schema=schema,
                    output_kind=output_kinds[i] if output_kinds else None,
                    output_category=(
                        output_categories[i] if output_categories else None
                    ),
                )
            )
        return out

    def _repair_ltr_texts(
        self,
        texts: list[str],
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        length: int,
        *,
        slot_contracts: list[list[str] | None] | None = None,
    ) -> list[str]:
        """Re-decode rows that fail stream_check using constrained LTR."""
        from concurrent.futures import ThreadPoolExecutor

        def _check(text: str, contract: list[str] | None) -> tuple[bool, str]:
            try:
                status = stream_check(text)
                if status.serialized and status.complete_ok:
                    ser = status.serialized.strip()
                    compact = ser.replace(" ", "")
                    if "Stack([])" not in compact and "Card([])" not in compact:
                        if contract:
                            from slm_training.dsl.placeholders import (
                                extract_placeholders,
                            )

                            preds = set(extract_placeholders(ser))
                            allowed = {
                                p if p.startswith(":") else f":{p}" for p in contract
                            }
                            if preds and not preds.issubset(allowed):
                                return False, text
                        return True, ser
            except Exception:  # noqa: BLE001
                pass
            return False, text

        # Parallel Node stream_check — grammar bridge is process-bound, so threads
        # overlap Python wait when the Node CLI is the bottleneck.
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(texts)))) as pool:
            checked = list(
                pool.map(
                    lambda item: _check(item[0], item[1]),
                    [
                        (
                            text,
                            slot_contracts[i]
                            if slot_contracts and i < len(slot_contracts)
                            else None,
                        )
                        for i, text in enumerate(texts)
                    ],
                )
            )

        repaired: list[str] = []
        for i, (ok, text) in enumerate(checked):
            if ok:
                repaired.append(text)
                continue
            row_ids = torch.full(
                (1, length),
                self.tokenizer.mask_id,
                dtype=torch.long,
                device=self.device_name,
            )
            row_ids[0, 0] = self.tokenizer.bos_id
            unknown = row_ids.eq(self.tokenizer.mask_id)
            filled = self._constrained_ltr_repair(
                row_ids,
                unknown,
                ctx[i : i + 1],
                ctx_pad[i : i + 1],
                slot_contract=(
                    slot_contracts[i]
                    if slot_contracts and i < len(slot_contracts)
                    else None
                ),
            )
            repaired.append(self._decode_ids(filled[0]))
        return repaired

    def _pick_best_of_n(
        self,
        candidates: list[str],
        gold: ExampleRecord | None,
    ) -> str:
        if len(candidates) == 1:
            return candidates[0]
        from slm_training.harnesses.preference import composite_reward

        best = candidates[0]
        best_score = -1.0
        for cand in candidates:
            score = float(composite_reward(cand, gold=gold, design_md=None))
            if score > best_score:
                best_score = score
                best = cand
        return best

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: list[str],
        golds: list[ExampleRecord | None] | None = None,
        *,
        max_len: int | None = None,
        grammar_constrained: bool | None = None,
        design_mds: list[str | None] | None = None,
    ) -> list[str]:
        """Batched generate — preferred for eval throughput."""
        from slm_training.runtime.telemetry import timed

        self.eval()
        if not prompts:
            return []
        n_samples = max(1, int(getattr(self.config, "best_of_n", 1) or 1))
        if n_samples > 1:
            pools: list[list[str]] = [[] for _ in prompts]
            prev = self.config.best_of_n
            self.config.best_of_n = 1
            try:
                with timed("generate_batch"):
                    for _ in range(n_samples):
                        sample = self._generate_batch_once(
                            prompts,
                            golds,
                            max_len=max_len,
                            grammar_constrained=grammar_constrained,
                            design_mds=design_mds,
                        )
                        for i, text in enumerate(sample):
                            pools[i].append(text)
            finally:
                self.config.best_of_n = prev
            out: list[str] = []
            with timed("best_of_n_rank"):
                for i, cands in enumerate(pools):
                    gold = golds[i] if golds else None
                    out.append(self._pick_best_of_n(cands, gold))
            return out
        with timed("generate_batch"):
            return self._generate_batch_once(
                prompts,
                golds,
                max_len=max_len,
                grammar_constrained=grammar_constrained,
                design_mds=design_mds,
            )

    @torch.inference_mode()
    def generate_batch_requests(
        self,
        requests: list[GenerationRequest],
        *,
        max_len: int | None = None,
        grammar_constrained: bool | None = None,
    ) -> list[str]:
        """Generate using production-available inputs only (no gold records).

        E35 honesty: when ``honest_slot_contract`` is set, surface
        ``request.slot_contract`` into the user-visible prompt via
        ``ensure_prompt_inventory`` (inventory-in-prompt API), then extract
        inventory from the prompt text — never a silent gold channel.
        """
        if not requests:
            return []
        self._last_generation_evidence = []
        if any(request.output_kind != "document" for request in requests) and (
            self.output_contract_version < 1
        ):
            raise ValueError(
                "checkpoint predates compact output contracts; request a document"
            )
        honest = bool(getattr(self.config, "honest_slot_contract", False))
        prompts: list[str] = []
        slot_contracts: list[list[str] | None] = []
        for r in requests:
            prompt = r.prompt
            contract = list(r.slot_contract) if r.slot_contract else None
            if honest and contract:
                prompt = ensure_prompt_inventory(prompt, contract)
            prompts.append(prompt)
            slot_contracts.append(contract)
        schemas = [r.schema for r in requests]
        design_mds = [r.design_md for r in requests]
        runtime_symbols = [list(r.effective_runtime_symbols()) for r in requests]
        requested_output_kinds = [r.output_kind for r in requests]
        output_kinds = (
            requested_output_kinds if self.output_contract_version >= 1 else None
        )
        output_categories = (
            [r.output_category for r in requests]
            if self.output_contract_version >= 1
            else None
        )
        if any(kind != "document" for kind in requested_output_kinds):
            grammar_constrained = False
        n_samples = max(1, int(getattr(self.config, "best_of_n", 1) or 1))
        if n_samples > 1:
            pools: list[list[str]] = [[] for _ in requests]
            prev = self.config.best_of_n
            self.config.best_of_n = 1
            try:
                for _ in range(n_samples):
                    sample = self._generate_batch_once(
                        prompts,
                        golds=None,
                        max_len=max_len,
                        grammar_constrained=grammar_constrained,
                        design_mds=design_mds,
                        slot_contracts=slot_contracts,
                        schemas=schemas,
                        runtime_symbols=runtime_symbols,
                        output_kinds=output_kinds,
                        output_categories=output_categories,
                    )
                    for i, text in enumerate(sample):
                        pools[i].append(text)
            finally:
                self.config.best_of_n = prev
            out: list[str] = []
            for cands in pools:
                out.append(self._pick_best_of_n(cands, None))
            # The retained candidate may not be the final sampled candidate.
            self._last_generation_evidence = []
            return out
        return self._generate_batch_once(
            prompts,
            golds=None,
            max_len=max_len,
            grammar_constrained=grammar_constrained,
            design_mds=design_mds,
            slot_contracts=slot_contracts,
            schemas=schemas,
            runtime_symbols=runtime_symbols,
            output_kinds=output_kinds,
            output_categories=output_categories,
        )

    def consume_generation_evidence(self) -> list[dict[str, object]]:
        """Return and clear evidence aligned with the last generated batch."""
        evidence = self._last_generation_evidence
        self._last_generation_evidence = []
        return evidence

    def _choice_generation_evidence(
        self,
        ids: torch.Tensor,
        contracts: list[list[str] | None],
    ) -> list[dict[str, object]]:
        """Persist the actual choice stream and legal reference decisions."""
        from slm_training.models.choice_tokenizer import ChoiceDecodeState

        rows: list[dict[str, object]] = []
        tok = self.tokenizer
        for row, contract in zip(ids.tolist(), contracts, strict=True):
            stream: list[int] = []
            for token_id in row:
                stream.append(int(token_id))
                if token_id == tok.eos_id:
                    break
            state = ChoiceDecodeState(tok, slot_count=len(contract or ()))
            reference_decisions: list[dict[str, object]] = []
            for position, token_id in enumerate(stream):
                if token_id == tok.bos_id:
                    continue
                token = str(tok.id_to_token.get(token_id, ""))
                remaining = max(1, len(stream) - position)
                legal = state.allowed_ids(remaining)
                legal_references = sorted(
                    str(tok.id_to_token[candidate])
                    for candidate in legal
                    if str(tok.id_to_token[candidate]).startswith("&")
                )
                if token.startswith("&") or legal_references:
                    reference_decisions.append(
                        {
                            "position": position,
                            "chosen": token,
                            "mode": state.mode,
                            "current_marker": state.current_marker,
                            "section_count": len(state.section_types),
                            "legal_candidate_count": len(legal),
                            "legal_references": legal_references,
                            **self._choice_phase_evidence(state),
                        }
                    )
                if token_id == tok.eos_id:
                    break
                if not state.advance_id(token_id):
                    break
            rows.append(
                {
                    "schema": "choice_decision_trace/v2",
                    "choice_token_count": len(stream),
                    "choice_tokens": [
                        str(tok.id_to_token.get(token_id, "")) for token_id in stream
                    ],
                    "reference_decisions": reference_decisions,
                }
            )
        return rows

    def _generate_batch_once(
        self,
        prompts: list[str],
        golds: list[ExampleRecord | None] | None = None,
        *,
        max_len: int | None = None,
        grammar_constrained: bool | None = None,
        design_mds: list[str | None] | None = None,
        slot_contracts: list[list[str] | None] | None = None,
        schemas: list[str | None] | None = None,
        runtime_symbols: list[list[RuntimeSymbol] | None] | None = None,
        output_kinds: list[str] | None = None,
        output_categories: list[str | None] | None = None,
    ) -> list[str]:
        use_grammar = (
            self.config.grammar_constrained
            if grammar_constrained is None
            else grammar_constrained
        )
        choice_constrained = False
        try:
            from slm_training.models.choice_tokenizer import is_choice_tokenizer

            if use_grammar and is_choice_tokenizer(self.tokenizer):
                choice_constrained = True
                # Choice ids are production decisions, not surface lexemes.
                # Their dedicated pushdown state owns legality below.
                use_grammar = False
        except Exception:  # noqa: BLE001
            pass
        length = max_len or self.gen_len or self.config.max_target_len
        length = max(8, min(int(length), self.config.max_target_len))
        use_contract_decode = bool(
            getattr(self.config, "slot_contract_constrained_decode", False)
        ) or bool(getattr(self.config, "template_fill_decode", False))
        honest = bool(getattr(self.config, "honest_slot_contract", False))
        # E35: surface inventory in the user-visible prompt when gold provides
        # slots but the prompt text does not (inventory-in-prompt API).
        if honest and golds:
            prompts = [
                ensure_prompt_inventory(
                    prompts[i],
                    list(golds[i].placeholders or []) if golds[i] is not None else None,
                )
                for i in range(len(prompts))
            ]
        if use_contract_decode:
            if not honest and slot_contracts is not None:
                self._slot_contracts = [list(c) if c else None for c in slot_contracts]
            else:
                self._slot_contracts = []
                for i, prompt in enumerate(prompts):
                    gold = golds[i] if golds else None
                    dm = design_mds[i] if design_mds else None
                    self._slot_contracts.append(
                        self._resolve_slot_contract(prompt, gold, dm)
                    )
        else:
            self._slot_contracts = None
        if not use_contract_decode:
            # E617: schema_role_slot_decode_weight, slot_coverage_close_decode_weight,
            # and the semantic_plan_typed_array_*/repeated_slot_margin decode weights
            # all gate their own bias functions on a populated `self._slot_contracts`
            # row (see `_schema_role_slot_bias`, `_slot_coverage_close_bias`,
            # `_semantic_plan_typed_array_nonempty_bias`,
            # `_semantic_plan_repeated_slot_bias`), which this method only populates
            # when `slot_contract_constrained_decode` (or `template_fill_decode`) is
            # enabled. Setting one of these weights without either flag used to
            # silently no-op every step (E611-E616 replayed a matched control/
            # treatment eval this way without ever observing a difference). Fail
            # loud instead of reproducing that footgun.
            _contract_gated_weight_names = (
                "schema_role_slot_decode_weight",
                "slot_coverage_close_decode_weight",
                "semantic_plan_typed_array_nonempty_margin_decode_weight",
                "semantic_plan_typed_array_item_margin_decode_weight",
                "semantic_plan_repeated_slot_margin_decode_weight",
            )
            _active_contract_gated_weights = sorted(
                name
                for name in _contract_gated_weight_names
                if float(getattr(self.config, name, 0.0) or 0.0) > 0.0
            )
            if _active_contract_gated_weights:
                raise ValueError(
                    "The following decode weights require "
                    "slot_contract_constrained_decode (or template_fill_decode) to "
                    "be enabled, or they silently no-op every decode step because "
                    "self._slot_contracts stays None: "
                    f"{_active_contract_gated_weights}. Pass "
                    "--slot-contract-constrained-decode (or --template-fill-decode)."
                )
        if bool(getattr(self.config, "semantic_role_contract_in_context", False)):
            if not honest:
                raise ValueError(
                    "semantic_role_contract_in_context requires honest_slot_contract"
                )
            prompts = [
                ensure_prompt_semantic_roles(
                    prompt,
                    (
                        self._slot_contracts[i]
                        if self._slot_contracts and i < len(self._slot_contracts)
                        else None
                    ),
                )
                for i, prompt in enumerate(prompts)
            ]
        role_weight = float(
            getattr(self.config, "semantic_role_decode_weight", 0.0) or 0.0
        )
        if role_weight > 0.0:
            if not honest or not bool(
                getattr(self.config, "semantic_role_contract_in_context", False)
            ):
                raise ValueError(
                    "semantic_role_decode_weight requires honest visible role context"
                )
            self._semantic_role_candidates = [
                prompt_semantic_role_candidates(
                    prompt,
                    (
                        self._slot_contracts[i]
                        if self._slot_contracts and i < len(self._slot_contracts)
                        else None
                    ),
                    include_schema_candidates=bool(
                        getattr(
                            self.config,
                            "semantic_role_schema_candidates",
                            False,
                        )
                    ),
                )
                for i, prompt in enumerate(prompts)
            ]
        else:
            self._semantic_role_candidates = None
        plan_weight = max(
            getattr(self.config, "semantic_plan_decode_weight", 0.0) or 0.0,
            getattr(
                self.config,
                "semantic_plan_margin_decode_weight",
                0.0,
            )
            or 0.0,
            getattr(self.config, "semantic_plan_seed_decode_weight", 0.0) or 0.0,
            getattr(self.config, "semantic_plan_binding_decode_weight", 0.0) or 0.0,
            getattr(self.config, "semantic_plan_root_decode_weight", 0.0) or 0.0,
            getattr(
                self.config,
                "semantic_plan_root_margin_decode_weight",
                0.0,
            )
            or 0.0,
            getattr(
                self.config,
                "semantic_plan_repeated_slot_margin_decode_weight",
                0.0,
            )
            or 0.0,
            getattr(
                self.config,
                "semantic_plan_typed_array_nonempty_margin_decode_weight",
                0.0,
            )
            or 0.0,
            getattr(
                self.config,
                "semantic_plan_typed_array_item_margin_decode_weight",
                0.0,
            )
            or 0.0,
        )
        if plan_weight > 0.0:
            if not choice_constrained:
                raise ValueError(
                    "semantic plan decode weights currently require choice-codec "
                    "constrained decode"
                )
            from slm_training.data.semantic_plan import OpenUISemanticPlanCompiler

            component_ids = self._component_inventory_token_ids()
            action_ids = []
            for token_id in component_ids:
                action = str(self.tokenizer.id_to_token.get(token_id, ""))
                for prefix in ("COMP:", "+"):
                    if action.startswith(prefix):
                        action = action[len(prefix) :]
                        break
                action_ids.append(action)
            compiler = OpenUISemanticPlanCompiler(honesty_mode="production")
            self._semantic_plan_action_scores = []
            self._semantic_plan_action_counts = []
            for prompt in prompts:
                plan = prompt_semantic_plan(prompt)
                features = compiler.annotate_actions(None, action_ids, plan)
                self._semantic_plan_action_scores.append(
                    {
                        token_id: feature.plan_confidence
                        for token_id, feature in zip(
                            component_ids, features, strict=True
                        )
                        if feature.component_family_compatible
                        and not feature.conflict_or_unknown
                    }
                )
                family_counts = Counter(
                    slot.component_family
                    for slot in (plan.role_slots if plan is not None else ())
                    if slot.component_family
                )
                self._semantic_plan_action_counts.append(
                    {
                        token_id: family_counts[action_id]
                        for token_id, action_id in zip(
                            component_ids, action_ids, strict=True
                        )
                        if family_counts[action_id] > 0
                    }
                )
        else:
            self._semantic_plan_action_scores = None
            self._semantic_plan_action_counts = None
        reference_weight = float(
            getattr(self.config, "visible_reference_decode_weight", 0.0) or 0.0
        )
        if reference_weight > 0.0 and (
            not choice_constrained
            or not honest
            or not bool(getattr(self.config, "slot_contract_constrained_decode", False))
        ):
            raise ValueError(
                "visible_reference_decode_weight requires honest choice-codec "
                "slot-constrained decode"
            )
        ctx_prompts = self._context_prompts(
            prompts,
            golds=golds,
            design_mds=design_mds,
            slot_contracts=slot_contracts,
            schemas=schemas,
            output_kinds=output_kinds,
            output_categories=output_categories,
        )
        ctx, ctx_pad = self._encode_context(ctx_prompts)
        feature_tables: list[object] = []
        try:
            from slm_training.models.dsl_tokenizer import SymbolTable

            for i in range(len(prompts)):
                symbols = runtime_symbols[i] if runtime_symbols else None
                if symbols:
                    table = SymbolTable.from_runtime_symbols(
                        symbols,
                        sym_slots=self.tokenizer.sym_slots,
                        bind_slots=self.tokenizer.bind_slots,
                        state_slots=self.tokenizer.state_slots,
                    )
                else:
                    placeholders = (
                        slot_contracts[i]
                        if slot_contracts
                        else list(golds[i].placeholders or [])
                        if golds and golds[i] is not None
                        else None
                    )
                    table = SymbolTable.from_placeholders(
                        placeholders,
                        max_slots=getattr(self.tokenizer, "sym_slots", 64),
                    )
                feature_tables.append(table)
        except Exception:  # noqa: BLE001
            feature_tables = []
        if choice_constrained:
            contracts = [
                self._slot_contracts[i]
                if self._slot_contracts and i < len(self._slot_contracts)
                else None
                for i in range(len(prompts))
            ]
            ids = self._choice_ltr_decode_batch(ctx, ctx_pad, length, contracts)
            self._last_generation_evidence = self._choice_generation_evidence(
                ids, contracts
            )
            return [
                self._decode_openui(ids[i], placeholders=contracts[i])
                for i in range(len(prompts))
            ]
        if bool(getattr(self.config, "contract_template_fastpath", False)):
            fast: list[str] = []
            for contract in self._slot_contracts or []:
                if not contract:
                    fast = []
                    break
                certified = self._canonical_valid_openui(
                    build_slot_contract_template(contract)
                )
                if certified is None:
                    fast = []
                    break
                fast.append(certified)
            if len(fast) == len(prompts):
                active = get_active_stats()
                if active is not None:
                    active.template_fastpath_count += len(fast)
                return fast
        row_lengths = [length] * len(prompts)
        if max_len is None:
            predicted_lengths = self._predict_target_lengths(ctx, ctx_pad)
            if predicted_lengths is not None:
                length = max(predicted_lengths)
                row_lengths = (
                    predicted_lengths
                    if bool(getattr(self.config, "compact_active_canvas", True))
                    else [length] * len(predicted_lengths)
                )

        compiler_mode = str(
            getattr(self.config, "compiler_decode_mode", "off") or "off"
        ).lower()
        if use_grammar and (self.config.grammar_ltr_primary or compiler_mode != "off"):
            self._set_runtime_symbol_features(feature_tables)
            self._current_runtime_table = (
                feature_tables[0] if len(feature_tables) == 1 else None
            )
            reserve = (
                int(getattr(self.config, "grammar_draft_window", 8) or 8)
                if compiler_mode != "off" and max_len is None
                else 0
            )
            repair_len = min(
                length + reserve,
                self.config.max_target_len,
                max(8, int(self.config.grammar_ltr_max_tokens)),
            )
            ids = self._greedy_ltr_decode_batch(ctx, ctx_pad, repair_len)
            texts = [self._decode_ids(ids[i]) for i in range(ids.size(0))]
            # Compiler modes already own constrained completion. Running the
            # legacy repair pass afterward discards certified tree output and
            # reintroduces the unconstrained suffix path.
            if self.config.grammar_ltr_repair and compiler_mode == "off":
                texts = self._repair_ltr_texts(
                    texts,
                    ctx,
                    ctx_pad,
                    repair_len,
                    slot_contracts=self._slot_contracts,
                )
            # Certify when grammar-constrained (finalize controls canned fallback).
            # R5: honor generate_max_attempts; when grammar_ltr_repair already ran
            # a BOS fill and the budget is 1, skip a redundant identical redo.
            max_attempts = max(
                1, int(getattr(self.config, "generate_max_attempts", 3) or 3)
            )
            ensure_attempts = (
                0
                if compiler_mode != "off"
                or (bool(self.config.grammar_ltr_repair) and max_attempts <= 1)
                else max_attempts
            )
            certified: list[str] = []
            for i, text in enumerate(texts):
                if feature_tables:
                    features = self._runtime_feature_tensor([feature_tables[i]])
                    self.denoiser.set_runtime_symbol_features(features)
                    self._current_runtime_table = feature_tables[i]
                contract = (
                    self._slot_contracts[i]
                    if self._slot_contracts and i < len(self._slot_contracts)
                    else None
                )
                certified.append(
                    self._ensure_valid_openui(
                        text,
                        ctx[i : i + 1],
                        ctx_pad[i : i + 1],
                        length,
                        attempts=ensure_attempts,
                        slot_contract=contract,
                    )
                )
            return certified

        # Fall back to per-item MaskGIT for non-LTR-primary path.
        out: list[str] = []
        for i in range(len(prompts)):
            if feature_tables:
                features = self._runtime_feature_tensor([feature_tables[i]])
                self.denoiser.set_runtime_symbol_features(features)
                self._current_runtime_table = feature_tables[i]
            contract = (
                self._slot_contracts[i]
                if self._slot_contracts and i < len(self._slot_contracts)
                else None
            )
            out.append(
                # MaskGIT is per-row, so attach the matching request table only.
                self._generate_maskgit_one(
                    ctx[i : i + 1],
                    ctx_pad[i : i + 1],
                    row_lengths[i],
                    use_grammar=use_grammar,
                    slot_contract=contract,
                )
            )
        return out

    def _remask_expand_positions(
        self,
        *,
        ids: torch.Tensor,
        unknown: torch.Tensor,
        conf: torch.Tensor,
        probs: torch.Tensor,
        grammar_remask: list[int],
        tracker: "StabilityTracker | None",
        remask_ratio: float,
        ctx: torch.Tensor | None,
        ctx_pad: torch.Tensor | None,
        allow_model_forwards: bool,
        stats: "SpeculativeStats | None" = None,
    ) -> set[tuple[int, int]] | None:
        """
        Select remask positions (E22/E33/E50/E70) for the given canvas.

        Shared between the real decode loop and E74 successor speculation so
        speculated canvases can reproduce the remask deterministically. When
        the configured policy needs extra model forwards (trust gate, CoRe
        perturbation) and ``allow_model_forwards`` is False, returns ``None``
        — the caller must treat the remask as unpredictable.
        """
        remask = list(grammar_remask)
        known = ~unknown
        remask_policy = str(
            getattr(self.config, "remask_policy", "confidence") or "confidence"
        ).lower()
        use_gate = bool(getattr(self.config, "remask_use_gate", False))
        needs_forwards = use_gate or remask_policy in {"core", "combined"}
        if needs_forwards and not allow_model_forwards:
            return None
        use_policy = bool(
            use_gate
            or getattr(self.config, "remask_use_entropy", False)
            or remask
            or remask_policy in {"core", "combined", "stability", "coverage"}
        )
        gate_trust = None
        entropy = None
        instability = None
        if use_policy or remask_policy in {"core", "combined"}:
            log_probs = torch.log(probs.clamp(min=1e-9))
            entropy = -(probs * log_probs).sum(dim=-1)
            if use_gate:
                try:
                    _logits_h, hidden = self.denoiser(
                        ids,
                        ctx,
                        pad_id=self.tokenizer.pad_id,
                        ctx_pad_mask=ctx_pad,
                        return_hidden=True,
                    )
                    if stats is not None:
                        stats.denoiser_forwards += 1
                    gate_trust = self.trust_gate(hidden)
                except Exception:  # noqa: BLE001
                    gate_trust = None
            if remask_policy in {"core", "combined"}:
                try:
                    perturb_frac = float(
                        getattr(self.config, "core_perturb_frac", 0.25) or 0.25
                    )
                    ids_pert = perturb_known_neighbors(
                        ids,
                        known,
                        mask_id=self.tokenizer.mask_id,
                        perturb_frac=perturb_frac,
                        protect_bos=True,
                    )
                    logits_pert = self.denoiser(
                        ids_pert,
                        ctx,
                        pad_id=self.tokenizer.pad_id,
                        ctx_pad_mask=ctx_pad,
                    )
                    if stats is not None:
                        stats.denoiser_forwards += 1
                    probs_pert = F.softmax(logits_pert, dim=-1)
                    instability = core_instability_scores(probs, probs_pert, ids, known)
                except Exception:  # noqa: BLE001
                    instability = None
            gate_threshold = float(
                getattr(self.config, "fastpath_gate_threshold", 0.5) or 0.5
            )
            if remask_policy == "stability":
                # E70: rank remasks by low persistence + high JSD.
                remask_flat = select_remask_stability_indices(
                    conf,
                    known,
                    remask_ratio=remask_ratio,
                    protect_bos=True,
                    instability=tracker.instability_scores()
                    if tracker is not None
                    else None,
                    grammar_positions=remask,
                    gate_trust=gate_trust,
                    entropy=entropy
                    if bool(getattr(self.config, "remask_use_entropy", False))
                    else None,
                    gate_threshold=gate_threshold,
                    combine_policy=bool(
                        use_gate or getattr(self.config, "remask_use_entropy", False)
                    ),
                )
            elif remask_policy in {"core", "combined"}:
                remask_flat = select_remask_core_indices(
                    conf,
                    known,
                    remask_ratio=remask_ratio,
                    protect_bos=True,
                    instability=instability,
                    grammar_positions=remask,
                    gate_trust=gate_trust,
                    entropy=entropy
                    if bool(getattr(self.config, "remask_use_entropy", False))
                    or remask_policy == "combined"
                    else None,
                    gate_threshold=gate_threshold,
                    combine_policy=remask_policy == "combined",
                )
            elif remask_policy == "coverage":
                # A3: bias remasking toward filler positions when the layout is
                # content-sparse, giving the model another pass to place missing
                # inventory content (soft sibling of the A4 hard contract).
                remask_flat = select_remask_coverage_indices(
                    conf,
                    known,
                    remask_ratio=remask_ratio,
                    protect_bos=True,
                    coverage_deficit=self._coverage_deficit(ids, known),
                    grammar_positions=remask,
                )
            else:
                remask_flat = select_remask_policy_indices(
                    conf,
                    known,
                    remask_ratio=remask_ratio,
                    protect_bos=True,
                    grammar_positions=remask,
                    gate_trust=gate_trust,
                    entropy=entropy
                    if bool(getattr(self.config, "remask_use_entropy", False))
                    or gate_trust is not None
                    else None,
                    gate_threshold=gate_threshold,
                )
        else:
            remask_flat = select_remask_indices(
                conf,
                known,
                remask_ratio=remask_ratio,
                protect_bos=True,
            )
        length = ids.size(-1)
        remask_span = str(getattr(self.config, "remask_span", "token") or "token")
        expand_positions: set[tuple[int, int]] = set()
        for idx in remask_flat:
            b = idx // length
            t = idx % length
            if t == 0:
                continue
            expand_positions.add((b, t))
            if remask_span == "statement":
                try:
                    from slm_training.models.dsl_tokenizer import (
                        is_dsl_native_tokenizer,
                    )

                    if is_dsl_native_tokenizer(self.tokenizer):
                        span = self.tokenizer.spanning_statement(ids[b].tolist(), t)
                        if span is not None:
                            for tt in range(span[0], span[1]):
                                if tt != 0:
                                    expand_positions.add((b, tt))
                except Exception:  # noqa: BLE001
                    pass
        return expand_positions

    def _generate_maskgit_one(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        length: int,
        *,
        use_grammar: bool,
        slot_contract: list[str] | None = None,
        seed_ids: list[int] | None = None,
    ) -> str:
        """Single-sequence MaskGIT unmasking (+ optional grammar repair)."""
        device = self.device_name
        rec = getattr(self, "trace_recorder", None)
        if rec is not None:
            rec.begin(
                length=int(length),
                use_grammar=bool(use_grammar),
                slot_contract=list(slot_contract) if slot_contract else None,
                gen_steps=int(self.config.gen_steps),
                remask_ratio=float(getattr(self.config, "remask_ratio", 0.0) or 0.0),
            )
        ids = torch.full(
            (1, length), self.tokenizer.mask_id, dtype=torch.long, device=device
        )
        ids[0, 0] = self.tokenizer.bos_id
        unknown = ids.eq(self.tokenizer.mask_id)

        if bool(getattr(self.config, "grammar_completion_bounds", False)):
            stats_bucket = get_active_stats()
            try:
                from slm_training.dsl.grammar.fastpath import engine_for_dsl
                from slm_training.models.grammar import active_dsl

                engine = engine_for_dsl(active_dsl())
                bound = (
                    engine.minimum_completion_tokens("") if engine is not None else None
                )
            except Exception:  # noqa: BLE001
                bound = None
            if stats_bucket is not None:
                if bound is None:
                    stats_bucket.completion_bound_unknown += 1
                else:
                    stats_bucket.completion_bound_known += 1

        # Compiler fallback: accepted prefix is authoritative and remains
        # visible while V7 denoises the unresolved suffix in parallel.
        if seed_ids:
            used = min(length, len(seed_ids))
            ids[0, :used] = torch.as_tensor(
                seed_ids[:used], dtype=torch.long, device=device
            )
            unknown[0, :used] = False
            ids[0, 0] = self.tokenizer.bos_id

        # E20: seed from slot-contract skeleton, remask binder/content positions.
        if (
            not seed_ids
            and bool(getattr(self.config, "template_fill_decode", False))
            and slot_contract
        ):
            template = build_slot_contract_template(slot_contract)
            if bool(getattr(self.config, "contract_template_fastpath", False)):
                certified = self._canonical_valid_openui(template)
                if certified is not None:
                    active = get_active_stats()
                    if active is not None:
                        active.template_fastpath_count += 1
                    if rec is not None:
                        rec.event("contract_template_fastpath")
                        rec.end(canvas=[], text=certified)
                    return certified
            seed = self._encode_openui(template, placeholders=list(slot_contract))[
                :length
            ]
            for i, tid in enumerate(seed):
                ids[0, i] = int(tid)
            unknown[0, :] = False
            if len(seed) < length:
                ids[0, len(seed) :] = self.tokenizer.pad_id
            for t in template_mask_positions(seed, self.tokenizer):
                if 0 < t < length:
                    ids[0, t] = self.tokenizer.mask_id
                    unknown[0, t] = True
            ids[0, 0] = self.tokenizer.bos_id
            unknown[0, 0] = False

        steps = max(1, self.config.gen_steps)
        remask_ratio = float(getattr(self.config, "remask_ratio", 0.0) or 0.0)
        # --- V7 speculative denoising state (all opt-in; defaults = V6 path) ---
        remask_policy_cfg = str(
            getattr(self.config, "remask_policy", "confidence") or "confidence"
        ).lower()
        stability_min_persistence = int(
            getattr(self.config, "stability_min_persistence", 0) or 0
        )
        cluster_mode = (
            str(getattr(self.config, "unmask_mode", "positions") or "positions").lower()
            == "cluster"
        )
        cluster_verify = cluster_mode and bool(
            getattr(self.config, "cluster_verify", False)
        )
        use_survival = bool(getattr(self.config, "survival_gate", False))
        survival_threshold = float(
            getattr(self.config, "survival_commit_threshold", 0.3) or 0.3
        )
        speculate = (
            cluster_verify
            and bool(getattr(self.config, "speculative_successor", False))
            and int(getattr(self.config, "speculative_fanout", 2) or 2) > 0
        )
        tracker: StabilityTracker | None = None
        if stability_min_persistence > 0 or remask_policy_cfg == "stability":
            tracker = StabilityTracker(
                jsd_weight=float(
                    getattr(self.config, "stability_jsd_weight", 1.0) or 1.0
                )
            )
        stats = self.speculative_stats
        stats.generates += 1
        successor_cache: SuccessorCache | None = None
        # A2 (ASAp for MaskGIT): adaptive removal of constraint-violating mass.
        asap: AsapLedger | None = (
            AsapLedger()
            if use_grammar and bool(getattr(self.config, "asap_decode", False))
            else None
        )
        for step in range(steps):
            if not unknown.any():
                if remask_ratio <= 0.0 or step >= steps - 1:
                    break
            need_hidden = use_survival
            need_attn = cluster_mode
            hidden: torch.Tensor | None = None
            attn: torch.Tensor | None = None
            cached = successor_cache.get(ids) if successor_cache is not None else None
            if cached is not None:
                logits, hidden, attn = cached
                logits = logits.clone()
                stats.successor_hits += 1
            else:
                if successor_cache is not None and len(successor_cache):
                    stats.successor_misses += 1
                if need_attn:
                    logits, hidden, attn = self.denoiser(
                        ids,
                        ctx,
                        pad_id=self.tokenizer.pad_id,
                        ctx_pad_mask=ctx_pad,
                        return_attn=True,
                    )
                elif need_hidden:
                    logits, hidden = self.denoiser(
                        ids,
                        ctx,
                        pad_id=self.tokenizer.pad_id,
                        ctx_pad_mask=ctx_pad,
                        return_hidden=True,
                    )
                else:
                    logits = self.denoiser(
                        ids, ctx, pad_id=self.tokenizer.pad_id, ctx_pad_mask=ctx_pad
                    )
                stats.denoiser_forwards += 1
                if rec is not None:
                    rec.forward()
            successor_cache = None
            logits = self._mask_inactive_dynamic_logits(logits)
            if use_grammar and self._effective_structural_bias():
                logits = apply_structural_bias(
                    logits,
                    self.tokenizer,
                    bias=self._effective_structural_bias(),
                )
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            if tracker is not None:
                tracker.update(probs)
            survival_scores: torch.Tensor | None = None
            if use_survival and hidden is not None:
                try:
                    survival_scores = self.survival_head(hidden)
                except Exception:  # noqa: BLE001
                    survival_scores = None
            conf_for_unmask = conf.masked_fill(~unknown, -1.0)
            if asap is not None and asap.has_penalties():
                # Distribution-aware ordering: a masked position whose top
                # mass was observed to violate must not outrank positions
                # where model and grammar agree (post-removal confidence).
                conf_for_unmask = asap.adjusted_confidence(
                    probs, conf_for_unmask, unknown
                )
            remaining = int(unknown.sum().item())
            mode = str(getattr(self.config, "parallel_unmask", "adaptive") or "topk")
            flat_idx = (
                select_unmask_indices(
                    conf_for_unmask,
                    unknown,
                    step=step,
                    steps=steps,
                    mode=mode,
                )
                if remaining > 0
                else []
            )
            # E70: only commit predictions whose argmax has persisted.
            if tracker is not None and stability_min_persistence > 0 and flat_idx:
                flat_idx = tracker.filter_commit_indices(
                    flat_idx,
                    length=length,
                    min_persistence=stability_min_persistence,
                )
            # E73 (non-cluster path): cumulative-survival commit budget.
            if (
                not cluster_mode
                and use_survival
                and survival_scores is not None
                and flat_idx
            ):
                flat_idx = filter_by_cumulative_survival(
                    flat_idx,
                    survival_scores,
                    threshold=survival_threshold,
                )
            newly: list[int] = []
            step_commits: list[dict] = []
            step_remasks: list[dict] = []
            use_fast = bool(getattr(self.config, "grammar_fastpath", True))
            mode = str(
                getattr(self.config, "grammar_fastpath_mode", "hybrid") or "hybrid"
            )
            admit_on = use_grammar and use_fast and mode in {"mask", "hybrid"}
            engine = None
            if admit_on:
                try:
                    from slm_training.dsl.grammar.fastpath import (
                        admit_fill,
                        engine_for_dsl,
                    )
                    from slm_training.models.grammar import active_dsl

                    engine = engine_for_dsl(active_dsl())
                    if engine is None:
                        admit_on = False
                except Exception:  # noqa: BLE001
                    engine = None
                    admit_on = False

            def _propose(b: int, t: int) -> int | None:
                """Constrained token proposal for one position (no commit)."""
                prefix = ids[b, :t].tolist()
                forced = (
                    force_emit_token_id(self.tokenizer, prefix) if use_fast else None
                )
                if forced is not None or use_grammar:
                    pick_logits = logits[b, t]
                    if asap is not None and asap.has_penalties(t):
                        # A2: proposal sees the ledger — observed violating
                        # mass at this position is removed in log domain.
                        pick_logits = asap.adjust_logits_row(pick_logits, t)
                    # Speculative / constrained pick — never commit illegal tokens.
                    choice = pick_constrained_token(
                        pick_logits,
                        self.tokenizer,
                        prefix,
                        top_k=self.config.grammar_top_k,
                        forced_token_id=forced,
                        slot_contract=slot_contract
                        if getattr(
                            self.config, "slot_contract_constrained_decode", False
                        )
                        else None,
                        **self._pick_kwargs(),
                    )
                    return choice  # None → leave masked for LTR repair
                return int(pred[b, t].item())

            if not cluster_mode:
                for idx in flat_idx:
                    b = idx // length
                    t = idx % length
                    if not unknown[b, t]:
                        continue
                    decision_trace = None
                    if rec is not None and getattr(rec, "record_support", False):
                        decision_trace = {
                            "pre_canvas": [int(value) for value in ids[b].tolist()],
                            "raw_id": int(pred[b, t].item()),
                            "raw_logit": float(logits[b, t, pred[b, t]].item()),
                        }
                    candidate = _propose(b, t)
                    if candidate is None:
                        continue
                    if admit_on and engine is not None and b == 0:
                        trial = ids[0].tolist()
                        trial[t] = candidate
                        try:
                            if not admit_fill(engine, self.tokenizer, trial):
                                if asap is not None:
                                    asap.penalize(
                                        t,
                                        candidate,
                                        float(probs[b, t, int(candidate)].item()),
                                    )
                                continue  # leave masked; try later / repair
                        except Exception:  # noqa: BLE001
                            if asap is not None:
                                asap.penalize(
                                    t,
                                    candidate,
                                    float(probs[b, t, int(candidate)].item()),
                                )
                            continue  # reject on admit probe failure
                    ids[b, t] = candidate
                    unknown[b, t] = False
                    if b == 0:
                        newly.append(t)
                        if rec is not None:
                            forced = (
                                force_emit_token_id(self.tokenizer, ids[b, :t].tolist())
                                if use_fast
                                else None
                            )
                            commit: dict = {
                                "t": t,
                                "id": int(candidate),
                                "lp": float(
                                    torch.log(
                                        probs[b, t, int(candidate)].clamp(min=1e-9)
                                    ).item()
                                ),
                                "forced": forced is not None,
                                "constrained": bool(use_grammar or forced is not None),
                                "phase": "maskgit",
                            }
                            if decision_trace is not None:
                                commit.update(decision_trace)
                                commit["selected_logit"] = float(
                                    logits[b, t, int(candidate)].item()
                                )
                            if (
                                getattr(rec, "record_support", False)
                                and use_grammar
                                and engine is not None
                            ):
                                try:
                                    from slm_training.dsl.grammar.fastpath.token_map import (
                                        allowed_id_set,
                                    )

                                    prefix_text = self.tokenizer.decode(
                                        ids[0, :t].tolist()
                                    )
                                    engine.set_prefix(prefix_text)
                                    allowed = allowed_id_set(
                                        self.tokenizer, engine.next_terminals()
                                    )
                                    if allowed:
                                        commit["allowed_id_set"] = sorted(
                                            int(x) for x in allowed
                                        )
                                except Exception:  # noqa: BLE001
                                    pass
                            step_commits.append(commit)
            else:
                # --- V7 cluster path (E71/E72/E73/E74); single-sequence B=1 ---
                proposals: dict[int, int] = {}
                for idx in flat_idx:
                    t = idx % length
                    if not unknown[0, t] or t in proposals:
                        continue
                    choice = _propose(0, t)
                    if choice is not None:
                        proposals[t] = int(choice)
                candidates = sorted(proposals)
                ordered: list = []
                if candidates and attn is not None:
                    graph_mode = str(
                        getattr(self.config, "constraint_graph_mode", "off") or "off"
                    ).lower()
                    explicit_edges = (
                        build_constraint_edges(ids[0].tolist(), self.tokenizer)
                        if graph_mode in {"grammar", "hybrid"}
                        else None
                    )
                    active_stats = get_active_stats()
                    if active_stats is not None and explicit_edges is not None:
                        active_stats.constraint_graph_edges += len(explicit_edges)
                    clusters = build_dependency_clusters(
                        attn[0],
                        candidates,
                        threshold=float(
                            getattr(self.config, "cluster_attn_threshold", 0.08) or 0.08
                        ),
                        max_size=max(
                            1, int(getattr(self.config, "cluster_max_size", 4) or 4)
                        ),
                        conf=conf[0],
                        explicit_edges=explicit_edges,
                        use_attention=graph_mode != "grammar",
                    )
                    ordered = order_clusters(
                        clusters,
                        conf=conf[0],
                        attn=attn[0] if graph_mode != "grammar" else None,
                        survival=survival_scores[0]
                        if survival_scores is not None
                        else None,
                    )
                    if use_survival and survival_scores is not None:
                        ordered = survival_commit_budget(
                            ordered, threshold=survival_threshold
                        )
                stats.clusters_proposed += len(ordered)
                admit_fn = None
                if admit_on and engine is not None:
                    from slm_training.dsl.grammar.fastpath import admit_fill as _admit

                    def admit_fn(trial: list[int]) -> bool:  # noqa: ANN001
                        return _admit(engine, self.tokenizer, trial)

                stream_fn = None
                if use_grammar and not bool(
                    getattr(self.config, "grammar_skip_exact_stream_probe", True)
                ):

                    def stream_fn(trial: list[int], newly_pos: list[int]) -> list[int]:
                        return filter_ids_by_stream(self.tokenizer, trial, newly_pos)

                if cluster_verify and ordered:
                    # E74: prepare successor states for likely outcomes while
                    # (or before) the grammar verifies the current transition.
                    def _speculate() -> SuccessorCache | None:
                        if not speculate or step >= steps - 1:
                            return None
                        # Skip speculation when remask needs extra model forwards
                        # (trust gate / CoRe): those canvases cannot be predicted
                        # without paying the remask cost, so the cache would miss.
                        remask_policy = str(
                            getattr(self.config, "remask_policy", "confidence")
                            or "confidence"
                        ).lower()
                        remask_needs_forward = bool(
                            getattr(self.config, "remask_use_gate", False)
                        ) or remask_policy in {"core", "combined"}
                        if remask_ratio > 0.0 and remask_needs_forward:
                            return None
                        outcome_canvases = enumerate_outcome_canvases(
                            ids,
                            ordered,
                            proposals,
                            fanout=max(
                                1,
                                int(getattr(self.config, "speculative_fanout", 2) or 2),
                            ),
                            eos_id=self.tokenizer.eos_id,
                            pad_id=self.tokenizer.pad_id,
                        )
                        if not outcome_canvases:
                            return None
                        # Simulate the deterministic post-commit remask so the
                        # speculated canvas matches the real next-step canvas.
                        # Policies needing extra forwards (gate/CoRe) return
                        # None → cache pre-remask canvases (honest misses).
                        if remask_ratio > 0.0:
                            simulated: list[tuple[int, torch.Tensor]] = []
                            for j, canvas in outcome_canvases:
                                unknown_out = canvas.eq(self.tokenizer.mask_id)
                                expand = self._remask_expand_positions(
                                    ids=canvas,
                                    unknown=unknown_out,
                                    conf=conf,
                                    probs=probs,
                                    grammar_remask=[],
                                    tracker=tracker,
                                    remask_ratio=remask_ratio,
                                    ctx=ctx,
                                    ctx_pad=ctx_pad,
                                    allow_model_forwards=False,
                                    stats=None,
                                )
                                if expand:
                                    canvas = canvas.clone()
                                    for b, t in expand:
                                        canvas[b, t] = self.tokenizer.mask_id
                                simulated.append((j, canvas))
                            outcome_canvases = simulated
                        batch = torch.cat([c for _, c in outcome_canvases], dim=0)
                        k = batch.size(0)
                        logits_k, hidden_k, attn_k = self.denoiser(
                            batch,
                            ctx.expand(k, -1, -1),
                            pad_id=self.tokenizer.pad_id,
                            ctx_pad_mask=ctx_pad.expand(k, -1)
                            if ctx_pad is not None
                            else None,
                            return_attn=True,
                        )
                        stats.denoiser_forwards += 1
                        stats.speculative_batches += 1
                        stats.speculative_canvases += k
                        cache = SuccessorCache()
                        for i, (_j, canvas) in enumerate(outcome_canvases):
                            cache.put(
                                canvas,
                                (
                                    logits_k[i : i + 1],
                                    hidden_k[i : i + 1],
                                    attn_k[i : i + 1],
                                ),
                            )
                        return cache

                    if bool(getattr(self.config, "speculative_overlap", False)):
                        from concurrent.futures import ThreadPoolExecutor

                        with ThreadPoolExecutor(max_workers=1) as pool:
                            fut = pool.submit(
                                verify_clusters_ordered,
                                ids,
                                ordered,
                                proposals,
                                admit=admit_fn,
                                stream_filter=stream_fn,
                            )
                            successor_cache = _speculate()
                            outcome = fut.result()
                    else:
                        successor_cache = _speculate()
                        outcome = verify_clusters_ordered(
                            ids,
                            ordered,
                            proposals,
                            admit=admit_fn,
                            stream_filter=stream_fn,
                        )
                    stats.clusters_accepted += outcome.accepted_clusters
                    if not outcome.all_accepted:
                        stats.clusters_rejected += 1
                    for t in outcome.accepted_positions:
                        ids[0, t] = proposals[t]
                        unknown[0, t] = False
                        newly.append(t)
                    # Rejected cluster stays masked (T2M); deferred clusters
                    # wait for the next pass conditioned on the new canvas.
                else:
                    # E71 without verify: commit only each cluster's anchor
                    # (highest-confidence member); coupled members wait.
                    for cluster in ordered:
                        anchors = sorted(
                            cluster.positions,
                            key=lambda t: float(conf[0, t].item()),
                            reverse=True,
                        )
                        t = anchors[0]
                        if not unknown[0, t]:
                            continue
                        candidate = proposals.get(t)
                        if candidate is None:
                            continue
                        if admit_fn is not None:
                            trial = ids[0].tolist()
                            trial[t] = candidate
                            try:
                                if not admit_fn(trial):
                                    continue
                            except Exception:  # noqa: BLE001
                                continue
                        ids[0, t] = candidate
                        unknown[0, t] = False
                        newly.append(t)
            _ = remaining  # kept for readability / future logging

            for b in range(ids.size(0)):
                eos_positions = (ids[b] == self.tokenizer.eos_id).nonzero(
                    as_tuple=False
                )
                if eos_positions.numel() == 0:
                    continue
                end = int(eos_positions[0].item())
                if end + 1 < length:
                    ids[b, end + 1 :] = self.tokenizer.pad_id
                    unknown[b, end + 1 :] = False

            if use_grammar and newly:
                remask = filter_ids_by_stream(self.tokenizer, ids[0].tolist(), newly)
                for t in remask:
                    if asap is not None:
                        # A2: a stream hard-error remask is an observed
                        # violation of the token just committed here.
                        died = int(ids[0, t].item())
                        asap.penalize(t, died, float(probs[0, t, died].item()))
                    ids[0, t] = self.tokenizer.mask_id
                    unknown[0, t] = True
                if rec is not None and remask:
                    step_remasks.append(
                        {
                            "positions": [int(t) for t in remask],
                            "reason": "grammar_stream",
                        }
                    )
            else:
                remask = []

            # E22 / E33 / E50 / E70: remask committed tokens → mask (T2M; never
            # token-edit). Selection is shared with E74 successor speculation
            # via _remask_expand_positions so speculated canvases match.
            if remask_ratio > 0.0 and step < steps - 1:
                expand_positions = self._remask_expand_positions(
                    ids=ids,
                    unknown=unknown,
                    conf=conf,
                    probs=probs,
                    grammar_remask=remask,
                    tracker=tracker,
                    remask_ratio=remask_ratio,
                    ctx=ctx,
                    ctx_pad=ctx_pad,
                    allow_model_forwards=True,
                    stats=stats,
                )
                assert expand_positions is not None
                for b, t in expand_positions:
                    # E51: remask_to_mask is mandatory (T2M); never token-edit.
                    ids[b, t] = self.tokenizer.mask_id
                    unknown[b, t] = True
                stats.remasked_positions += len(expand_positions)
                if rec is not None and expand_positions:
                    remask_policy = str(
                        getattr(self.config, "remask_policy", "confidence")
                        or "confidence"
                    ).lower()
                    step_remasks.append(
                        {
                            "positions": sorted(
                                int(t) for b, t in expand_positions if b == 0
                            ),
                            "reason": f"policy_{remask_policy}",
                        }
                    )

            if rec is not None:
                rec.step(
                    step,
                    canvas=ids[0].tolist(),
                    unknown=unknown[0].tolist(),
                    commits=step_commits,
                    remasks=step_remasks,
                )

        if asap is not None:
            active_stats = get_active_stats()
            if active_stats is not None:
                active_stats.asap_penalties += asap.penalties
                active_stats.asap_positions += len(asap._removed)

        if unknown.any():
            if use_grammar:
                ids = self._constrained_ltr_repair(
                    ids,
                    unknown,
                    ctx,
                    ctx_pad,
                    slot_contract=slot_contract,
                )
            else:
                logits = self.denoiser(
                    ids, ctx, pad_id=self.tokenizer.pad_id, ctx_pad_mask=ctx_pad
                )
                stats.denoiser_forwards += 1
                if rec is not None:
                    rec.forward()
                pred = logits.argmax(dim=-1)
                ids[unknown] = pred[unknown]
                for b in range(ids.size(0)):
                    eos_positions = (ids[b] == self.tokenizer.eos_id).nonzero(
                        as_tuple=False
                    )
                    if eos_positions.numel() == 0:
                        continue
                    end = int(eos_positions[0].item())
                    if end + 1 < length:
                        ids[b, end + 1 :] = self.tokenizer.pad_id

        text = self._decode_ids(ids[0])
        if rec is not None:
            rec.end(canvas=ids[0].tolist(), text=text)
        if use_grammar:
            repaired = self._repair_surface_syntax(text)
            canonical = self._canonical_valid_openui(repaired)
            if canonical is not None:
                return canonical
            # Grammar filtering can occasionally strand a well-trained MaskGIT
            # decode on a partial prefix. Retry once without token filtering,
            # then accept it only after deterministic syntax repair + validation.
            if rec is not None:
                rec.event("retry_unconstrained")
            active_stats = get_active_stats()
            if active_stats is not None:
                active_stats.unconstrained_retries += 1
            if not bool(getattr(self.config, "allow_unconstrained_fallback", True)):
                return text
            unconstrained = self._generate_maskgit_one(
                ctx,
                ctx_pad,
                length,
                use_grammar=False,
                slot_contract=slot_contract,
            )
            repaired = self._repair_surface_syntax(unconstrained)
            canonical = self._canonical_valid_openui(repaired)
            if canonical is not None:
                return canonical
            return self._ensure_valid_openui(
                text,
                ctx,
                ctx_pad,
                length,
                attempts=max(
                    1, int(getattr(self.config, "generate_max_attempts", 3) or 3)
                ),
                slot_contract=slot_contract,
            )
        return text

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        gold: ExampleRecord | None = None,
        max_len: int | None = None,
        grammar_constrained: bool | None = None,
        design_md: str | None = None,
    ) -> str:
        """Generate OpenUI for one prompt (batched LTR when enabled)."""
        return self.generate_batch(
            [prompt],
            golds=[gold],
            max_len=max_len,
            grammar_constrained=grammar_constrained,
            design_mds=[design_md],
        )[0]

    @torch.inference_mode()
    def generate_with_stats(
        self,
        prompt: str,
        gold: ExampleRecord | None = None,
        max_len: int | None = None,
        grammar_constrained: bool | None = None,
        design_md: str | None = None,
    ) -> tuple[str, DecodeStats]:
        """Generate one sample and return ``(text, DecodeStats)`` phase timings."""
        with collect_decode_stats() as stats:
            try:
                text = self.generate(
                    prompt,
                    gold=gold,
                    max_len=max_len,
                    grammar_constrained=grammar_constrained,
                    design_md=design_md,
                )
            except BaseException as exc:
                setattr(exc, "decode_stats", stats)
                raise
        return text, stats

    def artifact_identity(self) -> dict[str, str]:
        from slm_training.lineage.records import content_sha

        tokenizer_payload = getattr(self.tokenizer, "token_to_id", None)
        if tokenizer_payload is None:
            tokenizer_payload = getattr(self.tokenizer, "vocab", {})
        return {
            "kind": "twotower",
            "base_model_id": self.config.hf_model_name
            if is_hf_context(self.context)
            else "scratch",
            "base_model_revision": str(
                getattr(self.config, "hf_model_revision", None) or "local"
            ),
            "tokenizer_sha": content_sha(tokenizer_payload),
        }

    def compatibility_fingerprint(self) -> str:
        from slm_training.lineage.records import content_sha

        shapes = {name: tuple(value.shape) for name, value in self.state_dict().items()}
        return content_sha(
            {
                **self.artifact_identity(),
                "config": asdict(self.config),
                "parameter_shapes": shapes,
            }
        )

    def generate_constrained(self, prompt: str, **kwargs: object) -> str:
        from slm_training.dsl.parser import validate

        text = self.generate(prompt, grammar_constrained=True, **kwargs)
        program = validate(text)
        return (program.serialized or text).strip()

    def load_parent_weights(self, path: Path | str) -> None:
        """Branch semantics: copy model weights only, never optimizer/RNG state."""
        self.load(path)

    def export(self, path: Path | str, *, format: str = "onnx") -> tuple[Path, ...]:
        if format != "onnx":
            raise ValueError("TwoTower export supports format='onnx' only")
        from scripts.export_playground_onnx import export

        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=False)
        checkpoint = directory / "model.pt"
        self.save(checkpoint)
        context, denoiser = export(checkpoint)
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantized = []
        for raw in (context, denoiser):
            target = raw.with_name(f"{raw.stem}.int8.onnx")
            quantize_dynamic(raw, target, weight_type=QuantType.QInt8)
            raw.unlink()
            target.replace(raw)
            quantized.append(raw)
        checkpoint.unlink()
        artifacts = (
            *quantized,
            checkpoint.with_suffix(".tokenizer.json"),
            checkpoint.with_suffix(".meta.json"),
            checkpoint.with_name(checkpoint.stem + ".context.tokenizer.json"),
        )
        size = sum(item.stat().st_size for item in artifacts if item.exists())
        if size > 1_000_000_000:
            raise ValueError(f"export exceeds 1GB: {size} bytes")
        return tuple(item for item in artifacts if item.exists())

    def _state_dict_for_checkpoint(self) -> dict:
        state = {k: v.cpu() for k, v in self.state_dict().items()}
        # Keep checkpoints small: reload frozen HF backbone from hub/cache on load.
        if is_hf_context(self.context) and self.config.freeze_context:
            state = {
                k: v for k, v in state.items() if not k.startswith("context.backbone.")
            }
        return state

    def save(self, path: Path | str) -> None:
        # RSC-A06 (SLM-242): fail closed before a checkpoint config/manifest is
        # written -- a config mutated after construction (e.g. by
        # ``apply_runtime_overrides`` or direct attribute assignment) must
        # still satisfy every numeric weight/schedule rule before it becomes
        # durable evidence.
        validate_twotower_numeric_schedule(self.config)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "twotower",
            "config": asdict(self.config),
            "gen_len": self.gen_len,
            "output_contract_version": self.output_contract_version,
            "state_dict": self._state_dict_for_checkpoint(),
        }
        tok_path = path.with_suffix(".tokenizer.json")
        self.tokenizer.save(tok_path)
        ctx_tok_name = None
        if self.context_tokenizer is not self.tokenizer:
            ctx_tok_path = path.with_name(path.stem + ".context.tokenizer.json")
            self.context_tokenizer.save(ctx_tok_path)
            ctx_tok_name = ctx_tok_path.name
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "kind": "twotower",
                    "config": asdict(self.config),
                    "gen_len": self.gen_len,
                    "output_contract_version": self.output_contract_version,
                    "tokenizer": str(tok_path.name),
                    "context_tokenizer": ctx_tok_name,
                    "vocab_size": self.tokenizer.vocab_size,
                    "context_vocab_size": self.context_tokenizer.vocab_size,
                    "context_backend": self.config.context_backend,
                    "hf_model_name": self.config.hf_model_name
                    if is_hf_context(self.context)
                    else None,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        torch.save(payload, path)

    def load(self, path: Path | str, *, allow_tie_migration: bool = False) -> None:
        path = Path(path)
        payload = torch.load(path, map_location=self.device_name, weights_only=True)
        if payload.get("kind") != "twotower":
            raise ValueError(f"checkpoint kind {payload.get('kind')!r} is not twotower")
        _check_output_head_tie_migration(
            self,
            payload.get("config") or {},
            allow_tie_migration=allow_tie_migration,
        )
        _load_checkpoint_state(self, payload["state_dict"])
        source_config = payload.get("config") or {}
        restored_prior_fields: list[str] = []
        for field_name in (
            "slot_component_lexeme_priors",
            "slot_component_span_priors",
        ):
            values = source_config.get(field_name)
            if values is None:
                continue
            setattr(
                self.config,
                field_name,
                tuple(
                    (str(key), tuple(float(score) for score in scores))
                    for key, scores in values
                ),
            )
            restored_prior_fields.append(field_name)
        self.initialized_prior_fields = tuple(restored_prior_fields)
        # RSC-A06 (SLM-242): a legacy checkpoint's restored config values
        # (priors above, plus whatever ``self.config`` already carries from
        # construction) must still satisfy every numeric weight/schedule rule.
        # A config that predates a field defaults through ``_get`` and passes;
        # a config whose values are genuinely invalid (e.g. a pre-fix
        # recursive-depth-weights/denoiser_arch mismatch) must be migrated
        # explicitly rather than loaded silently -- see
        # ``slm_training.models.checkpoint_migrate``.
        validate_twotower_numeric_schedule(self.config)
        if "gen_len" in payload:
            self.gen_len = int(payload["gen_len"])
        self.output_contract_version = int(payload.get("output_contract_version", 0))
        tok_path = path.with_suffix(".tokenizer.json")
        if tok_path.exists():
            self.tokenizer = _load_any_tokenizer(tok_path)
        ctx_tok_path = path.with_name(path.stem + ".context.tokenizer.json")
        if ctx_tok_path.exists():
            self.context_tokenizer = _load_any_tokenizer(ctx_tok_path)
        else:
            self.context_tokenizer = self.tokenizer

    @classmethod
    def from_checkpoint(
        cls,
        path: Path | str,
        device: str | torch.device = "cpu",
        local_files_only: bool | None = None,
        *,
        allow_tie_migration: bool = False,
    ) -> TwoTowerModel:
        path = Path(path)
        payload = torch.load(path, map_location=device, weights_only=True)
        if not isinstance(payload, dict) or payload.get("kind") != "twotower":
            kind = (
                payload.get("kind")
                if isinstance(payload, dict)
                else type(payload).__name__
            )
            raise ValueError(f"checkpoint kind {kind!r} is not twotower")
        tok_path = path.with_suffix(".tokenizer.json")
        if not tok_path.exists():
            raise FileNotFoundError(f"missing tokenizer next to checkpoint: {tok_path}")
        tokenizer = _load_any_tokenizer(tok_path)
        ctx_tok_path = path.with_name(path.stem + ".context.tokenizer.json")
        context_tokenizer = (
            _load_any_tokenizer(ctx_tok_path) if ctx_tok_path.exists() else tokenizer
        )
        raw_cfg = dict(payload.get("config") or {})
        if isinstance(raw_cfg.get("grammar_ltr_stages"), list):
            raw_cfg["grammar_ltr_stages"] = tuple(raw_cfg["grammar_ltr_stages"])
        for key in ("diffusion_policies", "diffusion_length_buckets"):
            if isinstance(raw_cfg.get(key), list):
                raw_cfg[key] = tuple(raw_cfg[key])
        if raw_cfg.get("grammar_ltr_stages") is None:
            raw_cfg["grammar_ltr_stages"] = (64, 128, 192, 256)
        if local_files_only is not None:
            raw_cfg["local_files_only"] = bool(local_files_only)
        # Ignore unknown keys for forward/back compat
        valid = {f.name for f in TwoTowerConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        cfg = TwoTowerConfig(**{k: v for k, v in raw_cfg.items() if k in valid})
        model = cls(
            tokenizer=tokenizer,
            config=cfg,
            device=device,
            context_tokenizer=context_tokenizer,
        )
        _check_output_head_tie_migration(
            model,
            raw_cfg,
            allow_tie_migration=allow_tie_migration,
        )
        _load_checkpoint_state(model, payload["state_dict"])
        if "gen_len" in payload:
            model.gen_len = int(payload["gen_len"])
        model.output_contract_version = int(payload.get("output_contract_version", 0))
        return model

    @classmethod
    def from_records(
        cls,
        records: list[ExampleRecord],
        config: TwoTowerConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> TwoTowerModel:
        cfg = config or TwoTowerConfig()
        if _is_choice_output(cfg):
            from slm_training.models.choice_tokenizer import ChoiceTokenizer

            tokenizer = ChoiceTokenizer.build()
            # Scratch context keeps a prompt-word tokenizer (decoupled).
            context_tokenizer = OpenUITokenizer.build([r.prompt for r in records])
            max_target = max(
                (
                    len(
                        tokenizer.encode(
                            r.openui,
                            add_special=True,
                            placeholders=list(r.placeholders or []),
                        )
                    )
                    for r in records
                ),
                default=32,
            )
            max_prompt = max(
                (len(context_tokenizer.encode(r.prompt)) for r in records),
                default=16,
            )
        elif _is_lexer_output(cfg):
            from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

            if not bool(getattr(cfg, "symbol_anonymization", True)):
                # C4 fail-closed: the grammar gate's NAME accept-set is
                # BIND-only and macros/relative refs presuppose pooled ids, so
                # the surface arm refuses those combinations outright rather
                # than silently decoding garbage.
                if bool(getattr(cfg, "grammar_constrained", False)):
                    raise ValueError(
                        "symbol_anonymization=False is incompatible with "
                        "grammar_constrained decode (NAME gate admits only "
                        "<BIND_j> ids)"
                    )
                if bool(getattr(cfg, "macro_tokens", False)):
                    raise ValueError(
                        "symbol_anonymization=False is incompatible with "
                        "macro_tokens (tables are mined on anonymized ids)"
                    )
                if str(getattr(cfg, "bind_encoding", "absolute")) != "absolute":
                    raise ValueError(
                        "symbol_anonymization=False requires bind_encoding='absolute'"
                    )

            tokenizer = DSLNativeTokenizer.build(
                bind_encoding=str(
                    getattr(cfg, "bind_encoding", "absolute") or "absolute"
                )
            )
            if bool(getattr(cfg, "macro_tokens", False)):
                # C3: the induced table is persisted with the tokenizer, so
                # train and decode can never disagree about expansions.
                from slm_training.data.macro_induction import induce_macros

                result = induce_macros(
                    [r.openui for r in records if (r.openui or "").strip()],
                    tokenizer,
                )
                tokenizer.set_macro_expansions(result.expansions)
            # Scratch context keeps a prompt-word tokenizer (decoupled).
            ctx_texts = [r.prompt for r in records]
            context_tokenizer = OpenUITokenizer.build(ctx_texts)
            use_sym = bool(getattr(cfg, "use_symbol_table", True))
            max_target = max(
                (
                    len(
                        tokenizer.encode(
                            r.openui,
                            add_special=True,
                            use_symbol_table=use_sym,
                            placeholders=list(r.placeholders or []),
                            symbol_anonymization=bool(
                                getattr(cfg, "symbol_anonymization", True)
                            ),
                        )
                    )
                    for r in records
                ),
                default=32,
            )
            max_prompt = max(
                (len(context_tokenizer.encode(r.prompt)) for r in records),
                default=16,
            )
        else:
            texts = [r.prompt for r in records] + [r.openui for r in records]
            tokenizer = OpenUITokenizer.build(texts)
            context_tokenizer = tokenizer
            max_prompt = max(
                (len(tokenizer.encode(r.prompt)) for r in records), default=16
            )
            max_target = max(
                (len(tokenizer.encode(r.openui)) for r in records), default=32
            )
        cfg.max_prompt_len = max(cfg.max_prompt_len, max_prompt + 4)
        cfg.max_target_len = max(cfg.max_target_len, max_target + 8)
        model = cls(
            tokenizer=tokenizer,
            config=cfg,
            device=device,
            context_tokenizer=context_tokenizer,
        )
        balance_power = float(
            getattr(cfg, "slot_component_class_balance_power", 0.0) or 0.0
        )
        if balance_power > 0.0 and model.slot_component_head is not None:
            component_index = model._component_name_index()
            counts = [0] * len(component_index)
            for record in records:
                owners = model._slot_component_owners(record.openui)
                for slot in record.placeholders:
                    target = component_index.get(owners.get(slot, ""))
                    if target is not None:
                        counts[target] += 1
            observed = [count for count in counts if count > 0]
            if observed:
                total = float(sum(observed))
                classes = float(len(observed))
                weights = [
                    (total / (classes * count)) ** balance_power if count > 0 else 0.0
                    for count in counts
                ]
                observed_mean = (
                    sum(
                        weight * count
                        for weight, count in zip(weights, counts, strict=True)
                        if count > 0
                    )
                    / total
                )
                cfg.slot_component_class_weights = tuple(
                    weight / observed_mean if weight > 0.0 else 0.0
                    for weight in weights
                )
        prior_weight = float(
            getattr(cfg, "slot_component_lexeme_prior_weight", 0.0) or 0.0
        )
        if prior_weight > 0.0 and model.slot_component_head is not None:
            component_index = model._component_name_index()
            class_counts: Counter[int] = Counter()
            token_counts: dict[str, Counter[int]] = defaultdict(Counter)
            token_totals: Counter[str] = Counter()
            for record in records:
                owners = model._slot_component_owners(record.openui)
                for slot in record.placeholders:
                    target = component_index.get(owners.get(slot, ""))
                    if target is None:
                        continue
                    class_counts[target] += 1
                    for token in set(tokenize_text(slot)):
                        if not any(char.isalnum() for char in token):
                            continue
                        token_counts[token][target] += 1
                        token_totals[token] += 1
            total = sum(class_counts.values())
            classes = len(component_index)
            if total and classes:
                base = [
                    (class_counts[index] + 1.0) / (total + classes)
                    for index in range(classes)
                ]
                prior_mass = 0.5 * classes
                cfg.slot_component_lexeme_priors = tuple(
                    (
                        token,
                        tuple(
                            math.log(
                                (counts[index] + prior_mass * base[index])
                                / (token_totals[token] + prior_mass)
                            )
                            - math.log(base[index])
                            for index in range(classes)
                        ),
                    )
                    for token, counts in sorted(token_counts.items())
                    if token_totals[token] >= 2
                )
        span_weight = float(
            getattr(cfg, "slot_component_span_prior_weight", 0.0) or 0.0
        )
        if span_weight > 0.0 and model.slot_component_head is not None:
            component_index = model._component_name_index()
            component_ids = model._component_inventory_token_ids()
            slot_content_count = getattr(model.tokenizer, "slot_content_count", None)
            class_counts: Counter[int] = Counter()
            span_counts: dict[str, Counter[int]] = defaultdict(Counter)
            span_totals: Counter[str] = Counter()
            for record in records:
                owners = model._slot_component_owners(record.openui)
                slots = list(record.placeholders)
                for slot in slots:
                    target = component_index.get(owners.get(slot, ""))
                    if target is not None:
                        class_counts[target] += 1
                for index in range(len(slots) - 1):
                    owner = owners.get(slots[index], "")
                    if not owner or owner != owners.get(slots[index + 1], ""):
                        continue
                    target = component_index.get(owner)
                    if (
                        target is None
                        or not callable(slot_content_count)
                        or int(slot_content_count(component_ids[target])) != 2
                    ):
                        continue
                    key = "\x1f".join(
                        model._slot_role_token(slot)
                        for slot in slots[index : index + 2]
                    )
                    span_counts[key][target] += 1
                    span_totals[key] += 1
            total = sum(class_counts.values())
            classes = len(component_index)
            if total and classes:
                base = [
                    (class_counts[index] + 1.0) / (total + classes)
                    for index in range(classes)
                ]
                cfg.slot_component_span_priors = tuple(
                    (
                        key,
                        tuple(
                            (
                                math.log(
                                    (counts[index] + 0.5)
                                    / (span_totals[key] + 0.5 * classes)
                                )
                                - math.log(base[index])
                            )
                            if counts[index] > 0
                            else 0.0
                            for index in range(classes)
                        ),
                    )
                    for key, counts in sorted(span_counts.items())
                    if span_totals[key] >= 2
                )
        model.gen_len = max(max_target + 2, 16)
        if bool(getattr(cfg, "teacher_init_embeddings", False)):
            model._try_teacher_init_embeddings(records)
        return model

    def _try_teacher_init_embeddings(self, records: list[ExampleRecord]) -> None:
        """Initialize DSL symbol rows from a frozen HF teacher when available."""
        try:
            from slm_training.models.dsl_tokenizer import (
                TokenKind,
                is_dsl_native_tokenizer,
            )

            if not is_dsl_native_tokenizer(self.tokenizer):
                return
            if not is_hf_context(self.context):
                return
            assert isinstance(self.context, HFContextEncoder)

            init_source = str(
                getattr(self.config, "action_embedding_init", "none") or "none"
            )
            alias_sources = {
                "alias_aware_description",
                "alias_aware_signature_only",
                "alias_aware_shuffled",
            }
            canonical_sources = {
                "schema_description",
                "expanded_description",
                "shuffled",
                "description_without_canonical_name",
                "canonical_name_plus_description",
                "signature_only",
            }
            use_catalog = (
                init_source in canonical_sources or init_source in alias_sources
            )
            alias_mode = str(
                getattr(self.config, "action_alias_mode", "canonical") or "canonical"
            )
            name_mode = str(
                getattr(self.config, "action_description_name_mode", "schema")
                or "schema"
            )
            use_alias = (
                alias_mode not in {"off", "canonical"} or init_source in alias_sources
            )

            alias_map = None
            if use_catalog:
                from slm_training.dsl.action_descriptions import (
                    ActionDescriptionCatalog,
                )

                catalog = ActionDescriptionCatalog.build()

            if use_alias:
                from slm_training.dsl.action_descriptions import (
                    ActionAliasMap,
                    build_alias_map,
                )

                manifest_path = getattr(self.config, "action_alias_manifest", None)
                if manifest_path is not None and Path(manifest_path).is_file():
                    alias_map = ActionAliasMap.from_dict(
                        json.loads(Path(manifest_path).read_text(encoding="utf-8"))
                    )
                elif use_catalog:
                    alias_map = build_alias_map(
                        seed=int(getattr(self.config, "seed", 0) or 0),
                        pack_id=f"{init_source}:{alias_mode}",
                        action_keys=catalog.keys(),
                    )

            if use_catalog:
                descriptions = catalog.descriptions_for(
                    init_source,
                    alias_map=alias_map,
                    name_mode=name_mode,
                )
                fallback = catalog.descriptions_for("current_stub")
            else:
                catalog = None  # type: ignore[assignment]
                descriptions = {}
                fallback = {}

            # Map component / fixed-string tokens to short textual glosses.
            glosses: dict[int, str] = {}
            for tid, tok in self.tokenizer.id_to_token.items():
                kind = self.tokenizer.kind_of(tid)
                if kind == TokenKind.COMPONENT:
                    key = f"+{tok}"
                    if key in descriptions:
                        glosses[tid] = descriptions[key]
                    else:
                        glosses[tid] = fallback.get(key, f"{tok} UI component")
                elif kind == TokenKind.LIT and tok.startswith("STR:"):
                    glosses[tid] = tok[4:]
                elif kind == TokenKind.STRUCT and tok not in {"NL"}:
                    # Structural tokens use production-style keys when available.
                    key = tok
                    if key in descriptions:
                        glosses[tid] = descriptions[key]
                    else:
                        glosses[tid] = fallback.get(key, f"punctuation {tok}")
            if not glosses:
                return
            prompts = [glosses[i] for i in sorted(glosses)]
            ids = sorted(glosses)
            with torch.no_grad():
                hidden, _ = self.context.forward_prompts(
                    prompts,
                    max_len=min(32, self.config.max_prompt_len),
                    device=self.device_name,
                )
                # Mean-pool over sequence.
                pooled = hidden.mean(dim=1)
                if pooled.size(-1) != self.config.d_model:
                    # Project if HF hidden size differs (should match d_model via adapter).
                    return
                weight = self.denoiser.tok.weight.data
                for row, vec in zip(ids, pooled):
                    weight[row].copy_(vec)
        except Exception:  # noqa: BLE001
            return
