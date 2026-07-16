"""TwoTower OpenUI model: context encoder + trainable masked denoiser."""

from __future__ import annotations

import json
import math
import random
import re
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
    StabilityTracker,
    core_instability_scores,
    perturb_known_neighbors,
    select_remask_core_indices,
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
    inventory_from_prompt,
    template_mask_positions,
)
from slm_training.dsl.grammar.fastpath.gate import FastPathGate
from slm_training.models.tokenizer import OpenUITokenizer


def _is_lexer_output(config: "TwoTowerConfig | None") -> bool:
    if config is None:
        return False
    return str(getattr(config, "output_tokenizer", "compositional") or "").lower() in {
        "lexer",
        "dsl",
        "dsl_native",
        "native",
    }


def _load_any_tokenizer(path: Path | str):
    """Load compositional or lexer-native tokenizer from JSON sidecar."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
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
    compiler_alignment_stratified: bool = False
    compiler_alignment_semantic_exhaustive: bool = False
    symbol_boundary_loss_weight: float = 0.0
    # Extra CE weight on gold placeholder token positions (fidelity signal).
    fidelity_loss_weight: float = 0.5
    design_md_in_context: bool = True
    design_md_budget: int = 1800
    schema_in_context: bool = False
    slot_contract_in_context: bool = False
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
    fastpath_aux_weight: float = 0.0
    fastpath_gate_threshold: float = 0.5
    # E31: train/use FastPathGate trust head for remask.
    trust_gate_train: bool = False
    # V5: output-side representation
    # compositional = legacy OpenUITokenizer v2; lexer = DSLNativeTokenizer
    output_tokenizer: str = "compositional"
    # When output_tokenizer=lexer: map placeholders to <SYM_i> (E41+).
    use_symbol_table: bool = True
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
    # V8 request-conditioned dynamic vocabulary; ``none`` is checkpoint-identical.
    runtime_symbol_features: str = "none"  # none | surface | role_gated
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
        self.denoiser = DenoiserTower(
            vocab_size=tokenizer.vocab_size,
            d_model=self.config.d_model,
            n_layers=self.config.denoiser_layers,
            n_heads=self.config.n_heads,
            max_len=self.config.max_target_len,
            dropout=self.config.dropout,
            kind_ids=kind_ids,
            n_kinds=max(kind_ids) + 1 if kind_ids else 0,
        )
        self.length_head = (
            nn.Linear(self.config.d_model, len(self.config.diffusion_length_buckets))
            if str(getattr(self.config, "mask_pattern", "random")) == "diffusion"
            else None
        )
        # E31 BackPlay-lite: plug-in trust head over denoiser hiddens.
        self.trust_gate = FastPathGate(self.config.d_model)
        # E73 DSpark-lite: plug-in trajectory-survival head (V7).
        self.survival_head = FastPathGate(self.config.d_model)
        # V7 decode telemetry (MaskGIT path): forwards, successor hits/misses.
        self.speculative_stats = SpeculativeStats()
        self._rng = random.Random(self.config.seed)
        self.gen_len = self.config.max_target_len
        # Optional decode trajectory recorder (distill.DecodeTraceRecorder).
        # Zero-cost when None; not part of checkpoints.
        self.trace_recorder = None
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
        self._slot_contracts: list[list[str] | None] | None = None
        # Per-example symbol tables for lexer-native encode/decode.
        self._symbol_tables: dict[str, object] = {}
        self._current_runtime_table: object | None = None
        self.to(device)

    def _effective_structural_bias(self) -> float:
        if getattr(self.config, "grammar_trust_model", False):
            return 0.0
        return float(self.config.structural_bias or 0.0)

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
        return (p for p in self.parameters() if p.requires_grad)

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
                    text = self._format_one_context(
                        r.prompt,
                        r.design_md,
                        query_prompt=r.prompt,
                        slot_contract=self._resolve_slot_contract(
                            r.prompt, r, r.design_md
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
                            boundaries = [j for j, tid in enumerate(row) if tid in newline_ids]
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
                from slm_training.models.dsl_tokenizer import (
                    TokenKind,
                    is_dsl_native_tokenizer,
                )

                if is_dsl_native_tokenizer(self.tokenizer):
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
            from slm_training.models.dsl_tokenizer import (
                SymbolTable,
                is_dsl_native_tokenizer,
            )

            if is_dsl_native_tokenizer(self.tokenizer):
                use_sym = bool(getattr(self.config, "use_symbol_table", True))
                table = SymbolTable.from_placeholders(
                    placeholders, max_slots=self.tokenizer.sym_slots
                )
                if bool(getattr(self.config, "symbol_slot_augmentation", False)) and self.training:
                    key_seed = sum(ord(ch) for ch in (cache_key or openui))
                    table = table.permuted(int(self.config.seed) + key_seed)
                if cache_key is not None:
                    self._symbol_tables[cache_key] = table
                return self.tokenizer.encode(
                    openui,
                    table=table,
                    use_symbol_table=use_sym,
                    placeholders=placeholders,
                )
        except Exception:  # noqa: BLE001
            pass
        return self.tokenizer.encode(openui)

    def _runtime_feature_tensor(self, tables: list[object]) -> torch.Tensor | None:
        """Build per-example deltas for reserved symbol rows from existing embeddings."""
        mode = str(getattr(self.config, "runtime_symbol_features", "none") or "none")
        if mode == "none" or not tables:
            return None
        if mode not in {"surface", "role_gated"}:
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
                    features[row, token_id] = weight.index_select(0, index).mean(0)
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
            from slm_training.models.dsl_tokenizer import (
                SymbolTable,
                is_dsl_native_tokenizer,
            )

            if is_dsl_native_tokenizer(self.tokenizer):
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
                text = self._format_one_context(
                    r.prompt,
                    r.design_md,
                    query_prompt=r.prompt,
                    slot_contract=self._resolve_slot_contract(r.prompt, r, r.design_md)
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

        with timed("denoiser_forward"):
            self._set_runtime_symbol_features(
                [self._symbol_tables.get(key) for key in cache_keys]
            )
            logits = self.denoiser(
                noisy, ctx, pad_id=self.tokenizer.pad_id, ctx_pad_mask=ctx_pad
            )
        if predict_mask.any():
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = target_ids.reshape(-1)
            ce = F.cross_entropy(flat_logits, flat_targets, reduction="none")
            weights = torch.ones_like(ce)
            if ltr_w > 0.0 and fuse and ltr_suffix.any():
                suffix_flat = ltr_suffix.reshape(-1)
                weights = weights + (ltr_w * suffix_flat.float())
                prefix_w = float(getattr(self.config, "ltr_prefix_loss_weight", 0.0) or 0.0)
                if prefix_w > 0.0:
                    positions = torch.arange(target_ids.size(1), device=target_ids.device)
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
                (row_values * row_mask).sum(dim=1) / row_counts
            ).detach().cpu().tolist()
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
                from slm_training.models.dsl_tokenizer import (
                    SymbolTable,
                    TokenKind,
                    is_dsl_native_tokenizer,
                )

                if is_dsl_native_tokenizer(self.tokenizer):
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
                first_content = 1 if int(target_ids[i, 0]) == self.tokenizer.bos_id else 0
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
                getattr(
                    self.config, "compiler_alignment_semantic_exhaustive", False
                )
            )
            aligned_canvases: list[torch.Tensor] = []
            aligned_targets: list[int] = []
            aligned_positions: list[int] = []
            aligned_context_rows: list[int] = []
            aligned_kinds: list[str] = []
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
                position_index = torch.tensor(
                    aligned_positions,
                    device=aligned_logits.device,
                    dtype=torch.long,
                )
                batch_index = torch.arange(
                    aligned_rows, device=aligned_logits.device
                )
                target_tensor = torch.tensor(
                    aligned_targets,
                    device=aligned_logits.device,
                    dtype=target_ids.dtype,
                )
                alignment_losses = F.cross_entropy(
                    aligned_logits[batch_index, position_index],
                    target_tensor,
                    reduction="none",
                )
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
                    float(alignment_loss.detach().cpu())
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
        parts = re.split(r'("(?:\\.|[^"\\])*")', text)
        for index in range(0, len(parts), 2):
            parts[index] = re.sub(r"\s*=\s*=+\s*", " = ", parts[index])
            parts[index] = re.sub(r",\s*=\s*(?=[)\]])", "", parts[index])
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
        repair_commits: list[dict] = []

        def _record_commit(
            pos: int,
            token_id: int,
            logits_1d: torch.Tensor,
            *,
            forced: bool,
        ) -> None:
            if rec is None:
                return
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
                        "legal_candidates": int(stats.constrained_last_legal_candidates),
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
                t, int(choice), logits[0, local_t], forced=forced is not None
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
                    _record_commit(pos, int(nxt), logits[0, pos], forced=False)
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
        index = torch.as_tensor(
            candidate_ids, dtype=torch.long, device=hidden.device
        )
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

    def _select_compiler_path(
        self,
        prefix: list[int],
        paths: tuple,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        length: int,
        *,
        tree: bool,
    ) -> tuple[int, ...]:
        """Rank completion paths using gathered rows of the tied LM head."""
        if len(paths) == 1:
            return tuple(paths[0].token_ids)
        stats = get_active_stats()
        if stats is not None:
            stats.compiler_candidates += len(paths)

        def record_choice(
            chosen: int,
            scores: list[float],
            phase: str,
            *,
            first_edge_scores: list[float] | None = None,
        ) -> None:
            if stats is None or len(stats.constrained_selection_traces) >= 64:
                return
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
            for row, parent in enumerate(parents):
                candidate_ids = tuple(sorted(children[parent]))
                if len(candidate_ids) == 1:
                    edge_scores[(parent, candidate_ids[0])] = 0.0
                    continue
                scores = self._project_candidates(
                    hidden[row, len(parent)], candidate_ids
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
        if bool(getattr(self.config, "grammar_sample_decode", False)):
            temp = float(
                getattr(self.config, "grammar_sample_temperature", 0.8) or 0.8
            )
            probs = F.softmax(torch.tensor(path_scores) / temp, dim=0)
            chosen = int(torch.multinomial(probs, 1).item())
        else:
            chosen = max(range(len(paths)), key=path_scores.__getitem__)
        first_edge_scores = [
            edge_scores.get((tuple(prefix), int(path.token_ids[0])), 0.0)
            for path in paths
        ]
        record_choice(
            chosen,
            path_scores,
            "compiler_tree",
            first_edge_scores=first_edge_scores,
        )
        return tuple(paths[chosen].token_ids)

    def _compiler_ltr_decode_one(
        self,
        ctx: torch.Tensor,
        ctx_pad: torch.Tensor,
        length: int,
        *,
        mode: str,
        slot_contract: list[str] | None,
    ) -> torch.Tensor:
        from slm_training.dsl.grammar.fastpath.compiler_draft import (
            build_completion_forest,
        )

        if mode not in {"forced", "restricted", "tree"}:
            raise ValueError(
                "compiler_decode_mode must be off, forced, restricted, or tree"
            )
        state_rows = self._new_grammar_states(1)
        state = state_rows[0] if state_rows else make_grammar_state()
        prefix = [int(self.tokenizer.bos_id)]
        stats = get_active_stats()
        while len(prefix) < length:
            with timed_ms(stats, "compiler_ms"):
                forest = build_completion_forest(
                    self.tokenizer,
                    prefix,
                    state=state,
                    slot_contract=slot_contract,
                    max_path_tokens=int(
                        getattr(self.config, "grammar_draft_window", 8) or 8
                    ),
                )
            # Partial coverage still contains individually grammar-admitted
            # paths. Tree/restricted modes must consume those paths; falling
            # back merely because the vocabulary is not exhaustive discards
            # the deterministic symbolic constraint at exactly the branch
            # points it is meant to protect. Fall back only when no legal path
            # exists at all.
            if not forest.paths:
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
                selected = (int(choice if choice is not None else self.tokenizer.eos_id),)
            else:
                selected = self._select_compiler_path(
                    prefix,
                    forest.paths,
                    ctx,
                    ctx_pad,
                    length,
                    tree=mode == "tree",
                )
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
                break

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
    ) -> list[str] | None:
        """Return inventory for decode/context.

        E35 honest mode: inventory comes from the user-visible prompt/DESIGN.md
        only (never ``gold.placeholders``). When the prompt lacks an explicit
        inventory, a keyword heuristic is used. Non-honest mode (legacy V3)
        falls back to gold placeholders for template fill / conditioning.
        """
        dm = design_md
        if dm is None and gold is not None:
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
        if use_grammar and (
            self.config.grammar_ltr_primary or compiler_mode != "off"
        ):
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
            or remask_policy in {"core", "combined", "stability"}
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
                    engine.minimum_completion_tokens("")
                    if engine is not None
                    else None
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
                    # Speculative / constrained pick — never commit illegal tokens.
                    choice = pick_constrained_token(
                        logits[b, t],
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
                    candidate = _propose(b, t)
                    if candidate is None:
                        continue
                    if admit_on and engine is not None and b == 0:
                        trial = ids[0].tolist()
                        trial[t] = candidate
                        try:
                            if not admit_fill(engine, self.tokenizer, trial):
                                continue  # leave masked; try later / repair
                        except Exception:  # noqa: BLE001
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

    def load(self, path: Path | str) -> None:
        path = Path(path)
        payload = torch.load(path, map_location=self.device_name, weights_only=True)
        if payload.get("kind") != "twotower":
            raise ValueError(f"checkpoint kind {payload.get('kind')!r} is not twotower")
        _load_checkpoint_state(self, payload["state_dict"])
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
        if _is_lexer_output(cfg):
            from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

            tokenizer = DSLNativeTokenizer.build()
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
            # Map component / fixed-string tokens to short textual glosses.
            glosses: dict[int, str] = {}
            for tid, tok in self.tokenizer.id_to_token.items():
                kind = self.tokenizer.kind_of(tid)
                if kind == TokenKind.COMPONENT:
                    glosses[tid] = f"{tok} UI component"
                elif kind == TokenKind.LIT and tok.startswith("STR:"):
                    glosses[tid] = tok[4:]
                elif kind == TokenKind.STRUCT and tok not in {"NL"}:
                    glosses[tid] = f"punctuation {tok}"
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
