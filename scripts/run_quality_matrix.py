#!/usr/bin/env python3
"""Run the quality experiment matrix (docs/design/quality-experiment-matrix.md)."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from slm_training.harnesses.model_build import ModelBuildConfig, build_model, train
from slm_training.harnesses.model_build.data import load_train_records
from slm_training.harnesses.model_build.eval_runner import evaluate_suites
from slm_training.harnesses.model_build.eval_policy import (
    STRICT_COMPILER_TREE_POLICY,
)
from slm_training.runtime.telemetry import run_trace
from slm_training.harnesses.model_build.ship_gates import (
    DEFAULT_SHIP_GATES,
    evaluate_ship_gates,
)


SUITES = ["smoke", "held_out", "adversarial", "ood", "rico_held"]


def _copy_checkpoint(src: Path, dest: Path) -> Path:
    """Copy checkpoint + tokenizer/meta sidecars used by TwoTower.from_checkpoint."""
    import shutil

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    for side in (
        src.with_suffix(".tokenizer.json"),
        src.with_suffix(".meta.json"),
        src.with_name(src.stem + ".context.tokenizer.json"),
    ):
        if side.is_file():
            shutil.copy2(side, dest.parent / side.name)
    return dest


@dataclass(frozen=True)
class Experiment:
    eid: str
    run_id: str
    description: str
    train_dir: Path
    initialization: Literal["scratch", "parent", "eval_only", "process"] = "parent"
    parent_checkpoint: str | None = None
    # Train overrides
    fidelity_loss_weight: float = 0.0
    schema_in_context: bool = False
    retrieval_k: int = 0
    best_of_n: int = 1
    use_curriculum: bool = False
    grammar_ltr_repair: bool = False
    d_model: int = 128
    n_heads: int = 4
    context_layers: int = 2
    denoiser_layers: int = 4
    grammar_ltr_max_tokens: int = 192
    # Eval-only overlay on a prior run (skip train if set)
    eval_from_run: str | None = None
    design_md_in_context: bool | None = True
    runtime_override_fields: frozenset[str] | None = None
    # Absolute checkpoint seed (preferred over eval_from_run when set)
    seed_checkpoint: str | None = None
    preference: bool = False
    mix_curriculum: bool = True
    rl: bool = False
    slot_contract_in_context: bool = False
    slot_contract_constrained_decode: bool = False
    namespace_augment: bool = False
    ltr_loss_weight: float = 1.0
    ltr_prefix_loss_weight: float = 0.0
    symbol_boundary_loss_weight: float = 0.0
    # Eval-only overlay: decode sweep presets (E17)
    decode_sweep: str | None = None
    eval_from_checkpoint: str | None = None
    # V3 levers
    grammar_ltr_primary: bool = True
    template_fill_decode: bool = False
    contract_template_fastpath: bool = False
    mdlm_schedule: bool = False
    remask_ratio: float = 0.0
    gen_steps_override: int | None = None
    # V4 levers
    honest_slot_contract: bool = False
    suffix_rollback_window: int = 0
    remask_use_gate: bool = False
    remask_use_entropy: bool = False
    visible_corrupt_rate: float = 0.0
    trust_gate: bool = False
    grammar_fastpath_mode: str = "hybrid"
    # V5 levers: lexer-native output representation
    output_tokenizer: str = "compositional"
    use_symbol_table: bool = True
    factorized_embeddings: bool = False
    mask_pattern: str = "random"
    remask_span: str = "token"
    teacher_init_embeddings: bool = False
    # V6 levers: CoRe remask, T2M, slot-aware trust, grammar_diffusion
    remask_policy: str = "confidence"
    core_perturb_frac: float = 0.25
    remask_to_mask: bool = True
    slot_aware_trust_gate: bool = False
    model_name: str = "twotower"
    # V10 (B4): scratch | hf — AR→masked-denoiser adaptation backbone.
    denoiser_backend: str = "scratch"
    # V10 (C1): absolute | relative (De Bruijn binder references).
    bind_encoding: str = "absolute"
    # V12 (A2): ASAp-style distribution-aware constrained MaskGIT decode.
    asap_decode: bool = False
    # C3 (SLM-27): corpus-mined macro tokens with deterministic expansion.
    macro_tokens: bool = False
    # C4 (SLM-28): False = surface binder/state identifiers (byte channel).
    symbol_anonymization: bool = True
    # C4 pair runs decode unconstrained in BOTH arms (the NAME gate is
    # BIND-only, so constrained decode cannot emit surface identifiers).
    grammar_constrained: bool = True
    # V7 levers: speculative denoising (docs/design/speculative-denoising.md)
    stability_min_persistence: int = 0
    stability_jsd_weight: float = 1.0
    unmask_mode: str = "positions"
    cluster_attn_threshold: float = 0.08
    cluster_max_size: int = 4
    cluster_verify: bool = False
    # True = train survival head stage after SFT and use it at decode (E73).
    survival_gate: bool = False
    survival_commit_threshold: float = 0.3
    speculative_successor: bool = False
    speculative_fanout: int = 2
    speculative_overlap: bool = False
    # V8 dynamic-symbol / constraint-system levers.
    runtime_symbol_features: str = "none"
    symbol_slot_augmentation: bool = False
    semantic_candidate_masks: bool = False
    constraint_graph_mode: str = "off"
    grammar_completion_bounds: bool = False
    grammar_equivalence_cache: bool = False
    grammar_active_symbol_bitsets: bool = False
    compact_active_canvas: bool = True
    compiler_decode_mode: str = "off"
    compiler_search_mode: str = "greedy"
    compiler_search_trigger: str = "stagnation"
    compiler_search_width: int = 1
    compiler_search_noise: float = 0.0
    compiler_search_stagnation_patience: int = 2
    compiler_search_backtrack_limit: int = 8
    compiler_search_local_nogoods: bool = False
    grammar_finalize_validate: bool = False
    allow_unconstrained_fallback: bool = True
    component_inventory_loss_weight: float = 0.0
    component_inventory_decode_weight: float = 0.0
    component_plan_loss_weight: float = 0.0
    component_plan_decode_weight: float = 0.0
    component_edge_loss_weight: float = 0.0
    component_edge_alignment_loss_weight: float = 0.0
    component_edge_decode_weight: float = 0.0
    binder_component_plan_loss_weight: float = 0.0
    binder_component_plan_decode_weight: float = 0.0
    binder_topology_loss_weight: float = 0.0
    binder_topology_decode_weight: float = 0.0
    # V10 exact-state local preference process.
    local_parent_control: bool = False
    local_preference_objective: Literal[
        "ce_margin", "unlikelihood", "ftpo_single", "ftpo_set"
    ] | None = None
    local_preference_reference_tether: bool = False
    local_preference_balanced: bool = False
    local_preference_guarded_selection: bool = False
    local_preference_guarded_updates: bool = False
    local_preference_guard_backtrack_steps: int = 4
    local_preference_guard_by_decision_kind: bool = False
    local_preference_block_by_decision_kind: bool = False
    local_preference_gradient_combination: Literal["proposal", "pcgrad", "mgda"] = "proposal"
    local_preference_optimizer: Literal["adamw", "sgd"] = "adamw"
    binder_arity_loss_weight: float = 0.0
    binder_arity_decode_weight: float = 0.0


def _base_experiments(
    train_v1: Path,
    train_cur: Path,
    *,
    seed_checkpoint: Path | None = None,
    design_md_in_context: bool = True,
) -> list[Experiment]:
    seed = str(seed_checkpoint) if seed_checkpoint else None
    # Decode-only overlays must match the seed checkpoint's conditioning.
    seed_dm = False if seed else design_md_in_context
    return [
        Experiment(
            "E0",
            "qx_e0_baseline",
            "Ship-recipe baseline (v1, LTR primary, scratch)",
            train_v1,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E1",
            "qx_e1_repair",
            "Constrained LTR repair at eval (decode lever)",
            train_v1,
            grammar_ltr_repair=True,
            design_md_in_context=seed_dm,
            eval_from_run="qx_e0_baseline" if seed is None else None,
            seed_checkpoint=seed,
        ),
        Experiment(
            "E2",
            "qx_e2_curriculum",
            "Curriculum soft-mix A/B/C sampling",
            train_cur,
            use_curriculum=True,
            mix_curriculum=True,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E3",
            "qx_e3_fidelity",
            "Fidelity aux loss on placeholder tokens",
            train_v1,
            fidelity_loss_weight=1.0,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E4",
            "qx_e4_schema",
            "Schema-conditioned context",
            train_v1,
            schema_in_context=True,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E5",
            "qx_e5_pref_bon",
            "Soft preference pairs + best-of-N decode",
            train_v1,
            best_of_n=4,
            preference=True,
            design_md_in_context=seed_dm,
            eval_from_run="qx_e0_baseline" if seed is None else None,
            seed_checkpoint=seed,
        ),
        Experiment(
            "E6",
            "qx_e6_retrieve",
            "Retrieval skeleton bank (k=1)",
            train_v1,
            retrieval_k=1,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E7",
            "qx_e7_capacity",
            "Capacity upgrade (wider/deeper + longer LTR)",
            train_v1,
            d_model=192,
            n_heads=6,
            context_layers=3,
            denoiser_layers=6,
            grammar_ltr_max_tokens=192,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E8",
            "qx_e8_combo",
            "Stacked combo of all levers",
            train_cur,
            fidelity_loss_weight=1.0,
            schema_in_context=True,
            retrieval_k=1,
            best_of_n=4,
            use_curriculum=True,
            mix_curriculum=True,
            grammar_ltr_repair=True,
            d_model=192,
            n_heads=6,
            context_layers=3,
            denoiser_layers=6,
            grammar_ltr_max_tokens=192,
            preference=True,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E9b",
            "qx_e9b_fidelity_antileak",
            "Fidelity + soft curriculum mix + schema + LTR repair (anti-leak)",
            train_cur,
            fidelity_loss_weight=1.5,
            schema_in_context=True,
            use_curriculum=True,
            mix_curriculum=True,
            grammar_ltr_repair=True,
            best_of_n=4,
            d_model=192,
            n_heads=6,
            context_layers=3,
            denoiser_layers=6,
            grammar_ltr_max_tokens=192,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E10",
            "qx_e10_grpo",
            "Preference soft-pairs then GRPO-lite RL (seeds from E9b when present)",
            train_cur,
            fidelity_loss_weight=1.0,
            schema_in_context=True,
            use_curriculum=True,
            mix_curriculum=True,
            grammar_ltr_repair=True,
            best_of_n=4,
            preference=True,
            rl=True,
            eval_from_run="qx_e9b_fidelity_antileak",
            design_md_in_context=design_md_in_context,
        ),
    ]


def _v2_experiments(
    train_v1: Path,
    train_cur: Path,
    train_ns: Path,
    *,
    design_md_in_context: bool = True,
) -> list[Experiment]:
    """E11–E17: fixes for compositional placeholders + slot contract + LTR."""
    return [
        Experiment(
            "E11",
            "qx_e11_compositional_tok",
            "Compositional placeholder tokenizer (F1) baseline",
            train_v1,
            grammar_ltr_repair=True,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E12",
            "qx_e12_slot_contract",
            "F1 + slot contract conditioning + inventory decode (F2)",
            train_v1,
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            grammar_ltr_repair=True,
            fidelity_loss_weight=1.0,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E13",
            "qx_e13_ltr_aligned",
            "F1 + true weighted LTR loss + LTR-primary decode (F4)",
            train_v1,
            grammar_ltr_repair=True,
            ltr_loss_weight=2.0,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E14",
            "qx_e14_namespace_aug",
            "F1 + namespace augmentation, no slot contract (F5)",
            train_ns,
            grammar_ltr_repair=True,
            namespace_augment=True,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E15",
            "qx_e15_combo",
            "Combo: slot contract + LTR + leak-free curriculum + capacity",
            train_cur,
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            fidelity_loss_weight=1.5,
            schema_in_context=True,
            use_curriculum=True,
            mix_curriculum=True,
            grammar_ltr_repair=True,
            ltr_loss_weight=2.0,
            d_model=192,
            n_heads=6,
            context_layers=3,
            denoiser_layers=6,
            grammar_ltr_max_tokens=192,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E16",
            "qx_e16_long_train",
            "E15 recipe at extended step budget (use --steps 2000+)",
            train_cur,
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            fidelity_loss_weight=1.5,
            schema_in_context=True,
            use_curriculum=True,
            mix_curriculum=True,
            grammar_ltr_repair=True,
            ltr_loss_weight=2.0,
            d_model=192,
            n_heads=6,
            context_layers=3,
            denoiser_layers=6,
            grammar_ltr_max_tokens=192,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E17",
            "qx_e17_decode_sweep",
            "Decode-budget sweep on E15 checkpoint (eval-only)",
            train_cur,
            eval_from_run="qx_e15_combo",
            decode_sweep="gen16_repair_bon4",
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            grammar_ltr_repair=True,
            best_of_n=4,
            design_md_in_context=design_md_in_context,
        ),
    ]


def _v3_experiments(
    train_v1: Path,
    train_cur: Path,
    *,
    design_md_in_context: bool = True,
) -> list[Experiment]:
    """E18–E29: length-safe decode, train/infer match, template fill, MDLM, remask."""
    length_safe = 192
    capacity = dict(
        d_model=192,
        n_heads=6,
        context_layers=3,
        denoiser_layers=6,
        grammar_ltr_max_tokens=256,
    )
    return [
        Experiment(
            "E18",
            "qx_e18_length_safe",
            "Length-safe LTR budgets for compositional tokenizer",
            train_v1,
            grammar_ltr_repair=True,
            grammar_ltr_max_tokens=length_safe,
            grammar_ltr_primary=True,
            ltr_loss_weight=2.0,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E19a",
            "qx_e19a_maskgit_primary",
            "MaskGIT-primary decode matched to MaskGIT train (no LTR-primary)",
            train_v1,
            grammar_ltr_repair=True,
            grammar_ltr_max_tokens=length_safe,
            grammar_ltr_primary=False,
            ltr_loss_weight=0.5,
            gen_steps_override=16,
            remask_ratio=0.0,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E19b",
            "qx_e19b_ltr_matched",
            "LTR-primary decode with strong LTR loss + length-safe budget",
            train_v1,
            grammar_ltr_repair=True,
            grammar_ltr_max_tokens=length_safe,
            grammar_ltr_primary=True,
            ltr_loss_weight=2.5,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E20",
            "qx_e20_template_fill",
            "Slot-contract template-fill decode (inventory-bound skeleton)",
            train_v1,
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            template_fill_decode=True,
            grammar_ltr_repair=True,
            grammar_ltr_max_tokens=length_safe,
            grammar_ltr_primary=False,
            fidelity_loss_weight=1.0,
            gen_steps_override=12,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E21",
            "qx_e21_mdlm_schedule",
            "MDLM-faithful continuous-time mask ELBO + length-safe LTR",
            train_v1,
            grammar_ltr_repair=True,
            grammar_ltr_max_tokens=length_safe,
            grammar_ltr_primary=True,
            ltr_loss_weight=2.0,
            mdlm_schedule=True,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E22",
            "qx_e22_remask",
            "MaskGIT + confidence remasking (GIDD/ReMDM-lite)",
            train_v1,
            grammar_ltr_repair=True,
            grammar_ltr_max_tokens=length_safe,
            grammar_ltr_primary=False,
            remask_ratio=0.15,
            gen_steps_override=16,
            mdlm_schedule=True,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E29",
            "qx_e29_champion",
            "Champion: length-safe + slot contract + template + MDLM + remask + capacity",
            train_cur,
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            template_fill_decode=True,
            fidelity_loss_weight=1.5,
            schema_in_context=True,
            use_curriculum=True,
            mix_curriculum=True,
            grammar_ltr_repair=True,
            grammar_ltr_primary=False,
            ltr_loss_weight=2.0,
            mdlm_schedule=True,
            remask_ratio=0.12,
            gen_steps_override=16,
            best_of_n=4,
            design_md_in_context=design_md_in_context,
            **capacity,
        ),
    ]


def _v4_experiments(
    train_v1: Path,
    train_cur: Path,
    *,
    design_md_in_context: bool = True,
    seed_checkpoint: Path | None = None,
) -> list[Experiment]:
    """E30–E36: critic-guided revision + honest slot contract."""
    length_safe = 192
    capacity = dict(
        d_model=192,
        n_heads=6,
        context_layers=3,
        denoiser_layers=6,
        grammar_ltr_max_tokens=256,
    )
    seed = str(seed_checkpoint) if seed_checkpoint else None
    champion_base = dict(
        slot_contract_in_context=True,
        slot_contract_constrained_decode=True,
        template_fill_decode=True,
        fidelity_loss_weight=1.5,
        schema_in_context=True,
        grammar_ltr_repair=True,
        grammar_ltr_primary=False,
        ltr_loss_weight=2.0,
        mdlm_schedule=True,
        remask_ratio=0.12,
        gen_steps_override=16,
        design_md_in_context=design_md_in_context,
        **capacity,
    )
    return [
        Experiment(
            "E30",
            "qx_e30_suffix_rollback",
            "ReMDM-style LTR suffix rollback (inference-only revisable window)",
            train_v1,
            grammar_ltr_repair=True,
            grammar_ltr_max_tokens=length_safe,
            grammar_ltr_primary=True,
            ltr_loss_weight=2.0,
            suffix_rollback_window=8,
            seed_checkpoint=seed,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E31",
            "qx_e31_trust_gate",
            "BackPlay-lite FastPathGate on frozen denoiser errors",
            train_v1,
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            template_fill_decode=True,
            grammar_ltr_primary=False,
            remask_ratio=0.15,
            remask_use_gate=True,
            trust_gate=True,
            gen_steps_override=16,
            seed_checkpoint=seed,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E32",
            "qx_e32_visible_corrupt",
            "GIDD/RemeDi-lite visible-token corruption aux",
            train_v1,
            grammar_ltr_repair=True,
            grammar_ltr_max_tokens=length_safe,
            grammar_ltr_primary=False,
            mdlm_schedule=True,
            visible_corrupt_rate=0.08,
            remask_ratio=0.12,
            gen_steps_override=16,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E33",
            "qx_e33_remask_policy",
            "Combined remask: grammar + gate + entropy budget",
            train_v1,
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            template_fill_decode=True,
            grammar_ltr_primary=False,
            remask_ratio=0.15,
            remask_use_gate=True,
            remask_use_entropy=True,
            trust_gate=True,
            gen_steps_override=16,
            seed_checkpoint=seed,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E35",
            "qx_e35_honest_contract",
            "Honest inventory-in-prompt slot contract (no silent gold leakage)",
            train_cur if train_cur.exists() else train_v1,
            honest_slot_contract=True,
            use_curriculum=train_cur.exists(),
            mix_curriculum=True,
            best_of_n=4,
            **champion_base,
        ),
        Experiment(
            "E36",
            "qx_e36_decode_scaling",
            "Decode-time BoN + remask scaling sweep on E35/E29 recipe",
            train_v1,
            eval_from_run="qx_e35_honest_contract",
            decode_sweep="gen16_repair_bon4",
            honest_slot_contract=True,
            remask_ratio=0.2,
            remask_use_entropy=True,
            best_of_n=4,
            gen_steps_override=16,
            design_md_in_context=design_md_in_context,
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            template_fill_decode=True,
            grammar_ltr_primary=False,
            grammar_ltr_repair=True,
        ),
        Experiment(
            "E34",
            "qx_e34_latent_critics",
            "Deferred latent falsification MoE (placeholder row — skipped)",
            train_v1,
            design_md_in_context=design_md_in_context,
            # Marked via run_id; runner skips unless forced.
            seed_checkpoint=seed,
        ),
    ]


def _v5_experiments(
    train_v1: Path,
    train_cur: Path,
    *,
    design_md_in_context: bool = True,
) -> list[Experiment]:
    """E40–E46: lexer-native output vocabulary + Stage-2 structural levers."""
    length_safe = 192
    capacity = dict(
        d_model=192,
        n_heads=6,
        context_layers=3,
        denoiser_layers=6,
        grammar_ltr_max_tokens=256,
    )
    return [
        Experiment(
            "E40",
            "qx_e40_lexnative",
            "Lexer-native tokenizer without symbol table (literal channel)",
            train_v1,
            grammar_ltr_repair=True,
            grammar_ltr_max_tokens=length_safe,
            grammar_ltr_primary=False,
            ltr_loss_weight=0.5,
            gen_steps_override=16,
            remask_ratio=0.0,
            output_tokenizer="lexer",
            use_symbol_table=False,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E41",
            "qx_e41_symtable",
            "Lexer-native + dynamic symbol table for placeholders",
            train_v1,
            grammar_ltr_repair=True,
            grammar_ltr_max_tokens=length_safe,
            grammar_ltr_primary=False,
            ltr_loss_weight=0.5,
            fidelity_loss_weight=1.0,
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            gen_steps_override=16,
            output_tokenizer="lexer",
            use_symbol_table=True,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E42",
            "qx_e42_factorized",
            "E41 + kind-factorized embeddings",
            train_v1,
            grammar_ltr_repair=True,
            grammar_ltr_max_tokens=length_safe,
            grammar_ltr_primary=False,
            ltr_loss_weight=0.5,
            fidelity_loss_weight=1.0,
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            gen_steps_override=16,
            output_tokenizer="lexer",
            use_symbol_table=True,
            factorized_embeddings=True,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E43",
            "qx_e43_exact_masks",
            "Eval-only exact terminal→id masks on E41 checkpoint",
            train_v1,
            eval_from_run="qx_e41_symtable",
            grammar_ltr_repair=True,
            grammar_ltr_max_tokens=length_safe,
            grammar_ltr_primary=False,
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            gen_steps_override=16,
            output_tokenizer="lexer",
            use_symbol_table=True,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E47",
            "qx_e47_ltr_supervision",
            "E41 + doubled lexer-native LTR supervision",
            train_v1,
            grammar_ltr_repair=True,
            grammar_ltr_max_tokens=length_safe,
            grammar_ltr_primary=False,
            ltr_loss_weight=2.0,
            fidelity_loss_weight=1.0,
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            gen_steps_override=16,
            output_tokenizer="lexer",
            use_symbol_table=True,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E44",
            "qx_e44_structmask",
            "E41 + mixed statement masking + statement-span remask",
            train_v1,
            grammar_ltr_repair=True,
            grammar_ltr_max_tokens=length_safe,
            grammar_ltr_primary=False,
            ltr_loss_weight=0.5,
            fidelity_loss_weight=1.0,
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            remask_ratio=0.15,
            gen_steps_override=16,
            mdlm_schedule=True,
            output_tokenizer="lexer",
            use_symbol_table=True,
            mask_pattern="mixed",
            remask_span="statement",
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E45",
            "qx_e45_teacher_init",
            "E41 + teacher-initialized symbol embeddings (HF-cache gated)",
            train_v1,
            grammar_ltr_repair=True,
            grammar_ltr_max_tokens=length_safe,
            grammar_ltr_primary=False,
            ltr_loss_weight=0.5,
            fidelity_loss_weight=1.0,
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            gen_steps_override=16,
            output_tokenizer="lexer",
            use_symbol_table=True,
            teacher_init_embeddings=True,
            design_md_in_context=design_md_in_context,
        ),
        Experiment(
            "E46",
            "qx_e46_champion",
            "V5 champion: lexer+symtable+factorized+structmask+template+MDLM",
            train_cur,
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            template_fill_decode=True,
            fidelity_loss_weight=1.5,
            schema_in_context=True,
            use_curriculum=True,
            mix_curriculum=True,
            grammar_ltr_repair=True,
            grammar_ltr_primary=False,
            ltr_loss_weight=2.0,
            mdlm_schedule=True,
            remask_ratio=0.12,
            gen_steps_override=16,
            best_of_n=4,
            output_tokenizer="lexer",
            use_symbol_table=True,
            factorized_embeddings=True,
            mask_pattern="mixed",
            remask_span="statement",
            design_md_in_context=design_md_in_context,
            **capacity,
        ),
    ]


def _v6_experiments(
    train_v1: Path,
    train_cur: Path,
    *,
    design_md_in_context: bool = True,
    seed_checkpoint: Path | str | None = None,
) -> list[Experiment]:
    """E50–E55: CoRe remask, T2M, slot-aware trust, stacked honest V5 champion."""
    capacity = dict(
        d_model=192,
        n_heads=6,
        context_layers=3,
        denoiser_layers=6,
        grammar_ltr_max_tokens=256,
    )
    seed = str(seed_checkpoint) if seed_checkpoint else None
    v5_base = dict(
        slot_contract_in_context=True,
        slot_contract_constrained_decode=True,
        fidelity_loss_weight=1.5,
        schema_in_context=True,
        grammar_ltr_repair=True,
        grammar_ltr_primary=False,
        ltr_loss_weight=2.0,
        mdlm_schedule=True,
        remask_ratio=0.12,
        gen_steps_override=16,
        output_tokenizer="lexer",
        use_symbol_table=True,
        factorized_embeddings=True,
        mask_pattern="mixed",
        remask_span="statement",
        remask_to_mask=True,
        design_md_in_context=design_md_in_context,
        **capacity,
    )
    return [
        Experiment(
            "E50",
            "qx_e50_core_remask",
            "CoRe-lite context-robust remask (training-free) on V5 alphabet",
            train_v1,
            template_fill_decode=True,
            honest_slot_contract=True,
            remask_policy="core",
            core_perturb_frac=0.25,
            best_of_n=1,
            seed_checkpoint=seed,
            **v5_base,
        ),
        Experiment(
            "E51",
            "qx_e51_t2m_statement",
            "T2M remask-to-mask + statement-span remask discipline",
            train_v1,
            template_fill_decode=True,
            honest_slot_contract=True,
            remask_policy="confidence",
            best_of_n=1,
            seed_checkpoint=seed,
            **{
                **v5_base,
                "remask_ratio": 0.15,
                "remask_span": "statement",
                "remask_to_mask": True,
            },
        ),
        Experiment(
            "E52",
            "qx_e52_slot_trust",
            "Slot-aware FastPathGate trust head (placeholder binding errors)",
            train_v1,
            template_fill_decode=True,
            honest_slot_contract=True,
            remask_use_gate=True,
            remask_use_entropy=True,
            remask_policy="combined",
            trust_gate=True,
            slot_aware_trust_gate=True,
            best_of_n=1,
            seed_checkpoint=seed,
            **v5_base,
        ),
        Experiment(
            "E53",
            "qx_e53_honest_v5_champion",
            "Stacked honest V5 champion: E46+E35+E33+E50 CoRe remask",
            train_cur,
            template_fill_decode=True,
            honest_slot_contract=True,
            use_curriculum=True,
            mix_curriculum=True,
            remask_policy="combined",
            remask_use_gate=True,
            remask_use_entropy=True,
            core_perturb_frac=0.25,
            best_of_n=4,
            trust_gate=True,
            slot_aware_trust_gate=True,
            **v5_base,
        ),
        Experiment(
            "E54",
            "qx_e54_grammar_honest",
            "Grammar-diffusion with honest inventory-in-prompt (no gold channel)",
            train_v1,
            model_name="grammar_diffusion",
            slot_contract_in_context=True,
            slot_contract_constrained_decode=True,
            honest_slot_contract=True,
            fidelity_loss_weight=1.5,
            grammar_ltr_repair=True,
            grammar_ltr_primary=True,
            gen_steps_override=16,
            design_md_in_context=design_md_in_context,
            d_model=128,
            n_heads=4,
            context_layers=2,
            denoiser_layers=4,
            grammar_ltr_max_tokens=128,
        ),
        Experiment(
            "E55",
            "qx_e55_process",
            "Process stage on E53: preference + GRPO-lite (skip RL if no variance)",
            train_cur,
            template_fill_decode=True,
            honest_slot_contract=True,
            use_curriculum=True,
            mix_curriculum=True,
            remask_policy="combined",
            remask_use_gate=True,
            remask_use_entropy=True,
            core_perturb_frac=0.25,
            best_of_n=4,
            trust_gate=True,
            slot_aware_trust_gate=True,
            preference=True,
            rl=True,
            eval_from_run="qx_e53_honest_v5_champion",
            **v5_base,
        ),
    ]


def _v7_experiments(
    train_v1: Path,
    train_cur: Path,
    *,
    design_md_in_context: bool = True,
    seed_checkpoint: Path | str | None = None,
) -> list[Experiment]:
    """E70–E75: speculative denoising (stability, clusters, ordered verify,
    survival head, successor cache) — docs/design/speculative-denoising.md."""
    capacity = dict(
        d_model=192,
        n_heads=6,
        context_layers=3,
        denoiser_layers=6,
        grammar_ltr_max_tokens=256,
    )
    seed = str(seed_checkpoint) if seed_checkpoint else None
    v5_base = dict(
        slot_contract_in_context=True,
        slot_contract_constrained_decode=True,
        fidelity_loss_weight=1.5,
        schema_in_context=True,
        grammar_ltr_repair=True,
        grammar_ltr_primary=False,
        ltr_loss_weight=2.0,
        mdlm_schedule=True,
        remask_ratio=0.12,
        gen_steps_override=16,
        output_tokenizer="lexer",
        use_symbol_table=True,
        factorized_embeddings=True,
        mask_pattern="mixed",
        remask_span="statement",
        remask_to_mask=True,
        template_fill_decode=True,
        honest_slot_contract=True,
        design_md_in_context=design_md_in_context,
        **capacity,
    )
    return [
        Experiment(
            "E70",
            "qx_e70_stability",
            "LESS-lite stability remask + persistence commit gate",
            train_v1,
            remask_policy="stability",
            stability_min_persistence=1,
            stability_jsd_weight=1.0,
            best_of_n=1,
            seed_checkpoint=seed,
            **v5_base,
        ),
        Experiment(
            "E71",
            "qx_e71_clusters",
            "Attention dependency clusters (DAPD/DAWN-lite), anchor-first",
            train_v1,
            unmask_mode="cluster",
            best_of_n=1,
            seed_checkpoint=seed,
            **v5_base,
        ),
        Experiment(
            "E72",
            "qx_e72_cluster_verify",
            "Ordered cluster verification: outcome (j, repair)",
            train_v1,
            unmask_mode="cluster",
            cluster_verify=True,
            best_of_n=1,
            seed_checkpoint=seed,
            **v5_base,
        ),
        Experiment(
            "E73",
            "qx_e73_survival",
            "DSpark-lite survival head + cumulative commit budget",
            train_v1,
            unmask_mode="cluster",
            cluster_verify=True,
            survival_gate=True,
            best_of_n=1,
            seed_checkpoint=seed,
            **v5_base,
        ),
        Experiment(
            "E74",
            "qx_e74_successor",
            "Saguaro-SSD-lite successor cache (K=2 outcome fanout)",
            train_v1,
            unmask_mode="cluster",
            cluster_verify=True,
            speculative_successor=True,
            speculative_fanout=2,
            best_of_n=1,
            seed_checkpoint=seed,
            **v5_base,
        ),
        Experiment(
            "E75",
            "qx_e75_v7_champion",
            "V7 champion: E53 honest stack + stability + clusters + survival + successor cache",
            train_cur,
            use_curriculum=True,
            mix_curriculum=True,
            remask_policy="stability",
            stability_min_persistence=1,
            remask_use_gate=True,
            remask_use_entropy=True,
            unmask_mode="cluster",
            cluster_verify=True,
            survival_gate=True,
            speculative_successor=True,
            speculative_fanout=2,
            trust_gate=True,
            slot_aware_trust_gate=True,
            best_of_n=4,
            **v5_base,
        ),
        Experiment(
            "E76",
            "qx_e76_cache_reuse",
            "V7 champion without trust/entropy remask gates to measure successor reuse",
            train_cur,
            use_curriculum=True,
            mix_curriculum=True,
            remask_policy="stability",
            stability_min_persistence=1,
            unmask_mode="cluster",
            cluster_verify=True,
            survival_gate=True,
            speculative_successor=True,
            speculative_fanout=2,
            best_of_n=1,
            **v5_base,
        ),
        Experiment(
            "E77",
            "qx_e77_cache_bon4",
            "E76 corrected V7 cache path with champion best-of-4 selection",
            train_cur,
            use_curriculum=True,
            mix_curriculum=True,
            remask_policy="stability",
            stability_min_persistence=1,
            unmask_mode="cluster",
            cluster_verify=True,
            survival_gate=True,
            speculative_successor=True,
            speculative_fanout=2,
            best_of_n=4,
            **v5_base,
        ),
        Experiment(
            "E78",
            "qx_e78_slot_supervision_cache",
            "E76 corrected cache path with E75 trust and slot-aware supervision",
            train_cur,
            use_curriculum=True,
            mix_curriculum=True,
            remask_policy="stability",
            stability_min_persistence=1,
            unmask_mode="cluster",
            cluster_verify=True,
            survival_gate=True,
            speculative_successor=True,
            speculative_fanout=2,
            trust_gate=True,
            slot_aware_trust_gate=True,
            best_of_n=4,
            **v5_base,
        ),
        Experiment(
            "E80",
            "qx_e80_visible_contract",
            "E77 cache path retrained on prompts with visible slot contracts",
            train_cur,
            use_curriculum=True,
            mix_curriculum=True,
            remask_policy="stability",
            stability_min_persistence=1,
            unmask_mode="cluster",
            cluster_verify=True,
            survival_gate=True,
            speculative_successor=True,
            speculative_fanout=2,
            best_of_n=4,
            **v5_base,
        ),
        Experiment(
            "E82",
            "qx_e82_contract_template_fastpath",
            "E80 with certified contract-template fast path for latency upper bound",
            train_cur,
            use_curriculum=True,
            mix_curriculum=True,
            remask_policy="stability",
            stability_min_persistence=1,
            unmask_mode="cluster",
            cluster_verify=True,
            survival_gate=True,
            speculative_successor=True,
            speculative_fanout=2,
            best_of_n=1,
            contract_template_fastpath=True,
            **v5_base,
        ),
        Experiment(
            "E84",
            "qx_e84_ltr_primary_contract",
            "Visible-contract corpus with grammar LTR primary and learned decode",
            train_cur,
            use_curriculum=True,
            mix_curriculum=True,
            remask_policy="stability",
            stability_min_persistence=1,
            unmask_mode="cluster",
            cluster_verify=True,
            survival_gate=True,
            speculative_successor=True,
            speculative_fanout=2,
            best_of_n=1,
            contract_template_fastpath=False,
            **{**v5_base, "grammar_ltr_primary": True, "grammar_ltr_repair": True},
        ),
        Experiment(
            "E88",
            "qx_e88_structural_supervision_contract",
            "Visible-contract corpus with LTR, fidelity, and symbol-boundary supervision",
            train_cur,
            use_curriculum=True,
            mix_curriculum=True,
            remask_policy="stability",
            stability_min_persistence=1,
            best_of_n=1,
            contract_template_fastpath=False,
            symbol_boundary_loss_weight=2.0,
            **{**v5_base, "grammar_ltr_primary": True, "grammar_ltr_repair": True, "fidelity_loss_weight": 4.0},
        ),
    ]


def _v8_experiments(
    train_dir: Path,
    *,
    design_md_in_context: bool = True,
) -> list[Experiment]:
    """E200-E207: request-conditioned symbols and constraint-aware diffusion."""
    base = dict(
        output_tokenizer="lexer",
        use_symbol_table=True,
        factorized_embeddings=True,
        mask_pattern="diffusion",
        slot_contract_in_context=True,
        slot_contract_constrained_decode=True,
        honest_slot_contract=True,
        grammar_ltr_repair=True,
        grammar_ltr_primary=False,
        unmask_mode="cluster",
        cluster_verify=True,
        design_md_in_context=design_md_in_context,
    )
    return [
        Experiment("E200", "qx_e200_symbol_control", "Current fixed-row DSL symbol control", train_dir, **base),
        Experiment("E201", "qx_e201_alpha_shuffle", "Slot permutation and alpha-renaming augmentation", train_dir, symbol_slot_augmentation=True, **base),
        Experiment("E202", "qx_e202_surface_symbols", "Request-conditioned surface features for all symbol roles", train_dir, runtime_symbol_features="surface", **base),
        Experiment("E203", "qx_e203_role_gated", "Binder-invariant, entity/state-aware symbol features", train_dir, runtime_symbol_features="role_gated", **base),
        Experiment("E204", "qx_e204_semantic_masks", "Role-gated features plus active semantic candidate masks", train_dir, runtime_symbol_features="role_gated", semantic_candidate_masks=True, **base),
        Experiment("E205", "qx_e205_constraint_graph", "Hybrid grammar/attention constraint graph scheduling", train_dir, runtime_symbol_features="role_gated", semantic_candidate_masks=True, constraint_graph_mode="hybrid", **base),
        Experiment("E206", "qx_e206_fixed_canvas", "Complete V8 stack on fixed padded canvases", train_dir, runtime_symbol_features="role_gated", semantic_candidate_masks=True, constraint_graph_mode="hybrid", grammar_completion_bounds=True, grammar_equivalence_cache=True, grammar_active_symbol_bitsets=True, compact_active_canvas=False, **base),
        Experiment("E207", "qx_e207_compact_canvas", "Complete V8 stack on compact active canvases", train_dir, runtime_symbol_features="role_gated", semantic_candidate_masks=True, constraint_graph_mode="hybrid", grammar_completion_bounds=True, grammar_equivalence_cache=True, grammar_active_symbol_bitsets=True, compact_active_canvas=True, **base),
    ]


def _strict_compiler_tree_policy() -> dict[str, Any]:
    """Canonical honest compiler-tree evaluation policy for matched campaigns."""
    runtime_fields = frozenset(
        {
            "compiler_search_backtrack_limit",
            "compiler_search_local_nogoods",
            "compiler_search_mode",
            "compiler_search_noise",
            "compiler_search_stagnation_patience",
            "compiler_search_trigger",
            "compiler_search_width",
            *STRICT_COMPILER_TREE_POLICY,
        }
    )
    return dict(
        runtime_override_fields=runtime_fields,
        **STRICT_COMPILER_TREE_POLICY,
    )


def _v9_experiments(train_dir: Path) -> list[Experiment]:
    """E240-E247: eval-only compiler-lattice search campaign."""
    base = dict(**_strict_compiler_tree_policy(), initialization="eval_only")
    return [
        Experiment("E240", "qx_e240_compiler_tree_control", "Corrected greedy compiler-tree control", train_dir, **base),
        Experiment("E241", "qx_e241_lattice_rollback", "Hard/soft lattice with bounded rollback", train_dir, compiler_search_mode="lattice", **base),
        Experiment("E242", "qx_e242_stagnation_nogood", "Stagnation-visible localized conflict nogoods", train_dir, compiler_search_mode="lattice", compiler_search_local_nogoods=True, compiler_search_stagnation_patience=1, **base),
        Experiment("E243", "qx_e243_ptrm_triggered_w4", "PTRM-style width 4 triggered by stagnation", train_dir, compiler_search_mode="ptrm", compiler_search_trigger="stagnation", compiler_search_width=4, compiler_search_noise=1.0, compiler_search_local_nogoods=True, **base),
        Experiment("E244", "qx_e244_ptrm_always_w4", "Always-on PTRM-style width 4 matched control", train_dir, compiler_search_mode="ptrm", compiler_search_trigger="always", compiler_search_width=4, compiler_search_noise=1.0, compiler_search_local_nogoods=True, **base),
        Experiment("E245", "qx_e245_gram_diverse_w4", "GRAM-style semantic diversity at width 4", train_dir, compiler_search_mode="gram", compiler_search_trigger="stagnation", compiler_search_width=4, compiler_search_noise=1.0, compiler_search_local_nogoods=True, **base),
        Experiment("E246", "qx_e246_lattice_full_w4", "Full lattice stack at width 4", train_dir, compiler_search_mode="gram", compiler_search_trigger="stagnation", compiler_search_width=4, compiler_search_noise=1.0, compiler_search_backtrack_limit=8, compiler_search_local_nogoods=True, **base),
        Experiment("E247", "qx_e247_lattice_full_w8", "Full lattice stack width 8 scaling row", train_dir, compiler_search_mode="gram", compiler_search_trigger="stagnation", compiler_search_width=8, compiler_search_noise=1.0, compiler_search_backtrack_limit=8, compiler_search_local_nogoods=True, **base),
    ]


def _v10_experiments(train_dir: Path) -> list[Experiment]:
    """Exact-state local preference campaign."""
    base = _strict_compiler_tree_policy()
    return [
        Experiment(
            "E248",
            "qx_e248_local_parent_control",
            "Unchanged parent control for exact-state preference",
            train_dir,
            local_parent_control=True,
            **base,
        ),
        Experiment(
            "E249",
            "qx_e249_local_ce_margin",
            "Exact-event compiler CE plus margin",
            train_dir,
            local_preference_objective="ce_margin",
            **base,
        ),
        Experiment(
            "E250",
            "qx_e250_local_unlikelihood",
            "Exact-event bad-token unlikelihood",
            train_dir,
            local_preference_objective="unlikelihood",
            **base,
        ),
        Experiment(
            "E251",
            "qx_e251_local_ftpo_single",
            "Single-good/single-bad clipped FTPO",
            train_dir,
            local_preference_objective="ftpo_single",
            **base,
        ),
        Experiment(
            "E252",
            "qx_e252_local_ftpo_set",
            "Verifier-backed set FTPO",
            train_dir,
            local_preference_objective="ftpo_set",
            **base,
        ),
        Experiment(
            "E253",
            "qx_e253_local_ftpo_tether",
            "Set FTPO with frozen-reference logit tether",
            train_dir,
            local_preference_objective="ftpo_set",
            local_preference_reference_tether=True,
            **base,
        ),
        Experiment(
            "E254",
            "qx_e254_local_ftpo_balanced",
            "Tethered set FTPO with balanced event sampling",
            train_dir,
            local_preference_objective="ftpo_set",
            local_preference_reference_tether=True,
            local_preference_balanced=True,
            **base,
        ),
        Experiment(
            "E277",
            "qx_e277_broad_gold_ast_ftpo_set",
            "Broad grammar/AST-aligned set FTPO",
            train_dir,
            local_preference_objective="ftpo_set",
            **base,
        ),
        Experiment(
            "E278",
            "qx_e278_guarded_gold_ast_ftpo_set",
            "Held-out Pareto-guarded gold-AST set FTPO",
            train_dir,
            local_preference_objective="ftpo_set",
            local_preference_guarded_selection=True,
            **base,
        ),
        Experiment(
            "E265",
            "qx_e265_safe_gold_ast_ftpo_set",
            "Pareto-safe backtracked gold-AST set FTPO updates",
            train_dir,
            local_preference_objective="ftpo_set",
            local_preference_guarded_updates=True,
            **base,
        ),
        Experiment(
            "E266",
            "qx_e266_stratified_safe_gold_ast_ftpo_set",
            "Decision-kind-stratified safe gold-AST set FTPO",
            train_dir,
            local_preference_objective="ftpo_set",
            local_preference_guarded_updates=True,
            local_preference_guard_by_decision_kind=True,
            **base,
        ),
        Experiment(
            "E267",
            "qx_e267_block_stratified_safe_gold_ast_ftpo_set",
            "Decision-kind block-coordinate stratified safe set FTPO",
            train_dir,
            local_preference_objective="ftpo_set",
            local_preference_guarded_updates=True,
            local_preference_guard_by_decision_kind=True,
            local_preference_block_by_decision_kind=True,
            **base,
        ),
        Experiment(
            "E268",
            "qx_e268_projected_stratified_safe_gold_ast_ftpo_set",
            "Conflict-projected decision-kind stratified safe set FTPO",
            train_dir,
            local_preference_objective="ftpo_set",
            local_preference_guarded_updates=True,
            local_preference_guard_by_decision_kind=True,
            local_preference_gradient_combination="pcgrad",
            **base,
        ),
        Experiment(
            "E269",
            "qx_e269_mgda_stratified_safe_gold_ast_ftpo_set",
            "Minimum-norm common-descent decision-kind safe set FTPO",
            train_dir,
            local_preference_objective="ftpo_set",
            local_preference_guarded_updates=True,
            local_preference_guard_by_decision_kind=True,
            local_preference_gradient_combination="mgda",
            **base,
        ),
        Experiment(
            "E272",
            "qx_e272_mgda_sgd_stratified_safe_gold_ast_ftpo_set",
            "Minimum-norm decision-kind safe set FTPO with collinear SGD",
            train_dir,
            local_preference_objective="ftpo_set",
            local_preference_guarded_updates=True,
            local_preference_guard_by_decision_kind=True,
            local_preference_gradient_combination="mgda",
            local_preference_optimizer="sgd",
            **base,
        ),
    ]


def _v11_experiments(train_dir: Path) -> list[Experiment]:
    """E255-E257: Track B/C representation baselines.

    E255/E256 (B4): matched pair differing only in the denoiser backbone —
    from-scratch DenoiserTower vs the pretrained hf_model_name causal LM
    adapted into a bidirectional masked denoiser. Parallel MaskGIT decode
    (not LTR) keeps the 135M-backbone eval tractable and identical across
    the pair. E257 (C1): scope-as-relative-index binder references
    (<BINDDEF>/<BINDREL_±k>), matched against E255 on everything but
    bind_encoding.
    """
    base = dict(
        output_tokenizer="lexer",
        mask_pattern="diffusion",
        grammar_ltr_primary=False,
    )
    return [
        Experiment("E255", "qx_e255_b4_scratch_control", "B4 matched from-scratch denoiser control", train_dir, **base),
        Experiment("E256", "qx_e256_b4_ar_adapt", "B4 DiffuLLaMA-style SmolLM2 AR-to-masked-denoiser adaptation", train_dir, denoiser_backend="hf", **base),
        Experiment("E257", "qx_e257_c1_relative_bind", "C1 De Bruijn relative binder references", train_dir, bind_encoding="relative", **base),
    ]


def _v12_experiments(train_dir: Path) -> list[Experiment]:
    """E262: B1 choice-sequence codec (pure grammar-choice output stream).

    Trains/decodes over the ``choice`` output tokenizer: the model predicts
    only semantic decisions (which production, which slot filler) and the
    deterministic detokenizer reconstructs all surface syntax through the
    official lang-core serializer (fail-closed, so parse is a meaningful
    primary — the detokenizer never invents syntax for an invalid stream).
    Matched against E255 (v11 lexer-stream scratch control): identical
    diffusion masking and non-LTR MaskGIT decode, differing only in the
    output representation. v1 bypasses the surface-DFA token gate (choice
    ids are not surface lexemes; follow-up is a choice-native legal-decision
    gate). E2 semantic-density gates for the representation itself are
    pinned in tests/test_dsl/test_choice_codec.py and measured in
    docs/design/iter-b1-choice-sequence-codec-20260717.md.
    """
    return [
        Experiment(
            "E262",
            "qx_e262_b1_choice_codec",
            "B1 pure grammar-choice output stream (choice tokenizer)",
            train_dir,
            output_tokenizer="choice",
            mask_pattern="diffusion",
            grammar_ltr_primary=False,
        ),
    ]


def _v14_experiments(train_dir: Path) -> list[Experiment]:
    """E277 (A2): ASAp-style distribution-aware constrained MaskGIT decode.

    Observed constraint violations remove the violating token's mass at that
    canvas position from the next proposal, and unmask ordering uses
    post-removal confidence. Decode-only, so eval-only: route through a frozen
    E255 checkpoint via ``--parent`` — a matched pair differing only in
    ``asap_decode``.
    """
    base = dict(
        output_tokenizer="lexer",
        mask_pattern="diffusion",
        grammar_ltr_primary=False,
        initialization="eval_only",
        runtime_override_fields=frozenset({"asap_decode", "grammar_ltr_primary"}),
    )
    return [
        Experiment("E277", "qx_e277_a2_asap_decode", "A2 ASAp distribution-aware constrained MaskGIT decode", train_dir, asap_decode=True, **base),
    ]


def _v15_experiments(train_dir: Path) -> list[Experiment]:
    """E278 (C2): dynamic pseudo-embeddings for symbol tokens (SLM-26).

    ``runtime_symbol_features="replace"`` cancels the learned symbol-pool row
    with a deterministic byte-compositional vector (DyVo-style; weight tying
    and batching untouched). Matched against E255 on everything but the mode.
    """
    base = dict(
        output_tokenizer="lexer",
        mask_pattern="diffusion",
        grammar_ltr_primary=False,
    )
    return [
        Experiment("E278", "qx_e278_c2_pseudo_embeddings", "C2 dynamic pseudo-embeddings for symbol tokens", train_dir, runtime_symbol_features="replace", **base),
    ]


def _v16_experiments(train_dir: Path) -> list[Experiment]:
    """E280 (C3, SLM-27): corpus-mined macro tokens, matched against E255."""
    base = dict(
        output_tokenizer="lexer",
        mask_pattern="diffusion",
        grammar_ltr_primary=False,
    )
    return [
        Experiment(
            "E280",
            "qx_e280_c3_macro_tokens",
            "C3 corpus-mined macro tokens with deterministic expansion",
            train_dir,
            macro_tokens=True,
            **base,
        ),
    ]


def _v17_experiments(train_dir: Path) -> list[Experiment]:
    """E281/E282 (C4, SLM-28): names-disappear matched pair.

    Both arms decode unconstrained (grammar_constrained=False) because the
    NAME gate admits only <BIND_j> ids — the surface arm could never emit a
    byte-spelled identifier under the gate, which would confound the
    representation lever with a decode-legality artifact.
    """
    base = dict(
        output_tokenizer="lexer",
        mask_pattern="diffusion",
        grammar_ltr_primary=False,
        grammar_constrained=False,
    )
    return [
        Experiment(
            "E281",
            "qx_e281_c4_anon_control",
            "C4 anonymized-symbol control (unconstrained decode)",
            train_dir,
            **base,
        ),
        Experiment(
            "E282",
            "qx_e282_c4_surface_ids",
            "C4 surface binder/state identifiers via byte channel",
            train_dir,
            symbol_anonymization=False,
            **base,
        ),
    ]


def _apply_eval_checkpoint(
    experiments: list[Experiment], eval_checkpoint: Path | None
) -> list[Experiment]:
    """Route declared eval-only rows (V9) through one frozen checkpoint.

    Rows registered with ``initialization="eval_only"`` compare decode-time
    policies and must share identical checkpoint lineage; without an explicit
    checkpoint source the classifier would silently retrain each row.
    """
    if eval_checkpoint is None:
        return experiments
    return [
        replace(exp, eval_from_checkpoint=str(eval_checkpoint))
        if exp.initialization == "eval_only"
        and not exp.eval_from_run
        and not exp.eval_from_checkpoint
        else exp
        for exp in experiments
    ]


def _train_cfg(exp: Experiment, args: argparse.Namespace) -> ModelBuildConfig:
    return ModelBuildConfig(
        train_dir=exp.train_dir,
        test_dir=args.test_dir,
        suite="smoke",
        run_root=args.run_root,
        run_id=exp.run_id,
        runtime_override_fields=exp.runtime_override_fields,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        model_name=str(getattr(exp, "model_name", "twotower") or "twotower"),
        d_model=exp.d_model,
        n_heads=exp.n_heads,
        context_layers=exp.context_layers,
        denoiser_layers=exp.denoiser_layers,
        context_backend=args.context_backend,
        local_files_only=args.local_files_only,
        denoiser_backend=str(getattr(exp, "denoiser_backend", "scratch") or "scratch"),
        bind_encoding=str(getattr(exp, "bind_encoding", "absolute") or "absolute"),
        asap_decode=bool(getattr(exp, "asap_decode", False)),
        macro_tokens=bool(getattr(exp, "macro_tokens", False)),
        symbol_anonymization=bool(getattr(exp, "symbol_anonymization", True)),
        grammar_constrained=bool(getattr(exp, "grammar_constrained", True)),
        grammar_ltr_primary=bool(getattr(exp, "grammar_ltr_primary", True)),
        grammar_ltr_repair=exp.grammar_ltr_repair,
        grammar_ltr_max_tokens=exp.grammar_ltr_max_tokens,
        design_md_in_context=exp.design_md_in_context,
        ltr_loss_weight=getattr(exp, "ltr_loss_weight", 1.0),
        ltr_prefix_loss_weight=getattr(exp, "ltr_prefix_loss_weight", 0.0),
        symbol_boundary_loss_weight=getattr(exp, "symbol_boundary_loss_weight", 0.0),
        fidelity_loss_weight=exp.fidelity_loss_weight,
        schema_in_context=exp.schema_in_context,
        slot_contract_in_context=getattr(exp, "slot_contract_in_context", False),
        slot_contract_constrained_decode=getattr(
            exp, "slot_contract_constrained_decode", False
        ),
        template_fill_decode=bool(getattr(exp, "template_fill_decode", False)),
        contract_template_fastpath=bool(getattr(exp, "contract_template_fastpath", False)),
        honest_slot_contract=bool(getattr(exp, "honest_slot_contract", False)),
        retrieval_k=exp.retrieval_k,
        best_of_n=1,  # train without BoN cost; apply at eval
        use_curriculum=exp.use_curriculum,
        mix_curriculum=getattr(exp, "mix_curriculum", True),
        use_compile=bool(getattr(args, "compile", False)),
        parallel_unmask=str(getattr(args, "parallel_unmask", "adaptive") or "adaptive"),
        remask_ratio=float(getattr(exp, "remask_ratio", 0.0) or 0.0),
        remask_use_gate=bool(getattr(exp, "remask_use_gate", False)),
        remask_use_entropy=bool(getattr(exp, "remask_use_entropy", False)),
        remask_policy=str(getattr(exp, "remask_policy", "confidence") or "confidence"),
        core_perturb_frac=float(getattr(exp, "core_perturb_frac", 0.25) or 0.25),
        remask_to_mask=bool(getattr(exp, "remask_to_mask", True)),
        slot_aware_trust_gate=bool(getattr(exp, "slot_aware_trust_gate", False)),
        mdlm_schedule=bool(getattr(exp, "mdlm_schedule", False)),
        visible_corrupt_rate=float(getattr(exp, "visible_corrupt_rate", 0.0) or 0.0),
        suffix_rollback_window=int(getattr(exp, "suffix_rollback_window", 0) or 0),
        grammar_fastpath_mode=str(
            getattr(exp, "grammar_fastpath_mode", "hybrid") or "hybrid"
        ),
        compiler_decode_mode=str(getattr(exp, "compiler_decode_mode", "off") or "off"),
        compiler_search_mode=str(getattr(exp, "compiler_search_mode", "greedy") or "greedy"),
        compiler_search_trigger=str(getattr(exp, "compiler_search_trigger", "stagnation") or "stagnation"),
        compiler_search_width=max(1, int(getattr(exp, "compiler_search_width", 1) or 1)),
        compiler_search_noise=max(0.0, float(getattr(exp, "compiler_search_noise", 0.0) or 0.0)),
        compiler_search_stagnation_patience=max(1, int(getattr(exp, "compiler_search_stagnation_patience", 2) or 2)),
        compiler_search_backtrack_limit=max(0, int(getattr(exp, "compiler_search_backtrack_limit", 8) or 0)),
        compiler_search_local_nogoods=bool(
            getattr(exp, "compiler_search_local_nogoods", False)
        ),
        grammar_finalize_validate=bool(
            getattr(exp, "grammar_finalize_validate", False)
        ),
        allow_unconstrained_fallback=bool(
            getattr(exp, "allow_unconstrained_fallback", True)
        ),
        component_inventory_loss_weight=float(
            getattr(exp, "component_inventory_loss_weight", 0.0) or 0.0
        ),
        component_inventory_decode_weight=float(
            getattr(exp, "component_inventory_decode_weight", 0.0) or 0.0
        ),
        component_plan_loss_weight=float(
            getattr(exp, "component_plan_loss_weight", 0.0) or 0.0
        ),
        component_plan_decode_weight=float(
            getattr(exp, "component_plan_decode_weight", 0.0) or 0.0
        ),
        component_edge_loss_weight=float(
            getattr(exp, "component_edge_loss_weight", 0.0) or 0.0
        ),
        component_edge_alignment_loss_weight=float(
            getattr(exp, "component_edge_alignment_loss_weight", 0.0) or 0.0
        ),
        component_edge_decode_weight=float(
            getattr(exp, "component_edge_decode_weight", 0.0) or 0.0
        ),
        binder_component_plan_loss_weight=float(
            getattr(exp, "binder_component_plan_loss_weight", 0.0) or 0.0
        ),
        binder_component_plan_decode_weight=float(
            getattr(exp, "binder_component_plan_decode_weight", 0.0) or 0.0
        ),
        binder_topology_loss_weight=float(
            getattr(exp, "binder_topology_loss_weight", 0.0) or 0.0
        ),
        binder_topology_decode_weight=float(
            getattr(exp, "binder_topology_decode_weight", 0.0) or 0.0
        ),
        binder_arity_loss_weight=float(
            getattr(exp, "binder_arity_loss_weight", 0.0) or 0.0
        ),
        binder_arity_decode_weight=float(
            getattr(exp, "binder_arity_decode_weight", 0.0) or 0.0
        ),
        trust_gate_train=bool(getattr(exp, "trust_gate", False)),
        grad_accum_steps=max(1, int(getattr(args, "grad_accum", 1) or 1)),
        eval_every=args.eval_every,
        eval_suite="smoke",
        eval_suites="smoke,held_out" if args.eval_every else "",
        structural_bias=2.5,
        telemetry=True,
        gen_steps=int(
            getattr(exp, "gen_steps_override", None)
            or getattr(args, "gen_steps", 8)
            or 8
        ),
        output_tokenizer=str(
            getattr(exp, "output_tokenizer", "compositional") or "compositional"
        ),
        use_symbol_table=bool(getattr(exp, "use_symbol_table", True)),
        factorized_embeddings=bool(getattr(exp, "factorized_embeddings", False)),
        mask_pattern=str(getattr(exp, "mask_pattern", "random") or "random"),
        remask_span=str(getattr(exp, "remask_span", "token") or "token"),
        teacher_init_embeddings=bool(getattr(exp, "teacher_init_embeddings", False)),
        runtime_symbol_features=str(
            getattr(exp, "runtime_symbol_features", "none") or "none"
        ),
        symbol_slot_augmentation=bool(
            getattr(exp, "symbol_slot_augmentation", False)
        ),
        semantic_candidate_masks=bool(
            getattr(exp, "semantic_candidate_masks", False)
        ),
        constraint_graph_mode=str(
            getattr(exp, "constraint_graph_mode", "off") or "off"
        ),
        grammar_completion_bounds=bool(
            getattr(exp, "grammar_completion_bounds", False)
        ),
        grammar_equivalence_cache=bool(
            getattr(exp, "grammar_equivalence_cache", False)
        ),
        grammar_active_symbol_bitsets=bool(
            getattr(exp, "grammar_active_symbol_bitsets", False)
        ),
        compact_active_canvas=bool(
            getattr(exp, "compact_active_canvas", True)
        ),
        stability_min_persistence=int(
            getattr(exp, "stability_min_persistence", 0) or 0
        ),
        stability_jsd_weight=float(getattr(exp, "stability_jsd_weight", 1.0) or 1.0),
        unmask_mode=str(getattr(exp, "unmask_mode", "positions") or "positions"),
        cluster_attn_threshold=float(
            getattr(exp, "cluster_attn_threshold", 0.08) or 0.08
        ),
        cluster_max_size=int(getattr(exp, "cluster_max_size", 4) or 4),
        cluster_verify=bool(getattr(exp, "cluster_verify", False)),
        # Train the base model first; E73 fits the survival head in
        # _maybe_survival_gate before evaluation. Enabling decode-time survival
        # during SFT makes every training step take the expensive decode path.
        survival_gate=False,
        survival_commit_threshold=float(
            getattr(exp, "survival_commit_threshold", 0.3) or 0.3
        ),
        # Successor reuse is decode-only; keep it out of base SFT and enable it
        # through _eval_cfg after training.
        speculative_successor=False,
        speculative_fanout=int(getattr(exp, "speculative_fanout", 2) or 2),
        speculative_overlap=bool(getattr(exp, "speculative_overlap", False)),
    )


def _eval_cfg(exp: Experiment, args: argparse.Namespace) -> ModelBuildConfig:
    cfg = _train_cfg(exp, args)
    gen_steps = int(
        getattr(args, "override_gen_steps", None)
        or getattr(exp, "gen_steps_override", None)
        or args.gen_steps
        or 8
    )
    repair = exp.grammar_ltr_repair or exp.eid in {
        "E1",
        "E8",
        "E9b",
        "E10",
        "E11",
        "E12",
        "E13",
        "E14",
        "E15",
        "E16",
        "E17",
        "E18",
        "E19a",
        "E19b",
        "E20",
        "E21",
        "E22",
        "E29",
        "E30",
        "E31",
        "E32",
        "E33",
        "E35",
        "E36",
        "E40",
        "E41",
        "E42",
        "E43",
        "E44",
        "E45",
        "E46",
        "E50",
        "E51",
        "E52",
        "E53",
        "E54",
        "E55",
        "E70",
        "E71",
        "E72",
        "E73",
        "E74",
        "E75",
        "E76",
        "E77",
        "E78",
        "E80",
        "E82",
        "E84",
        "E88",
    }
    bon = exp.best_of_n
    if exp.decode_sweep == "gen16_repair_bon4":
        gen_steps = 16
        repair = True
        bon = 4
    elif exp.decode_sweep == "gen24_repair":
        gen_steps = 24
        repair = True
    elif exp.decode_sweep == "gen8_norepair":
        gen_steps = 8
        repair = False
    return replace(
        cfg,
        best_of_n=bon,
        gen_steps=gen_steps,
        grammar_ltr_repair=repair,
        grammar_ltr_primary=bool(getattr(exp, "grammar_ltr_primary", True)),
        template_fill_decode=bool(getattr(exp, "template_fill_decode", False)),
        contract_template_fastpath=bool(getattr(exp, "contract_template_fastpath", False)),
        honest_slot_contract=bool(getattr(exp, "honest_slot_contract", False)),
        remask_ratio=float(getattr(exp, "remask_ratio", 0.0) or 0.0),
        remask_use_gate=bool(getattr(exp, "remask_use_gate", False)),
        remask_use_entropy=bool(getattr(exp, "remask_use_entropy", False)),
        remask_policy=str(getattr(exp, "remask_policy", "confidence") or "confidence"),
        core_perturb_frac=float(getattr(exp, "core_perturb_frac", 0.25) or 0.25),
        remask_to_mask=bool(getattr(exp, "remask_to_mask", True)),
        slot_aware_trust_gate=bool(getattr(exp, "slot_aware_trust_gate", False)),
        mdlm_schedule=bool(getattr(exp, "mdlm_schedule", False)),
        suffix_rollback_window=int(getattr(exp, "suffix_rollback_window", 0) or 0),
        visible_corrupt_rate=float(getattr(exp, "visible_corrupt_rate", 0.0) or 0.0),
        cluster_verify=bool(getattr(exp, "cluster_verify", False)),
        survival_gate=bool(getattr(exp, "survival_gate", False)),
        speculative_successor=bool(getattr(exp, "speculative_successor", False)),
        speculative_fanout=int(getattr(exp, "speculative_fanout", 2) or 2),
        speculative_overlap=bool(getattr(exp, "speculative_overlap", False)),
        rico_eval_limit=args.rico_limit,
        eval_limit=args.eval_limit,
        run_id=exp.run_id,
    )


def _apply_decode_overrides(model: Any, exp: Experiment) -> None:
    """Align preference/RL rollout decode with the experiment recipe."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    for key, attr in (
        ("grammar_ltr_primary", "grammar_ltr_primary"),
        ("grammar_ltr_repair", "grammar_ltr_repair"),
        ("template_fill_decode", "template_fill_decode"),
        ("honest_slot_contract", "honest_slot_contract"),
        ("slot_contract_in_context", "slot_contract_in_context"),
        ("slot_contract_constrained_decode", "slot_contract_constrained_decode"),
        ("remask_ratio", "remask_ratio"),
        ("remask_use_gate", "remask_use_gate"),
        ("remask_use_entropy", "remask_use_entropy"),
        ("remask_policy", "remask_policy"),
        ("core_perturb_frac", "core_perturb_frac"),
        ("remask_to_mask", "remask_to_mask"),
        ("slot_aware_trust_gate", "slot_aware_trust_gate"),
        ("suffix_rollback_window", "suffix_rollback_window"),
        ("mdlm_schedule", "mdlm_schedule"),
        ("stability_min_persistence", "stability_min_persistence"),
        ("stability_jsd_weight", "stability_jsd_weight"),
        ("unmask_mode", "unmask_mode"),
        ("cluster_attn_threshold", "cluster_attn_threshold"),
        ("cluster_max_size", "cluster_max_size"),
        ("cluster_verify", "cluster_verify"),
        ("survival_gate", "survival_gate"),
        ("survival_commit_threshold", "survival_commit_threshold"),
        ("speculative_successor", "speculative_successor"),
        ("speculative_fanout", "speculative_fanout"),
        ("speculative_overlap", "speculative_overlap"),
    ):
        if hasattr(cfg, key):
            setattr(cfg, key, getattr(exp, attr, getattr(cfg, key)))


def _maybe_preference(exp: Experiment, ckpt: Path, args: argparse.Namespace) -> Path:
    if not exp.preference:
        return ckpt
    from slm_training.dsl.schema import load_jsonl
    from slm_training.models.twotower import TwoTowerModel
    from slm_training.harnesses.preference import (
        collect_pairs_with_generator,
        write_pairs,
    )
    from slm_training.harnesses.preference.train import train_preference_from_paths
    from slm_training.harnesses.quality import soft_corrupt_openui

    pairs_path = args.run_root / exp.run_id / "pairs.jsonl"
    records = load_jsonl(exp.train_dir / "records.jsonl")[: args.pref_limit]
    # Prefer soft-corrupt pairs (stable); optional live generator uses experiment decode.
    try:
        gen_model = TwoTowerModel.from_checkpoint(ckpt, device=args.device)
        _apply_decode_overrides(gen_model, exp)

        def _gen(r):  # noqa: ANN001
            pred = gen_model.generate(r.prompt, gold=r, design_md=r.design_md)
            return [r.openui, soft_corrupt_openui(r.openui), pred]

        pairs = collect_pairs_with_generator(
            records,
            _gen,
            prefer_valid_rejects=True,
            structure_only=True,
        )
    except Exception:  # noqa: BLE001
        pairs = collect_pairs_with_generator(
            records,
            lambda r: [r.openui, soft_corrupt_openui(r.openui)],
            prefer_valid_rejects=True,
            structure_only=True,
        )
    write_pairs(pairs_path, pairs)
    out_dir = args.run_root / exp.run_id / "pref"
    summary = train_preference_from_paths(
        ckpt,
        pairs_path,
        out_dir=out_dir,
        steps=args.pref_steps,
        device=args.device,
    )
    pref_ckpt = Path(summary.get("checkpoint") or (out_dir / "checkpoints" / "last.pt"))
    if pref_ckpt.is_file():
        dest = args.run_root / exp.run_id / "checkpoints" / "last.pt"
        return _copy_checkpoint(pref_ckpt, dest)
    return ckpt


def _maybe_local_preference(
    exp: Experiment, ckpt: Path, args: argparse.Namespace
) -> tuple[Path, dict[str, Any] | None]:
    objective = exp.local_preference_objective
    if objective is None:
        return ckpt, None
    if args.decision_events is None:
        raise ValueError(f"{exp.eid} requires --decision-events")
    from slm_training.harnesses.preference.local_train import train_local_from_paths

    tethered = bool(exp.local_preference_reference_tether)
    out_dir = args.run_root / exp.run_id / "local_preference"
    summary_path = out_dir / "local_preference_summary.json"
    if args.resume and summary_path.is_file():
        from slm_training.harnesses.preference.local_decisions import (
            load_decision_events,
        )

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        trained = Path(str(summary.get("checkpoint") or ""))
        expected_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        events = load_decision_events(args.decision_events)
        expected_counts = {
            split: sum(event.split == split for event in events)
            for split in ("train", "held_out")
        }
        matches = (
            summary.get("objective") == objective
            and int(summary.get("steps") or -1) == int(args.pref_steps)
            and bool(summary.get("balanced"))
            == bool(exp.local_preference_balanced)
            and bool(summary.get("reference_tethered")) == tethered
            and bool(summary.get("guarded_selection"))
            == bool(exp.local_preference_guarded_selection)
            and bool(summary.get("guarded_updates"))
            == bool(exp.local_preference_guarded_updates)
            and int(summary.get("guard_backtrack_steps") or 0)
            == (
                int(exp.local_preference_guard_backtrack_steps)
                if exp.local_preference_guarded_updates
                else 0
            )
            and bool(summary.get("guard_by_decision_kind"))
            == bool(exp.local_preference_guard_by_decision_kind)
            and bool(summary.get("block_by_decision_kind"))
            == bool(exp.local_preference_block_by_decision_kind)
            and summary.get("gradient_combination", "proposal")
            == exp.local_preference_gradient_combination
            and summary.get("optimizer", "adamw")
            == exp.local_preference_optimizer
            and summary.get("source_checkpoint_sha") == expected_sha
            and int(summary.get("train_events", -1)) == expected_counts["train"]
            and int(summary.get("held_out_events", -1))
            == expected_counts["held_out"]
            and trained.is_file()
        )
        if matches:
            dest = args.run_root / exp.run_id / "checkpoints" / "last.pt"
            return _copy_checkpoint(trained, dest), summary
        raise RuntimeError(f"{exp.eid} resume artifacts do not match this recipe")
    with run_trace(exp.run_id, "local_preference.train", run_dir=out_dir) as trace:
        started = time.perf_counter()
        summary = train_local_from_paths(
            ckpt,
            args.decision_events,
            out_dir=out_dir,
            objective=objective,
            reference_checkpoint=ckpt if tethered else None,
            steps=args.pref_steps,
            device=args.device,
            lr=args.local_pref_lr,
            epsilon=2.0,
            tau=1.0,
            non_target_tether=0.4 if tethered else 0.0,
            target_tether=0.05 if tethered else 0.0,
            target_grace=1.0,
            balanced=bool(exp.local_preference_balanced),
            seed=args.seed,
            validation_every=args.local_pref_validation_every,
            guarded_selection=bool(exp.local_preference_guarded_selection),
            guarded_updates=bool(exp.local_preference_guarded_updates),
            guard_backtrack_steps=int(exp.local_preference_guard_backtrack_steps),
            guard_by_decision_kind=bool(
                exp.local_preference_guard_by_decision_kind
            ),
            block_by_decision_kind=bool(
                exp.local_preference_block_by_decision_kind
            ),
            gradient_combination=exp.local_preference_gradient_combination,
            optimizer_name=exp.local_preference_optimizer,
        )
        summary["duration_seconds"] = time.perf_counter() - started
        selection = summary.get("validation_selection") or {}
        summary["validation_trials"] = sum(
            len(item.get("trials") or [])
            for item in selection.get("history") or []
        )
        summary["validation_event_forwards"] = (
            int(summary["validation_trials"])
            * int(summary.get("held_out_events") or 0)
        )
        summary["validation_batches"] = int(summary["validation_trials"]) * int(
            summary.get("validation_batch_groups") or 0
        )
        summary["trace_id"] = trace.trace_id
        summary["traceparent"] = trace.traceparent
        summary["trace_bundle"] = trace.bundle.as_posix()
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    trained = Path(summary["checkpoint"])
    dest = args.run_root / exp.run_id / "checkpoints" / "last.pt"
    return _copy_checkpoint(trained, dest), summary


def _maybe_rl(exp: Experiment, ckpt: Path, args: argparse.Namespace) -> Path:
    if not getattr(exp, "rl", False):
        return ckpt
    from slm_training.models.twotower import TwoTowerModel
    from slm_training.harnesses.rl import train_grpo_from_paths
    from slm_training.autoresearch.rl_gate import assert_rl_ready

    readiness = assert_rl_ready(getattr(args, "rl_readiness_report", None))

    # Prefetch decode overrides onto a temp copy the RL trainer will reload.
    try:
        model = TwoTowerModel.from_checkpoint(ckpt, device=args.device)
        _apply_decode_overrides(model, exp)
        tuned = args.run_root / exp.run_id / "rl_seed" / "last.pt"
        tuned.parent.mkdir(parents=True, exist_ok=True)
        model.save(tuned)
        ckpt_for_rl = tuned
    except Exception:  # noqa: BLE001
        ckpt_for_rl = ckpt

    out_dir = args.run_root / exp.run_id / "rl"
    summary = train_grpo_from_paths(
        ckpt_for_rl,
        exp.train_dir / "records.jsonl",
        out_dir=out_dir,
        steps=max(10, int(getattr(args, "rl_steps", 30) or 30)),
        group_size=max(2, int(getattr(args, "rl_group_size", 4) or 4)),
        device=args.device,
        ref_checkpoint=ckpt,
        limit=int(getattr(args, "pref_limit", 32) or 32),
        kl_beta=0.05,
        readiness_report=readiness,
    )
    rl_ckpt = Path(summary.get("checkpoint") or (out_dir / "model.pt"))
    if rl_ckpt.is_file():
        dest = args.run_root / exp.run_id / "checkpoints" / "last.pt"
        return _copy_checkpoint(rl_ckpt, dest)
    return ckpt


def _maybe_trust_gate(exp: Experiment, ckpt: Path, args: argparse.Namespace) -> Path:
    if not getattr(exp, "trust_gate", False):
        return ckpt
    from slm_training.dsl.grammar.fastpath.trust_train import (
        train_trust_gate_from_paths,
    )

    out_dir = args.run_root / exp.run_id / "trust_gate"
    summary = train_trust_gate_from_paths(
        ckpt,
        exp.train_dir / "records.jsonl",
        out_dir=out_dir,
        steps=max(1, int(getattr(args, "pref_steps", 30) or 30)),
        device=args.device,
        limit=int(getattr(args, "pref_limit", 40) or 40),
        slot_aware=bool(getattr(exp, "slot_aware_trust_gate", False)),
    )
    gate_ckpt = Path(summary.get("checkpoint") or (out_dir / "checkpoints" / "last.pt"))
    if gate_ckpt.is_file():
        dest = args.run_root / exp.run_id / "checkpoints" / "last.pt"
        return _copy_checkpoint(gate_ckpt, dest)
    return ckpt


def _maybe_survival_gate(exp: Experiment, ckpt: Path, args: argparse.Namespace) -> Path:
    """E73 (V7): train the trajectory-survival head after SFT/trust stages."""
    if not getattr(exp, "survival_gate", False):
        return ckpt
    from slm_training.dsl.grammar.fastpath.survival_train import (
        train_survival_gate_from_paths,
    )

    out_dir = args.run_root / exp.run_id / "survival_gate"
    summary = train_survival_gate_from_paths(
        ckpt,
        exp.train_dir / "records.jsonl",
        out_dir=out_dir,
        steps=max(20, int(getattr(args, "pref_steps", 30) or 30)),
        device=args.device,
        limit=int(getattr(args, "pref_limit", 40) or 40),
    )
    gate_ckpt = Path(summary.get("checkpoint") or (out_dir / "checkpoints" / "last.pt"))
    if gate_ckpt.is_file():
        dest = args.run_root / exp.run_id / "checkpoints" / "last.pt"
        return _copy_checkpoint(gate_ckpt, dest)
    return ckpt


def _summarize_board(board: dict[str, Any]) -> dict[str, Any]:
    suites = board.get("suites") or {}
    gates = evaluate_ship_gates(suites)

    def durable_decode_stats(metrics: dict[str, Any]) -> dict[str, Any]:
        """Persist aggregate telemetry, not per-token high-cardinality traces."""
        durable: dict[str, Any] = {}
        for key, value in (metrics.get("decode_stats") or {}).items():
            if isinstance(value, (bool, int, float, str)) or value is None:
                durable[key] = value
            elif (
                isinstance(value, dict)
                and len(value) <= 32
                and all(
                    isinstance(item, (bool, int, float, str)) or item is None
                    for item in value.values()
                )
            ):
                durable[key] = value
        return durable

    slim = {
        name: {
            "parse_rate": m.get("parse_rate"),
            "syntax_parse_rate": m.get("syntax_parse_rate"),
            "meaningful_program_rate": m.get("meaningful_program_rate"),
            "placeholder_fidelity": m.get("placeholder_fidelity"),
            "structural_similarity": m.get("structural_similarity"),
            "reward_score": m.get("reward_score"),
            "latency_ms_p50": m.get("latency_ms_p50"),
            "latency_ms_p95": m.get("latency_ms_p95"),
            "fallback_count": m.get("fallback_count"),
            "decode_timeout_count": m.get("decode_timeout_count"),
            "constrained_fallback_rate": m.get("constrained_fallback_rate"),
            "evaluation_policy": m.get("evaluation_policy"),
            "decode_stats": durable_decode_stats(m),
            "n": m.get("n"),
            # V7 decode telemetry (present when speculative levers are on).
            **(
                {"speculative_stats": m["speculative_stats"]}
                if m.get("speculative_stats")
                else {}
            ),
        }
        for name, m in suites.items()
    }
    return {
        "pass": gates.get("pass"),
        "failures": gates.get("failures"),
        "checkpoint_sha256": board.get("checkpoint_sha256"),
        "evaluated_at": board.get("evaluated_at"),
        "agentv": {
            "format": (board.get("agentv") or {}).get("format"),
            "sdk": (board.get("agentv") or {}).get("sdk"),
            "summary": (board.get("agentv") or {}).get("summary"),
        },
        "suites": slim,
    }


def run_one(exp: Experiment, args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_root / exp.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # E34 is deferred research — record a skipped result unless forced via env.
    if exp.eid == "E34" and not bool(getattr(args, "force_e34", False)):
        result = {
            "id": exp.eid,
            "run_id": exp.run_id,
            "description": exp.description,
            "checkpoint": None,
            "pass": None,
            "failures": ["deferred_latent_moe"],
            "suites": {},
            "skipped": True,
        }
        (run_dir / "matrix_result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        return result

    if exp.local_preference_objective is not None:
        if not exp.parent_checkpoint:
            raise ValueError(f"{exp.eid} local preference requires --parent")
        parent = Path(exp.parent_checkpoint)
        if not parent.is_file():
            raise FileNotFoundError(f"{exp.eid} needs parent checkpoint {parent}")
        ckpt = _copy_checkpoint(parent, run_dir / "checkpoints" / "last.pt")
    elif exp.initialization == "eval_only":
        if not exp.parent_checkpoint:
            raise ValueError(f"{exp.eid} eval-only initialization requires --parent")
        ckpt = Path(exp.parent_checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(f"{exp.eid} needs parent checkpoint {ckpt}")
    elif exp.initialization == "parent":
        if not exp.parent_checkpoint:
            raise ValueError(f"{exp.eid} parent initialization requires --parent")
        parent = Path(exp.parent_checkpoint)
        if not parent.is_file():
            raise FileNotFoundError(f"{exp.eid} needs parent checkpoint {parent}")
        cfg = _train_cfg(exp, args)
        model = build_model(cfg, load_train_records(exp.train_dir), checkpoint=parent)
        summary = train(cfg, model=model)
        ckpt = Path(summary["checkpoint"])
    elif exp.seed_checkpoint:
        src = Path(exp.seed_checkpoint)
        dest = run_dir / "checkpoints" / "last.pt"
        if not src.is_file():
            # Seed optional for V4/V6 decode overlays — train instead.
            if exp.eid in {
                "E30",
                "E31",
                "E33",
                "E50",
                "E51",
                "E52",
                "E70",
                "E71",
                "E72",
                "E73",
                "E74",
            }:
                summary = train(_train_cfg(exp, args))
                ckpt = Path(summary["checkpoint"])
            else:
                raise FileNotFoundError(f"{exp.eid} needs seed checkpoint {src}")
        else:
            ckpt = _copy_checkpoint(src, dest)
    elif exp.eval_from_checkpoint:
        src = Path(exp.eval_from_checkpoint)
        dest = run_dir / "checkpoints" / "last.pt"
        if not src.is_file():
            raise FileNotFoundError(f"{exp.eid} needs checkpoint {src}")
        ckpt = _copy_checkpoint(src, dest)
    elif exp.eval_from_run and not exp.preference:
        src = args.run_root / exp.eval_from_run / "checkpoints" / "last.pt"
        dest = run_dir / "checkpoints" / "last.pt"
        if not src.is_file():
            raise FileNotFoundError(f"{exp.eid} needs {src}")
        ckpt = _copy_checkpoint(src, dest)
    else:
        train_cfg = _train_cfg(exp, args)
        if exp.eval_from_run and exp.preference:
            # Start from prior checkpoint weights, then preference-train.
            src = args.run_root / exp.eval_from_run / "checkpoints" / "last.pt"
            dest = run_dir / "checkpoints" / "last.pt"
            ckpt = _copy_checkpoint(src, dest)
        else:
            summary = train(train_cfg)
            ckpt = Path(summary["checkpoint"])

    ckpt = _maybe_preference(exp, ckpt, args)
    ckpt, local_preference_summary = _maybe_local_preference(exp, ckpt, args)
    ckpt = _maybe_rl(exp, ckpt, args)
    ckpt = _maybe_trust_gate(exp, ckpt, args)
    ckpt = _maybe_survival_gate(exp, ckpt, args)
    eval_cfg = _eval_cfg(exp, args)
    with run_trace(exp.run_id, "eval", run_dir=run_dir) as trace:
        board = evaluate_suites(
            eval_cfg,
            args.suites,
            checkpoint=ckpt,
            write_gates=True,
        )
    result = {
        "id": exp.eid,
        "run_id": exp.run_id,
        "initialization": exp.initialization,
        "training_executed": exp.initialization != "eval_only",
        "parent_checkpoint": exp.parent_checkpoint,
        "description": exp.description,
        "honest_slot_contract": eval_cfg.honest_slot_contract,
        "design_md_in_context": eval_cfg.design_md_in_context,
        "schema_in_context": eval_cfg.schema_in_context,
        "slot_contract_in_context": eval_cfg.slot_contract_in_context,
        "slot_contract_constrained_decode": (eval_cfg.slot_contract_constrained_decode),
        "template_fill_decode": eval_cfg.template_fill_decode,
        "grammar_ltr_primary": eval_cfg.grammar_ltr_primary,
        "grammar_ltr_repair": eval_cfg.grammar_ltr_repair,
        "grammar_finalize_validate": eval_cfg.grammar_finalize_validate,
        "allow_unconstrained_fallback": eval_cfg.allow_unconstrained_fallback,
        "compiler_decode_mode": eval_cfg.compiler_decode_mode,
        "compiler_search_mode": eval_cfg.compiler_search_mode,
        "compiler_search_trigger": eval_cfg.compiler_search_trigger,
        "compiler_search_width": eval_cfg.compiler_search_width,
        "compiler_search_noise": eval_cfg.compiler_search_noise,
        "compiler_search_stagnation_patience": (
            eval_cfg.compiler_search_stagnation_patience
        ),
        "compiler_search_backtrack_limit": eval_cfg.compiler_search_backtrack_limit,
        "compiler_search_local_nogoods": eval_cfg.compiler_search_local_nogoods,
        "effective_gen_steps": eval_cfg.gen_steps,
        "best_of_n": eval_cfg.best_of_n,
        "train_dir": str(exp.train_dir),
        "train_content_fingerprint": json.loads(
            (exp.train_dir / "manifest.json").read_text(encoding="utf-8")
        ).get("content_fingerprint"),
        "checkpoint": str(ckpt),
        "trace_id": trace.trace_id,
        "traceparent": trace.traceparent,
        "trace_bundle": trace.bundle.as_posix(),
        "local_preference_objective": exp.local_preference_objective,
        "local_preference_reference_tether": (
            exp.local_preference_reference_tether
        ),
        "local_preference_balanced": exp.local_preference_balanced,
        "local_preference_guarded_selection": (
            exp.local_preference_guarded_selection
        ),
        "local_preference_guarded_updates": exp.local_preference_guarded_updates,
        "local_preference_guard_backtrack_steps": (
            exp.local_preference_guard_backtrack_steps
        ),
        "local_preference_guard_by_decision_kind": (
            exp.local_preference_guard_by_decision_kind
        ),
        "local_preference_block_by_decision_kind": (
            exp.local_preference_block_by_decision_kind
        ),
        "local_preference_gradient_combination": (
            exp.local_preference_gradient_combination
        ),
        "local_preference_optimizer": exp.local_preference_optimizer,
        "local_preference_summary": local_preference_summary,
        **_summarize_board(board),
    }
    (run_dir / "matrix_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("outputs/data/train/v1"))
    parser.add_argument(
        "--curriculum-dir",
        type=Path,
        default=Path("outputs/data/train/v1_curriculum"),
    )
    parser.add_argument("--test-dir", type=Path, default=Path("outputs/data/eval/v1"))
    parser.add_argument("--run-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=None,
        help=(
            "Summary mirror path (defaults to a run-root-specific file under "
            "docs/design for non-default run roots)."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--context-backend", choices=("scratch", "hf"), default="scratch"
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--rico-limit", type=int, default=32)
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Diagnostic-only cap per selected suite; omit for full evaluation.",
    )
    parser.add_argument(
        "--suites",
        default=",".join(SUITES),
        help="Comma-separated eval suites (default: all).",
    )
    parser.add_argument("--pref-steps", type=int, default=30)
    parser.add_argument("--pref-limit", type=int, default=40)
    parser.add_argument(
        "--decision-events",
        type=Path,
        default=None,
        help="DecisionEventV1 JSONL required by V10 intervention rows.",
    )
    parser.add_argument("--local-pref-lr", type=float, default=5e-5)
    parser.add_argument("--local-pref-validation-every", type=int, default=5)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed, recipe-matching local preference stage artifacts.",
    )
    parser.add_argument("--rl-steps", type=int, default=30)
    parser.add_argument("--rl-group-size", type=int, default=4)
    parser.add_argument("--rl-readiness-report", type=Path, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel experiment workers (thread pool; 1 = sequential).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile on train experiments.",
    )
    parser.add_argument(
        "--parallel-unmask",
        default="adaptive",
        choices=("topk", "confidence", "adaptive"),
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated experiment ids (e.g. E0,E1,E8).",
    )
    parser.add_argument(
        "--no-design-md-context",
        action="store_true",
        help="Train/eval without DESIGN.md in context (matches fixture-demo ship ckpt).",
    )
    parser.add_argument(
        "--seed-checkpoint",
        type=Path,
        default=None,
        help="Optional strong checkpoint for decode-only experiments (E1/E5).",
    )
    parser.add_argument(
        "--parent",
        type=Path,
        default=None,
        help="Named parent model checkpoint for branch-initialized matrix candidates.",
    )
    parser.add_argument(
        "--scratch-control",
        action="store_true",
        help="Explicitly run selected rows as non-deployable scratch controls.",
    )
    parser.add_argument(
        "--eval-checkpoint",
        type=Path,
        default=None,
        help=(
            "Frozen checkpoint for declared eval-only rows (V9 E240-E247): "
            "evaluate decode policies over one shared lineage without training."
        ),
    )
    parser.add_argument(
        "--build-curriculum",
        action="store_true",
        help="Build curriculum train corpus before running.",
    )
    parser.add_argument(
        "--namespace-dir",
        type=Path,
        default=Path("outputs/data/train/v1_namespace"),
    )
    parser.add_argument(
        "--matrix",
        choices=(
            "legacy",
            "v2",
            "v3",
            "v4",
            "v5",
            "v6",
            "v7",
            "v8",
            "v9",
            "v10",
            "v11",
            "v12",
            "v14",
            "v15",
            "v16",
            "v17",
            "all",
        ),
        default="v3",
        help="Experiment set through v10 local-decision rows E248-E254,"
        " v11 representation rows E255-E257, v12 choice-codec row E262,"
        " v14 decode-distortion row E277, v15 pseudo-embedding row E278,"
        " or all.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print selected experiment definitions without running them.",
    )
    parser.add_argument("--gen-steps", type=int, default=8)
    parser.add_argument(
        "--override-gen-steps",
        type=int,
        default=None,
        help="Force evaluation decode steps, overriding an experiment preset.",
    )
    parser.add_argument(
        "--force-e34",
        action="store_true",
        help="Run deferred E34 latent MoE placeholder (normally skipped).",
    )
    args = parser.parse_args(argv)
    # Modern curriculum rows (notably E53) use ``curriculum_dir`` as their
    # actual training input.  If a caller explicitly supplies a different
    # train corpus, do not silently substitute the stale default curriculum
    # snapshot; use the requested corpus unless a curriculum path was also
    # explicitly selected.
    if (
        args.train_dir != Path("outputs/data/train/v1")
        and args.curriculum_dir == Path("outputs/data/train/v1_curriculum")
    ):
        args.curriculum_dir = args.train_dir
    args.suites = tuple(
        value.strip() for value in args.suites.split(",") if value.strip()
    )
    unknown_suites = sorted(set(args.suites) - set(SUITES))
    if unknown_suites:
        parser.error(f"unknown suites: {','.join(unknown_suites)}")
    if not args.suites:
        parser.error("--suites must select at least one suite")
    if args.matrix in {"v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v14", "v15", "v16", "v17", "all"}:
        if (
            args.parent is None
            and not args.scratch_control
            and args.eval_checkpoint is None
        ):
            if not args.list:
                parser.error(
                    "modern matrices require --parent, explicit --scratch-control,"
                    " or --eval-checkpoint"
                )
    if args.matrix in {"v10", "all"} and args.parent is None and not args.list:
        parser.error("V10 exact-state rows require --parent")

    selected_ids = (
        {value.strip().upper() for value in args.only.split(",") if value.strip()}
        if args.only
        else None
    )
    needs_curriculum = selected_ids is None or bool(
        selected_ids
        & {
            "E2",
            "E8",
            "E9b",
            "E10",
            "E15",
            "E16",
            "E29",
            "E35",
            "E46",
            "E53",
            "E55",
            "E75",
        }
    )
    if not args.list and (
        args.build_curriculum
        or (not args.curriculum_dir.exists() and needs_curriculum)
    ):
        from slm_training.harnesses.train_data import TrainDataConfig, build_train_data

        build_train_data(
            TrainDataConfig(
                source="all",
                output_root=args.curriculum_dir.parent,
                version=args.curriculum_dir.name,
                synthesizer="quality",
                curriculum=True,
            )
        )

    needs_namespace = selected_ids is None or "E14" in selected_ids
    if not args.list and needs_namespace and not args.namespace_dir.exists():
        from slm_training.harnesses.train_data import TrainDataConfig, build_train_data

        build_train_data(
            TrainDataConfig(
                source="all",
                output_root=args.namespace_dir.parent,
                version=args.namespace_dir.name,
                synthesizer="quality",
                namespace_augment=True,
            )
        )

    design_md = not args.no_design_md_context
    experiments: list[Experiment] = []
    if args.matrix in {"legacy", "all"}:
        experiments.extend(
            _base_experiments(
                args.train_dir,
                args.curriculum_dir,
                seed_checkpoint=args.seed_checkpoint,
                design_md_in_context=design_md,
            )
        )
    if args.matrix in {"v2", "all"}:
        experiments.extend(
            _v2_experiments(
                args.train_dir,
                args.curriculum_dir,
                args.namespace_dir,
                design_md_in_context=design_md,
            )
        )
    if args.matrix in {"v3", "all"}:
        experiments.extend(
            _v3_experiments(
                args.train_dir,
                args.curriculum_dir,
                design_md_in_context=design_md,
            )
        )
    if args.matrix in {"v4", "all"}:
        experiments.extend(
            _v4_experiments(
                args.train_dir,
                args.curriculum_dir,
                design_md_in_context=design_md,
                seed_checkpoint=args.seed_checkpoint,
            )
        )
    if args.matrix in {"v5", "all"}:
        experiments.extend(
            _v5_experiments(
                args.train_dir,
                args.curriculum_dir,
                design_md_in_context=design_md,
            )
        )
    if args.matrix in {"v6", "all"}:
        experiments.extend(
            _v6_experiments(
                args.train_dir,
                args.curriculum_dir,
                design_md_in_context=design_md,
                seed_checkpoint=args.seed_checkpoint,
            )
        )
    if args.matrix in {"v7", "all"}:
        experiments.extend(
            _v7_experiments(
                args.train_dir,
                args.curriculum_dir,
                design_md_in_context=design_md,
                seed_checkpoint=args.seed_checkpoint,
            )
        )
    if args.matrix in {"v8", "all"}:
        experiments.extend(
            _v8_experiments(
                args.train_dir,
                design_md_in_context=design_md,
            )
        )
    if args.matrix in {"v9", "all"}:
        experiments.extend(_v9_experiments(args.train_dir))
    if args.matrix in {"v10", "all"}:
        experiments.extend(_v10_experiments(args.train_dir))
    if args.matrix in {"v11", "all"}:
        experiments.extend(_v11_experiments(args.train_dir))
    if args.matrix in {"v12", "all"}:
        experiments.extend(_v12_experiments(args.train_dir))
    if args.matrix in {"v14", "all"}:
        experiments.extend(_v14_experiments(args.train_dir))
    if args.matrix in {"v15", "all"}:
        experiments.extend(_v15_experiments(args.train_dir))
    if args.matrix in {"v16", "all"}:
        experiments.extend(_v16_experiments(args.train_dir))
    if args.matrix in {"v17", "all"}:
        experiments.extend(_v17_experiments(args.train_dir))
    if args.only:
        experiments = [e for e in experiments if e.eid in selected_ids]
    if args.list:
        print(
            json.dumps(
                [
                    {
                        "id": exp.eid,
                        "run_id": exp.run_id,
                        "description": exp.description,
                    }
                    for exp in experiments
                ],
                indent=2,
            )
        )
        return 0
    if any(exp.local_preference_objective for exp in experiments):
        if args.decision_events is None:
            parser.error("V10 intervention rows require --decision-events")

    if args.eval_checkpoint is not None and args.scratch_control:
        parser.error("--eval-checkpoint and --scratch-control are mutually exclusive")
    experiments = _apply_eval_checkpoint(experiments, args.eval_checkpoint)

    classified: list[Experiment] = []
    for exp in experiments:
        if exp.initialization == "eval_only":
            initialization = "eval_only"
        elif args.scratch_control or args.matrix in {"legacy", "v2", "v3"}:
            initialization = "scratch"
        elif exp.eval_from_run or exp.eval_from_checkpoint:
            initialization = "eval_only"
        elif exp.local_parent_control:
            initialization = "eval_only"
        elif (
            exp.preference
            or exp.local_preference_objective is not None
            or exp.rl
            or exp.trust_gate
            or exp.survival_gate
        ):
            initialization = "process"
        else:
            initialization = "parent"
        classified.append(
            replace(
                exp,
                initialization=initialization,
                parent_checkpoint=(
                    str(args.parent)
                    if args.parent is not None
                    and (
                        initialization in {"parent", "eval_only"}
                        or exp.local_parent_control
                        or exp.local_preference_objective is not None
                    )
                    else exp.parent_checkpoint
                ),
                seed_checkpoint=(
                    str(args.parent)
                    if args.parent is not None and initialization == "process"
                    else exp.seed_checkpoint
                ),
            )
        )
    experiments = classified

    # Ensure E0 runs before dependents when selected together.
    order = {e.eid: i for i, e in enumerate(experiments)}
    experiments = sorted(
        experiments,
        key=lambda e: (
            0 if e.eid == "E0" else 1 if e.eval_from_run else 2,
            order.get(e.eid, 99),
        ),
    )

    results: list[dict[str, Any]] = []
    progress_path = args.run_root / "quality_matrix_progress.json"
    active_experiment: Experiment | None = None

    def _persist_progress(status: str, active: Experiment | None = None) -> None:
        nonlocal active_experiment
        if active is not None:
            active_experiment = active
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(
            json.dumps(
                {
                    "status": status,
                    "matrix": args.matrix,
                    "completed": len(results),
                    "total": len(experiments),
                    "active": (
                        {"id": active.eid, "run_id": active.run_id}
                        if active is not None
                        else None
                    ),
                    "results": sorted(results, key=lambda r: r.get("id") or ""),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _mark_interrupted(signum: int, _frame: Any) -> None:
        """Persist resumable state before the supervisor stops the matrix."""
        _persist_progress("interrupted", active=active_experiment)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, _mark_interrupted)
    signal.signal(signal.SIGTERM, _mark_interrupted)

    workers = max(1, int(args.workers))
    # Seed/decode overlays that depend on another run stay sequential first.
    dependent = [
        e
        for e in experiments
        if e.eval_from_run or e.seed_checkpoint or e.eval_from_checkpoint
    ]
    independent = [e for e in experiments if e not in dependent]

    def _run(exp: Experiment) -> dict[str, Any]:
        _persist_progress("running", active=exp)
        print(json.dumps({"status": "start", "id": exp.eid, "run_id": exp.run_id}))
        try:
            result = run_one(exp, args)
        except BaseException as exc:  # noqa: BLE001 - preserve partial matrix evidence
            if isinstance(exc, (KeyboardInterrupt, GeneratorExit)):
                raise
            result = {
                "id": exp.eid,
                "run_id": exp.run_id,
                "pass": False,
                "failures": [f"exception: {type(exc).__name__}: {exc}"],
                "suites": {},
            }
        print(json.dumps({"status": "done", "id": exp.eid, "pass": result["pass"]}))
        return result

    # Run independent train experiments possibly in parallel.
    if workers > 1 and len(independent) > 1:
        from concurrent.futures import as_completed

        # Process pool can't pickle complex args cleanly — fall back to threads
        # for shared-memory CPU parallelism without re-importing CUDA contexts.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(workers, len(independent))) as pool:
            futs = {pool.submit(_run, exp): exp for exp in independent}
            for fut in as_completed(futs):
                results.append(fut.result())
                _persist_progress("running")
    else:
        for exp in independent:
            results.append(_run(exp))
            _persist_progress("running")

    for exp in dependent:
        results.append(_run(exp))
        _persist_progress("running")

    # Stable order by experiment id.
    results.sort(key=lambda r: r.get("id") or "")

    training_executed = any(result.get("training_executed") for result in results)
    design_policies = {
        result.get("design_md_in_context")
        for result in results
        if "design_md_in_context" in result
    }
    out = {
        "matrix": f"quality-experiment-matrix-{args.matrix}",
        "reference": "docs/design/quality-experiment-matrix.md",
        "gate_policy": {k: v for k, v in DEFAULT_SHIP_GATES.items()},
        "device": args.device,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "test_dir": str(args.test_dir),
        "design_md_in_context": (
            design_policies.pop() if len(design_policies) == 1 else None
        ),
        "training_executed": training_executed,
        "rico_eval_limit": args.rico_limit,
        "suites": sorted(args.suites),
        "steps": args.steps if training_executed else 0,
        "gen_steps": args.gen_steps,
        "context_backend": args.context_backend,
        "matrix_set": args.matrix,
        "results": results,
    }
    out_path = args.run_root / "quality_matrix_summary.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    _persist_progress("complete")
    # Also mirror under docs artifacts path for the PR.
    docs_out = args.docs_out or Path("docs/design") / (
        "quality-matrix-results.json"
        if args.run_root == Path("outputs/runs")
        else f"quality-matrix-results-{args.run_root.name}.json"
    )
    docs_out.parent.mkdir(parents=True, exist_ok=True)
    docs_out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(out_path), "n": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
