#!/usr/bin/env python3
"""Run the quality experiment matrix (docs/design/quality-experiment-matrix.md)."""

from __future__ import annotations

import argparse
import json
import signal
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from slm_training.harnesses.model_build import ModelBuildConfig, build_model, train
from slm_training.harnesses.model_build.data import load_train_records
from slm_training.harnesses.model_build.eval_runner import evaluate_suites
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
    design_md_in_context: bool = True
    # Absolute checkpoint seed (preferred over eval_from_run when set)
    seed_checkpoint: str | None = None
    preference: bool = False
    mix_curriculum: bool = True
    rl: bool = False
    slot_contract_in_context: bool = False
    slot_contract_constrained_decode: bool = False
    namespace_augment: bool = False
    ltr_loss_weight: float = 1.0
    # Eval-only overlay: decode sweep presets (E17)
    decode_sweep: str | None = None
    eval_from_checkpoint: str | None = None
    # V3 levers
    grammar_ltr_primary: bool = True
    template_fill_decode: bool = False
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
    ]


def _train_cfg(exp: Experiment, args: argparse.Namespace) -> ModelBuildConfig:
    return ModelBuildConfig(
        train_dir=exp.train_dir,
        test_dir=args.test_dir,
        suite="smoke",
        run_root=args.run_root,
        run_id=exp.run_id,
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
        grammar_constrained=True,
        grammar_ltr_primary=bool(getattr(exp, "grammar_ltr_primary", True)),
        grammar_ltr_repair=exp.grammar_ltr_repair,
        grammar_ltr_max_tokens=exp.grammar_ltr_max_tokens,
        design_md_in_context=exp.design_md_in_context,
        ltr_loss_weight=getattr(exp, "ltr_loss_weight", 1.0),
        fidelity_loss_weight=exp.fidelity_loss_weight,
        schema_in_context=exp.schema_in_context,
        slot_contract_in_context=getattr(exp, "slot_contract_in_context", False),
        slot_contract_constrained_decode=getattr(
            exp, "slot_contract_constrained_decode", False
        ),
        template_fill_decode=bool(getattr(exp, "template_fill_decode", False)),
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
    slim = {
        name: {
            "parse_rate": m.get("parse_rate"),
            "placeholder_fidelity": m.get("placeholder_fidelity"),
            "structural_similarity": m.get("structural_similarity"),
            "reward_score": m.get("reward_score"),
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

    if exp.initialization == "parent":
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
    ckpt = _maybe_rl(exp, ckpt, args)
    ckpt = _maybe_trust_gate(exp, ckpt, args)
    ckpt = _maybe_survival_gate(exp, ckpt, args)
    eval_cfg = _eval_cfg(exp, args)
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
        "parent_checkpoint": exp.parent_checkpoint,
        "description": exp.description,
        "honest_slot_contract": eval_cfg.honest_slot_contract,
        "schema_in_context": eval_cfg.schema_in_context,
        "slot_contract_in_context": eval_cfg.slot_contract_in_context,
        "slot_contract_constrained_decode": (eval_cfg.slot_contract_constrained_decode),
        "template_fill_decode": eval_cfg.template_fill_decode,
        "grammar_ltr_primary": eval_cfg.grammar_ltr_primary,
        "grammar_ltr_repair": eval_cfg.grammar_ltr_repair,
        "effective_gen_steps": eval_cfg.gen_steps,
        "best_of_n": eval_cfg.best_of_n,
        "train_dir": str(exp.train_dir),
        "train_content_fingerprint": json.loads(
            (exp.train_dir / "manifest.json").read_text(encoding="utf-8")
        ).get("content_fingerprint"),
        "checkpoint": str(ckpt),
        **_summarize_board(board),
    }
    (run_dir / "matrix_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("outputs/train_data/v1"))
    parser.add_argument(
        "--curriculum-dir",
        type=Path,
        default=Path("outputs/train_data/v1_curriculum"),
    )
    parser.add_argument("--test-dir", type=Path, default=Path("outputs/test_data/v1"))
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
        "--build-curriculum",
        action="store_true",
        help="Build curriculum train corpus before running.",
    )
    parser.add_argument(
        "--namespace-dir",
        type=Path,
        default=Path("outputs/train_data/v1_namespace"),
    )
    parser.add_argument(
        "--matrix",
        choices=("legacy", "v2", "v3", "v4", "v5", "v6", "v7", "all"),
        default="v3",
        help="Experiment set: legacy (E0–E10), v2 (E11–E17), v3 (E18–E29), v4 (E30–E36), v5 (E40–E46), v6 (E50–E55), v7 (E70–E75), or all.",
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
    args.suites = tuple(
        value.strip() for value in args.suites.split(",") if value.strip()
    )
    unknown_suites = sorted(set(args.suites) - set(SUITES))
    if unknown_suites:
        parser.error(f"unknown suites: {','.join(unknown_suites)}")
    if not args.suites:
        parser.error("--suites must select at least one suite")
    if args.matrix in {"v4", "v5", "v6", "v7", "all"}:
        if args.parent is None and not args.scratch_control:
            parser.error(
                "modern matrices require --parent or explicit --scratch-control"
            )

    needs_curriculum = args.only is None or any(
        x in (args.only or "")
        for x in (
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
        )
    )
    if args.build_curriculum or (not args.curriculum_dir.exists() and needs_curriculum):
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

    needs_namespace = args.only is None or any(x in (args.only or "") for x in ("E14",))
    if needs_namespace and not args.namespace_dir.exists():
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
    if args.only:
        wanted = {x.strip().upper() for x in args.only.split(",") if x.strip()}
        experiments = [e for e in experiments if e.eid in wanted]

    classified: list[Experiment] = []
    for exp in experiments:
        if args.scratch_control or args.matrix in {"legacy", "v2", "v3"}:
            initialization = "scratch"
        elif exp.eval_from_run or exp.eval_from_checkpoint:
            initialization = "eval_only"
        elif exp.preference or exp.rl or exp.trust_gate or exp.survival_gate:
            initialization = "process"
        else:
            initialization = "parent"
        classified.append(
            replace(
                exp,
                initialization=initialization,
                parent_checkpoint=(
                    str(args.parent)
                    if args.parent is not None and initialization == "parent"
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
        except Exception as exc:  # noqa: BLE001 - preserve partial matrix evidence
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

    out = {
        "matrix": f"quality-experiment-matrix-{args.matrix}",
        "reference": "docs/design/quality-experiment-matrix.md",
        "gate_policy": {k: v for k, v in DEFAULT_SHIP_GATES.items()},
        "device": args.device,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "test_dir": str(args.test_dir),
        "design_md_in_context": not args.no_design_md_context,
        "rico_eval_limit": args.rico_limit,
        "suites": sorted(args.suites),
        "steps": args.steps,
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
