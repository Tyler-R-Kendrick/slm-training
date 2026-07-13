"""Factory for model plug-ins."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.plugin import ModelPlugin, StubModel


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
    for key in (
        "grammar_constrained",
        "grammar_top_k",
        "structural_bias",
        "grammar_ltr_repair",
        "grammar_ltr_max_tokens",
        "grammar_ltr_primary",
        "grammar_finalize_validate",
        "design_md_budget",
        "schema_in_context",
        "slot_contract_in_context",
        "slot_contract_constrained_decode",
        "template_fill_decode",
        "honest_slot_contract",
        "retrieval_k",
        "best_of_n",
        "fidelity_loss_weight",
        "ltr_loss_weight",
        "parallel_unmask",
        "remask_ratio",
        "mdlm_schedule",
        "mdlm_eps",
        "cache_context",
        "fuse_ltr_loss",
        "grammar_fastpath",
        "grammar_fastpath_mode",
        "grammar_draft_window",
        "fastpath_aux_weight",
        "fastpath_gate_threshold",
        "suffix_rollback_window",
        "remask_use_gate",
        "remask_use_entropy",
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
    ):
        if hasattr(config, key) and hasattr(cfg, key):
            setattr(cfg, key, getattr(config, key))
    # Preserve checkpoint DESIGN.md conditioning unless caller sets an explicit bool.
    # Eval defaults must not force-enable gold DESIGN.md on no-design-md checkpoints.
    dm = getattr(config, "design_md_in_context", None)
    if dm is not None and hasattr(cfg, "design_md_in_context"):
        cfg.design_md_in_context = bool(dm)
    # Decode quality defaults often wanted at eval time.
    if getattr(config, "grammar_ltr_repair", False) and hasattr(cfg, "grammar_ltr_repair"):
        cfg.grammar_ltr_repair = True
    if int(getattr(config, "best_of_n", 1) or 1) > 1 and hasattr(cfg, "best_of_n"):
        cfg.best_of_n = int(config.best_of_n)
    if int(getattr(config, "retrieval_k", 0) or 0) > 0 and hasattr(model, "skeleton_bank"):
        try:
            from slm_training.harnesses.model_build.data import load_train_records
            from slm_training.retrieval import build_skeleton_bank

            if config.train_dir.exists():
                model.skeleton_bank = build_skeleton_bank(load_train_records(config.train_dir))
        except Exception:  # noqa: BLE001
            pass
    return model


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

    if name in {"twotower", "two_tower", "two-tower"}:
        from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

        if checkpoint and checkpoint.exists():
            loaded = TwoTowerModel.from_checkpoint(checkpoint, device=config.device)
            return apply_runtime_overrides(loaded, config)

        backend = (config.context_backend or "scratch").lower()
        freeze = config.freeze_context
        if backend in {"hf", "huggingface", "transformers"} and not freeze:
            # Match CLI: HF freezes by default unless explicitly disabled.
            freeze = True
        tt_cfg = TwoTowerConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            context_layers=config.context_layers,
            denoiser_layers=config.denoiser_layers,
            mask_min=config.mask_min,
            mask_max=config.mask_max,
            gen_steps=config.gen_steps,
            context_backend=backend,
            hf_model_name=config.hf_model_name,
            freeze_context=freeze,
            local_files_only=config.local_files_only,
            grammar_constrained=config.grammar_constrained,
            grammar_top_k=config.grammar_top_k,
            structural_bias=config.structural_bias,
            grammar_ltr_repair=config.grammar_ltr_repair,
            grammar_ltr_max_tokens=config.grammar_ltr_max_tokens,
            grammar_finalize_validate=getattr(config, "grammar_finalize_validate", False),
            ltr_loss_weight=getattr(config, "ltr_loss_weight", 0.5),
            fidelity_loss_weight=getattr(config, "fidelity_loss_weight", 0.0),
            grammar_ltr_primary=config.grammar_ltr_primary,
            design_md_in_context=(
                True
                if config.design_md_in_context is None
                else bool(config.design_md_in_context)
            ),
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
            mdlm_schedule=bool(getattr(config, "mdlm_schedule", False)),
            mdlm_eps=float(getattr(config, "mdlm_eps", 1e-3) or 1e-3),
            use_compile=getattr(config, "use_compile", False),
            compile_mode=getattr(config, "compile_mode", "default"),
            use_amp=getattr(config, "use_amp", False),
            cache_context=getattr(config, "cache_context", True),
            fuse_ltr_loss=getattr(config, "fuse_ltr_loss", True),
            grammar_fastpath=getattr(config, "grammar_fastpath", True),
            grammar_fastpath_mode=getattr(config, "grammar_fastpath_mode", "hybrid"),
            grammar_draft_window=int(getattr(config, "grammar_draft_window", 8) or 8),
            fastpath_aux_weight=getattr(config, "fastpath_aux_weight", 0.0),
            fastpath_gate_threshold=float(
                getattr(config, "fastpath_gate_threshold", 0.5) or 0.5
            ),
            suffix_rollback_window=int(
                getattr(config, "suffix_rollback_window", 0) or 0
            ),
            remask_use_gate=bool(getattr(config, "remask_use_gate", False)),
            remask_use_entropy=bool(getattr(config, "remask_use_entropy", False)),
            visible_corrupt_rate=float(
                getattr(config, "visible_corrupt_rate", 0.0) or 0.0
            ),
            trust_gate_train=bool(getattr(config, "trust_gate_train", False)),
            grammar_prefer_structural=getattr(config, "grammar_prefer_structural", True),
            grammar_trust_model=getattr(config, "grammar_trust_model", False),
            grammar_sample_decode=getattr(config, "grammar_sample_decode", False),
            grammar_sample_temperature=getattr(
                config, "grammar_sample_temperature", 0.8
            ),
            grammar_block_decode=getattr(config, "grammar_block_decode", False),
            grammar_block_size=getattr(config, "grammar_block_size", 32),
            seed=config.seed,
        )
        model = TwoTowerModel.from_records(records, config=tt_cfg, device=config.device)
        if int(getattr(config, "retrieval_k", 0) or 0) > 0:
            from slm_training.retrieval import build_skeleton_bank

            model.skeleton_bank = build_skeleton_bank(records)
        return model

    raise ValueError(f"unknown model_name {config.model_name!r}")
