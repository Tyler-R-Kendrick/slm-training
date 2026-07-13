"""Factory for model plug-ins."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.plugin import ModelPlugin, StubModel


def build_model(
    config: ModelBuildConfig,
    records: list[ExampleRecord],
    checkpoint: Path | None = None,
) -> Any:
    name = (config.model_name or "twotower").lower()
    if name == "stub":
        model: ModelPlugin = StubModel(noise_rate=config.noise_rate, seed=config.seed)
        if checkpoint and checkpoint.exists():
            model.load(checkpoint)
        return model

    if name in {"twotower", "two_tower", "two-tower"}:
        from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

        if checkpoint and checkpoint.exists():
            return TwoTowerModel.from_checkpoint(checkpoint, device=config.device)

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
            ltr_loss_weight=getattr(config, "ltr_loss_weight", 0.5),
            grammar_ltr_primary=config.grammar_ltr_primary,
            design_md_in_context=config.design_md_in_context,
            design_md_budget=config.design_md_budget,
            seed=config.seed,
        )
        return TwoTowerModel.from_records(records, config=tt_cfg, device=config.device)

    raise ValueError(f"unknown model_name {config.model_name!r}")
