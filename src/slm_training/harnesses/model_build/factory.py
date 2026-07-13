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

        tt_cfg = TwoTowerConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            context_layers=config.context_layers,
            denoiser_layers=config.denoiser_layers,
            mask_min=config.mask_min,
            mask_max=config.mask_max,
            gen_steps=config.gen_steps,
            freeze_context=config.freeze_context,
            seed=config.seed,
        )
        return TwoTowerModel.from_records(records, config=tt_cfg, device=config.device)

    raise ValueError(f"unknown model_name {config.model_name!r}")
