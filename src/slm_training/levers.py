"""Canonical experiment-lever discovery and repository run policy.

Change run policy here. Model/training lever defaults remain owned by
``ModelBuildConfig`` and are exposed here as one searchable catalog so scripts,
agents, and the web layer do not maintain parallel lists.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, Final


MAX_RUN_MINUTES: Final = 2
KILL_GRACE_SECONDS: Final = 10
MAX_RUN_SECONDS: Final = MAX_RUN_MINUTES * 60
INTERRUPT_AFTER_SECONDS: Final = MAX_RUN_SECONDS - KILL_GRACE_SECONDS
HF_JOB_TIMEOUT: Final = f"{MAX_RUN_MINUTES}m"

# Applicability lives beside discovery so CLIs and harness validation cannot
# drift from the human-visible lever catalog.
LEVER_OUTPUT_TOKENIZERS: Final = {
    "root_reference_arity_loss_weight": frozenset({"choice"}),
    "root_reference_arity_decode_weight": frozenset({"choice"}),
    "root_reference_identity_loss_weight": frozenset({"choice"}),
    "root_reference_identity_decode_weight": frozenset({"choice"}),
}
LEVER_MODELS: Final = {
    name: frozenset({"twotower"}) for name in LEVER_OUTPUT_TOKENIZERS
}


def incompatible_output_tokenizer_levers(config: Any) -> list[str]:
    """Return enabled levers that cannot execute for the selected codec."""
    output_tokenizer = getattr(config, "output_tokenizer", None)
    return [
        name
        for name, supported in LEVER_OUTPUT_TOKENIZERS.items()
        if isinstance((value := getattr(config, name, 0.0)), (int, float))
        and not isinstance(value, bool)
        and value != 0.0
        and output_tokenizer not in supported
    ]


def incompatible_model_levers(config: Any) -> list[str]:
    """Return enabled levers that cannot execute for the selected model."""
    model_name = getattr(config, "model_name", None)
    if model_name is None:
        return []
    return [
        name
        for name, supported in LEVER_MODELS.items()
        if isinstance((value := getattr(config, name, 0.0)), (int, float))
        and not isinstance(value, bool)
        and value != 0.0
        and model_name not in supported
    ]


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _category(name: str) -> str:
    if name in {"max_wall_minutes", "steps", "target_token_budget"}:
        return "run"
    if "decode" in name or name.startswith(("grammar_", "compiler_", "solver_")):
        return "decode"
    if name.endswith("_loss_weight") or name in {"lr", "batch_size", "optimizer_name"}:
        return "training"
    if name.startswith(("mixture_", "replay_")):
        return "data"
    if name.startswith(("eval_", "rico_eval_", "loss_eval_")):
        return "evaluation"
    return "model"


def lever_catalog() -> dict[str, dict[str, Any]]:
    """Return every user-facing build lever from its canonical dataclass."""
    from slm_training.harnesses.model_build.config import ModelBuildConfig
    from slm_training.models.twotower import TwoTowerConfig

    catalog: dict[str, dict[str, Any]] = {}
    checkpoint_defaults = {
        item.name: item.default
        for item in fields(TwoTowerConfig)
        if item.default is not MISSING
    }
    for item in fields(ModelBuildConfig):
        if item.default is not MISSING:
            default = item.default
        elif item.default_factory is not MISSING:
            default = item.default_factory()
        else:
            default = None
        catalog[item.name] = {
            "category": _category(item.name),
            "default": _json_value(default),
            "type": str(item.type),
            "source": "slm_training.harnesses.model_build.config.ModelBuildConfig",
        }
        if item.name in LEVER_OUTPUT_TOKENIZERS:
            catalog[item.name]["supported_output_tokenizers"] = sorted(
                LEVER_OUTPUT_TOKENIZERS[item.name]
            )
            catalog[item.name]["supported_models"] = sorted(LEVER_MODELS[item.name])
        if item.name in checkpoint_defaults and checkpoint_defaults[item.name] != default:
            catalog[item.name]["checkpoint_default"] = _json_value(
                checkpoint_defaults[item.name]
            )
            catalog[item.name]["contexts_diverge"] = True
    catalog["max_wall_minutes"].update(
        {
            "default": MAX_RUN_MINUTES,
            "maximum": MAX_RUN_MINUTES,
            "derived": {
                "interrupt_after_seconds": INTERRUPT_AFTER_SECONDS,
                "kill_grace_seconds": KILL_GRACE_SECONDS,
                "total_seconds": MAX_RUN_SECONDS,
                "hf_job_timeout": HF_JOB_TIMEOUT,
            },
            "source": "slm_training.levers.MAX_RUN_MINUTES",
        }
    )
    return catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List canonical OpenUI training levers.")
    parser.add_argument("--category", default=None)
    args = parser.parse_args(argv)
    catalog = lever_catalog()
    if args.category:
        catalog = {
            name: spec
            for name, spec in catalog.items()
            if spec["category"] == args.category
        }
    print(json.dumps(catalog, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
