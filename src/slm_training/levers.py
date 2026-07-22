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
# drift from the human-visible lever catalog. Each tuple is an OR of complete
# supported configurations; fields inside one configuration are ANDed.
_CHOICE: Final = {"model_name": "twotower", "output_tokenizer": "choice"}
_LEXER_COMPILER: Final = {
    "model_name": "twotower",
    "output_tokenizer": "lexer",
    "compiler_decode_mode": frozenset({"restricted", "tree"}),
}
_CHOICE_ONLY_DECODE_LEVERS: Final = (
    "slot_coverage_close_decode_weight",
    "schema_value_decode_weight",
    "schema_enum_close_decode_weight",
    "schema_open_decode_weight",
    "schema_opaque_decode_weight",
    "schema_opaque_close_decode_weight",
    "schema_role_slot_decode_weight",
    "required_slot_margin_decode_weight",
    "semantic_plan_decode_weight",
    "semantic_plan_margin_decode_weight",
    "semantic_plan_seed_decode_weight",
    "semantic_plan_inline_decode_weight",
    "semantic_plan_binding_decode_weight",
    "semantic_plan_root_decode_weight",
    "semantic_plan_root_margin_decode_weight",
    "semantic_plan_repeated_array_close_margin_decode_weight",
    "semantic_plan_repeated_slot_margin_decode_weight",
    "semantic_plan_typed_array_nonempty_margin_decode_weight",
    "semantic_plan_typed_array_item_margin_decode_weight",
    "visible_reference_decode_weight",
    "root_reference_identity_decode_weight",
)
_DUAL_PATH_DECODE_LEVERS: Final = (
    "component_inventory_decode_weight",
    "component_plan_decode_weight",
    "slot_component_decode_weight",
    "semantic_role_decode_weight",
    "root_reference_arity_decode_weight",
)
_COMPILER_PATH_DECODE_LEVERS: Final = (
    "component_edge_decode_weight",
    "binder_component_plan_decode_weight",
    "binder_topology_decode_weight",
    "binder_arity_decode_weight",
)
LEVER_REQUIREMENTS: Final = {
    **{name: (_CHOICE,) for name in _CHOICE_ONLY_DECODE_LEVERS},
    **{name: (_CHOICE, _LEXER_COMPILER) for name in _DUAL_PATH_DECODE_LEVERS},
    **{name: (_LEXER_COMPILER,) for name in _COMPILER_PATH_DECODE_LEVERS},
    "root_reference_arity_loss_weight": (_CHOICE, _LEXER_COMPILER),
    "root_reference_identity_loss_weight": (_CHOICE,),
}

# A decode head is usable only when its checkpoint trained that head. Without
# this contract a non-zero decode weight can instantiate or select random
# parameters while the run reports the lever as enabled.
TRAINED_DECODE_REQUIREMENTS: Final = {
    "component_inventory_decode_weight": ("component_inventory_loss_weight",),
    "component_plan_decode_weight": ("component_plan_loss_weight",),
    "slot_component_decode_weight": ("slot_component_loss_weight",),
    "component_edge_decode_weight": (
        "component_edge_loss_weight",
        "component_edge_alignment_loss_weight",
    ),
    "binder_component_plan_decode_weight": ("binder_component_plan_loss_weight",),
    "binder_topology_decode_weight": ("binder_topology_loss_weight",),
    "binder_arity_decode_weight": ("binder_arity_loss_weight",),
    "root_reference_arity_decode_weight": ("root_reference_arity_loss_weight",),
    "root_reference_identity_decode_weight": ("root_reference_identity_loss_weight",),
}


def _canonical_capability_value(field: str, value: Any) -> Any:
    normalized = str(value or "").lower()
    if field == "output_tokenizer":
        if normalized in {"choice", "choices", "choice_codec"}:
            return "choice"
        if normalized in {"lexer", "dsl", "dsl_native"}:
            return "lexer"
    return value


def _matches_requirement(config: Any, requirement: dict[str, Any]) -> bool:
    for field, expected in requirement.items():
        if field == "model_name" and not hasattr(config, field):
            continue
        actual = _canonical_capability_value(field, getattr(config, field, None))
        if isinstance(expected, frozenset):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def incompatible_lever_requirements(config: Any) -> dict[str, tuple[dict[str, Any], ...]]:
    """Return enabled levers for which no executable configuration exists."""
    return {
        name: requirements
        for name, requirements in LEVER_REQUIREMENTS.items()
        if isinstance((value := getattr(config, name, 0.0)), (int, float))
        and not isinstance(value, bool)
        and value != 0.0
        and not any(_matches_requirement(config, item) for item in requirements)
    }


def _positive_numeric(config: Any, field: str) -> bool:
    value = getattr(config, field, 0.0)
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 0.0
    )


def untrained_decode_levers(config: Any) -> dict[str, tuple[str, ...]]:
    """Return enabled learned decode levers without a trained owning objective."""
    return {
        name: owners
        for name, owners in TRAINED_DECODE_REQUIREMENTS.items()
        if _positive_numeric(config, name)
        and not any(_positive_numeric(config, owner) for owner in owners)
    }


def lever_configuration_errors(
    config: Any, *, require_trained_decode: bool = False
) -> tuple[str, ...]:
    """Return every canonical lever-contract violation for ``config``."""
    errors: list[str] = []
    incompatible = incompatible_lever_requirements(config)
    if incompatible:
        errors.append(f"unsupported enabled levers: {', '.join(incompatible)}")
    if require_trained_decode:
        untrained = untrained_decode_levers(config)
        errors.extend(
            f"{name} requires a trained checkpoint objective: {' or '.join(owners)}"
            for name, owners in untrained.items()
        )
    return tuple(errors)


def require_valid_lever_configuration(
    config: Any, *, require_trained_decode: bool = False, context: str = "config"
) -> None:
    """Fail before execution when a lever cannot have its advertised effect."""
    errors = lever_configuration_errors(
        config, require_trained_decode=require_trained_decode
    )
    if errors:
        raise ValueError(
            f"{context} has invalid enabled levers: {'; '.join(errors)}; "
            "inspect `python -m slm_training.levers` for supported configurations"
        )


def _requirement_json(requirement: dict[str, Any]) -> dict[str, Any]:
    return {name: _json_value(value) for name, value in requirement.items()}


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, frozenset):
        return sorted(_json_value(item) for item in value)
    if isinstance(value, tuple):
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
        if item.name in LEVER_REQUIREMENTS:
            catalog[item.name]["supported_configurations"] = [
                _requirement_json(requirement)
                for requirement in LEVER_REQUIREMENTS[item.name]
            ]
        if item.name in TRAINED_DECODE_REQUIREMENTS:
            catalog[item.name]["requires_trained_any"] = list(
                TRAINED_DECODE_REQUIREMENTS[item.name]
            )
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
    from slm_training.harnesses.model_build.eval_policy import EVALUATION_POLICIES

    catalog["evaluation_policy"].update(
        {
            "choices": sorted(EVALUATION_POLICIES),
            "source": "slm_training.harnesses.model_build.eval_policy.EVALUATION_POLICIES",
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
