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
# Harnesses must stop themselves early enough to serialize checkpoints and
# result metadata before the command-level interrupt fires.
HARNESS_FINALIZATION_RESERVE_SECONDS: Final = 15
MAX_HARNESS_WALL_SECONDS: Final = (
    INTERRUPT_AFTER_SECONDS - HARNESS_FINALIZATION_RESERVE_SECONDS
)
MAX_HARNESS_WALL_MINUTES: Final = MAX_HARNESS_WALL_SECONDS / 60
HF_JOB_TIMEOUT: Final = f"{MAX_RUN_MINUTES}m"
CHANGED_TEST_WORKERS: Final = 4
VERCEL_FUNCTION_INCLUDE_FILES: Final = (
    "docs/design/**",
    "docs/MODEL_CARD.md",
    "src/slm_training/resources/checkpoints/playground_demo/**",
)
VERCEL_FUNCTION_EXCLUDE_FILES: Final = (
    ".venv/**",
    ".vercel/**",
    "node_modules/**",
    "outputs/**",
    "tests/**",
    "src/apps/**",
    "**/governance/records.jsonl",
    # Historical semantic/schema-role corpora are provenance, not runtime data.
    "**/train/e177_*/**",
    "**/train/e21[48]_*/**",
    "**/train/e297_*/**",
    "**/train/scope_graded_v1/**",
)

# Active, source-controlled corpora. Historical snapshots remain immutable
# evidence but fail the canonical marker guard and are never CLI defaults.
DEFAULT_TRAIN_DATA_DIR: Final = Path(
    "src/slm_training/resources/data/train/e937_role_safe_all_targets_v2"
)
DEFAULT_EVAL_DATA_DIR: Final = Path(
    "src/slm_training/resources/data/eval/e938_role_safe_all_targets_v2"
)
DEFAULT_CONTEXT_BACKEND: Final = "scratch"

# Template markers are codec identities, never semantic supervision.  Keep this
# policy beside every other user-facing lever so there is one discoverable
# source of truth for training and decode configuration.
TEMPLATE_MARKERS_ARE_OPAQUE: Final = True
# A difficulty-scored rebuild is a hard-example selection operation, not a
# metadata-only annotation. Records in the easiest NLL tail are rejected by
# the data harness before training.
DIFFICULTY_EASY_TAIL_FRACTION: Final = 0.2
DEFAULT_OUTPUT_TOKENIZER: Final = "lexer"
DEFAULT_DECODE_TIMEOUT_SECONDS: Final = 12.0
CHECKPOINT_DECLARED_POLICY: Final = "checkpoint_declared"
STRICT_COMPILER_TREE_POLICY_ID: Final = "strict_compiler_tree"
# Every evaluation enforces the symbol-only completion boundary independently
# of checkpoint provenance. The compiler-tree bundle is the canonical default.
DEFAULT_EVALUATION_POLICY: Final = STRICT_COMPILER_TREE_POLICY_ID
STRICT_EVALUATION_POLICY: Final = {
    "grammar_constrained": True,
    "grammar_ltr_primary": True,
    "grammar_finalize_validate": True,
    "slot_contract_constrained_decode": True,
    "honest_slot_contract": True,
    "allow_unconstrained_fallback": False,
}
STRICT_COMPILER_TREE_POLICY: Final = {
    **STRICT_EVALUATION_POLICY,
    "output_tokenizer": DEFAULT_OUTPUT_TOKENIZER,
    "compiler_decode_mode": "tree",
    # Structural AST-plan scoring is part of the policy, not a caller-owned
    # pair of optional flags. This consumes no marker names or free-form text.
    "semantic_plan_decode_weight": 4.0,
    "semantic_plan_margin_decode_weight": 2.0,
}
PROHIBITED_TEMPLATE_SEMANTIC_LEVERS: Final = {
    "namespace_augment": "renames opaque markers into user-defined namespaces",
    "prompt_semantic_role_contract": "adds user-defined marker labels to training prompts",
    "semantic_role_contract_in_context": "exposes user-defined marker labels",
    "semantic_role_decode_weight": "scores user-defined marker labels",
    "semantic_role_schema_candidates": "maps user-defined marker labels to schema",
    "schema_role_slot_decode_weight": "maps user-defined marker labels to schema",
    "slot_coverage_close_decode_weight": "uses marker-label-derived schema reachability",
    "semantic_plan_repeated_slot_margin_decode_weight": (
        "groups markers by user-defined namespace labels"
    ),
}

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
    "schema_value_decode_weight",
    "schema_enum_close_decode_weight",
    "schema_open_decode_weight",
    "schema_opaque_decode_weight",
    "schema_opaque_close_decode_weight",
    "semantic_plan_seed_decode_weight",
    "semantic_plan_inline_decode_weight",
    "semantic_plan_binding_decode_weight",
    "semantic_plan_root_decode_weight",
    "semantic_plan_root_margin_decode_weight",
    "semantic_plan_repeated_array_close_margin_decode_weight",
    "visible_reference_decode_weight",
)
_DUAL_PATH_DECODE_LEVERS: Final = (
    "component_inventory_decode_weight",
    "component_plan_decode_weight",
    "slot_component_decode_weight",
    "required_slot_margin_decode_weight",
    "semantic_plan_decode_weight",
    "semantic_plan_margin_decode_weight",
    "semantic_plan_typed_array_nonempty_margin_decode_weight",
    "semantic_plan_typed_array_item_margin_decode_weight",
    "root_reference_arity_decode_weight",
    "root_reference_identity_decode_weight",
)
_COMPILER_PATH_DECODE_LEVERS: Final = (
    "compiler_schema_component_types",
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
    "root_reference_identity_loss_weight": (_CHOICE, _LEXER_COMPILER),
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

# Runtime prerequisites belong in the same registry as codec support.  Keeping
# them here makes construction fail before a harness creates run artifacts;
# model-side checks remain as defense in depth for callers that bypass the
# canonical factory.  Each tuple is an OR of complete companion configurations.
_SLOT_CONTRACT_DECODE: Final = (
    {
        "slot_contract_in_context": True,
        "slot_contract_constrained_decode": True,
    },
    {
        "slot_contract_in_context": True,
        "template_fill_decode": True,
    },
)
SLOT_CONTRACT_DECODE_LEVERS: Final = (
    "semantic_plan_typed_array_nonempty_margin_decode_weight",
    "semantic_plan_typed_array_item_margin_decode_weight",
    "required_slot_margin_decode_weight",
)
LEVER_COMPANION_REQUIREMENTS: Final = {
    **{name: _SLOT_CONTRACT_DECODE for name in SLOT_CONTRACT_DECODE_LEVERS},
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


def incompatible_lever_requirements(
    config: Any,
) -> dict[str, tuple[dict[str, Any], ...]]:
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
        isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0.0
    )


def untrained_decode_levers(config: Any) -> dict[str, tuple[str, ...]]:
    """Return enabled learned decode levers without a trained owning objective."""
    return {
        name: owners
        for name, owners in TRAINED_DECODE_REQUIREMENTS.items()
        if _positive_numeric(config, name)
        and not any(_positive_numeric(config, owner) for owner in owners)
    }


def missing_lever_companions(
    config: Any,
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Return enabled levers whose required runtime inputs cannot be built."""
    return {
        name: requirements
        for name, requirements in LEVER_COMPANION_REQUIREMENTS.items()
        if _positive_numeric(config, name)
        and not any(_matches_requirement(config, item) for item in requirements)
    }


def lever_configuration_errors(
    config: Any, *, require_trained_decode: bool = False
) -> tuple[str, ...]:
    """Return every canonical lever-contract violation for ``config``."""
    errors: list[str] = []
    if (
        TEMPLATE_MARKERS_ARE_OPAQUE
        and getattr(config, "symbol_anonymization", True) is False
    ):
        errors.append(
            "symbol_anonymization=False is prohibited because template markers are opaque"
        )
    prohibited = {
        name: reason
        for name, reason in PROHIBITED_TEMPLATE_SEMANTIC_LEVERS.items()
        if bool(getattr(config, name, False))
    }
    errors.extend(
        f"{name} is prohibited because template markers are opaque: {reason}"
        for name, reason in prohibited.items()
    )
    incompatible = incompatible_lever_requirements(config)
    if incompatible:
        errors.append(f"unsupported enabled levers: {', '.join(incompatible)}")
    missing_companions = missing_lever_companions(config)
    errors.extend(
        f"{name} requires one companion configuration: "
        f"{', '.join(str(_requirement_json(item)) for item in requirements)}"
        for name, requirements in missing_companions.items()
    )
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
        if item.name in LEVER_COMPANION_REQUIREMENTS:
            catalog[item.name]["requires_companion_configuration"] = [
                _requirement_json(requirement)
                for requirement in LEVER_COMPANION_REQUIREMENTS[item.name]
            ]
        if item.name in PROHIBITED_TEMPLATE_SEMANTIC_LEVERS:
            catalog[item.name]["prohibited"] = True
            catalog[item.name]["prohibited_reason"] = (
                PROHIBITED_TEMPLATE_SEMANTIC_LEVERS[item.name]
            )
        if item.name == "symbol_anonymization":
            catalog[item.name]["required"] = True
            catalog[item.name]["required_reason"] = (
                "template markers are opaque codec identities"
            )
        if (
            item.name in checkpoint_defaults
            and checkpoint_defaults[item.name] != default
        ):
            catalog[item.name]["checkpoint_default"] = _json_value(
                checkpoint_defaults[item.name]
            )
            catalog[item.name]["contexts_diverge"] = True
    for name, reason in PROHIBITED_TEMPLATE_SEMANTIC_LEVERS.items():
        catalog.setdefault(
            name,
            {
                "category": "data",
                "default": False,
                "type": "bool",
                "source": "slm_training.levers.PROHIBITED_TEMPLATE_SEMANTIC_LEVERS",
                "prohibited": True,
                "prohibited_reason": reason,
            },
        )
    catalog["max_wall_minutes"].update(
        {
            "default": MAX_HARNESS_WALL_MINUTES,
            "maximum": MAX_HARNESS_WALL_MINUTES,
            "derived": {
                "hard_cap_minutes": MAX_RUN_MINUTES,
                "interrupt_after_seconds": INTERRUPT_AFTER_SECONDS,
                "kill_grace_seconds": KILL_GRACE_SECONDS,
                "finalization_reserve_seconds": HARNESS_FINALIZATION_RESERVE_SECONDS,
                "total_seconds": MAX_RUN_SECONDS,
                "hf_job_timeout": HF_JOB_TIMEOUT,
            },
            "source": "slm_training.levers.MAX_HARNESS_WALL_MINUTES",
        }
    )
    catalog["changed_test_workers"] = {
        "category": "run",
        "default": CHANGED_TEST_WORKERS,
        "type": "int",
        "source": "slm_training.levers.CHANGED_TEST_WORKERS",
    }
    catalog["vercel_function_include_files"] = {
        "category": "run",
        "default": list(VERCEL_FUNCTION_INCLUDE_FILES),
        "type": "tuple[str, ...]",
        "source": "slm_training.levers.VERCEL_FUNCTION_INCLUDE_FILES",
    }
    catalog["template_markers_are_opaque"] = {
        "category": "data",
        "default": TEMPLATE_MARKERS_ARE_OPAQUE,
        "type": "bool",
        "source": "slm_training.levers.TEMPLATE_MARKERS_ARE_OPAQUE",
    }
    catalog["default_train_data_dir"] = {
        "category": "data",
        "default": str(DEFAULT_TRAIN_DATA_DIR),
        "type": "Path",
        "source": "slm_training.levers.DEFAULT_TRAIN_DATA_DIR",
    }
    catalog["default_eval_data_dir"] = {
        "category": "data",
        "default": str(DEFAULT_EVAL_DATA_DIR),
        "type": "Path",
        "source": "slm_training.levers.DEFAULT_EVAL_DATA_DIR",
    }
    catalog["default_context_backend"] = {
        "category": "model",
        "default": DEFAULT_CONTEXT_BACKEND,
        "type": "str",
        "choices": ["scratch", "hf"],
        "source": "slm_training.levers.DEFAULT_CONTEXT_BACKEND",
    }
    from slm_training.harnesses.model_build.eval_policy import EVALUATION_POLICIES

    catalog["evaluation_policy"].update(
        {
            "default": DEFAULT_EVALUATION_POLICY,
            "config_default": CHECKPOINT_DECLARED_POLICY,
            "choices": sorted(EVALUATION_POLICIES),
            "source": "slm_training.levers.DEFAULT_EVALUATION_POLICY",
        }
    )
    return catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List canonical OpenUI training levers."
    )
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
