"""Preflight effects for the existing continuous bank, not a second knob registry.

Only effects whose execution location is known are admissible for a new typed
design. Unknown keys fail closed; adding a dataclass field is not wiring proof.
Canonical capability/companion/constraint validators remain in ``levers``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.levers import (
    CAPACITY_SCALING_LEVERS,
    require_constrained_production_config,
    require_valid_lever_configuration,
)


@dataclass(frozen=True)
class LeverEffect:
    category: str
    execution_owner: str
    endpoints: tuple[str, ...]
    resource_effect: str
    runtime_dependency: str = "twotower"


_TRAIN = (
    "lr",
    "ltr_prefix_loss_weight",
    "component_token_loss_weight",
    "component_edge_token_loss_weight",
    "compiler_decision_token_loss_weight",
    "structure_token_loss_weight",
    "typed_family_balance_loss_weight",
    "ltr_tail_loss_weight",
    "compiler_alignment_loss_weight",
    "compiler_alignment_margin",
    "compiler_alignment_stratified",
    "compiler_alignment_semantic_exhaustive",
    "compiler_alignment_kind_filter",
    "component_plan_loss_weight",
    "solver_energy_loss_weight",
    "legal_edit_hazard_loss_weight",
    "component_edge_loss_weight",
    "component_edge_alignment_loss_weight",
    "component_inventory_loss_weight",
    "binder_topology_loss_weight",
    "binder_component_plan_loss_weight",
    "binder_arity_loss_weight",
    "slot_component_loss_weight",
    "symbol_boundary_loss_weight",
    "design_md_dropout",
    "fidelity_loss_weight",
    "semantic_contrast_loss_weight",
    "semantic_contrast_margin",
    "semantic_contrast_fraction",
    "symbol_slot_augmentation",
    "mask_pattern",
    "ambiguity_only_loss",
)
_DECODE = (
    "grammar_completion_bounds",
    "grammar_equivalence_cache",
    "grammar_draft_window",
    "compact_active_canvas",
    "component_plan_decode_weight",
    "solver_energy_decode_weight",
    "legal_edit_hazard_decode_weight",
    "component_edge_decode_weight",
    "component_inventory_decode_weight",
    "binder_topology_decode_weight",
    "binder_component_plan_decode_weight",
    "binder_arity_decode_weight",
    "slot_component_decode_weight",
    "compiler_decode_mode",
)
_DATA = (
    "train_version",
    "train_dir",
    "semantic_contrast_dir",
    "mixture_weights",
    "mixture_sampling_policy",
    "mixture_exposure_target_profile",
    "mixture_total_decision_budget",
    "mixture_per_root_cap",
    "mixture_per_template_cap",
    "mixture_max_importance_weight",
)


def bank_effects() -> dict[str, LeverEffect]:
    """Metadata overlay consumed by the existing bank category owner."""
    train = LeverEffect(
        "training",
        "models/twotower.py::training_loss",
        ("denoising_loss", "decoded_quality"),
        "training_compute",
    )
    decode = LeverEffect(
        "decode",
        "models/twotower.py::generate",
        ("decoded_quality", "latency"),
        "decode_compute",
    )
    data = LeverEffect(
        "data", "harnesses/model_build/train_loop.py", train.endpoints, "data_exposure"
    )
    effects = {
        **dict.fromkeys(_TRAIN, train),
        **dict.fromkeys(_DECODE, decode),
        **dict.fromkeys(_DATA, data),
    }
    for key in ("steps", "batch_size", "grad_accum_steps"):
        effects[key] = LeverEffect(
            "training",
            "harnesses/model_build/train_loop.py",
            train.endpoints,
            "training_duration",
        )
    for key in CAPACITY_SCALING_LEVERS:
        effects[key] = LeverEffect(
            "model",
            "harnesses/model_build/factory.py",
            train.endpoints,
            "charged_parameter_capacity",
        )
    return effects


def validate_lever_effects(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    varied_fields: tuple[str, ...],
    endpoint: str,
) -> None:
    """Structural no-op checks, not an empirical no-effect claim."""
    effects = bank_effects()
    for key in varied_fields:
        effect = effects.get(key)
        if effect is None:
            raise ValueError(f"unregistered executable lever: {key}")
        if key not in control or key not in candidate or control[key] == candidate[key]:
            raise ValueError(f"unchanged/unresolved lever: {key}")
        if endpoint not in effect.endpoints:
            raise ValueError(f"{key} cannot affect primary endpoint {endpoint}")
    config = _runtime_config(candidate)
    _require_live_objectives(varied_fields, config)
    require_constrained_production_config(config)
    require_valid_lever_configuration(config, require_trained_decode=True)
    if getattr(config, "model_name", "twotower") != "twotower":
        raise ValueError("bank effect evidence currently covers TwoTower only")


def _runtime_config(candidate: Mapping[str, Any]) -> SimpleNamespace:
    # Materialize actual runtime defaults, rather than inferred prefix semantics.
    names = {field.name for field in fields(ModelBuildConfig)}
    supplied = {key: value for key, value in candidate.items() if key in names}
    supplied.setdefault("train_dir", Path("."))
    defaults = ModelBuildConfig(**supplied)
    values = {field.name: getattr(defaults, field.name) for field in fields(defaults)}
    values.update(candidate)
    return SimpleNamespace(**values)


def _require_live_objectives(varied_fields, config) -> None:
    # These secondary parameters are read only inside their owning objective.
    dependencies = {
        "compiler_alignment_margin": "compiler_alignment_loss_weight",
        "compiler_alignment_stratified": "compiler_alignment_loss_weight",
        "compiler_alignment_kind_filter": "compiler_alignment_loss_weight",
        "compiler_alignment_semantic_exhaustive": "compiler_alignment_loss_weight",
        "semantic_contrast_margin": "semantic_contrast_loss_weight",
        "semantic_contrast_fraction": "semantic_contrast_loss_weight",
    }
    for field in varied_fields:
        owner = dependencies.get(field)
        if owner and not getattr(config, owner, 0) > 0:
            raise ValueError(f"inactive lever {field}: requires enabled {owner}")


def validate_compiled_effects(
    commands: list[list[str]],
    candidate: Mapping[str, Any],
    varied_fields: tuple[str, ...],
) -> None:
    """Require changed typed fields to survive the actual command compiler.

    This is an argv-bound wiring obligation, not a claim that model behavior
    improved. CLI/runtime and real parameter-effect regressions supply deeper proof.
    """
    train = [
        cmd for cmd in commands if len(cmd) > 2 and cmd[2] == "scripts.train_model"
    ]
    if len(train) != 1:
        raise ValueError("expected one compiled training invocation")
    argv = train[0][3:]
    for key in varied_fields:
        if key == "train_dir" and "--train-version" in argv:
            from scripts.train_model import resolve_published_train_version

            actual, _ = resolve_published_train_version(
                argv[argv.index("--train-version") + 1]
            )
            if Path(actual).resolve() != Path(candidate[key]).resolve():
                raise ValueError(
                    "compiled training snapshot differs from resolved config"
                )
            continue
        flag = {"grad_accum_steps": "--grad-accum"}.get(
            key, "--" + key.replace("_", "-")
        )
        value = candidate[key]
        if isinstance(value, bool):
            expected = flag if value else "--no-" + flag[2:]
            if expected not in argv:
                raise ValueError(f"compiled lever missing: {key}")
        elif (
            flag not in argv
            or argv.index(flag) + 1 >= len(argv)
            or argv[argv.index(flag) + 1] != str(value)
        ):
            raise ValueError(f"compiled lever missing or changed: {key}")
