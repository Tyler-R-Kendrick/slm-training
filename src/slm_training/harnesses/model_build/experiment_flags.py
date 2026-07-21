"""Bridge ModelBuildConfig ↔ OpenFeature experiment levers."""

from __future__ import annotations

from typing import Any, Mapping

from slm_training.flags import (
    EvaluationDetails,
    FlagClient,
    InMemoryProvider,
    apply_experiment_flags,
    experiment_context,
    ruleset_from_mapping,
)
from slm_training.flags.apply import assignments_payload
from slm_training.flags.bootstrap import client_from_environ
from slm_training.flags.levers import LEVER_BY_KEY
from slm_training.harnesses.model_build.config import ModelBuildConfig


def apply_levers_from_mapping(
    config: ModelBuildConfig,
    levers: Mapping[str, Any],
    *,
    experiment_id: str | None = None,
    matrix: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[ModelBuildConfig, list[EvaluationDetails]]:
    """Seed an in-memory ruleset from an experiment row / mapping and apply it."""
    client = FlagClient(InMemoryProvider(ruleset_from_mapping(dict(levers))))
    ctx = experiment_context(
        run_id=config.run_id,
        experiment_id=experiment_id,
        matrix=matrix,
        model_name=config.model_name,
        context_backend=getattr(config, "context_backend", None),
    )
    return apply_experiment_flags(
        config, client=client, context=ctx, overrides=overrides
    )


def apply_levers_from_environ(
    config: ModelBuildConfig,
    *,
    overrides: Mapping[str, Any] | None = None,
    experiment_id: str | None = None,
    matrix: str | None = None,
) -> tuple[ModelBuildConfig, list[EvaluationDetails]]:
    """Apply ``OPENUI_FLAGS_JSON`` / ``OPENUI_FLAGS_PATH`` when configured."""
    client = client_from_environ()
    if client is None:
        return config, []
    ctx = experiment_context(
        run_id=config.run_id,
        experiment_id=experiment_id,
        matrix=matrix,
        model_name=config.model_name,
        context_backend=getattr(config, "context_backend", None),
    )
    return apply_experiment_flags(
        config, client=client, context=ctx, overrides=overrides
    )


def cli_lever_overrides(args: Any) -> dict[str, Any]:
    """Collect explicit CLI lever overrides (store_true / non-default strings)."""
    overrides: dict[str, Any] = {}
    if getattr(args, "verified_solver_decode", False):
        overrides["verified_solver_decode"] = True
    if getattr(args, "honest_slot_contract", False) or getattr(args, "ship_gates", False):
        overrides["honest_slot_contract"] = True
    mode = getattr(args, "compiler_decode_mode", None)
    if mode is not None and mode != "off":
        overrides["compiler_decode_mode"] = mode
    if getattr(args, "asap_decode", False):
        overrides["asap_decode"] = True
    # Numbers/strings that are registered: honor CLI when present on args.
    for key in ("solver_max_nodes", "solver_unknown_policy", "solver_certificate_mode"):
        if key in LEVER_BY_KEY and hasattr(args, key):
            overrides[key] = getattr(args, key)
    return overrides


__all__ = [
    "apply_levers_from_environ",
    "apply_levers_from_mapping",
    "assignments_payload",
    "cli_lever_overrides",
]
