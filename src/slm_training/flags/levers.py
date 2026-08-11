"""Curated experiment-lever registry (flag key == ModelBuildConfig field)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from slm_training.flags.api import FlagValueType


@dataclass(frozen=True)
class LeverSpec:
    key: str
    value_type: FlagValueType
    default: Any
    description: str
    # Optional OpenFeature flag metadata attached to every evaluation.
    metadata: dict[str, Any] | None = None


def _bool(key: str, default: bool, description: str) -> LeverSpec:
    return LeverSpec(
        key=key,
        value_type=FlagValueType.BOOLEAN,
        default=default,
        description=description,
        metadata={"lever": key},
    )


def _str(key: str, default: str, description: str) -> LeverSpec:
    return LeverSpec(
        key=key,
        value_type=FlagValueType.STRING,
        default=default,
        description=description,
        metadata={"lever": key},
    )


def _num(key: str, default: float | int, description: str) -> LeverSpec:
    return LeverSpec(
        key=key,
        value_type=FlagValueType.NUMBER,
        default=default,
        description=description,
        metadata={"lever": key},
    )


# Keep this list small and intentional — only levers that are meaningful as
# experiment toggles / progressive-delivery flags. Matrix-only hyperparams stay
# on ModelBuildConfig without a flag key.
LEVER_FLAGS: tuple[LeverSpec, ...] = (
    _bool(
        "verified_solver_decode",
        False,
        "VSS1-03: prune compiler forest via certified exact closure before ranking",
    ),
    _bool(
        "topology_verified_solver",
        False,
        "VSS3-03: topology finite-domain solver integration",
    ),
    _bool(
        "topology_capsule_solver",
        False,
        "Topology capsule solver (requires topology_verified_solver)",
    ),
    _bool(
        "honest_slot_contract",
        False,
        "Honest slot contract — no silent gold.placeholders channel",
    ),
    _bool("asap_decode", False, "A2: ASAp-style constraint-mass removal in MaskGIT"),
    _bool(
        "compiler_search_local_nogoods",
        False,
        "Compiler search records local nogoods during backtrack",
    ),
    _str(
        "compiler_decode_mode",
        "off",
        "Compiler-guided decode mode: off | forced | restricted | tree",
    ),
    _str(
        "solver_unknown_policy",
        "keep_and_rank",
        "Policy when exact closure returns unknown",
    ),
    _str(
        "solver_certificate_mode",
        "summary",
        "Solver certificate verbosity: none | summary | full",
    ),
    _num("solver_max_nodes", 512, "Exact-closure node budget"),
    _num("decode_min_content", 0, "A4 minimum content floor (0=off, -1=auto)"),
)

LEVER_BY_KEY: dict[str, LeverSpec] = {spec.key: spec for spec in LEVER_FLAGS}


def lever_defaults() -> dict[str, Any]:
    return {spec.key: spec.default for spec in LEVER_FLAGS}


def coerce_lever_value(spec: LeverSpec, value: Any) -> Any:
    if spec.value_type is FlagValueType.BOOLEAN:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "on", "yes"}:
                return True
            if lowered in {"0", "false", "off", "no"}:
                return False
        raise TypeError(f"{spec.key}: expected bool, got {type(value).__name__}")
    if spec.value_type is FlagValueType.STRING:
        return str(value)
    if spec.value_type is FlagValueType.NUMBER:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{spec.key}: expected number, got {type(value).__name__}")
        # Preserve ints when the registry default is int.
        if isinstance(spec.default, int) and not isinstance(spec.default, bool):
            return int(value)
        return float(value)
    raise TypeError(f"{spec.key}: unsupported type {spec.value_type}")


FlagKind = Literal["boolean", "string", "number", "object"]
