"""Fail-closed execution-mode routing contract for DSH5-09 (SLM-417).

The router ranks only modes that the compiler has already declared available.
All-arm outcomes are evaluator-only: they may create an offline oracle label,
but never enter :class:`ExecutionRouterInputV1` or authorize execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from slm_training.models.operator_policy_view import validate_no_forbidden_fields

__all__ = [
    "ExecutionModeV1",
    "ExecutionRouterDecisionV1",
    "ExecutionRouterInputV1",
    "ExecutionRouterOutcomeV1",
    "choose_execution_mode",
    "oracle_execution_mode",
]


class ExecutionModeV1(str, Enum):
    CONTROL = "control"
    PRIMITIVE_OPERATOR = "primitive_operator"
    BULK_OPERATOR = "bulk_operator"
    TRANSACTION = "transaction"
    GENERATE_SUBTREE = "generate_subtree"
    FULL_REGENERATE = "full_regenerate"
    DEFER = "defer"


_ROUTER_FORBIDDEN_FIELDS = frozenset(
    {
        "oracle_mode",
        "selected_mode",
        "outcome",
        "success",
        "replay",
        "blast_radius",
        "work",
        "latency",
        "wall_ms",
        "post_execution",
    }
)


def _validate_router_fields(value: Any, *, path: str = "$") -> None:
    validate_no_forbidden_fields(value)
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in _ROUTER_FORBIDDEN_FIELDS:
                raise ValueError(f"router_input.forbidden_field:{path}.{key}")
            _validate_router_fields(item, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _validate_router_fields(item, path=f"{path}[{index}]")


@dataclass(frozen=True)
class ExecutionRouterInputV1:
    """Only pre-routing numeric facts and compiler availability reach a scorer."""

    features: Mapping[str, float]
    available_modes: frozenset[ExecutionModeV1]
    forced_mode: ExecutionModeV1 | None = None
    schema: str = "execution_router_input/v1"

    def __post_init__(self) -> None:
        if ExecutionModeV1.DEFER in self.available_modes:
            raise ValueError("DEFER is a fallback, not a selectable available mode")
        if not self.available_modes and self.forced_mode is not None:
            raise ValueError("forced_mode requires compiler availability")
        if self.forced_mode is not None and self.forced_mode not in self.available_modes:
            raise ValueError("forced_mode must be compiler-available")
        if any(not isinstance(value, (float, int)) for value in self.features.values()):
            raise ValueError("router features must be numeric")
        _validate_router_fields(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "features": dict(sorted(self.features.items())),
            "available_modes": sorted(mode.value for mode in self.available_modes),
            "forced_mode": None if self.forced_mode is None else self.forced_mode.value,
        }


@dataclass(frozen=True)
class ExecutionRouterOutcomeV1:
    """Evaluator-only all-arm result; never a router feature."""

    mode: ExecutionModeV1
    semantic_success: bool
    replay_exact: bool
    blast_radius: int
    work: float
    wall_ms: float

    def __post_init__(self) -> None:
        if self.mode is ExecutionModeV1.DEFER:
            raise ValueError("DEFER has no all-arm execution outcome")
        if self.blast_radius < 0 or self.work < 0 or self.wall_ms < 0:
            raise ValueError("outcome costs must be non-negative")


@dataclass(frozen=True)
class ExecutionRouterDecisionV1:
    mode: ExecutionModeV1
    source: str
    model_forwards: int

    def __post_init__(self) -> None:
        if self.source not in {"forced", "scored", "deterministic", "defer"}:
            raise ValueError("unsupported router decision source")
        if self.model_forwards < 0:
            raise ValueError("model_forwards must be non-negative")
        if self.mode is ExecutionModeV1.DEFER and self.source != "defer":
            raise ValueError("DEFER must have defer source")


def choose_execution_mode(
    router_input: ExecutionRouterInputV1,
    *,
    scores: Mapping[ExecutionModeV1, float] | None = None,
    deterministic_order: Sequence[ExecutionModeV1] = (
        ExecutionModeV1.CONTROL,
        ExecutionModeV1.PRIMITIVE_OPERATOR,
        ExecutionModeV1.BULK_OPERATOR,
        ExecutionModeV1.TRANSACTION,
        ExecutionModeV1.GENERATE_SUBTREE,
        ExecutionModeV1.FULL_REGENERATE,
    ),
) -> ExecutionRouterDecisionV1:
    """Choose only a live compiler mode, otherwise explicit DEFER."""
    if router_input.forced_mode is not None:
        return ExecutionRouterDecisionV1(router_input.forced_mode, "forced", 0)
    if scores:
        permitted = {
            mode: score
            for mode, score in scores.items()
            if mode in router_input.available_modes and mode is not ExecutionModeV1.DEFER
        }
        if permitted:
            mode = max(permitted, key=lambda candidate: (permitted[candidate], candidate.value))
            return ExecutionRouterDecisionV1(mode, "scored", 1)
    for mode in deterministic_order:
        if mode in router_input.available_modes:
            return ExecutionRouterDecisionV1(mode, "deterministic", 0)
    return ExecutionRouterDecisionV1(ExecutionModeV1.DEFER, "defer", 0)


def oracle_execution_mode(outcomes: Sequence[ExecutionRouterOutcomeV1]) -> ExecutionModeV1:
    """Offline-only lexicographic label; callers must keep it out of input."""
    if not outcomes:
        return ExecutionModeV1.DEFER
    return min(
        outcomes,
        key=lambda row: (
            not row.semantic_success,
            not row.replay_exact,
            row.blast_radius,
            row.work,
            row.wall_ms,
            row.mode.value,
        ),
    ).mode
