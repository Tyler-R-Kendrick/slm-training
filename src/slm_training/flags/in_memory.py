"""In-process OpenFeature provider backed by an explicit ruleset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from slm_training.flags.api import (
    ErrorCode,
    EvaluationContext,
    EvaluationDetails,
    FlagValueType,
    Reason,
)
from slm_training.flags.levers import LEVER_BY_KEY, LeverSpec, coerce_lever_value


@dataclass(frozen=True)
class FlagDefinition:
    key: str
    value_type: FlagValueType
    default: Any
    enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_lever(cls, spec: LeverSpec, value: Any | None = None) -> FlagDefinition:
        return cls(
            key=spec.key,
            value_type=spec.value_type,
            default=spec.default if value is None else coerce_lever_value(spec, value),
            metadata=dict(spec.metadata or {}),
        )


def ruleset_from_mapping(values: Mapping[str, Any]) -> dict[str, FlagDefinition]:
    """Build a ruleset from ``{flag_key: value}`` using the lever registry.

    Unknown keys are rejected — experiment levers must be registered.
    """
    out: dict[str, FlagDefinition] = {}
    for key, value in values.items():
        spec = LEVER_BY_KEY.get(key)
        if spec is None:
            raise KeyError(f"unknown experiment lever flag: {key!r}")
        out[key] = FlagDefinition.from_lever(spec, value)
    return out


def ruleset_from_defaults() -> dict[str, FlagDefinition]:
    return {key: FlagDefinition.from_lever(spec) for key, spec in LEVER_BY_KEY.items()}


class InMemoryProvider:
    """Deterministic in-process provider (flagd-style local ruleset)."""

    name = "InMemoryProvider"

    def __init__(self, flags: Mapping[str, FlagDefinition] | None = None) -> None:
        self._flags = dict(flags or {})

    def replace_flags(self, flags: Mapping[str, FlagDefinition]) -> None:
        self._flags = dict(flags)

    def update_flags(self, flags: Mapping[str, FlagDefinition]) -> None:
        self._flags.update(flags)

    def listed_keys(self) -> list[str]:
        return sorted(self._flags)

    def _resolve(
        self,
        flag_key: str,
        default: Any,
        expected: FlagValueType,
        context: EvaluationContext,
    ) -> EvaluationDetails:
        definition = self._flags.get(flag_key)
        metadata = {
            "provider": self.name,
        }
        if context.targeting_key:
            metadata["targeting_key"] = context.targeting_key
        for attr_key in ("experiment_id", "matrix", "model_name", "context_backend"):
            if attr_key in context.attributes:
                metadata[attr_key] = context.attributes[attr_key]

        if definition is None:
            return EvaluationDetails(
                flag_key=flag_key,
                value=default,
                variant=_variant(default),
                reason=Reason.DEFAULT,
                flag_metadata=metadata,
                error_code=ErrorCode.FLAG_NOT_FOUND,
                error_message=f"flag not in ruleset: {flag_key}",
            )
        if not definition.enabled:
            return EvaluationDetails(
                flag_key=flag_key,
                value=default,
                variant=_variant(default),
                reason=Reason.DISABLED,
                flag_metadata={**metadata, **dict(definition.metadata)},
            )
        if definition.value_type is not expected:
            return EvaluationDetails(
                flag_key=flag_key,
                value=default,
                variant=_variant(default),
                reason=Reason.ERROR,
                flag_metadata={**metadata, **dict(definition.metadata)},
                error_code=ErrorCode.TYPE_MISMATCH,
                error_message=(
                    f"{flag_key}: expected {expected.value}, "
                    f"ruleset has {definition.value_type.value}"
                ),
            )
        value = definition.default
        return EvaluationDetails(
            flag_key=flag_key,
            value=value,
            variant=_variant(value),
            reason=Reason.STATIC,
            flag_metadata={**metadata, **dict(definition.metadata)},
        )

    def resolve_boolean(
        self, flag_key: str, default: bool, context: EvaluationContext
    ) -> EvaluationDetails:
        return self._resolve(flag_key, default, FlagValueType.BOOLEAN, context)

    def resolve_string(
        self, flag_key: str, default: str, context: EvaluationContext
    ) -> EvaluationDetails:
        return self._resolve(flag_key, default, FlagValueType.STRING, context)

    def resolve_number(
        self, flag_key: str, default: float, context: EvaluationContext
    ) -> EvaluationDetails:
        return self._resolve(flag_key, default, FlagValueType.NUMBER, context)

    def resolve_object(
        self, flag_key: str, default: Mapping[str, Any], context: EvaluationContext
    ) -> EvaluationDetails:
        return self._resolve(flag_key, dict(default), FlagValueType.OBJECT, context)


def _variant(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)
