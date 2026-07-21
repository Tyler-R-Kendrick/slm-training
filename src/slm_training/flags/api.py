"""OpenFeature-shaped evaluation types and client (zero third-party deps).

Mirrors the CNCF OpenFeature evaluation surface closely enough that an
``openfeature-sdk`` adapter can wrap the same ``FlagProvider`` Protocol later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol


class FlagValueType(str, Enum):
    BOOLEAN = "boolean"
    STRING = "string"
    NUMBER = "number"
    OBJECT = "object"


class Reason(str, Enum):
    STATIC = "STATIC"
    DEFAULT = "DEFAULT"
    TARGETING_MATCH = "TARGETING_MATCH"
    SPLIT = "SPLIT"
    DISABLED = "DISABLED"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"


class ErrorCode(str, Enum):
    PROVIDER_NOT_READY = "PROVIDER_NOT_READY"
    FLAG_NOT_FOUND = "FLAG_NOT_FOUND"
    PARSE_ERROR = "PARSE_ERROR"
    TYPE_MISMATCH = "TYPE_MISMATCH"
    GENERAL = "GENERAL"


@dataclass(frozen=True)
class EvaluationContext:
    """OpenFeature evaluation context.

    ``targeting_key`` is the subject id (here: ``run_id``). Extra attributes
    carry experiment metadata (``experiment_id``, ``matrix``, …).
    """

    targeting_key: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def merged(self, other: EvaluationContext | None) -> EvaluationContext:
        if other is None:
            return self
        attrs = {**dict(self.attributes), **dict(other.attributes)}
        return EvaluationContext(
            targeting_key=other.targeting_key or self.targeting_key,
            attributes=attrs,
        )


@dataclass(frozen=True)
class EvaluationDetails:
    flag_key: str
    value: Any
    variant: str | None = None
    reason: Reason = Reason.UNKNOWN
    flag_metadata: Mapping[str, Any] = field(default_factory=dict)
    error_code: ErrorCode | None = None
    error_message: str | None = None

    def to_ofrep(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "value": self.value,
            "reason": self.reason.value,
        }
        if self.variant is not None:
            payload["variant"] = self.variant
        if self.flag_metadata:
            payload["metadata"] = dict(self.flag_metadata)
        if self.error_code is not None:
            payload["errorCode"] = self.error_code.value
        if self.error_message is not None:
            payload["errorDetails"] = self.error_message
        return payload


class FlagProvider(Protocol):
    """Minimal OpenFeature provider seam."""

    name: str

    def resolve_boolean(
        self, flag_key: str, default: bool, context: EvaluationContext
    ) -> EvaluationDetails: ...

    def resolve_string(
        self, flag_key: str, default: str, context: EvaluationContext
    ) -> EvaluationDetails: ...

    def resolve_number(
        self, flag_key: str, default: float, context: EvaluationContext
    ) -> EvaluationDetails: ...

    def resolve_object(
        self, flag_key: str, default: Mapping[str, Any], context: EvaluationContext
    ) -> EvaluationDetails: ...


class _NoopProvider:
    name = "No-op Provider"

    def resolve_boolean(
        self, flag_key: str, default: bool, context: EvaluationContext
    ) -> EvaluationDetails:
        return EvaluationDetails(
            flag_key=flag_key,
            value=default,
            variant=_variant(default),
            reason=Reason.DEFAULT,
        )

    def resolve_string(
        self, flag_key: str, default: str, context: EvaluationContext
    ) -> EvaluationDetails:
        return EvaluationDetails(
            flag_key=flag_key,
            value=default,
            variant=str(default),
            reason=Reason.DEFAULT,
        )

    def resolve_number(
        self, flag_key: str, default: float, context: EvaluationContext
    ) -> EvaluationDetails:
        return EvaluationDetails(
            flag_key=flag_key,
            value=default,
            variant=str(default),
            reason=Reason.DEFAULT,
        )

    def resolve_object(
        self, flag_key: str, default: Mapping[str, Any], context: EvaluationContext
    ) -> EvaluationDetails:
        return EvaluationDetails(
            flag_key=flag_key,
            value=dict(default),
            variant="default",
            reason=Reason.DEFAULT,
        )


def _variant(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)


class FlagClient:
    """OpenFeature-shaped client bound to one provider."""

    def __init__(
        self,
        provider: FlagProvider | None = None,
        *,
        domain: str = "openui",
        context: EvaluationContext | None = None,
    ) -> None:
        self._provider: FlagProvider = provider or _NoopProvider()
        self.domain = domain
        self._context = context or EvaluationContext()

    @property
    def provider_name(self) -> str:
        return getattr(self._provider, "name", "unknown")

    def set_context(self, context: EvaluationContext) -> None:
        self._context = context

    def get_boolean_value(
        self,
        flag_key: str,
        default: bool,
        context: EvaluationContext | None = None,
    ) -> bool:
        return bool(self.get_boolean_details(flag_key, default, context).value)

    def get_boolean_details(
        self,
        flag_key: str,
        default: bool,
        context: EvaluationContext | None = None,
    ) -> EvaluationDetails:
        return self._provider.resolve_boolean(
            flag_key, default, self._context.merged(context)
        )

    def get_string_value(
        self,
        flag_key: str,
        default: str,
        context: EvaluationContext | None = None,
    ) -> str:
        return str(self.get_string_details(flag_key, default, context).value)

    def get_string_details(
        self,
        flag_key: str,
        default: str,
        context: EvaluationContext | None = None,
    ) -> EvaluationDetails:
        return self._provider.resolve_string(
            flag_key, default, self._context.merged(context)
        )

    def get_number_value(
        self,
        flag_key: str,
        default: float,
        context: EvaluationContext | None = None,
    ) -> float:
        return float(self.get_number_details(flag_key, default, context).value)

    def get_number_details(
        self,
        flag_key: str,
        default: float,
        context: EvaluationContext | None = None,
    ) -> EvaluationDetails:
        return self._provider.resolve_number(
            flag_key, default, self._context.merged(context)
        )

    def get_object_value(
        self,
        flag_key: str,
        default: Mapping[str, Any],
        context: EvaluationContext | None = None,
    ) -> dict[str, Any]:
        value = self.get_object_details(flag_key, default, context).value
        return dict(value) if isinstance(value, Mapping) else dict(default)

    def get_object_details(
        self,
        flag_key: str,
        default: Mapping[str, Any],
        context: EvaluationContext | None = None,
    ) -> EvaluationDetails:
        return self._provider.resolve_object(
            flag_key, default, self._context.merged(context)
        )
