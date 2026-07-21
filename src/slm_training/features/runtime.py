"""OpenFeature runtime wiring for the control-plane server."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Literal

from openfeature import api
from openfeature.evaluation_context import EvaluationContext
from openfeature.provider.in_memory_provider import InMemoryFlag, InMemoryProvider

from slm_training.features.defaults import PRODUCT_FLAG_DEFAULTS
from slm_training.features.keys import PRODUCT_FLAG_KEYS

ProviderKind = Literal["in_memory", "posthog"]


def _merged_defaults(overrides: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(PRODUCT_FLAG_DEFAULTS)
    if overrides:
        merged.update(overrides)
    return merged


def _in_memory_flags(values: dict[str, Any]) -> dict[str, InMemoryFlag[Any]]:
    flags: dict[str, InMemoryFlag[Any]] = {}
    for key, value in values.items():
        flags[key] = InMemoryFlag(default_variant="default", variants={"default": value})
    return flags


def _parse_override_env() -> dict[str, Any]:
    raw = os.getenv("SLM_FEATURE_OVERRIDES", "").strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("SLM_FEATURE_OVERRIDES must be a JSON object")
    return parsed


def _posthog_api_key() -> str | None:
    for name in ("POSTHOG_API_KEY", "POSTHOG_PROJECT_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def _resolve_provider_kind(requested: str) -> ProviderKind:
    normalized = requested.strip().lower() or "auto"
    if normalized == "in_memory":
        return "in_memory"
    if normalized == "posthog":
        return "posthog"
    if normalized == "auto":
        return "posthog" if _posthog_api_key() else "in_memory"
    raise ValueError(
        f"unknown SLM_OPENFEATURE_PROVIDER={requested!r} "
        "(expected auto, in_memory, or posthog)"
    )


def _build_posthog_provider() -> tuple[Any, Callable[[], None] | None]:
    api_key = _posthog_api_key()
    if not api_key:
        raise RuntimeError("POSTHOG_API_KEY is required for posthog provider")
    try:
        import posthog
        from openfeature.contrib.provider.posthog import PostHogProvider
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "install openfeature-provider-posthog for PostHog provider "
            "(pip install -e '.[features-posthog]')"
        ) from exc

    host = os.getenv("POSTHOG_HOST", "https://us.i.posthog.com").strip()
    client = posthog.Posthog(api_key, host=host)
    provider = PostHogProvider(client, default_distinct_id="anonymous")
    return provider, client.shutdown


@dataclass
class FeatureRuntime:
    """Configured OpenFeature provider for product experiments."""

    provider: ProviderKind
    _shutdown: Callable[[], None] | None = None

    @classmethod
    def configure(cls, *, overrides: dict[str, Any] | None = None) -> FeatureRuntime:
        kind = _resolve_provider_kind(os.getenv("SLM_OPENFEATURE_PROVIDER", "auto"))
        env_overrides = _parse_override_env()
        merged_overrides = {**env_overrides, **(overrides or {})}

        shutdown: Callable[[], None] | None = None
        if kind == "posthog":
            provider, shutdown = _build_posthog_provider()
        else:
            provider = InMemoryProvider(_in_memory_flags(_merged_defaults(merged_overrides)))

        api.set_provider_and_wait(provider)
        return cls(provider=kind, _shutdown=shutdown)

    def evaluate_all(
        self,
        *,
        targeting_key: str = "anonymous",
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ctx = EvaluationContext(targeting_key=targeting_key, attributes=attributes or {})
        client = api.get_client()
        evaluated: dict[str, Any] = {}
        for key in PRODUCT_FLAG_KEYS:
            default = PRODUCT_FLAG_DEFAULTS[key]
            if isinstance(default, bool):
                evaluated[key] = client.get_boolean_value(key, default, ctx)
            elif isinstance(default, str):
                evaluated[key] = client.get_string_value(key, default, ctx)
            elif isinstance(default, int):
                evaluated[key] = client.get_integer_value(key, default, ctx)
            elif isinstance(default, float):
                evaluated[key] = client.get_float_value(key, default, ctx)
            elif isinstance(default, dict):
                evaluated[key] = client.get_object_value(key, default, ctx)
            else:  # pragma: no cover - defensive
                evaluated[key] = default
        return evaluated

    def bootstrap_payload(
        self,
        *,
        targeting_key: str = "anonymous",
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        posthog_key = _posthog_api_key()
        posthog_host = os.getenv("POSTHOG_HOST", "https://us.i.posthog.com").strip()
        return {
            "provider": self.provider,
            "posthog": (
                {"project_api_key": posthog_key, "host": posthog_host}
                if self.provider == "posthog" and posthog_key
                else None
            ),
            "defaults": dict(PRODUCT_FLAG_DEFAULTS),
            "evaluated": self.evaluate_all(
                targeting_key=targeting_key, attributes=attributes
            ),
            "targeting_key": targeting_key,
        }

    def shutdown(self) -> None:
        if self._shutdown is not None:
            self._shutdown()
        api.shutdown()
