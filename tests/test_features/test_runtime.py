"""Tests for OpenFeature product experiment runtime."""

from __future__ import annotations

import json

import pytest
from openfeature import api

from slm_training.features.defaults import PRODUCT_FLAG_DEFAULTS
from slm_training.features.keys import DASHBOARD_DEFAULT_RENDERER
from slm_training.features.runtime import FeatureRuntime


@pytest.fixture(autouse=True)
def _reset_openfeature():
    api.shutdown()
    yield
    api.shutdown()


def test_in_memory_evaluates_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_OPENFEATURE_PROVIDER", "in_memory")
    runtime = FeatureRuntime.configure()
    try:
        evaluated = runtime.evaluate_all(targeting_key="test-user")
        assert evaluated[DASHBOARD_DEFAULT_RENDERER] == "interpreted"
        assert evaluated == PRODUCT_FLAG_DEFAULTS
    finally:
        runtime.shutdown()


def test_override_env_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_OPENFEATURE_PROVIDER", "in_memory")
    monkeypatch.setenv(
        "SLM_FEATURE_OVERRIDES",
        json.dumps({DASHBOARD_DEFAULT_RENDERER: "compiled"}),
    )
    runtime = FeatureRuntime.configure()
    try:
        evaluated = runtime.evaluate_all()
        assert evaluated[DASHBOARD_DEFAULT_RENDERER] == "compiled"
    finally:
        runtime.shutdown()


def test_bootstrap_payload_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_OPENFEATURE_PROVIDER", "in_memory")
    runtime = FeatureRuntime.configure()
    try:
        payload = runtime.bootstrap_payload(targeting_key="anon-1")
        assert payload["provider"] == "in_memory"
        assert payload["posthog"] is None
        assert payload["defaults"] == PRODUCT_FLAG_DEFAULTS
        assert payload["evaluated"][DASHBOARD_DEFAULT_RENDERER] == "interpreted"
        assert payload["targeting_key"] == "anon-1"
    finally:
        runtime.shutdown()


def test_posthog_provider_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_OPENFEATURE_PROVIDER", "posthog")
    monkeypatch.delenv("POSTHOG_API_KEY", raising=False)
    monkeypatch.delenv("POSTHOG_PROJECT_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="POSTHOG_API_KEY"):
        FeatureRuntime.configure()
