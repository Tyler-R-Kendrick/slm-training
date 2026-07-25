"""Tests for OpenFeature product experiment runtime."""

from __future__ import annotations

import json

import pytest
from openfeature import api

from slm_training.features.defaults import PRODUCT_FLAG_DEFAULTS
from slm_training.features.keys import DASHBOARD_DEFAULT_RENDERER
from slm_training.features.levers import feature_flag_by_key, feature_flag_registry_payload
from slm_training.features.runtime import FeatureRuntime, _auto_provider_kind, _resolve_provider_kind


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
        assert payload["launchdarkly"] is False
        assert payload["defaults"] == PRODUCT_FLAG_DEFAULTS
        assert payload["evaluated"][DASHBOARD_DEFAULT_RENDERER] == "interpreted"
        assert payload["targeting_key"] == "anon-1"
        assert any(
            row["key"] == DASHBOARD_DEFAULT_RENDERER for row in payload["flags"]
        )
    finally:
        runtime.shutdown()


def test_posthog_provider_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_OPENFEATURE_PROVIDER", "posthog")
    monkeypatch.delenv("POSTHOG_API_KEY", raising=False)
    monkeypatch.delenv("POSTHOG_PROJECT_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="POSTHOG_API_KEY"):
        FeatureRuntime.configure()


def test_launchdarkly_provider_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_OPENFEATURE_PROVIDER", "launchdarkly")
    monkeypatch.delenv("LAUNCHDARKLY_SDK_KEY", raising=False)
    monkeypatch.delenv("LD_SDK_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LAUNCHDARKLY_SDK_KEY"):
        FeatureRuntime.configure()


def test_auto_prefers_launchdarkly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAUNCHDARKLY_SDK_KEY", "sdk-test")
    monkeypatch.setenv("POSTHOG_API_KEY", "phc_test")
    assert _auto_provider_kind() == "launchdarkly"
    assert _resolve_provider_kind("auto") == "launchdarkly"


def test_feature_flag_registry_lookup() -> None:
    payload = feature_flag_registry_payload()
    assert len(payload["flags"]) >= 3
    flag = feature_flag_by_key(DASHBOARD_DEFAULT_RENDERER)
    assert flag is not None
    assert flag.key == DASHBOARD_DEFAULT_RENDERER
    assert all({"lever_id", "flag_key"}.isdisjoint(row) for row in payload["flags"])
