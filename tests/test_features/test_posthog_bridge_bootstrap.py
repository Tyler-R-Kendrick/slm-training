"""Bootstrap payload carries the opt-in PostHog session-replay flag."""

from __future__ import annotations

import pytest
from openfeature import api

from slm_training.features.runtime import FeatureRuntime, _posthog_replay_enabled


@pytest.fixture(autouse=True)
def _reset_openfeature():
    api.shutdown()
    yield
    api.shutdown()


def test_replay_env_flag_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLM_POSTHOG_ENABLE_REPLAY", raising=False)
    assert _posthog_replay_enabled() is False
    monkeypatch.setenv("SLM_POSTHOG_ENABLE_REPLAY", "true")
    assert _posthog_replay_enabled() is True
    monkeypatch.setenv("SLM_POSTHOG_ENABLE_REPLAY", "0")
    assert _posthog_replay_enabled() is False


def test_bootstrap_posthog_block_defaults_replay_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTHOG_PROJECT_API_KEY", "phc_test")
    monkeypatch.delenv("SLM_POSTHOG_ENABLE_REPLAY", raising=False)
    # Provider kind is data on the dataclass; evaluation falls back to the
    # OpenFeature no-op provider, so no PostHog SDK/network is involved.
    runtime = FeatureRuntime(provider="posthog")
    payload = runtime.bootstrap_payload(targeting_key="anon-1")
    assert payload["posthog"] == {
        "project_api_key": "phc_test",
        "host": "https://us.i.posthog.com",
        "enable_replay": False,
    }


def test_bootstrap_posthog_block_replay_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTHOG_PROJECT_API_KEY", "phc_test")
    monkeypatch.setenv("POSTHOG_HOST", "https://eu.i.posthog.com")
    monkeypatch.setenv("SLM_POSTHOG_ENABLE_REPLAY", "yes")
    runtime = FeatureRuntime(provider="posthog")
    payload = runtime.bootstrap_payload(targeting_key="anon-1")
    assert payload["posthog"]["enable_replay"] is True
    assert payload["posthog"]["host"] == "https://eu.i.posthog.com"


def test_in_memory_bootstrap_still_has_no_posthog_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLM_OPENFEATURE_PROVIDER", "in_memory")
    monkeypatch.delenv("POSTHOG_PROJECT_API_KEY", raising=False)
    monkeypatch.delenv("POSTHOG_API_KEY", raising=False)
    runtime = FeatureRuntime.configure()
    try:
        assert runtime.bootstrap_payload()["posthog"] is None
    finally:
        runtime.shutdown()
