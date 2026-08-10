"""trace.py mirrors llm.* spans to the PostHog bridge without behavior change."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import slm_training.runtime.telemetry.posthog_bridge as bridge_module
from slm_training.runtime.telemetry import run_trace


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No LangSmith, no OTLP mirror, no PostHog network in these tests."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    for name in (
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
        "SLM_OTEL_PEERS",
        "POSTHOG_PROJECT_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_capture(**kwargs: Any) -> bool:
        calls.append(kwargs)
        return True

    monkeypatch.setattr(bridge_module, "capture_ai_generation", fake_capture)
    return calls


def test_llm_span_mirrors_one_ai_generation(
    tmp_path: Path, captured: list[dict[str, Any]]
) -> None:
    with run_trace(
        "gen-1",
        "generate",
        trace_root=tmp_path,
        attributes={
            "llm.provider": "local",
            "llm.model": "slm-frontier-v7",
            "llm.input_tokens": 128,
            "llm.output_tokens": 64,
            "llm.temperature": 0.2,
        },
    ) as trace:
        pass

    assert len(captured) == 1
    call = captured[0]
    assert call["trace_id"] == trace.trace_id
    assert call["model"] == "slm-frontier-v7"
    assert call["provider"] == "local"
    assert call["input_tokens"] == 128
    assert call["output_tokens"] == 64
    assert call["latency_ms"] >= 0
    assert call["error"] is None
    assert call["properties"]["slm.run.id"] == "gen-1"
    assert call["properties"]["slm.operation"] == "generate"
    assert call["properties"]["llm.temperature"] == 0.2
    # Core keys stay first-class arguments, not duplicated passthrough.
    assert "llm.model" not in call["properties"]


def test_content_bearing_llm_attribute_is_never_forwarded(
    tmp_path: Path, captured: list[dict[str, Any]]
) -> None:
    """Data-minimization: llm.* attributes are an allowlist, not a passthrough
    — a caller-set attribute carrying free-form content (prompt, completion,
    a raw response body) must never reach the third-party mirror just
    because it happens to start with "llm."."""
    with run_trace(
        "gen-2",
        "generate",
        trace_root=tmp_path,
        attributes={
            "llm.provider": "local",
            "llm.model": "slm-frontier-v7",
            "llm.prompt": "the user's secret prompt text",
            "llm.completion": "the model's generated completion text",
            "llm.raw_response": {"choices": ["sensitive content"]},
        },
    ):
        pass

    assert len(captured) == 1
    properties = captured[0]["properties"]
    assert "llm.prompt" not in properties
    assert "llm.completion" not in properties
    assert "llm.raw_response" not in properties


def test_non_llm_span_does_not_mirror(tmp_path: Path, captured: list[dict[str, Any]]) -> None:
    with run_trace("plain", "train", trace_root=tmp_path, attributes={"slm.step": 1}):
        pass
    assert captured == []


def test_failed_llm_span_mirrors_error(tmp_path: Path, captured: list[dict[str, Any]]) -> None:
    with pytest.raises(ValueError):
        with run_trace(
            "gen-err",
            "generate",
            trace_root=tmp_path,
            attributes={"llm.provider": "local", "llm.model": "m"},
        ):
            raise ValueError("bad decode")

    assert len(captured) == 1
    assert captured[0]["error"] == "ValueError: bad decode"


def test_mirror_failure_never_breaks_local_tracing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_capture(**kwargs: Any) -> bool:
        raise RuntimeError("bridge exploded")

    monkeypatch.setattr(bridge_module, "capture_ai_generation", broken_capture)
    with run_trace(
        "gen-broken",
        "generate",
        trace_root=tmp_path,
        attributes={"llm.provider": "local", "llm.model": "m"},
    ) as trace:
        pass

    traces = list((tmp_path / trace.trace_id / "signals" / "traces").glob("*.jsonl"))
    assert traces, "local OTLP span must still be written"
    row = json.loads(traces[0].read_text().splitlines()[0])
    span = row["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert span["status"]["code"] == 1


def test_no_key_means_no_network_and_no_behavior_change(tmp_path: Path) -> None:
    """Real bridge module, unset key: span finishes normally and bridge no-ops."""
    with run_trace(
        "gen-nokey",
        "generate",
        trace_root=tmp_path,
        attributes={"llm.provider": "local", "llm.model": "m"},
    ) as trace:
        pass
    assert (tmp_path / trace.trace_id / "manifest.json").is_file()
    assert bridge_module.get_bridge() is None
