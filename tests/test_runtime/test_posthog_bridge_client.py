"""Fail-soft PostHog bridge: queue/flush, no-op path, drop counters."""

from __future__ import annotations

import logging
import time
from typing import Any

import pytest

import slm_training.runtime.telemetry.posthog_bridge as bridge_module
from slm_training.runtime.telemetry.posthog_bridge import (
    PostHogBridge,
    capture_ai_generation,
    capture_ai_trace,
    default_distinct_id,
)


class FakeTransport:
    """Records batch payloads in place of urllib; no network I/O."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, payload: dict[str, Any]) -> None:
        if self.fail:
            raise OSError("offline")
        self.calls.append((url, payload))


@pytest.fixture(autouse=True)
def _isolated_bridge(monkeypatch: pytest.MonkeyPatch):
    """Keep every test offline and reset the module singleton + notice flag."""
    monkeypatch.delenv("POSTHOG_PROJECT_API_KEY", raising=False)
    monkeypatch.delenv("POSTHOG_HOST", raising=False)
    monkeypatch.delenv("SLM_RUN_ID", raising=False)
    monkeypatch.setattr(bridge_module, "_bridge", None)
    monkeypatch.setattr(bridge_module, "_noop_notice_logged", False)
    yield
    bridge_module.shutdown_bridge(timeout=0.5)


def _make_bridge(transport: FakeTransport, **kwargs: Any) -> PostHogBridge:
    defaults: dict[str, Any] = {
        "distinct_id": "run-test",
        "batch_size": 100,
        "flush_seconds": 60.0,  # keep the daemon thread idle; tests flush manually
        "transport": transport,
    }
    defaults.update(kwargs)
    return PostHogBridge("phc_test", "https://ph.example.test/", **defaults)


def test_batch_endpoint_payload_and_flush() -> None:
    transport = FakeTransport()
    bridge = _make_bridge(transport)
    try:
        assert bridge.capture("$ai_trace", {"$ai_trace_id": "abc", "$ai_span_name": "train"})
        assert bridge.capture("$ai_generation", {"$ai_trace_id": "abc"})
        bridge.flush()
    finally:
        bridge.close(timeout=0.5)

    assert len(transport.calls) == 1
    url, payload = transport.calls[0]
    assert url == "https://ph.example.test/batch/"
    assert payload["api_key"] == "phc_test"
    assert [row["event"] for row in payload["batch"]] == ["$ai_trace", "$ai_generation"]
    first = payload["batch"][0]
    assert first["distinct_id"] == "run-test"
    assert first["properties"]["distinct_id"] == "run-test"
    assert first["properties"]["$ai_span_name"] == "train"
    assert first["timestamp"].endswith("+00:00")
    assert bridge.stats.enqueued == 2
    assert bridge.stats.sent == 2
    assert bridge.stats.dropped == 0


def test_background_thread_flushes_on_batch_size() -> None:
    transport = FakeTransport()
    bridge = _make_bridge(transport, batch_size=2)
    try:
        bridge.capture("$ai_trace", {"$ai_trace_id": "a"})
        bridge.capture("$ai_trace", {"$ai_trace_id": "b"})
        deadline = time.monotonic() + 2.0
        while not transport.calls and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        bridge.close(timeout=0.5)
    assert transport.calls, "size-triggered flush did not run"
    assert len(transport.calls[0][1]["batch"]) == 2


def test_full_queue_drops_with_counter() -> None:
    transport = FakeTransport()
    bridge = _make_bridge(transport, max_queue=2)
    try:
        assert bridge.capture("$ai_trace", {"n": 1}) is True
        assert bridge.capture("$ai_trace", {"n": 2}) is True
        assert bridge.capture("$ai_trace", {"n": 3}) is False
        assert bridge.stats.dropped == 1
        bridge.flush()
        assert bridge.stats.sent == 2
    finally:
        bridge.close(timeout=0.5)


class _BlockingTransport:
    """Each send blocks for ``delay`` seconds — simulates a slow HTTP call."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.calls = 0

    def __call__(self, url: str, payload: dict[str, Any]) -> None:
        self.calls += 1
        time.sleep(self.delay)


def test_close_respects_one_absolute_deadline_with_full_queue() -> None:
    """A full queue behind a slow transport must not delay close() past its
    own timeout — the join and the final drain share one deadline, not
    (join timeout) + (unbounded drain), each send blocking further."""
    transport = _BlockingTransport(delay=0.2)
    bridge = _make_bridge(
        transport, max_queue=50, batch_size=1, flush_seconds=0.01
    )
    for i in range(20):
        bridge.capture("$ai_trace", {"n": i})

    started = time.monotonic()
    bridge.close(timeout=0.5)
    elapsed = time.monotonic() - started

    # Generous slack over the deadline for scheduling jitter and the one
    # in-flight send that can't be interrupted mid-call, but this must stay
    # far below "drain all 20 batches at 0.2s each" (~4s unbounded).
    assert elapsed < 1.5
    # close() itself is bounded (asserted above); the abandoned background
    # worker thread may still be mid-send for one batch when close()
    # returns (a blocking transport call can't be interrupted from another
    # thread) — give it a short grace window to finish before the final count.
    for _ in range(30):
        if bridge.stats.sent + bridge.stats.dropped >= 20:
            break
        time.sleep(0.1)
    assert bridge.stats.sent + bridge.stats.dropped == 20


def test_transport_failures_never_raise_and_count_drops() -> None:
    transport = FakeTransport(fail=True)
    bridge = _make_bridge(transport)
    try:
        bridge.capture("$ai_trace", {"n": 1})
        bridge.flush()  # must not raise
        assert bridge.stats.dropped == 1
        assert bridge.stats.last_error == "OSError"
    finally:
        bridge.close(timeout=0.5)


def test_noop_without_api_key_logs_single_notice(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=bridge_module.__name__):
        assert capture_ai_generation("abc", "slm-frontier", "local") is False
        assert capture_ai_trace("abc", "train") is False
    notices = [rec for rec in caplog.records if "PostHog mirror disabled" in rec.message]
    assert len(notices) == 1
    assert bridge_module.get_bridge() is None


def test_capture_ai_generation_maps_properties(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = FakeTransport()
    monkeypatch.setenv("POSTHOG_PROJECT_API_KEY", "phc_test")
    monkeypatch.setattr(bridge_module, "_bridge", _make_bridge(transport))

    ok = capture_ai_generation(
        "trace-1",
        "slm-frontier-v7",
        "local",
        input_tokens=128,
        output_tokens=64,
        latency_ms=1500.0,
        cost_usd=0.0,
        error="OSError: boom",
        properties={"slm.operation": "eval"},
    )
    assert ok is True
    bridge_module.get_bridge().flush()

    props = transport.calls[0][1]["batch"][0]["properties"]
    assert props["$ai_trace_id"] == "trace-1"
    assert props["$ai_model"] == "slm-frontier-v7"
    assert props["$ai_provider"] == "local"
    assert props["$ai_input_tokens"] == 128
    assert props["$ai_output_tokens"] == 64
    assert props["$ai_latency"] == pytest.approx(1.5)  # seconds, not ms
    assert props["$ai_is_error"] is True
    assert props["$ai_error"] == "OSError: boom"
    assert props["$ai_total_cost_usd"] == 0.0
    assert props["slm.operation"] == "eval"


def test_capture_never_raises_even_when_bridge_lookup_breaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> None:
        raise RuntimeError("broken bridge")

    monkeypatch.setattr(bridge_module, "get_bridge", _boom)
    assert capture_ai_generation("abc", "m", "p") is False
    assert capture_ai_trace("abc", "name") is False


def test_distinct_id_prefers_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLM_RUN_ID", "run_2026_08_09")
    assert default_distinct_id() == "run_2026_08_09"
    monkeypatch.delenv("SLM_RUN_ID")
    assert default_distinct_id()  # hostname fallback is non-empty


def test_shutdown_bridge_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = FakeTransport()
    monkeypatch.setenv("POSTHOG_PROJECT_API_KEY", "phc_test")
    monkeypatch.setattr(bridge_module, "_bridge", _make_bridge(transport))
    bridge_module.get_bridge().capture("$ai_trace", {"n": 1})
    bridge_module.shutdown_bridge(timeout=0.5)
    bridge_module.shutdown_bridge(timeout=0.5)
    assert transport.calls, "close() must drain the queue"
