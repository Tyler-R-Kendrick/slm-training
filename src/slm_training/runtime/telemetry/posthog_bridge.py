"""Fail-soft PostHog mirror for LLM analytics events.

The self-hosted OpenTelemetry stack (:mod:`slm_training.web.otel_hub` plus the
local OTLP JSONL bundles written by :mod:`slm_training.runtime.telemetry.trace`)
remains the source of truth for all telemetry. This module mirrors a small set
of LLM analytics events to PostHog so runs show up in its free-tier LLM
analytics and error-tracking views.

Design decisions (documented per WP-6):

* **Endpoint** — events are POSTed to ``{POSTHOG_HOST}/batch/`` as JSON
  ``{"api_key": ..., "batch": [...]}``. ``/batch/`` is PostHog's stable,
  documented public batch-capture endpoint (the singular ``/capture/`` accepts
  one event per request; ``/i/v0/e/`` is an SDK-internal alias). Batch items
  carry ``event``, ``distinct_id``, ``properties`` and an ISO-8601
  ``timestamp``.
* **Transport** — stdlib ``urllib.request`` only, matching
  ``trace.py::_mirror``; ``httpx`` is not a core dependency of this project
  (it appears only in optional extras), so the bridge must not import it.
* **``$ai_latency``** — PostHog's LLM analytics convention expresses latency
  in **seconds** (float); callers pass milliseconds and the bridge converts.
* **``distinct_id``** — a stable machine/run identifier: ``SLM_RUN_ID`` when
  set, else the hostname, else ``"slm-training"``. Runs are infrastructure,
  not people, so no user identity is ever attached.
* **Fail-soft** — when ``POSTHOG_PROJECT_API_KEY`` is unset the module no-ops
  after logging a single notice. Events are queued on a bounded in-memory
  queue drained by one daemon thread; when the queue is full events are
  dropped and counted. Nothing in this module ever raises into callers.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import socket
import threading
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "https://us.i.posthog.com"
_DEFAULT_MAX_QUEUE = 2048
_DEFAULT_BATCH_SIZE = 50
_DEFAULT_FLUSH_SECONDS = 2.0
_HTTP_TIMEOUT_SECONDS = 3.0
_ATEXIT_FLUSH_SECONDS = 2.0

# A transport posts one already-encoded batch payload to the endpoint URL.
Transport = Callable[[str, dict[str, Any]], None]


def _urllib_transport(url: str, payload: dict[str, Any]) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS):  # noqa: S310
        pass


def default_distinct_id() -> str:
    """Stable machine/run identifier: SLM_RUN_ID, else hostname, else literal."""
    run_id = os.getenv("SLM_RUN_ID", "").strip()
    if run_id:
        return run_id
    try:
        return socket.gethostname() or "slm-training"
    except OSError:  # pragma: no cover - hostname lookup is best-effort
        return "slm-training"


@dataclass
class BridgeStats:
    enqueued: int = 0
    sent: int = 0
    dropped: int = 0
    last_error: str | None = None


class PostHogBridge:
    """Bounded queue + single daemon flusher posting to PostHog ``/batch/``."""

    def __init__(
        self,
        api_key: str,
        host: str = _DEFAULT_HOST,
        *,
        distinct_id: str | None = None,
        max_queue: int = _DEFAULT_MAX_QUEUE,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        flush_seconds: float = _DEFAULT_FLUSH_SECONDS,
        transport: Transport | None = None,
    ) -> None:
        self.api_key = api_key
        self.endpoint = (host.strip().rstrip("/") or _DEFAULT_HOST) + "/batch/"
        self.distinct_id = distinct_id or default_distinct_id()
        self.batch_size = max(1, int(batch_size))
        self.flush_seconds = max(0.05, float(flush_seconds))
        self.stats = BridgeStats()
        self._transport = transport or _urllib_transport
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max(1, int(max_queue)))
        self._flush_lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._worker, name="posthog-bridge-flush", daemon=True
        )
        self._thread.start()

    def capture(self, event: str, properties: dict[str, Any]) -> bool:
        """Queue one event (fire-and-forget). Returns False when dropped."""
        row = {
            "event": event,
            "distinct_id": self.distinct_id,
            "properties": {"distinct_id": self.distinct_id, **properties},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self.stats.dropped += 1
            return False
        self.stats.enqueued += 1
        if self._queue.qsize() >= self.batch_size:
            self._wake.set()
        return True

    def flush(self) -> None:
        """Best-effort synchronous drain; bounded by the HTTP timeout, never raises."""
        self._flush_once()

    def close(self, timeout: float = _ATEXIT_FLUSH_SECONDS) -> None:
        """Stop the worker and drain what remains, bounded by ``timeout``."""
        self._closed.set()
        self._wake.set()
        self._thread.join(timeout=timeout)
        self._flush_once()

    # -- internals ---------------------------------------------------------

    def _worker(self) -> None:
        while not self._closed.is_set():
            self._wake.wait(timeout=self.flush_seconds)
            self._wake.clear()
            self._flush_once()

    def _flush_once(self) -> None:
        with self._flush_lock:
            while True:
                rows: list[dict[str, Any]] = []
                while len(rows) < self.batch_size:
                    try:
                        rows.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
                if not rows:
                    return
                try:
                    self._transport(self.endpoint, {"api_key": self.api_key, "batch": rows})
                    self.stats.sent += len(rows)
                except Exception as exc:  # noqa: BLE001 - mirror must never raise
                    self.stats.dropped += len(rows)
                    self.stats.last_error = type(exc).__name__


_bridge: PostHogBridge | None = None
_bridge_lock = threading.Lock()
_noop_notice_logged = False


def _api_key() -> str | None:
    value = os.getenv("POSTHOG_PROJECT_API_KEY", "").strip()
    return value or None


def _host() -> str:
    return os.getenv("POSTHOG_HOST", _DEFAULT_HOST).strip() or _DEFAULT_HOST


def get_bridge() -> PostHogBridge | None:
    """Return the process-wide bridge, or None (with one notice) when disabled."""
    global _bridge, _noop_notice_logged
    key = _api_key()
    if not key:
        if not _noop_notice_logged:
            _noop_notice_logged = True
            logger.info(
                "PostHog mirror disabled: POSTHOG_PROJECT_API_KEY is not set; "
                "LLM telemetry stays local/OTel only."
            )
        return None
    with _bridge_lock:
        if _bridge is None:
            _bridge = PostHogBridge(key, _host())
            atexit.register(shutdown_bridge)
        return _bridge


def shutdown_bridge(timeout: float = _ATEXIT_FLUSH_SECONDS) -> None:
    """Best-effort final flush (registered via atexit); safe to call twice."""
    global _bridge
    with _bridge_lock:
        bridge, _bridge = _bridge, None
    if bridge is not None:
        try:
            bridge.close(timeout=timeout)
        except Exception:  # noqa: BLE001 - shutdown must never raise
            pass


def capture_ai_generation(
    trace_id: str,
    model: str,
    provider: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    latency_ms: float | None = None,
    cost_usd: float | None = None,
    error: str | None = None,
    properties: dict[str, Any] | None = None,
) -> bool:
    """Mirror one LLM generation as a PostHog ``$ai_generation`` event.

    ``latency_ms`` is converted to seconds for ``$ai_latency`` (PostHog's LLM
    analytics convention). Returns True when queued; never raises.
    """
    try:
        bridge = get_bridge()
        if bridge is None:
            return False
        props: dict[str, Any] = {
            "$ai_trace_id": trace_id,
            "$ai_model": model,
            "$ai_provider": provider,
            "$ai_is_error": bool(error),
        }
        if input_tokens is not None:
            props["$ai_input_tokens"] = int(input_tokens)
        if output_tokens is not None:
            props["$ai_output_tokens"] = int(output_tokens)
        if latency_ms is not None:
            props["$ai_latency"] = round(float(latency_ms) / 1000.0, 6)
        if cost_usd is not None:
            props["$ai_total_cost_usd"] = float(cost_usd)
        if error:
            props["$ai_error"] = str(error)
        if properties:
            props.update(properties)
        return bridge.capture("$ai_generation", props)
    except Exception:  # noqa: BLE001 - mirror must never raise into callers
        return False


def capture_ai_trace(
    trace_id: str,
    name: str,
    properties: dict[str, Any] | None = None,
) -> bool:
    """Mirror one trace/root span as a PostHog ``$ai_trace`` event. Never raises."""
    try:
        bridge = get_bridge()
        if bridge is None:
            return False
        props: dict[str, Any] = {"$ai_trace_id": trace_id, "$ai_span_name": name}
        if properties:
            props.update(properties)
        return bridge.capture("$ai_trace", props)
    except Exception:  # noqa: BLE001 - mirror must never raise into callers
        return False


__all__ = [
    "BridgeStats",
    "PostHogBridge",
    "capture_ai_generation",
    "capture_ai_trace",
    "default_distinct_id",
    "get_bridge",
    "shutdown_bridge",
]
