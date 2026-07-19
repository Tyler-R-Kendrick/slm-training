"""In-memory OTLP hub: active-run registry, lazy SSE fan-out, peer federation.

Every app instance is a peer speaking the same protocol: standard OTLP/HTTP JSON
ingest (``POST /v1/traces|logs``, the paths ``RunTrace._mirror`` derives) plus a
tiny read API (``GET /api/otel/runs`` and ``.../events``). "The shared endpoint"
is a convention — any reachable instance listed in ``SLM_OTEL_PEERS``. Reads
federate across all peers at request time against each peer's *local-only* view,
so cyclic peer graphs are loop-safe by construction (no re-broadcast, no gossip).
State is in-memory and single-process (single-worker uvicorn); durable history
stays in each producer's local trace bundle JSONL.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import secrets
import threading
import time
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator

# Keep in sync with observability._RUN_ID_RE (dashboard links depend on it).
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_HF_WHOAMI_URL = "https://huggingface.co/api/whoami-v2"

MAX_INGEST_BYTES = 2 * 1024 * 1024
RING_BUFFER_SIZE = 512
QUEUE_SIZE = 256
MAX_SUBSCRIBERS = 32
MAX_RUNS = 256
ACTIVE_IDLE_STALE_SECONDS = 600.0
EVICT_TERMINAL_SECONDS = 1800.0
EVICT_STALE_SECONDS = 3600.0
DISK_ACTIVE_WINDOW_SECONDS = 600.0
FEDERATION_CACHE_SECONDS = 10.0

_TERMINAL = {"completed", "failed"}


def _sse(event: str, data: dict[str, Any]) -> str:
    # Same frame format as jobs._sse; duplicated because jobs.py is only
    # imported on execution-enabled deployments and the hub must stand alone.
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _flatten_attrs(rows: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if not isinstance(rows, list):
        return attrs
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = row.get("key")
        value = row.get("value")
        if not key or not isinstance(value, dict):
            continue
        if "stringValue" in value:
            attrs[key] = value["stringValue"]
        elif "boolValue" in value:
            attrs[key] = bool(value["boolValue"])
        elif "intValue" in value:
            try:
                # OTLP/JSON encodes 64-bit ints as strings.
                attrs[key] = int(value["intValue"])
            except (TypeError, ValueError):
                continue
        elif "doubleValue" in value:
            try:
                attrs[key] = float(value["doubleValue"])
            except (TypeError, ValueError):
                continue
    return attrs


def _iter_otlp_records(signal: str, payload: Any) -> Iterator[dict[str, Any]]:
    """Yield normalized events from an OTLP/JSON payload; tolerant of gaps."""
    if not isinstance(payload, dict):
        return
    if signal == "logs":
        resources, scopes_key, records_key = "resourceLogs", "scopeLogs", "logRecords"
    else:
        resources, scopes_key, records_key = "resourceSpans", "scopeSpans", "spans"
    for resource_row in payload.get(resources) or []:
        if not isinstance(resource_row, dict):
            continue
        resource_attrs = _flatten_attrs(
            (resource_row.get("resource") or {}).get("attributes")
        )
        for scope_row in resource_row.get(scopes_key) or []:
            if not isinstance(scope_row, dict):
                continue
            for record in scope_row.get(records_key) or []:
                if not isinstance(record, dict):
                    continue
                attrs = _flatten_attrs(record.get("attributes"))
                if signal == "logs":
                    body = (record.get("body") or {}).get("stringValue", "")
                    ts = record.get("timeUnixNano")
                    severity = record.get("severityText") or "INFO"
                    status_code = None
                else:
                    body = record.get("name", "")
                    ts = record.get("startTimeUnixNano")
                    severity = "INFO"
                    status = record.get("status")
                    status_code = (
                        status.get("code") if isinstance(status, dict) else None
                    )
                try:
                    ts_ns = int(ts)
                except (TypeError, ValueError):
                    ts_ns = time.time_ns()
                yield {
                    "signal": "log" if signal == "logs" else "span",
                    "ts": ts_ns,
                    "body": str(body),
                    "severity": str(severity),
                    "status_code": status_code,
                    "attrs": attrs,
                    "run_id": attrs.get("slm.run.id"),
                    "operation": attrs.get("slm.operation"),
                    "trace_id": record.get("traceId"),
                    "span_id": record.get("spanId"),
                    "instance_id": resource_attrs.get("service.instance.id"),
                }


def _http_get_json(url: str, timeout: float = 2.0) -> Any:
    from slm_training.runtime.telemetry.trace import _headers as sender_headers

    request = urllib.request.Request(url, headers=sender_headers())
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _hf_whoami(token: str) -> str | None:
    request = urllib.request.Request(
        _HF_WHOAMI_URL, headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=3.0) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    name = data.get("name") if isinstance(data, dict) else None
    return str(name) if name else None


def _tail_line(path: Path, window: int = 4096) -> str | None:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - window))
            lines = handle.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return None
    return lines[-1] if lines else None


@dataclass
class _SubState:
    queue: asyncio.Queue
    dropped: int = 0


@dataclass
class RunEntry:
    run_id: str
    operation: str = ""
    user: str | None = None
    trace_id: str | None = None
    instance_id: str | None = None
    status: str = "active"
    first_seen: float = 0.0
    last_seen: float = 0.0
    event_count: int = 0
    latest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "operation": self.operation,
            "user": self.user,
            "trace_id": self.trace_id,
            "instance_id": self.instance_id,
            "status": self.status,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "event_count": self.event_count,
            "latest": dict(self.latest),
            "source": "local",
        }


class OtelHub:
    """Active-run state ingested from OTLP, streamed lazily, merged from peers."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        outputs_dir: Path | str = Path("outputs"),
        peers: list[str] | None = None,
        auth_mode: str | None = None,
        token: str | None = None,
        now_fn: Callable[[], float] = time.time,
        fetch_json: Callable[[str], Any] = _http_get_json,
        whoami_fn: Callable[[str], str | None] = _hf_whoami,
        ping_interval: float = 15.0,
        poll_interval: float = 2.0,
        sweep_interval: float = 30.0,
    ) -> None:
        self.enabled = enabled
        self.hub_epoch = uuid.uuid4().hex
        self._outputs_dir = Path(outputs_dir)
        self._peers = [p.strip().rstrip("/") for p in (peers or []) if p.strip()]
        self._token = token
        mode = (auth_mode or "").strip().lower()
        if mode not in {"open", "token", "hf"}:
            mode = "token" if self._token else "open"
        self.auth_mode = mode
        self._now = now_fn
        self._fetch_json = fetch_json
        self._whoami_fn = whoami_fn
        self._ping_interval = ping_interval
        self._poll_interval = poll_interval
        self._sweep_interval = sweep_interval
        self._lock = threading.Lock()
        self._entries: dict[str, RunEntry] = {}
        self._events: dict[str, deque[dict[str, Any]]] = {}
        self._seq: dict[str, int] = {}
        self._subs: dict[str, dict[int, _SubState]] = {}
        self._token_cache: dict[str, tuple[str | None, float]] = {}
        self._peer_cache: list[dict[str, Any]] = []
        self._peer_cache_ts = float("-inf")
        self._peer_lock = threading.Lock()
        self._disk_cache: list[dict[str, Any]] = []
        self._disk_cache_ts = float("-inf")
        self._sweeper: asyncio.Task | None = None

    @property
    def peers(self) -> list[str]:
        return list(self._peers)

    def has_local(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._entries

    # -- lifecycle ----------------------------------------------------------
    async def start(self) -> None:
        if self.enabled and self._sweeper is None:
            self._sweeper = asyncio.create_task(self._sweep_loop())

    async def stop(self) -> None:
        if self._sweeper is not None:
            self._sweeper.cancel()
            try:
                await self._sweeper
            except asyncio.CancelledError:
                pass
            self._sweeper = None

    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(self._sweep_interval)
            self.sweep()

    def sweep(self) -> None:
        now = self._now()
        stale_transitions: list[RunEntry] = []
        with self._lock:
            for entry in self._entries.values():
                if (
                    entry.status == "active"
                    and now - entry.last_seen > ACTIVE_IDLE_STALE_SECONDS
                ):
                    entry.status = "stale"
                    stale_transitions.append(entry)
            evict = [
                run_id
                for run_id, entry in self._entries.items()
                if (
                    entry.status in _TERMINAL
                    and now - entry.last_seen > EVICT_TERMINAL_SECONDS
                )
                or (
                    entry.status == "stale"
                    and now - entry.last_seen > EVICT_STALE_SECONDS
                )
            ]
            if len(self._entries) - len(evict) > MAX_RUNS:
                survivors = sorted(
                    (e for r, e in self._entries.items() if r not in evict),
                    key=lambda e: e.last_seen,
                )
                overflow = len(survivors) - MAX_RUNS
                evict.extend(e.run_id for e in survivors[:overflow])
            for run_id in evict:
                self._entries.pop(run_id, None)
                self._events.pop(run_id, None)
                self._seq.pop(run_id, None)
        for entry in stale_transitions:
            self._broadcast(entry.run_id, "status", self._status_payload(entry))

    # -- auth ---------------------------------------------------------------
    async def authorize(self, authorization: str | None) -> tuple[bool, str | None]:
        if self.auth_mode == "open":
            return True, None
        scheme, _, supplied = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not supplied:
            return False, None
        if self.auth_mode == "token":
            ok = bool(self._token) and secrets.compare_digest(supplied, self._token)
            return ok, None
        key = hashlib.sha256(supplied.encode("utf-8")).hexdigest()
        now = self._now()
        cached = self._token_cache.get(key)
        if cached is not None:
            user, checked_at = cached
            ttl = 3600.0 if user else 60.0
            if now - checked_at < ttl:
                return bool(user), user
        loop = asyncio.get_running_loop()
        user = await loop.run_in_executor(None, self._whoami_fn, supplied)
        if len(self._token_cache) >= 128:
            self._token_cache.pop(next(iter(self._token_cache)))
        self._token_cache[key] = (user, now)
        return bool(user), user

    # -- ingest -------------------------------------------------------------
    def ingest(
        self, signal: str, payload: Any, *, user: str | None = None
    ) -> dict[str, Any]:
        rejected = 0
        for record in _iter_otlp_records(signal, payload):
            run_id = record.get("run_id")
            if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
                rejected += 1
                continue
            self._ingest_event(record, user)
        if rejected:
            key = "rejectedSpans" if signal == "traces" else "rejectedLogRecords"
            return {"partialSuccess": {key: rejected}}
        return {}

    def _ingest_event(self, record: dict[str, Any], user: str | None) -> None:
        run_id: str = record["run_id"]
        now = self._now()
        with self._lock:
            entry = self._entries.get(run_id)
            if entry is None:
                entry = RunEntry(run_id=run_id, first_seen=now)
                self._entries[run_id] = entry
            previous_status = entry.status
            entry.last_seen = now
            entry.event_count += 1
            if record.get("operation"):
                entry.operation = str(record["operation"])
            if record.get("trace_id"):
                entry.trace_id = str(record["trace_id"])
            if record.get("instance_id"):
                entry.instance_id = str(record["instance_id"])
            if user:
                entry.user = user
            body = record.get("body") or ""
            status_code = record.get("status_code")
            if body == "run.started":
                entry.status = "active"
            elif body == "run.completed" or status_code == 1:
                entry.status = "completed"
            elif body == "run.failed" or status_code == 2:
                entry.status = "failed"
            elif entry.status == "stale":
                entry.status = "active"
            attrs = record.get("attrs") or {}
            entry.latest["body"] = body
            entry.latest["ts"] = record.get("ts")
            if "slm.step" in attrs:
                entry.latest["step"] = attrs["slm.step"]
            if "slm.loss" in attrs:
                entry.latest["loss"] = attrs["slm.loss"]
            seq = self._seq.get(run_id, 0) + 1
            self._seq[run_id] = seq
            event = {**record, "seq": seq}
            self._events.setdefault(run_id, deque(maxlen=RING_BUFFER_SIZE)).append(
                event
            )
            status_changed = entry.status != previous_status
            status_payload = self._status_payload(entry)
        if status_changed:
            self._broadcast(run_id, "status", status_payload)
        self._broadcast(run_id, "otel", event)

    def _status_payload(self, entry: RunEntry) -> dict[str, Any]:
        return {**entry.to_dict(), "hub_epoch": self.hub_epoch}

    # -- reads --------------------------------------------------------------
    def local_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            entries = [entry.to_dict() for entry in self._entries.values()]
        entries.sort(key=lambda row: row["last_seen"], reverse=True)
        return entries

    def events(
        self, run_id: str, *, since: int = 0, limit: int = 200
    ) -> dict[str, Any]:
        with self._lock:
            entry = self._entries.get(run_id)
            run = self._status_payload(entry) if entry else None
            buffered = list(self._events.get(run_id, ()))
        selected = [event for event in buffered if event["seq"] > since][:limit]
        next_seq = selected[-1]["seq"] if selected else since
        return {
            "hub_epoch": self.hub_epoch,
            "run": run,
            "events": selected,
            "next": next_seq,
        }

    def peer_snapshot(self) -> list[dict[str, Any]]:
        """Fetch each peer's local-only view (blocking; call off the event loop)."""
        if not self._peers:
            return []
        now = self._now()
        with self._peer_lock:
            if now - self._peer_cache_ts < FEDERATION_CACHE_SECONDS:
                return self._peer_cache
        snapshot: list[dict[str, Any]] = []
        for peer in self._peers:
            try:
                data = self._fetch_json(f"{peer}/api/otel/runs?local=1")
                runs = [
                    {**row, "source": "peer", "peer": peer}
                    for row in (data.get("runs") or [])
                    if isinstance(row, dict) and row.get("run_id")
                ]
                snapshot.append({"url": peer, "ok": True, "error": None, "runs": runs})
            except Exception as exc:  # noqa: BLE001 — peer reachability is best-effort
                snapshot.append(
                    {"url": peer, "ok": False, "error": str(exc), "runs": []}
                )
        with self._peer_lock:
            self._peer_cache = snapshot
            self._peer_cache_ts = now
        return snapshot

    def disk_runs(self) -> list[dict[str, Any]]:
        """Zero-config fallback: recent local metrics.jsonl activity (list-only)."""
        now = self._now()
        if now - self._disk_cache_ts < FEDERATION_CACHE_SECONDS:
            return self._disk_cache
        rows: list[dict[str, Any]] = []
        for metrics in sorted(self._outputs_dir.glob("runs/*/metrics.jsonl")):
            run_id = metrics.parent.name
            if not _RUN_ID_RE.match(run_id):
                continue
            try:
                mtime = metrics.stat().st_mtime
            except OSError:
                continue
            if now - mtime > DISK_ACTIVE_WINDOW_SECONDS:
                continue
            latest: dict[str, Any] = {}
            last = _tail_line(metrics)
            if last:
                try:
                    row = json.loads(last)
                    if isinstance(row, dict):
                        if row.get("step") is not None:
                            latest["step"] = row["step"]
                        if row.get("loss") is not None:
                            latest["loss"] = row["loss"]
                except ValueError:
                    pass
            rows.append(
                {
                    "run_id": run_id,
                    "operation": "train",
                    "user": None,
                    "trace_id": None,
                    "instance_id": None,
                    "status": "active",
                    "first_seen": None,
                    "last_seen": mtime,
                    "event_count": 0,
                    "latest": latest,
                    "source": "disk",
                }
            )
        rows.sort(key=lambda row: row["last_seen"], reverse=True)
        self._disk_cache = rows
        self._disk_cache_ts = now
        return rows

    def merged_runs(self, *, local_only: bool = False) -> dict[str, Any]:
        runs = self.local_runs()
        payload: dict[str, Any] = {
            "enabled": self.enabled,
            "hub_epoch": self.hub_epoch,
            "auth_mode": self.auth_mode,
            "runs": runs,
            "peers": [],
        }
        if local_only:
            return payload
        seen = {row["run_id"] for row in runs}
        for peer in self.peer_snapshot():
            payload["peers"].append(
                {"url": peer["url"], "ok": peer["ok"], "error": peer["error"]}
            )
            for row in peer["runs"]:
                if row["run_id"] not in seen:
                    seen.add(row["run_id"])
                    runs.append(row)
        for row in self.disk_runs():
            if row["run_id"] not in seen:
                seen.add(row["run_id"])
                runs.append(row)
        return payload

    def find_peer_for(self, run_id: str) -> str | None:
        for peer in self.peer_snapshot():
            for row in peer["runs"]:
                if row["run_id"] == run_id:
                    return peer["url"]
        return None

    # -- streaming ----------------------------------------------------------
    def _broadcast(self, run_id: str, kind: str, data: dict[str, Any]) -> None:
        for state in list(self._subs.get(run_id, {}).values()):
            try:
                state.queue.put_nowait((kind, data))
            except asyncio.QueueFull:
                try:
                    state.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                state.dropped += 1
                try:
                    state.queue.put_nowait((kind, data))
                except asyncio.QueueFull:
                    state.dropped += 1

    def subscriber_count(self, run_id: str) -> int:
        return len(self._subs.get(run_id, {}))

    async def stream(self, run_id: str, *, since: int = 0) -> AsyncIterator[str]:
        subs = self._subs.setdefault(run_id, {})
        if len(subs) >= MAX_SUBSCRIBERS:
            yield _sse("error", {"error": "too many subscribers"})
            if not subs:
                self._subs.pop(run_id, None)
            return
        # Register before the first yield: events arriving while the client
        # consumes the replay land in the queue, deduped below via last_seq.
        state = _SubState(queue=asyncio.Queue(maxsize=QUEUE_SIZE))
        subs[id(state)] = state
        try:
            with self._lock:
                entry = self._entries.get(run_id)
                status = (
                    self._status_payload(entry)
                    if entry
                    else {
                        "run_id": run_id,
                        "status": "unknown",
                        "hub_epoch": self.hub_epoch,
                    }
                )
                replay = [
                    event
                    for event in self._events.get(run_id, ())
                    if event["seq"] > since
                ]
            last_seq = since
            # First frame is emitted without awaiting so TestClient streams and
            # EventSource handshakes always observe an immediate byte flow.
            yield _sse("status", status)
            for event in replay:
                last_seq = max(last_seq, event["seq"])
                yield _sse("otel", {**event, "hub_epoch": self.hub_epoch})
            while True:
                try:
                    kind, data = await asyncio.wait_for(
                        state.queue.get(), timeout=self._ping_interval
                    )
                except asyncio.TimeoutError:
                    yield _sse("ping", {"hub_epoch": self.hub_epoch})
                    continue
                if state.dropped:
                    yield _sse("dropped", {"count": state.dropped})
                    state.dropped = 0
                if kind == "otel":
                    seq = data.get("seq")
                    if isinstance(seq, int):
                        if seq <= last_seq:
                            continue
                        last_seq = seq
                    data = {**data, "hub_epoch": self.hub_epoch}
                yield _sse(kind, data)
        finally:
            subs.pop(id(state), None)
            if not subs:
                self._subs.pop(run_id, None)

    async def stream_remote(
        self, run_id: str, peer: str, *, since: int = 0
    ) -> AsyncIterator[str]:
        """Bridge a peer's run events into a local SSE stream by cursor-polling.

        The upstream connection only exists while this generator is being
        consumed — client disconnect cancels it at the next await point.
        """
        yield _sse(
            "status",
            {"run_id": run_id, "status": "connecting", "peer": peer,
             "hub_epoch": self.hub_epoch},
        )
        loop = asyncio.get_running_loop()
        cursor = since
        errors = 0
        idle_terminal_polls = 0
        last_emit = self._now()
        last_status: str | None = None
        while True:
            url = f"{peer}/api/otel/runs/{run_id}/events?since={cursor}&local=1"
            try:
                data = await loop.run_in_executor(None, self._fetch_json, url)
                errors = 0
            except Exception as exc:  # noqa: BLE001 — surface then keep trying briefly
                errors += 1
                if errors >= 5:
                    yield _sse("error", {"error": f"peer unreachable: {exc}"})
                    return
                await asyncio.sleep(self._poll_interval)
                continue
            run = data.get("run") if isinstance(data, dict) else None
            events = data.get("events") if isinstance(data, dict) else None
            for event in events or []:
                if isinstance(event, dict) and isinstance(event.get("seq"), int):
                    cursor = max(cursor, event["seq"])
                yield _sse("otel", {**event, "peer": peer})
                last_emit = self._now()
            status = run.get("status") if isinstance(run, dict) else None
            if run is not None and status != last_status:
                last_status = status
                yield _sse("status", {**run, "peer": peer})
                last_emit = self._now()
            if run is None or status in _TERMINAL:
                idle_terminal_polls = idle_terminal_polls + 1 if not events else 0
                if idle_terminal_polls >= 2:
                    if run is None:
                        yield _sse(
                            "status",
                            {"run_id": run_id, "status": "unknown", "peer": peer},
                        )
                    return
            else:
                idle_terminal_polls = 0
            if self._now() - last_emit >= self._ping_interval:
                yield _sse("ping", {"peer": peer, "hub_epoch": self.hub_epoch})
                last_emit = self._now()
            await asyncio.sleep(self._poll_interval)
