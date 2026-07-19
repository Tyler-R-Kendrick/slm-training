from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from slm_training.web import otel_hub
from slm_training.web.otel_hub import OtelHub, _iter_otlp_records


def _attr_rows(values: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, value in values.items():
        if isinstance(value, bool):
            encoded: dict[str, Any] = {"boolValue": value}
        elif isinstance(value, int):
            encoded = {"intValue": str(value)}
        elif isinstance(value, float):
            encoded = {"doubleValue": value}
        else:
            encoded = {"stringValue": str(value)}
        rows.append({"key": key, "value": encoded})
    return rows


def _log_payload(
    run_id: str | None,
    body: str,
    attrs: dict[str, Any] | None = None,
    *,
    instance: str = "inst-1",
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(attrs or {})
    if run_id is not None:
        merged.setdefault("slm.run.id", run_id)
        merged.setdefault("slm.operation", "train")
    return {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": _attr_rows({"service.instance.id": instance})
                },
                "scopeLogs": [
                    {
                        "scope": {"name": "slm_training", "version": "1"},
                        "logRecords": [
                            {
                                "timeUnixNano": "1000",
                                "severityText": "INFO",
                                "body": {"stringValue": body},
                                "attributes": _attr_rows(merged),
                                "traceId": "t" * 32,
                                "spanId": "s" * 16,
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _span_payload(run_id: str, status_code: int) -> dict[str, Any]:
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": _attr_rows({"service.instance.id": "inst-1"})
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "slm_training", "version": "1"},
                        "spans": [
                            {
                                "traceId": "t" * 32,
                                "spanId": "s" * 16,
                                "name": "train",
                                "startTimeUnixNano": "1000",
                                "endTimeUnixNano": "2000",
                                "attributes": _attr_rows(
                                    {"slm.run.id": run_id, "slm.operation": "train"}
                                ),
                                "status": {"code": status_code, "message": ""},
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _parse_frames(raw: str) -> list[tuple[str, dict[str, Any]]]:
    frames = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue
        event = data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        frames.append((event, data))
    return frames


class Clock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


async def _collect(gen, count: int, timeout: float = 2.0) -> list[str]:
    frames: list[str] = []

    async def pull() -> None:
        async for frame in gen:
            frames.append(frame)
            if len(frames) >= count:
                return

    try:
        await asyncio.wait_for(pull(), timeout)
    finally:
        await gen.aclose()
    return frames


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def test_parser_handles_string_int_values_and_gaps() -> None:
    payload = _log_payload("run-a", "train.progress", {"slm.step": 7, "slm.loss": 1.5})
    records = list(_iter_otlp_records("logs", payload))
    assert len(records) == 1
    record = records[0]
    assert record["run_id"] == "run-a"
    assert record["attrs"]["slm.step"] == 7
    assert record["attrs"]["slm.loss"] == 1.5
    assert record["instance_id"] == "inst-1"

    assert list(_iter_otlp_records("logs", {"resourceLogs": [{}]})) == []
    assert list(_iter_otlp_records("logs", "garbage")) == []
    assert list(_iter_otlp_records("traces", {"resourceSpans": [{"scopeSpans": []}]})) == []


def test_ingest_counts_records_without_run_id_as_partial_success() -> None:
    hub = OtelHub(now_fn=Clock())
    response = hub.ingest("logs", _log_payload(None, "orphan"))
    assert response == {"partialSuccess": {"rejectedLogRecords": 1}}
    response = hub.ingest("traces", {"resourceSpans": []})
    assert response == {}
    assert hub.local_runs() == []


def test_ingest_rejects_hostile_run_ids() -> None:
    hub = OtelHub(now_fn=Clock())
    response = hub.ingest("logs", _log_payload("../escape", "run.started"))
    assert response == {"partialSuccess": {"rejectedLogRecords": 1}}
    assert hub.local_runs() == []


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
def test_lifecycle_transitions_and_revival() -> None:
    clock = Clock()
    hub = OtelHub(now_fn=clock)
    hub.ingest("logs", _log_payload("run-a", "run.started"))
    (entry,) = hub.local_runs()
    assert entry["status"] == "active"
    assert entry["operation"] == "train"

    hub.ingest("logs", _log_payload("run-a", "run.completed"))
    assert hub.local_runs()[0]["status"] == "completed"

    # run ids are reused across phases (train -> eval): started revives.
    hub.ingest("logs", _log_payload("run-a", "run.started"))
    assert hub.local_runs()[0]["status"] == "active"

    hub.ingest("logs", _log_payload("run-a", "run.failed"))
    assert hub.local_runs()[0]["status"] == "failed"

    hub.ingest("traces", _span_payload("run-b", 1))
    hub.ingest("traces", _span_payload("run-c", 2))
    by_id = {row["run_id"]: row for row in hub.local_runs()}
    assert by_id["run-b"]["status"] == "completed"
    assert by_id["run-c"]["status"] == "failed"


def test_latest_summary_tracks_progress_attrs() -> None:
    hub = OtelHub(now_fn=Clock())
    hub.ingest("logs", _log_payload("run-a", "run.started"))
    hub.ingest(
        "logs",
        _log_payload("run-a", "train.progress", {"slm.step": 40, "slm.loss": 2.25}),
    )
    entry = hub.local_runs()[0]
    assert entry["latest"]["step"] == 40
    assert entry["latest"]["loss"] == 2.25
    assert entry["latest"]["body"] == "train.progress"
    assert entry["event_count"] == 2


def test_sweep_marks_stale_then_evicts() -> None:
    clock = Clock()
    hub = OtelHub(now_fn=clock)
    hub.ingest("logs", _log_payload("run-a", "run.started"))
    clock.t += otel_hub.ACTIVE_IDLE_STALE_SECONDS + 1
    hub.sweep()
    assert hub.local_runs()[0]["status"] == "stale"

    # Any fresh event revives a stale run (hub restarts mid-run recover).
    hub.ingest("logs", _log_payload("run-a", "train.progress", {"slm.step": 3}))
    assert hub.local_runs()[0]["status"] == "active"

    hub.ingest("logs", _log_payload("run-a", "run.completed"))
    clock.t += otel_hub.EVICT_TERMINAL_SECONDS + 1
    hub.sweep()
    assert hub.local_runs() == []


def test_sweep_enforces_max_runs_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(otel_hub, "MAX_RUNS", 3)
    clock = Clock()
    hub = OtelHub(now_fn=clock)
    for index in range(5):
        hub.ingest("logs", _log_payload(f"run-{index}", "run.started"))
        clock.t += 1
    hub.sweep()
    survivors = {row["run_id"] for row in hub.local_runs()}
    assert survivors == {"run-2", "run-3", "run-4"}


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #
async def test_stream_replays_then_delivers_live_events() -> None:
    hub = OtelHub(now_fn=Clock(), ping_interval=0.05)
    hub.ingest("logs", _log_payload("run-a", "run.started"))
    hub.ingest("logs", _log_payload("run-a", "train.progress", {"slm.step": 1}))

    gen = hub.stream("run-a")
    frames: list[str] = []
    frames.append(await gen.__anext__())  # status
    frames.append(await gen.__anext__())  # replay seq 1
    frames.append(await gen.__anext__())  # replay seq 2
    assert hub.subscriber_count("run-a") == 1

    hub.ingest("logs", _log_payload("run-a", "train.progress", {"slm.step": 2}))
    frames.append(await asyncio.wait_for(gen.__anext__(), 1.0))
    await gen.aclose()

    parsed = _parse_frames("".join(frames))
    kinds = [kind for kind, _ in parsed]
    assert kinds[0] == "status"
    assert parsed[0][1]["status"] == "active"
    otel_frames = [data for kind, data in parsed if kind == "otel"]
    assert [frame["seq"] for frame in otel_frames] == [1, 2, 3]
    assert all("hub_epoch" in data for _, data in parsed)
    assert hub.subscriber_count("run-a") == 0


async def test_stream_since_cursor_skips_replayed_events() -> None:
    hub = OtelHub(now_fn=Clock(), ping_interval=0.05)
    hub.ingest("logs", _log_payload("run-a", "run.started"))
    hub.ingest("logs", _log_payload("run-a", "train.progress", {"slm.step": 1}))
    frames = await _collect(hub.stream("run-a", since=1), 2)
    parsed = _parse_frames("".join(frames))
    otel_frames = [data for kind, data in parsed if kind == "otel"]
    assert [frame["seq"] for frame in otel_frames] == [2]


async def test_stream_emits_ping_when_idle_and_unknown_status() -> None:
    hub = OtelHub(now_fn=Clock(), ping_interval=0.01)
    frames = await _collect(hub.stream("missing"), 2)
    parsed = _parse_frames("".join(frames))
    assert parsed[0][0] == "status"
    assert parsed[0][1]["status"] == "unknown"
    assert parsed[1][0] == "ping"


async def test_stream_overflow_emits_dropped_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(otel_hub, "QUEUE_SIZE", 4)
    hub = OtelHub(now_fn=Clock(), ping_interval=0.05)
    hub.ingest("logs", _log_payload("run-a", "run.started"))
    gen = hub.stream("run-a")
    await gen.__anext__()  # status
    await gen.__anext__()  # replay of run.started
    for step in range(10):
        hub.ingest(
            "logs", _log_payload("run-a", "train.progress", {"slm.step": step})
        )
    frames = []
    for _ in range(5):
        frames.append(await asyncio.wait_for(gen.__anext__(), 1.0))
    await gen.aclose()
    parsed = _parse_frames("".join(frames))
    kinds = [kind for kind, _ in parsed]
    assert "dropped" in kinds
    dropped = next(data for kind, data in parsed if kind == "dropped")
    assert dropped["count"] > 0


async def test_no_subscriber_queues_without_streams() -> None:
    hub = OtelHub(now_fn=Clock())
    hub.ingest("logs", _log_payload("run-a", "run.started"))
    for step in range(50):
        hub.ingest("logs", _log_payload("run-a", "train.progress", {"slm.step": step}))
    assert hub.subscriber_count("run-a") == 0
    assert hub._subs == {}


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
async def test_authorize_modes() -> None:
    open_hub = OtelHub(now_fn=Clock())
    assert open_hub.auth_mode == "open"
    assert await open_hub.authorize(None) == (True, None)

    token_hub = OtelHub(now_fn=Clock(), token="shared-secret")
    assert token_hub.auth_mode == "token"
    assert await token_hub.authorize("Bearer shared-secret") == (True, None)
    assert (await token_hub.authorize("Bearer wrong"))[0] is False
    assert (await token_hub.authorize(None))[0] is False

    calls: list[str] = []

    def whoami(token: str) -> str | None:
        calls.append(token)
        return "tyler" if token == "hf_valid" else None

    hf_hub = OtelHub(now_fn=Clock(), auth_mode="hf", whoami_fn=whoami)
    assert await hf_hub.authorize("Bearer hf_valid") == (True, "tyler")
    assert await hf_hub.authorize("Bearer hf_valid") == (True, "tyler")
    assert calls == ["hf_valid"]  # cached second time
    assert (await hf_hub.authorize("Bearer hf_bogus"))[0] is False


def test_ingest_stamps_authenticated_user() -> None:
    hub = OtelHub(now_fn=Clock())
    hub.ingest("logs", _log_payload("run-a", "run.started"), user="tyler")
    hub.ingest("logs", _log_payload("run-a", "train.progress", {"slm.step": 1}))
    entry = hub.local_runs()[0]
    assert entry["user"] == "tyler"  # later anonymous events keep attribution


# --------------------------------------------------------------------------- #
# Federation + disk fallback
# --------------------------------------------------------------------------- #
def _peer_fetch(responses: dict[str, Any]):
    calls: list[str] = []

    def fetch(url: str) -> Any:
        calls.append(url)
        for prefix, payload in responses.items():
            if url.startswith(prefix):
                if isinstance(payload, Exception):
                    raise payload
                return payload
        raise AssertionError(f"unexpected url {url}")

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


def test_merged_runs_dedupes_with_precedence(tmp_path: Path) -> None:
    clock = Clock()
    outputs = tmp_path / "outputs"
    run_dir = outputs / "runs" / "run-disk"
    run_dir.mkdir(parents=True)
    metrics = run_dir / "metrics.jsonl"
    metrics.write_text('{"step": 5, "loss": 3.0}\n')
    shared_dir = outputs / "runs" / "run-a"
    shared_dir.mkdir(parents=True)
    (shared_dir / "metrics.jsonl").write_text('{"step": 1, "loss": 9.0}\n')
    clock.t = metrics.stat().st_mtime + 1  # inside the disk-active window

    peer_rows = {
        "runs": [
            {"run_id": "run-a", "status": "active", "source": "local"},
            {"run_id": "run-peer", "status": "active", "source": "local"},
        ]
    }
    fetch = _peer_fetch(
        {
            "http://peer-a/api/otel/runs": peer_rows,
            "http://peer-b/api/otel/runs": OSError("down"),
        }
    )
    hub = OtelHub(
        now_fn=clock,
        outputs_dir=outputs,
        peers=["http://peer-a/", "http://peer-b"],
        fetch_json=fetch,
    )
    hub.ingest("logs", _log_payload("run-a", "run.started"))

    payload = hub.merged_runs()
    by_id = {row["run_id"]: row for row in payload["runs"]}
    # local ingest wins over the peer copy of the same run
    assert by_id["run-a"]["source"] == "local"
    assert by_id["run-peer"]["source"] == "peer"
    assert by_id["run-peer"]["peer"] == "http://peer-a"
    assert by_id["run-disk"]["source"] == "disk"
    assert by_id["run-disk"]["latest"] == {"step": 5, "loss": 3.0}
    peers = {peer["url"]: peer for peer in payload["peers"]}
    assert peers["http://peer-a"]["ok"] is True
    assert peers["http://peer-b"]["ok"] is False
    assert "down" in peers["http://peer-b"]["error"]

    # local_only view never contacts peers (loop-safe federation contract)
    local_payload = hub.merged_runs(local_only=True)
    assert [row["run_id"] for row in local_payload["runs"]] == ["run-a"]


def test_find_peer_for_uses_cached_snapshot(tmp_path: Path) -> None:
    fetch = _peer_fetch(
        {
            "http://peer-a/api/otel/runs": {
                "runs": [{"run_id": "run-x", "status": "active"}]
            }
        }
    )
    hub = OtelHub(
        now_fn=Clock(), outputs_dir=tmp_path, peers=["http://peer-a"], fetch_json=fetch
    )
    assert hub.find_peer_for("run-x") == "http://peer-a"
    assert hub.find_peer_for("run-x") == "http://peer-a"
    assert len(fetch.calls) == 1  # second lookup served from the 10s cache
    assert hub.find_peer_for("missing") is None


def test_disk_runs_ignores_old_and_invalid(tmp_path: Path) -> None:
    clock = Clock()
    outputs = tmp_path / "outputs"
    fresh = outputs / "runs" / "fresh"
    fresh.mkdir(parents=True)
    (fresh / "metrics.jsonl").write_text('{"step": 2, "loss": 1.0}\nnot-json\n')
    old = outputs / "runs" / "old"
    old.mkdir(parents=True)
    (old / "metrics.jsonl").write_text('{"step": 1}\n')
    hub = OtelHub(now_fn=clock, outputs_dir=outputs)
    clock.t = (fresh / "metrics.jsonl").stat().st_mtime + 1
    import os

    stale_time = clock.t - otel_hub.DISK_ACTIVE_WINDOW_SECONDS - 5
    os.utime(old / "metrics.jsonl", (stale_time, stale_time))
    rows = hub.disk_runs()
    assert [row["run_id"] for row in rows] == ["fresh"]
    assert rows[0]["latest"] == {}  # tolerant of a torn/invalid last line


async def test_stream_remote_bridges_and_terminates() -> None:
    responses = [
        {
            "run": {"run_id": "run-x", "status": "active"},
            "events": [
                {"seq": 1, "body": "run.started"},
                {"seq": 2, "body": "train.progress"},
            ],
            "next": 2,
        },
        {"run": {"run_id": "run-x", "status": "completed"}, "events": [], "next": 2},
        {"run": {"run_id": "run-x", "status": "completed"}, "events": [], "next": 2},
        {"run": {"run_id": "run-x", "status": "completed"}, "events": [], "next": 2},
    ]
    urls: list[str] = []

    def fetch(url: str) -> Any:
        urls.append(url)
        return responses.pop(0) if responses else responses_exhausted()

    def responses_exhausted() -> Any:
        raise AssertionError("bridge should have terminated")

    hub = OtelHub(now_fn=Clock(), fetch_json=fetch, poll_interval=0.01)
    frames = []
    async for frame in hub.stream_remote("run-x", "http://peer-a"):
        frames.append(frame)
    parsed = _parse_frames("".join(frames))
    kinds = [kind for kind, _ in parsed]
    assert kinds[0] == "status"  # connecting
    assert kinds.count("otel") == 2
    statuses = [data.get("status") for kind, data in parsed if kind == "status"]
    assert statuses[-1] == "completed"
    assert urls[0].endswith("/api/otel/runs/run-x/events?since=0&local=1")
    assert urls[1].endswith("since=2&local=1")  # cursor advanced


async def test_stream_remote_reports_unreachable_peer() -> None:
    def fetch(url: str) -> Any:
        raise OSError("connection refused")

    hub = OtelHub(now_fn=Clock(), fetch_json=fetch, poll_interval=0.001)
    frames = []
    async for frame in hub.stream_remote("run-x", "http://peer-a"):
        frames.append(frame)
    parsed = _parse_frames("".join(frames))
    assert parsed[-1][0] == "error"
    assert "unreachable" in parsed[-1][1]["error"]


# --------------------------------------------------------------------------- #
# App wiring (HTTP surface)
# --------------------------------------------------------------------------- #
def _client(tmp_path: Path, **kwargs: Any):
    from fastapi.testclient import TestClient

    from slm_training.web.app import create_app

    return TestClient(create_app(execution=False, root=tmp_path, **kwargs))


def test_http_ingest_list_events_and_capabilities(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post("/v1/logs", json=_log_payload("run-a", "run.started"))
        assert response.status_code == 200
        assert response.json() == {}

        payload = client.get("/api/otel/runs").json()
        assert payload["enabled"] is True
        assert payload["runs"][0]["run_id"] == "run-a"
        assert payload["runs"][0]["status"] == "active"

        events = client.get("/api/otel/runs/run-a/events").json()
        assert [event["seq"] for event in events["events"]] == [1]

        caps = client.get("/api/capabilities").json()
        assert caps["otel"] == {
            "hub": True,
            "peers_configured": False,
            "auth_mode": "open",
        }


def test_http_ingest_rejects_bad_requests(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert client.post("/v1/metrics", json={}).status_code == 404
        assert (
            client.post(
                "/v1/logs",
                content=b"not json",
                headers={"content-type": "application/json"},
            ).status_code
            == 400
        )
        oversize = b"{" + b" " * otel_hub.MAX_INGEST_BYTES + b"}"
        assert (
            client.post(
                "/v1/logs",
                content=oversize,
                headers={"content-type": "application/json"},
            ).status_code
            == 413
        )
        # A declared oversize length is rejected from the header alone,
        # before any body bytes are buffered.
        assert (
            client.post(
                "/v1/logs",
                content=b"{}",
                headers={
                    "content-type": "application/json",
                    "content-length": str(otel_hub.MAX_INGEST_BYTES + 1),
                },
            ).status_code
            == 413
        )


def test_http_ingest_token_auth(tmp_path: Path) -> None:
    with _client(tmp_path, otel_token="secret") as client:
        payload = _log_payload("run-a", "run.started")
        assert client.post("/v1/logs", json=payload).status_code == 401
        assert (
            client.post(
                "/v1/logs", json=payload, headers={"authorization": "Bearer nope"}
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/v1/logs", json=payload, headers={"authorization": "Bearer secret"}
            ).status_code
            == 200
        )
        assert client.get("/api/capabilities").json()["otel"]["auth_mode"] == "token"


def test_http_ingest_hf_auth_stamps_user(tmp_path: Path) -> None:
    with _client(tmp_path, otel_auth_mode="hf") as client:
        client.app.state.otel._whoami_fn = (
            lambda token: "tyler" if token == "hf_ok" else None
        )
        payload = _log_payload("run-a", "run.started")
        assert client.post("/v1/logs", json=payload).status_code == 401
        assert (
            client.post(
                "/v1/logs", json=payload, headers={"authorization": "Bearer hf_bad"}
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/v1/logs", json=payload, headers={"authorization": "Bearer hf_ok"}
            ).status_code
            == 200
        )
        runs = client.get("/api/otel/runs").json()["runs"]
        assert runs[0]["user"] == "tyler"


def test_serverless_disables_hub_but_reads_through_peers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VERCEL", "1")
    with _client(tmp_path, otel_peers=["http://peer-a"]) as client:
        client.app.state.otel._fetch_json = _peer_fetch(
            {
                "http://peer-a/api/otel/runs?local=1": {
                    "runs": [{"run_id": "run-x", "status": "active"}]
                },
                "http://peer-a/api/otel/runs/run-x/events": {
                    "run": {"run_id": "run-x", "status": "completed"},
                    "events": [],
                    "next": 0,
                },
            }
        )
        assert (
            client.post(
                "/v1/logs", json=_log_payload("run-a", "run.started")
            ).status_code
            == 503
        )
        payload = client.get("/api/otel/runs").json()
        assert payload["enabled"] is False
        assert [row["run_id"] for row in payload["runs"]] == ["run-x"]
        assert payload["runs"][0]["source"] == "peer"
        # unknown run + hub disabled + not on any peer -> no stream possible
        assert client.get("/api/otel/runs/absent/stream").status_code == 503
        assert client.get("/api/capabilities").json()["otel"]["hub"] is False


async def test_http_sse_stream_replays_buffered_events(tmp_path: Path) -> None:
    # Neither TestClient (portal deadlock) nor httpx.ASGITransport (buffers the
    # full body) can consume a never-ending SSE response, so drive the ASGI app
    # directly: run it as a task, collect frames, cancel once we have enough.
    from slm_training.web.app import create_app

    app = create_app(execution=False, root=tmp_path)
    hub = app.state.otel
    hub.ingest("logs", _log_payload("run-a", "run.started"))
    hub.ingest("logs", _log_payload("run-a", "train.progress", {"slm.step": 2}))

    started: dict[str, Any] = {}
    body = bytearray()
    got_enough = asyncio.Event()

    async def receive() -> dict[str, Any]:
        await asyncio.Event().wait()  # the client never disconnects on its own
        raise AssertionError("unreachable")

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            started.update(message)
        elif message["type"] == "http.response.body":
            body.extend(message.get("body", b""))
            if body.count(b"\n\n") >= 3:
                got_enough.set()

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/otel/runs/run-a/stream",
        "raw_path": b"/api/otel/runs/run-a/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"hub.test"), (b"accept", b"text/event-stream")],
        "client": ("127.0.0.1", 40000),
        "server": ("hub.test", 80),
    }
    import contextlib

    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(got_enough.wait(), timeout=5.0)
    finally:
        task.cancel()
        with contextlib.suppress(BaseException):
            await task

    assert started["status"] == 200
    headers = {key.decode(): value.decode() for key, value in started["headers"]}
    assert headers["content-type"].startswith("text/event-stream")
    collected = _parse_frames(body.decode("utf-8"))
    kinds = [kind for kind, _ in collected[:3]]
    assert kinds == ["status", "otel", "otel"]
    assert collected[0][1]["status"] == "active"
    assert collected[2][1]["attrs"]["slm.step"] == 2
    # cancellation ran the generator's cleanup: no subscriber leaked
    assert hub.subscriber_count("run-a") == 0


def test_run_trace_mirror_roundtrip_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import urllib.request

    from slm_training.runtime.telemetry import run_trace

    with _client(tmp_path) as client:

        class _Response:
            def __enter__(self) -> "_Response":
                return self

            def __exit__(self, *args: Any) -> bool:
                return False

        def fake_urlopen(request: Any, timeout: float = 0) -> _Response:
            url = request.full_url
            path = url[url.index("/v1") :]
            response = client.post(
                path,
                content=request.data,
                headers={key: value for key, value in request.header_items()},
            )
            assert response.status_code == 200
            return _Response()

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://hub.example")

        with run_trace(
            "e2e-run", "train", trace_root=tmp_path / "traces"
        ) as trace:
            trace.log(
                "train.progress", attributes={"slm.step": 5, "slm.loss": 2.0}
            )

        runs = client.get("/api/otel/runs").json()["runs"]
        entry = next(row for row in runs if row["run_id"] == "e2e-run")
        assert entry["status"] == "completed"
        assert entry["latest"]["step"] == 5
        assert entry["operation"] == "train"
        assert entry["instance_id"]


# --------------------------------------------------------------------------- #
# Events cursor
# --------------------------------------------------------------------------- #
def test_events_cursor_pagination() -> None:
    hub = OtelHub(now_fn=Clock())
    hub.ingest("logs", _log_payload("run-a", "run.started"))
    for step in range(4):
        hub.ingest("logs", _log_payload("run-a", "train.progress", {"slm.step": step}))
    page = hub.events("run-a", since=0, limit=2)
    assert [event["seq"] for event in page["events"]] == [1, 2]
    assert page["next"] == 2
    assert page["run"]["status"] == "active"
    page = hub.events("run-a", since=page["next"], limit=100)
    assert [event["seq"] for event in page["events"]] == [3, 4, 5]
    assert hub.events("missing", since=0)["run"] is None
