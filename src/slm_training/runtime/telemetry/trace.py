"""W3C-correlated local OTLP JSONL traces with an optional OTLP/HTTP mirror."""

from __future__ import annotations

import contextvars
import json
import os
import secrets
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CURRENT: contextvars.ContextVar["RunTrace | None"] = contextvars.ContextVar(
    "slm_run_trace", default=None
)


def _hex_id(size: int) -> str:
    value = secrets.token_hex(size)
    return value if any(char != "0" for char in value) else "1".zfill(size * 2)


def _attrs(values: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, bool):
            encoded = {"boolValue": value}
        elif isinstance(value, int):
            encoded = {"intValue": str(value)}
        elif isinstance(value, float):
            encoded = {"doubleValue": value}
        else:
            encoded = {"stringValue": str(value)}
        rows.append({"key": key, "value": encoded})
    return rows


def _first_peer() -> str | None:
    peers = os.getenv("SLM_OTEL_PEERS", "")
    for peer in peers.split(","):
        if peer.strip():
            return peer.strip()
    return None


def _endpoint(signal: str) -> str | None:
    specific = os.getenv(f"OTEL_EXPORTER_OTLP_{signal.upper()}_ENDPOINT")
    base = specific or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or _first_peer()
    if not base:
        return None
    return base if specific else f"{base.rstrip('/')}/v1/{signal}"


def _headers() -> dict[str, str]:
    raw = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
    if raw:
        headers: dict[str, str] = {}
        for pair in raw.split(","):
            key, sep, value = pair.partition("=")
            if sep and key.strip():
                headers[key.strip()] = value.strip()
        return headers
    token = os.getenv("SLM_OTEL_TOKEN")
    if not token and os.getenv("SLM_OTEL_AUTH", "").strip().lower() == "hf":
        # HF_TOKEN is credential-bearing; forward it only on explicit opt-in.
        token = os.getenv("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


@dataclass
class RunTrace:
    run_id: str
    operation: str
    trace_root: Path = Path("outputs/traces")
    run_dir: Path | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""
    root_span_id: str = ""
    start_ns: int = 0

    def __post_init__(self) -> None:
        reference = self.run_dir / "trace.json" if self.run_dir else None
        existing: dict[str, Any] = {}
        if reference and reference.is_file():
            existing = json.loads(reference.read_text(encoding="utf-8"))
        self.trace_id = str(existing.get("trace_id") or _hex_id(16))
        self.root_span_id = str(existing.get("root_span_id") or _hex_id(8))
        first = not existing
        self.span_id = self.root_span_id if first else _hex_id(8)
        self.parent_span_id = "" if first else self.root_span_id
        self.start_ns = time.time_ns()
        self.trace_root = Path(self.trace_root)
        self.bundle = self.trace_root / self.trace_id
        self.instance_id = str(uuid.uuid4())
        self._token = None
        self._write_manifest()
        if reference:
            reference.parent.mkdir(parents=True, exist_ok=True)
            reference.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": self.run_id,
                        "trace_id": self.trace_id,
                        "root_span_id": self.root_span_id,
                        "traceparent": self.traceparent,
                        "bundle": self.bundle.as_posix(),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    @property
    def traceparent(self) -> str:
        return f"00-{self.trace_id}-{self.span_id}-01"

    def __enter__(self) -> "RunTrace":
        self._token = _CURRENT.set(self)
        self.log("run.started", attributes={"slm.operation": self.operation})
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        status = 2 if exc else 1
        self.log(
            "run.failed" if exc else "run.completed",
            severity="ERROR" if exc else "INFO",
            attributes={"error.type": exc_type.__name__ if exc_type else None},
        )
        payload = self._trace_payload(time.time_ns(), status, str(exc) if exc else "")
        self._append("traces", payload)
        self._mirror("traces", payload)
        if self._token is not None:
            _CURRENT.reset(self._token)

    def domain_path(self, kind: str, name: str = "records.jsonl") -> Path:
        if not kind.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"unsafe domain trace kind: {kind!r}")
        path = self.bundle / "domain" / kind / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def log(
        self,
        body: str,
        *,
        severity: str = "INFO",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        now = str(time.time_ns())
        record = {
            "timeUnixNano": now,
            "observedTimeUnixNano": now,
            "severityText": severity,
            "body": {"stringValue": body},
            "attributes": _attrs({**self._common_attributes(), **(attributes or {})}),
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "flags": 1,
        }
        payload = {
            "resourceLogs": [
                {
                    "resource": {"attributes": _attrs(self._resource_attributes())},
                    "scopeLogs": [
                        {
                            "scope": {"name": "slm_training", "version": "1"},
                            "logRecords": [record],
                        }
                    ],
                }
            ]
        }
        self._append("logs", payload)
        self._mirror("logs", payload)

    def _trace_payload(self, end_ns: int, status: int, message: str) -> dict[str, Any]:
        span = {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "name": self.operation,
            "kind": 1,
            "startTimeUnixNano": str(self.start_ns),
            "endTimeUnixNano": str(end_ns),
            "attributes": _attrs(self._common_attributes()),
            "status": {"code": status, "message": message},
        }
        if self.parent_span_id:
            span["parentSpanId"] = self.parent_span_id
        return {
            "resourceSpans": [
                {
                    "resource": {"attributes": _attrs(self._resource_attributes())},
                    "scopeSpans": [
                        {
                            "scope": {"name": "slm_training", "version": "1"},
                            "spans": [span],
                        }
                    ],
                }
            ]
        }

    def _common_attributes(self) -> dict[str, Any]:
        return {"slm.run.id": self.run_id, "slm.operation": self.operation, **self.attributes}

    def _resource_attributes(self) -> dict[str, Any]:
        return {
            "service.name": "slm-training",
            "service.namespace": "openui",
            "service.version": "0.1.0",
            "service.instance.id": self.instance_id,
        }

    def _append(self, signal: str, payload: dict[str, Any]) -> None:
        path = self.bundle / "signals" / signal / f"{self.instance_id}.otlp.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def _mirror(self, signal: str, payload: dict[str, Any]) -> None:
        endpoint = _endpoint(signal)
        if not endpoint:
            return
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **_headers()},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=2):  # noqa: S310
                pass
        except OSError as exc:
            self._write_manifest(export_error=str(exc))

    def _write_manifest(self, *, export_error: str | None = None) -> None:
        self.bundle = self.trace_root / self.trace_id
        self.bundle.mkdir(parents=True, exist_ok=True)
        path = self.bundle / "manifest.json"
        payload = {
            "schema_version": 1,
            "trace_id": self.trace_id,
            "run_ids": sorted(
                {
                    self.run_id,
                    *(
                        json.loads(path.read_text(encoding="utf-8")).get("run_ids", [])
                        if path.is_file()
                        else []
                    ),
                }
            ),
            "traceparent": self.traceparent,
            "otlp_json": True,
            "remote_endpoint_configured": bool(_endpoint("traces")),
            "last_export_error": export_error,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def current_trace() -> RunTrace | None:
    return _CURRENT.get()


def run_trace(
    run_id: str,
    operation: str,
    *,
    run_dir: Path | None = None,
    trace_root: Path | None = None,
    attributes: dict[str, Any] | None = None,
) -> RunTrace:
    return RunTrace(
        run_id=run_id,
        operation=operation,
        run_dir=run_dir,
        trace_root=(
            trace_root
            if trace_root is not None
            else run_dir.parent.parent / "traces"
            if run_dir is not None
            else Path("outputs/traces")
        ),
        attributes=attributes or {},
    )


__all__ = ["RunTrace", "current_trace", "run_trace"]
