"""W3C-correlated local OTLP JSONL traces with an optional OTLP/HTTP mirror."""

from __future__ import annotations

import contextvars
import json
import os
import secrets
import time
import urllib.request
import uuid
import warnings
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CURRENT: contextvars.ContextVar["RunTrace | None"] = contextvars.ContextVar(
    "slm_run_trace", default=None
)
_ENV_PATH = Path(__file__).resolve().parents[4] / ".env"


def _enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _key_value_env(name: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for pair in os.getenv(name, "").split(","):
        key, sep, value = pair.partition("=")
        if sep and key.strip():
            values[key.strip()] = value.strip()
    return values


def _timeout_seconds(name: str, default: float, maximum: float = 5.0) -> float:
    try:
        return min(maximum, max(0.1, float(os.getenv(name, default))))
    except ValueError:
        return default


def _load_local_env() -> None:
    """Load the repository's ignored local configuration without overriding CI."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_ENV_PATH, override=False)


def _langsmith_api_key() -> str | None:
    value = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not value or value == "replace_with_rotated_langsmith_key":
        return None
    return value


def _langsmith_flush_seconds() -> float:
    return _timeout_seconds("SLM_LANGSMITH_FLUSH_SECONDS", 2.0)


def _otlp_timeout_seconds() -> float:
    """Convert the standard OTLP millisecond setting to a bounded timeout."""
    try:
        value = float(os.getenv("OTEL_EXPORTER_OTLP_TIMEOUT", "1000")) / 1000
    except ValueError:
        value = 1.0
    return min(5.0, max(0.1, value))


class _LangSmithTrace:
    """Best-effort summary exporter; local OTLP remains the source of truth."""

    def __init__(self, trace: "RunTrace") -> None:
        _load_local_env()
        self.trace = trace
        self.client: Any | None = None
        self.error: str | None = None
        api_key = _langsmith_api_key()
        self.config = {
            "enabled": bool(api_key) and _enabled("LANGSMITH_TRACING"),
            "api_key_configured": bool(api_key),
            "project": os.getenv("LANGSMITH_PROJECT", "slm-training"),
            "endpoint": os.getenv("LANGSMITH_ENDPOINT") or None,
            "workspace_id_configured": bool(os.getenv("LANGSMITH_WORKSPACE_ID")),
        }
        self.run_id = uuid.UUID(hex=trace.trace_id)

    def start(self) -> None:
        if not self.config["enabled"]:
            return
        try:
            from langsmith import Client

            self.client = Client(
                api_key=_langsmith_api_key(),
                api_url=self.config["endpoint"],
                workspace_id=os.getenv("LANGSMITH_WORKSPACE_ID") or None,
                omit_traced_runtime_info=True,
            )
            self.client.create_run(
                id=self.run_id,
                project_name=self.config["project"],
                name=f"slm.{self.trace.operation}",
                run_type="chain",
                inputs={"run_id": self.trace.run_id, "operation": self.trace.operation},
                start_time=datetime.fromtimestamp(self.trace.start_ns / 1e9, tz=timezone.utc),
                extra={
                    "metadata": {
                        "w3c_trace_id": self.trace.trace_id,
                        "service": "slm-training",
                        **self.trace.attributes,
                    }
                },
            )
        except Exception as exc:  # noqa: BLE001 - observability must never stop a run
            self.client = None
            self._failed(exc)

    def summary(
        self,
        name: str,
        *,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        if self.client is None:
            return
        try:
            now = datetime.now(timezone.utc)
            self.client.create_run(
                id=uuid.uuid4(),
                trace_id=self.run_id,
                parent_run_id=self.run_id,
                project_name=self.config["project"],
                name=name,
                run_type="tool",
                inputs=inputs,
                outputs=outputs,
                start_time=now,
                end_time=now,
                extra={"metadata": {"w3c_trace_id": self.trace.trace_id, **metadata}},
            )
        except Exception as exc:  # noqa: BLE001 - observability must never stop a run
            self._failed(exc)

    def finish(self, error: BaseException | None) -> None:
        if self.client is None:
            return
        try:
            self.client.update_run(
                self.run_id,
                end_time=datetime.now(timezone.utc),
                outputs={"status": "failed" if error else "completed"},
                error="run failed; inspect local trace" if error else None,
            )
            self.client.flush(timeout=_langsmith_flush_seconds())
        except Exception as exc:  # noqa: BLE001 - observability must never stop a run
            self._failed(exc)

    def manifest(self) -> dict[str, Any]:
        return {**self.config, "trace_id": self.trace.trace_id, "last_export_error": self.error}

    def _failed(self, exc: Exception) -> None:
        self.error = type(exc).__name__
        warnings.warn(
            f"LangSmith export failed: {self.error}", RuntimeWarning, stacklevel=3
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
    if specific:
        return base
    normalized = base.rstrip("/")
    suffix = f"/v1/{signal}"
    return normalized if normalized.endswith(suffix) else f"{normalized}{suffix}"


def _headers() -> dict[str, str]:
    headers = _key_value_env("OTEL_EXPORTER_OTLP_HEADERS")
    if headers:
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
        self._langsmith = _LangSmithTrace(self)
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
        self._langsmith.start()
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
        self._langsmith.finish(exc)
        self._write_manifest()
        if self._token is not None:
            _CURRENT.reset(self._token)

    def record_summary(
        self,
        name: str,
        *,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        """Export a caller-curated aggregate only; never pass raw samples here."""
        self._langsmith.summary(name, inputs=inputs, outputs=outputs, metadata=metadata)

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
        configured = _key_value_env("OTEL_RESOURCE_ATTRIBUTES")
        return {
            "service.name": (
                os.getenv("OTEL_SERVICE_NAME")
                or configured.pop("service.name", None)
                or "slm-training"
            ),
            "service.namespace": configured.pop("service.namespace", "openui"),
            "service.version": configured.pop("service.version", "0.1.0"),
            "service.instance.id": self.instance_id,
            **configured,
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
            timeout = _otlp_timeout_seconds()
            with urllib.request.urlopen(request, timeout=timeout):  # noqa: S310
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
            "langsmith": self._langsmith.manifest(),
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
