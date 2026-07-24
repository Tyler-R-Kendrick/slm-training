from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

import slm_training.runtime.telemetry.trace as trace_module
from slm_training.runtime.telemetry import run_trace


@pytest.fixture(autouse=True)
def _disable_real_langsmith_exports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Focused tests use fakes when exercising the remote publisher."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")


def test_run_trace_is_w3c_correlated_and_reused(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "demo"
    trace_root = tmp_path / "traces"
    with run_trace("demo", "train", run_dir=run_dir, trace_root=trace_root) as first:
        first.log("checkpoint.saved", attributes={"slm.step": 1})
        trace_id = first.trace_id
        assert len(trace_id) == 32
        assert first.traceparent.startswith(f"00-{trace_id}-")
    with run_trace("demo", "eval", run_dir=run_dir, trace_root=trace_root) as second:
        assert second.trace_id == trace_id
        assert second.parent_span_id

    bundle = trace_root / trace_id
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["run_ids"] == ["demo"]
    traces = list((bundle / "signals" / "traces").glob("*.jsonl"))
    logs = list((bundle / "signals" / "logs").glob("*.jsonl"))
    assert len(traces) == 2
    assert len(logs) == 2
    row = json.loads(traces[0].read_text().splitlines()[0])
    span = row["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert span["traceId"] == trace_id


def test_domain_trace_path_is_centralized(tmp_path: Path) -> None:
    with run_trace("build-v1", "data.build", trace_root=tmp_path) as trace:
        path = trace.domain_path("synthesis")
        path.write_text("{}\n")
        assert path == tmp_path / trace.trace_id / "domain" / "synthesis" / "records.jsonl"


def test_langsmith_summary_uses_the_w3c_trace_and_safe_payload(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[str, tuple, dict]] = []

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            calls.append(("client", (), kwargs))

        def create_run(self, **kwargs) -> None:
            calls.append(("create", (), kwargs))

        def update_run(self, *args, **kwargs) -> None:
            calls.append(("update", args, kwargs))

        def flush(self, **kwargs) -> None:
            calls.append(("flush", (), kwargs))

    monkeypatch.setitem(sys.modules, "langsmith", SimpleNamespace(Client=FakeClient))
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    run_dir = tmp_path / "runs" / "langsmith"
    with run_trace("langsmith", "eval", run_dir=run_dir) as trace:
        trace.record_summary(
            "evaluation.summary",
            inputs={"run_id": "langsmith", "suites": ["smoke"]},
            outputs={"suites": {"smoke": {"parse_rate": 1.0}}},
            metadata={"version_stamp": {"stamp_schema": "version_stamp/v1"}},
        )

    creates = [payload for kind, _, payload in calls if kind == "create"]
    assert len(creates) == 2
    root, child = creates
    assert root["id"] == uuid.UUID(hex=trace.trace_id)
    assert child["trace_id"] == root["id"]
    assert child["parent_run_id"] == root["id"]
    assert "prompt" not in json.dumps(child, default=str)
    manifest = json.loads((tmp_path / "traces" / trace.trace_id / "manifest.json").read_text())
    assert manifest["langsmith"]["enabled"] is True
    assert manifest["langsmith"]["project"] == "slm-training"
    assert ("flush", (), {"timeout": 2.0}) in calls


def test_langsmith_loads_the_repository_env_file(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict] = []

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)

        def create_run(self, **kwargs) -> None:
            pass

        def update_run(self, *args, **kwargs) -> None:
            pass

        def flush(self, **kwargs) -> None:
            pass

    env_path = tmp_path / ".env"
    env_path.write_text(
        "LANGSMITH_TRACING=true\n"
        "LANGSMITH_API_KEY=test-key\n"
        "LANGSMITH_PROJECT=env-project\n",
        encoding="utf-8",
    )
    for name in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(trace_module, "_ENV_PATH", env_path)
    monkeypatch.setitem(sys.modules, "langsmith", SimpleNamespace(Client=FakeClient))
    with run_trace("env", "eval", trace_root=tmp_path) as trace:
        pass
    assert calls[0]["api_key"] == "test-key"
    manifest = json.loads((tmp_path / trace.trace_id / "manifest.json").read_text())
    assert manifest["langsmith"]["project"] == "env-project"
    assert manifest["langsmith"]["api_key_configured"] is True


def test_langsmith_placeholder_is_inert(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "replace_with_rotated_langsmith_key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    with run_trace("placeholder", "eval", trace_root=tmp_path) as trace:
        pass
    manifest = json.loads((tmp_path / trace.trace_id / "manifest.json").read_text())
    assert manifest["langsmith"]["enabled"] is False
    assert manifest["langsmith"]["api_key_configured"] is False


def test_langsmith_export_failures_do_not_stop_local_tracing(tmp_path: Path, monkeypatch) -> None:
    class FailingClient:
        def __init__(self, **kwargs) -> None:
            pass

        def create_run(self, **kwargs) -> None:
            raise OSError("offline")

    monkeypatch.setitem(sys.modules, "langsmith", SimpleNamespace(Client=FailingClient))
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    with pytest.warns(RuntimeWarning, match="LangSmith export failed"):
        with run_trace("offline", "eval", trace_root=tmp_path) as trace:
            trace.log("still.local")
    assert (tmp_path / trace.trace_id / "signals" / "traces").is_dir()
    manifest = json.loads((tmp_path / trace.trace_id / "manifest.json").read_text())
    assert manifest["langsmith"]["last_export_error"] == "OSError"


def test_endpoint_precedence_includes_peer_fallback(monkeypatch) -> None:
    from slm_training.runtime.telemetry.trace import _endpoint

    for name in (
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "SLM_OTEL_PEERS",
    ):
        monkeypatch.delenv(name, raising=False)
    assert _endpoint("logs") is None

    monkeypatch.setenv("SLM_OTEL_PEERS", " , http://peer-a:8765/ ,http://peer-b")
    assert _endpoint("logs") == "http://peer-a:8765/v1/logs"

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    assert _endpoint("logs") == "http://collector:4318/v1/logs"

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://exact/ingest")
    assert _endpoint("logs") == "http://exact/ingest"


def test_otlp_standard_endpoint_and_resource_config_are_honored(
    tmp_path: Path, monkeypatch
) -> None:
    from slm_training.runtime.telemetry.trace import _endpoint, _otlp_timeout_seconds

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318/v1/logs/")
    assert _endpoint("logs") == "http://collector:4318/v1/logs"

    monkeypatch.setenv("OTEL_SERVICE_NAME", "slm-training-ci")
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        "service.namespace=ci,deployment.environment=test",
    )
    with run_trace("resources", "eval", trace_root=tmp_path) as trace:
        attributes = trace._resource_attributes()
    assert attributes["service.name"] == "slm-training-ci"
    assert attributes["service.namespace"] == "ci"
    assert attributes["deployment.environment"] == "test"

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "1000")
    assert _otlp_timeout_seconds() == 1.0
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "90000")
    assert _otlp_timeout_seconds() == 5.0


def test_mirror_headers_resolution(monkeypatch) -> None:
    from slm_training.runtime.telemetry.trace import _headers

    for name in (
        "OTEL_EXPORTER_OTLP_HEADERS",
        "SLM_OTEL_TOKEN",
        "SLM_OTEL_AUTH",
        "HF_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    assert _headers() == {}

    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    assert _headers() == {}  # HF_TOKEN is never forwarded without opt-in

    monkeypatch.setenv("SLM_OTEL_AUTH", "hf")
    assert _headers() == {"Authorization": "Bearer hf_secret"}

    monkeypatch.setenv("SLM_OTEL_TOKEN", "shared")
    assert _headers() == {"Authorization": "Bearer shared"}

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "x-api-key=abc, y = 1=2")
    assert _headers() == {"x-api-key": "abc", "y": "1=2"}
