from __future__ import annotations

import json
from pathlib import Path

from slm_training.runtime.telemetry import run_trace


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
