"""Throttled OTLP train.progress emission from the training loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.train_loop import train
from slm_training.runtime.telemetry import run_trace

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)
CTA = 'root = Stack([cta])\ncta = Button(":cta.label")'


@pytest.fixture()
def train_dir(tmp_path: Path) -> Path:
    out = tmp_path / "train_data"
    out.mkdir(parents=True)
    write_jsonl(
        out / "records.jsonl",
        [
            ExampleRecord(
                id="a",
                prompt="Hero",
                openui=HERO,
                split="train",
                placeholders=[":hero.title", ":hero.body"],
            ),
            ExampleRecord(
                id="b",
                prompt="CTA",
                openui=CTA,
                split="train",
                placeholders=[":cta.label"],
            ),
        ],
    )
    return out


def _cfg(train_dir: Path, tmp_path: Path, run_id: str) -> ModelBuildConfig:
    return ModelBuildConfig(
        train_dir=train_dir,
        run_root=tmp_path / "runs",
        run_id=run_id,
        steps=4,
        batch_size=2,
        lr=3e-3,
        seed=0,
        model_name="twotower",
        d_model=32,
        n_heads=4,
        context_layers=1,
        denoiser_layers=1,
        context_backend="scratch",
        freeze_context=False,
        telemetry=False,
    )


def _log_bodies(trace_root: Path, trace_id: str) -> list[dict]:
    records: list[dict] = []
    for shard in (trace_root / trace_id / "signals" / "logs").glob("*.jsonl"):
        for line in shard.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            for resource in payload.get("resourceLogs", []):
                for scope in resource.get("scopeLogs", []):
                    records.extend(scope.get("logRecords", []))
    return records


def _attrs(record: dict) -> dict:
    out = {}
    for row in record.get("attributes", []):
        value = row.get("value", {})
        out[row["key"]] = next(iter(value.values()), None)
    return out


def test_progress_heartbeat_emitted_inside_run_trace(
    train_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
                 "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "SLM_OTEL_PEERS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLM_OTEL_PROGRESS_SECONDS", "0.0001")
    config = _cfg(train_dir, tmp_path, "progress_on")
    with run_trace("progress_on", "train", run_dir=config.run_dir) as trace:
        summary = train(config)
    assert summary["steps"] == 4
    progress = [
        record
        for record in _log_bodies(tmp_path / "traces", trace.trace_id)
        if record.get("body", {}).get("stringValue") == "train.progress"
    ]
    assert progress, "expected throttled train.progress heartbeats"
    attrs = _attrs(progress[-1])
    assert attrs["slm.run.id"] == "progress_on"
    assert int(attrs["slm.step"]) >= 1
    assert int(attrs["slm.steps.total"]) == 4
    assert "slm.loss" in attrs


def test_progress_heartbeat_disabled_by_env_and_without_trace(
    train_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLM_OTEL_PROGRESS_SECONDS", "0")
    config = _cfg(train_dir, tmp_path, "progress_off")
    with run_trace("progress_off", "train", run_dir=config.run_dir) as trace:
        train(config)
    bodies = {
        record.get("body", {}).get("stringValue")
        for record in _log_bodies(tmp_path / "traces", trace.trace_id)
    }
    assert "run.started" in bodies and "run.completed" in bodies
    assert "train.progress" not in bodies

    # No surrounding run_trace: the emitter is a silent no-op.
    monkeypatch.setenv("SLM_OTEL_PROGRESS_SECONDS", "0.0001")
    summary = train(_cfg(train_dir, tmp_path, "progress_untraced"))
    assert summary["steps"] == 4
