"""Persist explainable loss and phase insights for one completed run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = 1
MAX_CHART_POINTS = 1000


class InsightCause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal["collapse", "data", "optimization", "unknown"]
    title: str = Field(min_length=1, max_length=160)
    rationale: str = Field(min_length=1, max_length=1200)
    evidence: tuple[str, ...] = Field(default=(), max_length=8)
    suggestion: str = Field(min_length=1, max_length=1200)
    confidence: float = Field(ge=0, le=1)
    event_step: int | None = Field(default=None, ge=0)


class PhaseSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: str = Field(min_length=1, max_length=120)
    suggestion: str = Field(min_length=1, max_length=1200)


class GeneratedRunInsights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=1600)
    causes: tuple[InsightCause, ...] = Field(default=(), max_length=8)
    phase_suggestions: tuple[PhaseSuggestion, ...] = Field(default=(), max_length=12)


class RunInsightSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: Literal["browser", "openai"]
    runtime: str = Field(min_length=1, max_length=160)
    generated: GeneratedRunInsights
    response_id: str | None = Field(default=None, max_length=200)
    model: str | None = Field(default=None, max_length=200)
    usage: dict[str, Any] = Field(default_factory=dict)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        pass
    return rows


def _file_sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _fingerprint(run_dir: Path, scoreboard: dict[str, Any] | None) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metrics": _file_sha(run_dir / "metrics.jsonl"),
        "telemetry": _file_sha(run_dir / "train_telemetry.json"),
        "matrix_result": _file_sha(run_dir / "matrix_result.json"),
        "scoreboard": scoreboard or {},
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _phase_help(name: str) -> str:
    key = name.casefold().replace("_", " ")
    if "denoiser" in key or key == "forward":
        return "Profile model forwards first; test AMP/compile, batching, or fewer decode steps, then rerun quality guardrails."
    if "backward" in key:
        return "Test mixed precision or gradient accumulation and confirm the effective batch still preserves convergence."
    if "optim" in key:
        return "Reduce optimizer overhead only after checking parameter count, update frequency, and device placement."
    if "batch" in key or "data" in key:
        return "Pre-batch or cache deterministic transforms and keep data-quality checks unchanged."
    if "dfa" in key or "grammar" in key:
        return "Reuse incremental grammar state and cached legal-token sets before weakening any grammar checks."
    if "stream" in key or "verify" in key:
        return "Prefer incremental or chosen-token verification; keep final validation and parse guardrails enabled."
    if "sync" in key:
        return "Remove unnecessary host/device synchronization boundaries and measure again on the same device."
    if "save" in key or "checkpoint" in key:
        return "Measure storage throughput and reduce checkpoint frequency only if recovery requirements still hold."
    return "Profile this span in isolation and change one lever at a time, starting with the largest share of cycle time."


def _phases(
    telemetry: dict[str, Any] | None,
    matrix_result: dict[str, Any] | None,
    scoreboard: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    spans = (telemetry or {}).get("spans")
    if isinstance(spans, dict):
        for name, raw in spans.items():
            value = raw if isinstance(raw, dict) else {}
            rows.append(
                {
                    "name": str(name),
                    "label": str(name).replace("_", " "),
                    "value": float(value.get("pct") or 0),
                    "pct": float(value.get("pct") or 0),
                    "mean_ms": value.get("mean_ms"),
                    "total_ms": value.get("total_ms"),
                    "help": _phase_help(str(name)),
                }
            )
    if not rows:
        summary = (matrix_result or {}).get("phase_summary") or (scoreboard or {}).get(
            "phase_summary"
        )
        if isinstance(summary, dict):
            for key, value in summary.items():
                if not key.endswith("_ms_mean") or not isinstance(value, (int, float)):
                    continue
                name = key.removesuffix("_ms_mean")
                rows.append(
                    {
                        "name": name,
                        "label": name.replace("_", " "),
                        "value": float(value),
                        "mean_ms": float(value),
                        "pct": None,
                        "help": _phase_help(name),
                    }
                )
    rows.sort(key=lambda row: float(row.get("value") or 0), reverse=True)
    if rows:
        rows[0]["dominant"] = True
    return rows[:12]


def _loss_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        step = row.get("step", index)
        loss = row.get("loss")
        if not isinstance(step, (int, float)) or not isinstance(loss, (int, float)):
            continue
        numeric = float(loss)
        points.append(
            {
                "step": int(step),
                "loss": numeric if math.isfinite(numeric) else None,
                "raw_loss": numeric,
            }
        )
    points.sort(key=lambda row: row["step"])
    return points


def _event(
    kind: str,
    point: dict[str, Any],
    severity: str,
    finding: str,
    suggestion: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "step": point["step"],
        "loss": point["loss"],
        "severity": severity,
        "finding": finding,
        "suggestion": suggestion,
    }


def _collapse_events(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    finite: list[dict[str, Any]] = []
    for point in points:
        value = point["raw_loss"]
        if not math.isfinite(value):
            events.append(
                _event(
                    "non_finite",
                    point,
                    "critical",
                    "Loss became non-finite, which is a direct numerical-collapse signal.",
                    "Check learning rate, mixed-precision overflow, invalid batches, and gradient norms before rerunning the same recipe.",
                )
            )
            continue
        previous = finite[-8:]
        if len(previous) >= 5:
            values = [row["raw_loss"] for row in previous]
            median = statistics.median(values)
            mad = statistics.median(abs(item - median) for item in values)
            high_delta = max(6 * mad, 0.5 * abs(median), 1e-12)
            low_delta = max(6 * mad, 0.25 * abs(median), 1e-12)
            if value > median + high_delta:
                events.append(
                    _event(
                        "spike",
                        point,
                        "critical",
                        f"Loss spiked above its rolling baseline ({median:.4g}).",
                        "Compare the batch and optimizer state at this step; likely hypotheses include an excessive learning rate, unstable precision, or an outlier batch.",
                    )
                )
            elif median > 0 and value < 0.25 * median and median - value > low_delta:
                events.append(
                    _event(
                        "suspicious_drop",
                        point,
                        "warning",
                        f"Loss dropped abruptly below one quarter of its rolling baseline ({median:.4g}).",
                        "Verify target-token counts, masking, duplicate exposure, and leakage before treating the drop as genuine progress.",
                    )
                )
        window = [*finite[-4:], point]
        if (
            len(window) == 5
            and all(window[i + 1]["raw_loss"] > window[i]["raw_loss"] for i in range(4))
            and value >= 1.5 * min(row["raw_loss"] for row in window)
        ):
            events.append(
                _event(
                    "divergence",
                    point,
                    "critical",
                    "Loss rose for five consecutive recorded steps and ended at least 50% above the window minimum.",
                    "Test a lower learning rate or larger effective batch against the identical data snapshot and inspect gradients for the first rising step.",
                )
            )
        finite.append(point)

    # Repeated adjacent signals describe one episode; retain the latest marker.
    coalesced: list[dict[str, Any]] = []
    for event in events:
        if (
            coalesced
            and coalesced[-1]["kind"] == event["kind"]
            and event["step"] <= coalesced[-1]["step"] + 1
        ):
            coalesced[-1] = event
        else:
            coalesced.append(event)
    return coalesced


def _sample_points(
    points: list[dict[str, Any]], events: list[dict[str, Any]], limit: int = MAX_CHART_POINTS
) -> list[dict[str, Any]]:
    clean = [{"step": row["step"], "loss": row["loss"]} for row in points]
    if len(clean) <= limit:
        return clean
    event_steps = {event["step"] for event in events}
    kept = {0, len(clean) - 1}
    kept.update(index for index, row in enumerate(clean) if row["step"] in event_steps)
    budget = max(2, limit - len(kept))
    bucket_count = max(1, budget // 2)
    bucket_size = math.ceil(len(clean) / bucket_count)
    for start in range(0, len(clean), bucket_size):
        indexes = [
            index
            for index in range(start, min(len(clean), start + bucket_size))
            if clean[index]["loss"] is not None
        ]
        if not indexes:
            continue
        kept.add(min(indexes, key=lambda index: clean[index]["loss"]))
        kept.add(max(indexes, key=lambda index: clean[index]["loss"]))
    selected = sorted(kept)
    if len(selected) > limit:
        event_indexes = [index for index in selected if clean[index]["step"] in event_steps]
        others = [index for index in selected if index not in set(event_indexes)]
        stride = max(1, math.ceil(len(others) / max(1, limit - len(event_indexes))))
        selected = sorted((event_indexes + others[::stride])[:limit])
    return [clean[index] for index in selected]


def build_run_insights(
    run_dir: Path | str,
    *,
    run_id: str,
    scoreboard: dict[str, Any] | None = None,
    cached: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    fingerprint = _fingerprint(run_dir, scoreboard)
    if cached and cached.get("source_fingerprint") == fingerprint:
        return cached

    matrix_result = _read_json(run_dir / "matrix_result.json")
    telemetry = _read_json(run_dir / "train_telemetry.json")
    points = _loss_points(_read_jsonl(run_dir / "metrics.jsonl"))
    events = _collapse_events(points)
    phases = _phases(telemetry, matrix_result, scoreboard)
    insights = [
        {
            "category": "collapse" if event["severity"] == "critical" else "data",
            "finding": event["finding"],
            "suggestion": event["suggestion"],
            "confidence": 0.9 if event["kind"] == "non_finite" else 0.75,
            "step": event["step"],
            "source": "deterministic",
        }
        for event in events
    ]
    if phases:
        dominant = phases[0]
        insights.append(
            {
                "category": "optimization",
                "finding": f"{dominant['label']} is the largest recorded phase ({dominant['value']:.3g}{'%' if dominant.get('pct') is not None else ' ms mean'}).",
                "suggestion": dominant["help"],
                "confidence": 0.8,
                "source": "deterministic",
            }
        )
    critical = sum(event["severity"] == "critical" for event in events)
    status = "collapsed" if critical else "warning" if events else "healthy" if points else "unavailable"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "source_fingerprint": fingerprint,
        "generated_at": _utc_now(),
        "loss": {
            "status": status,
            "point_count": len(points),
            "points": _sample_points(points, events),
            "events": events,
        },
        "phases": phases,
        "insights": insights,
        "enrichment": None,
        "persistence": {"persisted": False, "path": str(run_dir / "run_insights.json")},
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_run_insights(
    run_dir: Path | str,
    *,
    run_id: str,
    scoreboard: dict[str, Any] | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    path = run_dir / "run_insights.json"
    report = build_run_insights(
        run_dir,
        run_id=run_id,
        scoreboard=scoreboard,
        cached=_read_json(path),
    )
    report["persistence"] = {"persisted": False, "path": str(path)}
    if persist:
        try:
            report["persistence"]["persisted"] = True
            _atomic_json(path, report)
        except OSError:
            report["persistence"]["persisted"] = False
    return report


def save_enrichment(
    run_dir: Path | str,
    *,
    run_id: str,
    submission: RunInsightSubmission,
    scoreboard: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    report = load_run_insights(run_dir, run_id=run_id, scoreboard=scoreboard)
    if submission.source_fingerprint != report["source_fingerprint"]:
        raise ValueError("run evidence changed; regenerate insights from the current report")
    report["enrichment"] = {
        **submission.model_dump(mode="json"),
        "generated_at": _utc_now(),
    }
    report["persistence"] = {
        "persisted": True,
        "path": str(run_dir / "run_insights.json"),
    }
    _atomic_json(run_dir / "run_insights.json", report)
    return report


def _dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {}
    return value if isinstance(value, dict) else {}


def enrich_with_openai(
    report: dict[str, Any], *, client: Any | None = None, model: str | None = None
) -> RunInsightSubmission:
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install slm-training[research] for OpenAI fallback") from exc
        client = OpenAI()
    model = model or os.getenv("OPENUI_INSIGHTS_MODEL", "gpt-5.6-sol")
    evidence = {
        "run_id": report.get("run_id"),
        "loss": report.get("loss"),
        "phases": report.get("phases"),
        "deterministic_insights": report.get("insights"),
    }
    response = client.responses.parse(
        model=model,
        store=False,
        input=[
            {
                "role": "system",
                "content": (
                    "Explain the supplied OpenUI training evidence. Treat deterministic "
                    "events as authoritative observations, label causes as hypotheses, cite "
                    "only supplied evidence fields, and suggest bounded experiments rather "
                    "than commands or gate changes."
                ),
            },
            {"role": "user", "content": json.dumps(evidence, sort_keys=True)},
        ],
        text_format=GeneratedRunInsights,
    )
    generated = response.output_parsed
    if not isinstance(generated, GeneratedRunInsights):
        generated = GeneratedRunInsights.model_validate(generated)
    return RunInsightSubmission(
        source_fingerprint=report["source_fingerprint"],
        provider="openai",
        runtime="openai-responses",
        generated=generated,
        response_id=getattr(response, "id", None),
        model=getattr(response, "model", None) or model,
        usage=_dump(getattr(response, "usage", None)),
    )


__all__ = [
    "GeneratedRunInsights",
    "RunInsightSubmission",
    "build_run_insights",
    "enrich_with_openai",
    "load_run_insights",
    "save_enrichment",
]
