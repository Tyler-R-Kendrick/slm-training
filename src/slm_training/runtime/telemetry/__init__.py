"""Lightweight cycle telemetry for train / inference bottleneck analysis."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from slm_training.runtime.telemetry.trace import RunTrace, current_trace, run_trace


@dataclass
class SpanStats:
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    samples_ns: list[int] = field(default_factory=list)

    def add(self, ms: float) -> None:
        value = float(ms)
        if not math.isfinite(value) or value < 0:
            raise ValueError("span duration must be finite and non-negative")
        self.add_ns(round(value * 1_000_000))

    def add_ns(self, ns: int) -> None:
        if ns < 0:
            raise ValueError("span duration must be non-negative")
        ms = ns / 1_000_000
        self.count += 1
        self.samples_ns.append(ns)
        self.total_ms += ms
        if ms > self.max_ms:
            self.max_ms = ms

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "samples_ns": list(self.samples_ns),
            "pct": None,  # filled by CycleTelemetry.summary
        }


@dataclass
class CycleTelemetry:
    """Accumulate named spans; emit JSON summaries for bottleneck ranking."""

    enabled: bool = True
    spans: dict[str, SpanStats] = field(default_factory=lambda: defaultdict(SpanStats))
    meta: dict[str, Any] = field(default_factory=dict)

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self.spans[name].add_ns(time.perf_counter_ns() - t0)

    def record_ms(self, name: str, ms: float) -> None:
        if self.enabled:
            self.spans[name].add(float(ms))

    def summary(self) -> dict[str, Any]:
        total = sum(s.total_ms for s in self.spans.values()) or 1.0
        by_name: dict[str, Any] = {}
        ranked: list[tuple[str, float]] = []
        for name, stats in self.spans.items():
            row = stats.to_dict()
            pct = 100.0 * stats.total_ms / total
            row["pct"] = round(pct, 2)
            by_name[name] = row
            ranked.append((name, pct))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return {
            "meta": dict(self.meta),
            "total_ms": round(sum(s.total_ms for s in self.spans.values()), 3),
            "spans": by_name,
            "bottlenecks": [{"name": n, "pct": round(p, 2)} for n, p in ranked[:8]],
        }

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2) + "\n", encoding="utf-8")
        return path

    def reset(self) -> None:
        self.spans = defaultdict(SpanStats)


# Process-global optional probe (train_loop / generate can attach).
_ACTIVE: CycleTelemetry | None = None


def get_telemetry() -> CycleTelemetry | None:
    return _ACTIVE


def set_telemetry(tel: CycleTelemetry | None) -> CycleTelemetry | None:
    global _ACTIVE
    prev = _ACTIVE
    _ACTIVE = tel
    return prev


@contextmanager
def bind_telemetry(tel: CycleTelemetry) -> Iterator[CycleTelemetry]:
    prev = set_telemetry(tel)
    try:
        yield tel
    finally:
        set_telemetry(prev)


@contextmanager
def timed(name: str) -> Iterator[None]:
    tel = get_telemetry()
    if tel is None:
        yield
        return
    with tel.span(name):
        yield


__all__ = [
    "CycleTelemetry",
    "RunTrace",
    "bind_telemetry",
    "current_trace",
    "get_telemetry",
    "run_trace",
    "set_telemetry",
    "timed",
]
