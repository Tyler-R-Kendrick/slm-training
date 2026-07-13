"""Per-phase decode latency instrumentation for generate / LTR paths."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator


@dataclass
class DecodeStats:
    """Accumulated wall-clock timings and counters for one generate call."""

    denoiser_ms: float = 0.0
    dfa_sync_ms: float = 0.0
    stream_check_ms: float = 0.0
    detok_ms: float = 0.0
    context_ms: float = 0.0
    finalize_ms: float = 0.0
    pick_ms: float = 0.0
    total_ms: float = 0.0
    forwards_count: int = 0
    probes_count: int = 0
    dfa_sync_count: int = 0
    tokens_emitted: int = 0
    attempts: int = 1
    accepted_run_tokens: int = 0  # P3 multi-token accepts beyond the first
    canvas_tokens: int = 0

    def add_ms(self, field_name: str, ms: float) -> None:
        setattr(self, field_name, float(getattr(self, field_name)) + float(ms))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def merge(self, other: "DecodeStats") -> None:
        for key, value in other.as_dict().items():
            if key == "attempts":
                self.attempts = max(self.attempts, int(value))
                continue
            cur = getattr(self, key)
            if isinstance(cur, (int, float)) and isinstance(value, (int, float)):
                setattr(self, key, cur + value)


@contextmanager
def timed_ms(stats: DecodeStats | None, field_name: str) -> Iterator[None]:
    """Accumulate wall time into ``stats.<field_name>`` when stats is set."""
    if stats is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        stats.add_ms(field_name, (time.perf_counter() - t0) * 1000.0)


# Thread-local-ish active stats for helpers that cannot take an explicit arg.
_ACTIVE: DecodeStats | None = None


def get_active_stats() -> DecodeStats | None:
    return _ACTIVE


def set_active_stats(stats: DecodeStats | None) -> DecodeStats | None:
    global _ACTIVE
    prev = _ACTIVE
    _ACTIVE = stats
    return prev


@contextmanager
def collect_decode_stats(stats: DecodeStats | None = None) -> Iterator[DecodeStats]:
    """Activate a DecodeStats collector for nested grammar/decode helpers."""
    bucket = stats if stats is not None else DecodeStats()
    prev = set_active_stats(bucket)
    t0 = time.perf_counter()
    try:
        yield bucket
    finally:
        bucket.total_ms += (time.perf_counter() - t0) * 1000.0
        set_active_stats(prev)


def aggregate_stats(rows: list[DecodeStats]) -> dict[str, Any]:
    """Mean / sum summary across multiple generate calls."""
    if not rows:
        return {}
    keys = [
        "denoiser_ms",
        "dfa_sync_ms",
        "stream_check_ms",
        "detok_ms",
        "context_ms",
        "finalize_ms",
        "pick_ms",
        "total_ms",
        "forwards_count",
        "probes_count",
        "dfa_sync_count",
        "tokens_emitted",
        "accepted_run_tokens",
        "canvas_tokens",
    ]
    out: dict[str, Any] = {"n": len(rows)}
    for key in keys:
        vals = [float(getattr(r, key)) for r in rows]
        out[f"{key}_sum"] = round(sum(vals), 3)
        out[f"{key}_mean"] = round(sum(vals) / len(vals), 3)
    totals = sorted(float(r.total_ms) for r in rows)
    mid = len(totals) // 2
    out["total_ms_p50"] = totals[mid] if totals else None
    out["total_ms_p95"] = totals[max(0, int(len(totals) * 0.95) - 1)] if totals else None
    return out


__all__ = [
    "DecodeStats",
    "aggregate_stats",
    "collect_decode_stats",
    "get_active_stats",
    "set_active_stats",
    "timed_ms",
]
