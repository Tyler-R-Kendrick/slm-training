"""Honest end-to-end cold/warm benchmark harness (PCT-003).

Extends the existing decode-telemetry measurement-stage taxonomy
(``MEASUREMENT_STAGES``/``COMPLETENESS_STATES`` in ``models/decode_stats.py``)
rather than inventing a second one. A cold trial always spawns a real
subprocess -- a fresh Python process, not simulated by clearing caches
in-process -- and reads back newline-delimited JSON phase markers it
emits; a warm trial runs an in-process callable and is labeled
``steady_state`` so it can never blend into cold numbers. A trial that
times out or whose subprocess exits non-zero is recorded as
``partial_timeout``/``aborted`` and is excluded from any "complete"
summary -- it is never silently treated as a finished measurement.
"""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

from slm_training.models.decode_stats import COMPLETENESS_STATES, MEASUREMENT_STAGES

BENCH_TRIAL_SCHEMA = "cold_warm_bench_trial/v1"

TrialKind = str  # "cold" | "warm"
_TRIAL_KINDS = frozenset({"cold", "warm"})

# One JSON object per line on stdout: {"phase": "<name>", "t": <perf_counter_seconds>}.
_PHASE_LINE_PREFIX = "PCT003_PHASE "


def phase_marker(phase: str) -> str:
    """Emit inside a benchmarked subprocess: ``print(phase_marker("model_load"))``."""
    return f"{_PHASE_LINE_PREFIX}{json.dumps({'phase': phase, 't': time.perf_counter()})}"


def _parse_phase_lines(stdout: str) -> tuple[dict[str, Any], ...]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.startswith(_PHASE_LINE_PREFIX):
            continue
        events.append(json.loads(line[len(_PHASE_LINE_PREFIX) :]))
    return tuple(events)


def _phase_durations(events: Sequence[dict[str, Any]]) -> tuple["PhaseTiming", ...]:
    """Consecutive-marker deltas -- the first marker's duration is time-since-launch."""
    timings: list[PhaseTiming] = []
    previous_t: float | None = None
    for event in events:
        t = float(event["t"])
        duration_ms = (t - previous_t) * 1000.0 if previous_t is not None else None
        timings.append(PhaseTiming(phase=str(event["phase"]), duration_ms=duration_ms))
        previous_t = t
    return tuple(timings)


def _config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True)
class HardwareIdentityV1:
    """Environment identity persisted with every trial (AC: identity persisted)."""

    hostname: str
    platform: str
    python_version: str
    cpu_count: int | None
    machine: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "hostname": self.hostname,
            "platform": self.platform,
            "python_version": self.python_version,
            "cpu_count": self.cpu_count,
            "machine": self.machine,
        }

    @classmethod
    def capture(cls) -> "HardwareIdentityV1":
        import os

        return cls(
            hostname=platform.node(),
            platform=platform.platform(),
            python_version=sys.version.split()[0],
            cpu_count=os.cpu_count(),
            machine=platform.machine(),
        )


@dataclass(frozen=True)
class PhaseTiming:
    phase: str
    duration_ms: float | None

    def to_dict(self) -> dict[str, Any]:
        return {"phase": self.phase, "duration_ms": self.duration_ms}


def _digest(bench: "BenchTrialV1") -> str:
    payload = bench.to_dict()
    payload.pop("trial_hash", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True)
class BenchTrialV1:
    """One immutable, tamper-evident cold or warm benchmark trial."""

    schema_version: str
    trial_kind: TrialKind
    measurement_stage: str
    completeness: str
    config_hash: str
    hardware_identity: dict[str, Any]
    phases: tuple[PhaseTiming, ...]
    total_ms: float | None
    peak_rss_kb: int | None
    exit_code: int | None
    trial_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trial_kind": self.trial_kind,
            "measurement_stage": self.measurement_stage,
            "completeness": self.completeness,
            "config_hash": self.config_hash,
            "hardware_identity": dict(self.hardware_identity),
            "phases": [p.to_dict() for p in self.phases],
            "total_ms": self.total_ms,
            "peak_rss_kb": self.peak_rss_kb,
            "exit_code": self.exit_code,
            "trial_hash": self.trial_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchTrialV1":
        if data.get("schema_version") != BENCH_TRIAL_SCHEMA:
            raise ValueError(
                f"unsupported cold/warm bench trial schema: {data.get('schema_version')!r}"
            )
        trial = cls(
            schema_version=data.get("schema_version", BENCH_TRIAL_SCHEMA),
            trial_kind=data.get("trial_kind", ""),
            measurement_stage=data.get("measurement_stage", "process_cold"),
            completeness=data.get("completeness", "aborted"),
            config_hash=data.get("config_hash", ""),
            hardware_identity=dict(data.get("hardware_identity", {})),
            phases=tuple(
                PhaseTiming(phase=p["phase"], duration_ms=p.get("duration_ms"))
                for p in data.get("phases", [])
            ),
            total_ms=data.get("total_ms"),
            peak_rss_kb=data.get("peak_rss_kb"),
            exit_code=data.get("exit_code"),
            trial_hash=data.get("trial_hash", ""),
        )
        if _digest(trial) != trial.trial_hash:
            raise ValueError(
                "cold/warm bench trial hash mismatch (tampered or corrupted payload)"
            )
        return trial


def _finalize(
    *,
    trial_kind: TrialKind,
    measurement_stage: str,
    completeness: str,
    config: dict[str, Any],
    phases: tuple[PhaseTiming, ...],
    total_ms: float | None,
    peak_rss_kb: int | None,
    exit_code: int | None,
) -> BenchTrialV1:
    if trial_kind not in _TRIAL_KINDS:
        raise ValueError(f"unknown trial_kind: {trial_kind!r}")
    if measurement_stage not in MEASUREMENT_STAGES:
        raise ValueError(f"unknown measurement_stage: {measurement_stage!r}")
    if completeness not in COMPLETENESS_STATES:
        raise ValueError(f"unknown completeness: {completeness!r}")
    trial = BenchTrialV1(
        schema_version=BENCH_TRIAL_SCHEMA,
        trial_kind=trial_kind,
        measurement_stage=measurement_stage,
        completeness=completeness,
        config_hash=_config_hash(config),
        hardware_identity=HardwareIdentityV1.capture().to_dict(),
        phases=phases,
        total_ms=total_ms,
        peak_rss_kb=peak_rss_kb,
        exit_code=exit_code,
        trial_hash="",
    )
    return replace(trial, trial_hash=_digest(trial))


def run_cold_trial(
    command: Sequence[str],
    *,
    config: dict[str, Any],
    timeout_s: float = 60.0,
) -> BenchTrialV1:
    """Spawn a real fresh subprocess and read back its emitted phase markers.

    Never simulates a cold start by clearing caches in this process -- the
    child process is a genuine fresh interpreter. A timeout or non-zero exit
    marks the trial ``partial_timeout``/``aborted`` rather than ``complete``;
    the recorded phases (if any) are kept, never fabricated or discarded.
    """
    rusage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    launch_t0 = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        events = _parse_phase_lines(stdout) if isinstance(stdout, str) else ()
        return _finalize(
            trial_kind="cold",
            measurement_stage="process_cold",
            completeness="partial_timeout",
            config=config,
            phases=_phase_durations(events),
            total_ms=None,
            peak_rss_kb=None,
            exit_code=None,
        )
    total_ms = (time.perf_counter() - launch_t0) * 1000.0
    rusage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak_rss_kb = max(0, rusage_after.ru_maxrss - rusage_before.ru_maxrss) or (
        rusage_after.ru_maxrss or None
    )
    events = _parse_phase_lines(completed.stdout)
    completeness = "complete" if completed.returncode == 0 else "aborted"
    return _finalize(
        trial_kind="cold",
        measurement_stage="process_cold",
        completeness=completeness,
        config=config,
        phases=_phase_durations(events),
        total_ms=total_ms,
        peak_rss_kb=peak_rss_kb,
        exit_code=completed.returncode,
    )


def run_warm_trial(
    fn: Callable[[], tuple[dict[str, Any], ...]],
    *,
    config: dict[str, Any],
) -> BenchTrialV1:
    """Run one in-process steady-state trial; ``fn`` returns its own phase events.

    Labeled ``steady_state``, structurally distinct from any cold trial's
    ``process_cold`` stage -- the two can never be blended by a downstream
    summary that only accepts a single ``measurement_stage``.
    """
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    try:
        events = fn()
    except Exception:  # noqa: BLE001 - a raising warm callable is an aborted trial
        return _finalize(
            trial_kind="warm",
            measurement_stage="steady_state",
            completeness="aborted",
            config=config,
            phases=(),
            total_ms=None,
            peak_rss_kb=None,
            exit_code=None,
        )
    total_ms = (time.perf_counter() - t0) * 1000.0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return _finalize(
        trial_kind="warm",
        measurement_stage="steady_state",
        completeness="complete",
        config=config,
        phases=_phase_durations(events),
        total_ms=total_ms,
        peak_rss_kb=max(0, rss_after - rss_before) or rss_after or None,
        exit_code=0,
    )


def trial_integrity_ok(trial: BenchTrialV1) -> bool:
    """Recompute the trial hash and compare; used by replay/tamper checks."""
    return _digest(trial) == trial.trial_hash


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100.0 * (len(ordered) - 1))))
    return ordered[idx]


def summarize_trials(trials: Sequence[BenchTrialV1]) -> dict[str, Any]:
    """p50/p95 total_ms per (trial_kind, config_hash), complete trials only.

    A control arm and an artifact-enabled arm land in separate summary keys
    only when they share an identical ``config_hash`` -- mismatched configs
    are reported, never silently averaged together (AC: matched requests/
    configs). Incomplete trials never enter a percentile.
    """
    groups: dict[tuple[str, str], list[float]] = {}
    excluded = 0
    for trial in trials:
        if trial.completeness != "complete" or trial.total_ms is None:
            excluded += 1
            continue
        groups.setdefault((trial.trial_kind, trial.config_hash), []).append(
            trial.total_ms
        )
    summary: dict[str, Any] = {"excluded_incomplete": excluded, "arms": {}}
    for (kind, config_hash), values in groups.items():
        summary["arms"][f"{kind}:{config_hash[:12]}"] = {
            "trial_kind": kind,
            "config_hash": config_hash,
            "n": len(values),
            "p50_ms": round(_percentile(values, 50), 3),
            "p95_ms": round(_percentile(values, 95), 3),
        }
    return summary
