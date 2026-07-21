"""SLM-224 (NCS0-04): SemanticFloorGateV1 statistical-power sweep.

SLM-223 calibrated ``SemanticFloorGateV1`` on the default 4-run / 2-family
synthetic fixture and got a ``no_signal`` disposition (LOFO balanced accuracy
0.500 == permutation-null mean 0.500). SLM-223's own honest caveats flagged
the likely cause: "the tiny fixture size (2-4 runs per fold) limits
statistical power". This harness asks whether that caveat is right.

It re-runs the *unmodified* SLM-223 gate pipeline
(``run_semantic_floor_gate_fixture``) at increasing ``synthetic_runs`` values
against the *same* SLM-215 synthetic generator (which bakes in a real,
per-matrix ``parse_rate = 0.4 + 0.1 * alpha_z + noise`` relationship — see
``slm215_spectral_atlas._synthetic_fixture_rows``) and records whether the
leave-one-family-out margin over the permutation-null baseline ever clears
the same 0.15 signal margin SLM-223 used.

No new statistical machinery is introduced: this harness only sweeps an
existing, already-parameterized fixture size and aggregates existing
per-point ``SemanticFloorGateReport`` objects. No model is trained, no GPU is
required, and no ship-gate or promotion claim is made.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm223_semantic_floor_gate import (
    DEFAULT_FLOOR_THRESHOLD,
    SIGNAL_MARGIN as _SLM223_SIGNAL_MARGIN,
    run_semantic_floor_gate_fixture,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "DEFAULT_SWEEP_GRID",
    "SIGNAL_MARGIN",
    "PowerSweepPoint",
    "PowerSweepReport",
    "render_markdown",
    "run_power_sweep_fixture",
]

MATRIX_VERSION = "ncs0-04-v1"
MATRIX_SET = "slm224_floor_gate_power_sweep"
EXPERIMENT_ID = "slm224-floor-gate-power-sweep"

SIGNAL_MARGIN = _SLM223_SIGNAL_MARGIN
DEFAULT_SWEEP_GRID: tuple[int, ...] = (4, 8, 16, 32, 64, 128)

_HYPOTHESIS = (
    "SLM-223's SemanticFloorGateV1 'no_signal' disposition on the default "
    "4-run/2-family fixture is a statistical-power artifact, not evidence "
    "that the underlying alpha_z-vs-parse_rate relationship fails to "
    "generalize across families: sweeping the same SLM-215 synthetic "
    "generator (which bakes in a real per-matrix parse_rate = 0.4 + "
    "0.1*alpha_z + noise relationship) to larger synthetic_runs values will "
    "produce a leave-one-family-out balanced accuracy that clears the "
    "permutation-null mean by the required 0.15 margin at some larger "
    "sample size."
)

_FALSIFIER = (
    "The LOFO-vs-permutation-null margin stays below 0.15 across the full "
    f"swept grid up to {DEFAULT_SWEEP_GRID[-1]} synthetic runs, i.e. more "
    "fixture size within the swept range never recovers signal -- "
    "suggesting SLM-223's no-signal result was not (only) a power problem."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: no trained model, checkpoint promotion, or GPU run.",
    "Reuses the unmodified SLM-223 gate pipeline and the unmodified SLM-215 synthetic "
    "generator at each grid point; no new calibration or statistical method is added.",
    "The synthetic generator always splits runs into exactly 2 families "
    "(run_idx % 2), so scaling synthetic_runs increases runs-per-family, not "
    "the number of families; this sweep cannot speak to whether more *families* "
    "would help, only whether more *runs per family* helps.",
    "The synthetic per-matrix signal (0.1 * alpha_z coefficient vs 0.05 noise sd) is "
    "a fixture design choice, not a measurement from a real checkpoint; a positive "
    "result here shows the gate mechanism *can* detect a real embedded signal given "
    "enough samples, not that any actual checkpoint atlas has this signal strength.",
    "No causal conclusion is drawn and no promotion or ship-gate claim is made; this "
    "is a diagnostic follow-up to SLM-223's own honest caveats.",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(value: Any) -> str:
    return _sha256(_canonical_json(value))


@dataclass(frozen=True)
class PowerSweepPoint:
    """One sweep grid point: a full SLM-223 gate run at a given fixture size."""

    synthetic_runs: int
    n_runs: int
    n_families: int
    real_balanced_accuracy: float | None
    permutation_null_mean: float | None
    margin: float | None
    disposition: str
    gate_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "synthetic_runs": self.synthetic_runs,
            "n_runs": self.n_runs,
            "n_families": self.n_families,
            "real_balanced_accuracy": self.real_balanced_accuracy,
            "permutation_null_mean": self.permutation_null_mean,
            "margin": self.margin,
            "disposition": self.disposition,
            "gate_hash": self.gate_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PowerSweepPoint":
        return cls(
            synthetic_runs=int(data.get("synthetic_runs", 0)),
            n_runs=int(data.get("n_runs", 0)),
            n_families=int(data.get("n_families", 0)),
            real_balanced_accuracy=data.get("real_balanced_accuracy"),
            permutation_null_mean=data.get("permutation_null_mean"),
            margin=data.get("margin"),
            disposition=str(data.get("disposition", "inconclusive")),
            gate_hash=str(data.get("gate_hash", "")),
        )


def _run_one_point(synthetic_runs: int, floor_threshold: float) -> PowerSweepPoint:
    report = run_semantic_floor_gate_fixture(
        synthetic_runs=synthetic_runs,
        floor_threshold=floor_threshold,
        run_id=f"{EXPERIMENT_ID}-n{synthetic_runs}",
    )
    null_mean = report.permutation_null.get("mean")
    margin = (
        report.real_balanced_accuracy - null_mean
        if report.real_balanced_accuracy is not None and null_mean is not None
        else None
    )
    return PowerSweepPoint(
        synthetic_runs=synthetic_runs,
        n_runs=report.n_runs,
        n_families=report.n_families,
        real_balanced_accuracy=report.real_balanced_accuracy,
        permutation_null_mean=null_mean,
        margin=margin,
        disposition=report.disposition,
        gate_hash=report.gate_hash,
    )


def _resolve_sweep_disposition(points: list[PowerSweepPoint]) -> tuple[str, str]:
    evaluated = [p for p in points if p.margin is not None]
    if not evaluated:
        return "inconclusive", "No sweep point produced a comparable margin."
    recovered = [p for p in evaluated if p.margin >= SIGNAL_MARGIN]
    if recovered:
        first = min(recovered, key=lambda p: p.synthetic_runs)
        return (
            "power_limited",
            f"Signal recovered at synthetic_runs={first.synthetic_runs} "
            f"(margin={first.margin:.3f} >= {SIGNAL_MARGIN:.2f}); SLM-223's "
            "no-signal result at synthetic_runs=4 is consistent with a "
            "statistical-power artifact, not a genuinely absent relationship.",
        )
    worst = evaluated[-1]
    return (
        "genuinely_no_signal_in_range",
        f"No swept point (up to synthetic_runs={worst.synthetic_runs}) reached the "
        f"{SIGNAL_MARGIN:.2f} margin; within this range, more fixture size did not "
        "recover signal for this LOFO/permutation-null protocol.",
    )


def run_power_sweep_fixture(
    *,
    sweep_grid: tuple[int, ...] = DEFAULT_SWEEP_GRID,
    floor_threshold: float = DEFAULT_FLOOR_THRESHOLD,
    run_id: str | None = None,
) -> "PowerSweepReport":
    """Run the SLM-223 gate pipeline at each grid point and summarize the trend."""
    points = [_run_one_point(n, floor_threshold) for n in sweep_grid]
    disposition, rationale = _resolve_sweep_disposition(points)

    payload = {
        "sweep_grid": list(sweep_grid),
        "point_hashes": [p.gate_hash for p in points],
        "floor_threshold": floor_threshold,
    }
    sweep_hash = _sha256(_canonical_json(payload))

    return PowerSweepReport(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        floor_threshold=floor_threshold,
        sweep_grid=tuple(sweep_grid),
        points=tuple(points),
        disposition=disposition,
        disposition_rationale=rationale,
        sweep_hash=sweep_hash,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm224_floor_gate_power_sweep",
            "harness.experiments.slm223_semantic_floor_gate",
            "harness.experiments.slm215_spectral_atlas",
            "harness.experiments.slm214_spectral_snapshot",
        ),
    )


@dataclass(frozen=True)
class PowerSweepReport:
    """Full fixture report for SLM-224."""

    schema: str = "PowerSweepReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm224-floor-gate-power-sweep"
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    floor_threshold: float = DEFAULT_FLOOR_THRESHOLD
    sweep_grid: tuple[int, ...] = DEFAULT_SWEEP_GRID
    points: tuple[PowerSweepPoint, ...] = ()
    disposition: str = "inconclusive"
    disposition_rationale: str = ""
    sweep_hash: str = ""
    honest_caveats: tuple[str, ...] = _HONEST_CAVEATS
    version_stamp: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "hypothesis": self.hypothesis,
            "falsifier": self.falsifier,
            "floor_threshold": self.floor_threshold,
            "sweep_grid": list(self.sweep_grid),
            "points": [p.to_dict() for p in self.points],
            "disposition": self.disposition,
            "disposition_rationale": self.disposition_rationale,
            "sweep_hash": self.sweep_hash,
            "honest_caveats": list(self.honest_caveats),
            "version_stamp": dict(self.version_stamp),
            "timestamp": self.timestamp,
        }

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PowerSweepReport":
        return cls(
            schema=str(data.get("schema", "PowerSweepReportV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", EXPERIMENT_ID)),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            floor_threshold=float(data.get("floor_threshold", DEFAULT_FLOOR_THRESHOLD)),
            sweep_grid=tuple(data.get("sweep_grid", DEFAULT_SWEEP_GRID)),
            points=tuple(PowerSweepPoint.from_dict(p) for p in data.get("points", ())),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            sweep_hash=str(data.get("sweep_hash", "")),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


def render_markdown(report: PowerSweepReport) -> str:
    lines = [
        f"# SLM-224 (NCS0-04): SemanticFloorGateV1 power sweep ({report.run_id})",
        "",
        f"**Matrix set:** `{report.matrix_set}`",
        f"**Version:** `{report.matrix_version}`",
        f"**Status:** {report.status}",
        f"**Claim class:** {report.claim_class}",
        f"**Floor threshold:** parse_rate < {report.floor_threshold}",
        f"**Sweep grid (synthetic_runs):** {list(report.sweep_grid)}",
        f"**Disposition:** {report.disposition} — {report.disposition_rationale}",
        "",
        "## Hypothesis",
        "",
        report.hypothesis,
        "",
        "## Falsifier",
        "",
        report.falsifier,
        "",
        "## Honest caveats",
        "",
        *(f"- {c}" for c in report.honest_caveats),
        "",
        "## Sweep results",
        "",
        "| synthetic_runs | n_runs | n_families | LOFO balanced acc | perm-null mean | margin | disposition |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p in sorted(report.points, key=lambda p: p.synthetic_runs):
        acc = f"{p.real_balanced_accuracy:.3f}" if p.real_balanced_accuracy is not None else "—"
        nm = f"{p.permutation_null_mean:.3f}" if p.permutation_null_mean is not None else "—"
        mg = f"{p.margin:.3f}" if p.margin is not None else "—"
        lines.append(
            f"| {p.synthetic_runs} | {p.n_runs} | {p.n_families} | {acc} | {nm} | {mg} | {p.disposition} |"
        )
    lines += [
        "",
        "## No-go for promotion",
        "",
        "This report is wiring/fixture evidence only. No checkpoint promotion, GPU "
        "train, or ship gate is claimed. It reruns SLM-223's unmodified gate "
        "pipeline at larger synthetic fixture sizes to test whether SLM-223's "
        "no-signal result was a statistical-power artifact of a 4-run fixture; it "
        "does not itself certify `SemanticFloorGateV1` as a promotion or ship gate.",
        "",
    ]
    return "\n".join(lines)
