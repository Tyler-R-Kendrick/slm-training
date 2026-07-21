"""SLM-225 (NCS0-05): SemanticFloorGateV1 family-count sweep.

SLM-224 swept ``synthetic_runs`` from 4 to 128 and found no signal recovery
along that axis (``genuinely_no_signal_in_range``), but its own honest
caveats flagged an untested axis: the SLM-215 synthetic generator always
splits runs into exactly 2 families (``run_idx % 2``), regardless of
``synthetic_runs``, so SLM-224 could only speak to runs-per-family power, not
family-count power.

SLM-215's ``run_spectral_atlas_fixture`` (and SLM-223's
``run_semantic_floor_gate_fixture``) now accept an optional, backward-
compatible ``n_families`` parameter (default 2, unchanged for existing
callers). This harness reruns the *unmodified* SLM-223 gate pipeline at
increasing ``n_families`` values while holding runs-per-family fixed at 4
(so ``synthetic_runs = n_families * 4``), isolating the family-count axis
from the runs-per-family axis SLM-224 already tested, and records whether
the leave-one-family-out margin over the permutation-null baseline ever
clears the same 0.15 signal margin SLM-223 and SLM-224 used.

No new statistical machinery is introduced: this harness only sweeps the
newly-parameterized ``n_families`` fixture knob and aggregates existing
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
    "DEFAULT_RUNS_PER_FAMILY",
    "SIGNAL_MARGIN",
    "FamilySweepPoint",
    "FamilySweepReport",
    "render_markdown",
    "run_family_sweep_fixture",
]

MATRIX_VERSION = "ncs0-05-v1"
MATRIX_SET = "slm225_floor_gate_family_sweep"
EXPERIMENT_ID = "slm225-floor-gate-family-sweep"

SIGNAL_MARGIN = _SLM223_SIGNAL_MARGIN
DEFAULT_RUNS_PER_FAMILY = 4
DEFAULT_SWEEP_GRID: tuple[int, ...] = (2, 4, 8, 16, 32)

_HYPOTHESIS = (
    "SLM-223's SemanticFloorGateV1 'no_signal' disposition, and SLM-224's "
    "'genuinely_no_signal_in_range' finding along the runs-per-family axis, "
    "leave the family-count axis untested: the SLM-215 synthetic generator "
    "used to always fix families at exactly 2. Holding runs-per-family "
    "constant at 4 and sweeping n_families (via the new backward-compatible "
    "n_families parameter) will produce a leave-one-family-out balanced "
    "accuracy that clears the permutation-null mean by the required 0.15 "
    "margin at some larger family count -- i.e. more distinct LOFO folds, "
    "not more runs per fold, recovers signal."
)

_FALSIFIER = (
    "The LOFO-vs-permutation-null margin stays below 0.15 across the full "
    f"swept grid up to n_families={DEFAULT_SWEEP_GRID[-1]} (with runs-per-family "
    "held fixed at 4), i.e. more families within the swept range never "
    "recovers signal -- suggesting the family-count axis is not the "
    "explanation for SLM-223/SLM-224's no-signal results either."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: no trained model, checkpoint promotion, or GPU run.",
    "Reuses the unmodified SLM-223 gate pipeline at each grid point; the only new "
    "code is the backward-compatible n_families parameter threaded through the "
    "existing SLM-215 synthetic generator (default unchanged at n_families=2), and "
    "this sweep harness itself. No new calibration or statistical method is added.",
    "Runs-per-family is held fixed at "
    f"{DEFAULT_RUNS_PER_FAMILY} so synthetic_runs scales with n_families "
    "(synthetic_runs = n_families * runs_per_family); this isolates the "
    "family-count axis from the runs-per-family axis SLM-224 already swept, but "
    "the two axes are not fully orthogonal in one combined sweep -- a full "
    "2D grid over both axes independently is out of scope here.",
    "The synthetic per-matrix signal (0.1 * alpha_z coefficient vs 0.05 noise sd) is "
    "a fixture design choice, not a measurement from a real checkpoint; a positive "
    "result here shows the gate mechanism *can* detect a real embedded signal given "
    "enough families, not that any actual checkpoint atlas has this signal strength.",
    "No causal conclusion is drawn and no promotion or ship-gate claim is made; this "
    "is a diagnostic follow-up to SLM-224's own honest caveat about the untested "
    "family-count axis.",
    "Each grid point's permutation-null baseline is drawn from only 20 random label "
    "permutations, and lower n_families points have very few LOFO folds; margins are "
    "expected to be noisy from point to point. A single grid point crossing the 0.15 "
    "margin is read as an existence result (the mechanism can clear the margin "
    "somewhere in range), not as evidence of a clean monotonic family-count effect -- "
    "the full swept trend, including any non-monotonic points, should be read together.",
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
class FamilySweepPoint:
    """One sweep grid point: a full SLM-223 gate run at a given family count."""

    n_families_requested: int
    runs_per_family: int
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
            "n_families_requested": self.n_families_requested,
            "runs_per_family": self.runs_per_family,
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
    def from_dict(cls, data: dict[str, Any]) -> "FamilySweepPoint":
        return cls(
            n_families_requested=int(data.get("n_families_requested", 0)),
            runs_per_family=int(data.get("runs_per_family", DEFAULT_RUNS_PER_FAMILY)),
            synthetic_runs=int(data.get("synthetic_runs", 0)),
            n_runs=int(data.get("n_runs", 0)),
            n_families=int(data.get("n_families", 0)),
            real_balanced_accuracy=data.get("real_balanced_accuracy"),
            permutation_null_mean=data.get("permutation_null_mean"),
            margin=data.get("margin"),
            disposition=str(data.get("disposition", "inconclusive")),
            gate_hash=str(data.get("gate_hash", "")),
        )


def _run_one_point(n_families: int, runs_per_family: int, floor_threshold: float) -> FamilySweepPoint:
    synthetic_runs = n_families * runs_per_family
    report = run_semantic_floor_gate_fixture(
        synthetic_runs=synthetic_runs,
        n_families=n_families,
        floor_threshold=floor_threshold,
        run_id=f"{EXPERIMENT_ID}-f{n_families}",
    )
    null_mean = report.permutation_null.get("mean")
    margin = (
        report.real_balanced_accuracy - null_mean
        if report.real_balanced_accuracy is not None and null_mean is not None
        else None
    )
    return FamilySweepPoint(
        n_families_requested=n_families,
        runs_per_family=runs_per_family,
        synthetic_runs=synthetic_runs,
        n_runs=report.n_runs,
        n_families=report.n_families,
        real_balanced_accuracy=report.real_balanced_accuracy,
        permutation_null_mean=null_mean,
        margin=margin,
        disposition=report.disposition,
        gate_hash=report.gate_hash,
    )


def _resolve_sweep_disposition(points: list[FamilySweepPoint]) -> tuple[str, str]:
    evaluated = [p for p in points if p.margin is not None]
    if not evaluated:
        return "inconclusive", "No sweep point produced a comparable margin."
    recovered = [p for p in evaluated if p.margin >= SIGNAL_MARGIN]
    if recovered:
        first = min(recovered, key=lambda p: p.n_families_requested)
        return (
            "family_count_limited",
            f"Signal recovered at n_families={first.n_families_requested} "
            f"(margin={first.margin:.3f} >= {SIGNAL_MARGIN:.2f}); SLM-223/SLM-224's "
            "no-signal results are consistent with a family-count-limited artifact, "
            "not a genuinely absent relationship.",
        )
    worst = evaluated[-1]
    return (
        "genuinely_no_signal_in_range",
        f"No swept point (up to n_families={worst.n_families_requested}) reached the "
        f"{SIGNAL_MARGIN:.2f} margin; within this range, more families did not "
        "recover signal for this LOFO/permutation-null protocol.",
    )


def run_family_sweep_fixture(
    *,
    sweep_grid: tuple[int, ...] = DEFAULT_SWEEP_GRID,
    runs_per_family: int = DEFAULT_RUNS_PER_FAMILY,
    floor_threshold: float = DEFAULT_FLOOR_THRESHOLD,
    run_id: str | None = None,
) -> "FamilySweepReport":
    """Run the SLM-223 gate pipeline at each family-count grid point."""
    points = [_run_one_point(n, runs_per_family, floor_threshold) for n in sweep_grid]
    disposition, rationale = _resolve_sweep_disposition(points)

    payload = {
        "sweep_grid": list(sweep_grid),
        "runs_per_family": runs_per_family,
        "point_hashes": [p.gate_hash for p in points],
        "floor_threshold": floor_threshold,
    }
    sweep_hash = _sha256(_canonical_json(payload))

    return FamilySweepReport(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        floor_threshold=floor_threshold,
        sweep_grid=tuple(sweep_grid),
        runs_per_family=runs_per_family,
        points=tuple(points),
        disposition=disposition,
        disposition_rationale=rationale,
        sweep_hash=sweep_hash,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm225_floor_gate_family_sweep",
            "harness.experiments.slm224_floor_gate_power_sweep",
            "harness.experiments.slm223_semantic_floor_gate",
            "harness.experiments.slm215_spectral_atlas",
            "harness.experiments.slm214_spectral_snapshot",
        ),
    )


@dataclass(frozen=True)
class FamilySweepReport:
    """Full fixture report for SLM-225."""

    schema: str = "FamilySweepReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm225-floor-gate-family-sweep"
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    floor_threshold: float = DEFAULT_FLOOR_THRESHOLD
    sweep_grid: tuple[int, ...] = DEFAULT_SWEEP_GRID
    runs_per_family: int = DEFAULT_RUNS_PER_FAMILY
    points: tuple[FamilySweepPoint, ...] = ()
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
            "runs_per_family": self.runs_per_family,
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
    def from_dict(cls, data: dict[str, Any]) -> "FamilySweepReport":
        return cls(
            schema=str(data.get("schema", "FamilySweepReportV1")),
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
            runs_per_family=int(data.get("runs_per_family", DEFAULT_RUNS_PER_FAMILY)),
            points=tuple(FamilySweepPoint.from_dict(p) for p in data.get("points", ())),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            sweep_hash=str(data.get("sweep_hash", "")),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


def render_markdown(report: FamilySweepReport) -> str:
    lines = [
        f"# SLM-225 (NCS0-05): SemanticFloorGateV1 family-count sweep ({report.run_id})",
        "",
        f"**Matrix set:** `{report.matrix_set}`",
        f"**Version:** `{report.matrix_version}`",
        f"**Status:** {report.status}",
        f"**Claim class:** {report.claim_class}",
        f"**Floor threshold:** parse_rate < {report.floor_threshold}",
        f"**Runs per family (fixed):** {report.runs_per_family}",
        f"**Sweep grid (n_families):** {list(report.sweep_grid)}",
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
        "| n_families | runs_per_family | synthetic_runs | n_runs | n_families (actual) | LOFO balanced acc | perm-null mean | margin | disposition |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p in sorted(report.points, key=lambda p: p.n_families_requested):
        acc = f"{p.real_balanced_accuracy:.3f}" if p.real_balanced_accuracy is not None else "—"
        nm = f"{p.permutation_null_mean:.3f}" if p.permutation_null_mean is not None else "—"
        mg = f"{p.margin:.3f}" if p.margin is not None else "—"
        lines.append(
            f"| {p.n_families_requested} | {p.runs_per_family} | {p.synthetic_runs} | {p.n_runs} | "
            f"{p.n_families} | {acc} | {nm} | {mg} | {p.disposition} |"
        )
    lines += [
        "",
        "## No-go for promotion",
        "",
        "This report is wiring/fixture evidence only. No checkpoint promotion, GPU "
        "train, or ship gate is claimed. It reruns SLM-223's unmodified gate "
        "pipeline at increasing n_families values (runs-per-family held fixed) to "
        "test whether the family-count axis SLM-224 could not test recovers "
        "signal; it does not itself certify `SemanticFloorGateV1` as a promotion "
        "or ship gate.",
        "",
    ]
    return "\n".join(lines)
