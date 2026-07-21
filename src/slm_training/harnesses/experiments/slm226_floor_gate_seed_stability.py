"""SLM-226 (NCS0-06): SemanticFloorGateV1 permutation-null seed-stability sweep.

SLM-225 swept ``n_families`` from 2 to 32 (runs-per-family held fixed at 4)
and found a non-monotonic pattern: the LOFO-vs-permutation-null margin cleared
the 0.15 signal threshold at ``n_families=4`` (0.188) and ``n_families=32``
(0.229), but *not* at ``n_families=8`` (0.094) or ``n_families=16`` (0.000) --
a dip sandwiched between two signal-clearing points. SLM-225's own honest
caveats flagged the likely culprit without testing it: each grid point's
permutation-null baseline is drawn from only 20 label permutations at a single
fixed seed (11), so margins are "expected to be noisy from point to point."

This harness reruns the *unmodified* SLM-223 gate pipeline at each of
SLM-225's grid points, holding the synthetic data fixed (seed 42, unchanged)
but varying only the permutation-null seed via the new backward-compatible
``permutation_seed`` parameter (SLM-223). If the dip at n_families=8/16 is
explained by permutation-null sampling noise, some seeds at those points
should push the margin across the 0.15 threshold. If the dip is stable across
seeds, permutation-null noise does not explain it (though this does not rule
out synthetic-data-seed noise, which is out of scope here -- see honest
caveats).

No new statistical machinery is introduced: this harness only varies the
already-existing ``permutation_seed`` knob and aggregates existing per-point
``SemanticFloorGateReport`` objects. No model is trained, no GPU is required,
and no ship-gate or promotion claim is made.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm223_semantic_floor_gate import (
    DEFAULT_FLOOR_THRESHOLD,
    SIGNAL_MARGIN as _SLM223_SIGNAL_MARGIN,
    run_semantic_floor_gate_fixture,
)
from slm_training.harnesses.experiments.slm225_floor_gate_family_sweep import (
    DEFAULT_RUNS_PER_FAMILY,
    DEFAULT_SWEEP_GRID,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "DEFAULT_SWEEP_GRID",
    "DEFAULT_RUNS_PER_FAMILY",
    "DEFAULT_SEEDS",
    "SIGNAL_MARGIN",
    "SeedStabilityPoint",
    "SeedStabilityReport",
    "render_markdown",
    "run_seed_stability_fixture",
]

MATRIX_VERSION = "ncs0-06-v1"
MATRIX_SET = "slm226_floor_gate_seed_stability"
EXPERIMENT_ID = "slm226-floor-gate-seed-stability"

SIGNAL_MARGIN = _SLM223_SIGNAL_MARGIN
# 8 permutation-null seeds distinct from SLM-223/224/225's hardcoded seed=11
# (kept as the first entry so the default-seed point is reproduced exactly).
DEFAULT_SEEDS: tuple[int, ...] = (11, 3, 7, 19, 23, 29, 31, 37)

_HYPOTHESIS = (
    "SLM-225's non-monotonic dip at n_families=8 (margin=0.094) and "
    "n_families=16 (margin=0.000) -- both below the 0.15 signal margin, "
    "sandwiched between signal-clearing points at n_families=4 (0.188) and "
    "n_families=32 (0.229) -- is explained by permutation-null sampling noise: "
    "each grid point's null baseline is drawn from only 20 permutations at a "
    "single fixed seed. Rerunning the unmodified SLM-223 gate pipeline at each "
    "SLM-225 grid point across multiple permutation-null seeds (synthetic data "
    "held fixed) will show the margin at n_families=8 and/or 16 crossing 0.15 "
    "for at least one alternate seed, i.e. the dip is a resampling artifact, "
    "not a stable non-monotonic effect."
)

_FALSIFIER = (
    "The margin at n_families=8 and n_families=16 stays below the 0.15 signal "
    "threshold for every swept permutation-null seed -- i.e. the dip is stable "
    "under permutation-null resampling and is not explained by null-baseline "
    "sampling noise alone (though a residual synthetic-data-seed noise source "
    "remains untested; see honest caveats)."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: no trained model, checkpoint promotion, or GPU run.",
    "Reuses the unmodified SLM-223 gate pipeline and SLM-225's grid at each point; the "
    "only new code is the backward-compatible permutation_seed parameter threaded "
    "through SLM-223's run_semantic_floor_gate_fixture (default unchanged at seed=11), "
    "and this sweep harness itself. No new calibration or statistical method is added.",
    "This sweep varies only the permutation-null seed; the synthetic per-run data "
    "itself is generated with a single hardcoded seed (42, unchanged) in SLM-215's "
    "generator. A residual noise source -- sensitivity of the real LOFO balanced "
    "accuracy to the synthetic-data seed -- is NOT tested here and remains an open "
    "axis; a stable-under-permutation-resampling result at n_families=8/16 does not "
    "rule out that a different synthetic-data seed would shift real_balanced_accuracy "
    "itself and produce a different dip pattern.",
    "Only 8 permutation-null seeds are swept per grid point (each itself averaging "
    "20 permutation draws); this bounds, but does not eliminate, sampling uncertainty "
    "in the seed-to-seed margin statistics reported here.",
    "No causal conclusion is drawn and no promotion or ship-gate claim is made; this "
    "is a diagnostic follow-up to SLM-225's own honest caveat about permutation-null "
    "sampling noise as a candidate explanation for its non-monotonic sweep.",
    "The real_balanced_accuracy at each n_families point is identical across all "
    "swept seeds by construction (permutation_seed only affects the null baseline), "
    "so this harness cannot itself distinguish 'genuine non-monotonicity in the real "
    "signal' from 'non-monotonicity in the real signal driven by the fixed "
    "synthetic-data seed' -- it can only test whether the null-baseline axis alone "
    "explains the dip.",
    "In the committed fixture run, the permutation-null mean (and hence the margin) "
    "was empirically identical across all 8 swept seeds (margin_std=0.000) at "
    "n_families=2, 8, and 16, and only showed seed-to-seed variance at n_families=32; "
    "read literally, this suggests the label-permutation space at smaller run counts "
    "is constrained enough that 20 draws already converge to the same mean regardless "
    "of RNG seed for this fixture's class balance -- itself a fixture-scale artifact, "
    "not evidence that a real checkpoint atlas would behave identically.",
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
class SeedStabilityPoint:
    """Aggregated margin statistics across permutation-null seeds at one n_families grid point."""

    n_families: int
    runs_per_family: int
    synthetic_runs: int
    seeds: tuple[int, ...]
    margins: tuple[float, ...]
    real_balanced_accuracy: float | None
    margin_mean: float | None
    margin_std: float | None
    margin_min: float | None
    margin_max: float | None
    seeds_crossing_margin: int
    slm225_disposition: str
    stability: str
    gate_hashes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_families": self.n_families,
            "runs_per_family": self.runs_per_family,
            "synthetic_runs": self.synthetic_runs,
            "seeds": list(self.seeds),
            "margins": list(self.margins),
            "real_balanced_accuracy": self.real_balanced_accuracy,
            "margin_mean": self.margin_mean,
            "margin_std": self.margin_std,
            "margin_min": self.margin_min,
            "margin_max": self.margin_max,
            "seeds_crossing_margin": self.seeds_crossing_margin,
            "slm225_disposition": self.slm225_disposition,
            "stability": self.stability,
            "gate_hashes": list(self.gate_hashes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeedStabilityPoint":
        return cls(
            n_families=int(data.get("n_families", 0)),
            runs_per_family=int(data.get("runs_per_family", DEFAULT_RUNS_PER_FAMILY)),
            synthetic_runs=int(data.get("synthetic_runs", 0)),
            seeds=tuple(int(s) for s in data.get("seeds", ())),
            margins=tuple(float(m) for m in data.get("margins", ())),
            real_balanced_accuracy=data.get("real_balanced_accuracy"),
            margin_mean=data.get("margin_mean"),
            margin_std=data.get("margin_std"),
            margin_min=data.get("margin_min"),
            margin_max=data.get("margin_max"),
            seeds_crossing_margin=int(data.get("seeds_crossing_margin", 0)),
            slm225_disposition=str(data.get("slm225_disposition", "")),
            stability=str(data.get("stability", "inconclusive")),
            gate_hashes=tuple(str(h) for h in data.get("gate_hashes", ())),
        )


def _run_one_point(
    n_families: int, runs_per_family: int, floor_threshold: float, seeds: tuple[int, ...]
) -> SeedStabilityPoint:
    synthetic_runs = n_families * runs_per_family
    margins: list[float] = []
    gate_hashes: list[str] = []
    real_accs: list[float] = []
    for seed in seeds:
        report = run_semantic_floor_gate_fixture(
            synthetic_runs=synthetic_runs,
            n_families=n_families,
            floor_threshold=floor_threshold,
            permutation_seed=seed,
            run_id=f"{EXPERIMENT_ID}-f{n_families}-s{seed}",
        )
        null_mean = report.permutation_null.get("mean")
        if report.real_balanced_accuracy is not None and null_mean is not None:
            margins.append(report.real_balanced_accuracy - null_mean)
            real_accs.append(report.real_balanced_accuracy)
        gate_hashes.append(report.gate_hash)

    real_acc = real_accs[0] if real_accs else None
    margin_mean = float(statistics.fmean(margins)) if margins else None
    margin_std = float(statistics.pstdev(margins)) if len(margins) > 1 else (0.0 if margins else None)
    margin_min = float(min(margins)) if margins else None
    margin_max = float(max(margins)) if margins else None
    seeds_crossing = sum(1 for m in margins if m >= SIGNAL_MARGIN)

    if not margins:
        slm225_disposition = "inconclusive"
    elif margin_max is not None and margin_max >= SIGNAL_MARGIN and margin_min is not None and margin_min < SIGNAL_MARGIN:
        slm225_disposition = "mixed"
    elif margin_min is not None and margin_min >= SIGNAL_MARGIN:
        slm225_disposition = "signal_predictive"
    else:
        slm225_disposition = "no_signal"

    if not margins:
        stability = "inconclusive"
    elif seeds_crossing > 0 and seeds_crossing < len(margins):
        stability = "seed_sensitive"
    elif seeds_crossing == 0:
        stability = "stable_no_signal"
    else:
        stability = "stable_signal"

    return SeedStabilityPoint(
        n_families=n_families,
        runs_per_family=runs_per_family,
        synthetic_runs=synthetic_runs,
        seeds=seeds,
        margins=tuple(margins),
        real_balanced_accuracy=real_acc,
        margin_mean=margin_mean,
        margin_std=margin_std,
        margin_min=margin_min,
        margin_max=margin_max,
        seeds_crossing_margin=seeds_crossing,
        slm225_disposition=slm225_disposition,
        stability=stability,
        gate_hashes=tuple(gate_hashes),
    )


def _resolve_sweep_disposition(
    points: list[SeedStabilityPoint], dip_families: tuple[int, ...]
) -> tuple[str, str]:
    dip_points = [p for p in points if p.n_families in dip_families and p.margins]
    if not dip_points:
        return "inconclusive", "No dip-family grid point produced a comparable margin."
    if any(p.stability == "seed_sensitive" for p in dip_points):
        sensitive = [p for p in dip_points if p.stability == "seed_sensitive"]
        names = ", ".join(f"n_families={p.n_families}" for p in sensitive)
        return (
            "permutation_noise_explains_dip",
            f"At least one dip point ({names}) has a permutation-null-seed-dependent "
            f"margin that crosses the {SIGNAL_MARGIN:.2f} threshold for some but not "
            "all seeds -- SLM-225's non-monotonic dip is at least partly explained by "
            "permutation-null sampling noise.",
        )
    return (
        "dip_stable_under_permutation_resampling",
        f"Every dip point ({', '.join(f'n_families={p.n_families}' for p in dip_points)}) "
        f"stayed below the {SIGNAL_MARGIN:.2f} margin across all {len(dip_points[0].seeds)} "
        "swept permutation-null seeds -- the dip is not explained by permutation-null "
        "sampling noise alone (synthetic-data-seed noise remains untested).",
    )


def run_seed_stability_fixture(
    *,
    sweep_grid: tuple[int, ...] = DEFAULT_SWEEP_GRID,
    runs_per_family: int = DEFAULT_RUNS_PER_FAMILY,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    floor_threshold: float = DEFAULT_FLOOR_THRESHOLD,
    dip_families: tuple[int, ...] = (8, 16),
    run_id: str | None = None,
) -> "SeedStabilityReport":
    """Rerun the SLM-223 gate pipeline at each SLM-225 grid point across multiple permutation-null seeds."""
    points = [_run_one_point(n, runs_per_family, floor_threshold, seeds) for n in sweep_grid]
    disposition, rationale = _resolve_sweep_disposition(points, dip_families)

    payload = {
        "sweep_grid": list(sweep_grid),
        "runs_per_family": runs_per_family,
        "seeds": list(seeds),
        "point_gate_hashes": [list(p.gate_hashes) for p in points],
        "floor_threshold": floor_threshold,
    }
    sweep_hash = _sha256(_canonical_json(payload))

    return SeedStabilityReport(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        floor_threshold=floor_threshold,
        sweep_grid=tuple(sweep_grid),
        runs_per_family=runs_per_family,
        seeds=tuple(seeds),
        dip_families=tuple(dip_families),
        points=tuple(points),
        disposition=disposition,
        disposition_rationale=rationale,
        sweep_hash=sweep_hash,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm226_floor_gate_seed_stability",
            "harness.experiments.slm225_floor_gate_family_sweep",
            "harness.experiments.slm224_floor_gate_power_sweep",
            "harness.experiments.slm223_semantic_floor_gate",
            "harness.experiments.slm215_spectral_atlas",
            "harness.experiments.slm214_spectral_snapshot",
        ),
    )


@dataclass(frozen=True)
class SeedStabilityReport:
    """Full fixture report for SLM-226."""

    schema: str = "SeedStabilityReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm226-floor-gate-seed-stability"
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    floor_threshold: float = DEFAULT_FLOOR_THRESHOLD
    sweep_grid: tuple[int, ...] = DEFAULT_SWEEP_GRID
    runs_per_family: int = DEFAULT_RUNS_PER_FAMILY
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    dip_families: tuple[int, ...] = (8, 16)
    points: tuple[SeedStabilityPoint, ...] = ()
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
            "seeds": list(self.seeds),
            "dip_families": list(self.dip_families),
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
    def from_dict(cls, data: dict[str, Any]) -> "SeedStabilityReport":
        return cls(
            schema=str(data.get("schema", "SeedStabilityReportV1")),
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
            seeds=tuple(data.get("seeds", DEFAULT_SEEDS)),
            dip_families=tuple(data.get("dip_families", (8, 16))),
            points=tuple(SeedStabilityPoint.from_dict(p) for p in data.get("points", ())),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            sweep_hash=str(data.get("sweep_hash", "")),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


def render_markdown(report: SeedStabilityReport) -> str:
    lines = [
        f"# SLM-226 (NCS0-06): SemanticFloorGateV1 permutation-null seed-stability sweep ({report.run_id})",
        "",
        f"**Matrix set:** `{report.matrix_set}`",
        f"**Version:** `{report.matrix_version}`",
        f"**Status:** {report.status}",
        f"**Claim class:** {report.claim_class}",
        f"**Floor threshold:** parse_rate < {report.floor_threshold}",
        f"**Runs per family (fixed):** {report.runs_per_family}",
        f"**Sweep grid (n_families):** {list(report.sweep_grid)}",
        f"**Permutation-null seeds swept:** {list(report.seeds)}",
        f"**SLM-225 dip grid points under test:** {list(report.dip_families)}",
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
        "| n_families | real LOFO bal. acc | margin mean | margin std | margin min | margin max | seeds crossing 0.15 / total | SLM-225 disposition (this run) | stability |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p in sorted(report.points, key=lambda p: p.n_families):
        acc = f"{p.real_balanced_accuracy:.3f}" if p.real_balanced_accuracy is not None else "—"
        mm = f"{p.margin_mean:.3f}" if p.margin_mean is not None else "—"
        ms = f"{p.margin_std:.3f}" if p.margin_std is not None else "—"
        mn = f"{p.margin_min:.3f}" if p.margin_min is not None else "—"
        mx = f"{p.margin_max:.3f}" if p.margin_max is not None else "—"
        lines.append(
            f"| {p.n_families} | {acc} | {mm} | {ms} | {mn} | {mx} | "
            f"{p.seeds_crossing_margin} / {len(p.margins)} | {p.slm225_disposition} | {p.stability} |"
        )
    lines += [
        "",
        "## No-go for promotion",
        "",
        "This report is wiring/fixture evidence only. No checkpoint promotion, GPU "
        "train, or ship gate is claimed. It reruns SLM-223's unmodified gate "
        "pipeline at SLM-225's grid points across multiple permutation-null seeds "
        "to test whether SLM-225's non-monotonic n_families=8/16 dip is explained "
        "by permutation-null sampling noise; it does not itself certify "
        "`SemanticFloorGateV1` as a promotion or ship gate, and does not test "
        "synthetic-data-seed noise (a separate, still-open axis).",
        "",
    ]
    return "\n".join(lines)
