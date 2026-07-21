"""SLM-223 (NCS0-03): SemanticFloorGateV1 — a calibrated pre-eval gate that
flags checkpoints likely to be under a parse-rate "floor" using role-weighted
spectral signal from SpectralAtlasV1.

CPU-only fixture/wiring harness. Consumes SpectralAtlasV1 rows (SLM-215),
learns per-role weights and a direction+threshold on out-of-fold training
families only (leave-one-family-out), and evaluates whether the resulting
gate separates floor-risk checkpoints from healthy ones better than a
label-permuted control.

No model is trained, no GPU is required, and no ship-gate claim is made. The
gate is explicitly a diagnostic pre-screen candidate, not a promotion gate:
SLM-215 named ``SemanticFloorGateV1`` as a prerequisite for production-quality
atlas claims, and this harness is the first wiring pass at that gate.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm215_spectral_atlas import (
    SpectralAtlasV1,
    run_spectral_atlas_fixture,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "DEFAULT_FLOOR_THRESHOLD",
    "SIGNAL_MARGIN",
    "FloorGateRunRow",
    "SemanticFloorGateReport",
    "render_markdown",
    "run_semantic_floor_gate_fixture",
]

MATRIX_VERSION = "ncs0-03-v1"
MATRIX_SET = "slm223_semantic_floor_gate"
EXPERIMENT_ID = "slm223-semantic-floor-gate"

DEFAULT_FLOOR_THRESHOLD = 0.5
_PERMUTATION_DRAWS = 20
SIGNAL_MARGIN = 0.15
_SIGNAL_MARGIN = SIGNAL_MARGIN  # backward-compatible alias for existing internal references

_HYPOTHESIS = (
    "A role-weighted aggregate of SpectralAtlasV1 alpha_z, calibrated on "
    "out-of-fold training families only (direction + median threshold), can "
    "flag checkpoints under a parse-rate floor with a leave-one-family-out "
    "balanced accuracy that exceeds a label-permuted control."
)

_FALSIFIER = (
    "The calibrated gate's leave-one-family-out balanced accuracy does not "
    "exceed the label-permutation-null mean by at least "
    f"{_SIGNAL_MARGIN:.2f}, or there are too few families/runs to calibrate "
    "out-of-fold at all."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: no trained model, checkpoint promotion, or GPU run.",
    "Built entirely on SLM-215 SpectralAtlasV1 rows (real or synthetic); no new "
    "spectral measurement is taken here.",
    "This is a diagnostic pre-screen candidate, not a promotion or ship gate. It "
    "does not replace full suite evaluation and makes no readiness claim.",
    "Role weights and thresholds are learned only from training-fold data inside "
    "leave-one-family-out; the tiny fixture size (2-4 runs per fold) limits "
    "statistical power and the calibration is expected to be noisy.",
    "The floor threshold on parse_rate is a fixed configuration knob, not fit "
    "from data; changing it changes which checkpoints count as floor-risk.",
    "No causal conclusion is drawn; a positive result only means the alpha_z "
    "signal correlates with the floor label in this fixture, not that it causes it.",
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
class FloorGateRunRow:
    """One checkpoint/run's floor-gate inputs, label, and (when evaluated) decision."""

    gate_version: str = "SemanticFloorGateV1"
    run_id: str = ""
    family: str = "unknown"
    n_matrices: int = 0
    mean_alpha_z: float | None = None
    weighted_alpha_z: float | None = None
    parse_rate: float | None = None
    floor_label: bool | None = None
    gate_flag: bool | None = None
    fold: str = "unassigned"

    @property
    def correct(self) -> bool | None:
        if self.floor_label is None or self.gate_flag is None:
            return None
        return self.floor_label == self.gate_flag

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_version": self.gate_version,
            "run_id": self.run_id,
            "family": self.family,
            "n_matrices": self.n_matrices,
            "mean_alpha_z": self.mean_alpha_z,
            "weighted_alpha_z": self.weighted_alpha_z,
            "parse_rate": self.parse_rate,
            "floor_label": self.floor_label,
            "gate_flag": self.gate_flag,
            "correct": self.correct,
            "fold": self.fold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FloorGateRunRow":
        return cls(
            gate_version=str(data.get("gate_version", "SemanticFloorGateV1")),
            run_id=str(data.get("run_id", "")),
            family=str(data.get("family", "unknown")),
            n_matrices=int(data.get("n_matrices", 0)),
            mean_alpha_z=data.get("mean_alpha_z"),
            weighted_alpha_z=data.get("weighted_alpha_z"),
            parse_rate=data.get("parse_rate"),
            floor_label=data.get("floor_label"),
            gate_flag=data.get("gate_flag"),
            fold=str(data.get("fold", "unassigned")),
        )


def _group_by_run(rows: list[SpectralAtlasV1]) -> dict[str, list[SpectralAtlasV1]]:
    by_run: dict[str, list[SpectralAtlasV1]] = {}
    for row in rows:
        by_run.setdefault(row.run_id, []).append(row)
    return by_run


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    den = (den_x * den_y) ** 0.5
    return float(num / den) if den > 0 else 0.0


def _learn_role_weights(
    train_matrix_rows: list[SpectralAtlasV1], parse_rate_by_run: dict[str, float | None]
) -> dict[str, float]:
    """Learn per-role weight = Pearson(alpha_z, parse_rate) within role, train-fold only.

    Uses ``parse_rate_by_run`` (not the raw ``SpectralAtlasV1.parse_rate`` field) as
    the target so that permutation-null draws, which pass a shuffled
    ``parse_rate_by_run``, re-derive role weights under the shuffled labels too —
    a proper "shuffle y, refit everything" permutation test rather than only
    shuffling the downstream threshold calibration.
    """
    by_role: dict[str, list[tuple[float, float]]] = {}
    for row in train_matrix_rows:
        target = parse_rate_by_run.get(row.run_id)
        if row.alpha_z is None or target is None:
            continue
        by_role.setdefault(row.semantic_role, []).append((row.alpha_z, target))
    weights: dict[str, float] = {}
    for role, pairs in by_role.items():
        if len(pairs) < 3:
            continue
        weights[role] = _pearson([p[0] for p in pairs], [p[1] for p in pairs])
    return weights


def _weighted_run_alpha_z(
    run_matrix_rows: list[SpectralAtlasV1], role_weights: dict[str, float]
) -> float | None:
    eligible = [r for r in run_matrix_rows if r.alpha_z is not None]
    if not eligible:
        return None
    weighted_sum = 0.0
    weight_total = 0.0
    for row in eligible:
        w = role_weights.get(row.semantic_role, 0.0)
        weighted_sum += w * row.alpha_z
        weight_total += abs(w)
    if weight_total <= 0:
        return float(sum(r.alpha_z for r in eligible) / len(eligible))
    return float(weighted_sum / weight_total)


def _mean_run_alpha_z(run_matrix_rows: list[SpectralAtlasV1]) -> float | None:
    eligible = [r for r in run_matrix_rows if r.alpha_z is not None]
    if not eligible:
        return None
    return float(sum(r.alpha_z for r in eligible) / len(eligible))


def _mean_run_parse_rate(run_matrix_rows: list[SpectralAtlasV1]) -> float | None:
    eligible = [r for r in run_matrix_rows if r.parse_rate is not None]
    if not eligible:
        return None
    return float(sum(r.parse_rate for r in eligible) / len(eligible))


def _calibrate_direction_and_threshold(
    train_run_ids: list[str],
    weighted_alpha_z: dict[str, float],
    floor_label: dict[str, bool],
) -> tuple[str, float] | None:
    """Learn (direction, threshold) from training-fold run aggregates only.

    direction == 'low_is_risk' means values <= threshold are flagged floor-risk;
    'high_is_risk' means values >= threshold are flagged floor-risk. Threshold is
    the median of train-run weighted_alpha_z (a simple, non-searched split to
    avoid overfitting a tiny fixture).
    """
    values = [weighted_alpha_z[r] for r in train_run_ids if r in weighted_alpha_z]
    if len(values) < 2:
        return None
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 0:
        threshold = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    else:
        threshold = sorted_vals[mid]

    corr = _pearson(
        [weighted_alpha_z[r] for r in train_run_ids if r in weighted_alpha_z],
        [1.0 if floor_label[r] else 0.0 for r in train_run_ids if r in weighted_alpha_z],
    )
    # Positive corr(alpha_z, floor_risk) => high values are risky; negative => low is risky.
    direction = "high_is_risk" if corr >= 0 else "low_is_risk"
    return direction, threshold


def _apply_gate(value: float, direction: str, threshold: float) -> bool:
    if direction == "high_is_risk":
        return value >= threshold
    return value <= threshold


def _balanced_accuracy(rows: list[FloorGateRunRow]) -> float | None:
    labeled = [r for r in rows if r.floor_label is not None and r.gate_flag is not None]
    if not labeled:
        return None
    positives = [r for r in labeled if r.floor_label]
    negatives = [r for r in labeled if not r.floor_label]
    sensitivity = (
        sum(1 for r in positives if r.gate_flag) / len(positives) if positives else None
    )
    specificity = (
        sum(1 for r in negatives if not r.gate_flag) / len(negatives) if negatives else None
    )
    parts = [v for v in (sensitivity, specificity) if v is not None]
    if not parts:
        return None
    return float(sum(parts) / len(parts))


def _run_lofo_pipeline(
    run_ids: list[str],
    families: dict[str, str],
    matrix_rows_by_run: dict[str, list[SpectralAtlasV1]],
    parse_rate_by_run: dict[str, float | None],
    floor_threshold: float,
) -> list[FloorGateRunRow]:
    """One leave-one-family-out pass: calibrate on train families, evaluate held-out."""
    distinct_families = sorted({families[r] for r in run_ids})
    floor_label = {
        r: (parse_rate_by_run[r] < floor_threshold)
        for r in run_ids
        if parse_rate_by_run.get(r) is not None
    }
    out_rows: list[FloorGateRunRow] = []

    if len(distinct_families) < 2:
        for run_id in run_ids:
            out_rows.append(
                FloorGateRunRow(
                    run_id=run_id,
                    family=families[run_id],
                    n_matrices=len(matrix_rows_by_run.get(run_id, [])),
                    mean_alpha_z=_mean_run_alpha_z(matrix_rows_by_run.get(run_id, [])),
                    weighted_alpha_z=None,
                    parse_rate=parse_rate_by_run.get(run_id),
                    floor_label=floor_label.get(run_id),
                    gate_flag=None,
                    fold="insufficient_families",
                )
            )
        return out_rows

    for held_out in distinct_families:
        train_run_ids = [r for r in run_ids if families[r] != held_out]
        test_run_ids = [r for r in run_ids if families[r] == held_out]
        train_matrix_rows = [
            row for r in train_run_ids for row in matrix_rows_by_run.get(r, [])
        ]
        role_weights = _learn_role_weights(train_matrix_rows, parse_rate_by_run)

        weighted_by_run: dict[str, float] = {}
        for r in run_ids:
            wa = _weighted_run_alpha_z(matrix_rows_by_run.get(r, []), role_weights)
            if wa is not None:
                weighted_by_run[r] = wa

        calibration = _calibrate_direction_and_threshold(
            [r for r in train_run_ids if r in floor_label], weighted_by_run, floor_label
        )
        for run_id in test_run_ids:
            wa = weighted_by_run.get(run_id)
            gate_flag = None
            if calibration is not None and wa is not None:
                direction, threshold = calibration
                gate_flag = _apply_gate(wa, direction, threshold)
            out_rows.append(
                FloorGateRunRow(
                    run_id=run_id,
                    family=families[run_id],
                    n_matrices=len(matrix_rows_by_run.get(run_id, [])),
                    mean_alpha_z=_mean_run_alpha_z(matrix_rows_by_run.get(run_id, [])),
                    weighted_alpha_z=wa,
                    parse_rate=parse_rate_by_run.get(run_id),
                    floor_label=floor_label.get(run_id),
                    gate_flag=gate_flag,
                    fold=f"held_out_{held_out}",
                )
            )
    return out_rows


def _permutation_null(
    run_ids: list[str],
    families: dict[str, str],
    matrix_rows_by_run: dict[str, list[SpectralAtlasV1]],
    parse_rate_by_run: dict[str, float | None],
    floor_threshold: float,
    draws: int = _PERMUTATION_DRAWS,
    seed: int = 11,
) -> dict[str, Any]:
    """Shuffle floor labels across runs and re-run the LOFO pipeline each draw."""
    rng = random.Random(seed)
    real_labels = {
        r: parse_rate_by_run[r] < floor_threshold
        for r in run_ids
        if parse_rate_by_run.get(r) is not None
    }
    label_run_ids = list(real_labels.keys())
    accuracies: list[float] = []
    for _ in range(draws):
        shuffled_values = list(real_labels.values())
        rng.shuffle(shuffled_values)
        shuffled_labels = dict(zip(label_run_ids, shuffled_values))
        shuffled_parse = {
            r: (0.0 if shuffled_labels.get(r) else 1.0) if r in shuffled_labels else parse_rate_by_run.get(r)
            for r in run_ids
        }
        # Reuse the LOFO pipeline against a shuffled binary "parse" surrogate at
        # the same floor_threshold (0.5), which reproduces the shuffled label.
        rows = _run_lofo_pipeline(
            run_ids, families, matrix_rows_by_run, shuffled_parse, floor_threshold=0.5
        )
        acc = _balanced_accuracy(rows)
        if acc is not None:
            accuracies.append(acc)
    if not accuracies:
        return {"status": "insufficient_data", "mean": None, "draws": draws}
    mean = float(sum(accuracies) / len(accuracies))
    return {"status": "evaluated", "mean": mean, "draws": len(accuracies)}


def _resolve_disposition(
    lofo_rows: list[FloorGateRunRow], null_result: dict[str, Any]
) -> tuple[str, str]:
    evaluated = [r for r in lofo_rows if r.gate_flag is not None]
    if not evaluated:
        return "inconclusive", "No leave-one-family-out gate decisions could be made (insufficient families/runs)."
    real_acc = _balanced_accuracy(lofo_rows)
    if real_acc is None:
        return "inconclusive", "Balanced accuracy could not be computed (missing labels)."
    if null_result.get("status") != "evaluated":
        return "inconclusive", "Permutation-null baseline could not be computed."
    null_mean = null_result["mean"]
    margin = real_acc - null_mean
    if margin >= _SIGNAL_MARGIN:
        return (
            "signal_predictive",
            f"LOFO balanced accuracy {real_acc:.3f} exceeds permutation-null mean "
            f"{null_mean:.3f} by {margin:.3f} (>= {_SIGNAL_MARGIN:.2f} margin).",
        )
    return (
        "no_signal",
        f"LOFO balanced accuracy {real_acc:.3f} does not clear the permutation-null "
        f"mean {null_mean:.3f} by the required {_SIGNAL_MARGIN:.2f} margin (margin={margin:.3f}).",
    )


def run_semantic_floor_gate_fixture(
    reports_dir: Path | None = None,
    *,
    synthetic_runs: int = 4,
    n_families: int = 2,
    floor_threshold: float = DEFAULT_FLOOR_THRESHOLD,
    run_id: str | None = None,
) -> "SemanticFloorGateReport":
    """Build a SemanticFloorGateV1 fixture report from a SpectralAtlasV1 source.

    ``n_families`` (default 2, backward-compatible) is forwarded to the
    SLM-215 synthetic generator via ``run_spectral_atlas_fixture``; added for
    SLM-225's family-count power sweep.
    """
    atlas = run_spectral_atlas_fixture(
        reports_dir=reports_dir, synthetic_runs=synthetic_runs, n_families=n_families
    )
    matrix_rows_by_run = _group_by_run(list(atlas.rows))
    families = {r: rows[0].family for r, rows in matrix_rows_by_run.items()}
    parse_rate_by_run = {r: _mean_run_parse_rate(rows) for r, rows in matrix_rows_by_run.items()}
    run_ids = sorted(matrix_rows_by_run.keys())

    lofo_rows = _run_lofo_pipeline(
        run_ids, families, matrix_rows_by_run, parse_rate_by_run, floor_threshold
    )
    null_result = _permutation_null(
        run_ids, families, matrix_rows_by_run, parse_rate_by_run, floor_threshold
    )
    real_balanced_accuracy = _balanced_accuracy(lofo_rows)
    disposition, rationale = _resolve_disposition(lofo_rows, null_result)

    # Full-data (non-held-out) calibration, reported for transparency only; it is
    # never used for the disposition, which relies solely on out-of-fold LOFO rows.
    full_role_weights = _learn_role_weights(list(atlas.rows), parse_rate_by_run)

    n_families = len({families[r] for r in run_ids})
    payload = {
        "row_digests": sorted(_digest(r.to_dict()) for r in lofo_rows),
        "null_result": null_result,
        "real_balanced_accuracy": real_balanced_accuracy,
    }
    gate_hash = _sha256(_canonical_json(payload))

    return SemanticFloorGateReport(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        floor_threshold=floor_threshold,
        atlas_hash=atlas.atlas_hash,
        rows=tuple(lofo_rows),
        n_runs=len(run_ids),
        n_families=n_families,
        full_role_weights=full_role_weights,
        real_balanced_accuracy=real_balanced_accuracy,
        permutation_null=null_result,
        gate_hash=gate_hash,
        disposition=disposition,
        disposition_rationale=rationale,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm223_semantic_floor_gate",
            "harness.experiments.slm215_spectral_atlas",
            "harness.experiments.slm214_spectral_snapshot",
        ),
    )


@dataclass(frozen=True)
class SemanticFloorGateReport:
    """Full fixture report for SLM-223."""

    schema: str = "SemanticFloorGateReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm223-semantic-floor-gate"
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    floor_threshold: float = DEFAULT_FLOOR_THRESHOLD
    atlas_hash: str = ""
    rows: tuple[FloorGateRunRow, ...] = ()
    n_runs: int = 0
    n_families: int = 0
    full_role_weights: dict[str, float] = field(default_factory=dict)
    real_balanced_accuracy: float | None = None
    permutation_null: dict[str, Any] = field(default_factory=dict)
    gate_hash: str = ""
    disposition: str = "inconclusive"
    disposition_rationale: str = ""
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
            "atlas_hash": self.atlas_hash,
            "rows": [r.to_dict() for r in self.rows],
            "n_runs": self.n_runs,
            "n_families": self.n_families,
            "full_role_weights": dict(self.full_role_weights),
            "real_balanced_accuracy": self.real_balanced_accuracy,
            "permutation_null": dict(self.permutation_null),
            "gate_hash": self.gate_hash,
            "disposition": self.disposition,
            "disposition_rationale": self.disposition_rationale,
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
    def from_dict(cls, data: dict[str, Any]) -> "SemanticFloorGateReport":
        return cls(
            schema=str(data.get("schema", "SemanticFloorGateReportV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", EXPERIMENT_ID)),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            floor_threshold=float(data.get("floor_threshold", DEFAULT_FLOOR_THRESHOLD)),
            atlas_hash=str(data.get("atlas_hash", "")),
            rows=tuple(FloorGateRunRow.from_dict(r) for r in data.get("rows", ())),
            n_runs=int(data.get("n_runs", 0)),
            n_families=int(data.get("n_families", 0)),
            full_role_weights=dict(data.get("full_role_weights", {})),
            real_balanced_accuracy=data.get("real_balanced_accuracy"),
            permutation_null=dict(data.get("permutation_null", {})),
            gate_hash=str(data.get("gate_hash", "")),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


def render_markdown(report: SemanticFloorGateReport) -> str:
    lines = [
        f"# SLM-223 (NCS0-03): SemanticFloorGateV1 fixture ({report.run_id})",
        "",
        f"**Matrix set:** `{report.matrix_set}`",
        f"**Version:** `{report.matrix_version}`",
        f"**Status:** {report.status}",
        f"**Claim class:** {report.claim_class}",
        f"**Floor threshold:** parse_rate < {report.floor_threshold}",
        f"**Atlas hash:** `{report.atlas_hash[:16]}...`",
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
        "## Summary",
        "",
        f"- Runs: {report.n_runs}",
        f"- Families: {report.n_families}",
        f"- LOFO real balanced accuracy: "
        f"{report.real_balanced_accuracy:.3f}" if report.real_balanced_accuracy is not None else "- LOFO real balanced accuracy: —",
        f"- Permutation-null mean: {report.permutation_null.get('mean')}",
        f"- Permutation draws evaluated: {report.permutation_null.get('draws')}",
        "",
        "## Per-run gate decisions (leave-one-family-out)",
        "",
        "| run_id | family | fold | mean α z | weighted α z | parse_rate | floor_label | gate_flag | correct |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(report.rows, key=lambda r: r.run_id):
        maz = f"{row.mean_alpha_z:.3f}" if row.mean_alpha_z is not None else "—"
        waz = f"{row.weighted_alpha_z:.3f}" if row.weighted_alpha_z is not None else "—"
        pr = f"{row.parse_rate:.3f}" if row.parse_rate is not None else "—"
        lines.append(
            f"| {row.run_id} | {row.family} | {row.fold} | {maz} | {waz} | {pr} | "
            f"{row.floor_label} | {row.gate_flag} | {row.correct} |"
        )
    lines += [
        "",
        "## Full-data role weights (reported only, not used for disposition)",
        "",
        "| role | weight (Pearson α z vs parse) |",
        "| --- | --- |",
    ]
    for role, weight in sorted(report.full_role_weights.items()):
        lines.append(f"| {role} | {weight:.3f} |")
    lines += [
        "",
        "## No-go for promotion",
        "",
        "This report is wiring/fixture evidence only. No checkpoint promotion, GPU "
        "train, or ship gate is claimed. `SemanticFloorGateV1` is a diagnostic "
        "pre-screen candidate; it does not replace full suite evaluation.",
        "",
    ]
    return "\n".join(lines)
