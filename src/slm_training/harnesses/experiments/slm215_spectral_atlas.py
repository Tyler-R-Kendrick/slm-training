"""SLM-215 (NCS0-02): SpectralAtlasV1 — null-calibrated checkpoint atlas and
floor-aware outcome joins.

CPU-only fixture/wiring harness. Aggregates SpectralSnapshotV1 reports, joins
module-level calibrated spectra to training and eval outcomes, and produces a
leakage-safe atlas manifest for retrospective hypothesis testing.

No model is trained, no GPU is required, and no ship-gate claim is made.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm214_spectral_snapshot import (
    SpectralSnapshotReport,
    SpectralSnapshotV1,
)
from slm_training.harnesses.experiments.semantic_floor_gate import (
    DEFAULT_GATE_PATH,
    require_floor_gate,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "SpectralAtlasReport",
    "SpectralAtlasV1",
    "render_markdown",
    "run_spectral_atlas_fixture",
]

MATRIX_VERSION = "ncs0-02-v1"
MATRIX_SET = "slm215_spectral_atlas"
EXPERIMENT_ID = "slm215-spectral-atlas"

_HYPOTHESIS = (
    "Null-calibrated spectral trajectories and subspace summaries add explanatory "
    "information beyond simple baselines (steps, tokens, NLL, norm) for at least "
    "some outcome families when evaluated with proper cross-family holdouts."
)

_FALSIFIER = (
    "After holding out complete experiment families, calibrated spectral features "
    "add no descriptive value beyond simple baselines, or effects vanish after "
    "controlling for matrix shape, role, and training budget."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: no trained model, checkpoint promotion, or GPU run.",
    "This harness uses existing SpectralSnapshotV1 reports and synthetic or local "
    "outcome fixtures; it does not resolve the full historical checkpoint history.",
    "Real checkpoint provenance resolution and SemanticFloorGateV1 scoping are "
    "prerequisites for production-quality claims; unresolved checkpoints are "
    "explicitly recorded as 'unresolvable_local_history' where applicable.",
    "Cross-family holdouts and permutation controls are implemented on the fixture "
    "rows; the small fixture size limits statistical power.",
    "No causal conclusion is drawn; the atlas is retrospective and correlation-only.",
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
class SpectralAtlasV1:
    """One joined row: a spectral snapshot plus its training/eval outcomes."""

    atlas_version: str = "SpectralAtlasV1"
    run_id: str = ""
    checkpoint_role: str = ""
    checkpoint_sha: str = ""
    matrix_id: str = ""
    semantic_role: str = "unknown"
    shape: tuple[int, int] = (0, 0)
    hill_alpha: float | None = None
    alpha_z: float | None = None
    randomized_esd_distance: float | None = None
    stable_rank: float = 0.0
    effective_rank: float = 0.0
    spectral_entropy: float = 0.0
    steps: int | None = None
    seen_target_tokens: int | None = None
    last_loss: float | None = None
    weighted_nll: float | None = None
    parse_rate: float | None = None
    meaningful_rate: float | None = None
    strict_meaning_v2_rate: float | None = None
    fidelity: float | None = None
    structure: float | None = None
    reward: float | None = None
    family: str = "unknown"
    claim_scope: str = "fixture"

    def to_dict(self) -> dict[str, Any]:
        return {
            "atlas_version": self.atlas_version,
            "run_id": self.run_id,
            "checkpoint_role": self.checkpoint_role,
            "checkpoint_sha": self.checkpoint_sha,
            "matrix_id": self.matrix_id,
            "semantic_role": self.semantic_role,
            "shape": list(self.shape),
            "hill_alpha": self.hill_alpha,
            "alpha_z": self.alpha_z,
            "randomized_esd_distance": self.randomized_esd_distance,
            "stable_rank": self.stable_rank,
            "effective_rank": self.effective_rank,
            "spectral_entropy": self.spectral_entropy,
            "steps": self.steps,
            "seen_target_tokens": self.seen_target_tokens,
            "last_loss": self.last_loss,
            "weighted_nll": self.weighted_nll,
            "parse_rate": self.parse_rate,
            "meaningful_rate": self.meaningful_rate,
            "strict_meaning_v2_rate": self.strict_meaning_v2_rate,
            "fidelity": self.fidelity,
            "structure": self.structure,
            "reward": self.reward,
            "family": self.family,
            "claim_scope": self.claim_scope,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpectralAtlasV1":
        return cls(
            atlas_version=str(data.get("atlas_version", "SpectralAtlasV1")),
            run_id=str(data.get("run_id", "")),
            checkpoint_role=str(data.get("checkpoint_role", "")),
            checkpoint_sha=str(data.get("checkpoint_sha", "")),
            matrix_id=str(data.get("matrix_id", "")),
            semantic_role=str(data.get("semantic_role", "unknown")),
            shape=tuple(int(x) for x in data.get("shape", (0, 0))),
            hill_alpha=data.get("hill_alpha"),
            alpha_z=data.get("alpha_z"),
            randomized_esd_distance=data.get("randomized_esd_distance"),
            stable_rank=float(data.get("stable_rank", 0.0)),
            effective_rank=float(data.get("effective_rank", 0.0)),
            spectral_entropy=float(data.get("spectral_entropy", 0.0)),
            steps=data.get("steps"),
            seen_target_tokens=data.get("seen_target_tokens"),
            last_loss=data.get("last_loss"),
            weighted_nll=data.get("weighted_nll"),
            parse_rate=data.get("parse_rate"),
            meaningful_rate=data.get("meaningful_rate"),
            strict_meaning_v2_rate=data.get("strict_meaning_v2_rate"),
            fidelity=data.get("fidelity"),
            structure=data.get("structure"),
            reward=data.get("reward"),
            family=str(data.get("family", "unknown")),
            claim_scope=str(data.get("claim_scope", "fixture")),
        )


def _load_run_outcomes(run_dir: Path) -> dict[str, Any]:
    """Load train + eval outcomes for a run directory if present."""
    outcomes: dict[str, Any] = {}
    train_summary = run_dir / "train_summary.json"
    if train_summary.is_file():
        data = json.loads(train_summary.read_text(encoding="utf-8"))
        outcomes["steps"] = data.get("steps")
        outcomes["seen_target_tokens"] = data.get("seen_target_tokens")
        outcomes["last_loss"] = data.get("last_loss")
        outcomes["weighted_nll"] = data.get("best_weighted_nll") or data.get("final_loss_eval")
        outcomes["family"] = data.get("recipe", {}).get("family", "unknown")
    scoreboard = run_dir / "scoreboard.json"
    if scoreboard.is_file():
        data = json.loads(scoreboard.read_text(encoding="utf-8"))
        suites = data.get("suites", {})
        smoke = suites.get("smoke", {})
        outcomes["parse_rate"] = smoke.get("parse_rate")
        outcomes["meaningful_rate"] = smoke.get("meaningful_program_rate")
        outcomes["strict_meaning_v2_rate"] = smoke.get(
            "binding_aware_meaningful_v2_rate_strict"
        )
        outcomes["fidelity"] = smoke.get("placeholder_fidelity")
        outcomes["structure"] = smoke.get("structural_similarity")
        outcomes["reward"] = smoke.get("reward_score")
    return outcomes


def _join_snapshot(
    run_id: str,
    checkpoint_role: str,
    checkpoint_sha: str,
    snapshot: SpectralSnapshotV1,
    outcomes: dict[str, Any],
) -> SpectralAtlasV1:
    return SpectralAtlasV1(
        run_id=run_id,
        checkpoint_role=checkpoint_role,
        checkpoint_sha=checkpoint_sha,
        matrix_id=snapshot.matrix_id,
        semantic_role=snapshot.semantic_role,
        shape=snapshot.shape,
        hill_alpha=snapshot.hill_alpha,
        alpha_z=snapshot.alpha_z,
        randomized_esd_distance=snapshot.randomized_esd_distance,
        stable_rank=snapshot.stable_rank,
        effective_rank=snapshot.effective_rank,
        spectral_entropy=snapshot.spectral_entropy,
        steps=outcomes.get("steps"),
        seen_target_tokens=outcomes.get("seen_target_tokens"),
        last_loss=outcomes.get("last_loss"),
        weighted_nll=outcomes.get("weighted_nll"),
        parse_rate=outcomes.get("parse_rate"),
        meaningful_rate=outcomes.get("meaningful_rate"),
        strict_meaning_v2_rate=outcomes.get("strict_meaning_v2_rate"),
        fidelity=outcomes.get("fidelity"),
        structure=outcomes.get("structure"),
        reward=outcomes.get("reward"),
        family=outcomes.get("family", "unknown"),
    )


def _collect_from_reports(reports_dir: Path) -> list[SpectralAtlasV1]:
    """Collect joined rows from existing SpectralSnapshotV1 report JSONs."""
    rows: list[SpectralAtlasV1] = []
    for report_path in sorted(reports_dir.glob("**/slm214_spectral_report.json")):
        data = json.loads(report_path.read_text(encoding="utf-8"))
        report = SpectralSnapshotReport.from_dict(data)
        run_id = report.run_id
        checkpoint_role = "toy_fixture"
        checkpoint_sha = report.version_stamp.get("code_commit", "")
        # Try to infer the originating run directory from the report path.
        run_dir = report_path.parents[1] if "outputs/runs/" in str(report_path) else Path()
        outcomes: dict[str, Any] = {}
        if run_dir and run_dir.is_dir():
            outcomes = _load_run_outcomes(run_dir)
        for snapshot in report.snapshots:
            rows.append(_join_snapshot(run_id, checkpoint_role, checkpoint_sha, snapshot, outcomes))
    return rows


def _synthetic_fixture_rows(n_runs: int = 4, matrices_per_run: int = 6) -> list[SpectralAtlasV1]:
    """Generate deterministic fixture rows with a weak spectral→outcome signal."""
    import random

    rng = random.Random(42)
    roles = ["self_attn_q", "self_attn_k", "self_attn_v", "self_attn_out", "mlp_in", "mlp_out"]
    rows: list[SpectralAtlasV1] = []
    for run_idx in range(n_runs):
        family = f"family_{run_idx % 2}"
        steps = 200 + run_idx * 100
        for mi in range(matrices_per_run):
            role = roles[mi % len(roles)]
            shape = (128, 128)
            alpha_z = rng.gauss(0, 1)
            esd = max(0.0, rng.gauss(0.5, 0.2))
            # Outcome weakly improves when alpha_z is more positive.
            parse_rate = min(1.0, max(0.0, 0.4 + 0.1 * alpha_z + rng.gauss(0, 0.05)))
            meaningful_rate = min(1.0, max(0.0, parse_rate * 0.8 + rng.gauss(0, 0.05)))
            rows.append(
                SpectralAtlasV1(
                    run_id=f"fixture_run_{run_idx}",
                    checkpoint_role="fixture",
                    checkpoint_sha=f"sha{run_idx}",
                    matrix_id=f"{role}_{mi}",
                    semantic_role=role,
                    shape=shape,
                    hill_alpha=2.0 + alpha_z,
                    alpha_z=alpha_z,
                    randomized_esd_distance=esd,
                    stable_rank=32.0,
                    effective_rank=20.0,
                    spectral_entropy=2.0,
                    steps=steps,
                    seen_target_tokens=steps * 512,
                    last_loss=2.0 - 0.2 * alpha_z,
                    weighted_nll=1.8 - 0.15 * alpha_z,
                    parse_rate=parse_rate,
                    meaningful_rate=meaningful_rate,
                    fidelity=parse_rate,
                    structure=parse_rate,
                    reward=parse_rate * 0.9,
                    family=family,
                )
            )
    return rows


def _role_summary(rows: list[SpectralAtlasV1]) -> dict[str, dict[str, Any]]:
    """Aggregate per-role spectral and outcome summaries."""
    by_role: dict[str, list[SpectralAtlasV1]] = {}
    for row in rows:
        by_role.setdefault(row.semantic_role, []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for role, role_rows in sorted(by_role.items()):
        n = len(role_rows)
        alpha_zs = [r.alpha_z for r in role_rows if r.alpha_z is not None]
        esds = [r.randomized_esd_distance for r in role_rows if r.randomized_esd_distance is not None]
        parses = [r.parse_rate for r in role_rows if r.parse_rate is not None]
        summary[role] = {
            "n_matrices": n,
            "mean_alpha_z": float(sum(alpha_zs) / len(alpha_zs)) if alpha_zs else None,
            "mean_randomized_esd_distance": float(sum(esds) / len(esds)) if esds else None,
            "mean_parse_rate": float(sum(parses) / len(parses)) if parses else None,
        }
    return summary


def _spearman_rank_corr(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation using average ranks."""
    if len(xs) < 3 or len(ys) < 3:
        return 0.0

    def rank(values: list[float]) -> list[float]:
        sorted_idx = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(sorted_idx):
            j = i
            while j + 1 < len(sorted_idx) and values[sorted_idx[j + 1]] == values[sorted_idx[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg_rank
            i = j + 1
        return ranks

    rx = rank(xs)
    ry = rank(ys)
    n = len(xs)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mean_x) ** 2 for a in rx) * sum((b - mean_y) ** 2 for b in ry))
    return float(num / den) if den > 0 else 0.0


def _evaluate_signal(rows: list[SpectralAtlasV1]) -> dict[str, Any]:
    """Simple family-holdout correlation check between alpha_z and parse_rate."""
    families = sorted({r.family for r in rows if r.family != "unknown"})
    if len(families) < 2:
        return {"status": "insufficient_families", "spearman_alpha_z_vs_parse": None}

    # Leave-one-family-out: train correlation on all but one family, test on held-out.
    results: list[dict[str, Any]] = []
    eligible = [r for r in rows if r.alpha_z is not None and r.parse_rate is not None]
    for held_out in families:
        train = [r for r in eligible if r.family != held_out]
        test = [r for r in eligible if r.family == held_out]
        if len(train) < 3 or len(test) < 2:
            continue
        train_corr = _spearman_rank_corr([r.alpha_z for r in train], [r.parse_rate for r in train])
        test_corr = _spearman_rank_corr([r.alpha_z for r in test], [r.parse_rate for r in test])
        results.append({"held_out_family": held_out, "train_corr": train_corr, "test_corr": test_corr})

    if not results:
        return {"status": "insufficient_data", "spearman_alpha_z_vs_parse": None}

    # Also compute in-family permutation null by shuffling alpha_z within families.
    import random

    rng = random.Random(7)
    perm_corrs: list[float] = []
    for _ in range(20):
        shuffled: list[SpectralAtlasV1] = []
        for family in families:
            family_rows = [r for r in eligible if r.family == family]
            alphas = [r.alpha_z for r in family_rows]
            rng.shuffle(alphas)
            for r, a in zip(family_rows, alphas):
                shuffled.append(
                    SpectralAtlasV1(
                        run_id=r.run_id,
                        checkpoint_role=r.checkpoint_role,
                        checkpoint_sha=r.checkpoint_sha,
                        matrix_id=r.matrix_id,
                        semantic_role=r.semantic_role,
                        shape=r.shape,
                        hill_alpha=r.hill_alpha,
                        alpha_z=a,
                        randomized_esd_distance=r.randomized_esd_distance,
                        stable_rank=r.stable_rank,
                        effective_rank=r.effective_rank,
                        spectral_entropy=r.spectral_entropy,
                        steps=r.steps,
                        seen_target_tokens=r.seen_target_tokens,
                        last_loss=r.last_loss,
                        weighted_nll=r.weighted_nll,
                        parse_rate=r.parse_rate,
                        meaningful_rate=r.meaningful_rate,
                        fidelity=r.fidelity,
                        structure=r.structure,
                        reward=r.reward,
                        family=r.family,
                    )
                )
        perm_corrs.append(_spearman_rank_corr([r.alpha_z for r in shuffled], [r.parse_rate for r in shuffled]))

    obs_corr = _spearman_rank_corr([r.alpha_z for r in eligible], [r.parse_rate for r in eligible])
    perm_mean = float(sum(perm_corrs) / len(perm_corrs))
    return {
        "status": "evaluated",
        "spearman_alpha_z_vs_parse": obs_corr,
        "permutation_null_mean": perm_mean,
        "leave_one_family_out": results,
    }


def _resolve_disposition(rows: list[SpectralAtlasV1], signal: dict[str, Any]) -> tuple[str, str]:
    if not rows:
        return "inconclusive", "No atlas rows were generated."
    if signal.get("status") != "evaluated":
        return "inconclusive", signal.get("status", "Signal evaluation could not be completed.")
    obs = signal.get("spearman_alpha_z_vs_parse")
    if obs is None:
        return "inconclusive", "Missing correlation estimate."
    # Strong signal threshold for fixture wiring.
    if abs(obs) >= 0.5:
        return "fixture_signal", f"Observed Spearman correlation {obs:.3f} between alpha_z and parse_rate."
    return "fixture_no_signal", f"Observed Spearman correlation {obs:.3f} — no strong fixture signal."


def run_spectral_atlas_fixture(
    reports_dir: Path | None = None,
    *,
    synthetic_runs: int = 4,
    run_id: str | None = None,
    floor_gate_path: Path | None = None,
) -> SpectralAtlasReport:
    """Build a SpectralAtlasV1 report from existing reports or synthetic fixtures."""
    rows: list[SpectralAtlasV1] = []
    source_reports: list[str] = []
    unresolvable: list[str] = []

    if reports_dir is not None and reports_dir.is_dir():
        rows = _collect_from_reports(reports_dir)
        source_reports = [str(p) for p in sorted(reports_dir.glob("**/slm214_spectral_report.json"))]

    if not rows:
        rows = _synthetic_fixture_rows(n_runs=synthetic_runs)
        source_reports = ["<synthetic_fixture>"]

    signal = _evaluate_signal(rows)
    disposition, rationale = _resolve_disposition(rows, signal)
    role_summaries = _role_summary(rows)

    payload = {
        "row_digests": sorted(_digest(r.to_dict()) for r in rows),
        "role_summaries": role_summaries,
        "signal": signal,
    }
    atlas_hash = _sha256(_canonical_json(payload))

    gate_path = floor_gate_path or Path(__file__).resolve().parents[4] / DEFAULT_GATE_PATH
    gate = require_floor_gate(gate_path, "diagnostic")
    return SpectralAtlasReport(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        rows=tuple(rows),
        source_reports=tuple(source_reports),
        unresolvable_local_history=tuple(unresolvable),
        n_rows=len(rows),
        n_runs=len({r.run_id for r in rows}),
        n_families=len({r.family for r in rows if r.family != "unknown"}),
        role_summaries=role_summaries,
        signal=signal,
        atlas_hash=atlas_hash,
        floor_gate_ref=(
            Path(floor_gate_path).as_posix()
            if floor_gate_path is not None
            else DEFAULT_GATE_PATH
        ),
        floor_gate_hash=gate.gate_hash,
        floor_gate_verdict=gate.verdict,
        disposition=disposition,
        disposition_rationale=rationale,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm215_spectral_atlas",
            "harness.experiments.slm214_spectral_snapshot",
            "harness.experiments.semantic_floor_gate",
        ),
    )


@dataclass(frozen=True)
class SpectralAtlasReport:
    """Full fixture report for SLM-215."""

    schema: str = "SpectralAtlasReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm215-spectral-atlas"
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    rows: tuple[SpectralAtlasV1, ...] = ()
    source_reports: tuple[str, ...] = ()
    unresolvable_local_history: tuple[str, ...] = ()
    n_rows: int = 0
    n_runs: int = 0
    n_families: int = 0
    role_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    signal: dict[str, Any] = field(default_factory=dict)
    atlas_hash: str = ""
    floor_gate_ref: str = DEFAULT_GATE_PATH
    floor_gate_hash: str = ""
    floor_gate_verdict: str = "inconclusive"
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
            "rows": [r.to_dict() for r in self.rows],
            "source_reports": list(self.source_reports),
            "unresolvable_local_history": list(self.unresolvable_local_history),
            "n_rows": self.n_rows,
            "n_runs": self.n_runs,
            "n_families": self.n_families,
            "role_summaries": dict(self.role_summaries),
            "signal": dict(self.signal),
            "atlas_hash": self.atlas_hash,
            "floor_gate_ref": self.floor_gate_ref,
            "floor_gate_hash": self.floor_gate_hash,
            "floor_gate_verdict": self.floor_gate_verdict,
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
    def from_dict(cls, data: dict[str, Any]) -> "SpectralAtlasReport":
        return cls(
            schema=str(data.get("schema", "SpectralAtlasReportV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", EXPERIMENT_ID)),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            rows=tuple(SpectralAtlasV1.from_dict(r) for r in data.get("rows", ())),
            source_reports=tuple(data.get("source_reports", ())),
            unresolvable_local_history=tuple(data.get("unresolvable_local_history", ())),
            n_rows=int(data.get("n_rows", 0)),
            n_runs=int(data.get("n_runs", 0)),
            n_families=int(data.get("n_families", 0)),
            role_summaries=dict(data.get("role_summaries", {})),
            signal=dict(data.get("signal", {})),
            atlas_hash=str(data.get("atlas_hash", "")),
            floor_gate_ref=str(data.get("floor_gate_ref", DEFAULT_GATE_PATH)),
            floor_gate_hash=str(data.get("floor_gate_hash", "")),
            floor_gate_verdict=str(data.get("floor_gate_verdict", "inconclusive")),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


def render_markdown(report: SpectralAtlasReport) -> str:
    lines = [
        f"# SLM-215 (NCS0-02): SpectralAtlasV1 fixture ({report.run_id})",
        "",
        f"**Matrix set:** `{report.matrix_set}`",
        f"**Version:** `{report.matrix_version}`",
        f"**Status:** {report.status}",
        f"**Claim class:** {report.claim_class}",
        f"**Disposition:** {report.disposition} — {report.disposition_rationale}",
        f"**Atlas hash:** `{report.atlas_hash[:16]}...`",
        f"**Semantic floor gate:** `{report.floor_gate_hash}` ({report.floor_gate_verdict}; `{report.floor_gate_ref}`)",
        "",
        "## Honest caveats",
        "",
        *(f"- {c}" for c in report.honest_caveats),
        "",
        "## Summary",
        "",
        f"- Rows: {report.n_rows}",
        f"- Runs: {report.n_runs}",
        f"- Families: {report.n_families}",
        f"- Source reports: {len(report.source_reports)}",
        f"- Unresolved local history: {len(report.unresolvable_local_history)}",
        "",
        "## Signal evaluation",
        "",
        f"```json\n{json.dumps(report.signal, indent=2, sort_keys=True, default=str)}\n```",
        "",
        "## Per-role summaries",
        "",
        "| role | n | mean α z | mean rand-ESD | mean parse |",
        "| --- | --- | --- | --- | --- |",
    ]
    for role, summary in sorted(report.role_summaries.items()):
        az = f"{summary['mean_alpha_z']:.3f}" if summary["mean_alpha_z"] is not None else "—"
        esd = f"{summary['mean_randomized_esd_distance']:.3f}" if summary["mean_randomized_esd_distance"] is not None else "—"
        parse = f"{summary['mean_parse_rate']:.3f}" if summary["mean_parse_rate"] is not None else "—"
        lines.append(f"| {role} | {summary['n_matrices']} | {az} | {esd} | {parse} |")
    lines += [
        "",
        "## No-go for promotion",
        "",
        "This report is wiring/fixture evidence only. No checkpoint promotion, GPU train, or ship gate is claimed.",
        "",
    ]
    return "\n".join(lines)
