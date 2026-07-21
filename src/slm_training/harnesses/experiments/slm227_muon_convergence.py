"""SLM-227 (NCS2-04): Muon/AdamW hybrid convergence-direction sweep.

SLM-222 wired the Muon/AdamW hybrid optimizer (partitioning, checkpoint
fingerprint, fail-closed cross-optimizer resume) and ran a 2-step,
single-record smoke test whose own honest caveats said "no convergence, final
loss, or downstream eval metrics" conclusion could be drawn. This module
answers the narrow, CPU-only follow-up question SLM-222 explicitly left open:
holding initialization, learning rate, and data fixed, does the Muon arm's
final training loss move in a *consistent direction* relative to the AdamW
arm once the run is long enough (more steps, more records, multiple seeds)
for the two update rules to diverge?

This is still wiring/fixture evidence: a tiny scratch-backend TwoTower model
overfitting a handful of synthetic records for a few dozen steps says nothing
about downstream OpenUI quality. The full O0-O4 matched, capacity- and
data-matched AdamW-vs-Muon campaign (SLM-222's stated future work) still
requires local E224+ checkpoints and GPU time. What *is* new here relative to
SLM-222: a real (not hand-typed) multi-seed loss-trajectory comparison, with
an explicit falsifier about direction consistency rather than "training runs
without NaN".
"""

from __future__ import annotations

import hashlib
import json
import statistics
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.model_build import train
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "DEFAULT_SEEDS",
    "DEFAULT_STEPS",
    "DEFAULT_N_RECORDS",
    "SeedArmResult",
    "SeedComparison",
    "MuonConvergenceReport",
    "render_markdown",
    "run_muon_convergence_sweep",
]

MATRIX_VERSION = "ncs2-04-v1"
MATRIX_SET = "slm227_muon_convergence"
EXPERIMENT_ID = "slm227-muon-convergence"

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_STEPS = 40
DEFAULT_N_RECORDS = 4

_HYPOTHESIS = (
    "At matched initialization, learning rate, data, and step budget, the "
    "Muon/AdamW hybrid optimizer's final training loss moves in a consistent "
    "direction relative to plain AdamW across seeds, once run long enough "
    "(more steps and records than SLM-222's 2-step single-record smoke test) "
    "for the orthogonalized-momentum update to diverge from AdamW."
)

_FALSIFIER = (
    "The Muon arm's final loss is not consistently lower (or higher) than "
    "AdamW's across the swept seeds (no seed-majority direction), or either "
    "arm produces a non-finite loss or non-finite parameters at any seed."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: no trained model, checkpoint promotion, GPU run, or ship-gate claim.",
    "The full O0-O4 matched AdamW-vs-Muon campaign (capacity- and data-matched, with spectral LR control) "
    "still requires local E224+ checkpoints and dedicated GPU time and remains future work.",
    "The fixture uses a tiny scratch-context model overfitting a handful of synthetic records; a consistent "
    "direction here is evidence about this optimizer pair's fixture-scale dynamics, not about downstream "
    "OpenUI quality, generalization, or which optimizer to ship.",
    "Learning rate is matched but not tuned per optimizer; a real campaign would sweep LR separately for "
    "each arm before comparing convergence.",
)

_HERO = 'root = Stack([hero], "column")\nhero_title = TextContent(":hero.title")\nhero_body = TextContent(":hero.body")\nhero = Card([hero_title, hero_body])'


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_records(n_records: int) -> list[ExampleRecord]:
    return [
        ExampleRecord(id=f"r{i}", prompt=f"Hero variant {i}", openui=_HERO, split="train")
        for i in range(n_records)
    ]


def _write_train_dir(path: Path, records: list[ExampleRecord]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    records_path = path / "records.jsonl"
    write_jsonl(records_path, records)
    content = records_path.read_bytes()
    manifest = {
        "version": "slm227-fixture",
        "kind": "train",
        "records": str(records_path),
        "record_count": len(records),
        "content_fingerprint": hashlib.sha256(content).hexdigest(),
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _read_loss_trajectory(run_dir: Path) -> list[float]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    losses: list[float] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "loss" in row:
            losses.append(float(row["loss"]))
    return losses


@dataclass(frozen=True)
class SeedArmResult:
    """Per-optimizer, per-seed arm result."""

    optimizer_name: str = "adamw"
    seed: int = 0
    steps_completed: int = 0
    first_loss: float | None = None
    last_loss: float | None = None
    finite_throughout: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer_name": self.optimizer_name,
            "seed": self.seed,
            "steps_completed": self.steps_completed,
            "first_loss": self.first_loss,
            "last_loss": self.last_loss,
            "finite_throughout": self.finite_throughout,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeedArmResult":
        return cls(
            optimizer_name=str(data.get("optimizer_name", "adamw")),
            seed=int(data.get("seed", 0)),
            steps_completed=int(data.get("steps_completed", 0)),
            first_loss=data.get("first_loss"),
            last_loss=data.get("last_loss"),
            finite_throughout=bool(data.get("finite_throughout", True)),
        )


@dataclass(frozen=True)
class SeedComparison:
    """Muon vs AdamW comparison for a single seed."""

    seed: int
    adamw: SeedArmResult
    muon: SeedArmResult
    muon_minus_adamw_last_loss: float | None
    winner: str  # "muon" | "adamw" | "tie" | "unstable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "adamw": self.adamw.to_dict(),
            "muon": self.muon.to_dict(),
            "muon_minus_adamw_last_loss": self.muon_minus_adamw_last_loss,
            "winner": self.winner,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeedComparison":
        return cls(
            seed=int(data.get("seed", 0)),
            adamw=SeedArmResult.from_dict(data.get("adamw", {})),
            muon=SeedArmResult.from_dict(data.get("muon", {})),
            muon_minus_adamw_last_loss=data.get("muon_minus_adamw_last_loss"),
            winner=str(data.get("winner", "unstable")),
        )


@dataclass(frozen=True)
class MuonConvergenceReport:
    """Fixture report for SLM-227."""

    schema: str = "MuonConvergenceReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm227-muon-convergence"
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    steps: int = DEFAULT_STEPS
    n_records: int = DEFAULT_N_RECORDS
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    comparisons: tuple[SeedComparison, ...] = field(default_factory=tuple)
    muon_wins: int = 0
    adamw_wins: int = 0
    ties: int = 0
    unstable_seeds: int = 0
    mean_delta: float | None = None
    stdev_delta: float | None = None
    all_finite: bool = True
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
            "steps": self.steps,
            "n_records": self.n_records,
            "seeds": list(self.seeds),
            "comparisons": [c.to_dict() for c in self.comparisons],
            "muon_wins": self.muon_wins,
            "adamw_wins": self.adamw_wins,
            "ties": self.ties,
            "unstable_seeds": self.unstable_seeds,
            "mean_delta": self.mean_delta,
            "stdev_delta": self.stdev_delta,
            "all_finite": self.all_finite,
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
    def from_dict(cls, data: dict[str, Any]) -> "MuonConvergenceReport":
        return cls(
            schema=str(data.get("schema", "MuonConvergenceReportV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", EXPERIMENT_ID)),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            steps=int(data.get("steps", DEFAULT_STEPS)),
            n_records=int(data.get("n_records", DEFAULT_N_RECORDS)),
            seeds=tuple(data.get("seeds", DEFAULT_SEEDS)),
            comparisons=tuple(SeedComparison.from_dict(c) for c in data.get("comparisons", ())),
            muon_wins=int(data.get("muon_wins", 0)),
            adamw_wins=int(data.get("adamw_wins", 0)),
            ties=int(data.get("ties", 0)),
            unstable_seeds=int(data.get("unstable_seeds", 0)),
            mean_delta=data.get("mean_delta"),
            stdev_delta=data.get("stdev_delta"),
            all_finite=bool(data.get("all_finite", True)),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


def _run_arm(
    *,
    train_dir: Path,
    run_root: Path,
    optimizer_name: str,
    run_id: str,
    steps: int,
    batch_size: int,
    seed: int,
) -> SeedArmResult:
    config = ModelBuildConfig(
        train_dir=train_dir,
        test_dir=None,
        suite="smoke",
        run_root=run_root,
        run_id=run_id,
        steps=steps,
        batch_size=batch_size,
        lr=3e-4,
        optimizer_name=optimizer_name,
        muon_lr=3e-4,
        adamw_lr=3e-4,
        weight_decay=0.0,
        muon_momentum=0.9,
        muon_nesterov=False,
        muon_ns_steps=5,
        device="cpu",
        model_name="twotower",
        d_model=32,
        n_heads=4,
        context_layers=1,
        denoiser_layers=1,
        context_backend="scratch",
        denoiser_backend="scratch",
        grammar_constrained=False,
        full_state_checkpoint=False,
        sync_checkpoints=False,
        seed=seed,
    )
    summary = train(config)
    losses = _read_loss_trajectory(config.run_dir)
    finite_throughout = bool(losses) and all(
        torch.isfinite(torch.tensor(float(v))) for v in losses
    )
    return SeedArmResult(
        optimizer_name=optimizer_name,
        seed=seed,
        steps_completed=int(summary.get("steps") or 0),
        first_loss=losses[0] if losses else summary.get("last_loss"),
        last_loss=summary.get("last_loss"),
        finite_throughout=finite_throughout,
    )


def run_muon_convergence_sweep(
    *,
    steps: int = DEFAULT_STEPS,
    n_records: int = DEFAULT_N_RECORDS,
    batch_size: int = 2,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    run_id: str | None = None,
    out_dir: Path | str | None = None,
) -> MuonConvergenceReport:
    """Run the SLM-227 Muon/AdamW convergence-direction sweep."""
    records = _build_records(n_records)
    comparisons: list[SeedComparison] = []

    with tempfile.TemporaryDirectory(prefix="slm227-") as tmp:
        tmp_path = Path(tmp)
        train_dir = tmp_path / "train"
        run_root = tmp_path / "runs"
        _write_train_dir(train_dir, records)

        for seed in seeds:
            adamw_arm = _run_arm(
                train_dir=train_dir,
                run_root=run_root,
                optimizer_name="adamw",
                run_id=f"{EXPERIMENT_ID}-adamw-{seed}",
                steps=steps,
                batch_size=batch_size,
                seed=seed,
            )
            muon_arm = _run_arm(
                train_dir=train_dir,
                run_root=run_root,
                optimizer_name="muon_hybrid",
                run_id=f"{EXPERIMENT_ID}-muon-{seed}",
                steps=steps,
                batch_size=batch_size,
                seed=seed,
            )

            stable = adamw_arm.finite_throughout and muon_arm.finite_throughout
            delta: float | None = None
            winner = "unstable"
            if stable and adamw_arm.last_loss is not None and muon_arm.last_loss is not None:
                delta = muon_arm.last_loss - adamw_arm.last_loss
                if abs(delta) < 1e-9:
                    winner = "tie"
                elif delta < 0:
                    winner = "muon"
                else:
                    winner = "adamw"

            comparisons.append(
                SeedComparison(
                    seed=seed,
                    adamw=adamw_arm,
                    muon=muon_arm,
                    muon_minus_adamw_last_loss=delta,
                    winner=winner,
                )
            )

    muon_wins = sum(1 for c in comparisons if c.winner == "muon")
    adamw_wins = sum(1 for c in comparisons if c.winner == "adamw")
    ties = sum(1 for c in comparisons if c.winner == "tie")
    unstable_seeds = sum(1 for c in comparisons if c.winner == "unstable")
    all_finite = unstable_seeds == 0

    deltas = [c.muon_minus_adamw_last_loss for c in comparisons if c.muon_minus_adamw_last_loss is not None]
    mean_delta = statistics.fmean(deltas) if deltas else None
    stdev_delta = statistics.pstdev(deltas) if len(deltas) > 1 else (0.0 if deltas else None)

    n_decided = muon_wins + adamw_wins
    if not all_finite:
        disposition = "unstable"
        rationale = (
            f"{unstable_seeds}/{len(comparisons)} seeds produced a non-finite loss in at least one "
            "arm; the comparison cannot speak to convergence direction until that is fixed."
        )
    elif n_decided == 0:
        disposition = "no_signal"
        rationale = "Every seed tied within numerical noise; no direction to report."
    elif muon_wins == n_decided:
        disposition = "consistent_muon_lower_loss"
        rationale = (
            f"Muon's final loss was lower than AdamW's in all {muon_wins}/{n_decided} decided seeds "
            f"(mean delta {mean_delta:+.4f})."
        )
    elif adamw_wins == n_decided:
        disposition = "consistent_adamw_lower_loss"
        rationale = (
            f"AdamW's final loss was lower than Muon's in all {adamw_wins}/{n_decided} decided seeds "
            f"(mean delta {mean_delta:+.4f})."
        )
    elif muon_wins > adamw_wins:
        disposition = "majority_muon_lower_loss"
        rationale = (
            f"Muon's final loss was lower in a majority of seeds ({muon_wins}/{n_decided}), but not "
            f"all; direction is not fully consistent (mean delta {mean_delta:+.4f})."
        )
    elif adamw_wins > muon_wins:
        disposition = "majority_adamw_lower_loss"
        rationale = (
            f"AdamW's final loss was lower in a majority of seeds ({adamw_wins}/{n_decided}), but not "
            f"all; direction is not fully consistent (mean delta {mean_delta:+.4f})."
        )
    else:
        disposition = "mixed_no_signal"
        rationale = (
            f"Wins were split evenly ({muon_wins} muon / {adamw_wins} adamw); no consistent direction "
            f"(mean delta {mean_delta:+.4f})."
        )

    report = MuonConvergenceReport(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        steps=steps,
        n_records=n_records,
        seeds=tuple(seeds),
        comparisons=tuple(comparisons),
        muon_wins=muon_wins,
        adamw_wins=adamw_wins,
        ties=ties,
        unstable_seeds=unstable_seeds,
        mean_delta=mean_delta,
        stdev_delta=stdev_delta,
        all_finite=all_finite,
        disposition=disposition,
        disposition_rationale=rationale,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm227_muon_convergence",
            "harness.experiments.slm222_muon_baseline",
            "harness.model_build.train",
            "model.twotower",
        ),
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(out_dir / f"iter-slm227-muon-convergence-{_today_yyyymmdd()}.json")
    return report


def render_markdown(report: MuonConvergenceReport) -> str:
    """Render a compact design note for the fixture."""
    lines = [
        f"# SLM-227 (NCS2-04): Muon/AdamW convergence-direction sweep ({report.run_id})",
        "",
        f"**Matrix set:** `{report.matrix_set}`",
        f"**Version:** `{report.matrix_version}`",
        f"**Status:** {report.status}",
        f"**Claim class:** {report.claim_class}",
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
        "## Sweep",
        "",
        f"- steps per arm: {report.steps}",
        f"- records: {report.n_records}",
        f"- seeds: {list(report.seeds)}",
        "",
        "## Per-seed results",
        "",
        "| seed | adamw last_loss | muon last_loss | muon - adamw | winner |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in report.comparisons:
        a_loss = f"{c.adamw.last_loss:.4f}" if c.adamw.last_loss is not None else "n/a"
        m_loss = f"{c.muon.last_loss:.4f}" if c.muon.last_loss is not None else "n/a"
        delta = f"{c.muon_minus_adamw_last_loss:+.4f}" if c.muon_minus_adamw_last_loss is not None else "n/a"
        lines.append(f"| {c.seed} | {a_loss} | {m_loss} | {delta} | {c.winner} |")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- muon_wins: {report.muon_wins}",
            f"- adamw_wins: {report.adamw_wins}",
            f"- ties: {report.ties}",
            f"- unstable_seeds: {report.unstable_seeds}",
            f"- mean_delta (muon - adamw): {report.mean_delta}",
            f"- stdev_delta: {report.stdev_delta}",
            f"- all_finite: {report.all_finite}",
            "",
            "## Disposition",
            "",
            f"**{report.disposition}**",
            "",
            report.disposition_rationale,
            "",
            "## Go / no-go decision",
            "",
            "**No-go for promotion.** This report is wiring/fixture evidence only over a tiny "
            "scratch-backend model and synthetic overfit data. No checkpoint, GPU train, or ship "
            "gate is claimed, and a consistent per-seed loss direction here does not establish that "
            "either optimizer is better for real OpenUI training.",
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_slm227_muon_convergence --mode plan-only",
            "python -m scripts.run_slm227_muon_convergence --mode fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    out = Path("docs/design")
    report = run_muon_convergence_sweep(out_dir=out)
    (out / f"iter-slm227-muon-convergence-{_today_yyyymmdd()}.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(
        f"disposition={report.disposition} muon_wins={report.muon_wins} "
        f"adamw_wins={report.adamw_wins} ties={report.ties} "
        f"unstable={report.unstable_seeds} mean_delta={report.mean_delta}"
    )
