"""SLM-238 (GAE0-01): gradient-accumulation vs. physical-batch equivalence sweep.

Two single-run July-2026 telemetry probes touched ``grad_accum_steps`` and
flagged the same open question without answering it:

- ``docs/design/iter-telemetry-gradaccum2-20260715.md`` ran one CPU scratch
  step at ``batch_size=4, grad_accum=2`` and reported timing only, explicitly
  noting "quality and convergence still need a controlled multi-step
  comparison."
- ``docs/design/iter-telemetry-effective-batch-20260715.md`` compared one
  physical-batch-8 run against one batch-4/grad_accum-2 run at a single seed
  and found held-out NLL within ~0.1 nats, concluding this "supports
  effective batch size as the relevant comparison, not an
  accumulation-specific quality change" -- but a single seed cannot
  distinguish a genuine mechanism match from coincidence.

This module is the controlled follow-up, in the same spirit as SLM-227's
multi-seed extension of SLM-222's single-run Muon smoke test: it holds
initialization, learning rate, optimizer, total record set, and effective
batch size fixed, and asks across several seeds whether the real
``model_build`` train loop's ``grad_accum_steps=N`` (micro-batch ``B``)
training trajectory is a *close, unbiased* numerical stand-in for direct
training with ``batch_size=B*N, grad_accum=1`` -- not by inspecting the code,
but by running both through the unmodified
:func:`slm_training.harnesses.model_build.train_loop.train` and comparing
final training loss and the ``grad_accum`` / ``effective_batch_size``
telemetry fields it actually writes.

Both arms are stochastic in one respect the code does not control for: the
per-batch mask/corruption draw (``torch.rand`` in ``TwoTowerModel.
_mask_targets``) is consumed from the global RNG stream one call per batch,
so a single 8-record forward call and two sequential 4-record forward calls
draw *different* random mask realizations even from matched initial RNG
state. Exact bit-for-bit gradient equality is therefore not expected or
tested; this experiment instead asks whether the resulting loss trajectories
stay close (within a pre-registered tolerance) rather than diverging into
different training regimes.
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
    "CLOSE_RELATIVE_TOLERANCE",
    "ArmResult",
    "SeedComparison",
    "GradAccumEquivalenceReport",
    "render_markdown",
    "run_gradaccum_equivalence_sweep",
]

MATRIX_VERSION = "gae0-01-v1"
MATRIX_SET = "slm238_gradaccum_equivalence"
EXPERIMENT_ID = "slm238-gae0-01-gradaccum-equivalence"

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_STEPS = 40
DEFAULT_N_RECORDS = 8

# Pre-registered "close approximation" tolerance on the relative difference
# between the two arms' final training loss, chosen before running the sweep.
CLOSE_RELATIVE_TOLERANCE = 0.15

_HYPOTHESIS = (
    "At matched initialization, learning rate, optimizer, and total record "
    "set, training the real model_build train loop with grad_accum_steps=2 "
    "(micro-batch 4) reaches a final training loss within "
    f"{CLOSE_RELATIVE_TOLERANCE:.0%} relative difference of direct training "
    "with batch_size=8, grad_accum=1 (same effective batch size), across "
    "multiple seeds -- i.e. gradient accumulation is a close, unbiased "
    "numerical stand-in for a larger physical batch at fixture scale, not a "
    "systematically different training regime, and the grad_accum / "
    "effective_batch_size telemetry fields correctly report the accounting."
)

_FALSIFIER = (
    "The accum arm's final loss differs from the direct arm's by more than "
    f"{CLOSE_RELATIVE_TOLERANCE:.0%} relative in a consistent direction "
    "across a majority of seeds (a systematic bias, not stochastic mask-draw "
    "noise), or the accel.grad_accum / accel.effective_batch_size telemetry "
    "fields do not match the configured values, or either arm produces a "
    "non-finite loss."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: a tiny scratch-context TwoTower model "
    "overfitting 8 synthetic records for 40 optimizer steps says nothing "
    "about gradient-accumulation behavior on a real training corpus or at "
    "production model scale.",
    "The two arms are not expected to be bit-identical: mask/corruption "
    "randomness is drawn per forward call from the global RNG stream, so an "
    "8-record single call and two sequential 4-record calls consume that "
    "stream differently even from matched initial state. This experiment "
    "measures whether the resulting loss trajectories stay close, not "
    "whether they are numerically identical.",
    "Learning rate, weight decay, and optimizer (AdamW) are matched but not "
    "independently retuned per arm; a production comparison would also "
    "check gradient-norm distributions and multi-epoch generalization, not "
    "just final training loss on a fixture that only ever overfits.",
    "The CLOSE_RELATIVE_TOLERANCE=15% threshold was chosen before running "
    "the sweep and is a fixture-scale judgment call, not a derived "
    "statistical bound.",
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
        "version": "slm238-fixture",
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


def _read_last_row(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return {}
    last: dict[str, Any] = {}
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        last = json.loads(line)
    return last


@dataclass(frozen=True)
class ArmResult:
    """Per-arm, per-seed result."""

    arm: str = "direct"  # "direct" | "accum"
    seed: int = 0
    steps_completed: int = 0
    batch_size: int = 0
    grad_accum: int = 1
    effective_batch_size: int = 0
    first_loss: float | None = None
    last_loss: float | None = None
    finite_throughout: bool = True
    metadata_ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "seed": self.seed,
            "steps_completed": self.steps_completed,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "effective_batch_size": self.effective_batch_size,
            "first_loss": self.first_loss,
            "last_loss": self.last_loss,
            "finite_throughout": self.finite_throughout,
            "metadata_ok": self.metadata_ok,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArmResult":
        return cls(
            arm=str(data.get("arm", "direct")),
            seed=int(data.get("seed", 0)),
            steps_completed=int(data.get("steps_completed", 0)),
            batch_size=int(data.get("batch_size", 0)),
            grad_accum=int(data.get("grad_accum", 1)),
            effective_batch_size=int(data.get("effective_batch_size", 0)),
            first_loss=data.get("first_loss"),
            last_loss=data.get("last_loss"),
            finite_throughout=bool(data.get("finite_throughout", True)),
            metadata_ok=bool(data.get("metadata_ok", True)),
        )


@dataclass(frozen=True)
class SeedComparison:
    """Direct-batch vs. grad-accumulation comparison for a single seed."""

    seed: int
    direct: ArmResult
    accum: ArmResult
    accum_minus_direct_last_loss: float | None
    relative_diff: float | None
    winner: str  # "accum" | "direct" | "tie" | "unstable"
    close: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "direct": self.direct.to_dict(),
            "accum": self.accum.to_dict(),
            "accum_minus_direct_last_loss": self.accum_minus_direct_last_loss,
            "relative_diff": self.relative_diff,
            "winner": self.winner,
            "close": self.close,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeedComparison":
        return cls(
            seed=int(data.get("seed", 0)),
            direct=ArmResult.from_dict(data.get("direct", {})),
            accum=ArmResult.from_dict(data.get("accum", {})),
            accum_minus_direct_last_loss=data.get("accum_minus_direct_last_loss"),
            relative_diff=data.get("relative_diff"),
            winner=str(data.get("winner", "unstable")),
            close=bool(data.get("close", False)),
        )


@dataclass(frozen=True)
class GradAccumEquivalenceReport:
    """Fixture report for SLM-238."""

    schema: str = "GradAccumEquivalenceReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm238-gradaccum-equivalence"
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    steps: int = DEFAULT_STEPS
    n_records: int = DEFAULT_N_RECORDS
    close_relative_tolerance: float = CLOSE_RELATIVE_TOLERANCE
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    comparisons: tuple[SeedComparison, ...] = field(default_factory=tuple)
    accum_wins: int = 0
    direct_wins: int = 0
    ties: int = 0
    unstable_seeds: int = 0
    close_seeds: int = 0
    mean_relative_diff: float | None = None
    mean_delta: float | None = None
    stdev_delta: float | None = None
    all_finite: bool = True
    all_metadata_ok: bool = True
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
            "close_relative_tolerance": self.close_relative_tolerance,
            "seeds": list(self.seeds),
            "comparisons": [c.to_dict() for c in self.comparisons],
            "accum_wins": self.accum_wins,
            "direct_wins": self.direct_wins,
            "ties": self.ties,
            "unstable_seeds": self.unstable_seeds,
            "close_seeds": self.close_seeds,
            "mean_relative_diff": self.mean_relative_diff,
            "mean_delta": self.mean_delta,
            "stdev_delta": self.stdev_delta,
            "all_finite": self.all_finite,
            "all_metadata_ok": self.all_metadata_ok,
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
    def from_dict(cls, data: dict[str, Any]) -> "GradAccumEquivalenceReport":
        return cls(
            schema=str(data.get("schema", "GradAccumEquivalenceReportV1")),
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
            close_relative_tolerance=float(
                data.get("close_relative_tolerance", CLOSE_RELATIVE_TOLERANCE)
            ),
            seeds=tuple(data.get("seeds", DEFAULT_SEEDS)),
            comparisons=tuple(
                SeedComparison.from_dict(c) for c in data.get("comparisons", ())
            ),
            accum_wins=int(data.get("accum_wins", 0)),
            direct_wins=int(data.get("direct_wins", 0)),
            ties=int(data.get("ties", 0)),
            unstable_seeds=int(data.get("unstable_seeds", 0)),
            close_seeds=int(data.get("close_seeds", 0)),
            mean_relative_diff=data.get("mean_relative_diff"),
            mean_delta=data.get("mean_delta"),
            stdev_delta=data.get("stdev_delta"),
            all_finite=bool(data.get("all_finite", True)),
            all_metadata_ok=bool(data.get("all_metadata_ok", True)),
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
    arm: str,
    run_id: str,
    steps: int,
    batch_size: int,
    grad_accum: int,
    expected_effective_batch_size: int,
    seed: int,
) -> ArmResult:
    config = ModelBuildConfig(
        train_dir=train_dir,
        test_dir=None,
        suite="smoke",
        run_root=run_root,
        run_id=run_id,
        steps=steps,
        batch_size=batch_size,
        grad_accum_steps=grad_accum,
        lr=3e-4,
        optimizer_name="adamw",
        weight_decay=0.0,
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
    accel = summary.get("accel") or {}
    reported_grad_accum = int(accel.get("grad_accum", -1))
    reported_effective_batch_size = int(accel.get("effective_batch_size", -1))
    last_row = _read_last_row(config.run_dir)
    row_grad_accum = int(last_row.get("grad_accum", reported_grad_accum))
    row_batch_size = int(last_row.get("batch_size", -1))
    metadata_ok = (
        reported_grad_accum == grad_accum
        and reported_effective_batch_size == expected_effective_batch_size
        and row_grad_accum == grad_accum
        and row_batch_size == expected_effective_batch_size
    )
    return ArmResult(
        arm=arm,
        seed=seed,
        steps_completed=int(summary.get("steps") or 0),
        batch_size=batch_size,
        grad_accum=grad_accum,
        effective_batch_size=reported_effective_batch_size,
        first_loss=losses[0] if losses else summary.get("last_loss"),
        last_loss=summary.get("last_loss"),
        finite_throughout=finite_throughout,
        metadata_ok=metadata_ok,
    )


def run_gradaccum_equivalence_sweep(
    *,
    steps: int = DEFAULT_STEPS,
    n_records: int = DEFAULT_N_RECORDS,
    close_relative_tolerance: float = CLOSE_RELATIVE_TOLERANCE,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    run_id: str | None = None,
    out_dir: Path | str | None = None,
) -> GradAccumEquivalenceReport:
    """Run the SLM-238 gradient-accumulation equivalence sweep."""
    if n_records % 2 != 0:
        raise ValueError("n_records must be even (accum arm halves the batch)")
    records = _build_records(n_records)
    direct_batch_size = n_records
    accum_batch_size = n_records // 2
    accum_grad_accum = 2
    expected_effective_batch_size = direct_batch_size
    comparisons: list[SeedComparison] = []

    with tempfile.TemporaryDirectory(prefix="slm238-") as tmp:
        tmp_path = Path(tmp)
        train_dir = tmp_path / "train"
        run_root = tmp_path / "runs"
        _write_train_dir(train_dir, records)

        for seed in seeds:
            direct_arm = _run_arm(
                train_dir=train_dir,
                run_root=run_root,
                arm="direct",
                run_id=f"{EXPERIMENT_ID}-direct-{seed}",
                steps=steps,
                batch_size=direct_batch_size,
                grad_accum=1,
                expected_effective_batch_size=expected_effective_batch_size,
                seed=seed,
            )
            accum_arm = _run_arm(
                train_dir=train_dir,
                run_root=run_root,
                arm="accum",
                run_id=f"{EXPERIMENT_ID}-accum-{seed}",
                steps=steps,
                batch_size=accum_batch_size,
                grad_accum=accum_grad_accum,
                expected_effective_batch_size=expected_effective_batch_size,
                seed=seed,
            )

            stable = direct_arm.finite_throughout and accum_arm.finite_throughout
            delta: float | None = None
            relative_diff: float | None = None
            winner = "unstable"
            close = False
            if (
                stable
                and direct_arm.last_loss is not None
                and accum_arm.last_loss is not None
            ):
                delta = accum_arm.last_loss - direct_arm.last_loss
                denom = abs(direct_arm.last_loss) if direct_arm.last_loss else 1.0
                relative_diff = abs(delta) / max(denom, 1e-9)
                close = relative_diff <= close_relative_tolerance
                if abs(delta) < 1e-9:
                    winner = "tie"
                elif delta < 0:
                    winner = "accum"
                else:
                    winner = "direct"

            comparisons.append(
                SeedComparison(
                    seed=seed,
                    direct=direct_arm,
                    accum=accum_arm,
                    accum_minus_direct_last_loss=delta,
                    relative_diff=relative_diff,
                    winner=winner,
                    close=close,
                )
            )

    accum_wins = sum(1 for c in comparisons if c.winner == "accum")
    direct_wins = sum(1 for c in comparisons if c.winner == "direct")
    ties = sum(1 for c in comparisons if c.winner == "tie")
    unstable_seeds = sum(1 for c in comparisons if c.winner == "unstable")
    close_seeds = sum(1 for c in comparisons if c.close)
    all_finite = unstable_seeds == 0
    all_metadata_ok = all(
        c.direct.metadata_ok and c.accum.metadata_ok for c in comparisons
    )

    rel_diffs = [c.relative_diff for c in comparisons if c.relative_diff is not None]
    mean_relative_diff = statistics.fmean(rel_diffs) if rel_diffs else None

    deltas = [
        c.accum_minus_direct_last_loss
        for c in comparisons
        if c.accum_minus_direct_last_loss is not None
    ]
    mean_delta = statistics.fmean(deltas) if deltas else None
    stdev_delta = statistics.pstdev(deltas) if len(deltas) > 1 else (0.0 if deltas else None)

    n_decided = len(comparisons) - unstable_seeds

    if not all_finite:
        disposition = "unstable"
        rationale = (
            f"{unstable_seeds}/{len(comparisons)} seeds produced a non-finite loss in at "
            "least one arm; the equivalence question cannot be answered until that is fixed."
        )
    elif not all_metadata_ok:
        disposition = "metadata_gap"
        rationale = (
            "At least one seed's train_summary accel.grad_accum / "
            "accel.effective_batch_size fields (or the matching metrics.jsonl row) did not "
            "match the configured grad_accum_steps / batch_size*grad_accum -- the "
            "accumulation accounting contract itself is not trustworthy here, independent "
            "of loss closeness."
        )
    elif n_decided > 0 and close_seeds == n_decided:
        disposition = "close_approximation_confirmed"
        rationale = (
            f"All {close_seeds}/{n_decided} decided seeds stayed within the pre-registered "
            f"{close_relative_tolerance:.0%} relative-difference tolerance "
            f"(mean relative diff {mean_relative_diff:.4f}, mean delta {mean_delta:+.4f}). "
            "Gradient accumulation behaves as a close, unbiased stand-in for the equivalent "
            "physical batch at this fixture scale."
        )
    elif n_decided > 0 and (accum_wins == n_decided or direct_wins == n_decided):
        disposition = "consistent_direction_but_diverges"
        leader = "accum" if accum_wins == n_decided else "direct"
        rationale = (
            f"Every decided seed favored the {leader} arm, but "
            f"{n_decided - close_seeds}/{n_decided} seeds exceeded the "
            f"{close_relative_tolerance:.0%} tolerance (mean relative diff "
            f"{mean_relative_diff:.4f}, mean delta {mean_delta:+.4f}). A systematic, "
            "not merely noisy, gap between the two arms at this fixture scale."
        )
    elif n_decided == 0:
        disposition = "no_signal"
        rationale = "Every seed tied within numerical noise; no relative-difference signal to report."
    else:
        disposition = "inconsistent_and_diverges"
        rationale = (
            f"Wins were split ({accum_wins} accum / {direct_wins} direct) and "
            f"{n_decided - close_seeds}/{n_decided} seeds exceeded the "
            f"{close_relative_tolerance:.0%} tolerance (mean relative diff "
            f"{mean_relative_diff:.4f}). No reliable direction or closeness."
        )

    report = GradAccumEquivalenceReport(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        steps=steps,
        n_records=n_records,
        close_relative_tolerance=close_relative_tolerance,
        seeds=tuple(seeds),
        comparisons=tuple(comparisons),
        accum_wins=accum_wins,
        direct_wins=direct_wins,
        ties=ties,
        unstable_seeds=unstable_seeds,
        close_seeds=close_seeds,
        mean_relative_diff=mean_relative_diff,
        mean_delta=mean_delta,
        stdev_delta=stdev_delta,
        all_finite=all_finite,
        all_metadata_ok=all_metadata_ok,
        disposition=disposition,
        disposition_rationale=rationale,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm238_gradaccum_equivalence",
            "harness.experiments.slm227_muon_convergence",
            "harness.model_build.train",
            "model.twotower",
        ),
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(
            out_dir
            / f"iter-slm238-gae0-01-gradaccum-equivalence-{_today_yyyymmdd()}.json"
        )
    return report


def render_markdown(report: GradAccumEquivalenceReport) -> str:
    """Render a compact design note for the fixture."""
    lines = [
        f"# SLM-238 (GAE0-01): gradient-accumulation equivalence sweep ({report.run_id})",
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
        f"- optimizer steps per arm: {report.steps}",
        f"- records: {report.n_records}",
        f"- close-relative-difference tolerance: {report.close_relative_tolerance:.0%}",
        f"- seeds: {list(report.seeds)}",
        "",
        "## Per-seed results",
        "",
        "| seed | direct last_loss | accum last_loss | delta (accum-direct) | rel diff | close? | winner |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in report.comparisons:
        d_loss = f"{c.direct.last_loss:.4f}" if c.direct.last_loss is not None else "n/a"
        a_loss = f"{c.accum.last_loss:.4f}" if c.accum.last_loss is not None else "n/a"
        delta = (
            f"{c.accum_minus_direct_last_loss:+.4f}"
            if c.accum_minus_direct_last_loss is not None
            else "n/a"
        )
        rel = f"{c.relative_diff:.4f}" if c.relative_diff is not None else "n/a"
        lines.append(
            f"| {c.seed} | {d_loss} | {a_loss} | {delta} | {rel} | {c.close} | {c.winner} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- accum_wins: {report.accum_wins}",
            f"- direct_wins: {report.direct_wins}",
            f"- ties: {report.ties}",
            f"- unstable_seeds: {report.unstable_seeds}",
            f"- close_seeds (within tolerance): {report.close_seeds}/{len(report.comparisons)}",
            f"- mean_relative_diff: {report.mean_relative_diff}",
            f"- mean_delta (accum - direct): {report.mean_delta}",
            f"- stdev_delta: {report.stdev_delta}",
            f"- all_finite: {report.all_finite}",
            f"- all_metadata_ok (accel.grad_accum / accel.effective_batch_size match config): {report.all_metadata_ok}",
            "",
            "## Disposition",
            "",
            f"**{report.disposition}**",
            "",
            report.disposition_rationale,
            "",
            "## Go / no-go decision",
            "",
            (
                "**No-go for promotion; positive mechanism characterization.** This report is "
                "wiring/fixture evidence only over a tiny scratch-backend model and synthetic "
                "overfit data. No checkpoint, GPU train, or ship gate is claimed. A "
                "`close_approximation_confirmed` disposition supports treating grad_accum_steps "
                "as a faithful physical-batch stand-in at fixture scale for future scaling-ladder "
                "or memory-constrained runs; any other disposition means that substitution should "
                "not be assumed without a matched-batch control."
            ),
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_slm238_gradaccum_equivalence --mode plan-only",
            "python -m scripts.run_slm238_gradaccum_equivalence --mode fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    out = Path("docs/design")
    report = run_gradaccum_equivalence_sweep(out_dir=out)
    (
        out / f"iter-slm238-gae0-01-gradaccum-equivalence-{_today_yyyymmdd()}.md"
    ).write_text(render_markdown(report), encoding="utf-8")
    print(
        f"disposition={report.disposition} accum_wins={report.accum_wins} "
        f"direct_wins={report.direct_wins} ties={report.ties} "
        f"unstable={report.unstable_seeds} close_seeds={report.close_seeds} "
        f"mean_relative_diff={report.mean_relative_diff}"
    )
