"""SLM-240 (LRS0-01): learning-rate schedule gap probe.

Every fixture harness in this session's Muon lineage (SLM-222, SLM-227) held
the learning rate fixed and compared *optimizers*, never asking whether the
canonical ``model_build`` train loop applies any learning-rate *schedule* at
all. Reading ``harnesses/model_build/config.py`` and ``train_loop.py`` end to
end turns up no warmup/decay knob on ``ModelBuildConfig`` and no
``torch.optim.lr_scheduler`` import or ``.step()`` call anywhere in
``train_loop.py`` -- the optimizer is constructed once with a static
``config.lr`` (or ``muon_lr``/``adamw_lr`` per group) and never adjusted
again. The per-step ``metrics.jsonl`` row also omits the applied learning
rate entirely (only ``step``/``loss``/``batch_size``/... are written), so
even if a schedule existed nothing downstream could see it.

This module tests that reading, not by re-reading the source again, but by
instrumenting the *live* optimizer during real ``train()`` runs: it
monkeypatches ``torch.optim.AdamW.step`` and ``MuonHybrid.step`` to record a
snapshot of every parameter group's ``lr`` immediately before each call,
then restores the original methods when the run completes. The spy never
alters control flow -- it always delegates to the original bound method --
so this is instrumentation, not a change to optimizer behavior, and no
production file is edited.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.model_build import train
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.optimizers.muon import MuonHybrid
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "DEFAULT_SEEDS",
    "DEFAULT_STEPS",
    "DEFAULT_N_RECORDS",
    "DEFAULT_OPTIMIZERS",
    "ArmResult",
    "LrScheduleGapReport",
    "render_markdown",
    "run_lr_schedule_gap_probe",
]

MATRIX_VERSION = "lrs0-01-v1"
MATRIX_SET = "slm240_lr_schedule_gap"
EXPERIMENT_ID = "slm240-lrs0-01-lr-schedule-gap"

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
DEFAULT_STEPS = 20
DEFAULT_N_RECORDS = 4
DEFAULT_OPTIMIZERS: tuple[str, ...] = ("adamw", "muon_hybrid")

# Distinct, easy-to-tell-apart configured rates so a mixed-up group or a
# silently-defaulted lr would show up as a mismatch rather than hiding
# behind a coincidental equality.
_BASE_LR = 3e-4
_MUON_LR = 5e-4
_ADAMW_LR = 1e-4

_HYPOTHESIS = (
    "The real model_build train loop (harnesses/model_build/train_loop.py::"
    "train) applies a bit-identical, constant learning rate to every "
    "optimizer parameter group for the entire duration of a run, for both "
    "supported optimizers (adamw, muon_hybrid), with no warmup ramp-up and "
    "no decay -- and the per-step metrics.jsonl row never records the "
    "applied learning rate, so no existing telemetry could detect a "
    "schedule even if one existed."
)

_FALSIFIER = (
    "Any optimizer.step() call recorded during a >=10-step training run "
    "reports a parameter-group lr that differs from the group's first "
    "recorded lr (i.e. some schedule already governs it in this code path), "
    "or a first recorded lr does not match the configured lr/muon_lr/"
    "adamw_lr value, or any metrics.jsonl row for the run contains an "
    "'lr' or 'learning_rate' key."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: a tiny scratch-backend TwoTower model "
    "training on 4 synthetic records for 20 optimizer steps says nothing "
    "about whether a schedule *should* exist or what it would do to real "
    "convergence -- this probe only asks whether one exists today.",
    "Instrumentation is a monkeypatch spy on torch.optim.AdamW.step and "
    "MuonHybrid.step, scoped to a try/finally around a single train() call "
    "and always delegating to the original method; no production file is "
    "edited and no optimizer behavior is altered, but a spy is still an "
    "external observation mechanism rather than a first-class API, so it "
    "would need to be re-verified if either optimizer's step() signature "
    "changes.",
    "This finding is about the current absence of a mechanism, not a claim "
    "that a schedule would help or hurt convergence at any scale; it also "
    "does not cover the causal-LM/HF Trainer track (models/"
    "causal_lm_openui.py), which does configure lr_scheduler_type and "
    "warmup_ratio through the standard HF Trainer.",
    "Only two optimizer configurations (adamw, muon_hybrid) and three seeds "
    "were probed; this does not rule out a schedule being applied only "
    "under some other config combination this probe did not exercise.",
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
    manifest = {
        "version": "slm240-fixture",
        "kind": "train",
        "records": str(records_path),
        "record_count": len(records),
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _read_metrics_rows(run_dir: Path) -> list[dict[str, Any]]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _snapshot_param_groups(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    return [
        {
            "optimizer": str(group.get("optimizer", "adamw")),
            "lr": float(group["lr"]),
        }
        for group in optimizer.param_groups
    ]


def _run_instrumented(config: ModelBuildConfig) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    """Run the real train() loop while spying on live optimizer.step() calls.

    Always delegates to the original bound method; only records a snapshot
    of each parameter group's ``lr`` immediately before the call.
    """
    snapshots: list[list[dict[str, Any]]] = []
    orig_adamw_step = torch.optim.AdamW.step
    orig_muon_step = MuonHybrid.step

    def spy_adamw_step(self: torch.optim.AdamW, *args: Any, **kwargs: Any) -> Any:
        snapshots.append(_snapshot_param_groups(self))
        return orig_adamw_step(self, *args, **kwargs)

    def spy_muon_step(self: MuonHybrid, *args: Any, **kwargs: Any) -> Any:
        snapshots.append(_snapshot_param_groups(self))
        return orig_muon_step(self, *args, **kwargs)

    torch.optim.AdamW.step = spy_adamw_step  # type: ignore[method-assign]
    MuonHybrid.step = spy_muon_step  # type: ignore[method-assign]
    try:
        summary = train(config)
    finally:
        torch.optim.AdamW.step = orig_adamw_step  # type: ignore[method-assign]
        MuonHybrid.step = orig_muon_step  # type: ignore[method-assign]
    return summary, snapshots


@dataclass(frozen=True)
class ArmResult:
    """Per (optimizer, seed) result."""

    optimizer_name: str = "adamw"
    seed: int = 0
    steps_recorded: int = 0
    configured_lrs: dict[str, float] = field(default_factory=dict)
    first_snapshot: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    lr_constant: bool = True
    lr_matches_config: bool = True
    metrics_logs_lr: bool = False
    finite_throughout: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer_name": self.optimizer_name,
            "seed": self.seed,
            "steps_recorded": self.steps_recorded,
            "configured_lrs": dict(self.configured_lrs),
            "first_snapshot": [dict(s) for s in self.first_snapshot],
            "lr_constant": self.lr_constant,
            "lr_matches_config": self.lr_matches_config,
            "metrics_logs_lr": self.metrics_logs_lr,
            "finite_throughout": self.finite_throughout,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArmResult":
        return cls(
            optimizer_name=str(data.get("optimizer_name", "adamw")),
            seed=int(data.get("seed", 0)),
            steps_recorded=int(data.get("steps_recorded", 0)),
            configured_lrs=dict(data.get("configured_lrs", {})),
            first_snapshot=tuple(data.get("first_snapshot", ())),
            lr_constant=bool(data.get("lr_constant", True)),
            lr_matches_config=bool(data.get("lr_matches_config", True)),
            metrics_logs_lr=bool(data.get("metrics_logs_lr", False)),
            finite_throughout=bool(data.get("finite_throughout", True)),
        )


@dataclass(frozen=True)
class LrScheduleGapReport:
    """Fixture report for SLM-240."""

    schema: str = "LrScheduleGapReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm240-lr-schedule-gap"
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    steps: int = DEFAULT_STEPS
    n_records: int = DEFAULT_N_RECORDS
    optimizers: tuple[str, ...] = DEFAULT_OPTIMIZERS
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    arms: tuple[ArmResult, ...] = field(default_factory=tuple)
    all_lr_constant: bool = True
    all_lr_matches_config: bool = True
    any_metrics_log_lr: bool = False
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
            "optimizers": list(self.optimizers),
            "seeds": list(self.seeds),
            "arms": [a.to_dict() for a in self.arms],
            "all_lr_constant": self.all_lr_constant,
            "all_lr_matches_config": self.all_lr_matches_config,
            "any_metrics_log_lr": self.any_metrics_log_lr,
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
    def from_dict(cls, data: dict[str, Any]) -> "LrScheduleGapReport":
        return cls(
            schema=str(data.get("schema", "LrScheduleGapReportV1")),
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
            optimizers=tuple(data.get("optimizers", DEFAULT_OPTIMIZERS)),
            seeds=tuple(data.get("seeds", DEFAULT_SEEDS)),
            arms=tuple(ArmResult.from_dict(a) for a in data.get("arms", ())),
            all_lr_constant=bool(data.get("all_lr_constant", True)),
            all_lr_matches_config=bool(data.get("all_lr_matches_config", True)),
            any_metrics_log_lr=bool(data.get("any_metrics_log_lr", False)),
            all_finite=bool(data.get("all_finite", True)),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


def _expected_lrs(optimizer_name: str) -> dict[str, float]:
    if optimizer_name == "muon_hybrid":
        return {"muon": _MUON_LR, "adamw": _ADAMW_LR}
    return {"adamw": _BASE_LR}


def _run_arm(
    *,
    train_dir: Path,
    run_root: Path,
    optimizer_name: str,
    run_id: str,
    steps: int,
    batch_size: int,
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
        grad_accum_steps=1,
        lr=_BASE_LR,
        optimizer_name=optimizer_name,
        muon_lr=_MUON_LR if optimizer_name == "muon_hybrid" else None,
        adamw_lr=_ADAMW_LR if optimizer_name == "muon_hybrid" else None,
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
    summary, snapshots = _run_instrumented(config)
    metrics_rows = _read_metrics_rows(config.run_dir)
    metrics_logs_lr = any(
        ("lr" in row or "learning_rate" in row) for row in metrics_rows
    )

    expected = _expected_lrs(optimizer_name)
    finite_throughout = bool(snapshots) and all(
        all(torch.isfinite(torch.tensor(float(g["lr"]))) for g in snap)
        for snap in snapshots
    )

    first_snapshot: tuple[dict[str, Any], ...] = (
        tuple(snapshots[0]) if snapshots else ()
    )
    lr_constant = bool(snapshots) and all(
        snap == snapshots[0] for snap in snapshots
    )
    lr_matches_config = bool(first_snapshot) and all(
        abs(entry["lr"] - expected.get(entry["optimizer"], float("nan"))) < 1e-12
        for entry in first_snapshot
    )
    # Also require every expected optimizer family to actually appear (a
    # missing group -- e.g. no muon-eligible matrices at this tiny width --
    # would otherwise vacuously "match").
    seen_families = {entry["optimizer"] for entry in first_snapshot}
    lr_matches_config = lr_matches_config and seen_families == set(expected)

    del summary  # only used for side effects (metrics.jsonl on disk)
    return ArmResult(
        optimizer_name=optimizer_name,
        seed=seed,
        steps_recorded=len(snapshots),
        configured_lrs=expected,
        first_snapshot=first_snapshot,
        lr_constant=lr_constant,
        lr_matches_config=lr_matches_config,
        metrics_logs_lr=metrics_logs_lr,
        finite_throughout=finite_throughout,
    )


def run_lr_schedule_gap_probe(
    *,
    steps: int = DEFAULT_STEPS,
    n_records: int = DEFAULT_N_RECORDS,
    optimizers: tuple[str, ...] = DEFAULT_OPTIMIZERS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    run_id: str | None = None,
    out_dir: Path | str | None = None,
) -> LrScheduleGapReport:
    """Run the SLM-240 learning-rate schedule gap probe."""
    records = _build_records(n_records)
    arms: list[ArmResult] = []

    with tempfile.TemporaryDirectory(prefix="slm240-") as tmp:
        tmp_path = Path(tmp)
        train_dir = tmp_path / "train"
        run_root = tmp_path / "runs"
        _write_train_dir(train_dir, records)

        for optimizer_name in optimizers:
            for seed in seeds:
                arm = _run_arm(
                    train_dir=train_dir,
                    run_root=run_root,
                    optimizer_name=optimizer_name,
                    run_id=f"{EXPERIMENT_ID}-{optimizer_name}-{seed}",
                    steps=steps,
                    batch_size=n_records,
                    seed=seed,
                )
                arms.append(arm)

    all_lr_constant = all(a.lr_constant for a in arms)
    all_lr_matches_config = all(a.lr_matches_config for a in arms)
    any_metrics_log_lr = any(a.metrics_logs_lr for a in arms)
    all_finite = all(a.finite_throughout for a in arms)

    if not all_finite:
        disposition = "unstable"
        rationale = (
            "At least one arm produced a non-finite recorded lr during the "
            "run; the schedule question cannot be answered until that is "
            "fixed."
        )
    elif not all_lr_matches_config:
        disposition = "config_mismatch"
        rationale = (
            "At least one arm's first recorded optimizer.step() lr did not "
            "match its configured lr/muon_lr/adamw_lr (or an expected "
            "optimizer family never appeared in the live param groups) -- "
            "the configured-lr contract itself is not trustworthy here, "
            "independent of whether a schedule exists."
        )
    elif not all_lr_constant:
        disposition = "schedule_detected"
        rationale = (
            "At least one arm's recorded lr changed across steps -- the "
            "prior reading that the train loop applies no schedule is "
            "refuted for that configuration."
        )
    elif any_metrics_log_lr:
        disposition = "gap_confirmed_telemetry_partial"
        rationale = (
            "Every arm's live optimizer lr was constant and matched the "
            "configured value across all steps (confirming no warmup/decay "
            "schedule exists), but at least one arm's metrics.jsonl row did "
            "log an lr/learning_rate field, so the telemetry-blind-spot half "
            "of the hypothesis does not hold everywhere."
        )
    else:
        disposition = "gap_confirmed"
        rationale = (
            "Every arm's live optimizer.step() lr was bit-identical across "
            "all recorded steps and matched the configured lr/muon_lr/"
            "adamw_lr value, for both adamw and muon_hybrid, across all "
            "seeds -- the model_build train loop applies no learning-rate "
            "schedule (no warmup ramp, no decay) today. No metrics.jsonl row "
            "in any arm logged an lr/learning_rate field, so this would be "
            "invisible to existing per-step telemetry even if a bug caused "
            "the lr to drift."
        )

    report = LrScheduleGapReport(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        steps=steps,
        n_records=n_records,
        optimizers=tuple(optimizers),
        seeds=tuple(seeds),
        arms=tuple(arms),
        all_lr_constant=all_lr_constant,
        all_lr_matches_config=all_lr_matches_config,
        any_metrics_log_lr=any_metrics_log_lr,
        all_finite=all_finite,
        disposition=disposition,
        disposition_rationale=rationale,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm240_lr_schedule_gap",
            "harness.model_build.train",
            "model.twotower",
        ),
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(
            out_dir
            / f"iter-slm240-lrs0-01-lr-schedule-gap-{_today_yyyymmdd()}.json"
        )
    return report


def render_markdown(report: LrScheduleGapReport) -> str:
    """Render a compact design note for the fixture."""
    lines = [
        f"# SLM-240 (LRS0-01): learning-rate schedule gap probe ({report.run_id})",
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
        "## Probe",
        "",
        f"- optimizer steps per arm: {report.steps}",
        f"- records: {report.n_records}",
        f"- optimizers: {list(report.optimizers)}",
        f"- seeds: {list(report.seeds)}",
        "",
        "## Per-arm results",
        "",
        "| optimizer | seed | steps recorded | configured lrs | lr constant? | lr matches config? | metrics logs lr? | finite? |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for a in report.arms:
        cfg = ", ".join(f"{k}={v:g}" for k, v in sorted(a.configured_lrs.items()))
        lines.append(
            f"| {a.optimizer_name} | {a.seed} | {a.steps_recorded} | {cfg} | "
            f"{a.lr_constant} | {a.lr_matches_config} | {a.metrics_logs_lr} | "
            f"{a.finite_throughout} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- all_lr_constant: {report.all_lr_constant}",
            f"- all_lr_matches_config: {report.all_lr_matches_config}",
            f"- any_metrics_log_lr: {report.any_metrics_log_lr}",
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
            (
                "**No-go for any 'schedule already works' claim; honest gap "
                "confirmation.** This is wiring/fixture evidence over a tiny "
                "scratch-backend model and synthetic overfit data, not a "
                "quality or ship claim. A `gap_confirmed` disposition means "
                "the model_build TwoTower train loop has no learning-rate "
                "warmup or decay mechanism today, and no per-step telemetry "
                "field would surface one if it existed -- both real gaps for "
                "a future SLM to close, not something this probe fixes."
            ),
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_slm240_lr_schedule_gap --mode plan-only",
            "python -m scripts.run_slm240_lr_schedule_gap --mode fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    out = Path("docs/design")
    report = run_lr_schedule_gap_probe(out_dir=out)
    (
        out / f"iter-slm240-lrs0-01-lr-schedule-gap-{_today_yyyymmdd()}.md"
    ).write_text(render_markdown(report), encoding="utf-8")
    print(
        f"disposition={report.disposition} all_lr_constant={report.all_lr_constant} "
        f"all_lr_matches_config={report.all_lr_matches_config} "
        f"any_metrics_log_lr={report.any_metrics_log_lr}"
    )
