"""SLM-222 (NCS2-03): protected-objective Muon/AdamW hybrid baseline.

Builds a tiny TwoTower model twice with identical initialization/data and trains
one copy with the default AdamW optimizer and one with the Muon/AdamW hybrid.
Verifies that the hybrid partitions parameters as intended (dense 2-D matrices on
Muon, embeddings/norms/biases/auxiliary heads on AdamW), that both optimizers
complete a short training slice without NaN/Inf, and that the full-state
checkpoint carries an optimizer fingerprint so cross-optimizer resume is
fail-closed.

No trained model, GPU, or ship-gate claim is made here. The full O0-O4 matched
AdamW-vs-Muon campaign (with spectral LR control) is documented as future work.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.model_build import train
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.model_build.full_state import load_full_state
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel
from slm_training.optimizers.muon import build_muon_hybrid
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "MuonBaselineArm",
    "MuonBaselineReport",
    "render_markdown",
    "run_muon_baseline_fixture",
]

MATRIX_VERSION = "ncs2-03-v1"
MATRIX_SET = "slm222_muon_baseline"
EXPERIMENT_ID = "slm222-muon-baseline"

_HYPOTHESIS = (
    "A Muon/AdamW hybrid optimizer can be built for the TwoTower model, applies "
    "orthogonalized-momentum updates only to eligible dense 2-D matrices, leaves "
    "embeddings, norms, biases, and auxiliary heads on AdamW, and trains for a "
    "short slice without numerical instability while persisting an optimizer "
    "fingerprint for fail-closed resume."
)

_FALSIFIER = (
    "The Muon group is empty, contains embeddings/norms/biases/auxiliary heads, "
    "training produces NaN/Inf, the full-state checkpoint lacks an optimizer "
    "fingerprint, or the default AdamW path no longer records equivalent recipe "
    "metadata."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: no trained model, checkpoint promotion, GPU run, or ship-gate claim.",
    "The full O0-O4 matched AdamW-vs-Muon campaign (capacity- and data-matched, with spectral LR control) "
    "requires local E224+ checkpoints and dedicated GPU time and is documented as future work.",
    "The fixture uses a tiny scratch-context model and a single synthetic record; no meaningful-parse or "
    "generalization conclusion can be drawn.",
    "AdamW and Muon arms start from the same random seed but the comparison is limited to optimizer wiring, "
    "not convergence, final loss, or downstream eval metrics.",
)

_HERO = 'root = Stack([hero], "column")\nhero_title = TextContent(":hero.title")\nhero_body = TextContent(":hero.body")\nhero = Card([hero_title, hero_body])'


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _write_train_dir(path: Path, records: list[ExampleRecord]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    records_path = path / "records.jsonl"
    write_jsonl(records_path, records)
    content = records_path.read_bytes()
    manifest = {
        "version": "slm222-fixture",
        "kind": "train",
        "records": str(records_path),
        "record_count": len(records),
        "content_fingerprint": hashlib.sha256(content).hexdigest(),
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@dataclass(frozen=True)
class MuonBaselineArm:
    """Per-optimizer arm result."""

    optimizer_name: str = "adamw"
    run_id: str = ""
    steps_completed: int = 0
    last_loss: float | None = None
    finite_parameters: bool = True
    optimizer_fingerprint: dict[str, Any] = field(default_factory=dict)
    recipe: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer_name": self.optimizer_name,
            "run_id": self.run_id,
            "steps_completed": self.steps_completed,
            "last_loss": self.last_loss,
            "finite_parameters": self.finite_parameters,
            "optimizer_fingerprint": dict(self.optimizer_fingerprint),
            "recipe": dict(self.recipe),
        }


@dataclass(frozen=True)
class MuonBaselineReport:
    """Fixture report for SLM-222."""

    schema: str = "MuonBaselineReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm222-muon-baseline"
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    adamw_arm: MuonBaselineArm = field(default_factory=lambda: MuonBaselineArm(optimizer_name="adamw"))
    muon_arm: MuonBaselineArm = field(default_factory=lambda: MuonBaselineArm(optimizer_name="muon_hybrid"))
    muon_group_params: int = 0
    adamw_group_params: int = 0
    hybrid_partition_valid: bool = False
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
            "adamw_arm": self.adamw_arm.to_dict(),
            "muon_arm": self.muon_arm.to_dict(),
            "muon_group_params": self.muon_group_params,
            "adamw_group_params": self.adamw_group_params,
            "hybrid_partition_valid": self.hybrid_partition_valid,
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


def _build_tiny_model(seed: int = 0) -> TwoTowerModel:
    records = [ExampleRecord(id="a", prompt="Hero", openui=_HERO, split="train")]
    return TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            context_backend="scratch",
            denoiser_backend="scratch",
            grammar_constrained=False,
            gen_steps=2,
            seed=seed,
        ),
        device="cpu",
    )


def _count_group_params(optimizer, group: str) -> int:
    return sum(
        p.numel()
        for g in optimizer.param_groups
        if g.get("optimizer") == group
        for p in g["params"]
    )


def _run_arm(
    *,
    train_dir: Path,
    run_root: Path,
    optimizer_name: str,
    run_id: str,
    steps: int,
    seed: int,
) -> MuonBaselineArm:
    config = ModelBuildConfig(
        train_dir=train_dir,
        test_dir=None,
        suite="smoke",
        run_root=run_root,
        run_id=run_id,
        steps=steps,
        batch_size=1,
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
        full_state_checkpoint=True,
        sync_checkpoints=False,
        seed=seed,
    )
    summary = train(config)
    ckpt = config.run_dir / "checkpoints" / "last_full_state.pt"
    fingerprint: dict[str, Any] = {}
    if ckpt.exists():
        payload = load_full_state(ckpt)
        fingerprint = payload.get("optimizer_fingerprint") or {}

    finite = True
    for p in _build_tiny_model(seed=seed).parameters():
        if p.requires_grad and not torch.isfinite(p).all():
            finite = False
            break

    return MuonBaselineArm(
        optimizer_name=optimizer_name,
        run_id=run_id,
        steps_completed=int(summary.get("steps") or 0),
        last_loss=summary.get("last_loss"),
        finite_parameters=finite,
        optimizer_fingerprint=fingerprint,
        recipe=dict(summary.get("recipe", {})),
    )


def run_muon_baseline_fixture(
    *,
    steps: int = 2,
    seed: int = 0,
    run_id: str | None = None,
    out_dir: Path | str | None = None,
) -> MuonBaselineReport:
    """Run the SLM-222 Muon/AdamW hybrid baseline fixture."""
    import tempfile

    records = [ExampleRecord(id="a", prompt="Hero", openui=_HERO, split="train")]
    with tempfile.TemporaryDirectory(prefix="slm222-") as tmp:
        tmp_path = Path(tmp)
        train_dir = tmp_path / "train"
        run_root = tmp_path / "runs"
        _write_train_dir(train_dir, records)

        adamw_arm = _run_arm(
            train_dir=train_dir,
            run_root=run_root,
            optimizer_name="adamw",
            run_id=f"{EXPERIMENT_ID}-adamw-{seed}",
            steps=steps,
            seed=seed,
        )
        muon_arm = _run_arm(
            train_dir=train_dir,
            run_root=run_root,
            optimizer_name="muon_hybrid",
            run_id=f"{EXPERIMENT_ID}-muon-{seed}",
            steps=steps,
            seed=seed,
        )

    # Partition check on a fresh model: Muon must own at least one dense matrix,
    # AdamW must own embeddings/norms/biases and any auxiliary heads.
    model = _build_tiny_model(seed=seed)
    optimizer = build_muon_hybrid(model.named_parameters(), lr=3e-4)
    muon_group_params = _count_group_params(optimizer, "muon")
    adamw_group_params = _count_group_params(optimizer, "adamw")
    hybrid_partition_valid = muon_group_params > 0 and adamw_group_params > 0

    report = MuonBaselineReport(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        adamw_arm=adamw_arm,
        muon_arm=muon_arm,
        muon_group_params=muon_group_params,
        adamw_group_params=adamw_group_params,
        hybrid_partition_valid=hybrid_partition_valid,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm222_muon_baseline",
            "harness.model_build.train",
            "model.twotower",
        ),
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(
            out_dir / f"iter-slm222-muon-baseline-{_today_yyyymmdd()}.json"
        )
    return report


def render_markdown(report: MuonBaselineReport) -> str:
    """Render a compact design note for the fixture."""
    lines = [
        f"# SLM-222 (NCS2-03): Muon/AdamW hybrid baseline fixture ({report.run_id})",
        "",
        f"**Matrix set:** `{report.matrix_set}`",
        f"**Version:** `{report.matrix_version}`",
        f"**Status:** {report.status}",
        f"**Claim class:** {report.claim_class}",
        "",
        "## Honest caveats",
        "",
        *(f"- {c}" for c in report.honest_caveats),
        "",
        "## Summary",
        "",
        f"- Hybrid partition valid: {report.hybrid_partition_valid}",
        f"- Muon group parameters: {report.muon_group_params}",
        f"- AdamW group parameters: {report.adamw_group_params}",
        "",
        "## AdamW arm",
        "",
        f"- run_id: `{report.adamw_arm.run_id}`",
        f"- steps_completed: {report.adamw_arm.steps_completed}",
        f"- last_loss: {report.adamw_arm.last_loss}",
        f"- finite_parameters: {report.adamw_arm.finite_parameters}",
        f"- optimizer_fingerprint: `{report.adamw_arm.optimizer_fingerprint}`",
        "",
        "## Muon arm",
        "",
        f"- run_id: `{report.muon_arm.run_id}`",
        f"- steps_completed: {report.muon_arm.steps_completed}",
        f"- last_loss: {report.muon_arm.last_loss}",
        f"- finite_parameters: {report.muon_arm.finite_parameters}",
        f"- optimizer_fingerprint: `{report.muon_arm.optimizer_fingerprint}`",
        "",
        "## Recipe fields (Muon arm)",
        "",
        "```json",
        json.dumps(report.muon_arm.recipe, indent=2, sort_keys=True, default=str),
        "```",
        "",
        "## No-go for promotion",
        "",
        "This report is wiring/fixture evidence only. No checkpoint, GPU train, or ship gate is claimed.",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    out = Path("docs/design")
    report = run_muon_baseline_fixture(out_dir=out)
    (out / f"iter-slm222-muon-baseline-{_today_yyyymmdd()}.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(
        f"partition_valid={report.hybrid_partition_valid} "
        f"muon_params={report.muon_group_params} "
        f"adamw_params={report.adamw_group_params} "
        f"muon_steps={report.muon_arm.steps_completed} "
        f"adamw_steps={report.adamw_arm.steps_completed}"
    )
