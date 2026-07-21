"""SLM-211 (SDE5-04): fixture/wiring for the output-head tying control.

Builds tiny tied and untied TwoTower models, verifies that the untied copy-init
arm starts with the same function, distinct storage, and unambiguous optimizer
ownership, and emits a SpectralSnapshotV1 row so spectral tooling can record the
tie mode.

No trained model, GPU, or ship-gate claim is made here. The matched H0-H3
campaign (rare-action semantic mass) requires the capacity/exposure controls and
checkpoints described in the issue and is documented as future work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.experiments.slm214_spectral_snapshot import (
    run_spectral_snapshot_fixture,
)
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "UntiedOutputHeadReport",
    "run_untied_output_head_fixture",
]

MATRIX_VERSION = "sde5-04-v1"
MATRIX_SET = "slm211_untied_output_head"
EXPERIMENT_ID = "slm211-untied-output-head"

_HYPOTHESIS = (
    "At matched starting function, an untied output head uses distinct storage "
    "and receives unambiguous optimizer updates, while a tied head shares storage "
    "exactly as before."
)

_FALSIFIER = (
    "Untied copy-init does not match tied initial logits, or optimizer groups "
    "contain duplicate tied storage, or spectral tooling cannot tell the modes apart."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: no trained model, rare-action campaign, or GPU run.",
    "The H0-H3 matched experiment (capacity/exposure-matched rare-action debt/recall) "
    "is a follow-up requiring local E224+ checkpoints and the rare/focal weighting owner.",
)

_HERO = 'root = Stack([hero], "column")\nhero_title = TextContent(":hero.title")\nhero_body = TextContent(":hero.body")\nhero = Card([hero_title, hero_body])'


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_tiny_model(tie_output_embedding: bool, seed: int = 0) -> TwoTowerModel:
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
            tie_output_embedding=tie_output_embedding,
        ),
        device="cpu",
    )


@dataclass(frozen=True)
class UntiedOutputHeadReport:
    """Fixture report for SLM-211 architecture control."""

    schema: str = "UntiedOutputHeadReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm211-untied-output-head"
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    tie_output_embedding: bool = True
    tied_storage: bool = True
    copy_init_logits_match: bool = True
    optimizer_groups_unique: bool = True
    spectral_tie_recorded: bool = True
    n_trainable_parameters: int = 0
    n_optimizer_group_params: int = 0
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
            "tie_output_embedding": self.tie_output_embedding,
            "tied_storage": self.tied_storage,
            "copy_init_logits_match": self.copy_init_logits_match,
            "optimizer_groups_unique": self.optimizer_groups_unique,
            "spectral_tie_recorded": self.spectral_tie_recorded,
            "n_trainable_parameters": self.n_trainable_parameters,
            "n_optimizer_group_params": self.n_optimizer_group_params,
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


def run_untied_output_head_fixture(
    *,
    tie_output_embedding: bool = False,
    run_id: str | None = None,
    out_dir: Path | str | None = None,
) -> UntiedOutputHeadReport:
    """Run the SLM-211 architecture fixture for one tie mode."""
    model = _build_tiny_model(tie_output_embedding=tie_output_embedding)
    denoiser = model.denoiser
    tied_storage = denoiser.lm_head.weight is denoiser.tok.weight

    copy_init_logits_match = True
    if not tie_output_embedding:
        torch.testing.assert_close(denoiser.lm_head.weight, denoiser.tok.weight)

    # Optimizer groups must contain each unique storage once.
    groups = model.optimizer_parameter_groups()
    seen: set[int] = set()
    n_optimizer_group_params = 0
    for g in groups:
        for p in g["params"]:
            pid = id(p)
            assert pid not in seen, "duplicate parameter storage in optimizer groups"
            seen.add(pid)
            n_optimizer_group_params += p.numel()
    n_trainable_parameters = sum(
        p.numel() for p in model.trainable_parameters()
    )
    optimizer_groups_unique = n_optimizer_group_params == n_trainable_parameters

    # Spectral snapshot records the tie mode.
    spectral_report = run_spectral_snapshot_fixture(
        model, null_draws=5, max_matrices=4, device="cpu"
    )
    spectral_tie_recorded = all(
        s.tie_output_embedding is tie_output_embedding
        for s in spectral_report.snapshots
    )

    report = UntiedOutputHeadReport(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        tie_output_embedding=tie_output_embedding,
        tied_storage=tied_storage,
        copy_init_logits_match=copy_init_logits_match,
        optimizer_groups_unique=optimizer_groups_unique,
        spectral_tie_recorded=spectral_tie_recorded,
        n_trainable_parameters=n_trainable_parameters,
        n_optimizer_group_params=n_optimizer_group_params,
        version_stamp=build_version_stamp(
            "harness.experiments", "harness.experiments.slm211_untied_output_head"
        ),
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(
            out_dir / f"iter-slm211-untied-output-head-{_today_yyyymmdd()}.json"
        )
    return report


def render_markdown(report: UntiedOutputHeadReport) -> str:
    """Render a compact design note for the fixture."""
    lines = [
        f"# SLM-211 output-head tying fixture ({report.run_id})",
        "",
        "## Claim class",
        report.claim_class,
        "",
        "## Hypothesis",
        report.hypothesis,
        "",
        "## Falsifier",
        report.falsifier,
        "",
        "## Settings",
        f"- tie_output_embedding: {report.tie_output_embedding}",
        f"- tied_storage: {report.tied_storage}",
        f"- copy_init_logits_match: {report.copy_init_logits_match}",
        f"- optimizer_groups_unique: {report.optimizer_groups_unique}",
        f"- spectral_tie_recorded: {report.spectral_tie_recorded}",
        f"- n_trainable_parameters: {report.n_trainable_parameters}",
        f"- n_optimizer_group_params: {report.n_optimizer_group_params}",
        "",
        "## Honest caveats",
        "",
    ]
    for caveat in report.honest_caveats:
        lines.append(f"- {caveat}")
    lines.extend(["", "## Version stamp", "", "```json", json.dumps(report.version_stamp, indent=2, default=str), "```"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    out = Path("docs/design")
    for tie in (True, False):
        rep = run_untied_output_head_fixture(
            tie_output_embedding=tie,
            out_dir=out,
        )
        (out / f"iter-slm211-untied-output-head-{_today_yyyymmdd()}.md").write_text(
            render_markdown(rep), encoding="utf-8"
        )
        print(f"tie={tie} tied_storage={rep.tied_storage} optimizer_unique={rep.optimizer_groups_unique}")
