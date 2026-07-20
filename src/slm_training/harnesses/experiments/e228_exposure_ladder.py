"""SLM-109 E228 exposure-ladder manifest and fixture runner.

EFS1-02: run a ≥100× training-exposure checkpoint ladder on the frozen E228
legal-candidate-margin recipe. This module provides the recipe-freeze manifest,
ladder-point definitions, and a torch-free fixture runner. The actual 128× ladder
requires a GPU host and durable checkpoint provenance (SLM-103).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.versioning import build_version_stamp

__all__ = [
    "E228ExposureLadderManifest",
    "E228ExposureReport",
    "LADDER_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "build_e228_exposure_ladder",
    "build_e228_recipe_config",
    "render_markdown",
    "run_fixture_ladder",
    "validate_manifest",
]

MATRIX_VERSION = "efs1-02-v1"
MATRIX_SET = "e228-exposure-ladder"
LADDER_ID = "e228-exposure"

# Original E228 cumulative target-token exposure from committed evidence.
E228_TARGET_TOKENS = 6401

# Preregistered exposure multipliers (1× reproduction, then 4× ladder steps).
EXPOSURE_MULTIPLIERS = (1, 4, 16, 64, 128)


@dataclass(frozen=True)
class E228ExposureLadderManifest:
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    ladder_id: str = LADDER_ID
    hypothesis: str = (
        "Binding-aware semantic quality on the E228 recipe exhibits a clear exposure "
        "threshold between 4× and 128× cumulative target tokens."
    )
    falsifier: str = (
        "Semantic metrics remain flat or degrade through ≥100× exposure while train "
        "loss improves or saturates."
    )
    # Frozen E228 recipe fields (from docs/design/iter-e228-candidate-margin-alignment-20260716.json).
    base_recipe: dict[str, Any] = field(
        default_factory=lambda: {
            "device": "cpu",
            "steps": 32,
            "batch_size": 4,
            "learning_rate": 0.0003,
            "seed": 0,
            "context_backend": "hf_local_files_only",
            "train_version": "e218_schema_normalized_judge_v5",
            "eval_version": "remediated",
            "output_tokenizer": "lexer",
            "compiler_alignment_loss_weight": 1.0,
            "compiler_alignment_margin": 1.0,
            "compiler_alignment_stratified": True,
            "compiler_alignment_semantic_exhaustive": True,
            "alignment_candidate_scope": "lark_compiler_forest",
            "compiler_decode_mode": "tree",
            "schema_in_context": True,
            "slot_contract_in_context": True,
            "slot_contract_constrained_decode": True,
            "honest_slot_contract": True,
            "design_md_in_context": False,
            "allow_unconstrained_fallback": False,
            "mixture_sampling_policy": "quota_capacity_aware",
            "checkpoint_sync": False,
        }
    )
    base_target_tokens: int = E228_TARGET_TOKENS
    multipliers: tuple[int, ...] = EXPOSURE_MULTIPLIERS
    seeds: tuple[int, ...] = (0, 1, 2)
    claim_class: str = "frontier"
    status: str = "not_run"
    # Required for frontier execution.
    parent_checkpoint_uri: str | None = None
    checkpoint_bucket: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["base_recipe_hash"] = self.recipe_hash()
        return data

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    def recipe_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.base_recipe, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class ExposurePointResult:
    multiplier: int
    target_tokens: int
    seed: int
    status: str
    run_id: str | None = None
    checkpoint_uri: str | None = None
    train_loss: float | None = None
    binding_aware_meaningful_v2_rate_strict: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class E228ExposureReport:
    matrix_set: str
    matrix_version: str
    ladder_id: str
    run_id: str
    status: str
    manifest: E228ExposureLadderManifest
    points: list[ExposurePointResult]
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "ladder_id": self.ladder_id,
            "run_id": self.run_id,
            "status": self.status,
            "manifest": self.manifest.to_dict(),
            "points": [p.to_dict() for p in self.points],
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


def build_e228_exposure_ladder(
    *,
    parent_checkpoint_uri: str | None = None,
    checkpoint_bucket: str | None = None,
    seeds: tuple[int, ...] = EXPOSURE_MULTIPLIERS,
    multipliers: tuple[int, ...] = EXPOSURE_MULTIPLIERS,
) -> E228ExposureLadderManifest:
    """Return the preregistered SLM-109 exposure-ladder manifest."""
    if parent_checkpoint_uri is None:
        status = "not_run"
    else:
        status = "frontier_pending_gpu"
    return E228ExposureLadderManifest(
        parent_checkpoint_uri=parent_checkpoint_uri,
        checkpoint_bucket=checkpoint_bucket,
        seeds=seeds,
        multipliers=multipliers,
        status=status,
    )


def validate_manifest(manifest: E228ExposureLadderManifest) -> list[str]:
    errors: list[str] = []
    if not manifest.multipliers:
        errors.append("multipliers must not be empty")
    if 1 not in manifest.multipliers:
        errors.append("1× reproduction multiplier is required")
    if max(manifest.multipliers, default=0) < 100:
        errors.append("at least one multiplier must be ≥100")
    if not manifest.seeds:
        errors.append("seeds must not be empty")
    if manifest.claim_class in {"frontier", "ship_candidate"}:
        if not manifest.parent_checkpoint_uri:
            errors.append("frontier/ship_candidate manifest requires parent_checkpoint_uri")
        if not manifest.checkpoint_bucket:
            errors.append("frontier/ship_candidate manifest requires checkpoint_bucket")
    return errors


def build_e228_recipe_config(manifest: E228ExposureLadderManifest) -> ModelBuildConfig:
    """Map the frozen E228 recipe to a ModelBuildConfig for training.

    The returned config is intentionally minimal; callers overlay run_id,
    target_token_budget, and resume path per ladder point.
    """
    recipe = dict(manifest.base_recipe)
    return ModelBuildConfig(
        train_dir=Path("outputs/data/train") / str(recipe.get("train_version", "v1")),
        test_dir=Path("outputs/data/eval") / str(recipe.get("eval_version", "v1")),
        device=str(recipe.get("device", "cpu")),
        batch_size=int(recipe.get("batch_size", 4)),
        lr=float(recipe.get("learning_rate", 3e-4)),
        seed=int(recipe.get("seed", 0)),
        context_backend=str(recipe.get("context_backend", "scratch")),
        local_files_only=bool(recipe.get("local_files_only", False)),
        output_tokenizer=str(recipe.get("output_tokenizer", "compositional")),
        compiler_alignment_loss_weight=float(
            recipe.get("compiler_alignment_loss_weight", 0.0)
        ),
        compiler_alignment_margin=float(recipe.get("compiler_alignment_margin", 0.0)),
        compiler_alignment_stratified=bool(
            recipe.get("compiler_alignment_stratified", False)
        ),
        compiler_alignment_semantic_exhaustive=bool(
            recipe.get("compiler_alignment_semantic_exhaustive", False)
        ),
        compiler_decode_mode=str(recipe.get("compiler_decode_mode", "off")),
        schema_in_context=bool(recipe.get("schema_in_context", False)),
        slot_contract_in_context=bool(recipe.get("slot_contract_in_context", False)),
        slot_contract_constrained_decode=bool(
            recipe.get("slot_contract_constrained_decode", False)
        ),
        honest_slot_contract=bool(recipe.get("honest_slot_contract", False)),
        design_md_in_context=bool(recipe.get("design_md_in_context", True)),
        allow_unconstrained_fallback=bool(
            recipe.get("allow_unconstrained_fallback", True)
        ),
        grammar_constrained=True,
        grammar_ltr_primary=True,
        max_wall_minutes=3.0,
    )


def run_fixture_ladder(
    manifest: E228ExposureLadderManifest,
    *,
    run_id: str = "slm109_fixture",
    output_dir: Path | None = None,
) -> E228ExposureReport:
    """Torch-free fixture that validates the ladder manifest and emits a plan."""
    points: list[ExposurePointResult] = []
    time.perf_counter()  # mark start timing for future telemetry
    for multiplier in manifest.multipliers:
        target_tokens = manifest.base_target_tokens * multiplier
        for seed in manifest.seeds:
            points.append(
                ExposurePointResult(
                    multiplier=multiplier,
                    target_tokens=target_tokens,
                    seed=seed,
                    status="fixture_planned",
                    run_id=f"{run_id}_m{multiplier}_s{seed}",
                    checkpoint_uri=None,
                    train_loss=None,
                    binding_aware_meaningful_v2_rate_strict=None,
                    notes=[
                        f"planned {multiplier}× exposure = {target_tokens} target tokens",
                        "fixture-only: no model trained",
                    ],
                )
            )
    report = E228ExposureReport(
        matrix_set=manifest.matrix_set,
        matrix_version=manifest.matrix_version,
        ladder_id=manifest.ladder_id,
        run_id=run_id,
        status="fixture",
        manifest=manifest,
        points=points,
        version_stamp=build_version_stamp("harness.experiments"),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "e228_exposure_report.json")
    return report


def render_markdown(report: E228ExposureReport) -> str:
    lines = [
        f"# SLM-109 — E228 ≥100× exposure ladder ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`  ",
        f"Version: `{report.matrix_version}`  ",
        f"Status: **{report.status}**  ",
        "",
        "## Hypothesis",
        "",
        report.manifest.hypothesis,
        "",
        "## Falsifier",
        "",
        report.manifest.falsifier,
        "",
        "## Frozen E228 recipe (SHA-256)",
        "",
        f"```\n{report.manifest.recipe_hash()}\n```",
        "",
        "## Ladder points",
        "",
        "| Multiplier | Target tokens | Seeds | Status |",
        "| --- | --- | --- | --- |",
    ]
    for mult in report.manifest.multipliers:
        target = report.manifest.base_target_tokens * mult
        statuses = {p.status for p in report.points if p.multiplier == mult}
        lines.append(
            f"| {mult}× | {target} | {len(report.manifest.seeds)} | {','.join(sorted(statuses))} |"
        )
    lines.extend(["", "## Results", ""])
    for point in report.points:
        lines.append(
            f"- **{point.multiplier}× seed {point.seed}**: {point.status} — "
            f"run_id `{point.run_id}`, target {point.target_tokens} tokens"
        )
        for note in point.notes:
            lines.append(f"  - {note}")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "Fixture/plan only. The actual 128× ladder requires a GPU host, the original "
            "E228 checkpoint or a verified 1× reproduction, and durable HF bucket sync "
            "per SLM-103. No exposure claim or ship gate is made from this artifact.",
            "",
        ]
    )
    return "\n".join(lines)
