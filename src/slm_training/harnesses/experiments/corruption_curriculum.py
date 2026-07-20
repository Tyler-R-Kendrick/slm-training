"""SLM-120 EFS3-02 near-solved semantic corruption curriculum manifest.

This module provides the preregistered curriculum manifest, recipe-freeze hash,
and torch-free fixture runner. The actual A–D arm trains require a GPU host and
durable checkpoint provenance (SLM-103 / EFS1 readout).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slm_training.data.corrupt.trace import SeverityLevel
from slm_training.versioning import build_version_stamp

__all__ = [
    "CORRUPTION_CURRICULUM_ID",
    "CorruptionCurriculumManifest",
    "CorruptionCurriculumReport",
    "CurriculumArmResult",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "build_corruption_curriculum_manifest",
    "render_markdown",
    "run_fixture_curriculum",
    "validate_manifest",
]

MATRIX_VERSION = "efs3-02-v1"
MATRIX_SET = "corruption-curriculum"
CORRUPTION_CURRICULUM_ID = "near-solved-semantic"

# Preregistered near-solved shares (S1+S2) for arms A–E.
# Arm A is the clean control; arms B–D are the factorial; arm E is a stress test.
NEAR_SOLVED_SHARES = (0.0, 0.05, 0.10, 0.15, 0.30)

# Severity mix inside the near-solved share: 50/50 S1/S2.
S1_S2_SPLIT = (0.5, 0.5)


@dataclass(frozen=True)
class CorruptionCurriculumManifest:
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    curriculum_id: str = CORRUPTION_CURRICULUM_ID
    hypothesis: str = (
        "Including a small fraction of one- and two-error states improves correction "
        "of local semantic failures and preserves already-correct states, yielding "
        "higher binding-aware meaningful quality without reducing performance on "
        "heavily corrupted/full-generation cases."
    )
    falsifier: str = (
        "Near-solved mass produces no recovery/stability gain, causes copying/early-stop "
        "behavior, or degrades from-scratch semantic generation at matched exposure."
    )
    # Frozen base recipe from the E228 legal-candidate-margin candidate
    # (docs/design/iter-e228-candidate-margin-alignment-20260716.json).
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
            "max_wall_minutes": 3.0,
        }
    )
    near_solved_shares: tuple[float, ...] = NEAR_SOLVED_SHARES
    s1_s2_split: tuple[float, ...] = S1_S2_SPLIT
    seeds: tuple[int, ...] = (0, 1, 2)
    claim_class: str = "frontier"
    status: str = "not_run"
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
class CurriculumArmResult:
    arm_label: str
    near_solved_share: float
    seed: int
    status: str
    run_id: str | None = None
    checkpoint_uri: str | None = None
    train_loss: float | None = None
    binding_aware_meaningful_v2_rate_strict: float | None = None
    s0_stability_rate: float | None = None
    s1_recovery_rate: float | None = None
    s2_recovery_rate: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class CorruptionCurriculumReport:
    matrix_set: str
    matrix_version: str
    curriculum_id: str
    run_id: str
    status: str
    manifest: CorruptionCurriculumManifest
    arms: list[CurriculumArmResult]
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "curriculum_id": self.curriculum_id,
            "run_id": self.run_id,
            "status": self.status,
            "manifest": self.manifest.to_dict(),
            "arms": [a.to_dict() for a in self.arms],
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


def build_corruption_curriculum_manifest(
    *,
    parent_checkpoint_uri: str | None = None,
    checkpoint_bucket: str | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
    near_solved_shares: tuple[float, ...] = NEAR_SOLVED_SHARES,
) -> CorruptionCurriculumManifest:
    """Return the preregistered SLM-120 corruption-curriculum manifest."""
    if parent_checkpoint_uri is None:
        status = "not_run"
        claim_class = "wiring"
    else:
        status = "frontier_pending_gpu"
        claim_class = "frontier"
    return CorruptionCurriculumManifest(
        parent_checkpoint_uri=parent_checkpoint_uri,
        checkpoint_bucket=checkpoint_bucket,
        seeds=seeds,
        near_solved_shares=near_solved_shares,
        status=status,
        claim_class=claim_class,
    )


def validate_manifest(manifest: CorruptionCurriculumManifest) -> list[str]:
    errors: list[str] = []
    if not manifest.near_solved_shares:
        errors.append("near_solved_shares must not be empty")
    if 0.0 not in manifest.near_solved_shares:
        errors.append("0.0 control share is required")
    if any(not 0.0 <= share <= 1.0 for share in manifest.near_solved_shares):
        errors.append("near_solved_shares must be in [0, 1]")
    if not manifest.seeds:
        errors.append("seeds must not be empty")
    if len(manifest.s1_s2_split) != 2:
        errors.append("s1_s2_split must have two entries")
    elif abs(sum(manifest.s1_s2_split) - 1.0) > 1e-9:
        errors.append("s1_s2_split must sum to 1.0")
    if manifest.claim_class in {"frontier", "ship_candidate"}:
        if not manifest.parent_checkpoint_uri:
            errors.append("frontier/ship_candidate manifest requires parent_checkpoint_uri")
        if not manifest.checkpoint_bucket:
            errors.append("frontier/ship_candidate manifest requires checkpoint_bucket")
    return errors


def _arm_label(share: float) -> str:
    if share == 0.0:
        return "A_control"
    return f"B{int(share * 100):02d}"


def run_fixture_curriculum(
    manifest: CorruptionCurriculumManifest,
    *,
    run_id: str = "slm120_fixture",
    output_dir: Path | None = None,
) -> CorruptionCurriculumReport:
    """Torch-free fixture that validates the curriculum manifest and emits a plan."""
    arms: list[CurriculumArmResult] = []
    time.perf_counter()  # mark start timing for future telemetry
    for share in manifest.near_solved_shares:
        label = _arm_label(share)
        for seed in manifest.seeds:
            arms.append(
                CurriculumArmResult(
                    arm_label=label,
                    near_solved_share=share,
                    seed=seed,
                    status="fixture_planned",
                    run_id=f"{run_id}_{label}_s{seed}",
                    checkpoint_uri=None,
                    train_loss=None,
                    binding_aware_meaningful_v2_rate_strict=None,
                    s0_stability_rate=None,
                    s1_recovery_rate=None,
                    s2_recovery_rate=None,
                    notes=[
                        f"planned near-solved share {share:.0%} seed {seed}",
                        "fixture-only: no model trained",
                        f"S1/S2 split: {manifest.s1_s2_split[0]:.0%}/{manifest.s1_s2_split[1]:.0%}",
                    ],
                )
            )
    report = CorruptionCurriculumReport(
        matrix_set=manifest.matrix_set,
        matrix_version=manifest.matrix_version,
        curriculum_id=manifest.curriculum_id,
        run_id=run_id,
        status="fixture",
        manifest=manifest,
        arms=arms,
        version_stamp=build_version_stamp("harness.experiments"),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "corruption_curriculum_report.json")
    return report


def render_markdown(report: CorruptionCurriculumReport) -> str:
    lines = [
        f"# SLM-120 — Near-solved semantic corruption curriculum ({report.run_id})",
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
        "## Frozen base recipe (SHA-256)",
        "",
        f"```\n{report.manifest.recipe_hash()}\n```",
        "",
        "## Curriculum arms",
        "",
        "| Arm | Near-solved share (S1+S2) | Seeds | Status |",
        "| --- | --- | --- | --- |",
    ]
    for share in report.manifest.near_solved_shares:
        label = _arm_label(share)
        statuses = {a.status for a in report.arms if a.near_solved_share == share}
        lines.append(
            f"| {label} | {share:.0%} | {len(report.manifest.seeds)} | {','.join(sorted(statuses))} |"
        )
    lines.extend(["", "## Results", ""])
    for arm in report.arms:
        lines.append(
            f"- **{arm.arm_label} seed {arm.seed}**: {arm.status} — "
            f"run_id `{arm.run_id}`, near-solved share {arm.near_solved_share:.0%}"
        )
        for note in arm.notes:
            lines.append(f"  - {note}")
    lines.extend(
        [
            "",
            "## Severity taxonomy",
            "",
            "| Level | Name | Description |",
            "| --- | --- | --- |",
            f"| S0 | `{SeverityLevel.S0_CLEAN.value}` | no corruption; stability/identity target |",
            f"| S1 | `{SeverityLevel.S1_NEAR_SOLVED_1.value}` | exactly one semantic corruption |",
            f"| S2 | `{SeverityLevel.S2_NEAR_SOLVED_2.value}` | exactly two semantic corruptions |",
            f"| S3 | `{SeverityLevel.S3_MEDIUM.value}` | 3–5 semantic corruptions |",
            f"| S4 | `{SeverityLevel.S4_HEAVY.value}` | current full/high-mask corruption |",
            "",
            "## Verdict",
            "",
            "Fixture/plan only. The actual A–D curriculum trains require a GPU host, "
            "the EFS1-decided base recipe/checkpoint, and durable HF bucket sync per "
            "SLM-103. No curriculum claim or ship gate is made from this artifact.",
            "",
        ]
    )
    return "\n".join(lines)
