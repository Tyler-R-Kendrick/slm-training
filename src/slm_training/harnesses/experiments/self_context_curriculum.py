"""SLM-300 AP-015 self-context / scheduled-corruption exposure-bias manifest.

This module provides the preregistered self-context curriculum manifest, a
pure deterministic policy-origin mixture calculation, and a torch-free
fixture runner. It follows the SLM-120 ``corruption_curriculum`` module
pattern: the manifest/mixture math is real, versioned, and testable; the
actual multi-seed training arms require a GPU host and durable checkpoint
provenance (SLM-103 / EFS1 readout) and are out of scope for this wiring
slice.

Objective (AP-015): determine whether train-decode state mismatch (exposure
bias) explains binding/reference recovery failures, and whether mixing
model-generated ("self-context") partial states into the corruption
schedule -- instead of exclusively gold-label-derived corruption -- improves
recovery without a hidden training-budget increase or validity regression.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from slm_training.harness_core.versioning import build_version_stamp
from slm_training.levers import MAX_RUN_MINUTES

__all__ = [
    "DEFAULT_LAGGED_POLICY_SHARE",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "SELF_CONTEXT_CURRICULUM_ID",
    "SELF_CONTEXT_RATES",
    "PolicyOrigin",
    "SelfContextArmResult",
    "SelfContextCurriculumManifest",
    "SelfContextCurriculumReport",
    "build_self_context_manifest",
    "policy_origin_mixture",
    "render_markdown",
    "run_fixture_self_context_curriculum",
    "validate_manifest",
]

MATRIX_VERSION = "ap015-v1"
MATRIX_SET = "self-context-exposure-bias"
SELF_CONTEXT_CURRICULUM_ID = "self-context-scheduled-corruption"

# Preregistered self-context rates: 0.0 is the mandatory clean/legacy control;
# 0.10/0.25/0.50 are the scheduled-corruption mixture arms from the issue.
SELF_CONTEXT_RATES = (0.0, 0.10, 0.25, 0.50)

# Of the self-context mass, the share drawn from a lagged/EMA policy
# checkpoint rather than the live/current policy. A lagged origin damps the
# instability of purely on-policy self-distillation (moving target), matching
# the "current-versus-lagged policy origin" provenance field required by the
# issue's acceptance criteria.
DEFAULT_LAGGED_POLICY_SHARE = 0.5


class PolicyOrigin(str, Enum):
    """Provenance of one corrupted/partial training state."""

    GOLD = "gold"
    MODEL_CURRENT = "model_current"
    MODEL_LAGGED = "model_lagged"


def policy_origin_mixture(
    self_context_rate: float,
    lagged_policy_share: float = DEFAULT_LAGGED_POLICY_SHARE,
) -> dict[str, float]:
    """Return the exact {gold, model_current, model_lagged} mass split.

    Pure function, no model/training required: ``self_context_rate`` is the
    total probability mass drawn from model-generated partial states (as
    opposed to gold/locked-label corruption); ``lagged_policy_share`` splits
    that self-context mass between the live policy and a lagged/EMA
    checkpoint. At ``self_context_rate == 0.0`` the mixture is always
    ``{"gold": 1.0, "model_current": 0.0, "model_lagged": 0.0}`` regardless of
    ``lagged_policy_share`` -- this is the exact wiring proof that the
    mixture-zero arm reproduces legacy (gold-only) corruption behavior.
    """
    if not 0.0 <= self_context_rate <= 1.0:
        raise ValueError("self_context_rate must be in [0, 1]")
    if not 0.0 <= lagged_policy_share <= 1.0:
        raise ValueError("lagged_policy_share must be in [0, 1]")
    gold = 1.0 - self_context_rate
    model_lagged = self_context_rate * lagged_policy_share
    model_current = self_context_rate - model_lagged
    return {
        PolicyOrigin.GOLD.value: gold,
        PolicyOrigin.MODEL_CURRENT.value: model_current,
        PolicyOrigin.MODEL_LAGGED.value: model_lagged,
    }


@dataclass(frozen=True)
class SelfContextCurriculumManifest:
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    curriculum_id: str = SELF_CONTEXT_CURRICULUM_ID
    hypothesis: str = (
        "Train-decode state mismatch (exposure bias) contributes to "
        "binding/reference recovery failures, and mixing verifier-aware, "
        "model-generated partial states into the corruption schedule at a "
        "moderate self-context rate improves recovery without validity "
        "regression or a hidden training-budget increase."
    )
    falsifier: str = (
        "Self-context mixture produces no paired strict-semantic recovery "
        "gain over the gold-only (rate=0) control, or any gain requires "
        "validity regression greater than 0.01 or more total tokens/updates "
        "than the matched control."
    )
    # Base recipe fields mirror the SLM-120 corruption-curriculum base recipe
    # shape so both curricula stay directly comparable at matched exposure.
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
            "checkpoint_sync": False,
            "max_wall_minutes": float(MAX_RUN_MINUTES),
        }
    )
    self_context_rates: tuple[float, ...] = SELF_CONTEXT_RATES
    lagged_policy_share: float = DEFAULT_LAGGED_POLICY_SHARE
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
class SelfContextArmResult:
    arm_label: str
    self_context_rate: float
    seed: int
    status: str
    policy_origin_mixture: dict[str, float]
    run_id: str | None = None
    checkpoint_uri: str | None = None
    train_loss: float | None = None
    binding_aware_meaningful_v2_rate_strict: float | None = None
    validity_regression: float | None = None
    binder_reference_recovery: dict[str, float] | None = None
    denoising_round_recovery: dict[str, float] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class SelfContextCurriculumReport:
    matrix_set: str
    matrix_version: str
    curriculum_id: str
    run_id: str
    status: str
    manifest: SelfContextCurriculumManifest
    arms: list[SelfContextArmResult]
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


def build_self_context_manifest(
    *,
    parent_checkpoint_uri: str | None = None,
    checkpoint_bucket: str | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
    self_context_rates: tuple[float, ...] = SELF_CONTEXT_RATES,
    lagged_policy_share: float = DEFAULT_LAGGED_POLICY_SHARE,
) -> SelfContextCurriculumManifest:
    """Return the preregistered SLM-300 self-context curriculum manifest."""
    if parent_checkpoint_uri is None:
        status = "not_run"
        claim_class = "wiring"
    else:
        status = "frontier_pending_gpu"
        claim_class = "frontier"
    return SelfContextCurriculumManifest(
        parent_checkpoint_uri=parent_checkpoint_uri,
        checkpoint_bucket=checkpoint_bucket,
        seeds=seeds,
        self_context_rates=self_context_rates,
        lagged_policy_share=lagged_policy_share,
        status=status,
        claim_class=claim_class,
    )


def validate_manifest(manifest: SelfContextCurriculumManifest) -> list[str]:
    errors: list[str] = []
    if not manifest.self_context_rates:
        errors.append("self_context_rates must not be empty")
    if 0.0 not in manifest.self_context_rates:
        errors.append("0.0 control rate is required (mixture zero must reproduce legacy behavior)")
    if any(not 0.0 <= rate <= 1.0 for rate in manifest.self_context_rates):
        errors.append("self_context_rates must be in [0, 1]")
    if not 0.0 <= manifest.lagged_policy_share <= 1.0:
        errors.append("lagged_policy_share must be in [0, 1]")
    if not manifest.seeds:
        errors.append("seeds must not be empty")
    if manifest.claim_class in {"frontier", "ship_candidate"}:
        if not manifest.parent_checkpoint_uri:
            errors.append("frontier/ship_candidate manifest requires parent_checkpoint_uri")
        if not manifest.checkpoint_bucket:
            errors.append("frontier/ship_candidate manifest requires checkpoint_bucket")
    return errors


def _arm_label(rate: float) -> str:
    if rate == 0.0:
        return "A_control"
    return f"SC{rate * 100:g}".replace(".", "p")


def run_fixture_self_context_curriculum(
    manifest: SelfContextCurriculumManifest,
    *,
    run_id: str = "slm300_fixture",
    output_dir: Path | None = None,
) -> SelfContextCurriculumReport:
    """Torch-free fixture that validates the manifest and emits a plan.

    The policy-origin mixture is computed exactly (it is a pure function of
    the manifest, not a training result); every other per-arm metric remains
    ``None`` because no model is trained in this wiring slice.

    Raises:
        ValueError: if ``manifest`` fails :func:`validate_manifest` (e.g. a
            missing ``0.0`` control rate or empty seeds).
    """
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("; ".join(errors))

    arms: list[SelfContextArmResult] = []
    time.perf_counter()  # mark start timing for future telemetry
    for rate in manifest.self_context_rates:
        label = _arm_label(rate)
        mixture = policy_origin_mixture(rate, manifest.lagged_policy_share)
        for seed in manifest.seeds:
            notes = [
                f"planned self-context rate {rate:.0%} seed {seed}",
                "fixture-only: no model trained",
                f"policy origin mixture: gold={mixture[PolicyOrigin.GOLD.value]:.2f} "
                f"model_current={mixture[PolicyOrigin.MODEL_CURRENT.value]:.2f} "
                f"model_lagged={mixture[PolicyOrigin.MODEL_LAGGED.value]:.2f}",
            ]
            if rate == 0.0:
                notes.append("mixture zero reproduces legacy gold-only corruption exactly")
            arms.append(
                SelfContextArmResult(
                    arm_label=label,
                    self_context_rate=rate,
                    seed=seed,
                    status="fixture_planned",
                    policy_origin_mixture=mixture,
                    run_id=f"{run_id}_{label}_s{seed}",
                    notes=notes,
                )
            )
    report = SelfContextCurriculumReport(
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
        report.to_json(output_dir / "self_context_curriculum_report.json")
    return report


def render_markdown(report: SelfContextCurriculumReport) -> str:
    lines = [
        f"# SLM-300 AP-015 — Self-context exposure-bias curriculum ({report.run_id})",
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
        "| Arm | Self-context rate | Lagged share | Seeds | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for rate in report.manifest.self_context_rates:
        label = _arm_label(rate)
        statuses = {a.status for a in report.arms if a.self_context_rate == rate}
        lines.append(
            f"| {label} | {rate:.0%} | {report.manifest.lagged_policy_share:.0%} | "
            f"{len(report.manifest.seeds)} | {','.join(sorted(statuses))} |"
        )
    lines.extend(["", "## Policy origin mixture", "", "| Arm | gold | model_current | model_lagged |", "| --- | --- | --- | --- |"])
    seen_labels: set[str] = set()
    for arm in report.arms:
        if arm.arm_label in seen_labels:
            continue
        seen_labels.add(arm.arm_label)
        mix = arm.policy_origin_mixture
        lines.append(
            f"| {arm.arm_label} | {mix[PolicyOrigin.GOLD.value]:.2f} | "
            f"{mix[PolicyOrigin.MODEL_CURRENT.value]:.2f} | "
            f"{mix[PolicyOrigin.MODEL_LAGGED.value]:.2f} |"
        )
    lines.extend(["", "## Results", ""])
    for arm in report.arms:
        lines.append(
            f"- **{arm.arm_label} seed {arm.seed}**: {arm.status} — "
            f"run_id `{arm.run_id}`, self-context rate {arm.self_context_rate:.0%}"
        )
        for note in arm.notes:
            lines.append(f"  - {note}")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "Fixture/plan only. The policy-origin mixture math is exact and tested "
            "(including the mixture-zero legacy-equivalence invariant); the actual "
            "multi-seed self-context training arms require a GPU host, the "
            "certified baseline checkpoint, and durable HF bucket sync per SLM-103. "
            "No recovery or ship-gate claim is made from this artifact.",
            "",
        ]
    )
    return "\n".join(lines)
