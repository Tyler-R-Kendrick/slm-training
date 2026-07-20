"""SLM-124 EFS3-03 B3 surface-vs-choice capacity ladder v2 plan/fixture harness.

This module preregisters the corrected B3 capacity-ladder v2 (post-E288 choice-native
decoder) and emits a torch-free fixture plan. The actual 18-run ladder requires GPU
hosts, durable checkpoints, and the EFS1 exposure decision from SLM-109.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.ladder import (
    CAPACITY_ARMS,
    CAPACITY_WIDTHS,
    capacity_ladder_arms,
    ladder_run_id,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "B3_CAPACITY_V2_ID",
    "B3CapacityV2Arm",
    "B3CapacityV2Manifest",
    "B3CapacityV2Report",
    "B3CapacityV2Row",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "build_b3_capacity_v2_manifest",
    "render_markdown",
    "run_fixture_ladder",
    "validate_manifest",
]

MATRIX_VERSION = "efs3-03-v1"
MATRIX_SET = "b3-capacity-v2"
B3_CAPACITY_V2_ID = "efs-b3-capacity-v2"

# Decode fingerprints: stable hashes of the frozen decode configuration for each arm.
# These are wiring placeholders; frontier runs must record the actual decoder version.
SURFACE_DECODE_FINGERPRINT = "surface_lexer_v1:grammar_constrained=True,ltr_primary=False"
CHOICE_DECODE_FINGERPRINT = "choice_native_v1:grammar_constrained=True,e288_forced_singleton=True"


@dataclass(frozen=True)
class B3CapacityV2Arm:
    representation: str
    decode_fingerprint: str
    widths: tuple[int, ...]
    seeds: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "decode_fingerprint": self.decode_fingerprint,
            "widths": list(self.widths),
            "seeds": list(self.seeds),
        }


@dataclass(frozen=True)
class B3CapacityV2Row:
    arm: str
    representation: str
    d_model: int
    seed: int
    run_id: str
    target_token_budget: int
    status: str
    checkpoint_uri: str | None = None
    binding_aware_meaningful_v2_rate_strict: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class B3CapacityV2Manifest:
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    b3_id: str = B3_CAPACITY_V2_ID
    hypothesis: str = (
        "Externalizing non-lexical syntax into the choice codec shifts the "
        "semantic quality-vs-capacity curve left: at a preregistered semantic target, "
        "the choice representation reaches the target with fewer parameters or a "
        "smaller d_model than the surface representation after correcting the E288 "
        "choice-native decoder state."
    )
    falsifier: str = (
        "After matched semantic-example exposure and correct representation-native "
        "decoding, the surface and choice curves are equivalent or surface is better "
        "across all three widths; the 2.25× target-bit reduction then does not "
        "translate into useful model-capacity savings under this recipe."
    )
    base_recipe: dict[str, Any] = field(
        default_factory=lambda: {
            "device": "cpu",
            "context_backend": "scratch",
            "denoiser_backend": "scratch",
            "batch_size": 2,
            "learning_rate": 0.0003,
            "mask_pattern": "diffusion",
            "grammar_ltr_primary": False,
            "grammar_constrained": True,
            "parallel_unmask": "adaptive",
            "gen_steps": 8,
            "best_of_n": 1,
            "train_version": "e218_schema_normalized_judge_v5",
            "eval_version": "remediated",
            "eval_suites": "smoke,held_out,adversarial,ood,rico_held",
            "base_token_budget": 50000,
            "max_wall_minutes": 3.0,
            "checkpoint_sync": False,
        }
    )
    arms: tuple[B3CapacityV2Arm, ...] = (
        B3CapacityV2Arm(
            representation="lexer",
            decode_fingerprint=SURFACE_DECODE_FINGERPRINT,
            widths=CAPACITY_WIDTHS,
            seeds=(0, 1, 2),
        ),
        B3CapacityV2Arm(
            representation="choice",
            decode_fingerprint=CHOICE_DECODE_FINGERPRINT,
            widths=CAPACITY_WIDTHS,
            seeds=(0, 1, 2),
        ),
    )
    primary_metric: str = "binding_aware_meaningful_v2_rate_strict"
    claim_class: str = "frontier"
    status: str = "not_run"
    parent_checkpoint_uri: str | None = None
    checkpoint_bucket: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["base_recipe_hash"] = self.recipe_hash()
        data["arms"] = [a.to_dict() for a in self.arms]
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
class B3CapacityV2Report:
    matrix_set: str
    matrix_version: str
    b3_id: str
    run_id: str
    status: str
    manifest: B3CapacityV2Manifest
    rows: list[B3CapacityV2Row]
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "b3_id": self.b3_id,
            "run_id": self.run_id,
            "status": self.status,
            "manifest": self.manifest.to_dict(),
            "rows": [r.to_dict() for r in self.rows],
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


def build_b3_capacity_v2_manifest(
    *,
    parent_checkpoint_uri: str | None = None,
    checkpoint_bucket: str | None = None,
    widths: tuple[int, ...] = CAPACITY_WIDTHS,
    seeds: tuple[int, ...] = (0, 1, 2),
    representations: tuple[str, ...] = CAPACITY_ARMS,
) -> B3CapacityV2Manifest:
    """Return the preregistered SLM-124 B3 capacity-ladder v2 manifest."""
    def _arm(rep: str) -> B3CapacityV2Arm:
        known = {
            "lexer": SURFACE_DECODE_FINGERPRINT,
            "choice": CHOICE_DECODE_FINGERPRINT,
        }
        return B3CapacityV2Arm(
            representation=rep,
            decode_fingerprint=known.get(rep, ""),
            widths=widths,
            seeds=seeds,
        )

    arms = tuple(_arm(r) for r in representations)
    if parent_checkpoint_uri is None:
        status = "not_run"
        claim_class = "wiring"
    else:
        status = "frontier_pending_gpu"
        claim_class = "frontier"
    return B3CapacityV2Manifest(
        arms=arms,
        parent_checkpoint_uri=parent_checkpoint_uri,
        checkpoint_bucket=checkpoint_bucket,
        status=status,
        claim_class=claim_class,
    )


def validate_manifest(manifest: B3CapacityV2Manifest) -> list[str]:
    errors: list[str] = []
    if not manifest.arms:
        errors.append("arms must not be empty")
    valid_reps = set(CAPACITY_ARMS)
    for arm in manifest.arms:
        if arm.representation not in valid_reps:
            errors.append(f"unsupported representation: {arm.representation}")
        if not arm.widths:
            errors.append(f"{arm.representation}: widths must not be empty")
        if not arm.seeds:
            errors.append(f"{arm.representation}: seeds must not be empty")
        if not arm.decode_fingerprint:
            errors.append(f"{arm.representation}: decode_fingerprint must not be empty")
    representations = {a.representation for a in manifest.arms}
    if len(representations) != len(manifest.arms):
        errors.append("representations must be unique")
    if manifest.claim_class in {"frontier", "ship_candidate"}:
        if not manifest.parent_checkpoint_uri:
            errors.append("frontier/ship_candidate manifest requires parent_checkpoint_uri")
        if not manifest.checkpoint_bucket:
            errors.append("frontier/ship_candidate manifest requires checkpoint_bucket")
    return errors


def run_fixture_ladder(
    manifest: B3CapacityV2Manifest,
    *,
    run_id: str = "slm124_fixture",
    output_dir: Path | None = None,
) -> B3CapacityV2Report:
    """Torch-free fixture that validates the v2 manifest and emits a plan."""
    rows: list[B3CapacityV2Row] = []
    time.perf_counter()
    for arm in manifest.arms:
        ladder = capacity_ladder_arms(
            base_token_budget=manifest.base_recipe["base_token_budget"],
            widths=arm.widths,
            horizons=(1.0,),
            arms=(arm.representation,),
        )[arm.representation]
        for point in ladder.points:
            for seed in arm.seeds:
                rows.append(
                    B3CapacityV2Row(
                        arm=arm.representation,
                        representation=arm.representation,
                        d_model=point.d_model,
                        seed=seed,
                        run_id=ladder_run_id(ladder.ladder_id, point, seed),
                        target_token_budget=point.target_token_budget,
                        status="fixture_planned",
                        checkpoint_uri=None,
                        binding_aware_meaningful_v2_rate_strict=None,
                        notes=[
                            f"planned {arm.representation} width {point.d_model} seed {seed}",
                            "fixture-only: no model trained",
                            f"decode_fingerprint: {arm.decode_fingerprint}",
                        ],
                    )
                )
    report = B3CapacityV2Report(
        matrix_set=manifest.matrix_set,
        matrix_version=manifest.matrix_version,
        b3_id=manifest.b3_id,
        run_id=run_id,
        status="fixture",
        manifest=manifest,
        rows=rows,
        version_stamp=build_version_stamp("harness.experiments"),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "b3_capacity_v2_report.json")
    return report


def render_markdown(report: B3CapacityV2Report) -> str:
    lines = [
        f"# SLM-124 — B3 capacity ladder v2 ({report.run_id})",
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
        "## Arms",
        "",
        "| Arm | Representation | Widths | Seeds | Decode fingerprint |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arm in report.manifest.arms:
        lines.append(
            f"| {arm.representation} | {arm.representation} | "
            f"{','.join(map(str, arm.widths))} | {','.join(map(str, arm.seeds))} | "
            f"`{arm.decode_fingerprint}` |"
        )
    lines.extend(["", "## Rows", "", "| Arm | d_model | Seed | Run id | Status |", "| --- | --- | --- | --- | --- |"])
    for row in report.rows:
        lines.append(
            f"| {row.arm} | {row.d_model} | {row.seed} | `{row.run_id}` | {row.status} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "Fixture/plan only. The actual B3 v2 capacity ladder requires 18 matched "
            "trains (2 representations × 3 widths × 3 seeds), a GPU host, durable HF "
            "bucket sync per SLM-103, and the EFS1 exposure decision from SLM-109. "
            "No capacity-quality claim or ship gate is made from this artifact.",
            "",
        ]
    )
    return "\n".join(lines)
