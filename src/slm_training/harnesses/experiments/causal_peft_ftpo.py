"""SLM-121 LDI1-02 causal PEFT FTPO manifest and fixture runner.

This module provides the preregistered causal-adapter FTPO manifest, recipe-freeze
hash, and torch-free fixture runner. The actual objective/matrix trains require a
GPU host, a causal base checkpoint, and durable provenance (SLM-103).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slm_training.levers import MAX_RUN_MINUTES
from slm_training.versioning import build_version_stamp

__all__ = [
    "CAUSAL_PEFT_FTPO_ID",
    "CausalPeftFtpoManifest",
    "CausalPeftFtpoReport",
    "FtpoArmResult",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "build_causal_peft_ftpo_manifest",
    "render_markdown",
    "run_fixture_ftpo",
    "validate_manifest",
]

MATRIX_VERSION = "ldi1-02-v1"
MATRIX_SET = "causal-peft-ftpo"
CAUSAL_PEFT_FTPO_ID = "causal-peft-ftpo"

# Preregistered FTPO objective arms. Each arm trains a removable PEFT adapter
# on exact-state DecisionEventV2 action tables.
FTPO_OBJECTIVES = ("unlikelihood", "ftpo_single", "ftpo_set", "legal_set_mass")

# Primary adapter method is LoRA; DoRA/PiSSA/AdaLoRA are config-ready but only
# supported when the installed PEFT exposes the documented flags.
ADAPTER_METHODS = ("lora", "dora", "pissa", "adalora")


@dataclass(frozen=True)
class CausalPeftFtpoManifest:
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    ftpo_id: str = CAUSAL_PEFT_FTPO_ID
    hypothesis: str = (
        "Updating only small PEFT adapters on exact-state causal decision events "
        "with FTPO objectives improves binding-aware meaningful-program rate while "
        "preserving base-model legality and keeping the adapter removable."
    )
    falsifier: str = (
        "PEFT FTPO adapters fail to move good legal actions above bad legal actions "
        "in logit space, degrade reference-locality metrics, or do not improve "
        "binding-aware meaningful outcomes at matched compute."
    )
    # Frozen base recipe (E228 legal-candidate-margin backbone).
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
            "max_wall_minutes": float(MAX_RUN_MINUTES),
            # PEFT / FTPO specific defaults
            "adapter_method": "lora",
            "adapter_rank": 8,
            "adapter_alpha": 16,
            "adapter_dropout": 0.0,
            "adapter_target_modules": ("q_proj", "v_proj"),
            "ftpo_epsilon": 2.0,
            "ftpo_tau": 1.0,
            "non_target_tether": 0.4,
            "target_tether": 0.05,
            "target_tether_grace": 1.0,
        }
    )
    objectives: tuple[str, ...] = FTPO_OBJECTIVES
    adapter_methods: tuple[str, ...] = ADAPTER_METHODS
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
class FtpoArmResult:
    objective: str
    adapter_method: str
    seed: int
    status: str
    run_id: str | None = None
    checkpoint_uri: str | None = None
    train_loss: float | None = None
    binding_aware_meaningful_v2_rate_strict: float | None = None
    reference_locality_drift: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class CausalPeftFtpoReport:
    matrix_set: str
    matrix_version: str
    ftpo_id: str
    run_id: str
    status: str
    manifest: CausalPeftFtpoManifest
    arms: list[FtpoArmResult]
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "ftpo_id": self.ftpo_id,
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


def build_causal_peft_ftpo_manifest(
    *,
    parent_checkpoint_uri: str | None = None,
    checkpoint_bucket: str | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
    objectives: tuple[str, ...] = FTPO_OBJECTIVES,
    adapter_methods: tuple[str, ...] = ("lora",),
) -> CausalPeftFtpoManifest:
    """Return the preregistered SLM-121 causal PEFT FTPO manifest."""
    if parent_checkpoint_uri is None:
        status = "not_run"
        claim_class = "wiring"
    else:
        status = "frontier_pending_gpu"
        claim_class = "frontier"
    return CausalPeftFtpoManifest(
        parent_checkpoint_uri=parent_checkpoint_uri,
        checkpoint_bucket=checkpoint_bucket,
        seeds=seeds,
        objectives=objectives,
        adapter_methods=adapter_methods,
        status=status,
        claim_class=claim_class,
    )


def validate_manifest(manifest: CausalPeftFtpoManifest) -> list[str]:
    errors: list[str] = []
    if not manifest.objectives:
        errors.append("objectives must not be empty")
    supported = set(FTPO_OBJECTIVES)
    for objective in manifest.objectives:
        if objective not in supported:
            errors.append(f"unsupported objective: {objective}")
    if not manifest.adapter_methods:
        errors.append("adapter_methods must not be empty")
    supported_methods = set(ADAPTER_METHODS)
    for method in manifest.adapter_methods:
        if method not in supported_methods:
            errors.append(f"unsupported adapter method: {method}")
    if not manifest.seeds:
        errors.append("seeds must not be empty")
    if manifest.claim_class in {"frontier", "ship_candidate"}:
        if not manifest.parent_checkpoint_uri:
            errors.append("frontier/ship_candidate manifest requires parent_checkpoint_uri")
        if not manifest.checkpoint_bucket:
            errors.append("frontier/ship_candidate manifest requires checkpoint_bucket")
    return errors


def _arm_label(objective: str, adapter_method: str) -> str:
    return f"{objective}_{adapter_method}"


def run_fixture_ftpo(
    manifest: CausalPeftFtpoManifest,
    *,
    run_id: str = "slm121_fixture",
    output_dir: Path | None = None,
) -> CausalPeftFtpoReport:
    """Torch-free fixture that validates the FTPO manifest and emits a plan."""
    arms: list[FtpoArmResult] = []
    time.perf_counter()  # mark start timing for future telemetry
    for objective in manifest.objectives:
        for adapter_method in manifest.adapter_methods:
            for seed in manifest.seeds:
                arms.append(
                    FtpoArmResult(
                        objective=objective,
                        adapter_method=adapter_method,
                        seed=seed,
                        status="fixture_planned",
                        run_id=f"{run_id}_{objective}_{adapter_method}_s{seed}",
                        checkpoint_uri=None,
                        train_loss=None,
                        binding_aware_meaningful_v2_rate_strict=None,
                        reference_locality_drift=None,
                        notes=[
                            f"planned {objective} objective with {adapter_method} adapter seed {seed}",
                            "fixture-only: no model trained",
                        ],
                    )
                )
    report = CausalPeftFtpoReport(
        matrix_set=manifest.matrix_set,
        matrix_version=manifest.matrix_version,
        ftpo_id=manifest.ftpo_id,
        run_id=run_id,
        status="fixture",
        manifest=manifest,
        arms=arms,
        version_stamp=build_version_stamp("harness.experiments"),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "causal_peft_ftpo_report.json")
    return report


def render_markdown(report: CausalPeftFtpoReport) -> str:
    lines = [
        f"# SLM-121 — Causal PEFT FTPO ({report.run_id})",
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
        "## FTPO arms",
        "",
        "| Objective | Adapter method | Seeds | Status |",
        "| --- | --- | --- | --- |",
    ]
    for objective in report.manifest.objectives:
        for adapter_method in report.manifest.adapter_methods:
            statuses = {
                a.status
                for a in report.arms
                if a.objective == objective and a.adapter_method == adapter_method
            }
            lines.append(
                f"| {objective} | {adapter_method} | {len(report.manifest.seeds)} | {','.join(sorted(statuses))} |"
            )
    lines.extend(["", "## Results", ""])
    for arm in report.arms:
        lines.append(
            f"- **{arm.objective} / {arm.adapter_method} seed {arm.seed}**: {arm.status} — "
            f"run_id `{arm.run_id}`"
        )
        for note in arm.notes:
            lines.append(f"  - {note}")
    lines.extend(
        [
            "",
            "## Required objectives",
            "",
            "| Objective | Purpose |",
            "| --- | --- |",
            "| `unlikelihood` | negative control over bad legal actions |",
            "| `ftpo_single` | exactly one good vs one bad action |",
            "| `ftpo_set` | weighted good × bad margins |",
            "| `legal_set_mass` | shift legal-space mass from bad set to good set |",
            "",
            "## Verdict",
            "",
            "Fixture/plan only. The actual causal PEFT FTPO trains require a GPU host, "
            "a causal base checkpoint, an admitted DecisionEventV2 corpus, and durable "
            "HF bucket sync per SLM-103. No adapter quality claim or ship gate is made "
            "from this artifact.",
            "",
        ]
    )
    return "\n".join(lines)
