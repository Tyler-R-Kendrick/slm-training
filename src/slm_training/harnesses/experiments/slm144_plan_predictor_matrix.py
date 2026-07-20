"""SLM-144 SPV1-01: archetype + role-set predictor fixture matrix.

Wiring-only evidence. The CPU fixture trains the smallest possible predictor on
a synthetic corpus and compares baseline, frequency-prior, learned head, and
oracle arms. No production TwoTower wiring is touched and no ship claim is made.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from slm_training.models.semantic_plan_predictor import (
    PlanBatchCollator,
    PlanTrainingExample,
    predict_role_set_from_logits,
    predict_serialized_inventory,
    train_fixture_predictor,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "Slm144Arm",
    "Slm144Manifest",
    "Slm144Report",
    "Slm144Row",
    "build_slm144_manifest",
    "render_markdown",
    "run_fixture_matrix",
]

MATRIX_VERSION = "spv1-01-v1"
MATRIX_SET = "slm144_plan_predictor"


@dataclass(frozen=True)
class Slm144Arm:
    """Description of one matrix arm."""

    arm_id: str
    description: str
    archetype_source: str  # none | frequency | predicted | gold
    role_set_source: str  # none | frequency | serialized | predicted | gold
    status: str = "fixture"

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class Slm144Row:
    """Result row for one arm."""

    arm_id: str
    status: str
    archetype_accuracy: float | None = None
    role_precision: float | None = None
    role_recall: float | None = None
    role_f1: float | None = None
    inventory_token_accuracy: float | None = None
    latency_ms: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class Slm144Manifest:
    """Fixture manifest for SLM-144."""

    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    hypothesis: str = (
        "A tiny CPU predictor can learn archetype and role-set factors from a "
        "component-family count vector, and learned heads outperform frequency "
        "baselines on a controlled fixture corpus."
    )
    arms: list[Slm144Arm] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["arms"] = [arm.to_dict() for arm in self.arms]
        return data

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class Slm144Report:
    """Container for the full fixture matrix run."""

    matrix_set: str
    matrix_version: str
    run_id: str
    status: str
    manifest: Slm144Manifest
    rows: list[Slm144Row]
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "run_id": self.run_id,
            "status": self.status,
            "manifest": self.manifest.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str) + "\n",
            encoding="utf-8",
        )


def build_slm144_manifest() -> Slm144Manifest:
    """Return the default SLM-144 fixture arm manifest."""
    arms = [
        Slm144Arm(
            arm_id="baseline_none",
            description="Empty plan baseline: no archetype and no roles.",
            archetype_source="none",
            role_set_source="none",
        ),
        Slm144Arm(
            arm_id="frequency_prior",
            description="Train-set majority archetype and union role set.",
            archetype_source="frequency",
            role_set_source="frequency",
        ),
        Slm144Arm(
            arm_id="serialized_inventory",
            description="Serialized inventory head decoded into a role set.",
            archetype_source="none",
            role_set_source="serialized",
        ),
        Slm144Arm(
            arm_id="set_matching",
            description="Learned-slot bipartite matching role-set head.",
            archetype_source="none",
            role_set_source="predicted",
        ),
        Slm144Arm(
            arm_id="gold_archetype",
            description="Gold archetype + learned role-set head.",
            archetype_source="gold",
            role_set_source="predicted",
        ),
        Slm144Arm(
            arm_id="gold_role_set",
            description="Learned archetype head + gold role set.",
            archetype_source="predicted",
            role_set_source="gold",
        ),
        Slm144Arm(
            arm_id="gold_both",
            description="Oracle upper bound using gold archetype and role set.",
            archetype_source="gold",
            role_set_source="gold",
        ),
    ]
    return Slm144Manifest(arms=arms)


def _frequency_priors(
    train_examples: Sequence[PlanTrainingExample],
) -> tuple[int, set[int]]:
    labels = [e.archetype_label for e in train_examples]
    archetype = max(set(labels), key=labels.count)
    role_union: set[int] = set()
    for ex in train_examples:
        role_union.update(ex.role_set_mask.nonzero(as_tuple=True)[0].tolist())
    return archetype, role_union


def _set_metrics(
    pred_sets: Sequence[set[int]],
    gold_sets: Sequence[set[int]],
) -> dict[str, float]:
    tp = sum(len(p & g) for p, g in zip(pred_sets, gold_sets))
    pred_total = sum(len(p) for p in pred_sets)
    gold_total = sum(len(g) for g in gold_sets)
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _evaluate_predictions(
    arm: Slm144Arm,
    train_examples: Sequence[PlanTrainingExample],
    val_examples: Sequence[PlanTrainingExample],
    bundle: dict[str, Any],
) -> Slm144Row:
    collator = PlanBatchCollator()
    batch = collator(val_examples)
    inputs = batch["input_features"]
    gold_labels = batch["archetype_labels"]
    gold_masks = batch["role_set_masks"]
    gold_serial = batch["serialized_roles"]

    freq_archetype, freq_roles = _frequency_priors(train_examples)
    gold_role_sets = [
        set(gold_masks[i].nonzero(as_tuple=True)[0].tolist())
        for i in range(gold_masks.shape[0])
    ]

    t0 = time.perf_counter()

    archetype_head = bundle["archetype_head"]
    role_head = bundle["role_set_head"]
    inventory_head = bundle["serialized_inventory_head"]
    archetype_head.eval()
    role_head.eval()
    inventory_head.eval()

    with torch.no_grad():
        arch_logits = archetype_head(inputs)
        role_logits = role_head(inputs)
        inv_teacher_logits = inventory_head.teacher_forward(inputs, gold_serial)
        inv_greedy_logits = inventory_head(inputs)

    if arm.archetype_source == "gold":
        archetype_preds = gold_labels.tolist()
    elif arm.archetype_source == "frequency":
        archetype_preds = [freq_archetype] * len(val_examples)
    elif arm.archetype_source == "predicted":
        archetype_preds = arch_logits.argmax(dim=-1).tolist()
    else:
        archetype_preds = [-1] * len(val_examples)

    archetype_acc = (
        sum(1 for p, g in zip(archetype_preds, gold_labels.tolist()) if p == g)
        / len(val_examples)
    )

    if arm.role_set_source == "gold":
        pred_role_sets = gold_role_sets
    elif arm.role_set_source == "frequency":
        pred_role_sets = [freq_roles] * len(val_examples)
    elif arm.role_set_source == "serialized":
        pred_role_sets = [
            set(predict_serialized_inventory(inv_greedy_logits[i]))
            for i in range(inv_greedy_logits.shape[0])
        ]
    elif arm.role_set_source == "predicted":
        blank = role_head.blank_role
        pred_role_sets = [
            set(predict_role_set_from_logits(role_logits[i], blank))
            for i in range(role_logits.shape[0])
        ]
    else:
        pred_role_sets = [set() for _ in val_examples]

    role_metrics = _set_metrics(pred_role_sets, gold_role_sets)

    inventory_acc: float | None = None
    if arm.role_set_source == "serialized":
        inventory_acc = (
            (inv_teacher_logits.argmax(dim=-1) == gold_serial).float().mean().item()
        )

    latency_ms = (time.perf_counter() - t0) * 1000.0

    notes = [
        f"archetype_source={arm.archetype_source}",
        f"role_set_source={arm.role_set_source}",
        "fixture-only: tiny CPU net, no real checkpoint",
    ]
    return Slm144Row(
        arm_id=arm.arm_id,
        status="fixture",
        archetype_accuracy=archetype_acc,
        role_precision=role_metrics["precision"],
        role_recall=role_metrics["recall"],
        role_f1=role_metrics["f1"],
        inventory_token_accuracy=inventory_acc,
        latency_ms=latency_ms,
        notes=notes,
    )


def run_fixture_matrix(
    train_examples: Sequence[PlanTrainingExample],
    val_examples: Sequence[PlanTrainingExample],
    *,
    run_id: str = "slm144_fixture",
    output_dir: Path | None = None,
    epochs: int = 40,
    batch_size: int = 8,
) -> Slm144Report:
    """Train the fixture predictor and evaluate every arm on CPU."""
    manifest = build_slm144_manifest()

    t0 = time.perf_counter()
    bundle = train_fixture_predictor(
        train_examples,
        val_examples,
        epochs=epochs,
        batch_size=batch_size,
    )
    train_time_ms = (time.perf_counter() - t0) * 1000.0

    rows: list[Slm144Row] = []
    for arm in manifest.arms:
        row = _evaluate_predictions(arm, train_examples, val_examples, bundle)
        if arm.arm_id in {"serialized_inventory", "set_matching", "gold_archetype", "gold_role_set"}:
            row.notes.append(f"training wall time: {train_time_ms:.1f} ms")
        rows.append(row)

    report = Slm144Report(
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        run_id=run_id,
        status="fixture",
        manifest=manifest,
        rows=rows,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm144_plan_predictor",
        ),
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm144_plan_predictor_report.json")

    return report


def render_markdown(report: Slm144Report) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-144 / SPV1-01: Archetype + role-set predictor fixture matrix ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`  ",
        f"Version: `{report.matrix_version}`  ",
        f"Status: **{report.status}**  ",
        "",
        "**Claim class:** wiring / fixture only. No GPU was used, no production "
        "TwoTower wiring was touched, and no ship-gate claim is made.",
        "",
        "## Manifest",
        "",
        f"Hypothesis: {report.manifest.hypothesis}",
        "",
        "| Arm | Archetype | Role set | Status |",
        "| --- | --- | --- | --- |",
    ]
    for arm in report.manifest.arms:
        lines.append(
            f"| {arm.arm_id} | {arm.archetype_source} | {arm.role_set_source} | {arm.status} |"
        )

    lines.extend(["", "## Results", ""])
    for row in report.rows:
        lines.append(f"### {row.arm_id}")
        lines.append(f"- archetype accuracy: {row.archetype_accuracy}")
        lines.append(f"- role precision: {row.role_precision}")
        lines.append(f"- role recall: {row.role_recall}")
        lines.append(f"- role F1: {row.role_f1}")
        if row.inventory_token_accuracy is not None:
            lines.append(f"- inventory token accuracy: {row.inventory_token_accuracy}")
        for note in row.notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.extend(
        [
            "## Verdict",
            "",
            "Fixture wiring only. The arms exercise baseline, frequency, "
            "learned serialized-inventory, learned set-matching, and oracle "
            "upper-bound code paths on a deterministic 64-example corpus. "
            "Generalization to real OpenUI programs requires a trained model, "
            "held-out suites, and honest ship-gate evaluation.",
            "",
        ]
    )
    return "\n".join(lines)
