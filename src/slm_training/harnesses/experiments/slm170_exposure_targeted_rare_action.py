"""SLM-170 (SDE2-03): exposure-targeted rare-action sampling wiring/fixture harness.

Measures whether an action-exposure objective can up-weight rare grammar actions
within a fixed total decision budget while respecting per-root and per-template
diversity caps.  The fixture uses synthetic ``ExampleRecord`` rows with skewed
action counts and runs ``sample_mixture_batch`` under the new
``exposure_targeted`` policy.

This is a wiring fixture: no model is trained, no GPU is required, and the
default ``with_replacement`` sampler remains unchanged.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slm_training.data.mixture import (
    build_exposure_ledger,
    record_action_counts,
    sample_mixture_batch,
)
from slm_training.dsl.schema import ExampleRecord
from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "EXPERIMENT_ID",
    "ARM_NAMES",
    "RareActionArm",
    "RareActionMetrics",
    "RareActionReport",
    "build_cells",
    "validate_manifest",
    "run_fixture_campaign",
    "render_markdown",
    "resolve_disposition",
]

MATRIX_VERSION = "sde2-03-v1"
MATRIX_SET = "slm170_exposure_targeted_rare_action"
EXPERIMENT_ID = "slm170-exposure-targeted-rare-action"

_DEFAULT_SEEDS = (0, 1, 2)
_DEFAULT_TOTAL_DECISION_BUDGET = 64
_DEFAULT_PER_ROOT_CAP = 4
_DEFAULT_PER_TEMPLATE_CAP = 4
_DEFAULT_MAX_IMPORTANCE_WEIGHT = 10.0

ARM_NAMES = (
    "current",
    "e396_balanced",
    "sqrt_inverse_frequency",
    "minimum_exposure_floor",
    "root_template_balanced_control",
)

# Synthetic component inventory skewed toward common/rare actions.
_COMMON_ACTIONS = ("Button", "TextContent", "Card")
_RARE_ACTIONS = ("ImageGallery", "Chart", "Map", "Calendar")


def _project_root() -> Path:
    """Return the repository root relative to this module."""
    return Path(__file__).resolve().parents[4]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class RareActionArm:
    """One exposure-targeted sampling arm plus derived recipe fields."""

    arm_id: str
    arm_name: str
    policy: str
    seed: int
    total_decision_budget: int
    per_root_cap: int
    per_template_cap: int
    max_importance_weight: float
    action_targets: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RareActionArm":
        return cls(
            arm_id=str(data["arm_id"]),
            arm_name=str(data["arm_name"]),
            policy=str(data["policy"]),
            seed=int(data["seed"]),
            total_decision_budget=int(data["total_decision_budget"]),
            per_root_cap=int(data["per_root_cap"]),
            per_template_cap=int(data["per_template_cap"]),
            max_importance_weight=float(data["max_importance_weight"]),
            action_targets=(
                dict(data["action_targets"]) if data.get("action_targets") else None
            ),
        )


@dataclass(frozen=True)
class RareActionMetrics:
    """Per-arm, per-seed synthetic fixture metrics."""

    arm_id: str
    arm_name: str
    policy: str
    seed: int
    total_decision_budget: int
    sampled_records: int
    observed_decisions: int
    rare_action_exposure: float
    common_action_exposure: float
    rare_to_common_ratio: float
    unique_roots: int
    unique_templates: int
    budget_adherence: float
    cap_violations: int
    rare_action_recall: float
    wall_seconds: float
    notes: list[str] = field(default_factory=list)
    ledger: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        out = dict(asdict(self))
        out["ledger"] = self.ledger
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RareActionMetrics":
        return cls(
            arm_id=str(data["arm_id"]),
            arm_name=str(data["arm_name"]),
            policy=str(data["policy"]),
            seed=int(data["seed"]),
            total_decision_budget=int(data["total_decision_budget"]),
            sampled_records=int(data["sampled_records"]),
            observed_decisions=int(data["observed_decisions"]),
            rare_action_exposure=float(data["rare_action_exposure"]),
            common_action_exposure=float(data["common_action_exposure"]),
            rare_to_common_ratio=float(data["rare_to_common_ratio"]),
            unique_roots=int(data["unique_roots"]),
            unique_templates=int(data["unique_templates"]),
            budget_adherence=float(data["budget_adherence"]),
            cap_violations=int(data["cap_violations"]),
            rare_action_recall=float(data["rare_action_recall"]),
            wall_seconds=float(data["wall_seconds"]),
            notes=list(data.get("notes", [])),
            ledger=dict(data.get("ledger", {})),
        )


@dataclass(frozen=True)
class RareActionReport:
    """Full fixture report for SLM-170."""

    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    claim_class: str
    hypothesis: str
    falsifier: str
    cells: tuple[RareActionArm, ...]
    rows: list[RareActionMetrics]
    arm_means: dict[str, dict[str, float]]
    disposition: str
    disposition_rationale: str
    dependency_caveats: list[str]
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "hypothesis": self.hypothesis,
            "falsifier": self.falsifier,
            "cells": [cell.to_dict() for cell in self.cells],
            "rows": [row.to_dict() for row in self.rows],
            "arm_means": {k: dict(v) for k, v in self.arm_means.items()},
            "disposition": self.disposition,
            "disposition_rationale": self.disposition_rationale,
            "dependency_caveats": list(self.dependency_caveats),
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RareActionReport":
        return cls(
            matrix_set=data.get("matrix_set", MATRIX_SET),
            matrix_version=data.get("matrix_version", MATRIX_VERSION),
            experiment_id=data.get("experiment_id", EXPERIMENT_ID),
            run_id=data.get("run_id", "slm170_fixture"),
            status=data.get("status", "fixture"),
            claim_class=data.get("claim_class", "wiring"),
            hypothesis=data.get(
                "hypothesis",
                "Exposure-targeted sampling with bounded importance weights and diversity "
                "caps increases rare-action exposure within a fixed total decision budget.",
            ),
            falsifier=data.get(
                "falsifier",
                "Exposure-targeted sampling fails to increase rare-action exposure or "
                "violates the total budget / diversity caps.",
            ),
            cells=tuple(RareActionArm.from_dict(c) for c in data.get("cells", [])),
            rows=[RareActionMetrics.from_dict(r) for r in data.get("rows", [])],
            arm_means={k: dict(v) for k, v in data.get("arm_means", {}).items()},
            disposition=data.get("disposition", "inconclusive"),
            disposition_rationale=data.get(
                "disposition_rationale", "no rationale provided"
            ),
            dependency_caveats=list(data.get("dependency_caveats", [])),
            version_stamp=data.get("version_stamp", {}),
        )


def _arm_label(arm_name: str, seed: int) -> str:
    return f"{arm_name}__s{seed}"


def _build_synthetic_records(
    *,
    n_common: int = 40,
    n_rare_each: int = 3,
    seed: int = 0,
) -> list[ExampleRecord]:
    """Build deterministic records with heavily skewed action counts."""
    rng = random.Random(seed)
    records: list[ExampleRecord] = []
    root_id = 0
    template_id = 0

    def _record(action: str, rid: str) -> ExampleRecord:
        nonlocal root_id, template_id
        root_id += 1
        template_id += 1
        return ExampleRecord(
            id=rid,
            prompt=f"prompt for {action} {rid}",
            openui=f'root = {action}("value")',
            split="train",
            source="fixture",
            meta={
                "source_family": "fixture",
                "root_parent_id": f"root_{root_id // 2:04d}",
                "parent_id": f"template_{template_id // 2:04d}",
            },
        )

    # Common actions: many records each.
    for i in range(n_common):
        action = rng.choice(_COMMON_ACTIONS)
        records.append(_record(action, f"common_{i:03d}"))

    # Rare actions: only a few records each.
    for action in _RARE_ACTIONS:
        for i in range(n_rare_each):
            records.append(_record(action, f"rare_{action}_{i:02d}"))

    rng.shuffle(records)
    return records


def _frequency_targets(records: list[ExampleRecord], budget: int) -> dict[str, float]:
    """Sqrt-inverse-frequency targets capped at the budget share."""
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record_action_counts(record))
    if not counts:
        return {}
    raw = {a: 1.0 / math.sqrt(max(1, c)) for a, c in counts.items()}
    total = sum(raw.values()) or 1.0
    return {a: budget * v / total for a, v in raw.items()}


def _balanced_targets(budget: int) -> dict[str, float]:
    """Uniform targets across the synthetic action vocabulary."""
    actions = list(_COMMON_ACTIONS) + list(_RARE_ACTIONS)
    share = budget / len(actions)
    return {a: share for a in actions}


def _floor_targets(records: list[ExampleRecord], budget: int) -> dict[str, float]:
    """Balanced targets with a hard floor so every rare action is covered."""
    floor = budget / (len(_COMMON_ACTIONS) + len(_RARE_ACTIONS))
    targets = _balanced_targets(budget)
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record_action_counts(record))
    for action in _RARE_ACTIONS:
        targets[action] = max(targets.get(action, 0.0), floor * 1.5)
    # Re-normalize to the budget.
    total = sum(targets.values()) or 1.0
    return {a: budget * v / total for a, v in targets.items()}


def _run_arm(arm: RareActionArm, records: list[ExampleRecord]) -> RareActionMetrics:
    """Sample one arm and build its exposure ledger."""
    start = time.perf_counter()
    rng = random.Random(arm.seed)

    if arm.policy == "exposure_targeted":
        action_targets = arm.action_targets
        sampled = sample_mixture_batch(
            records,
            weights={"fixture": 1.0},
            batch_size=arm.total_decision_budget,
            rng=rng,
            sampling_policy="exposure_targeted",
            action_targets=action_targets,
            total_decision_budget=arm.total_decision_budget,
            per_root_cap=arm.per_root_cap,
            per_template_cap=arm.per_template_cap,
            max_importance_weight=arm.max_importance_weight,
        )
    else:
        sampled = sample_mixture_batch(
            records,
            weights={"fixture": 1.0},
            batch_size=arm.total_decision_budget,
            rng=rng,
            sampling_policy="with_replacement",
        )

    selected_ids = {r.id for r in sampled}
    ledger = build_exposure_ledger(records, selected_ids=selected_ids)
    observed = ledger.get("observed_decisions_per_run", {})
    rare_total = sum(observed.get(a, 0) for a in _RARE_ACTIONS)
    common_total = sum(observed.get(a, 0) for a in _COMMON_ACTIONS)
    rare_recall = sum(1 for a in _RARE_ACTIONS if observed.get(a, 0) > 0) / len(
        _RARE_ACTIONS
    )

    roots: set[str] = set()
    templates: set[str] = set()
    cap_violations = 0
    root_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    for record in sampled:
        meta = record.meta or {}
        root = str(meta.get("root_parent_id") or record.id)
        template = str(meta.get("parent_id") or meta.get("source_family") or record.source)
        roots.add(root)
        templates.add(template)
        root_counts[root] += 1
        template_counts[template] += 1
        if root_counts[root] > arm.per_root_cap:
            cap_violations += 1
        if template_counts[template] > arm.per_template_cap:
            cap_violations += 1

    elapsed = time.perf_counter() - start
    wall_seconds = _clamp(
        elapsed + 0.001 * len(sampled), low=0.001, high=10.0
    )

    notes = [
        f"policy={arm.policy}",
        f"total_decision_budget={arm.total_decision_budget}",
        "fixture-only: synthetic rare-action exposure comparison",
    ]
    if arm.policy == "exposure_targeted":
        notes.append(f"max_importance_weight={arm.max_importance_weight}")

    return RareActionMetrics(
        arm_id=arm.arm_id,
        arm_name=arm.arm_name,
        policy=arm.policy,
        seed=arm.seed,
        total_decision_budget=arm.total_decision_budget,
        sampled_records=len(sampled),
        observed_decisions=ledger.get("aggregate", {}).get("selected_decisions", 0),
        rare_action_exposure=rare_total,
        common_action_exposure=common_total,
        rare_to_common_ratio=(
            rare_total / max(1, common_total) / (len(_RARE_ACTIONS) / len(_COMMON_ACTIONS))
        ),
        unique_roots=len(roots),
        unique_templates=len(templates),
        budget_adherence=len(sampled) / max(1, arm.total_decision_budget),
        cap_violations=cap_violations,
        rare_action_recall=rare_recall,
        wall_seconds=wall_seconds,
        notes=notes,
        ledger=ledger,
    )


def build_cells(
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    *,
    total_decision_budget: int = _DEFAULT_TOTAL_DECISION_BUDGET,
    per_root_cap: int = _DEFAULT_PER_ROOT_CAP,
    per_template_cap: int = _DEFAULT_PER_TEMPLATE_CAP,
    max_importance_weight: float = _DEFAULT_MAX_IMPORTANCE_WEIGHT,
) -> tuple[RareActionArm, ...]:
    """Build the policy × seeds cells for the fixture."""
    cells: list[RareActionArm] = []
    for seed in seeds:
        for arm_name in ARM_NAMES:
            policy = "with_replacement" if arm_name == "current" else "exposure_targeted"
            cells.append(
                RareActionArm(
                    arm_id=_arm_label(arm_name, seed),
                    arm_name=arm_name,
                    policy=policy,
                    seed=seed,
                    total_decision_budget=total_decision_budget,
                    per_root_cap=per_root_cap,
                    per_template_cap=per_template_cap,
                    max_importance_weight=max_importance_weight,
                    action_targets=None,
                )
            )
    return tuple(cells)


def validate_manifest(cells: tuple[RareActionArm, ...]) -> list[str]:
    """Validate the rare-action sampling manifest."""
    errors: list[str] = []
    if not cells:
        errors.append("cells must not be empty")
    seen: set[str] = set()
    for cell in cells:
        if cell.arm_id in seen:
            errors.append(f"duplicate arm_id: {cell.arm_id}")
        seen.add(cell.arm_id)
        if cell.arm_name not in ARM_NAMES:
            errors.append(f"{cell.arm_id}: invalid arm_name {cell.arm_name!r}")
        if cell.policy not in {"with_replacement", "exposure_targeted"}:
            errors.append(f"{cell.arm_id}: invalid policy {cell.policy!r}")
        if cell.total_decision_budget <= 0:
            errors.append(f"{cell.arm_id}: total_decision_budget must be positive")
        if cell.per_root_cap <= 0 or cell.per_template_cap <= 0:
            errors.append(f"{cell.arm_id}: caps must be positive")
        if cell.max_importance_weight < 1.0:
            errors.append(f"{cell.arm_id}: max_importance_weight must be at least 1.0")
    return errors


def _arm_means(rows: list[RareActionMetrics]) -> dict[str, dict[str, float]]:
    """Aggregate per-arm means across seeds."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        bucket = grouped.setdefault(row.arm_name, {})
        for key in (
            "sampled_records",
            "observed_decisions",
            "rare_action_exposure",
            "common_action_exposure",
            "rare_to_common_ratio",
            "unique_roots",
            "unique_templates",
            "budget_adherence",
            "cap_violations",
            "rare_action_recall",
            "wall_seconds",
        ):
            bucket.setdefault(key, []).append(float(getattr(row, key)))
    return {
        arm: {key: statistics.mean(values) for key, values in metrics.items()}
        for arm, metrics in grouped.items()
    }


def resolve_disposition(
    arm_means: dict[str, dict[str, float]]
) -> tuple[str, str]:
    """Return (disposition, rationale) from the per-arm means."""
    current = arm_means.get("current", {})
    balanced = arm_means.get("e396_balanced", {})
    sqrt_inv = arm_means.get("sqrt_inverse_frequency", {})
    floor = arm_means.get("minimum_exposure_floor", {})
    control = arm_means.get("root_template_balanced_control", {})

    current_recall = current.get("rare_action_recall", 0.0)
    best_recall = max(
        balanced.get("rare_action_recall", 0.0),
        sqrt_inv.get("rare_action_recall", 0.0),
        floor.get("rare_action_recall", 0.0),
        control.get("rare_action_recall", 0.0),
    )
    best_ratio = max(
        balanced.get("rare_to_common_ratio", 0.0),
        sqrt_inv.get("rare_to_common_ratio", 0.0),
        floor.get("rare_to_common_ratio", 0.0),
        control.get("rare_to_common_ratio", 0.0),
    )
    any_caps = any(
        arm_means.get(arm, {}).get("cap_violations", 0.0) > 0
        for arm in ARM_NAMES
        if arm != "current"
    )

    if any_caps:
        return (
            "inconclusive",
            "At least one exposure-targeted arm violated a per-root or per-template cap; "
            "the wiring needs a tighter bound before claiming a rare-action win.",
        )
    if best_recall <= current_recall + 0.05 and best_ratio <= 1.05:
        return (
            "no_exposure_lift",
            "Exposure-targeted arms do not materially increase rare-action recall or "
            "rare-to-common exposure ratio versus the with_replacement baseline.",
        )
    if best_ratio >= 1.5 and best_recall >= current_recall + 0.20:
        return (
            "useful_rare_action_exposure",
            "Exposure-targeted sampling materially increases rare-action exposure and "
            "recall within the fixed budget and diversity caps.",
        )
    if best_recall >= current_recall + 0.10:
        return (
            "modest_rare_action_lift",
            "Exposure-targeted sampling shows a measurable rare-action recall lift but "
            "the rare-to-common ratio improvement is modest.",
        )
    return (
        "inconclusive",
        "The exposure-targeted pattern is mixed; additional seeds or a larger synthetic "
        "corpus are needed to falsify the hypothesis.",
    )


def run_fixture_campaign(
    cells: tuple[RareActionArm, ...] | None = None,
    *,
    run_id: str = "slm170-exposure-targeted-rare-action",
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    n_common: int = 40,
    n_rare_each: int = 3,
) -> RareActionReport:
    """Run the SLM-170 exposure-targeted rare-action sampling fixture campaign."""
    cells = cells or build_cells(seeds)
    errors = validate_manifest(cells)
    if errors:
        raise ValueError("manifest validation failed: " + "; ".join(errors))

    records = _build_synthetic_records(n_common=n_common, n_rare_each=n_rare_each)

    rows: list[RareActionMetrics] = []
    for cell in cells:
        # Resolve action targets per arm when not explicitly provided.
        if cell.policy == "exposure_targeted" and cell.action_targets is None:
            if cell.arm_name == "e396_balanced":
                action_targets = _balanced_targets(cell.total_decision_budget)
            elif cell.arm_name == "sqrt_inverse_frequency":
                action_targets = _frequency_targets(records, cell.total_decision_budget)
            elif cell.arm_name == "minimum_exposure_floor":
                action_targets = _floor_targets(records, cell.total_decision_budget)
            elif cell.arm_name == "root_template_balanced_control":
                action_targets = _balanced_targets(cell.total_decision_budget)
            else:
                action_targets = _frequency_targets(records, cell.total_decision_budget)
            cell = RareActionArm(
                arm_id=cell.arm_id,
                arm_name=cell.arm_name,
                policy=cell.policy,
                seed=cell.seed,
                total_decision_budget=cell.total_decision_budget,
                per_root_cap=cell.per_root_cap,
                per_template_cap=cell.per_template_cap,
                max_importance_weight=cell.max_importance_weight,
                action_targets=action_targets,
            )
        rows.append(_run_arm(cell, records))

    means = _arm_means(rows)
    disposition, rationale = resolve_disposition(means)

    hypothesis = (
        "Exposure-targeted sampling with bounded importance weights and diversity "
        "caps increases rare-action exposure within a fixed total decision budget."
    )
    falsifier = (
        "Exposure-targeted sampling fails to increase rare-action exposure or "
        "violates the total budget / diversity caps."
    )

    report = RareActionReport(
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
        status="fixture",
        claim_class="wiring",
        hypothesis=hypothesis,
        falsifier=falsifier,
        cells=cells,
        rows=rows,
        arm_means=means,
        disposition=disposition,
        disposition_rationale=rationale,
        dependency_caveats=[
            "Depends on slm_training.data.mixture exposure_targeted sampler.",
            "Synthetic records use a fixed action vocabulary; real corpus rare-action "
            "distributions may differ.",
            "No model is trained; this is wiring evidence only.",
        ],
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm170_exposure_targeted_rare_action",
        ),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm170_exposure_targeted_rare_action_report.json")
    return report


def render_markdown(report: RareActionReport) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-170 (SDE2-03): exposure-targeted rare-action sampling fixture ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`",
        "",
        f"Version: `{report.matrix_version}`",
        "",
        f"Status: **{report.status}**",
        "",
        "**Claim class:** wiring / fixture only. No GPU was used, no trainable weights "
        "were updated, and no ship-gate claim is made.",
        "",
        "## Hypothesis",
        "",
        report.hypothesis,
        "",
        "## Falsifier",
        "",
        report.falsifier,
        "",
        "## Sampling arms",
        "",
        "| arm_id | arm_name | policy | seed | total_decision_budget | per_root_cap | per_template_cap | max_importance_weight |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in report.cells:
        lines.append(
            f"| {cell.arm_id} | {cell.arm_name} | {cell.policy} | {cell.seed} | "
            f"{cell.total_decision_budget} | {cell.per_root_cap} | "
            f"{cell.per_template_cap} | {cell.max_importance_weight} |"
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| arm_id | arm_name | seed | sampled_records | observed_decisions | rare_exposure | "
            "common_exposure | rare_recall | unique_roots | unique_templates | cap_violations | "
            "budget_adherence | wall_seconds |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.rows:
        lines.append(
            f"| {row.arm_id} | {row.arm_name} | {row.seed} | {row.sampled_records} | "
            f"{row.observed_decisions} | {row.rare_action_exposure:.1f} | "
            f"{row.common_action_exposure:.1f} | {row.rare_action_recall:.3f} | "
            f"{row.unique_roots} | {row.unique_templates} | {row.cap_violations} | "
            f"{row.budget_adherence:.3f} | {row.wall_seconds:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Per-arm means",
            "",
            "| arm_name | sampled_records | rare_exposure | common_exposure | rare_recall | "
            "rare_to_common_ratio | unique_roots | unique_templates | cap_violations |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for arm_name in ARM_NAMES:
        if arm_name not in report.arm_means:
            continue
        m = report.arm_means[arm_name]
        lines.append(
            f"| {arm_name} | {m.get('sampled_records', 0.0):.1f} | "
            f"{m.get('rare_action_exposure', 0.0):.1f} | "
            f"{m.get('common_action_exposure', 0.0):.1f} | "
            f"{m.get('rare_action_recall', 0.0):.3f} | "
            f"{m.get('rare_to_common_ratio', 0.0):.3f} | "
            f"{m.get('unique_roots', 0.0):.1f} | "
            f"{m.get('unique_templates', 0.0):.1f} | "
            f"{m.get('cap_violations', 0.0):.1f} |"
        )

    lines.extend(
        [
            "",
            "## Disposition",
            "",
            f"**{report.disposition}**",
            "",
            report.disposition_rationale,
            "",
            "## Go / no-go decision",
            "",
            "**No-go for promotion.** This is a wiring fixture. The sampler, targets, "
            "and caps are exercised over deterministic synthetic records, but no real "
            "model was trained or evaluated. The mechanism remains "
            "``retain_diagnostic`` / ``blocked_pending_real_model`` until a trained "
            "model and AgentV evaluation are available.",
            "",
            "## Honest caveats",
            "",
            "- Metrics are generated by a deterministic synthetic corpus, not a trained model.",
            "- Action counts are derived from a regex over the synthetic OpenUI target; "
            "  real records may use a richer action vocabulary.",
            "- Per-root and per-template caps are synthetic; real corpus lineage fields "
            "  may require different cap semantics.",
            "- No ship-gate claim is made; this is wiring evidence only.",
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_slm170_exposure_targeted_rare_action_fixture --mode plan-only",
            "python -m scripts.run_slm170_exposure_targeted_rare_action_fixture --mode fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
