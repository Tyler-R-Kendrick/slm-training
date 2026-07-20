"""SLM-172 (SDE2-05): calibrate canonical render-tree and visual-diff surrogates.

Wiring/fixture-only harness that builds deterministic OpenUI (pred, gold) pairs,
runs the tiered ``render_equivalence`` metric, and checks that canonical exact
match, render-tree subscores, and the optional visual-diff surrogate behave as
expected across exact, alpha-renamed, style-only, topology-corrupted,
binding-corrupted, component-substitution, and metric-gaming cases.

No model is trained, no GPU is required, and the default render-equivalence
mode remains ``off``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slm_training.data.semantic_contrast.builder import SemanticContrastBuilder
from slm_training.data.semantic_contrast.schema import ContrastFamily
from slm_training.evals.metric_gaming import build_minimal_valid_trap_cases
from slm_training.evals.render_equivalence import render_equivalence
from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "EXPERIMENT_ID",
    "ARM_NAMES",
    "RenderEquivalenceArm",
    "RenderEquivalenceMetrics",
    "RenderEquivalenceReport",
    "build_cells",
    "validate_manifest",
    "run_fixture_campaign",
    "render_markdown",
    "resolve_disposition",
]

MATRIX_VERSION = "sde2-05-v1"
MATRIX_SET = "slm172_render_equivalence"
EXPERIMENT_ID = "slm172-render-equivalence"

_DEFAULT_SEEDS = (0, 1, 2)

ARM_NAMES = (
    "canonical_exact",
    "alpha_renamed",
    "style_only_change",
    "topology_corruption",
    "binding_corruption",
    "component_substitution",
    "metric_gaming_minimal_valid",
)

_BASE_PROGRAM = (
    'root = Stack([title, cta])\n'
    'title = TextContent(":card.title")\n'
    'cta = Button(":card.action")\n'
)


@dataclass(frozen=True)
class RenderEquivalenceArm:
    """One render-equivalence fixture arm."""

    arm_id: str
    arm_name: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RenderEquivalenceArm":
        return cls(
            arm_id=str(data["arm_id"]),
            arm_name=str(data["arm_name"]),
            seed=int(data["seed"]),
        )


@dataclass(frozen=True)
class RenderEquivalenceMetrics:
    """Per-arm, per-seed render-equivalence fixture metrics."""

    arm_id: str
    arm_name: str
    seed: int
    pred: str
    gold: str
    tier0_canonical_exact: bool
    tier0_binding_graph_equal: bool
    tier1_component_type_match: float
    tier1_role_match: float
    tier1_topology_match: float
    tier1_cardinality_match: float
    tier1_binding_graph_match: float
    tier1_interaction_dependency_match: float
    tier1_normalized_render_tree_distance: float
    tier2_status: str
    tier2_visual_similarity: float | None
    equivalent: bool
    reason_codes: tuple[str, ...]
    wall_seconds: float

    def to_dict(self) -> dict[str, Any]:
        out = dict(asdict(self))
        out["reason_codes"] = list(self.reason_codes)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RenderEquivalenceMetrics":
        return cls(
            arm_id=str(data["arm_id"]),
            arm_name=str(data["arm_name"]),
            seed=int(data["seed"]),
            pred=str(data["pred"]),
            gold=str(data["gold"]),
            tier0_canonical_exact=bool(data["tier0_canonical_exact"]),
            tier0_binding_graph_equal=bool(data["tier0_binding_graph_equal"]),
            tier1_component_type_match=float(data["tier1_component_type_match"]),
            tier1_role_match=float(data["tier1_role_match"]),
            tier1_topology_match=float(data["tier1_topology_match"]),
            tier1_cardinality_match=float(data["tier1_cardinality_match"]),
            tier1_binding_graph_match=float(data["tier1_binding_graph_match"]),
            tier1_interaction_dependency_match=float(
                data["tier1_interaction_dependency_match"]
            ),
            tier1_normalized_render_tree_distance=float(
                data["tier1_normalized_render_tree_distance"]
            ),
            tier2_status=str(data["tier2_status"]),
            tier2_visual_similarity=(
                float(data["tier2_visual_similarity"])
                if data.get("tier2_visual_similarity") is not None
                else None
            ),
            equivalent=bool(data["equivalent"]),
            reason_codes=tuple(data.get("reason_codes", [])),
            wall_seconds=float(data["wall_seconds"]),
        )


@dataclass(frozen=True)
class RenderEquivalenceReport:
    """Full fixture report for SLM-172."""

    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    claim_class: str
    hypothesis: str
    falsifier: str
    cells: tuple[RenderEquivalenceArm, ...]
    rows: list[RenderEquivalenceMetrics]
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
    def from_dict(cls, data: dict[str, Any]) -> "RenderEquivalenceReport":
        return cls(
            matrix_set=data.get("matrix_set", MATRIX_SET),
            matrix_version=data.get("matrix_version", MATRIX_VERSION),
            experiment_id=data.get("experiment_id", EXPERIMENT_ID),
            run_id=data.get("run_id", "slm172_fixture"),
            status=data.get("status", "fixture"),
            claim_class=data.get("claim_class", "wiring"),
            hypothesis=data.get(
                "hypothesis",
                "Canonical AST signature, normalized render-tree overlap, and optional "
                "visual-diff surrogate agree on semantic equivalence and reject "
                "structural corruptions.",
            ),
            falsifier=data.get(
                "falsifier",
                "A structural corruption (topology, binding, component substitution) is "
                "marked equivalent, or a canonical exact match is rejected.",
            ),
            cells=tuple(RenderEquivalenceArm.from_dict(c) for c in data.get("cells", [])),
            rows=[RenderEquivalenceMetrics.from_dict(r) for r in data.get("rows", [])],
            arm_means={k: dict(v) for k, v in data.get("arm_means", {}).items()},
            disposition=data.get("disposition", "inconclusive"),
            disposition_rationale=data.get(
                "disposition_rationale", "no rationale provided"
            ),
            dependency_caveats=list(data.get("dependency_caveats", [])),
            version_stamp=data.get("version_stamp", {}),
        )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _arm_label(arm_name: str, seed: int) -> str:
    return f"{arm_name}__s{seed}"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _build_semantic_contrast_pairs(seed: int) -> dict[str, tuple[str, str]]:
    """Build a small semantic-contrast corpus and return positive/negative pairs.

    Returns a mapping from corruption family to ``(positive_openui, negative_openui)``.
    Falls back to hand-crafted examples if the builder produces no admitted pair
    for a requested family.
    """
    builder = SemanticContrastBuilder(
        output_root="outputs/data",
        dataset_id=f"slm172_render_equivalence_fixture_s{seed}",
        seed=seed,
        source_count=4,
        splits=("test",),
        split_weights=(1.0,),
    )
    try:
        builder.build()
    except Exception:
        pass
    pairs_path = builder.output_dir / "pairs.jsonl"
    pairs: dict[str, tuple[str, str]] = {}
    if pairs_path.is_file():
        for line in pairs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("admitted"):
                continue
            family = str(record.get("family") or "")
            positive = (
                record.get("positive", {}).get("record", {}).get("openui", "")
            )
            negative = (
                record.get("negative", {}).get("record", {}).get("openui", "")
            )
            if positive and negative and family not in pairs:
                pairs[family] = (positive, negative)
    # Fallbacks in case the builder is empty or missing a family.
    if ContrastFamily.TOPOLOGY.value not in pairs:
        positive = _BASE_PROGRAM
        negative = (
            'root = Stack([Card([title]), cta])\n'
            'title = TextContent(":card.title")\n'
            'cta = Button(":card.action")\n'
        )
        pairs[ContrastFamily.TOPOLOGY.value] = (positive, negative)
    if ContrastFamily.BINDING.value not in pairs:
        positive = _BASE_PROGRAM
        negative = (
            'root = Stack([title, cta])\n'
            'title = TextContent(":card.title")\n'
            'cta = Button(":card.action")\n'
            'dead = TextContent(":dead.text")\n'
        )
        pairs[ContrastFamily.BINDING.value] = (positive, negative)
    return pairs


def _build_metric_gaming_pair(seed: int) -> tuple[str, str]:
    """Return a hard-valid but semantically insufficient (pred, gold) pair."""
    cases = build_minimal_valid_trap_cases(seed)
    for case in cases:
        if case.expected_verdict is False and case.gold_openui:
            return (case.pred_openui, case.gold_openui)
    # Fallback if the fixture shape changes.
    return (
        'root = Stack([TextContent(":placeholder.text")])\n',
        _BASE_PROGRAM,
    )


def _build_pair(arm_name: str, seed: int) -> tuple[str, str]:
    """Return the (pred, gold) OpenUI pair for one arm."""
    if arm_name == "canonical_exact":
        return (_BASE_PROGRAM, _BASE_PROGRAM)

    if arm_name == "alpha_renamed":
        pred = (
            'root = Stack([t, c])\n'
            't = TextContent(":card.title")\n'
            'c = Button(":card.action")\n'
        )
        return (pred, _BASE_PROGRAM)

    if arm_name == "style_only_change":
        pred = (
            'root = Stack([title, cta])\n'
            'title = TextContent(":card.title")\n'
            'cta = Button(":card.action", "primary")\n'
        )
        return (pred, _BASE_PROGRAM)

    if arm_name == "component_substitution":
        pred = (
            'root = Stack([title, cta])\n'
            'title = TextContent(":card.title")\n'
            'cta = TextContent(":card.action")\n'
        )
        return (pred, _BASE_PROGRAM)

    pairs = _build_semantic_contrast_pairs(seed)

    if arm_name == "topology_corruption":
        return pairs.get(
            ContrastFamily.TOPOLOGY.value,
            (_BASE_PROGRAM, _BASE_PROGRAM),
        )

    if arm_name == "binding_corruption":
        return pairs.get(
            ContrastFamily.BINDING.value,
            (_BASE_PROGRAM, _BASE_PROGRAM),
        )

    if arm_name == "metric_gaming_minimal_valid":
        return _build_metric_gaming_pair(seed)

    raise ValueError(f"unknown arm_name: {arm_name!r}")


def _run_arm(arm: RenderEquivalenceArm) -> RenderEquivalenceMetrics:
    start = time.perf_counter()
    pred, gold = _build_pair(arm.arm_name, arm.seed)
    report = render_equivalence(pred, gold)
    elapsed = time.perf_counter() - start
    return RenderEquivalenceMetrics(
        arm_id=arm.arm_id,
        arm_name=arm.arm_name,
        seed=arm.seed,
        pred=pred,
        gold=gold,
        tier0_canonical_exact=report.tier0.canonical_exact,
        tier0_binding_graph_equal=report.tier0.binding_graph_equal,
        tier1_component_type_match=report.tier1.component_type_match,
        tier1_role_match=report.tier1.role_match,
        tier1_topology_match=report.tier1.topology_match,
        tier1_cardinality_match=report.tier1.cardinality_match,
        tier1_binding_graph_match=report.tier1.binding_graph_match,
        tier1_interaction_dependency_match=report.tier1.interaction_dependency_match,
        tier1_normalized_render_tree_distance=report.tier1.normalized_render_tree_distance,
        tier2_status=report.tier2.status,
        tier2_visual_similarity=report.tier2.visual_similarity,
        equivalent=report.equivalent,
        reason_codes=report.reason_codes,
        wall_seconds=_clamp(elapsed + 0.001, low=0.001, high=10.0),
    )


def build_cells(seeds: tuple[int, ...] = _DEFAULT_SEEDS) -> tuple[RenderEquivalenceArm, ...]:
    """Build the arm × seeds cells for the fixture."""
    cells: list[RenderEquivalenceArm] = []
    for seed in seeds:
        for arm_name in ARM_NAMES:
            cells.append(
                RenderEquivalenceArm(
                    arm_id=_arm_label(arm_name, seed),
                    arm_name=arm_name,
                    seed=seed,
                )
            )
    return tuple(cells)


def validate_manifest(cells: tuple[RenderEquivalenceArm, ...]) -> list[str]:
    """Validate the render-equivalence manifest."""
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
    return errors


def _arm_means(rows: list[RenderEquivalenceMetrics]) -> dict[str, dict[str, float]]:
    """Aggregate per-arm means across seeds."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        bucket = grouped.setdefault(row.arm_name, {})
        for key in (
            "tier0_canonical_exact",
            "tier0_binding_graph_equal",
            "tier1_component_type_match",
            "tier1_role_match",
            "tier1_topology_match",
            "tier1_cardinality_match",
            "tier1_binding_graph_match",
            "tier1_interaction_dependency_match",
            "tier1_normalized_render_tree_distance",
            "equivalent",
            "wall_seconds",
        ):
            bucket.setdefault(key, []).append(float(getattr(row, key) or 0.0))
    return {
        arm: {key: sum(values) / len(values) for key, values in metrics.items()}
        for arm, metrics in grouped.items()
    }


def resolve_disposition(
    arm_means: dict[str, dict[str, float]]
) -> tuple[str, str]:
    """Return (disposition, rationale) from per-arm means."""
    exact = arm_means.get("canonical_exact", {}).get("equivalent", 0.0)
    alpha = arm_means.get("alpha_renamed", {}).get("equivalent", 0.0)
    style = arm_means.get("style_only_change", {}).get("equivalent", 0.0)

    if exact < 1.0 or alpha < 1.0 or style < 1.0:
        return (
            "canonical_signature_unreliable",
            "Canonical-exact / alpha-renamed / style-only pairs were not all marked "
            "equivalent; the tier-0 signature is not calibrated.",
        )

    corrupted_arms = (
        "topology_corruption",
        "binding_corruption",
        "component_substitution",
        "metric_gaming_minimal_valid",
    )
    leak = any(arm_means.get(arm, {}).get("equivalent", 0.0) > 0.0 for arm in corrupted_arms)
    if leak:
        return (
            "semantic_leak",
            "At least one corrupted arm was marked equivalent; the surrogates are too "
            "permissive.",
        )

    return (
        "calibrated",
        "Canonical exact, alpha-renamed, and style-only pairs are equivalent, while "
        "all structural corruptions and metric-gaming traps are rejected.",
    )


def run_fixture_campaign(
    cells: tuple[RenderEquivalenceArm, ...] | None = None,
    *,
    run_id: str = "slm172-render-equivalence",
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
) -> RenderEquivalenceReport:
    """Run the SLM-172 render-equivalence fixture campaign."""
    cells = cells or build_cells(seeds)
    errors = validate_manifest(cells)
    if errors:
        raise ValueError("manifest validation failed: " + "; ".join(errors))

    rows: list[RenderEquivalenceMetrics] = []
    for cell in cells:
        rows.append(_run_arm(cell))

    means = _arm_means(rows)
    disposition, rationale = resolve_disposition(means)

    report = RenderEquivalenceReport(
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
        status="fixture",
        claim_class="wiring",
        hypothesis=(
            "Canonical AST signature, normalized render-tree overlap, and optional "
            "visual-diff surrogate agree on semantic equivalence and reject structural "
            "corruptions."
        ),
        falsifier=(
            "A structural corruption (topology, binding, component substitution) is "
            "marked equivalent, or a canonical exact match is rejected."
        ),
        cells=cells,
        rows=rows,
        arm_means=means,
        disposition=disposition,
        disposition_rationale=rationale,
        dependency_caveats=[
            "Depends on slm_training.evals.render_equivalence; tier-2 visual diff is "
            "capability-gated and reports not_available when Playwright/chromium is "
            "unavailable.",
            "Semantic-contrast pairs are sampled from a small deterministic builder "
            "corpus; real OOD distributions may differ.",
            "No model is trained; this is wiring evidence only.",
        ],
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm172_render_equivalence",
            "evals.render_equivalence",
        ),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm172_render_equivalence_report.json")
    return report


def render_markdown(report: RenderEquivalenceReport) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-172 (SDE2-05): render-equivalence fixture ({report.run_id})",
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
        "## Render-equivalence arms",
        "",
        "| arm_id | arm_name | seed |",
        "| --- | --- | --- |",
    ]
    for cell in report.cells:
        lines.append(f"| {cell.arm_id} | {cell.arm_name} | {cell.seed} |")

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| arm_id | arm_name | seed | canonical_exact | binding_graph_equal | "
            "component_type | role | topology | cardinality | binding_graph | "
            "interaction_dep | render_tree_dist | tier2_status | equivalent | wall_seconds |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.rows:
        lines.append(
            f"| {row.arm_id} | {row.arm_name} | {row.seed} | "
            f"{int(row.tier0_canonical_exact)} | {int(row.tier0_binding_graph_equal)} | "
            f"{row.tier1_component_type_match:.3f} | {row.tier1_role_match:.3f} | "
            f"{row.tier1_topology_match:.3f} | {row.tier1_cardinality_match:.3f} | "
            f"{row.tier1_binding_graph_match:.3f} | "
            f"{row.tier1_interaction_dependency_match:.3f} | "
            f"{row.tier1_normalized_render_tree_distance:.3f} | "
            f"{row.tier2_status} | {int(row.equivalent)} | {row.wall_seconds:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Per-arm means",
            "",
            "| arm_name | equivalent_rate | canonical_exact | binding_graph_equal | "
            "component_type | role | topology | cardinality | binding_graph | "
            "interaction_dep | render_tree_dist |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for arm_name in ARM_NAMES:
        if arm_name not in report.arm_means:
            continue
        m = report.arm_means[arm_name]
        lines.append(
            f"| {arm_name} | {m.get('equivalent', 0.0):.3f} | "
            f"{m.get('tier0_canonical_exact', 0.0):.3f} | "
            f"{m.get('tier0_binding_graph_equal', 0.0):.3f} | "
            f"{m.get('tier1_component_type_match', 0.0):.3f} | "
            f"{m.get('tier1_role_match', 0.0):.3f} | "
            f"{m.get('tier1_topology_match', 0.0):.3f} | "
            f"{m.get('tier1_cardinality_match', 0.0):.3f} | "
            f"{m.get('tier1_binding_graph_match', 0.0):.3f} | "
            f"{m.get('tier1_interaction_dependency_match', 0.0):.3f} | "
            f"{m.get('tier1_normalized_render_tree_distance', 0.0):.3f} |"
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
            "**No-go for promotion.** This is a wiring fixture. The render-equivalence "
            "surrogates are exercised over deterministic synthetic and contrast pairs, "
            "but no real model was trained or evaluated. The mechanism remains "
            "``retain_diagnostic`` / ``blocked_pending_real_model`` until a trained model "
            "and AgentV evaluation are available.",
            "",
            "## Honest caveats",
            "",
            "- Tier-2 visual diff is capability-gated; in most CI/fixture environments it "
            "  reports ``not_available``.",
            "- Semantic-contrast pairs come from a small deterministic builder corpus, not "
            "  a trained model or real user distribution.",
            "- Component substitution and metric-gaming cases are hand-selected traps, not "
            "  a representative sample.",
            "- No ship-gate claim is made; this is wiring evidence only.",
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_slm172_render_equivalence_fixture --mode plan-only",
            "python -m scripts.run_slm172_render_equivalence_fixture --mode fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
