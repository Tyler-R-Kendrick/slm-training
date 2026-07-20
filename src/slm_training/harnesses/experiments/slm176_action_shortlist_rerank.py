"""SLM-176 (P14): description-based retrieve-then-rerank over live legal actions.

Wiring/fixture-only harness that checks whether a deterministic description
retrieval shortlist preserves the top full-set candidate for synthetic legal
sets.  No model is trained and no GPU is required.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from slm_training.dsl.action_descriptions import (
    ActionDescription,
    ActionDescriptionCatalog,
    FixtureDescriptionEncoder,
)
from slm_training.dsl.action_shortlist import (
    ActionShortlistPolicy,
    build_query_vector,
    retrieve_then_rerank,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "EXPERIMENT_ID",
    "ShortlistScenario",
    "ShortlistMetrics",
    "ShortlistReport",
    "build_cells",
    "validate_manifest",
    "run_fixture_campaign",
    "render_markdown",
]

MATRIX_VERSION = "p14-v1"
MATRIX_SET = "slm176_action_shortlist_rerank"
EXPERIMENT_ID = "slm176-action-shortlist-rerank"

_DEFAULT_SEEDS = (0, 1, 2)
_DEFAULT_D_MODEL = 64


@dataclass(frozen=True)
class ShortlistScenario:
    """One synthetic legal-set / query scenario."""

    scenario_id: str
    legal_set_size: int
    k: int
    seed: int
    query_hint: str

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShortlistScenario":
        return cls(
            scenario_id=str(data["scenario_id"]),
            legal_set_size=int(data["legal_set_size"]),
            k=int(data["k"]),
            seed=int(data["seed"]),
            query_hint=str(data["query_hint"]),
        )


@dataclass(frozen=True)
class ShortlistMetrics:
    """Per-scenario fixture metrics."""

    scenario_id: str
    legal_set_size: int
    k: int
    seed: int
    shortlist_size: int
    full_top1_in_shortlist: bool
    full_top5_in_shortlist: bool
    fallback_reason: str | None
    recall_at_k: float
    wall_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShortlistMetrics":
        return cls(
            scenario_id=str(data["scenario_id"]),
            legal_set_size=int(data["legal_set_size"]),
            k=int(data["k"]),
            seed=int(data["seed"]),
            shortlist_size=int(data["shortlist_size"]),
            full_top1_in_shortlist=bool(data["full_top1_in_shortlist"]),
            full_top5_in_shortlist=bool(data["full_top5_in_shortlist"]),
            fallback_reason=data.get("fallback_reason"),
            recall_at_k=float(data["recall_at_k"]),
            wall_seconds=float(data["wall_seconds"]),
        )


@dataclass(frozen=True)
class ShortlistReport:
    """Full fixture report for SLM-176."""

    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    claim_class: str
    hypothesis: str
    falsifier: str
    cells: tuple[ShortlistScenario, ...]
    rows: list[ShortlistMetrics]
    mean_recall_at_k: float
    mean_full_top1_retained: float
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
            "mean_recall_at_k": self.mean_recall_at_k,
            "mean_full_top1_retained": self.mean_full_top1_retained,
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
    def from_dict(cls, data: dict[str, Any]) -> "ShortlistReport":
        return cls(
            matrix_set=data.get("matrix_set", MATRIX_SET),
            matrix_version=data.get("matrix_version", MATRIX_VERSION),
            experiment_id=data.get("experiment_id", EXPERIMENT_ID),
            run_id=data.get("run_id", "slm176_fixture"),
            status=data.get("status", "fixture"),
            claim_class=data.get("claim_class", "wiring"),
            hypothesis=data.get(
                "hypothesis",
                "A deterministic description-retrieval shortlist preserves the "
                "full-set top candidate for synthetic legal action sets.",
            ),
            falsifier=data.get(
                "falsifier",
                "The retrieval shortlist drops the full-set top-1 candidate or "
                "collapses to fallback for every non-trivial legal set.",
            ),
            cells=tuple(ShortlistScenario.from_dict(c) for c in data.get("cells", [])),
            rows=[ShortlistMetrics.from_dict(r) for r in data.get("rows", [])],
            mean_recall_at_k=float(data.get("mean_recall_at_k", 0.0)),
            mean_full_top1_retained=float(data.get("mean_full_top1_retained", 0.0)),
            disposition=data.get("disposition", "inconclusive"),
            disposition_rationale=data.get(
                "disposition_rationale", "no rationale provided"
            ),
            dependency_caveats=list(data.get("dependency_caveats", [])),
            version_stamp=data.get("version_stamp", {}),
        )


def _build_fixture_catalog() -> ActionDescriptionCatalog:
    """Small deterministic catalog for wiring tests."""
    entries = [
        ActionDescription(
            action_key="+Card",
            short_name="Card",
            signature="+Card()",
            description="Card container for grouping content.",
            result_type="element",
            argument_roles=(),
            sibling_family="container",
            provenance="schema",
        ),
        ActionDescription(
            action_key="+Stack",
            short_name="Stack",
            signature="+Stack()",
            description="Stack layout for vertical arrangement.",
            result_type="element",
            argument_roles=(),
            sibling_family="container",
            provenance="schema",
        ),
        ActionDescription(
            action_key="+Button",
            short_name="Button",
            signature="+Button()",
            description="Button action for user clicks.",
            result_type="element",
            argument_roles=(),
            sibling_family="action",
            provenance="schema",
        ),
        ActionDescription(
            action_key="+Input",
            short_name="Input",
            signature="+Input()",
            description="Input field for text entry.",
            result_type="element",
            argument_roles=(),
            sibling_family="input",
            provenance="schema",
        ),
        ActionDescription(
            action_key="+Select",
            short_name="Select",
            signature="+Select()",
            description="Select dropdown for choosing among options.",
            result_type="element",
            argument_roles=(),
            sibling_family="input",
            provenance="schema",
        ),
        ActionDescription(
            action_key="+Label",
            short_name="Label",
            signature="+Label()",
            description="Label for descriptive text.",
            result_type="element",
            argument_roles=(),
            sibling_family="content",
            provenance="schema",
        ),
        ActionDescription(
            action_key="+Image",
            short_name="Image",
            signature="+Image()",
            description="Image for visual content.",
            result_type="element",
            argument_roles=(),
            sibling_family="content",
            provenance="schema",
        ),
        ActionDescription(
            action_key="+Form",
            short_name="Form",
            signature="+Form()",
            description="Form for collecting input values.",
            result_type="element",
            argument_roles=(),
            sibling_family="form",
            provenance="schema",
        ),
        ActionDescription(
            action_key="+Table",
            short_name="Table",
            signature="+Table()",
            description="Table for tabular data display.",
            result_type="element",
            argument_roles=(),
            sibling_family="data",
            provenance="schema",
        ),
        ActionDescription(
            action_key="+BarChart",
            short_name="BarChart",
            signature="+BarChart()",
            description="Bar chart for categorical data visualization.",
            result_type="element",
            argument_roles=(),
            sibling_family="data",
            provenance="schema",
        ),
        ActionDescription(
            action_key="-",
            short_name="close",
            signature="-",
            description="Close a component or builtin expression.",
            result_type=None,
            argument_roles=(),
            sibling_family=None,
            provenance="structural",
        ),
        ActionDescription(
            action_key="r=",
            short_name="root_statement",
            signature="r=",
            description="Root statement marker.",
            result_type=None,
            argument_roles=(),
            sibling_family=None,
            provenance="structural",
        ),
        ActionDescription(
            action_key="a=",
            short_name="action_statement",
            signature="a=",
            description="Action statement marker.",
            result_type=None,
            argument_roles=(),
            sibling_family=None,
            provenance="structural",
        ),
        ActionDescription(
            action_key="*Run",
            short_name="Run",
            signature="*Run(args: any)",
            description="Builtin aggregate action Run.",
            result_type="any",
            argument_roles=("args:any",),
            sibling_family="builtin",
            provenance="builtin",
        ),
        ActionDescription(
            action_key="*Fetch",
            short_name="Fetch",
            signature="*Fetch(args: any)",
            description="Builtin aggregate action Fetch.",
            result_type="any",
            argument_roles=("args:any",),
            sibling_family="builtin",
            provenance="builtin",
        ),
    ]
    return ActionDescriptionCatalog(entries=tuple(entries))


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _legal_set_for_seed(
    catalog: ActionDescriptionCatalog,
    seed: int,
    size: int,
) -> tuple[str, ...]:
    """Deterministic synthetic legal subset of ``size`` action keys."""
    keys = list(catalog.keys())
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(len(keys), generator=rng)
    selected = [keys[int(i)] for i in perm[:size].tolist()]
    return tuple(sorted(selected))


def _query_hint_to_vector(
    hint: str,
    d_model: int,
    catalog: ActionDescriptionCatalog,
) -> torch.Tensor:
    encoder = FixtureDescriptionEncoder(d_model)
    return build_query_vector(hint, catalog, encoder)


def _run_scenario(
    scenario: ShortlistScenario,
    catalog: ActionDescriptionCatalog,
    d_model: int,
) -> ShortlistMetrics:
    start = time.perf_counter()
    legal_action_ids = _legal_set_for_seed(
        catalog, scenario.seed, scenario.legal_set_size
    )
    action_vectors = catalog.fixture_vectors(d_model, source="schema_description")
    query_vector = _query_hint_to_vector(scenario.query_hint, d_model, catalog)

    policy = ActionShortlistPolicy(
        mode="description_retrieval",
        k=scenario.k,
        min_legal_size=1,
        score_margin=0.0,
    )

    shortlist, scores, fallback_reason = retrieve_then_rerank(
        legal_action_ids,
        query_vector,
        action_vectors,
        policy,
    )

    # Full-set ranking by retrieval score.
    full_ranked = sorted(
        [aid for aid in legal_action_ids if aid in scores],
        key=lambda aid: scores.get(aid, -float("inf")),
        reverse=True,
    )
    top1 = full_ranked[0] if full_ranked else None
    top5 = set(full_ranked[:5])

    shortlist_set = set(shortlist)
    elapsed = time.perf_counter() - start
    return ShortlistMetrics(
        scenario_id=scenario.scenario_id,
        legal_set_size=len(legal_action_ids),
        k=scenario.k,
        seed=scenario.seed,
        shortlist_size=len(shortlist),
        full_top1_in_shortlist=top1 in shortlist_set if top1 is not None else False,
        full_top5_in_shortlist=bool(top5 & shortlist_set),
        fallback_reason=fallback_reason,
        recall_at_k=_clamp(
            len(shortlist_set & set(full_ranked[: scenario.k])) / max(1, scenario.k)
        ),
        wall_seconds=_clamp(elapsed + 0.001, low=0.001, high=10.0),
    )


def build_cells(
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    sizes: tuple[int, ...] = (8, 16, 32),
    ks: tuple[int, ...] = (4, 8),
) -> tuple[ShortlistScenario, ...]:
    """Build scenario × seed cells for the fixture."""
    hints = (
        "container layout",
        "user input form",
        "data visualization",
        "action button",
        "structural root",
    )
    cells: list[ShortlistScenario] = []
    for size in sizes:
        for k in ks:
            for seed in seeds:
                hint = hints[(size + k + seed) % len(hints)]
                scenario_id = f"size{size}_k{k}_s{seed}"
                cells.append(
                    ShortlistScenario(
                        scenario_id=scenario_id,
                        legal_set_size=size,
                        k=k,
                        seed=seed,
                        query_hint=hint,
                    )
                )
    return tuple(cells)


def validate_manifest(cells: tuple[ShortlistScenario, ...]) -> list[str]:
    """Validate the shortlist scenario manifest."""
    errors: list[str] = []
    if not cells:
        errors.append("cells must not be empty")
    seen: set[str] = set()
    for cell in cells:
        if cell.scenario_id in seen:
            errors.append(f"duplicate scenario_id: {cell.scenario_id}")
        seen.add(cell.scenario_id)
        if cell.legal_set_size <= 0:
            errors.append(f"{cell.scenario_id}: legal_set_size must be positive")
        if cell.k < 0:
            errors.append(f"{cell.scenario_id}: k must be non-negative")
    return errors


def _resolve_disposition(
    rows: list[ShortlistMetrics],
) -> tuple[str, str, float, float]:
    """Return (disposition, rationale, mean_recall, mean_top1_retained)."""
    if not rows:
        return (
            "no_data",
            "No scenarios were run.",
            0.0,
            0.0,
        )
    non_fallback = [r for r in rows if r.fallback_reason is None]
    if not non_fallback:
        return (
            "always_fallback",
            "Every scenario fell back to the full legal set; retrieval did not engage.",
            0.0,
            0.0,
        )
    mean_recall = sum(r.recall_at_k for r in non_fallback) / len(non_fallback)
    top1_retained = sum(1.0 for r in non_fallback if r.full_top1_in_shortlist) / len(
        non_fallback
    )

    if mean_recall < 0.5:
        return (
            "low_recall",
            "Retrieval shortlist recall@k is below 50%; the shortlist is losing "
            "too many top full-set candidates.",
            mean_recall,
            top1_retained,
        )
    if top1_retained < 0.5:
        return (
            "top1_not_retained",
            "The full-set top-1 candidate is retained in fewer than 50% of "
            "non-fallback scenarios.",
            mean_recall,
            top1_retained,
        )
    return (
        "shortlist_wiring_ok",
        "Deterministic description retrieval retains the full-set top candidate "
        "and achieves reasonable recall@k on synthetic legal sets.  Wiring is "
        "ready for a trained-model test.",
        mean_recall,
        top1_retained,
    )


def run_fixture_campaign(
    cells: tuple[ShortlistScenario, ...] | None = None,
    *,
    run_id: str = "slm176-action-shortlist-rerank",
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    d_model: int = _DEFAULT_D_MODEL,
) -> ShortlistReport:
    """Run the SLM-176 action-shortlist rerank fixture campaign."""
    cells = cells or build_cells(seeds)
    errors = validate_manifest(cells)
    if errors:
        raise ValueError("manifest validation failed: " + "; ".join(errors))

    catalog = _build_fixture_catalog()
    rows: list[ShortlistMetrics] = []
    for cell in cells:
        rows.append(_run_scenario(cell, catalog, d_model))

    disposition, rationale, mean_recall, top1_retained = _resolve_disposition(rows)

    report = ShortlistReport(
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
        status="fixture",
        claim_class="wiring",
        hypothesis=(
            "A deterministic description-retrieval shortlist preserves the "
            "full-set top candidate for synthetic legal action sets."
        ),
        falsifier=(
            "The retrieval shortlist drops the full-set top-1 candidate or "
            "collapses to fallback for every non-trivial legal set."
        ),
        cells=cells,
        rows=rows,
        mean_recall_at_k=mean_recall,
        mean_full_top1_retained=top1_retained,
        disposition=disposition,
        disposition_rationale=rationale,
        dependency_caveats=[
            "Depends on slm_training.dsl.action_shortlist and the deterministic "
            "FixtureDescriptionEncoder; real text encoders may produce different geometry.",
            "Legal sets are synthetic permutations of a small fixture catalog, not "
            "live compiler legal actions from a real decode step.",
            "No model is trained; this is wiring evidence only.",
        ],
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm176_action_shortlist_rerank",
            "dsl.action_shortlist",
            "dsl.action_descriptions",
        ),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm176_action_shortlist_rerank_report.json")
    return report


def render_markdown(report: ShortlistReport) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-176 (P14): action-shortlist retrieve-then-rerank fixture ({report.run_id})",
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
        "## Scenarios",
        "",
        "| scenario_id | legal_set_size | k | seed | query_hint |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cell in report.cells:
        lines.append(
            f"| {cell.scenario_id} | {cell.legal_set_size} | {cell.k} | "
            f"{cell.seed} | {cell.query_hint} |"
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| scenario_id | shortlist_size | top1_retained | top5_retained | "
            "fallback | recall@k | wall_seconds |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.rows:
        fallback = row.fallback_reason if row.fallback_reason is not None else "-"
        lines.append(
            f"| {row.scenario_id} | {row.shortlist_size} | "
            f"{row.full_top1_in_shortlist} | {row.full_top5_in_shortlist} | "
            f"{fallback} | {row.recall_at_k:.3f} | {row.wall_seconds:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- mean recall@k: **{report.mean_recall_at_k:.3f}**",
            f"- mean full-set top-1 retained: **{report.mean_full_top1_retained:.3f}**",
            "",
            "## Disposition",
            "",
            f"**{report.disposition}**",
            "",
            report.disposition_rationale,
            "",
            "## Go / no-go decision",
            "",
            "**No-go for promotion.** This is a wiring fixture. The retrieval, "
            "shortlist, and rerank plumbing are exercised over a deterministic "
            "synthetic encoder and catalog, but no real model was trained or "
            "evaluated. The mechanism remains ``retain_diagnostic`` / "
            "``blocked_pending_real_model`` until a trained model and AgentV "
            "evaluation are available.",
            "",
            "## Honest caveats",
            "",
            "- The FixtureDescriptionEncoder is a deterministic hash surrogate, not a "
            "  trained language model; geometry may differ with real text encoders.",
            "- Synthetic legal sets are random permutations, not live compiler output.",
            "- No ship-gate claim is made; this is wiring evidence only.",
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_slm176_action_shortlist_rerank_fixture --mode plan-only",
            "python -m scripts.run_slm176_action_shortlist_rerank_fixture --mode fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
