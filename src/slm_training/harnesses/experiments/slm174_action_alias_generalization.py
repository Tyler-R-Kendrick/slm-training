"""SLM-174 (SDE2-07): test description-mediated generalization with anonymized aliases.

Wiring/fixture-only harness that encodes action descriptions under several
name/alias regimes and checks whether aliased descriptions remain semantically
clusterable.  No model is trained and no GPU is required.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from slm_training.dsl.action_descriptions import (
    ActionDescriptionCatalog,
    FixtureDescriptionEncoder,
    build_alias_map,
    compute_nearest_neighbor_metrics,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "EXPERIMENT_ID",
    "ARM_NAMES",
    "AliasArm",
    "AliasMetrics",
    "AliasReport",
    "build_cells",
    "validate_manifest",
    "run_fixture_campaign",
    "render_markdown",
    "resolve_disposition",
]

MATRIX_VERSION = "sde2-07-v1"
MATRIX_SET = "slm174_action_alias_generalization"
EXPERIMENT_ID = "slm174-action-alias-generalization"

_DEFAULT_SEEDS = (0, 1, 2)

ARM_NAMES = (
    "canonical_name_plus_description",
    "canonical_name_description_without_name",
    "fixed_alias_description_without_name",
    "multiple_alias_augmentation_held_out",
    "multiple_alias_shuffled_descriptions",
    "alias_signature_only",
    "canonical_evaluated_under_unseen_alias",
)

_ARM_SOURCE = {
    "canonical_name_plus_description": "canonical_name_plus_description",
    "canonical_name_description_without_name": "description_without_canonical_name",
    "fixed_alias_description_without_name": "alias_aware_description",
    "multiple_alias_augmentation_held_out": "alias_aware_description",
    "multiple_alias_shuffled_descriptions": "alias_aware_shuffled",
    "alias_signature_only": "alias_aware_signature_only",
}

_DEFAULT_D_MODEL = 64


@dataclass(frozen=True)
class AliasArm:
    """One SLM-174 fixture arm."""

    arm_id: str
    arm_name: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AliasArm":
        return cls(
            arm_id=str(data["arm_id"]),
            arm_name=str(data["arm_name"]),
            seed=int(data["seed"]),
        )


@dataclass(frozen=True)
class AliasMetrics:
    """Per-arm, per-seed alias-generalization fixture metrics."""

    arm_id: str
    arm_name: str
    seed: int
    n_actions: int
    mean_nearest_cosine: float
    family_purity: float
    held_out_transfer_score: float | None
    canonical_unseen_alias_score: float | None
    leakage_findings: tuple[str, ...]
    wall_seconds: float

    def to_dict(self) -> dict[str, Any]:
        out = dict(asdict(self))
        out["leakage_findings"] = list(self.leakage_findings)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AliasMetrics":
        return cls(
            arm_id=str(data["arm_id"]),
            arm_name=str(data["arm_name"]),
            seed=int(data["seed"]),
            n_actions=int(data["n_actions"]),
            mean_nearest_cosine=float(data["mean_nearest_cosine"]),
            family_purity=float(data["family_purity"]),
            held_out_transfer_score=(
                float(data["held_out_transfer_score"])
                if data.get("held_out_transfer_score") is not None
                else None
            ),
            canonical_unseen_alias_score=(
                float(data["canonical_unseen_alias_score"])
                if data.get("canonical_unseen_alias_score") is not None
                else None
            ),
            leakage_findings=tuple(data.get("leakage_findings", [])),
            wall_seconds=float(data["wall_seconds"]),
        )


@dataclass(frozen=True)
class AliasReport:
    """Full fixture report for SLM-174."""

    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    claim_class: str
    hypothesis: str
    falsifier: str
    cells: tuple[AliasArm, ...]
    rows: list[AliasMetrics]
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
    def from_dict(cls, data: dict[str, Any]) -> "AliasReport":
        return cls(
            matrix_set=data.get("matrix_set", MATRIX_SET),
            matrix_version=data.get("matrix_version", MATRIX_VERSION),
            experiment_id=data.get("experiment_id", EXPERIMENT_ID),
            run_id=data.get("run_id", "slm174_fixture"),
            status=data.get("status", "fixture"),
            claim_class=data.get("claim_class", "wiring"),
            hypothesis=data.get(
                "hypothesis",
                "Action descriptions remain semantically clusterable even when canonical "
                "action names are replaced by opaque aliases.",
            ),
            falsifier=data.get(
                "falsifier",
                "Aliased descriptions collapse into a single cluster or nearest-neighbor "
                "geometry no longer reflects sibling families.",
            ),
            cells=tuple(AliasArm.from_dict(c) for c in data.get("cells", [])),
            rows=[AliasMetrics.from_dict(r) for r in data.get("rows", [])],
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


def _encode_descriptions(
    descriptions: dict[str, str], d_model: int
) -> dict[str, torch.Tensor]:
    encoder = FixtureDescriptionEncoder(d_model)
    return {key: encoder.encode(text) for key, text in descriptions.items()}


def _family_purity(vectors: dict[str, torch.Tensor], catalog: ActionDescriptionCatalog) -> float:
    """Fraction of actions whose nearest neighbor shares a sibling family."""
    if not vectors:
        return 0.0
    keys = sorted(vectors)
    matrix = torch.stack([vectors[k] for k in keys])
    sims = F.cosine_similarity(matrix.unsqueeze(1), matrix.unsqueeze(0), dim=-1)
    sims.fill_diagonal_(-1.0)
    best_indices = sims.argmax(dim=-1)
    hits = 0
    for i, key in enumerate(keys):
        entry = catalog.by_key.get(key)
        neighbor_entry = catalog.by_key.get(keys[int(best_indices[i].item())])
        if entry is not None and neighbor_entry is not None:
            if (
                entry.sibling_family is not None
                and entry.sibling_family == neighbor_entry.sibling_family
            ):
                hits += 1
    return hits / len(keys)


def _held_out_transfer_score(
    catalog: ActionDescriptionCatalog,
    d_model: int,
    train_seeds: tuple[int, ...] = (0, 1),
    held_out_seed: int = 2,
) -> float:
    """Mean same-action cosine minus mean different-action cosine across alias maps."""
    keys = list(catalog.keys())
    train_vectors: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
    for seed in train_seeds:
        alias_map = build_alias_map(seed, "held_out_train", keys)
        descriptions = catalog.descriptions_for(
            "alias_aware_description", alias_map=alias_map
        )
        vectors = _encode_descriptions(descriptions, d_model)
        for key in keys:
            train_vectors[key].append(vectors[key])
    train_mean = {
        key: torch.stack(vecs).mean(dim=0) for key, vecs in train_vectors.items()
    }

    held_out_map = build_alias_map(held_out_seed, "held_out_test", keys)
    held_out_desc = catalog.descriptions_for(
        "alias_aware_description", alias_map=held_out_map
    )
    held_out_vectors = _encode_descriptions(held_out_desc, d_model)

    same_scores: list[float] = []
    diff_scores: list[float] = []
    for key in keys:
        same_scores.append(
            F.cosine_similarity(
                train_mean[key].unsqueeze(0),
                held_out_vectors[key].unsqueeze(0),
                dim=-1,
            )
            .item()
        )
        for other in keys:
            if other == key:
                continue
            diff_scores.append(
                F.cosine_similarity(
                    train_mean[key].unsqueeze(0),
                    held_out_vectors[other].unsqueeze(0),
                    dim=-1,
                )
                .item()
            )
    return _clamp(float(torch.tensor(same_scores).mean().item())) - _clamp(
        float(torch.tensor(diff_scores).mean().item())
    )


def _canonical_unseen_alias_score(
    catalog: ActionDescriptionCatalog,
    d_model: int,
    unseen_seed: int = 99,
) -> float:
    """Same-action cosine between canonical and unseen-alias vectors minus different-action."""
    keys = list(catalog.keys())
    canonical = _encode_descriptions(
        catalog.descriptions_for("schema_description"), d_model
    )
    alias_map = build_alias_map(unseen_seed, "canonical_unseen", keys)
    aliased = _encode_descriptions(
        catalog.descriptions_for("alias_aware_description", alias_map=alias_map),
        d_model,
    )
    same_scores = [
        F.cosine_similarity(
            canonical[key].unsqueeze(0), aliased[key].unsqueeze(0), dim=-1
        ).item()
        for key in keys
    ]
    diff_scores: list[float] = []
    for key in keys:
        for other in keys:
            if other == key:
                continue
            diff_scores.append(
                F.cosine_similarity(
                    canonical[key].unsqueeze(0),
                    aliased[other].unsqueeze(0),
                    dim=-1,
                )
                .item()
            )
    return _clamp(float(torch.tensor(same_scores).mean().item())) - _clamp(
        float(torch.tensor(diff_scores).mean().item())
    )


def _run_arm(
    arm: AliasArm,
    catalog: ActionDescriptionCatalog,
    d_model: int,
) -> AliasMetrics:
    start = time.perf_counter()
    keys = list(catalog.keys())
    arm_name = arm.arm_name

    if arm_name == "canonical_evaluated_under_unseen_alias":
        descriptions = catalog.descriptions_for("schema_description")
        vectors = _encode_descriptions(descriptions, d_model)
        nn = compute_nearest_neighbor_metrics(vectors)
        purity = _family_purity(vectors, catalog)
        canonical_unseen = _canonical_unseen_alias_score(catalog, d_model)
        leakage: tuple[str, ...] = ()
    elif arm_name == "multiple_alias_augmentation_held_out":
        descriptions = catalog.descriptions_for("schema_description")
        vectors = _encode_descriptions(descriptions, d_model)
        nn = compute_nearest_neighbor_metrics(vectors)
        purity = _family_purity(vectors, catalog)
        canonical_unseen = None
        leakage = ()
    else:
        source = _ARM_SOURCE[arm_name]
        alias_map = None
        if source.startswith("alias_aware"):
            alias_map = build_alias_map(arm.seed, f"slm174:{arm_name}", keys)
        descriptions = catalog.descriptions_for(source, alias_map=alias_map)
        vectors = _encode_descriptions(descriptions, d_model)
        nn = compute_nearest_neighbor_metrics(vectors)
        purity = _family_purity(vectors, catalog)
        canonical_unseen = None
        if alias_map is not None and arm_name != "multiple_alias_shuffled_descriptions":
            findings = alias_map.validate_no_leakage(
                descriptions, entries=catalog.by_key
            )
            leakage = tuple(findings)
        else:
            leakage = ()

    held_out = None
    if arm_name == "multiple_alias_augmentation_held_out":
        held_out = _held_out_transfer_score(catalog, d_model)

    elapsed = time.perf_counter() - start
    return AliasMetrics(
        arm_id=arm.arm_id,
        arm_name=arm.arm_name,
        seed=arm.seed,
        n_actions=len(vectors),
        mean_nearest_cosine=float(nn["mean_nearest_cosine"]),
        family_purity=purity,
        held_out_transfer_score=_clamp(held_out) if held_out is not None else None,
        canonical_unseen_alias_score=_clamp(canonical_unseen)
        if canonical_unseen is not None
        else None,
        leakage_findings=leakage,
        wall_seconds=_clamp(elapsed + 0.001, low=0.001, high=10.0),
    )


def build_cells(seeds: tuple[int, ...] = _DEFAULT_SEEDS) -> tuple[AliasArm, ...]:
    """Build the arm × seeds cells for the fixture."""
    cells: list[AliasArm] = []
    for seed in seeds:
        for arm_name in ARM_NAMES:
            cells.append(
                AliasArm(
                    arm_id=_arm_label(arm_name, seed),
                    arm_name=arm_name,
                    seed=seed,
                )
            )
    return tuple(cells)


def validate_manifest(cells: tuple[AliasArm, ...]) -> list[str]:
    """Validate the alias-generalization manifest."""
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


def _arm_means(rows: list[AliasMetrics]) -> dict[str, dict[str, float]]:
    """Aggregate per-arm means across seeds."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        bucket = grouped.setdefault(row.arm_name, {})
        for key in (
            "n_actions",
            "mean_nearest_cosine",
            "family_purity",
            "held_out_transfer_score",
            "canonical_unseen_alias_score",
            "wall_seconds",
        ):
            value = getattr(row, key)
            if value is not None:
                bucket.setdefault(key, []).append(float(value))
    return {
        arm: {key: sum(values) / len(values) for key, values in metrics.items()}
        for arm, metrics in grouped.items()
    }


def resolve_disposition(
    arm_means: dict[str, dict[str, float]]
) -> tuple[str, str]:
    """Return (disposition, rationale) from per-arm means."""
    canonical_plus = arm_means.get("canonical_name_plus_description", {}).get(
        "family_purity", 0.0
    )
    canonical_no_name = arm_means.get(
        "canonical_name_description_without_name", {}
    ).get("family_purity", 0.0)
    fixed_alias = arm_means.get("fixed_alias_description_without_name", {}).get(
        "family_purity", 0.0
    )
    shuffled = arm_means.get("multiple_alias_shuffled_descriptions", {}).get(
        "family_purity", 0.0
    )
    signature_only = arm_means.get("alias_signature_only", {}).get(
        "family_purity", 0.0
    )
    held_out = arm_means.get("multiple_alias_augmentation_held_out", {}).get(
        "held_out_transfer_score"
    )
    canonical_unseen = arm_means.get(
        "canonical_evaluated_under_unseen_alias", {}
    ).get("canonical_unseen_alias_score")

    if canonical_plus < 0.5:
        return (
            "baseline_unreliable",
            "Canonical-name-plus-description baseline does not cluster by family; "
            "the fixture encoder or catalog is not representative.",
        )

    if fixed_alias < canonical_no_name * 0.75:
        return (
            "alias_breaks_clustering",
            "Fixed-alias descriptions cluster substantially worse than canonical "
            "descriptions without names; aliases are not carrying semantic signal.",
        )

    if shuffled > fixed_alias * 0.9:
        return (
            "shuffled_control_fails",
            "Shuffled alias-aware descriptions cluster almost as well as ordered ones; "
            "nearest-neighbor geometry is not driven by semantic content.",
        )

    if signature_only > fixed_alias * 0.9:
        return (
            "signature_only_equivalent",
            "Alias+signature-only performs nearly as well as full descriptions; "
            "the description text is not adding measurable separability.",
        )

    if held_out is not None and held_out <= 0.0:
        return (
            "held_out_transfer_fails",
            "Training on multiple alias maps does not transfer to a held-out alias map.",
        )

    if canonical_unseen is not None and canonical_unseen <= 0.0:
        return (
            "canonical_unseen_alias_fails",
            "Canonical embeddings do not align with unseen-alias embeddings.",
        )

    return (
        "alias_generalization_wired",
        "Canonical and aliased descriptions cluster by sibling family, shuffled "
        "descriptions perform worse than ordered aliases, and held-out / unseen-alias "
        "transfer is positive.  The wiring is ready for a trained-model test.",
    )


def run_fixture_campaign(
    cells: tuple[AliasArm, ...] | None = None,
    *,
    run_id: str = "slm174-action-alias-generalization",
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    d_model: int = _DEFAULT_D_MODEL,
) -> AliasReport:
    """Run the SLM-174 action-alias generalization fixture campaign."""
    cells = cells or build_cells(seeds)
    errors = validate_manifest(cells)
    if errors:
        raise ValueError("manifest validation failed: " + "; ".join(errors))

    catalog = ActionDescriptionCatalog.build()
    rows: list[AliasMetrics] = []
    for cell in cells:
        rows.append(_run_arm(cell, catalog, d_model))

    means = _arm_means(rows)
    disposition, rationale = resolve_disposition(means)

    report = AliasReport(
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
        status="fixture",
        claim_class="wiring",
        hypothesis=(
            "Action descriptions remain semantically clusterable even when canonical "
            "action names are replaced by opaque aliases."
        ),
        falsifier=(
            "Aliased descriptions collapse into a single cluster or nearest-neighbor "
            "geometry no longer reflects sibling families."
        ),
        cells=cells,
        rows=rows,
        arm_means=means,
        disposition=disposition,
        disposition_rationale=rationale,
        dependency_caveats=[
            "Depends on slm_training.dsl.action_descriptions and the deterministic "
            "FixtureDescriptionEncoder; real HF text encoders may produce different geometry.",
            "Sibling-family grouping is a coarse semantic proxy; real action similarity "
            "is richer than family identity.",
            "No model is trained; this is wiring evidence only.",
        ],
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm174_action_alias_generalization",
            "dsl.action_descriptions",
        ),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm174_action_alias_generalization_report.json")
    return report


def render_markdown(report: AliasReport) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-174 (SDE2-07): action-alias generalization fixture ({report.run_id})",
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
        "## Alias-generalization arms",
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
            "| arm_id | arm_name | seed | n_actions | mean_nearest_cosine | family_purity | "
            "held_out_transfer | canonical_unseen | leakage | wall_seconds |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.rows:
        held_out = (
            f"{row.held_out_transfer_score:.3f}"
            if row.held_out_transfer_score is not None
            else "-"
        )
        canonical_unseen = (
            f"{row.canonical_unseen_alias_score:.3f}"
            if row.canonical_unseen_alias_score is not None
            else "-"
        )
        leakage = "yes" if row.leakage_findings else "no"
        lines.append(
            f"| {row.arm_id} | {row.arm_name} | {row.seed} | {row.n_actions} | "
            f"{row.mean_nearest_cosine:.3f} | {row.family_purity:.3f} | "
            f"{held_out} | {canonical_unseen} | {leakage} | {row.wall_seconds:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Per-arm means",
            "",
            "| arm_name | mean_nearest_cosine | family_purity | held_out_transfer | canonical_unseen |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for arm_name in ARM_NAMES:
        if arm_name not in report.arm_means:
            continue
        m = report.arm_means[arm_name]
        held_out = (
            f"{m.get('held_out_transfer_score', 0.0):.3f}"
            if "held_out_transfer_score" in m
            else "-"
        )
        canonical_unseen = (
            f"{m.get('canonical_unseen_alias_score', 0.0):.3f}"
            if "canonical_unseen_alias_score" in m
            else "-"
        )
        lines.append(
            f"| {arm_name} | {m.get('mean_nearest_cosine', 0.0):.3f} | "
            f"{m.get('family_purity', 0.0):.3f} | {held_out} | {canonical_unseen} |"
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
            "**No-go for promotion.** This is a wiring fixture. The alias map, "
            "description sources, and clustering telemetry are exercised over a deterministic "
            "synthetic encoder, but no real model was trained or evaluated. The mechanism "
            "remains ``retain_diagnostic`` / ``blocked_pending_real_model`` until a trained "
            "model and AgentV evaluation are available.",
            "",
            "## Honest caveats",
            "",
            "- The FixtureDescriptionEncoder is a deterministic hash surrogate, not a trained "
            "  language model; geometry may differ with real text encoders.",
            "- Sibling-family grouping is a coarse semantic proxy.",
            "- No ship-gate claim is made; this is wiring evidence only.",
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_slm174_action_alias_generalization_fixture --mode plan-only",
            "python -m scripts.run_slm174_action_alias_generalization_fixture --mode fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
