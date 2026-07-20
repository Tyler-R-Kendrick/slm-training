"""SLM-163 (SDE1-01): schema-description action-embedding wiring fixture.

Deterministic, CPU-only harness that compares action-embedding initializations
built from the OpenUI schema description catalog.  It trains no GPU model and
makes no ship-gate claim.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slm_training.dsl.action_descriptions import (
    ActionDescriptionCatalog,
    FixtureDescriptionEncoder,
    centroid_distance,
    compute_nearest_neighbor_metrics,
    coverage_report,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "EXPERIMENT_ID",
    "InitArm",
    "EmbeddingMetrics",
    "SchemaActionEmbeddingReport",
    "build_manifest",
    "validate_manifest",
    "run_fixture_campaign",
    "render_markdown",
]

MATRIX_VERSION = "sde1-01-v1"
MATRIX_SET = "slm163_schema_action_embedding"
EXPERIMENT_ID = "slm163-schema-action-embedding"

_DEFAULT_SEEDS = (0, 1, 2)
_DEFAULT_D_MODEL = 64

_RARE_COMPONENTS = {"+Form", "+Tabs", "+SwitchItem", "+Slider"}
_COMMON_COMPONENTS = {"+Card", "+Stack", "+Button", "+TextContent", "+Input"}
_SIBLING_PAIRS = [
    ({"+Stack"}, {"+Card"}),
    ({"+Tabs"}, {"+Accordion"}),
]


@dataclass(frozen=True)
class InitArm:
    """One initialization source arm in the SLM-163 campaign."""

    arm_id: str
    source: str
    name: str
    description: str
    promotable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InitArm":
        return cls(
            arm_id=str(data["arm_id"]),
            source=str(data["source"]),
            name=str(data["name"]),
            description=str(data["description"]),
            promotable=bool(data.get("promotable", False)),
        )


@dataclass(frozen=True)
class EmbeddingMetrics:
    """Per-arm, per-seed embedding-quality metrics."""

    arm_id: str
    source: str
    seed: int
    d_model: int
    n_actions: int
    coverage_fraction: float
    mean_nearest_cosine: float
    sibling_separation: float
    rare_common_centroid_distance: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingMetrics":
        return cls(
            arm_id=str(data["arm_id"]),
            source=str(data["source"]),
            seed=int(data["seed"]),
            d_model=int(data["d_model"]),
            n_actions=int(data["n_actions"]),
            coverage_fraction=float(data["coverage_fraction"]),
            mean_nearest_cosine=float(data["mean_nearest_cosine"]),
            sibling_separation=float(data["sibling_separation"]),
            rare_common_centroid_distance=float(data["rare_common_centroid_distance"]),
            notes=list(data.get("notes", [])),
        )


@dataclass(frozen=True)
class SchemaActionEmbeddingReport:
    """Full fixture report for SLM-163."""

    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    claim_class: str
    d_model: int
    arms: tuple[InitArm, ...]
    rows: list[EmbeddingMetrics]
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "d_model": self.d_model,
            "arms": [arm.to_dict() for arm in self.arms],
            "rows": [row.to_dict() for row in self.rows],
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchemaActionEmbeddingReport":
        return cls(
            matrix_set=data.get("matrix_set", MATRIX_SET),
            matrix_version=data.get("matrix_version", MATRIX_VERSION),
            experiment_id=data.get("experiment_id", EXPERIMENT_ID),
            run_id=data.get("run_id", "slm163_fixture"),
            status=data.get("status", "fixture"),
            claim_class=data.get("claim_class", "wiring"),
            d_model=int(data.get("d_model", _DEFAULT_D_MODEL)),
            arms=tuple(InitArm.from_dict(a) for a in data.get("arms", [])),
            rows=[EmbeddingMetrics.from_dict(r) for r in data.get("rows", [])],
            version_stamp=data.get("version_stamp", {}),
        )


def build_manifest() -> tuple[InitArm, ...]:
    """Return the default SLM-163 fixture manifest arms."""
    return (
        InitArm(
            arm_id="A_none",
            source="none",
            name="none",
            description="No action descriptions; embeddings remain randomly initialized.",
        ),
        InitArm(
            arm_id="B_current_stub",
            source="current_stub",
            name="current_stub",
            description="Short stub glosses matching the existing teacher_init_embeddings path.",
        ),
        InitArm(
            arm_id="C_schema_description",
            source="schema_description",
            name="schema_description",
            description="Schema-derived descriptions with property signatures and roles.",
        ),
        InitArm(
            arm_id="D_expanded_description",
            source="expanded_description",
            name="expanded_description",
            description="Rich teacher-style descriptions from the committed expanded JSON file.",
        ),
        InitArm(
            arm_id="E_shuffled",
            source="shuffled",
            name="shuffled",
            description="Control arm: schema descriptions randomly reassigned to action keys.",
        ),
    )


def validate_manifest(arms: tuple[InitArm, ...]) -> list[str]:
    """Validate manifest shape."""
    errors: list[str] = []
    if not arms:
        errors.append("arms must not be empty")
    seen: set[str] = set()
    for arm in arms:
        if arm.arm_id in seen:
            errors.append(f"duplicate arm_id: {arm.arm_id}")
        seen.add(arm.arm_id)
        if arm.promotable:
            errors.append(f"{arm.arm_id}: fixture arms must be non-promotable")
    return errors


def _pairwise_separation(vectors: dict[str, Any], set_a: set[str], set_b: set[str]) -> float:
    """Average cosine distance between all pairs across ``set_a`` and ``set_b``."""
    import torch

    present_a = [vectors[k] for k in set_a if k in vectors]
    present_b = [vectors[k] for k in set_b if k in vectors]
    if not present_a or not present_b:
        return 0.0
    mat_a = torch.stack(present_a)
    mat_b = torch.stack(present_b)
    # cosine_similarity expects (*, D); use 2D expansion.
    sims = torch.nn.functional.cosine_similarity(
        mat_a.unsqueeze(1), mat_b.unsqueeze(0), dim=-1
    )
    return float((1.0 - sims).mean().item())


def _compute_arm_metrics(
    arm: InitArm,
    seed: int,
    d_model: int,
    catalog: ActionDescriptionCatalog,
) -> EmbeddingMetrics:
    import torch

    descriptions = catalog.descriptions_for(arm.source)
    encoder = FixtureDescriptionEncoder(d_model)
    rng = torch.Generator()
    rng.manual_seed(seed)

    vectors: dict[str, Any] = {}
    for key, text in descriptions.items():
        # Blend a deterministic encoding with a tiny seed-dependent noise so that
        # repeated seeds are identical but different seeds are not degenerate.
        base = encoder.encode(text)
        noise = torch.randn(d_model, generator=rng) * 0.02
        vectors[key] = base + noise

    cov = coverage_report(descriptions, catalog)
    nn_metrics = compute_nearest_neighbor_metrics(vectors)

    sibling_sep = 0.0
    if len(_SIBLING_PAIRS) > 0:
        sep_values = [
            _pairwise_separation(vectors, a, b) for a, b in _SIBLING_PAIRS
        ]
        sibling_sep = sum(sep_values) / len(sep_values)

    rare_common = centroid_distance(vectors, _RARE_COMPONENTS, _COMMON_COMPONENTS)

    return EmbeddingMetrics(
        arm_id=arm.arm_id,
        source=arm.source,
        seed=seed,
        d_model=d_model,
        n_actions=len(vectors),
        coverage_fraction=cov["coverage_fraction"],
        mean_nearest_cosine=nn_metrics["mean_nearest_cosine"],
        sibling_separation=sibling_sep,
        rare_common_centroid_distance=rare_common,
        notes=[
            f"source={arm.source}",
            "fixture-only: hash-based encoder, no learned model",
        ],
    )


def run_fixture_campaign(
    arms: tuple[InitArm, ...] | None = None,
    *,
    run_id: str = "slm163-schema-action-embedding",
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    d_model: int = _DEFAULT_D_MODEL,
) -> SchemaActionEmbeddingReport:
    """Run the SLM-163 schema-description action-embedding fixture campaign."""
    arms = arms or build_manifest()
    errors = validate_manifest(arms)
    if errors:
        raise ValueError("manifest validation failed: " + "; ".join(errors))

    catalog = ActionDescriptionCatalog.build()
    rows: list[EmbeddingMetrics] = []
    for arm in arms:
        if arm.source == "none":
            # No vectors; record zero metrics as the baseline.
            for seed in seeds:
                rows.append(
                    EmbeddingMetrics(
                        arm_id=arm.arm_id,
                        source=arm.source,
                        seed=seed,
                        d_model=d_model,
                        n_actions=0,
                        coverage_fraction=0.0,
                        mean_nearest_cosine=0.0,
                        sibling_separation=0.0,
                        rare_common_centroid_distance=0.0,
                        notes=["baseline: no action descriptions"],
                    )
                )
            continue
        for seed in seeds:
            rows.append(_compute_arm_metrics(arm, seed, d_model, catalog))

    report = SchemaActionEmbeddingReport(
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
        status="fixture",
        claim_class="wiring",
        d_model=d_model,
        arms=arms,
        rows=rows,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm163_schema_action_embedding",
        ),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm163_schema_action_embedding_report.json")
    return report


def render_markdown(report: SchemaActionEmbeddingReport) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-163 (SDE1-01): Schema-description action-embedding fixture ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`",
        "",
        f"Version: `{report.matrix_version}`",
        "",
        f"Status: **{report.status}**",
        "",
        "**Claim class:** wiring / fixture only. No GPU was used, no production "
        "TwoTower wiring was touched, and no ship-gate claim is made.",
        "",
        "## Hypothesis",
        "",
        "Schema-derived action descriptions produce action embeddings that are more "
        "structured than random or stub initializations, as measured by coverage, "
        "nearest-neighbor cosine separation, sibling-family separation, and "
        "rare-vs-common centroid distance.",
        "",
        "## Falsifier",
        "",
        "Schema descriptions do not improve any of the above metrics over the "
        "current_stub baseline, or the shuffled control arm performs as well as "
        "the schema-driven arms.",
        "",
        "## Arms",
        "",
        "| Arm | Source | Promotable | Description |",
        "| --- | --- | --- | --- |",
    ]
    for arm in report.arms:
        lines.append(
            f"| {arm.arm_id} | {arm.source} | {arm.promotable} | {arm.description} |"
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Arm | Seed | d_model | Actions | Coverage | Mean NN cos | Sibling sep | Rare-common dist |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.rows:
        lines.append(
            f"| {row.arm_id} | {row.seed} | {row.d_model} | {row.n_actions} | "
            f"{row.coverage_fraction:.3f} | {row.mean_nearest_cosine:.3f} | "
            f"{row.sibling_separation:.3f} | {row.rare_common_centroid_distance:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Go / no-go decision",
            "",
            "**No-go for promotion.** Every arm is explicitly non-promotable. The "
            "harness proves the wiring and metrics plumbing over deterministic "
            "schema-derived descriptions, but it does not train or evaluate a real "
            "model. The mechanism remains ``retain_diagnostic`` / "
            "``blocked_pending_real_model`` until a trained scorer and AgentV "
            "evaluation are available.",
            "",
            "## Honest caveats",
            "",
            "- The encoder is a deterministic hash projection, not a trained language model.",
            "- Embeddings are perturbed by a tiny seed-dependent noise vector so that "
            "  different seeds are not degenerate.",
            "- Sibling and rare/common metrics use hand-picked component families as a "
            "  sanity check, not a learned taxonomy.",
            "- No Pareto or ship-gate claim is made; this is wiring evidence only.",
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_slm163_schema_action_embedding_fixture --mode plan-only",
            "python -m scripts.run_slm163_schema_action_embedding_fixture --mode fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
