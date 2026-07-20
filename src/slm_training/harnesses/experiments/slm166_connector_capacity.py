"""SLM-166 (SDE1-04): semantic connector capacity wiring/fixture harness.

This is a fixture-only comparison of connector variants between a frozen context
encoder and the sparse grammar-action scorer.  No connector is wired into live
decode; metrics are deterministic and CPU-safe.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from slm_training.dsl.action_descriptions import (
    ActionDescriptionCatalog,
    FixtureDescriptionEncoder,
)
from slm_training.models.semantic_connector import (
    SemanticConnector,
    count_connector_parameters,
    estimate_connector_flops,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "EXPERIMENT_ID",
    "ARM_NAMES",
    "TRAIN_SCOPES",
    "CONNECTOR_TYPES",
    "ConnectorArm",
    "ConnectorMetrics",
    "ConnectorReport",
    "build_cells",
    "validate_manifest",
    "run_fixture_campaign",
    "render_markdown",
    "resolve_disposition",
]

MATRIX_VERSION = "sde1-04-v1"
MATRIX_SET = "slm166_connector_capacity"
EXPERIMENT_ID = "slm166-connector-capacity"

_DEFAULT_SEEDS = (0, 1, 2)
_DEFAULT_BATCH = 2
_DEFAULT_SEQ = 8
_DEFAULT_D_MODEL = 64
_DEFAULT_TARGET_DECISIONS = 1000

TRAIN_SCOPES = (
    "current",
    "connector_only",
    "connector_plus_action_residuals",
    "small_model",
)
CONNECTOR_TYPES = ("none", "linear", "low_rank", "cross_attention")
ARM_NAMES = (
    "current",
    "linear",
    "low_rank",
    "cross_attention",
    "linear_plus_action_residuals",
    "linear_shuffled_context",
    "local_target",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConnectorArm:
    """One connector-capacity arm plus derived recipe fields."""

    arm_id: str
    arm_name: str
    connector_type: str
    train_scope: str
    seed: int
    d_model: int
    connector_hidden_dim: int
    connector_rank: int
    connector_n_queries: int
    connector_freeze_encoder: bool
    target_decisions: int

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConnectorArm":
        return cls(
            arm_id=str(data["arm_id"]),
            arm_name=str(data["arm_name"]),
            connector_type=str(data["connector_type"]),
            train_scope=str(data["train_scope"]),
            seed=int(data["seed"]),
            d_model=int(data["d_model"]),
            connector_hidden_dim=int(data["connector_hidden_dim"]),
            connector_rank=int(data["connector_rank"]),
            connector_n_queries=int(data["connector_n_queries"]),
            connector_freeze_encoder=bool(data["connector_freeze_encoder"]),
            target_decisions=int(data["target_decisions"]),
        )


@dataclass(frozen=True)
class ConnectorMetrics:
    """Per-arm, per-seed synthetic fixture metrics."""

    arm_id: str
    arm_name: str
    connector_type: str
    train_scope: str
    seed: int
    trainable_params: int
    frozen_params: int
    estimated_flops: int
    rare_component_recall: float
    meaningful_program_rate: float
    common_component_recall: float
    parse_validity_rate: float
    first_attempt_quality: float
    target_decisions: int
    wall_seconds: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConnectorMetrics":
        return cls(
            arm_id=str(data["arm_id"]),
            arm_name=str(data["arm_name"]),
            connector_type=str(data["connector_type"]),
            train_scope=str(data["train_scope"]),
            seed=int(data["seed"]),
            trainable_params=int(data["trainable_params"]),
            frozen_params=int(data["frozen_params"]),
            estimated_flops=int(data["estimated_flops"]),
            rare_component_recall=float(data["rare_component_recall"]),
            meaningful_program_rate=float(data["meaningful_program_rate"]),
            common_component_recall=float(data["common_component_recall"]),
            parse_validity_rate=float(data["parse_validity_rate"]),
            first_attempt_quality=float(data["first_attempt_quality"]),
            target_decisions=int(data["target_decisions"]),
            wall_seconds=float(data["wall_seconds"]),
            notes=list(data.get("notes", [])),
        )


@dataclass(frozen=True)
class ConnectorReport:
    """Full fixture report for SLM-166."""

    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    claim_class: str
    hypothesis: str
    falsifier: str
    cells: tuple[ConnectorArm, ...]
    rows: list[ConnectorMetrics]
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
    def from_dict(cls, data: dict[str, Any]) -> "ConnectorReport":
        return cls(
            matrix_set=data.get("matrix_set", MATRIX_SET),
            matrix_version=data.get("matrix_version", MATRIX_VERSION),
            experiment_id=data.get("experiment_id", EXPERIMENT_ID),
            run_id=data.get("run_id", "slm166_fixture"),
            status=data.get("status", "fixture"),
            claim_class=data.get("claim_class", "wiring"),
            hypothesis=data.get(
                "hypothesis",
                "A nonlinear cross-attention connector improves rare-component recall over a "
                "linear projection, but still falls short of a small-model control unless the "
                "decoder is also adapted.",
            ),
            falsifier=data.get(
                "falsifier",
                "The linear connector matches or exceeds the cross-attention connector on "
                "rare-component recall, or the cross-attention connector already matches the "
                "small-model control.",
            ),
            cells=tuple(ConnectorArm.from_dict(c) for c in data.get("cells", [])),
            rows=[ConnectorMetrics.from_dict(r) for r in data.get("rows", [])],
            arm_means={
                k: dict(v) for k, v in data.get("arm_means", {}).items()
            },
            disposition=data.get("disposition", "inconclusive"),
            disposition_rationale=data.get(
                "disposition_rationale", "no rationale provided"
            ),
            dependency_caveats=list(data.get("dependency_caveats", [])),
            version_stamp=data.get("version_stamp", {}),
        )


def _project_root() -> Path:
    """Return the repository root relative to this module."""
    return Path(__file__).resolve().parents[4]


def _hash_noise(payload: str, span: float = 0.01) -> float:
    """Deterministic noise in ``[-span, span]`` from ``payload``."""
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    normalized = int(digest[:16], 16) / (2 ** 64)
    return (normalized * 2.0 - 1.0) * span


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _arm_label(arm_name: str, seed: int) -> str:
    return f"{arm_name}__s{seed}"


def _arm_config(arm_name: str) -> dict[str, Any]:
    """Return the connector type and capacity overrides for an arm."""
    if arm_name == "current":
        return {
            "connector_type": "none",
            "train_scope": "current",
            "connector_hidden_dim": 256,
            "connector_rank": 32,
            "connector_n_queries": 4,
        }
    if arm_name == "linear":
        return {
            "connector_type": "linear",
            "train_scope": "connector_only",
            "connector_hidden_dim": 256,
            "connector_rank": 32,
            "connector_n_queries": 4,
        }
    if arm_name == "low_rank":
        return {
            "connector_type": "low_rank",
            "train_scope": "connector_only",
            "connector_hidden_dim": 256,
            "connector_rank": 32,
            "connector_n_queries": 4,
        }
    if arm_name == "cross_attention":
        return {
            "connector_type": "cross_attention",
            "train_scope": "connector_only",
            "connector_hidden_dim": 256,
            "connector_rank": 32,
            "connector_n_queries": 4,
        }
    if arm_name == "linear_plus_action_residuals":
        return {
            "connector_type": "linear",
            "train_scope": "connector_plus_action_residuals",
            "connector_hidden_dim": 256,
            "connector_rank": 32,
            "connector_n_queries": 4,
        }
    if arm_name == "linear_shuffled_context":
        return {
            "connector_type": "linear",
            "train_scope": "connector_only",
            "connector_hidden_dim": 256,
            "connector_rank": 32,
            "connector_n_queries": 4,
        }
    if arm_name == "local_target":
        # Full small-model control: same cross-attention family, but with more
        # queries to act as an upper-bound capacity reference.
        return {
            "connector_type": "cross_attention",
            "train_scope": "small_model",
            "connector_hidden_dim": 512,
            "connector_rank": 64,
            "connector_n_queries": 8,
        }
    raise ValueError(f"unknown arm_name: {arm_name!r}")


def build_cells(
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    *,
    d_model: int = _DEFAULT_D_MODEL,
    target_decisions: int = _DEFAULT_TARGET_DECISIONS,
) -> tuple[ConnectorArm, ...]:
    """Build the 7 arms × seeds connector-capacity cells."""
    cells: list[ConnectorArm] = []
    for seed in seeds:
        for arm_name in ARM_NAMES:
            cfg = _arm_config(arm_name)
            cells.append(
                ConnectorArm(
                    arm_id=_arm_label(arm_name, seed),
                    arm_name=arm_name,
                    connector_type=cfg["connector_type"],
                    train_scope=cfg["train_scope"],
                    seed=seed,
                    d_model=d_model,
                    connector_hidden_dim=cfg["connector_hidden_dim"],
                    connector_rank=cfg["connector_rank"],
                    connector_n_queries=cfg["connector_n_queries"],
                    connector_freeze_encoder=True,
                    target_decisions=target_decisions,
                )
            )
    return tuple(cells)


def validate_manifest(cells: tuple[ConnectorArm, ...]) -> list[str]:
    """Validate the connector-capacity manifest."""
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
        if cell.connector_type not in CONNECTOR_TYPES:
            errors.append(
                f"{cell.arm_id}: invalid connector_type {cell.connector_type!r}"
            )
        if cell.train_scope not in TRAIN_SCOPES:
            errors.append(
                f"{cell.arm_id}: invalid train_scope {cell.train_scope!r}"
            )
    return errors


def _make_synthetic_tensors(
    batch_size: int,
    seq_len: int,
    d_model: int,
    seed: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic synthetic context vectors and mask."""
    rng = random.Random(seed)
    # Build values with Python random so the fixture is stable across torch
    # versions and CPU/GPU.
    values = [
        [rng.gauss(0.0, 1.0) for _ in range(d_model)] for _ in range(batch_size * seq_len)
    ]
    context_vectors = torch.tensor(values, dtype=torch.float32, device=device).reshape(
        batch_size, seq_len, d_model
    )
    mask_values = [
        [rng.random() < 0.8 for _ in range(seq_len)] for _ in range(batch_size)
    ]
    mask = torch.tensor(mask_values, dtype=torch.bool, device=device)
    # Guarantee at least one valid token per row so pooling is well-defined.
    mask[:, 0] = True
    return context_vectors, mask


def _action_description_residual(
    batch_size: int,
    d_model: int,
    seed: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Return a deterministic action-description residual vector for the batch."""
    catalog = ActionDescriptionCatalog.build()
    encoder = FixtureDescriptionEncoder(d_model)
    keys = catalog.keys()
    if not keys:
        return torch.zeros(batch_size, d_model, device=device)
    rng = random.Random(seed)
    sample_keys = [rng.choice(keys) for _ in range(min(8, len(keys)))]
    vectors = [encoder.encode(catalog.by_key[k].description) for k in sample_keys]
    residual = torch.stack(vectors).mean(dim=0)
    return residual.unsqueeze(0).expand(batch_size, -1).to(device=device)


def _shuffle_context(
    context_vectors: torch.Tensor, seed: int
) -> torch.Tensor:
    """Shuffle the sequence dimension of ``context_vectors`` deterministically."""
    seq_len = context_vectors.shape[1]
    rng = random.Random(seed)
    perm = list(range(seq_len))
    rng.shuffle(perm)
    return context_vectors[:, perm, :]


def _build_connector(cell: ConnectorArm, device: str = "cpu") -> SemanticConnector:
    return SemanticConnector(
        connector_type=cell.connector_type,
        d_model=cell.d_model,
        connector_hidden_dim=cell.connector_hidden_dim,
        connector_rank=cell.connector_rank,
        connector_n_queries=cell.connector_n_queries,
        connector_freeze_encoder=cell.connector_freeze_encoder,
    ).to(device)


def _base_rare_recall(arm_name: str) -> float:
    """Deterministic rare-recall baseline for each arm.

    The ordering is:
    current ≈ linear_shuffled_context < linear < linear_plus_action_residuals
    < low_rank < cross_attention < local_target.
    """
    return {
        "current": 0.35,
        "linear_shuffled_context": 0.36,
        "linear": 0.40,
        "linear_plus_action_residuals": 0.43,
        "low_rank": 0.45,
        "cross_attention": 0.50,
        "local_target": 0.55,
    }[arm_name]


def _simulate_cell(
    cell: ConnectorArm,
    *,
    batch_size: int = _DEFAULT_BATCH,
    seq_len: int = _DEFAULT_SEQ,
    device: str = "cpu",
) -> ConnectorMetrics:
    """Instantiate the connector, run a forward pass, and emit synthetic metrics."""
    start = time.perf_counter()
    context_vectors, mask = _make_synthetic_tensors(
        batch_size, seq_len, cell.d_model, cell.seed, device
    )

    notes: list[str] = [
        f"connector_type={cell.connector_type}",
        f"train_scope={cell.train_scope}",
        "fixture-only: synthetic capacity comparison",
    ]

    if cell.arm_name == "linear_plus_action_residuals":
        residual = _action_description_residual(batch_size, cell.d_model, cell.seed, device)
        # Add the residual to every position so the connector sees action-biased
        # context without changing the sequence geometry.
        context_vectors = context_vectors + residual.unsqueeze(1)
        notes.append("added deterministic action-description residual")

    if cell.arm_name == "linear_shuffled_context":
        context_vectors = _shuffle_context(context_vectors, cell.seed)
        notes.append("shuffled context sequence order")

    connector = _build_connector(cell, device)
    with torch.no_grad():
        output = connector(context_vectors, mask)

    trainable_params = count_connector_parameters(connector)
    frozen_params = sum(
        int(p.numel()) for p in connector.parameters() if not p.requires_grad
    )
    estimated_flops = estimate_connector_flops(
        connector, batch_size, seq_len, cell.d_model
    )

    # Synthetic quality is driven by the arm's capacity baseline, not the random
    # forward pass, so the matrix is deterministic and reproduces the intended
    # capacity ordering.
    base = _base_rare_recall(cell.arm_name)
    rare_recall = _clamp(
        base + _hash_noise(f"{cell.arm_id}:{MATRIX_VERSION}", span=0.01)
    )
    meaningful_program_rate = _clamp(
        rare_recall - 0.02 + _hash_noise(f"mp:{cell.arm_id}")
    )
    common_component_recall = _clamp(
        rare_recall + 0.08 + _hash_noise(f"cc:{cell.arm_id}")
    )
    parse_validity_rate = _clamp(
        rare_recall + 0.05 + _hash_noise(f"pv:{cell.arm_id}")
    )
    first_attempt_quality = _clamp(
        rare_recall - 0.01 + _hash_noise(f"faq:{cell.arm_id}")
    )

    elapsed = time.perf_counter() - start
    wall_seconds = _clamp(
        elapsed
        + 0.001 * (trainable_params / max(1, cell.d_model * cell.d_model))
        + _hash_noise(f"wall:{cell.arm_id}", span=0.02),
        low=0.001,
        high=10.0,
    )

    # Touch output so the compiler/linter cannot complain about unused variables.
    _ = output.to_dict()

    return ConnectorMetrics(
        arm_id=cell.arm_id,
        arm_name=cell.arm_name,
        connector_type=cell.connector_type,
        train_scope=cell.train_scope,
        seed=cell.seed,
        trainable_params=trainable_params,
        frozen_params=frozen_params,
        estimated_flops=estimated_flops,
        rare_component_recall=rare_recall,
        meaningful_program_rate=meaningful_program_rate,
        common_component_recall=common_component_recall,
        parse_validity_rate=parse_validity_rate,
        first_attempt_quality=first_attempt_quality,
        target_decisions=cell.target_decisions,
        wall_seconds=wall_seconds,
        notes=notes,
    )


def _arm_means(rows: list[ConnectorMetrics]) -> dict[str, dict[str, float]]:
    """Aggregate per-arm means across seeds."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        bucket = grouped.setdefault(row.arm_name, {})
        for key in (
            "rare_component_recall",
            "meaningful_program_rate",
            "common_component_recall",
            "parse_validity_rate",
            "first_attempt_quality",
            "trainable_params",
            "estimated_flops",
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
    current = arm_means.get("current", {}).get("rare_component_recall", 0.0)
    linear = arm_means.get("linear", {}).get("rare_component_recall", 0.0)
    low_rank = arm_means.get("low_rank", {}).get("rare_component_recall", 0.0)
    cross = arm_means.get("cross_attention", {}).get("rare_component_recall", 0.0)
    control = arm_means.get("local_target", {}).get("rare_component_recall", 0.0)

    diff_linear = linear - current
    diff_cross = cross - low_rank
    diff_control = control - cross

    if diff_linear < 0.03 and diff_cross < 0.03:
        return (
            "data_or_objective_limited",
            "Neither linear nor cross-attention connectors improve over the current "
            "baseline; the limitation appears to be data or objective rather than "
            "connector capacity.",
        )
    if diff_linear >= 0.05 and cross <= linear + 0.02:
        return (
            "linear_sufficient",
            "The linear connector already captures most of the gains; adding nonlinear "
            "capacity does not materially improve rare-component recall.",
        )
    if diff_cross >= 0.04 and diff_control <= 0.04:
        return (
            "nonlinear_connector_needed",
            "The cross-attention connector closes most of the gap to the small-model "
            "control, so a nonlinear connector is warranted.",
        )
    if diff_cross >= 0.04 and diff_control > 0.04:
        return (
            "decoder_adaptation_needed",
            "The cross-attention connector improves over linear/low-rank variants but "
            "still falls short of the small-model control, suggesting decoder-side "
            "adaptation is also needed.",
        )
    return (
        "inconclusive",
        "The capacity ordering is inconsistent with the expected progression; "
        "additional seeds or real-model measurements are needed.",
    )


def run_fixture_campaign(
    cells: tuple[ConnectorArm, ...] | None = None,
    *,
    run_id: str = "slm166-connector-capacity",
    output_dir: Path | None = None,
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    device: str = "cpu",
) -> ConnectorReport:
    """Run the SLM-166 connector-capacity fixture campaign."""
    cells = cells or build_cells(seeds)
    errors = validate_manifest(cells)
    if errors:
        raise ValueError("manifest validation failed: " + "; ".join(errors))

    rows = [_simulate_cell(cell, device=device) for cell in cells]
    means = _arm_means(rows)
    disposition, rationale = resolve_disposition(means)

    hypothesis = (
        "A nonlinear cross-attention connector between the frozen context encoder "
        "and the sparse grammar-action scorer improves rare-component recall over a "
        "linear projection and a low-rank bottleneck, but a full small-model control "
        "still outperforms it unless the decoder is also adapted."
    )
    falsifier = (
        "The linear connector matches or exceeds the cross-attention connector on "
        "rare-component recall, or the cross-attention connector already matches the "
        "small-model control."
    )

    report = ConnectorReport(
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
        dependency_caveats=[],
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm166_connector_capacity",
            "model.twotower",
        ),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm166_connector_capacity_report.json")
    return report


def render_markdown(report: ConnectorReport) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-166 (SDE1-04): semantic connector capacity fixture ({report.run_id})",
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
        report.hypothesis,
        "",
        "## Falsifier",
        "",
        report.falsifier,
        "",
        "## Connector arms",
        "",
        "| arm_id | arm_name | connector_type | train_scope | seed | d_model | hidden_dim | rank | n_queries |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in report.cells:
        lines.append(
            f"| {cell.arm_id} | {cell.arm_name} | {cell.connector_type} | "
            f"{cell.train_scope} | {cell.seed} | {cell.d_model} | "
            f"{cell.connector_hidden_dim} | {cell.connector_rank} | "
            f"{cell.connector_n_queries} |"
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| arm_id | arm_name | seed | trainable_params | frozen_params | estimated_flops | rare_recall | meaningful_program_rate | common_recall | parse_validity | first_attempt_quality | wall_seconds |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.rows:
        lines.append(
            f"| {row.arm_id} | {row.arm_name} | {row.seed} | "
            f"{row.trainable_params} | {row.frozen_params} | {row.estimated_flops} | "
            f"{row.rare_component_recall:.3f} | {row.meaningful_program_rate:.3f} | "
            f"{row.common_component_recall:.3f} | {row.parse_validity_rate:.3f} | "
            f"{row.first_attempt_quality:.3f} | {row.wall_seconds:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Per-arm means",
            "",
            "| arm_name | rare_recall | meaningful_program_rate | common_recall | parse_validity | first_attempt_quality | trainable_params | estimated_flops |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for arm_name in ARM_NAMES:
        if arm_name not in report.arm_means:
            continue
        m = report.arm_means[arm_name]
        lines.append(
            f"| {arm_name} | {m.get('rare_component_recall', 0.0):.3f} | "
            f"{m.get('meaningful_program_rate', 0.0):.3f} | "
            f"{m.get('common_component_recall', 0.0):.3f} | "
            f"{m.get('parse_validity_rate', 0.0):.3f} | "
            f"{m.get('first_attempt_quality', 0.0):.3f} | "
            f"{m.get('trainable_params', 0.0):.0f} | "
            f"{m.get('estimated_flops', 0.0):.0f} |"
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
            "**No-go for promotion.** This is a wiring fixture. The connector variants, "
            "parameter counts, FLOP estimates, and synthetic metrics are exercised over "
            "deterministic inputs, but no real model was trained or evaluated. The "
            "mechanism remains ``retain_diagnostic`` / ``blocked_pending_real_model`` "
            "until a trained scorer and AgentV evaluation are available.",
            "",
            "## Honest caveats",
            "",
            "- Metrics are generated by a deterministic capacity simulator, not a trained model.",
            "- Seed noise is bounded to ±0.01 so the capacity ordering dominates.",
            "- The synthetic simulator is tuned to make cross-attention better than linear/low-rank "
            "  but still short of the small-model control; real measurements could differ.",
            "- No Pareto or ship-gate claim is made; this is wiring evidence only.",
            "",
            "## Reproducibility",
            "",
            "```bash",
            "python -m scripts.run_slm166_connector_capacity_fixture --mode plan-only",
            "python -m scripts.run_slm166_connector_capacity_fixture --mode fixture",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
