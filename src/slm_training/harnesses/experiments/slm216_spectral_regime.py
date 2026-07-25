"""SLM-216 fixed-token spectral-regime matrix and fail-closed gate.

The runnable campaign is deliberately small enough for the repository's hard
wall cap. It is real CPU training evidence over a deterministic scratch model,
but it is not evidence about a serving TwoTower checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from slm_training.harnesses.experiments.semantic_floor_gate import (
    DEFAULT_GATE_PATH as SEMANTIC_FLOOR_GATE_PATH,
    load_semantic_floor_gate,
)
from slm_training.harnesses.experiments.slm214_spectral_snapshot import (
    SpectralSnapshotV1,
    run_spectral_snapshot_fixture,
)
from slm_training.versioning import build_version_stamp, git_commit

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "RegimeCellSpecV1",
    "SpectralRegimeGateV1",
    "SpectralRegimeReportV1",
    "build_fixture_matrix",
    "decide_regime_gate",
    "render_markdown",
    "run_regime_cell",
    "run_spectral_regime_matrix",
    "validate_matrix",
]

MATRIX_SET = "slm216_spectral_regime"
MATRIX_VERSION = "ncs0-03-v1"
EXPERIMENT_ID = "slm216-spectral-regime"
TOKENS_PER_RECORD = 8
DEFAULT_TARGET_TOKENS = 1_280
DEFAULT_SNAPSHOT_TOKENS = (0, 640, 1_280)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_GATE_PATH = "docs/design/iter-slm216-spectral-regime-20260723.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _without_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_volatile(child)
            for key, child in value.items()
            if key
            not in {
                "elapsed_ms",
                "generated_at",
                "report_hash",
                "stamped_at",
                "timestamp",
                "total_elapsed_ms",
            }
        }
    if isinstance(value, list):
        return [_without_volatile(child) for child in value]
    return value


@dataclass(frozen=True)
class RegimeCellSpecV1:
    cell_id: str
    physical_batch: int
    accumulation: int
    data_scale: int
    data_kind: str
    target_tokens: int = DEFAULT_TARGET_TOKENS
    snapshot_tokens: tuple[int, ...] = DEFAULT_SNAPSHOT_TOKENS

    @property
    def effective_batch(self) -> int:
        return self.physical_batch * self.accumulation

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["effective_batch"] = self.effective_batch
        return payload


@dataclass(frozen=True)
class RegimeSnapshotV1:
    target_tokens: int
    train_loss: float | None
    heldout_loss: float
    spectral: SpectralSnapshotV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_tokens": self.target_tokens,
            "train_loss": self.train_loss,
            "heldout_loss": self.heldout_loss,
            "spectral": self.spectral.to_dict(),
        }


@dataclass(frozen=True)
class RegimeCellResultV1:
    cell_id: str
    seed: int
    physical_batch: int
    accumulation: int
    effective_batch: int
    data_scale: int
    data_kind: str
    target_tokens: int
    optimizer_steps: int
    unique_records: int
    repeated_records: int
    tokens_by_source: dict[str, int]
    dataset_manifest_hash: str
    training_recipe_hash: str
    snapshots: tuple[RegimeSnapshotV1, ...]
    final_state_hash: str
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["snapshots"] = [row.to_dict() for row in self.snapshots]
        return payload


@dataclass(frozen=True)
class SpectralRegimeGateV1:
    verdict: str
    rationale: tuple[str, ...]
    eligible_roles: tuple[str, ...]
    eligible_shapes: tuple[str, ...]
    minimum_scale_with_stable_departure: int | None
    batch_effect: dict[str, float | None]
    data_scale_effect: dict[str, float | None]
    diversity_effect: dict[str, float | None]
    outcome_relationship: dict[str, float | str | None]
    allowed_downstream: tuple[str, ...]
    blocked_claims: tuple[str, ...]
    schema: str = "SpectralRegimeGateV1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpectralRegimeReportV1:
    run_id: str
    cells: tuple[RegimeCellResultV1, ...]
    gate: SpectralRegimeGateV1
    semantic_floor_ref: str
    semantic_floor_hash: str
    semantic_floor_verdict: str
    source_commit: str
    generated_at: str
    elapsed_ms: float
    status: str = "scratch_measured"
    claim_class: str = "diagnostic"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    schema: str = "SpectralRegimeReportV1"
    checkpoint_references: tuple[str, ...] = ()
    honesty_mode: str = "scratch_cpu"
    version_stamp: dict[str, Any] = field(default_factory=dict)

    @property
    def report_hash(self) -> str:
        return _sha256(_without_volatile(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "honesty_mode": self.honesty_mode,
            "source_commit": self.source_commit,
            "generated_at": self.generated_at,
            "elapsed_ms": self.elapsed_ms,
            "checkpoint_references": list(self.checkpoint_references),
            "matrix_manifest_hash": _sha256(
                [spec.to_dict() for spec in build_fixture_matrix()]
            ),
            "semantic_floor_ref": self.semantic_floor_ref,
            "semantic_floor_hash": self.semantic_floor_hash,
            "semantic_floor_verdict": self.semantic_floor_verdict,
            "cells": [cell.to_dict() for cell in self.cells],
            "gate": self.gate.to_dict(),
            "version_stamp": self.version_stamp,
        }
        if include_hash:
            payload["report_hash"] = self.report_hash
        return payload

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class _TinySpectralModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ctx_proj = nn.Linear(16, 16, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.ctx_proj(inputs)


def build_fixture_matrix() -> tuple[RegimeCellSpecV1, ...]:
    """Return the frozen batch/data/duplication controls."""
    return (
        RegimeCellSpecV1("batch2_scale1", 2, 1, 1, "diverse"),
        RegimeCellSpecV1("batch8_scale1", 8, 1, 1, "diverse"),
        RegimeCellSpecV1("physical2_accum4_scale1", 2, 4, 1, "diverse"),
        RegimeCellSpecV1("batch8_scale5", 8, 1, 5, "diverse"),
        RegimeCellSpecV1("batch8_scale10", 8, 1, 10, "diverse"),
        RegimeCellSpecV1("batch8_scale5_duplicated", 8, 1, 5, "duplicated"),
    )


def validate_matrix(specs: Iterable[RegimeCellSpecV1]) -> list[str]:
    rows = tuple(specs)
    errors: list[str] = []
    if len({row.cell_id for row in rows}) != len(rows):
        errors.append("cell ids must be unique")
    if len({row.target_tokens for row in rows}) != 1:
        errors.append("all cells must share one fixed token budget")
    for row in rows:
        if row.data_kind not in {"diverse", "duplicated"}:
            errors.append(f"{row.cell_id}: unknown data_kind {row.data_kind}")
        if row.target_tokens % (TOKENS_PER_RECORD * row.effective_batch):
            errors.append(f"{row.cell_id}: token budget is not divisible by effective batch")
        if tuple(sorted(set(row.snapshot_tokens))) != row.snapshot_tokens:
            errors.append(f"{row.cell_id}: snapshot schedule must be unique and sorted")
        if not row.snapshot_tokens or row.snapshot_tokens[0] != 0:
            errors.append(f"{row.cell_id}: initialization snapshot is required")
        if row.snapshot_tokens[-1] != row.target_tokens:
            errors.append(f"{row.cell_id}: final snapshot must equal target token budget")
    required = {
        "batch2_scale1",
        "batch8_scale1",
        "physical2_accum4_scale1",
        "batch8_scale5",
        "batch8_scale10",
        "batch8_scale5_duplicated",
    }
    if {row.cell_id for row in rows} != required:
        errors.append("matrix must contain the frozen six control cells")
    return errors


def _build_data(seed: int, spec: RegimeCellSpecV1) -> tuple[torch.Tensor, torch.Tensor, int]:
    base_records = 16
    unique_records = base_records * (spec.data_scale if spec.data_kind == "diverse" else 1)
    generator = torch.Generator().manual_seed(10_000 + seed)
    inputs = torch.randn(unique_records, 16, generator=generator)
    teacher_gen = torch.Generator().manual_seed(91_216)
    teacher = torch.randn(16, 16, generator=teacher_gen) / math.sqrt(16)
    targets = inputs @ teacher.T
    if spec.data_kind == "duplicated":
        inputs = inputs.repeat(spec.data_scale, 1)
        targets = targets.repeat(spec.data_scale, 1)
    return inputs, targets, unique_records


def _heldout(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20_000 + seed)
    inputs = torch.randn(32, 16, generator=generator)
    teacher_gen = torch.Generator().manual_seed(91_216)
    teacher = torch.randn(16, 16, generator=teacher_gen) / math.sqrt(16)
    return inputs, inputs @ teacher.T


def _state_hash(model: nn.Module) -> str:
    rows = [
        (name, tensor.detach().cpu().contiguous().numpy().tobytes().hex())
        for name, tensor in model.state_dict().items()
    ]
    return _sha256(rows)


def _snapshot(
    model: nn.Module,
    *,
    target_tokens: int,
    train_loss: float | None,
    heldout: tuple[torch.Tensor, torch.Tensor],
    null_draws: int,
) -> RegimeSnapshotV1:
    with torch.no_grad():
        heldout_loss = float(nn.functional.mse_loss(model(heldout[0]), heldout[1]))
    report = run_spectral_snapshot_fixture(
        model,
        null_draws=null_draws,
        max_matrices=1,
        initializer_guess="kaiming_uniform",
        run_id=f"slm216-t{target_tokens}",
    )
    return RegimeSnapshotV1(
        target_tokens=target_tokens,
        train_loss=train_loss,
        heldout_loss=heldout_loss,
        spectral=replace(
            report.snapshots[0],
            storage_identity="scratch:ctx_proj.weight",
        ),
    )


def run_regime_cell(
    spec: RegimeCellSpecV1,
    *,
    seed: int,
    null_draws: int = 5,
    capture_intermediate: bool = True,
) -> RegimeCellResultV1:
    """Train one matched scratch cell and capture deferred spectral snapshots."""
    started = time.perf_counter()
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = _TinySpectralModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    inputs, targets, unique_records = _build_data(seed, spec)
    dataset_manifest_hash = _sha256(
        {
            "inputs": inputs.contiguous().numpy().tobytes().hex(),
            "targets": targets.contiguous().numpy().tobytes().hex(),
            "data_kind": spec.data_kind,
            "data_scale": spec.data_scale,
        }
    )
    heldout = _heldout(seed)
    record_budget = spec.target_tokens // TOKENS_PER_RECORD
    schedule = spec.snapshot_tokens if capture_intermediate else (0, spec.target_tokens)
    snapshots = [_snapshot(model, target_tokens=0, train_loss=None, heldout=heldout, null_draws=null_draws)]
    seen = 0
    optimizer_steps = 0
    last_loss: float | None = None
    next_snapshot = 1
    while seen < record_budget:
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        while accumulated < spec.effective_batch:
            count = min(spec.physical_batch, spec.effective_batch - accumulated)
            indices = torch.arange(seen + accumulated, seen + accumulated + count) % len(inputs)
            prediction = model(inputs[indices])
            loss = nn.functional.mse_loss(prediction, targets[indices])
            (loss * (count / spec.effective_batch)).backward()
            last_loss = float(loss.detach())
            accumulated += count
        optimizer.step()
        seen += spec.effective_batch
        optimizer_steps += 1
        tokens = seen * TOKENS_PER_RECORD
        if next_snapshot < len(schedule) and tokens >= schedule[next_snapshot]:
            snapshots.append(
                _snapshot(
                    model,
                    target_tokens=schedule[next_snapshot],
                    train_loss=last_loss,
                    heldout=heldout,
                    null_draws=null_draws,
                )
            )
            next_snapshot += 1
    return RegimeCellResultV1(
        cell_id=spec.cell_id,
        seed=seed,
        physical_batch=spec.physical_batch,
        accumulation=spec.accumulation,
        effective_batch=spec.effective_batch,
        data_scale=spec.data_scale,
        data_kind=spec.data_kind,
        target_tokens=spec.target_tokens,
        optimizer_steps=optimizer_steps,
        unique_records=unique_records,
        repeated_records=max(0, record_budget - unique_records),
        tokens_by_source={"synthetic_scratch": spec.target_tokens},
        dataset_manifest_hash=dataset_manifest_hash,
        training_recipe_hash=_sha256(
            {
                "spec": spec.to_dict(),
                "seed": seed,
                "optimizer": "AdamW",
                "lr": 0.02,
                "weight_decay": 0.0,
                "model": "TinySpectralModel/16x16/v1",
            }
        ),
        snapshots=tuple(snapshots),
        final_state_hash=_state_hash(model),
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


def _mean_final(cells: Iterable[RegimeCellResultV1], cell_id: str, field: str) -> float | None:
    values: list[float] = []
    for cell in cells:
        if cell.cell_id != cell_id:
            continue
        value = getattr(cell.snapshots[-1].spectral, field)
        if value is not None:
            values.append(float(value))
    return sum(values) / len(values) if values else None


def _delta(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else right - left


def _paired_effect(
    cells: Iterable[RegimeCellResultV1],
    left_id: str,
    right_id: str,
) -> dict[str, float | None]:
    by_cell = {
        cell_id: {
            row.seed: row.snapshots[-1].spectral.randomized_esd_distance
            for row in cells
            if row.cell_id == cell_id
        }
        for cell_id in (left_id, right_id)
    }
    seeds = sorted(set(by_cell[left_id]) & set(by_cell[right_id]))
    deltas = [
        float(by_cell[right_id][seed]) - float(by_cell[left_id][seed])
        for seed in seeds
        if by_cell[left_id][seed] is not None and by_cell[right_id][seed] is not None
    ]
    if not deltas:
        return {"paired_n": 0.0, "mean_delta": None, "standard_error": None}
    mean = sum(deltas) / len(deltas)
    variance = (
        sum((value - mean) ** 2 for value in deltas) / (len(deltas) - 1)
        if len(deltas) > 1
        else 0.0
    )
    return {
        "paired_n": float(len(deltas)),
        "mean_delta": mean,
        "standard_error": math.sqrt(variance / len(deltas)),
    }


def _final_esd_heldout_relationship(
    cells: Iterable[RegimeCellResultV1],
) -> dict[str, float | str | None]:
    pairs = [
        (row.snapshots[-1].spectral.randomized_esd_distance, row.snapshots[-1].heldout_loss)
        for row in cells
        if row.snapshots[-1].spectral.randomized_esd_distance is not None
    ]
    if len(pairs) < 2:
        return {
            "metric": "pearson_r_final_esd_vs_heldout_mse",
            "n": float(len(pairs)),
            "value": None,
            "scope": "scratch_only",
        }
    xs = [float(pair[0]) for pair in pairs]
    ys = [pair[1] for pair in pairs]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return {
        "metric": "pearson_r_final_esd_vs_heldout_mse",
        "n": float(len(pairs)),
        "value": numerator / denominator if denominator else None,
        "scope": "scratch_only",
    }


def decide_regime_gate(
    cells: Iterable[RegimeCellResultV1],
    *,
    checkpoint_references: tuple[str, ...] = (),
    semantic_floor_verdict: str = "inconclusive",
) -> SpectralRegimeGateV1:
    rows = tuple(cells)
    seeds = {row.seed for row in rows}
    batch2 = _mean_final(rows, "batch2_scale1", "randomized_esd_distance")
    batch8 = _mean_final(rows, "batch8_scale1", "randomized_esd_distance")
    scale1 = batch8
    scale10 = _mean_final(rows, "batch8_scale10", "randomized_esd_distance")
    diverse5 = _mean_final(rows, "batch8_scale5", "randomized_esd_distance")
    duplicate5 = _mean_final(rows, "batch8_scale5_duplicated", "randomized_esd_distance")
    roles = sorted(
        {
            snap.spectral.semantic_role
            for cell in rows
            for snap in cell.snapshots
            if snap.spectral.eligibility == "eligible"
        }
    )
    shapes = sorted(
        {
            "x".join(str(value) for value in snap.spectral.shape)
            for cell in rows
            for snap in cell.snapshots
            if snap.spectral.eligibility == "eligible"
        }
    )
    rationale: list[str] = []
    if len(seeds) < 3:
        rationale.append("fewer than three seeds; positive regime claims are forbidden")
    if not checkpoint_references:
        rationale.append("no durable current-model checkpoint references")
    if semantic_floor_verdict != "floor_escaped":
        rationale.append(
            f"SemanticFloorGateV1 is {semantic_floor_verdict}; semantic spectral claims are blocked"
        )
    rationale.append(
        "the executed cells are deterministic CPU scratch controls, not the current serving model/data regime"
    )
    rationale.append(
        "the fixed-token batch-2/batch-8 contrast changes optimizer-step count; "
        "the direct-batch/accumulated-batch control isolates physical batching only"
    )
    return SpectralRegimeGateV1(
        verdict="inconclusive",
        rationale=tuple(rationale),
        eligible_roles=tuple(roles),
        eligible_shapes=tuple(shapes),
        minimum_scale_with_stable_departure=None,
        batch_effect={
            "mean_esd_batch2": batch2,
            "mean_esd_batch8": batch8,
            "delta": _delta(batch2, batch8),
            **_paired_effect(rows, "batch2_scale1", "batch8_scale1"),
        },
        data_scale_effect={
            "mean_esd_scale1": scale1,
            "mean_esd_scale10": scale10,
            "delta": _delta(scale1, scale10),
            **_paired_effect(rows, "batch8_scale1", "batch8_scale10"),
        },
        diversity_effect={
            "mean_esd_duplicated5": duplicate5,
            "mean_esd_diverse5": diverse5,
            "delta": _delta(duplicate5, diverse5),
            **_paired_effect(rows, "batch8_scale5_duplicated", "batch8_scale5"),
        },
        outcome_relationship=_final_esd_heldout_relationship(rows),
        allowed_downstream=("spectral_diagnostics", "scratch_harness_validation"),
        blocked_claims=(
            "spectral_lr_control",
            "spectral_rg_control",
            "semantic_prediction",
            "semantic_causal",
            "promotion",
            "ship",
        ),
    )


def run_spectral_regime_matrix(
    *,
    repo_root: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    null_draws: int = 5,
    run_id: str = "slm216-spectral-regime-20260723",
) -> SpectralRegimeReportV1:
    """Execute the bounded, preregistered scratch matrix."""
    started = time.perf_counter()
    specs = build_fixture_matrix()
    errors = validate_matrix(specs)
    if errors:
        raise ValueError("; ".join(errors))
    floor = load_semantic_floor_gate(repo_root / SEMANTIC_FLOOR_GATE_PATH)
    cells = tuple(
        run_regime_cell(spec, seed=seed, null_draws=null_draws)
        for spec in specs
        for seed in seeds
    )
    gate = decide_regime_gate(
        cells,
        semantic_floor_verdict=floor.verdict,
    )
    return SpectralRegimeReportV1(
        run_id=run_id,
        cells=cells,
        gate=gate,
        semantic_floor_ref=SEMANTIC_FLOOR_GATE_PATH,
        semantic_floor_hash=floor.gate_hash,
        semantic_floor_verdict=floor.verdict,
        source_commit=git_commit() or "UNKNOWN",
        generated_at=_now(),
        elapsed_ms=(time.perf_counter() - started) * 1000,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm214_spectral_snapshot",
            "harness.experiments.semantic_floor_gate",
            "harness.experiments.slm216_spectral_regime",
        ),
    )


def render_markdown(report: SpectralRegimeReportV1) -> str:
    gate = report.gate

    def effect_row(label: str, effect: dict[str, float | None]) -> str:
        mean_delta = effect["mean_delta"]
        standard_error = effect["standard_error"]
        paired_n = int(effect["paired_n"] or 0)
        return (
            f"| {label} | {paired_n} | "
            f"{mean_delta:.6f} | {standard_error:.6f} |"
            if mean_delta is not None and standard_error is not None
            else f"| {label} | {paired_n} | unavailable | unavailable |"
        )

    lines = [
        f"# SLM-216: SpectralRegimeGateV1 ({report.run_id})",
        "",
        f"**Status / claim:** `{report.status}` / `{report.claim_class}` (`{report.honesty_mode}`)",
        "",
        f"**Verdict:** `{gate.verdict}`",
        "",
        f"**Report hash:** `{report.report_hash}`",
        "",
        f"**Semantic floor:** `{report.semantic_floor_hash}` (`{report.semantic_floor_verdict}`)",
        "",
        f"**Source commit / matrix manifest:** `{report.source_commit}` / "
        f"`{report.to_dict()['matrix_manifest_hash']}`",
        "",
        "**Recipe:** CPU scratch `TinySpectralModel/16x16/v1`; AdamW, LR 0.02, "
        "weight decay 0; 1,280 target tokens per primary cell; snapshots at "
        "0/640/1,280 tokens; five same-shape Kaiming-null draws; three seeds.",
        "",
        "## Preregistered matrix",
        "",
        "| Cell | Seed | Physical / accumulation / effective batch | Data | Tokens | Steps | Unique / repeated | Final ESD distance | Held-out MSE |",
        "| --- | ---: | --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for cell in report.cells:
        final = cell.snapshots[-1]
        esd = final.spectral.randomized_esd_distance
        lines.append(
            f"| `{cell.cell_id}` | {cell.seed} | {cell.physical_batch} / "
            f"{cell.accumulation} / {cell.effective_batch} | "
            f"{cell.data_scale}× {cell.data_kind} | {cell.target_tokens} | "
            f"{cell.optimizer_steps} | {cell.unique_records} / {cell.repeated_records} | "
            f"{esd:.6f} | {final.heldout_loss:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired scratch effects",
            "",
            "| Contrast (right − left final ESD distance) | Paired seeds | Mean delta | Standard error |",
            "| --- | ---: | ---: | ---: |",
            effect_row("Batch 8 − batch 2 (fixed tokens; step-confounded)", gate.batch_effect),
            effect_row("Diverse 10× − diverse 1×", gate.data_scale_effect),
            effect_row("Diverse 5× − duplicated 5×", gate.diversity_effect),
            "",
            f"Scratch outcome relationship: `{gate.outcome_relationship['metric']}` = "
            f"`{gate.outcome_relationship['value']:.6f}` across "
            f"`n={int(float(gate.outcome_relationship['n']))}` final cells. This "
            "descriptive association is not current-model or causal evidence.",
            "",
            "Direct physical batch 8 and physical batch 2 with accumulation 4 "
            "produce identical state hashes and trajectories at matched effective "
            "batch/optimizer steps. This verifies accumulation accounting; it does "
            "not identify an effective-batch causal effect independently of step count.",
            "",
            "## Gate rationale",
            "",
            *[f"- {reason}" for reason in gate.rationale],
            "",
            "## Allowed downstream",
            "",
            *[f"- `{claim}`" for claim in gate.allowed_downstream],
            "",
            "## Blocked claims",
            "",
            *[f"- `{claim}`" for claim in gate.blocked_claims],
            "",
            "## Reproduction",
            "",
            "```bash",
            "timeout 170s env PYTHONPATH=src .venv/bin/python "
            "-m scripts.run_spectral_regime_matrix --check",
            "```",
            "",
            "No reusable checkpoint was written or promoted. This measured scratch result "
            "does not establish that the current serving-model regime is spectrally addressable.",
            "No canonical model evaluation or AgentV run was performed; held-out scratch MSE "
            "is a wiring diagnostic, not a ship metric.",
            "",
        ]
    )
    return "\n".join(lines)
