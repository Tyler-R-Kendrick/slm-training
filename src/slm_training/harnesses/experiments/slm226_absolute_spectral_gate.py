"""SLM-226 finite-size boundary and fail-closed absolute spectral target gate."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn

from slm_training.harnesses.experiments.slm214_spectral_snapshot import (
    SpectralSnapshotV1,
    make_pareto_tail_matrix,
    make_spiked_matrix,
    run_spectral_snapshot_fixture,
    sample_null_summary,
)
from slm_training.versioning import build_version_stamp, git_commit

MATRIX_SET = "slm226_absolute_spectral_boundary"
MATRIX_VERSION = "ncs4-01-v1"
DEFAULT_REPORT_PATH = "docs/design/iter-slm226-absolute-spectral-gate-20260723.json"
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_NULL_DRAWS = 200


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


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


@contextmanager
def _single_threaded_torch() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@dataclass(frozen=True)
class WidthShapeSpecV1:
    width: int
    role: str
    rows: int
    cols: int
    head_count: int
    head_dim: int = 64
    initializer: str = "gaussian"

    @property
    def shape_id(self) -> str:
        return f"{self.rows}x{self.cols}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "shape_id": self.shape_id}


@dataclass(frozen=True)
class NullShapeEvidenceV1:
    spec: WidthShapeSpecV1
    draws: int
    null_key: str
    mean_alpha: float
    sd_alpha: float
    alpha_95_interval: tuple[float, float]
    alpha_two_z: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spec"] = self.spec.to_dict()
        return payload


@dataclass(frozen=True)
class TrainedShapeEvidenceV1:
    spec: WidthShapeSpecV1
    seed: int
    target_tokens: int
    optimizer_steps: int
    init_snapshot: SpectralSnapshotV1
    final_snapshot: SpectralSnapshotV1
    final_loss: float
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "seed": self.seed,
            "target_tokens": self.target_tokens,
            "optimizer_steps": self.optimizer_steps,
            "init_snapshot": self.init_snapshot.to_dict(),
            "final_snapshot": self.final_snapshot.to_dict(),
            "final_loss": self.final_loss,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass(frozen=True)
class SyntheticControlV1:
    shape_id: str
    kind: str
    declared_alpha: float | None
    snapshot: SpectralSnapshotV1

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape_id": self.shape_id,
            "kind": self.kind,
            "declared_alpha": self.declared_alpha,
            "snapshot": self.snapshot.to_dict(),
        }


@dataclass(frozen=True)
class AbsoluteSpectralTargetGateV1:
    verdict: str
    rationale: tuple[str, ...]
    authorized_roles: tuple[str, ...]
    authorized_shapes: tuple[str, ...]
    allowed_downstream: tuple[str, ...]
    blocked_interventions: tuple[str, ...]
    minimum_tested_width: int
    absolute_target: float
    causal_shape_effect_supported: bool
    durable_checkpoint_families: int
    semantic_floor_verdict: str
    schema: str = "AbsoluteSpectralTargetGateV1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AbsoluteSpectralBoundaryReportV1:
    run_id: str
    null_shapes: tuple[NullShapeEvidenceV1, ...]
    trained_shapes: tuple[TrainedShapeEvidenceV1, ...]
    controls: tuple[SyntheticControlV1, ...]
    gate: AbsoluteSpectralTargetGateV1
    prerequisite_refs: dict[str, str]
    source_commit: str
    generated_at: str
    elapsed_ms: float
    schema: str = "AbsoluteSpectralBoundaryReportV1"
    status: str = "scratch_measured"
    claim_class: str = "descriptive_diagnostic"
    honesty_mode: str = "scratch_cpu_no_durable_checkpoint"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    checkpoint_references: tuple[str, ...] = ()
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
                [spec.to_dict() for spec in build_width_manifest()]
            ),
            "prerequisite_refs": self.prerequisite_refs,
            "null_shapes": [row.to_dict() for row in self.null_shapes],
            "trained_shapes": [row.to_dict() for row in self.trained_shapes],
            "controls": [row.to_dict() for row in self.controls],
            "gate": self.gate.to_dict(),
            "version_stamp": self.version_stamp,
        }
        if include_hash:
            payload["report_hash"] = self.report_hash
        return payload


def build_width_manifest() -> tuple[WidthShapeSpecV1, ...]:
    """Return the frozen width/shape family."""
    return (
        WidthShapeSpecV1(128, "ctx_proj", 128, 128, 2),
        WidthShapeSpecV1(256, "ctx_proj", 256, 128, 4),
        WidthShapeSpecV1(512, "ctx_proj", 512, 128, 8),
    )


class _WidthProbe(nn.Module):
    def __init__(self, spec: WidthShapeSpecV1, seed: int) -> None:
        super().__init__()
        self.ctx_proj = nn.Linear(spec.cols, spec.rows, bias=False)
        generator = torch.Generator().manual_seed(22_600 + seed + spec.width)
        with torch.no_grad():
            self.ctx_proj.weight.copy_(
                torch.randn(spec.rows, spec.cols, generator=generator)
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.ctx_proj(value)


def _calibrate(
    snapshot: SpectralSnapshotV1,
    summary: dict[str, Any],
) -> SpectralSnapshotV1:
    mean_alpha = float(summary["mean_alpha"])
    sd_alpha = float(summary["sd_alpha"])
    singular = torch.tensor(snapshot.singular_values)
    mean_singular = torch.tensor(summary["mean_singular_values"])
    count = min(len(singular), len(mean_singular))
    distance = float(torch.linalg.vector_norm(singular[:count] - mean_singular[:count]))
    return replace(
        snapshot,
        null_key=str(summary["null_key"]),
        null_draws=int(summary["draws"]),
        null_mean_alpha=mean_alpha,
        null_sd_alpha=sd_alpha,
        alpha_z=(
            (float(snapshot.hill_alpha) - mean_alpha) / sd_alpha
            if snapshot.hill_alpha is not None and sd_alpha > 0
            else None
        ),
        randomized_esd_distance=distance / max(snapshot.frobenius_norm, 1e-30),
    )


def _snapshot(model: nn.Module, summary: dict[str, Any], run_id: str) -> SpectralSnapshotV1:
    report = run_spectral_snapshot_fixture(
        model,
        null_draws=3,
        max_matrices=1,
        initializer_guess="gaussian",
        run_id=run_id,
    )
    return _calibrate(
        replace(
            report.snapshots[0],
            storage_identity=f"scratch:{run_id}:ctx_proj.weight",
        ),
        summary,
    )


def _run_trained_shape(
    spec: WidthShapeSpecV1,
    *,
    seed: int,
    null_summary: dict[str, Any],
    steps: int = 8,
    batch_size: int = 8,
) -> TrainedShapeEvidenceV1:
    started = time.perf_counter()
    model = _WidthProbe(spec, seed)
    initial = _snapshot(model, null_summary, f"slm226-{spec.shape_id}-s{seed}-init")
    data_generator = torch.Generator().manual_seed(32_600 + seed)
    inputs = torch.randn(batch_size, spec.cols, generator=data_generator)
    teacher_generator = torch.Generator().manual_seed(42_600 + spec.width)
    teacher = torch.randn(spec.rows, spec.cols, generator=teacher_generator)
    targets = inputs @ teacher.T
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    final_loss = 0.0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    final = _snapshot(model, null_summary, f"slm226-{spec.shape_id}-s{seed}-final")
    return TrainedShapeEvidenceV1(
        spec=spec,
        seed=seed,
        target_tokens=steps * batch_size * spec.cols,
        optimizer_steps=steps,
        init_snapshot=initial,
        final_snapshot=final,
        final_loss=final_loss,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


def decide_absolute_gate(
    *,
    causal_shape_effect_supported: bool,
    durable_checkpoint_families: int,
    semantic_floor_verdict: str,
) -> AbsoluteSpectralTargetGateV1:
    rationale = [
        "same-shape random nulls place raw alpha near the proposed target at finite width",
        "scratch probes are descriptive and do not represent a durable serving checkpoint family",
    ]
    if not causal_shape_effect_supported:
        rationale.append("SLM-221 found no reproducible singular-value-shape causal effect")
    if durable_checkpoint_families < 1:
        rationale.append("no provenance-resolvable durable checkpoint family is available")
    if semantic_floor_verdict != "floor_escaped":
        rationale.append(
            f"SemanticFloorGateV1 is {semantic_floor_verdict}; semantic outcome use is blocked"
        )
    return AbsoluteSpectralTargetGateV1(
        verdict="descriptive_only",
        rationale=tuple(rationale),
        authorized_roles=(),
        authorized_shapes=(),
        allowed_downstream=("spectral_diagnostics", "null_calibration"),
        blocked_interventions=("ww_pgd", "trace_log", "alpha_target"),
        minimum_tested_width=128,
        absolute_target=2.0,
        causal_shape_effect_supported=causal_shape_effect_supported,
        durable_checkpoint_families=durable_checkpoint_families,
        semantic_floor_verdict=semantic_floor_verdict,
    )


def require_absolute_spectral_gate(
    gate: AbsoluteSpectralTargetGateV1,
    *,
    claim_or_intervention: str,
    role: str,
    shape: str,
) -> None:
    """Fail closed unless a gate authorizes this exact intervention/role/shape."""
    if claim_or_intervention not in {"ww_pgd", "trace_log", "alpha_target"}:
        raise ValueError(f"unknown absolute spectral intervention: {claim_or_intervention}")
    if (
        gate.verdict != "absolute_target_addressable"
        or role not in gate.authorized_roles
        or shape not in gate.authorized_shapes
    ):
        raise RuntimeError(
            f"{claim_or_intervention} is not authorized for {role}/{shape}: "
            f"gate verdict is {gate.verdict}"
        )


def run_absolute_spectral_boundary(
    *,
    repo_root: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    null_draws: int = DEFAULT_NULL_DRAWS,
    run_id: str = "slm226-absolute-spectral-gate-20260723",
) -> AbsoluteSpectralBoundaryReportV1:
    """Run the bounded scratch/null boundary study."""
    if null_draws < 3:
        raise ValueError("null_draws must be at least 3")
    started = time.perf_counter()
    specs = build_width_manifest()
    summaries: dict[str, dict[str, Any]] = {}
    null_rows: list[NullShapeEvidenceV1] = []
    trained_rows: list[TrainedShapeEvidenceV1] = []
    controls: list[SyntheticControlV1] = []
    with _single_threaded_torch(), torch.random.fork_rng():
        torch.manual_seed(22_600)
        for spec in specs:
            summary = sample_null_summary(
                spec.rows,
                spec.cols,
                torch.float32,
                spec.initializer,
                draws=null_draws,
            )
            summaries[spec.shape_id] = summary
            mean = float(summary["mean_alpha"])
            sd = float(summary["sd_alpha"])
            null_rows.append(
                NullShapeEvidenceV1(
                    spec=spec,
                    draws=null_draws,
                    null_key=str(summary["null_key"]),
                    mean_alpha=mean,
                    sd_alpha=sd,
                    alpha_95_interval=(mean - 1.96 * sd, mean + 1.96 * sd),
                    alpha_two_z=(2.0 - mean) / sd,
                )
            )
        for spec in specs:
            for seed in seeds:
                trained_rows.append(
                    _run_trained_shape(
                        spec,
                        seed=seed,
                        null_summary=summaries[spec.shape_id],
                    )
                )
            for kind, declared, matrix in (
                (
                    "pareto",
                    2.0,
                    make_pareto_tail_matrix(
                        spec.rows,
                        spec.cols,
                        alpha_true=2.0,
                        rank=min(spec.rows, spec.cols),
                    ),
                ),
                ("spiked", None, make_spiked_matrix(spec.rows, spec.cols)),
            ):
                holder = _WidthProbe(spec, 99)
                with torch.no_grad():
                    holder.ctx_proj.weight.copy_(matrix)
                controls.append(
                    SyntheticControlV1(
                        shape_id=spec.shape_id,
                        kind=kind,
                        declared_alpha=declared,
                        snapshot=_snapshot(
                            holder,
                            summaries[spec.shape_id],
                            f"slm226-{spec.shape_id}-{kind}",
                        ),
                    )
                )
    semantic_gate_path = repo_root / "docs/design/semantic-floor-gate-v1.json"
    semantic_gate = json.loads(semantic_gate_path.read_text(encoding="utf-8"))
    regime_path = repo_root / "docs/design/iter-slm216-spectral-regime-20260723.json"
    regime = json.loads(regime_path.read_text(encoding="utf-8"))
    gate = decide_absolute_gate(
        causal_shape_effect_supported=False,
        durable_checkpoint_families=0,
        semantic_floor_verdict=str(semantic_gate["verdict"]),
    )
    return AbsoluteSpectralBoundaryReportV1(
        run_id=run_id,
        null_shapes=tuple(null_rows),
        trained_shapes=tuple(trained_rows),
        controls=tuple(controls),
        gate=gate,
        prerequisite_refs={
            "SLM-216": str(regime["report_hash"]),
            "SLM-221": "closed_no_eligible_perturbation_bands",
            "SLM-223": "closed_no_authorized_spectral_control",
            "SemanticFloorGateV1": str(semantic_gate["gate_hash"]),
        },
        source_commit=git_commit() or "UNKNOWN",
        generated_at=_now(),
        elapsed_ms=(time.perf_counter() - started) * 1000,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm214_spectral_snapshot",
            "harness.experiments.slm216_spectral_regime",
            "harness.experiments.slm226_absolute_spectral_gate",
        ),
    )


def render_markdown(report: AbsoluteSpectralBoundaryReportV1) -> str:
    lines = [
        f"# SLM-226: AbsoluteSpectralTargetGateV1 ({report.run_id})",
        "",
        f"**Verdict:** `{report.gate.verdict}`",
        "",
        f"**Report hash:** `{report.report_hash}`",
        "",
        f"**Status / claim:** `{report.status}` / `{report.claim_class}` "
        f"(`{report.honesty_mode}`)",
        "",
        "## Null finite-size boundary",
        "",
        "| Width / role / shape | Draws | Mean alpha | SD | 95% null interval | z(alpha=2) |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in report.null_shapes:
        lines.append(
            f"| {row.spec.width} / `{row.spec.role}` / `{row.spec.shape_id}` | "
            f"{row.draws} | {row.mean_alpha:.6f} | {row.sd_alpha:.6f} | "
            f"[{row.alpha_95_interval[0]:.6f}, {row.alpha_95_interval[1]:.6f}] | "
            f"{row.alpha_two_z:.3f} |"
        )
    lines.extend(
        [
            "",
            "Raw alpha is shape-dependent and is never interpreted without this "
            "same-shape null. In particular, proximity to alpha=2 is not an "
            "authorization signal.",
            "",
            "## Scratch trained probes",
            "",
            "| Width / shape | Seed | Steps / tokens | Init alpha | Final alpha | "
            "Init/final null distance | Final MSE |",
            "| --- | ---: | --- | ---: | ---: | --- | ---: |",
        ]
    )
    for row in report.trained_shapes:
        lines.append(
            f"| {row.spec.width} / `{row.spec.shape_id}` | {row.seed} | "
            f"{row.optimizer_steps} / {row.target_tokens} | "
            f"{row.init_snapshot.hill_alpha:.6f} | {row.final_snapshot.hill_alpha:.6f} | "
            f"{row.init_snapshot.randomized_esd_distance:.6f} / "
            f"{row.final_snapshot.randomized_esd_distance:.6f} | "
            f"{row.final_loss:.6f} |"
        )
    lines.extend(
        [
            "",
            "The probes are deterministic CPU linear-role diagnostics, not full "
            "TwoTower checkpoints or quality evidence. They cannot establish an "
            "absolute optimum or minimum production width.",
            "",
            "## Gate rationale",
            "",
            *[f"- {reason}" for reason in report.gate.rationale],
            "",
            "## Disposition",
            "",
            "Only null-calibrated spectral diagnostics remain allowed. `ww_pgd`, "
            "`trace_log`, and `alpha_target` are blocked for every role and shape; "
            "the guard helper fails closed on this gate.",
            "",
            "**Recipe:** CPU, one PyTorch thread; Gaussian 128x128, 256x128, and "
            f"512x128 nulls ({report.null_shapes[0].draws} draws per shape); "
            "Pareto/spiked controls; three deterministic seeds; AdamW, 8 steps, "
            "no reusable checkpoint.",
            "",
            "No canonical model evaluation or AgentV run was performed because this "
            "was a spectral profile, not a model-quality evaluation. No checkpoint "
            "was written or promoted.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "timeout 170s env PYTHONPATH=src "
            "/home/codex/repos/slm-training/.venv/bin/python "
            "-m scripts.run_absolute_spectral_gate --check",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
