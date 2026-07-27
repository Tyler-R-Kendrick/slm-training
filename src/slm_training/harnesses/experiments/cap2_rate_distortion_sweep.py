"""CAP2-06 program-factor latent size/width/codec rate-distortion sweep (AP-030 / SLM-328).

Sweeps latent count (positions/codebook exponent) x encoder width (hidden_dim)
x codec family (uniform scalar, mixed-radix FSQ, binary LFQ, learned VQ,
continuous) over the SLM-325 `SemanticPlanV1` program-factor tensorizer, to
find the smallest oracle latent representation that preserves each factor
family and to identify which factor crosses a distortion threshold first as
capacity shrinks.

Scope decision (see the design doc's honest caveats): this issue's own
"Implementation" section only requires adding a token/tree decoder "after
factor gates pass," i.e. real `meaning-v2`, binder/reference F1, canonical
AST, and G0-G8 metrics against generated OpenUI programs are a later,
gated stage that needs the full compiler/evaluator pipeline wired to a real
decoder -- infrastructure this bounded CPU wiring pass does not build. This
harness measures the stage the issue's implementation section actually
describes here: per-factor-family reconstruction distortion (MSE) as a
function of latent count, width, and codec family, using a single linear
decoder matched to the full concatenated factor vector (sliced per family
for reporting) rather than one bespoke decoder network per family. The
distortion-threshold "gate" in this pass is an explicit MSE proxy, not
meaning-v2/binder-F1; promotion past this stage into real semantic-quality
evaluation is future work, not claimed here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.models.binary_lfq import BinaryLFQCodec, BinaryLFQConfig
from slm_training.models.continuous_latent import ContinuousLatentCodec, ContinuousLatentConfig
from slm_training.models.latent_codec import LatentCodec
from slm_training.models.latent_codec_trainer import audit_no_bypass
from slm_training.models.learned_vq import LearnedVQCodec, LearnedVQConfig
from slm_training.models.mixed_radix_fsq import MixedRadixFSQCodec, MixedRadixFSQConfig
from slm_training.models.semantic_plan_factors import (
    PROGRAM_FACTOR_FAMILIES,
    ProgramFactorTensorizerConfig,
    concat_factor_tensors,
    factor_dims,
    tensorize_semantic_plan,
)
from slm_training.models.uniform_scalar_codec import (
    UniformScalarCodec,
    UniformScalarCodecConfig,
)

__all__ = [
    "RATE_DISTORTION_CODECS",
    "RATE_DISTORTION_LATENT_COUNTS",
    "RATE_DISTORTION_WIDTHS",
    "DISTORTION_THRESHOLD",
    "RateDistortionCell",
    "RateDistortionResult",
    "RateDistortionSweepReport",
    "FactorReconstructionModel",
    "train_factor_reconstruction",
    "evaluate_factor_reconstruction",
    "build_sweep",
    "evaluate_cell",
    "run_sweep",
    "select_retained_configs",
]

# Literal sweep axes from the issue's own "Implementation" bullet.
RATE_DISTORTION_LATENT_COUNTS: tuple[int, ...] = (4, 8, 16, 32)
RATE_DISTORTION_WIDTHS: tuple[int, ...] = (64, 128, 256)
RATE_DISTORTION_CODECS: tuple[str, ...] = ("uniform_scalar", "fsq", "lfq", "vq", "continuous")

# MSE proxy threshold. Program-factor tensors are normalized into [0, 1] by
# the tensorizer, so an MSE budget of 0.05 is a deliberately generous, small
# distortion tolerance -- a proxy for the issue's 0.10 meaning-v2/binder-F1
# margin, not a claim of numeric equivalence to that metric.
DISTORTION_THRESHOLD: float = 0.05

# Fixed per-coordinate FSQ level and VQ latent width, held constant across the
# sweep so num_latents/width remain the only varied axes for that family.
_FSQ_LEVEL_PER_COORD = 4
_VQ_LATENT_DIM = 8


@dataclass(frozen=True)
class RateDistortionCell:
    """One (codec, num_latents, width, seed) sweep cell."""

    codec: str
    num_latents: int
    width: int
    seed: int = 0
    train_steps: int = 400
    lr: float = 2e-2

    @property
    def cell_id(self) -> str:
        return f"{self.codec}_n{self.num_latents}_w{self.width}_s{self.seed}"

    @property
    def capacity(self) -> float:
        """Nominal discrete capacity (continuous latents report latent_dim)."""
        if self.codec == "uniform_scalar":
            return 2**self.num_latents
        if self.codec == "fsq":
            return _FSQ_LEVEL_PER_COORD**self.num_latents
        if self.codec == "lfq":
            return 2**self.num_latents
        if self.codec == "vq":
            return float(2**self.num_latents)
        if self.codec == "continuous":
            return float(self.num_latents)
        raise ValueError(f"unknown codec {self.codec!r}")


@dataclass(frozen=True)
class RateDistortionResult:
    """Measured result for one sweep cell."""

    cell_id: str
    codec: str
    num_latents: int
    width: int
    seed: int
    capacity: float
    nominal_bits: float
    serialized_bytes_per_example: float
    nominal_bytes_per_example: float
    mse_overall: float
    mse_per_factor: dict[str, float]
    max_factor_mse: float
    factors_over_threshold: tuple[str, ...]
    no_bypass_ok: bool
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "codec": self.codec,
            "num_latents": self.num_latents,
            "width": self.width,
            "seed": self.seed,
            "capacity": self.capacity,
            "nominal_bits": self.nominal_bits,
            "serialized_bytes_per_example": self.serialized_bytes_per_example,
            "nominal_bytes_per_example": self.nominal_bytes_per_example,
            "mse_overall": self.mse_overall,
            "mse_per_factor": self.mse_per_factor,
            "max_factor_mse": self.max_factor_mse,
            "factors_over_threshold": list(self.factors_over_threshold),
            "no_bypass_ok": self.no_bypass_ok,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class RetainedConfig:
    """Smallest passing (codec, num_latents, width) cell for one codec family."""

    codec: str
    num_latents: int
    width: int
    seed0_result: RateDistortionResult
    confirmation_seeds: tuple[RateDistortionResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "codec": self.codec,
            "num_latents": self.num_latents,
            "width": self.width,
            "seed0_result": self.seed0_result.to_dict(),
            "confirmation_seeds": [r.to_dict() for r in self.confirmation_seeds],
        }


@dataclass(frozen=True)
class RateDistortionSweepReport:
    """Full CAP2-06 staged rate-distortion frontier report."""

    run_id: str
    num_records: int
    threshold: float
    cells: tuple[RateDistortionResult, ...]
    retained: tuple[RetainedConfig, ...]
    version: str = "cap2-06-v1"
    timestamp: str = field(default_factory=lambda: _utc_now())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "version": self.version,
            "timestamp": self.timestamp,
            "num_records": self.num_records,
            "threshold": self.threshold,
            "cells": [c.to_dict() for c in self.cells],
            "retained": [r.to_dict() for r in self.retained],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _hash_run_id(parts: tuple[Any, ...]) -> str:
    import hashlib

    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def _decoder_input_dim(codec: LatentCodec) -> int:
    """Mirror LatentCodecModel._decoder_input_dim for the regression decoder."""
    spec = codec.spec
    if spec.name in ("learned_vq", "continuous"):
        return codec.config.latent_dim  # type: ignore[attr-defined]
    if spec.name == "binary_lfq":
        return codec.config.d  # type: ignore[attr-defined]
    return sum(spec.levels)


class FactorReconstructionModel(nn.Module):
    """Wrap a LatentCodec with a single linear decoder regressing the full
    concatenated program-factor vector back from the codec's declared
    decode_input output (never from the raw features directly).
    """

    def __init__(self, codec: LatentCodec, feature_dim: int) -> None:
        super().__init__()
        self.codec = codec
        self.feature_dim = feature_dim
        self.decoder = nn.Linear(_decoder_input_dim(codec), feature_dim)

    def forward(self, x: torch.Tensor, *, hard: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoding = self.codec.encode(x, hard=hard)
        decoder_input = self.codec.decode_input(encoding)
        return self.decoder(decoder_input), encoding.code_index


def train_factor_reconstruction(
    model: FactorReconstructionModel,
    features: torch.Tensor,
    *,
    steps: int = 400,
    lr: float = 2e-2,
) -> dict[str, float]:
    """Train the regression decoder to reconstruct the raw factor vector."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        encoding = model.codec.encode(features, hard=False)
        decoder_input = model.codec.decode_input(encoding)
        recon = model.decoder(decoder_input)
        loss = F.mse_loss(recon, features)
        for aux in encoding.aux_loss.values():
            loss = loss + aux
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return {"final_loss": losses[-1], "steps": steps, "lr": lr}


def evaluate_factor_reconstruction(
    model: FactorReconstructionModel,
    features: torch.Tensor,
    dims: dict[str, int],
) -> dict[str, Any]:
    """Hard-code evaluation: overall and per-factor-family reconstruction MSE."""
    model.eval()
    with torch.no_grad():
        recon, _ = model(features, hard=True)
        mse_overall = F.mse_loss(recon, features).item()
        mse_per_factor: dict[str, float] = {}
        offset = 0
        for name in PROGRAM_FACTOR_FAMILIES:
            width = dims[name]
            mse_per_factor[name] = F.mse_loss(
                recon[:, offset : offset + width], features[:, offset : offset + width]
            ).item()
            offset += width
        encodings = [model.codec.encode(features[i : i + 1], hard=True) for i in range(features.shape[0])]
        diag = model.codec.diagnostics(encodings)
    return {
        "mse_overall": mse_overall,
        "mse_per_factor": mse_per_factor,
        "occupied_codewords": diag.occupied_codes,
        "utilization": diag.utilization,
        "empirical_entropy_bits": diag.empirical_entropy_bits,
    }


def _build_codec(cell: RateDistortionCell, feature_dim: int) -> LatentCodec:
    width = cell.width
    n = cell.num_latents
    if cell.codec == "uniform_scalar":
        cfg = UniformScalarCodecConfig(
            num_states=1, K=2, d=n, hidden_dim=width, feature_dim=feature_dim, mode="semantic_trace"
        )
        return UniformScalarCodec(cfg)
    if cell.codec == "fsq":
        cfg = MixedRadixFSQConfig(
            num_states=1,
            levels=(_FSQ_LEVEL_PER_COORD,) * n,
            hidden_dim=width,
            feature_dim=feature_dim,
            mode="semantic_trace",
        )
        return MixedRadixFSQCodec(cfg)
    if cell.codec == "lfq":
        cfg = BinaryLFQConfig(
            num_states=1, d=n, hidden_dim=width, feature_dim=feature_dim, mode="semantic_trace"
        )
        return BinaryLFQCodec(cfg)
    if cell.codec == "vq":
        cfg = LearnedVQConfig(
            num_states=1,
            codebook_size=2**n,
            latent_dim=_VQ_LATENT_DIM,
            hidden_dim=width,
            feature_dim=feature_dim,
            mode="semantic_trace",
        )
        return LearnedVQCodec(cfg)
    if cell.codec == "continuous":
        cfg = ContinuousLatentConfig(
            num_states=1, latent_dim=n, hidden_dim=width, feature_dim=feature_dim, mode="semantic_trace"
        )
        return ContinuousLatentCodec(cfg)
    raise ValueError(f"unknown codec {cell.codec!r}")


def build_sweep(
    *,
    codecs: tuple[str, ...] = RATE_DISTORTION_CODECS,
    num_latents_values: tuple[int, ...] = RATE_DISTORTION_LATENT_COUNTS,
    widths: tuple[int, ...] = RATE_DISTORTION_WIDTHS,
    seed: int = 0,
    train_steps: int = 400,
) -> list[RateDistortionCell]:
    """Build the full staged (codec, num_latents, width) grid at one screening seed."""
    cells: list[RateDistortionCell] = []
    for codec in codecs:
        for n in num_latents_values:
            for width in widths:
                cells.append(
                    RateDistortionCell(codec, n, width, seed=seed, train_steps=train_steps)
                )
    return cells


def evaluate_cell(
    cell: RateDistortionCell,
    features: torch.Tensor,
    dims: dict[str, int],
) -> RateDistortionResult:
    """Train and evaluate one sweep cell."""
    start = time.monotonic()
    feature_dim = features.shape[-1]
    torch.manual_seed(cell.seed)
    codec = _build_codec(cell, feature_dim)
    model = FactorReconstructionModel(codec, feature_dim)
    train_factor_reconstruction(model, features, steps=cell.train_steps, lr=cell.lr)
    metrics = evaluate_factor_reconstruction(model, features, dims)
    max_factor_mse = max(metrics["mse_per_factor"].values())
    over_threshold = tuple(
        sorted(name for name, mse in metrics["mse_per_factor"].items() if mse > DISTORTION_THRESHOLD)
    )
    no_bypass_ok = audit_no_bypass(model, features)
    storage = codec.physical_storage((features.shape[0],))
    return RateDistortionResult(
        cell_id=cell.cell_id,
        codec=cell.codec,
        num_latents=cell.num_latents,
        width=cell.width,
        seed=cell.seed,
        capacity=cell.capacity,
        nominal_bits=codec.nominal_bits(),
        serialized_bytes_per_example=storage.bytes_per_example,
        nominal_bytes_per_example=storage.nominal_bytes,
        mse_overall=metrics["mse_overall"],
        mse_per_factor=metrics["mse_per_factor"],
        max_factor_mse=max_factor_mse,
        factors_over_threshold=over_threshold,
        no_bypass_ok=no_bypass_ok,
        elapsed_seconds=time.monotonic() - start,
    )


def select_retained_configs(
    cells: list[RateDistortionCell],
    results: dict[str, RateDistortionResult],
    features: torch.Tensor,
    dims: dict[str, int],
    *,
    confirmation_seeds: tuple[int, ...] = (1, 2),
) -> list[RetainedConfig]:
    """Per codec, pick the smallest-capacity screened cell that passes the
    threshold on every factor family, and re-run it at confirmation seeds.

    "Smallest" orders by (num_latents, width) ascending -- the same order the
    grid is declared in -- so ties prefer fewer latent positions over a wider
    encoder. A codec with no passing cell in the screened grid is omitted
    from the retained set entirely (an explicit failed-to-clear-gate
    disposition, not silently promoted).

    A cell must also pass ``no_bypass_ok`` to be eligible: a cell whose
    encoder collapsed to a constant/dead code can coincidentally reconstruct
    a low-variance factor family well (e.g. by learning the sample mean),
    which would otherwise look like a passing low-distortion configuration
    despite not actually using its input. Low distortion from a codec that
    doesn't functionally depend on its input is not a real rate-distortion
    result.
    """
    retained: list[RetainedConfig] = []
    by_codec: dict[str, list[RateDistortionCell]] = {}
    for cell in cells:
        by_codec.setdefault(cell.codec, []).append(cell)

    for codec, codec_cells in by_codec.items():
        ordered = sorted(codec_cells, key=lambda c: (c.num_latents, c.width))
        passing = [
            c
            for c in ordered
            if not results[c.cell_id].factors_over_threshold and results[c.cell_id].no_bypass_ok
        ]
        if not passing:
            continue
        winner = passing[0]
        confirm_results = []
        for seed in confirmation_seeds:
            confirm_cell = RateDistortionCell(
                codec, winner.num_latents, winner.width, seed=seed, train_steps=winner.train_steps
            )
            confirm_results.append(evaluate_cell(confirm_cell, features, dims))
        retained.append(
            RetainedConfig(
                codec=codec,
                num_latents=winner.num_latents,
                width=winner.width,
                seed0_result=results[winner.cell_id],
                confirmation_seeds=tuple(confirm_results),
            )
        )
    return retained


def run_sweep(
    *,
    fixture_count: int = 12,
    fixture_seed: int = 0,
    codecs: tuple[str, ...] = RATE_DISTORTION_CODECS,
    num_latents_values: tuple[int, ...] = RATE_DISTORTION_LATENT_COUNTS,
    widths: tuple[int, ...] = RATE_DISTORTION_WIDTHS,
    train_steps: int = 400,
    tensorizer_config: ProgramFactorTensorizerConfig | None = None,
) -> RateDistortionSweepReport:
    """Run the full staged CAP2-06 rate-distortion sweep on the SLM-144 fixture corpus."""
    from slm_training.harnesses.experiments.cap2_semantic_plan_bottleneck import load_fixture_plans

    cfg = tensorizer_config or ProgramFactorTensorizerConfig()
    dims = factor_dims(cfg)
    plans: list[SemanticPlanV1] = load_fixture_plans(count=fixture_count, seed=fixture_seed)
    features = torch.stack([concat_factor_tensors(tensorize_semantic_plan(plan, cfg)) for plan in plans])

    cells = build_sweep(
        codecs=codecs, num_latents_values=num_latents_values, widths=widths, train_steps=train_steps
    )
    results: dict[str, RateDistortionResult] = {}
    for cell in cells:
        results[cell.cell_id] = evaluate_cell(cell, features, dims)

    retained = select_retained_configs(cells, results, features, dims)

    run_id = _hash_run_id(("cap2-06", len(plans), tuple(c.cell_id for c in cells)))
    return RateDistortionSweepReport(
        run_id=run_id,
        num_records=len(plans),
        threshold=DISTORTION_THRESHOLD,
        cells=tuple(results[c.cell_id] for c in cells),
        retained=tuple(retained),
    )
