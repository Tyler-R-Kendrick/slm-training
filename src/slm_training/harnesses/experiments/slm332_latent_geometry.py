"""SLM-332's codec-only hard-valid latent-geometry training harness.

This harness deliberately stops at the existing program-factor codec boundary.
It trains a matched reconstruction-only control and a combined reconstruction,
hard-valid contrast, and local-noise arm over admitted AP-009 binding pairs.
It cannot turn a latent vector back into an OpenUI program, so it records the
AP-030 semantic gate as unavailable and never returns a promotable result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.models.program_latent_objectives import (
    ProgramLatentObjective,
    combined_program_latent_objective,
)
from slm_training.models.semantic_plan_factors import (
    ProgramFactorTensorizerConfig,
    concat_factor_tensors,
    tensorize_semantic_plan,
)

from .cap2_rate_distortion_sweep import FactorReconstructionModel

DEFAULT_AP009_PAIRS = Path(
    "src/slm_training/resources/data/eval/openui_hard_valid_v1/pairs.jsonl"
)
_BINDING_TRANSFORMS = frozenset({"binding_swap_symbol", "binding_swap_symbol_reverse"})

__all__ = [
    "DEFAULT_AP009_PAIRS",
    "BindingContrastCorpus",
    "GeometryTrainingConfig",
    "GeometryMeasurement",
    "load_admitted_binding_contrasts",
    "train_geometry_arm",
    "measure_geometry_arm",
]


@dataclass(frozen=True)
class BindingContrastCorpus:
    """Canonical factor tensors for admitted AP-009 binding/reference pairs."""

    positive: torch.Tensor
    negative: torch.Tensor
    pair_ids: tuple[str, ...]
    transform_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.pair_ids:
            raise ValueError("SLM-332 requires at least one admitted binding contrast")
        if self.positive.shape != self.negative.shape:
            raise ValueError("positive and negative contrast tensors must have the same shape")
        if self.positive.shape[0] != len(self.pair_ids):
            raise ValueError("pair ids must align with contrast tensors")
        if not set(self.transform_ids).issubset(_BINDING_TRANSFORMS):
            raise ValueError("contrast corpus contains a non-binding AP-009 transform")


@dataclass(frozen=True)
class GeometryTrainingConfig:
    steps: int = 400
    lr: float = 2e-2
    contrast_margin: float = 1.0
    contrast_weight: float = 1.0
    noise_std: float = 0.05
    noise_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.lr <= 0 or self.noise_std < 0:
            raise ValueError("lr must be positive and noise_std non-negative")


@dataclass(frozen=True)
class GeometryMeasurement:
    reconstruction_mse: float
    contrast_margin_loss: float
    contrast_distance: float
    within_noise_mse: float
    effective_rank: float
    ap030_oracle_semantic_gate: str = "unavailable_no_verified_latent_to_program_decoder"
    interpolation_traversal: str = "unavailable_no_verified_latent_to_program_decoder"
    promotion_eligible: bool = False


def load_admitted_binding_contrasts(
    path: Path = DEFAULT_AP009_PAIRS,
    *,
    tensorizer_config: ProgramFactorTensorizerConfig | None = None,
) -> BindingContrastCorpus:
    """Load only admitted AP-009 binding/reference contrasts.

    Source plans, rather than generated text or metadata shortcuts, are
    tensorized with the shared semantic-plan factorizer. A missing binding
    transform is an error: silently falling back to generic semantic pairs
    would defeat the issue's binder/reference coverage requirement.
    """
    cfg = tensorizer_config or ProgramFactorTensorizerConfig()
    positives: list[torch.Tensor] = []
    negatives: list[torch.Tensor] = []
    pair_ids: list[str] = []
    transforms: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        transform = row["transform_id"]
        if not row["admitted"] or transform not in _BINDING_TRANSFORMS:
            continue
        positive = SemanticPlanV1.from_dict(row["positive"]["source_plan"])
        negative = SemanticPlanV1.from_dict(row["negative"]["source_plan"])
        positives.append(concat_factor_tensors(tensorize_semantic_plan(positive, cfg)))
        negatives.append(concat_factor_tensors(tensorize_semantic_plan(negative, cfg)))
        pair_ids.append(row["pair_id"])
        transforms.append(transform)
    if not positives:
        raise ValueError(f"no admitted AP-009 binding contrasts in {path}")
    return BindingContrastCorpus(
        positive=torch.stack(positives),
        negative=torch.stack(negatives),
        pair_ids=tuple(pair_ids),
        transform_ids=tuple(transforms),
    )


def _latents(model: FactorReconstructionModel, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    encoding = model.codec.encode(features, hard=False)
    aux = sum(encoding.aux_loss.values(), start=features.new_zeros(()))
    return model.codec.decode_input(encoding), aux


def train_geometry_arm(
    model: FactorReconstructionModel,
    corpus: BindingContrastCorpus,
    config: GeometryTrainingConfig,
) -> list[ProgramLatentObjective]:
    """Train one preregistered objective arm without any non-codec bypass."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history: list[ProgramLatentObjective] = []
    model.train()
    for _ in range(config.steps):
        optimizer.zero_grad()
        positive_latents, positive_aux = _latents(model, corpus.positive)
        negative_latents, negative_aux = _latents(model, corpus.negative)
        reconstruction = F.mse_loss(model.decoder(positive_latents), corpus.positive)
        noisy = positive_latents + torch.randn_like(positive_latents) * config.noise_std
        objective = combined_program_latent_objective(
            reconstruction,
            positive_latents,
            negative_latents,
            contrast_margin=config.contrast_margin,
            contrast_weight=config.contrast_weight,
            noisy_latents=noisy,
            noise_weight=config.noise_weight,
        )
        (objective.total + positive_aux + negative_aux).backward()
        optimizer.step()
        history.append(objective)
    return history


def _effective_rank(latents: torch.Tensor) -> float:
    centered = latents - latents.mean(dim=0, keepdim=True)
    energy = torch.linalg.svdvals(centered).square()
    total = energy.sum()
    if total <= 0:
        return 0.0
    probabilities = energy / total
    entropy = -(probabilities * probabilities.clamp_min(torch.finfo(energy.dtype).tiny).log()).sum()
    return float(entropy.exp().item())


def measure_geometry_arm(
    model: FactorReconstructionModel,
    corpus: BindingContrastCorpus,
    config: GeometryTrainingConfig,
) -> GeometryMeasurement:
    """Measure geometry proxies and fail closed on semantic/promotion authority."""
    model.eval()
    with torch.no_grad():
        positive_latents, _ = _latents(model, corpus.positive)
        negative_latents, _ = _latents(model, corpus.negative)
        reconstruction = F.mse_loss(model.decoder(positive_latents), corpus.positive)
        noisy = positive_latents + torch.randn_like(positive_latents) * config.noise_std
        objective = combined_program_latent_objective(
            reconstruction,
            positive_latents,
            negative_latents,
            contrast_margin=config.contrast_margin,
            contrast_weight=config.contrast_weight,
            noisy_latents=noisy,
            noise_weight=config.noise_weight,
        )
        return GeometryMeasurement(
            reconstruction_mse=float(reconstruction.item()),
            contrast_margin_loss=float(objective.contrast.item()),
            contrast_distance=float(torch.linalg.vector_norm(positive_latents - negative_latents, dim=-1).mean().item()),
            within_noise_mse=float(F.mse_loss(noisy, positive_latents).item()),
            effective_rank=_effective_rank(positive_latents),
        )
