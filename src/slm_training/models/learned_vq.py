"""Learned vector-quantization (VQ) latent codec.

A small learned codebook is shared across the batch.  The encoder outputs a
continuous vector and the nearest codebook entry is selected.  Training uses the
straight-through estimator; evaluation is hard nearest-neighbor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from slm_training.models.latent_codec import (
    LatentCodec,
    LatentCodecSpec,
    LatentDiagnostics,
    LatentEncoding,
    StorageEstimate,
    _empirical_entropy_bits,
)


@dataclass(frozen=True)
class LearnedVQConfig:
    """Configuration for a learned VQ codec."""

    num_states: int
    codebook_size: int
    latent_dim: int
    hidden_dim: int = 64
    feature_dim: int | None = None
    mode: str = "oracle_state"  # "oracle_state" | "semantic_trace"
    commitment_cost: float = 0.25


class LearnedVQCodec(nn.Module, LatentCodec):
    """Learned VQ latent codec with explicit commitment loss and dead-code tracking."""

    def __init__(self, config: LearnedVQConfig) -> None:
        super().__init__()
        self.config = config
        if config.codebook_size < 2:
            raise ValueError("codebook_size must be >= 2")
        if config.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        import math

        self.spec = LatentCodecSpec(
            name="learned_vq",
            levels=(config.codebook_size,),
            nominal_bits=math.log2(config.codebook_size),
            hard_only=True,
        )
        self.codebook = nn.Parameter(torch.randn(config.codebook_size, config.latent_dim))

        if config.mode == "oracle_state":
            self.encoder: nn.Module | None = None
            self.state_latents = nn.Parameter(torch.randn(config.num_states, config.latent_dim))
            self.code_logits: nn.Linear | None = None
        elif config.mode == "semantic_trace":
            if config.feature_dim is None:
                raise ValueError("semantic_trace mode requires feature_dim")
            self.encoder = nn.Sequential(
                nn.Linear(config.feature_dim, config.hidden_dim),
                nn.ReLU(),
            )
            self.state_latents = None
            self.code_logits = nn.Linear(config.hidden_dim, config.latent_dim)
        else:
            raise ValueError(f"unknown mode {config.mode!r}")

    def _latents(self, x: torch.Tensor) -> torch.Tensor:
        """Return continuous latent vectors [batch, latent_dim]."""
        if self.state_latents is not None:
            return self.state_latents[x]
        assert self.encoder is not None and self.code_logits is not None
        return self.code_logits(self.encoder(x))

    def _nearest_code(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return hard code indices [batch] and quantized vectors [batch, latent_dim]."""
        distances = torch.cdist(latents, self.codebook, p=2)
        indices = distances.argmin(dim=-1)
        quantized = self.codebook[indices]
        return indices, quantized

    def encode(self, hidden: torch.Tensor, *, hard: bool) -> LatentEncoding:
        latents = self._latents(hidden)
        indices, quantized = self._nearest_code(latents)
        if self.training and not hard:
            # STE: forward uses quantized, backward flows through latents.
            ste_latents = latents + (quantized - latents).detach()
            relaxed = ste_latents
        else:
            relaxed = None
        aux_loss: dict[str, torch.Tensor] = {}
        if self.training and self.config.commitment_cost > 0.0:
            aux_loss["commitment"] = self.config.commitment_cost * F.mse_loss(latents, quantized.detach())
        return LatentEncoding(
            hard=indices.unsqueeze(-1),
            code_index=indices,
            relaxed=relaxed,
            aux_loss=aux_loss,
            metadata={"latents": latents, "quantized": quantized},
        )

    def decode_input(self, encoding: LatentEncoding) -> torch.Tensor:
        # Training uses the STE latent; evaluation uses the codebook lookup.
        if encoding.relaxed is not None:
            return encoding.relaxed
        indices = encoding.code_index
        assert indices is not None
        return self.codebook[indices]

    def nominal_bits(self) -> float:
        return self.spec.nominal_bits

    def physical_storage(self, batch_shape: tuple[int, ...]) -> StorageEstimate:
        slots = 1
        nominal_bytes = self.nominal_bits() / 8.0
        # Index only: ceil(log2(codebook_size)) bits per example.
        import math

        bytes_per_example = math.ceil(self.nominal_bits()) / 8.0
        return StorageEstimate(
            bytes_per_example=bytes_per_example,
            nominal_bytes=nominal_bytes,
            slots=slots,
        )

    def diagnostics(self, encodings: list[LatentEncoding]) -> LatentDiagnostics:
        indices = torch.cat([e.code_index for e in encodings if e.code_index is not None])
        occupied = int(torch.unique(indices).numel())
        capacity = self.config.codebook_size
        dead = capacity - occupied
        return LatentDiagnostics(
            occupied_codes=occupied,
            utilization=occupied / max(1, capacity),
            empirical_entropy_bits=_empirical_entropy_bits(indices),
            dead_coordinates=dead,
            notes=(f"learned VQ codebook_size={self.config.codebook_size} latent_dim={self.config.latent_dim}",),
        )
