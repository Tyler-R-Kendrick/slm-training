"""Continuous rate-controlled latent codec.

This is a control arm for the CAP2 matrix: a continuous bottleneck with an
explicit Gaussian noise or rate/KL penalty.  It is not a discrete code, so the
no-bypass audit checks that the decoder input equals the declared latent vector
and that a noise/rate policy is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from slm_training.models.latent_codec import (
    LatentCodec,
    LatentCodecSpec,
    LatentDiagnostics,
    LatentEncoding,
    StorageEstimate,
)


@dataclass(frozen=True)
class ContinuousLatentConfig:
    """Configuration for a continuous latent codec.

    Attributes:
        num_states: number of distinct inputs (used by oracle_state mode).
        latent_dim: dimension of the continuous latent vector.
        hidden_dim: MLP hidden dimension for semantic_trace mode.
        feature_dim: input feature dimension for semantic_trace mode.
        mode: "oracle_state" or "semantic_trace".
        noise_std: Gaussian noise stddev during training (0.0 for deterministic eval).
        rate_penalty: weight on the squared latent norm rate penalty.
    """

    num_states: int
    latent_dim: int
    hidden_dim: int = 64
    feature_dim: int | None = None
    mode: str = "oracle_state"
    noise_std: float = 0.0
    rate_penalty: float = 0.0


class ContinuousLatentCodec(nn.Module, LatentCodec):
    """Continuous latent with explicit noise/rate policy."""

    def __init__(self, config: ContinuousLatentConfig) -> None:
        super().__init__()
        self.config = config
        if config.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        # Continuous latents are not discrete; nominal_bits is the latent
        # dimension under a unit-variance Gaussian coding assumption, but the
        # spec records this as a continuous dimension count, not equal information.
        self.spec = LatentCodecSpec(
            name="continuous",
            levels=(config.latent_dim,),
            nominal_bits=float(config.latent_dim),
            hard_only=False,
        )

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

    def encode(self, hidden: torch.Tensor, *, hard: bool) -> LatentEncoding:
        latents = self._latents(hidden)
        if self.training and not hard and self.config.noise_std > 0.0:
            noisy = latents + torch.randn_like(latents) * self.config.noise_std
            relaxed = noisy
        else:
            relaxed = None
        # For continuous latents the "hard" symbols are just the latent values
        # themselves; there is no discrete code index.
        code_index = torch.zeros(latents.shape[0], dtype=torch.long, device=latents.device)
        aux_loss: dict[str, torch.Tensor] = {}
        if self.training and self.config.rate_penalty > 0.0:
            aux_loss["rate_penalty"] = self.config.rate_penalty * (latents ** 2).mean()
        return LatentEncoding(
            hard=latents,
            code_index=code_index,
            relaxed=relaxed,
            aux_loss=aux_loss,
            metadata={
                "noise_std": self.config.noise_std,
                "rate_penalty": self.config.rate_penalty,
            },
        )

    def decode_input(self, encoding: LatentEncoding) -> torch.Tensor:
        if encoding.relaxed is not None:
            return encoding.relaxed
        return encoding.hard

    def nominal_bits(self) -> float:
        return self.spec.nominal_bits

    def physical_storage(self, batch_shape: tuple[int, ...]) -> StorageEstimate:
        # Report actual float32 bytes; nominal_bytes is the dimension count only.
        bytes_per_example = self.config.latent_dim * 4
        return StorageEstimate(
            bytes_per_example=bytes_per_example,
            nominal_bytes=self.nominal_bits() / 8.0,
            slots=self.config.latent_dim,
        )

    def diagnostics(self, encodings: list[LatentEncoding]) -> LatentDiagnostics:
        # Occupancy is meaningless for continuous latents; report effective support.
        latents = torch.cat([e.hard for e in encodings])
        # Effective rank via SVD energy (cheap diagnostic).
        if latents.shape[0] > 1 and latents.shape[1] > 0:
            centered = latents - latents.mean(dim=0, keepdim=True)
            _, s, _ = torch.svd(centered)
            total = (s ** 2).sum()
            effective_rank = (
                ((s ** 2) / total).sum().item() if total > 0 else float("nan")
            )
        else:
            effective_rank = float("nan")
        return LatentDiagnostics(
            occupied_codes=latents.shape[0],
            utilization=1.0,
            empirical_entropy_bits=0.0,
            dead_coordinates=0,
            notes=(
                f"continuous latent_dim={self.config.latent_dim} "
                f"noise_std={self.config.noise_std} rate_penalty={self.config.rate_penalty} "
                f"effective_rank={effective_rank:.3f}",
            ),
        )
