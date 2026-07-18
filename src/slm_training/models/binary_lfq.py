"""Binary lookup-free quantization (LFQ) latent codec.

LFQ here means a binary sign code: each latent coordinate is rounded to +1/-1
(or 0/1) with no learned codebook lookup.  The decoder sees only the hard binary
vector.  Entropy regularization can be added during training but is reported,
not hidden.
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
    _empirical_entropy_bits,
)


@dataclass(frozen=True)
class BinaryLFQConfig:
    """Configuration for a binary LFQ codec."""

    num_states: int
    d: int
    hidden_dim: int = 64
    feature_dim: int | None = None
    mode: str = "oracle_state"  # "oracle_state" | "semantic_trace"
    entropy_regularization: float = 0.0


class BinaryLFQCodec(nn.Module, LatentCodec):
    """Binary lookup-free quantization latent codec.

    Each coordinate is a hard sign {-1, +1}.  Training uses a straight-through
    sign estimator.  No codebook is required, hence "lookup-free".
    """

    def __init__(self, config: BinaryLFQConfig) -> None:
        super().__init__()
        self.config = config
        if config.d <= 0:
            raise ValueError("d must be positive")
        self.spec = LatentCodecSpec(
            name="binary_lfq",
            levels=(2,) * config.d,
            nominal_bits=float(config.d),
            hard_only=True,
        )

        if config.mode == "oracle_state":
            self.encoder: nn.Module | None = None
            self.state_latents = nn.Parameter(torch.randn(config.num_states, config.d))
            self.code_logits: nn.Linear | None = None
        elif config.mode == "semantic_trace":
            if config.feature_dim is None:
                raise ValueError("semantic_trace mode requires feature_dim")
            self.encoder = nn.Sequential(
                nn.Linear(config.feature_dim, config.hidden_dim),
                nn.ReLU(),
            )
            self.state_latents = None
            self.code_logits = nn.Linear(config.hidden_dim, config.d)
        else:
            raise ValueError(f"unknown mode {config.mode!r}")

    def _latents(self, x: torch.Tensor) -> torch.Tensor:
        """Return continuous pre-sign latents [batch, d]."""
        if self.state_latents is not None:
            return self.state_latents[x]
        assert self.encoder is not None and self.code_logits is not None
        return self.code_logits(self.encoder(x))

    def encode(self, hidden: torch.Tensor, *, hard: bool) -> LatentEncoding:
        latents = self._latents(hidden)
        hard_bits = (latents >= 0).long()
        hard_signed = hard_bits * 2 - 1
        if self.training and not hard:
            # STE through sign: forward uses hard sign, backward sees tanh slope.
            soft_signed = torch.tanh(latents)
            relaxed = soft_signed + hard_signed.detach() - soft_signed.detach()
        else:
            relaxed = None
        code_index = torch.zeros(hard_bits.shape[0], dtype=torch.long, device=hard_bits.device)
        for i in range(self.config.d):
            code_index = code_index * 2 + hard_bits[:, i]
        aux_loss: dict[str, torch.Tensor] = {}
        if self.training and self.config.entropy_regularization > 0.0:
            # Simple per-coordinate entropy bonus (maximize entropy -> push bits toward 0.5).
            probs = torch.sigmoid(latents)
            per_coord_entropy = -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
            aux_loss["entropy_regularization"] = -self.config.entropy_regularization * per_coord_entropy.mean()
        return LatentEncoding(
            hard=hard_bits,
            code_index=code_index,
            relaxed=relaxed,
            aux_loss=aux_loss,
            metadata={"latents": latents, "signed": hard_signed},
        )

    def decode_input(self, encoding: LatentEncoding) -> torch.Tensor:
        # Training may use the relaxed soft sign; evaluation uses hard {-1,+1}.
        if encoding.relaxed is not None:
            return encoding.relaxed
        return encoding.hard * 2.0 - 1.0

    def nominal_bits(self) -> float:
        return self.spec.nominal_bits

    def physical_storage(self, batch_shape: tuple[int, ...]) -> StorageEstimate:
        slots = self.config.d
        nominal_bytes = self.nominal_bits() / 8.0
        bytes_per_example = self.config.d / 8.0  # one bit per coordinate
        return StorageEstimate(
            bytes_per_example=bytes_per_example,
            nominal_bytes=nominal_bytes,
            slots=slots,
        )

    def diagnostics(self, encodings: list[LatentEncoding]) -> LatentDiagnostics:
        indices = torch.cat([e.code_index for e in encodings if e.code_index is not None])
        occupied = int(torch.unique(indices).numel())
        capacity = 2 ** self.config.d
        dead = 0
        if encodings:
            bits = torch.cat([e.hard for e in encodings])
            for coord in range(bits.shape[-1]):
                if torch.unique(bits[:, coord]).numel() <= 1:
                    dead += 1
        return LatentDiagnostics(
            occupied_codes=occupied,
            utilization=occupied / max(1, capacity),
            empirical_entropy_bits=_empirical_entropy_bits(indices),
            dead_coordinates=dead,
            notes=(f"binary LFQ d={self.config.d}",),
        )
