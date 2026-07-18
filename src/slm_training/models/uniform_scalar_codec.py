"""Uniform K-ary scalar latent codec (CAP2 baseline).

This is the uniform-radix counterpart to :class:`KaryBottleneck` expressed as a
:class:`LatentCodec`.  It is intentionally simple: ``d`` categorical coordinates
each taking ``K`` values, with a straight-through estimator for training.
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
    _index_from_mixed_radix,
    _one_hot_mixed_radix,
)


@dataclass(frozen=True)
class UniformScalarCodecConfig:
    """Configuration for a uniform K-ary scalar codec."""

    num_states: int
    K: int
    d: int
    hidden_dim: int = 64
    feature_dim: int | None = None
    mode: str = "oracle_state"  # "oracle_state" | "semantic_trace"


class UniformScalarCodec(nn.Module, LatentCodec):
    """Uniform K-ary scalar latent codec.

    The decoder receives only the hard one-hot code during deterministic
    evaluation.  Training may use a soft straight-through relaxation.
    """

    def __init__(self, config: UniformScalarCodecConfig) -> None:
        super().__init__()
        self.config = config
        self.spec = LatentCodecSpec(
            name="uniform_scalar",
            levels=(config.K,) * config.d,
            nominal_bits=float(config.d * math_log2(config.K)),
            hard_only=True,
        )

        if config.mode == "oracle_state":
            self.encoder: nn.Module | None = None
            self.state_code_logits = nn.Parameter(
                torch.randn(config.num_states, config.d, config.K)
            )
            self.code_logits: nn.Linear | None = None
        elif config.mode == "semantic_trace":
            if config.feature_dim is None:
                raise ValueError("semantic_trace mode requires feature_dim")
            self.encoder = nn.Sequential(
                nn.Linear(config.feature_dim, config.hidden_dim),
                nn.ReLU(),
            )
            self.state_code_logits = None
            self.code_logits = nn.Linear(config.hidden_dim, config.d * config.K)
        else:
            raise ValueError(f"unknown mode {config.mode!r}")

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-coordinate logits [batch, d, K]."""
        if self.state_code_logits is not None:
            return self.state_code_logits[x]
        assert self.encoder is not None and self.code_logits is not None
        h = self.encoder(x)
        return self.code_logits(h).view(-1, self.config.d, self.config.K)

    def encode(self, hidden: torch.Tensor, *, hard: bool) -> LatentEncoding:
        logits = self._logits(hidden)
        hard_symbols = logits.argmax(dim=-1)
        one_hot = F.one_hot(hard_symbols, self.config.K).float()
        if self.training and not hard:
            probs = F.softmax(logits, dim=-1)
            soft_one_hot = one_hot + probs - probs.detach()
        else:
            soft_one_hot = one_hot
        code_index = _index_from_mixed_radix(
            hard_symbols, (self.config.K,) * self.config.d
        )
        return LatentEncoding(
            hard=hard_symbols,
            code_index=code_index,
            relaxed=soft_one_hot.view(-1, self.config.d * self.config.K)
            if self.training and not hard
            else None,
            metadata={"logits": logits, "one_hot": one_hot},
        )

    def decode_input(self, encoding: LatentEncoding) -> torch.Tensor:
        # Training may use the relaxed soft one-hot; evaluation uses hard symbols only.
        if encoding.relaxed is not None:
            return encoding.relaxed
        symbols = encoding.hard
        return _one_hot_mixed_radix(symbols, self.spec.levels)

    def nominal_bits(self) -> float:
        return self.spec.nominal_bits

    def physical_storage(self, batch_shape: tuple[int, ...]) -> StorageEstimate:
        slots = self.config.d
        nominal_bytes = self.nominal_bits() / 8.0
        # Pack each coordinate into ceil(log2(K)) bits; here we report the ideal.
        bytes_per_example = nominal_bytes
        return StorageEstimate(
            bytes_per_example=bytes_per_example,
            nominal_bytes=nominal_bytes,
            slots=slots,
        )

    def diagnostics(self, encodings: list[LatentEncoding]) -> LatentDiagnostics:
        indices = torch.cat([e.code_index for e in encodings if e.code_index is not None])
        occupied = int(torch.unique(indices).numel())
        capacity = self.config.K ** self.config.d
        dead = 0
        if encodings:
            symbols = torch.cat([e.hard for e in encodings])
            for coord in range(symbols.shape[-1]):
                if torch.unique(symbols[:, coord]).numel() <= 1:
                    dead += 1
        return LatentDiagnostics(
            occupied_codes=occupied,
            utilization=occupied / max(1, capacity),
            empirical_entropy_bits=_empirical_entropy_bits(indices),
            dead_coordinates=dead,
            notes=(f"uniform scalar K={self.config.K} d={self.config.d}",),
        )


def math_log2(x: int) -> float:
    import math

    return math.log2(x)
