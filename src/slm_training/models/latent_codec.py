"""Common latent-codec interface for CAP2 discrete/continuous bottleneck experiments.

This module defines the shared contract used by the strict bottleneck harness so
that uniform scalar, mixed-radix FSQ, binary LFQ, learned VQ, relaxed discrete,
and continuous rate-controlled latents can be compared with the same evaluation
path.  It intentionally stays small: the harness only needs integer codes, a
decoder-from-code path, and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn


@dataclass(frozen=True)
class LatentCodecSpec:
    """Static description of a latent codec.

    Attributes:
        name: codec family id, e.g. "uniform_scalar", "mixed_radix_fsq",
            "binary_lfq", "learned_vq", "relaxed_discrete", "continuous".
        levels: per-coordinate alphabet sizes.  For a uniform K-ary code this is
            ``(K,) * d``; for mixed-radix FSQ it is the level vector ``L``.
        nominal_bits: theoretical capacity ``sum(log2(level_i))``.
        hard_only: if True the codec never produces relaxed values (discrete
            families); continuous families set False.
    """

    name: str
    levels: tuple[int, ...]
    nominal_bits: float
    hard_only: bool = True


@dataclass
class LatentEncoding:
    """Output of a codec ``encode`` call.

    Attributes:
        hard: integer code symbols [batch, d] or [batch].  This is the only
            tensor that may reach the decoder during deterministic evaluation.
        code_index: flattened integer code index [batch], or None if the codec
            does not define a flat ordering.
        relaxed: optional relaxed values used only during training (e.g. soft
            probabilities or continuous latents before rounding).
        aux_loss: quantization / commitment / entropy regularization loss terms
            that should be added during training.
        metadata: codec-specific metadata used by diagnostics/audit.
    """

    hard: torch.Tensor
    code_index: torch.Tensor | None = None
    relaxed: torch.Tensor | None = None
    aux_loss: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StorageEstimate:
    """Physical latent storage estimate.

    Attributes:
        bytes_per_example: best-case packed bytes for one example.
        nominal_bytes: bytes implied by nominal_bits (information-theoretic).
        slots: number of discrete slots occupied per example.
    """

    bytes_per_example: float
    nominal_bytes: float
    slots: int


@dataclass
class LatentDiagnostics:
    """Diagnostics collected over a batch of encodings.

    Attributes:
        occupied_codes: number of distinct flat code indices observed.
        utilization: occupied_codes / total_flat_capacity.
        empirical_entropy_bits: Shannon entropy of the observed code index
            distribution.
        dead_coordinates: coordinates whose empirical marginal entropy is zero.
        notes: human-readable observations.
    """

    occupied_codes: int
    utilization: float
    empirical_entropy_bits: float
    dead_coordinates: int
    notes: tuple[str, ...] = ()


@runtime_checkable
class LatentCodec(Protocol):
    """Shared interface for all latent codecs in the CAP2 matrix.

    Concrete implementations must also inherit from ``torch.nn.Module`` but the
    protocol itself cannot inherit from it (Python protocol rules).
    """

    spec: LatentCodecSpec

    def encode(self, hidden: torch.Tensor, *, hard: bool) -> LatentEncoding:
        """Map hidden features to a latent encoding."""
        ...

    def decode_input(self, encoding: LatentEncoding) -> torch.Tensor:
        """Prepare the decoder input from a latent encoding.

        During deterministic evaluation this must use only ``encoding.hard``.
        """
        ...

    def nominal_bits(self) -> float:
        """Return the theoretical discrete capacity in bits."""
        ...

    def physical_storage(self, batch_shape: tuple[int, ...]) -> StorageEstimate:
        """Estimate physical bytes for the given batch shape."""
        ...

    def diagnostics(self, encodings: list[LatentEncoding]) -> LatentDiagnostics:
        """Compute diagnostics over a list of encodings."""
        ...


def _one_hot_mixed_radix(symbols: torch.Tensor, levels: tuple[int, ...]) -> torch.Tensor:
    """One-hot encode a mixed-radix symbol tensor [batch, d] with per-coordinate levels.

    Returns a flat concatenated one-hot tensor [batch, sum(levels)].
    """
    parts: list[torch.Tensor] = []
    for coord, level in enumerate(levels):
        parts.append(
            nn.functional.one_hot(symbols[:, coord], level).to(dtype=torch.float32)
        )
    return torch.cat(parts, dim=-1)


def _index_from_mixed_radix(symbols: torch.Tensor, levels: tuple[int, ...]) -> torch.Tensor:
    """Convert mixed-radix symbols [batch, d] to a flat integer index [batch]."""
    batch_size = symbols.shape[0]
    index = torch.zeros(batch_size, dtype=torch.long, device=symbols.device)
    for coord, level in enumerate(levels):
        index = index * level + symbols[:, coord]
    return index


def _symbols_from_index(index: torch.Tensor, levels: tuple[int, ...]) -> torch.Tensor:
    """Convert flat integer index [batch] to mixed-radix symbols [batch, d]."""
    symbols: list[torch.Tensor] = []
    rem = index.clone()
    for level in reversed(levels):
        symbols.append(rem % level)
        rem = rem // level
    return torch.stack(list(reversed(symbols)), dim=-1)


def _empirical_entropy_bits(indices: torch.Tensor) -> float:
    """Shannon entropy in bits of an integer index tensor [batch]."""
    if indices.numel() == 0:
        return 0.0
    unique, counts = torch.unique(indices, return_counts=True)
    probs = counts.float() / counts.sum()
    entropy = -(probs * torch.log2(probs)).sum()
    return float(entropy.item()) if not torch.isnan(entropy) else 0.0
