"""Mixed-radix finite-scalar-quantization (FSQ) latent codec.

Each coordinate has its own number of levels ``L_i``.  The total nominal capacity
is ``prod(L_i)``.  This codec is deterministic at evaluation and uses a
straight-through estimator during training.
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
class MixedRadixFSQConfig:
    """Configuration for a mixed-radix FSQ codec."""

    num_states: int
    levels: tuple[int, ...]
    hidden_dim: int = 64
    feature_dim: int | None = None
    mode: str = "oracle_state"  # "oracle_state" | "semantic_trace"


class MixedRadixFSQCodec(nn.Module, LatentCodec):
    """Mixed-radix FSQ latent codec.

    The encoder produces one scalar per coordinate; each coordinate is rounded
    to an integer in ``[0, L_i)``.  The decoder sees only the hard integer code.
    """

    def __init__(self, config: MixedRadixFSQConfig) -> None:
        super().__init__()
        self.config = config
        if any(level < 2 for level in config.levels):
            raise ValueError(f"all levels must be >= 2, got {config.levels}")
        import math

        nominal_bits = sum(math.log2(level) for level in config.levels)
        self.spec = LatentCodecSpec(
            name="mixed_radix_fsq",
            levels=config.levels,
            nominal_bits=nominal_bits,
            hard_only=True,
        )

        if config.mode == "oracle_state":
            self.encoder: nn.Module | None = None
            # Per-state, per-coordinate logits over each level.
            self.state_code_logits = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(config.num_states, level))
                    for level in config.levels
                ]
            )
            self.code_logits: nn.Module | None = None
        elif config.mode == "semantic_trace":
            if config.feature_dim is None:
                raise ValueError("semantic_trace mode requires feature_dim")
            self.encoder = nn.Sequential(
                nn.Linear(config.feature_dim, config.hidden_dim),
                nn.ReLU(),
            )
            self.state_code_logits = None
            self.code_logits = nn.Linear(config.hidden_dim, sum(config.levels))
        else:
            raise ValueError(f"unknown mode {config.mode!r}")

    def _logits(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return per-coordinate logits, one tensor [batch, L_i] per coordinate."""
        if self.state_code_logits is not None:
            return [logits[x] for logits in self.state_code_logits]
        assert self.encoder is not None and self.code_logits is not None
        h = self.encoder(x)
        flat = self.code_logits(h)
        parts = torch.split(flat, self.config.levels, dim=-1)
        return list(parts)

    def encode(self, hidden: torch.Tensor, *, hard: bool) -> LatentEncoding:
        logits_per_coord = self._logits(hidden)
        hard_symbols = torch.stack(
            [logits.argmax(dim=-1) for logits in logits_per_coord], dim=-1
        )
        one_hots = [
            F.one_hot(hard_symbols[:, i], level).float()
            for i, level in enumerate(self.config.levels)
        ]
        if self.training and not hard:
            soft_one_hots = []
            for i, logits in enumerate(logits_per_coord):
                probs = F.softmax(logits, dim=-1)
                soft_one_hots.append(one_hots[i] + probs - probs.detach())
            relaxed = torch.cat(soft_one_hots, dim=-1)
        else:
            relaxed = None
        code_index = _index_from_mixed_radix(hard_symbols, self.config.levels)
        return LatentEncoding(
            hard=hard_symbols,
            code_index=code_index,
            relaxed=relaxed,
            metadata={"logits": logits_per_coord, "one_hots": one_hots},
        )

    def decode_input(self, encoding: LatentEncoding) -> torch.Tensor:
        if encoding.relaxed is not None:
            return encoding.relaxed
        return _one_hot_mixed_radix(encoding.hard, self.spec.levels)

    def nominal_bits(self) -> float:
        return self.spec.nominal_bits

    def physical_storage(self, batch_shape: tuple[int, ...]) -> StorageEstimate:
        import math

        slots = len(self.config.levels)
        nominal_bytes = self.nominal_bits() / 8.0
        # Ideal packing: ceil(log2(level_i)) bits per coordinate.
        bits_per_example = sum(math.ceil(math.log2(level)) for level in self.config.levels)
        return StorageEstimate(
            bytes_per_example=bits_per_example / 8.0,
            nominal_bytes=nominal_bytes,
            slots=slots,
        )

    def diagnostics(self, encodings: list[LatentEncoding]) -> LatentDiagnostics:
        indices = torch.cat([e.code_index for e in encodings if e.code_index is not None])
        occupied = int(torch.unique(indices).numel())
        capacity = 1
        for level in self.config.levels:
            capacity *= level
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
            notes=(f"mixed-radix FSQ levels={self.config.levels}",),
        )


def suggest_mixed_radix_levels(
    target_bits: float,
    *,
    max_levels: int = 6,
    allowed_levels: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8),
    source_hash: str = "",
) -> list[tuple[int, ...]]:
    """Return candidate mixed-radix level vectors meeting ``target_bits``.

    The allocator emits alternatives sorted by nominal-bit excess rather than a
    single "optimal" vector, because the best choice depends on trainability,
    hardware, and robustness assumptions that are fixed only at experiment time.
    """
    import math

    if target_bits <= 0:
        raise ValueError("target_bits must be positive")
    candidates: list[tuple[float, tuple[int, ...]]] = []
    # Greedy constructive search: build vectors left-to-right up to max_levels.
    stack: list[tuple[float, tuple[int, ...]]] = [(0.0, ())]
    while stack:
        bits_so_far, levels_so_far = stack.pop()
        if len(levels_so_far) >= max_levels:
            continue
        for level in allowed_levels:
            new_levels = levels_so_far + (level,)
            new_bits = bits_so_far + math.log2(level)
            if new_bits >= target_bits:
                excess = new_bits - target_bits
                candidates.append((excess, new_levels))
            else:
                # Only continue if adding the largest level could still reach target.
                if new_bits + math.log2(max(allowed_levels)) * (
                    max_levels - len(new_levels)
                ) >= target_bits:
                    stack.append((new_bits, new_levels))
    candidates.sort(key=lambda x: (x[0], len(x[1]), x[1]))
    # De-duplicate and return top alternatives.
    seen: set[tuple[int, ...]] = set()
    result: list[tuple[int, ...]] = []
    for _, levels in candidates:
        if levels not in seen:
            seen.add(levels)
            result.append(levels)
            if len(result) >= 8:
                break
    return result
