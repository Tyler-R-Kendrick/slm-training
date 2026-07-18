"""A removable LoRA-style low-rank delta wrapper for a single linear module (LDI2-01).

For a frozen parent linear ``W`` (bias untouched):

    y = W x + scale * B(A(dropout(x))),    scale = alpha / rank

``B`` is zero-initialized, so a freshly attached adapter is output-identical to the
parent bit-for-bit. Only ``A``/``B`` receive gradients; the parent weight and bias are
frozen. The adapter can be disabled (restoring the exact parent map even after ``A``/``B``
have changed), and merged one-way into a standalone ``nn.Linear`` on a copy — merging
never mutates the removable adapter. This is standard low-rank delta; it is deliberately
not called DoRA/PiSSA, which are not implemented here.
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["LowRankAdapter"]


class LowRankAdapter(nn.Module):
    """Wrap a frozen ``nn.Linear`` with a removable low-rank delta."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LowRankAdapter wraps an nn.Linear module")
        if int(rank) <= 0:
            raise ValueError("adapter rank must be positive")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout: nn.Module = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        weight = base.weight
        self.lora_A = nn.Parameter(
            torch.zeros(self.rank, base.in_features, dtype=weight.dtype, device=weight.device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, self.rank, dtype=weight.dtype, device=weight.device)
        )
        # A is randomized; B stays zero so the initial delta is exactly zero.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable_adapter(self) -> None:
        self._enabled = True

    def disable_adapter(self) -> None:
        self._enabled = False

    def adapter_parameters(self) -> list[nn.Parameter]:
        return [self.lora_A, self.lora_B]

    def delta_weight(self) -> torch.Tensor:
        """The dense delta ``scale * (B @ A)`` in the parent weight's dtype/device."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def merged_weight(self) -> torch.Tensor:
        """The parent weight with the delta folded in (does not mutate the parent)."""
        return self.base.weight.data + self.delta_weight()

    def merged_linear(self) -> nn.Linear:
        """A standalone ``nn.Linear`` equal to the adapter-enabled map, wrapper-free.

        The parent and this adapter are left untouched — merge is one-way on a copy.
        """
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
        ).to(self.base.weight.device, self.base.weight.dtype)
        with torch.no_grad():
            merged.weight.copy_(self.merged_weight())
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias.data)
        return merged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self._enabled:
            return out
        delta = nn.functional.linear(self.dropout(x), self.lora_A)
        delta = nn.functional.linear(delta, self.lora_B)
        return out + self.scaling * delta
