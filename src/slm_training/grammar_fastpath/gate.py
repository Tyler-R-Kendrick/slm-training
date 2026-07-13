"""Tiny trust gate for LayerSkip-style early exit (optional)."""

from __future__ import annotations

import torch
import torch.nn as nn


class FastPathGate(nn.Module):
    """Sigmoid trust head on hidden states — does not override DFA correctness."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [B, T, D] -> trust [B, T] in (0,1)."""
        return torch.sigmoid(self.proj(hidden).squeeze(-1))
