"""Tiny trust gate for LayerSkip-style early exit (optional).

Also hosts the A2 (SLM-38) distribution-aware constrained-decode primitives:
the single-step ASAp ledger + re-weighting used by the MaskGIT unmask loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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


# --- A2 (SLM-38): distribution-aware constrained decode (single-step ASAp) ---
#
# Grammar-Aligned Decoding / ASAp (Park et al., NeurIPS 2024,
# "Grammar-Aligned Decoding", arXiv:2405.21047) shows that hard constraint
# masking renormalizes the model distribution over only the grammar-legal
# tokens at each step. That renormalization discards the probability mass the
# constraint removed and *provably* distorts the sampled distribution away from
# the model's true grammatical continuation preferences — combined with a
# length prior it pulls decode toward grammatical-but-empty programs (the A1
# emptiness diagnosis, ``evals/emptiness_probe.py``). Full ASAp corrects this by
# tracking an approximate expected-future grammaticality per prefix and
# re-weighting across samples. The primitives here implement the honest
# *single-step* approximation: correct using the removed mass observed at the
# current position instead of a full expected-future computation.


@dataclass
class AsapLedger:
    """Ledger of grammar-removed probability mass over a constrained decode.

    Records, per committed MaskGIT position, the model probability mass the
    grammar constraint removed (``removed_mass`` below). This is the live
    telemetry that proves the ASAp correction is active and quantifies the
    distortion the constraint introduces; it is *not* a ship metric.
    """

    positions: int = 0
    removed_mass_sum: float = 0.0
    max_removed_mass: float = 0.0
    nonzero_removed: int = 0

    def record(self, removed: float) -> None:
        r = float(removed)
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        self.positions += 1
        self.removed_mass_sum += r
        if r > self.max_removed_mass:
            self.max_removed_mass = r
        if r > 1e-9:
            self.nonzero_removed += 1

    @property
    def mean_removed_mass(self) -> float:
        return self.removed_mass_sum / self.positions if self.positions else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "positions": self.positions,
            "removed_mass_sum": round(self.removed_mass_sum, 6),
            "mean_removed_mass": round(self.mean_removed_mass, 6),
            "max_removed_mass": round(self.max_removed_mass, 6),
            "nonzero_removed": self.nonzero_removed,
        }


def removed_mass(probs_1d: torch.Tensor, legal_ids: Iterable[int]) -> float:
    """Model probability mass on grammar-illegal tokens at one position.

    ``probs_1d`` is a full-vocabulary distribution (sums to 1). ``legal_ids`` is
    the grammar-legal continuation set. Returns ``1.0`` when the legal set is
    empty (fail-closed: the constraint removed *all* mass) and is clamped to
    ``[0, 1]`` so numerical drift never yields a NaN/negative ledger entry.
    """
    vocab = int(probs_1d.numel())
    legal = sorted({int(i) for i in legal_ids if 0 <= int(i) < vocab})
    if not legal:
        return 1.0
    idx = torch.as_tensor(legal, dtype=torch.long, device=probs_1d.device)
    s = float(probs_1d.index_select(0, idx).sum().item())
    m = 1.0 - s
    if m < 0.0:
        m = 0.0
    elif m > 1.0:
        m = 1.0
    return m


def asap_reweight(
    probs_1d: torch.Tensor,
    legal_ids: Iterable[int],
    *,
    alpha: float = 1.0,
    eps: float = 1e-12,
) -> tuple[list[int], torch.Tensor]:
    """Single-step ASAp-corrected distribution over the grammar-legal tokens.

    Plain constraint decoding renormalizes ``q0(t) = p(t) / S`` for legal ``t``
    (``S`` = legal mass = ``1 - M``; ``M`` = removed mass). That renormalization
    inflates every legal token's confidence by exactly the mass ``M`` the
    constraint removed — the ASAp distortion. The single-step correction damps
    the renormalization in proportion to ``M``::

        gamma = clip(1 - alpha * M, 1e-3, 1)     # alpha in [0, 1]
        w(t)  = p(t) ** gamma
        q(t)  = w(t) / sum_t w(t)

    ``x ** gamma`` is monotone for ``x > 0``, so the winning legal token never
    changes (and an illegal token is never admitted — fail-closed), but the
    legal distribution is *flattened* where the constraint removed a lot of
    mass, so the confidence-scheduled MaskGIT unmask loop defers those
    high-distortion positions. ``alpha == 0`` or ``M == 0`` reproduces plain
    renormalization exactly. Returns ``(legal_ids_sorted, q)`` with ``q`` a
    valid distribution (sums to 1, non-negative). An empty legal set returns
    ``([], empty_tensor)`` (fail-closed).
    """
    vocab = int(probs_1d.numel())
    legal = sorted({int(i) for i in legal_ids if 0 <= int(i) < vocab})
    if not legal:
        return [], probs_1d.new_zeros(0)
    idx = torch.as_tensor(legal, dtype=torch.long, device=probs_1d.device)
    p = probs_1d.index_select(0, idx).double().clamp(min=eps)
    m = (1.0 - p.sum()).clamp(min=0.0, max=1.0)
    a = float(max(0.0, min(1.0, alpha)))
    gamma = float((1.0 - a * m).clamp(min=1e-3, max=1.0).item())
    w = p.pow(gamma)
    q = (w / w.sum()).to(probs_1d.dtype)
    return legal, q
