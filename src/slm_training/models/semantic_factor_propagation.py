"""Transparent HyCE-style restart diffusion over factor-node incidence.

Advisory only: convergence is not semantic correctness. Uses NumPy only.
No graph framework dependency.

Reference operator (column-vector convention)::

    S = B D_e^{-1} B^T D_v^{-1}
    c^{t+1} = λ r + (1-λ) S c^t

Under positive degrees, S is column-stochastic and the map is an ℓ₁ contraction
with rate (1-λ) for λ ∈ (0,1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from slm_training.data.progspec.semantic_evidence import FactorGraphView

__all__ = [
    "PropagationConfig",
    "PropagationResult",
    "build_incidence_matrix",
    "column_stochastic_S",
    "propagate_restart",
    "split_polarity_channels",
]


@dataclass(frozen=True)
class PropagationConfig:
    lambda_restart: float = 0.15
    max_steps: int = 32
    tol: float = 1e-8
    gamma_contradict: float = 1.0


@dataclass(frozen=True)
class PropagationResult:
    confidence: np.ndarray
    steps: int
    residual_l1: float
    column_sums: np.ndarray
    converged: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": [float(x) for x in self.confidence.tolist()],
            "steps": int(self.steps),
            "residual_l1": float(self.residual_l1),
            "column_sums": [float(x) for x in self.column_sums.tolist()],
            "converged": bool(self.converged),
            "notes": list(self.notes),
        }


def build_incidence_matrix(view: FactorGraphView) -> np.ndarray:
    """Binary |V|×|F| incidence matrix B from a factor-node view."""

    n_v = len(view.vertex_ids)
    n_f = len(view.factor_ids)
    B = np.zeros((n_v, n_f), dtype=np.float64)
    for v_idx, f_idx in view.incidence:
        B[int(v_idx), int(f_idx)] = 1.0
    return B


def column_stochastic_S(B: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Build S = B D_e^{-1} B^T D_v^{-1} after dropping zero-degree rows/cols notes."""

    notes: list[str] = []
    if B.size == 0:
        return np.zeros((0, 0), dtype=np.float64), np.zeros(0), ("empty_incidence",)

    d_e = B.sum(axis=0)  # factor degrees
    d_v = B.sum(axis=1)  # vertex degrees
    if np.any(d_e <= 0) or np.any(d_v <= 0):
        keep_f = d_e > 0
        keep_v = d_v > 0
        notes.append("dropped_isolated")
        B = B[np.ix_(keep_v, keep_f)]
        d_e = B.sum(axis=0)
        d_v = B.sum(axis=1)
    if B.size == 0 or np.any(d_e <= 0) or np.any(d_v <= 0):
        n = B.shape[0]
        return np.eye(n, dtype=np.float64), np.ones(n), tuple(notes + ["degenerate"])

    De_inv = np.diag(1.0 / d_e)
    Dv_inv = np.diag(1.0 / d_v)
    S = B @ De_inv @ B.T @ Dv_inv
    column_sums = S.sum(axis=0)
    return S, column_sums, tuple(notes)


def propagate_restart(
    B: np.ndarray,
    restart: np.ndarray,
    *,
    config: PropagationConfig | None = None,
) -> PropagationResult:
    """Iterate c <- λ r + (1-λ) S c until tol or max_steps."""

    cfg = config or PropagationConfig()
    lam = float(cfg.lambda_restart)
    if not (0.0 < lam <= 1.0):
        raise ValueError("lambda_restart must be in (0,1]")
    S, column_sums, notes = column_stochastic_S(B)
    n = S.shape[0]
    r = np.asarray(restart, dtype=np.float64).reshape(-1)
    if r.size != n:
        # Align by truncation/pad when isolates were dropped — fail closed to uniform.
        r = np.ones(n, dtype=np.float64) / max(1, n)
        notes = notes + ("restart_realigned_uniform",)
    r_sum = float(r.sum())
    if r_sum <= 0:
        r = np.ones(n, dtype=np.float64) / max(1, n)
    else:
        r = r / r_sum
    c = r.copy()
    residual = float("inf")
    steps = 0
    for steps in range(1, int(cfg.max_steps) + 1):
        c_next = lam * r + (1.0 - lam) * (S @ c)
        residual = float(np.abs(c_next - c).sum())
        c = c_next
        if residual <= float(cfg.tol):
            return PropagationResult(
                confidence=c,
                steps=steps,
                residual_l1=residual,
                column_sums=column_sums,
                converged=True,
                notes=notes,
            )
    return PropagationResult(
        confidence=c,
        steps=int(cfg.max_steps),
        residual_l1=residual,
        column_sums=column_sums,
        converged=False,
        notes=notes + ("max_steps",),
    )


def split_polarity_channels(
    view: FactorGraphView,
    *,
    config: PropagationConfig | None = None,
) -> tuple[PropagationResult, PropagationResult, np.ndarray]:
    """Positive/negative channels; Δ = c+ − γ c− (never negative stochastic S)."""

    cfg = config or PropagationConfig()
    B = build_incidence_matrix(view)
    n = B.shape[0]
    # Restart mass from polarity-tagged factors projected to members.
    r_pos = np.zeros(n, dtype=np.float64)
    r_neg = np.zeros(n, dtype=np.float64)
    for v_idx, f_idx in view.incidence:
        pol = view.polarity[f_idx] if f_idx < len(view.polarity) else "supports"
        if pol == "contradicts":
            r_neg[v_idx] += 1.0
        else:
            r_pos[v_idx] += 1.0
    if r_pos.sum() <= 0:
        r_pos = np.ones(n) / max(1, n)
    if r_neg.sum() <= 0:
        r_neg = np.zeros(n)
    pos = propagate_restart(B, r_pos, config=cfg)
    if r_neg.sum() > 0:
        neg = propagate_restart(B, r_neg, config=cfg)
    else:
        neg = PropagationResult(
            confidence=np.zeros(n),
            steps=0,
            residual_l1=0.0,
            column_sums=pos.column_sums,
            converged=True,
            notes=("no_contradiction_channel",),
        )
    delta = pos.confidence - float(cfg.gamma_contradict) * neg.confidence
    return pos, neg, delta
