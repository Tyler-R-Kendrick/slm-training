"""VSS3-02 (SLM-70): cost-to-go energy scorer for the verified solver.

The scorer's *only* runtime authority is to **order** the exact live candidates
supplied by the solver's `CandidateRanker` seam. It never defines candidate
membership, certifies support, suppresses `UNKNOWN`, or bypasses the final
verifier. Exact closure and the hard live set are computed first; energy ranking
may change candidate order (and thus expanded nodes / backtracks / latency) and
nothing else.

The learned target is **search cost-to-go**, not program likelihood:

    E_theta(state, hole, candidate) ~= expected remaining exact search work

Lower energy ranks earlier. Training consumes the replay-verified VSS3-01
(`solver_supervision`) `candidate_cost` rows, and only where `cost_observed` is
true; `UNKNOWN`/censored rows are masked from the hard losses, never relabeled
worst-cost. Low energy is a ranking hint, never a correctness claim.

This module is a self-contained adapter: features enter as tensors
(`CandidateEnergyInput`), so the scorer needs no invasive model surgery and old
checkpoints load unchanged (the head is not baked into the base state dict).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from slm_training.dsl.solver.state import DomainValue, FiniteDomainState, HoleId

WORK_TARGET_VERSION = "v1"

# Versioned scalar work coefficients (stored in dataset/checkpoint metadata).
DEFAULT_WORK_WEIGHTS: dict[str, float] = {
    "expanded_nodes": 1.0,
    "verifier_calls": 1.0,
    "backtracks": 1.0,
    "decisions": 1.0,
}


def work_scalar(
    *,
    expanded_nodes: float,
    verifier_calls: float,
    backtracks: float,
    decisions: float,
    weights: dict[str, float] | None = None,
) -> float:
    w = weights or DEFAULT_WORK_WEIGHTS
    return (
        w["expanded_nodes"] * float(expanded_nodes)
        + w["verifier_calls"] * float(verifier_calls)
        + w["backtracks"] * float(backtracks)
        + w["decisions"] * float(decisions)
    )


def cost_target_from_row(row: dict, weights: dict[str, float] | None = None) -> float:
    """`log1p(work)` target from a VSS3-01 `candidate_cost` row."""
    work = work_scalar(
        expanded_nodes=row.get("remaining_expanded_nodes", 0),
        verifier_calls=row.get("remaining_verifier_calls", 0),
        backtracks=row.get("remaining_backtracks", 0),
        decisions=row.get("remaining_decisions", 0),
        weights=weights,
    )
    return math.log1p(work)


# --------------------------------------------------------------------------- #
# Scorer I/O contract
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CandidateEnergyInput:
    """Decision-time features for one hole's live candidates (post-closure).

    Carries only tensors + the state fingerprint + candidate identities. No final
    source, future decision, verifier report, or witness text ever enters here.
    """

    state_fingerprint: str
    state_features: torch.Tensor  # [F_state]
    hole_features: torch.Tensor  # [F_hole]
    candidate_ids: tuple[DomainValue, ...]
    candidate_features: torch.Tensor  # [num_candidates, F_cand]


@dataclass(frozen=True)
class CandidateEnergyOutput:
    energies: torch.Tensor  # [num_candidates]
    candidate_ids: tuple[DomainValue, ...]
    scorer_id: str


class CandidateEnergyScorer(nn.Module):
    """Small MLP mapping (state, hole, candidate) features to a scalar energy."""

    def __init__(
        self,
        *,
        state_dim: int,
        hole_dim: int,
        candidate_dim: int,
        hidden_dim: int = 64,
        scorer_id: str = "solver-energy-v1",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hole_dim = hole_dim
        self.candidate_dim = candidate_dim
        self._scorer_id = scorer_id
        self.net = nn.Sequential(
            nn.Linear(state_dim + hole_dim + candidate_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def scorer_id(self) -> str:
        return self._scorer_id

    def energies(self, inp: CandidateEnergyInput) -> torch.Tensor:
        n = inp.candidate_features.shape[0]
        state = inp.state_features.reshape(1, -1).expand(n, -1)
        hole = inp.hole_features.reshape(1, -1).expand(n, -1)
        x = torch.cat([state, hole, inp.candidate_features], dim=-1)
        return self.net(x).squeeze(-1)

    def forward(self, inp: CandidateEnergyInput) -> CandidateEnergyOutput:
        return CandidateEnergyOutput(
            energies=self.energies(inp),
            candidate_ids=inp.candidate_ids,
            scorer_id=self.scorer_id,
        )


# --------------------------------------------------------------------------- #
# Ranking adapter — permutation-only, membership-preserving
# --------------------------------------------------------------------------- #


def _multiset(values: tuple[DomainValue, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = repr(value.to_dict())
        counts[key] = counts.get(key, 0) + 1
    return counts


class CandidateEnergyRanker:
    """`CandidateRanker` adapter: energy orders live values, never alters them.

    Guarantees a permutation of the supplied `values`. Any scorer defect — a
    count mismatch, duplicate, NaN, or infinite energy — triggers the deterministic
    fallback (canonical identity order) and is counted in `fallback_count`; the
    hard domain is never changed.
    """

    def __init__(
        self,
        scorer: CandidateEnergyScorer,
        feature_fn,
        *,
        fallback: str = "deterministic",
    ) -> None:
        self.scorer = scorer
        self.feature_fn = feature_fn
        self.fallback = fallback
        self.fallback_count = 0
        self.rank_count = 0

    @property
    def ranker_id(self) -> str:
        return f"energy:{getattr(self.scorer, 'scorer_id', 'unknown')}"

    def rank(
        self,
        state: FiniteDomainState,
        hole_id: HoleId,
        values: tuple[DomainValue, ...],
    ) -> tuple[DomainValue, ...]:
        self.rank_count += 1
        if len(values) <= 1:
            return tuple(values)
        try:
            ordered = self._energy_order(state, hole_id, values)
        except Exception:  # noqa: BLE001 — any scorer defect must fail safe
            self.fallback_count += 1
            return tuple(values)  # deterministic identity fallback
        # Fail-closed permutation check: identical multiset, no add/drop.
        if _multiset(ordered) != _multiset(values):
            self.fallback_count += 1
            return tuple(values)
        return ordered

    def _energy_order(
        self,
        state: FiniteDomainState,
        hole_id: HoleId,
        values: tuple[DomainValue, ...],
    ) -> tuple[DomainValue, ...]:
        with torch.no_grad():
            inp = self.feature_fn(state, hole_id, values)
            energies = self.scorer.energies(inp).reshape(-1)
        if energies.shape[0] != len(values):
            raise ValueError("scorer returned a different candidate count")
        if not bool(torch.isfinite(energies).all()):
            raise ValueError("scorer produced a non-finite energy")
        # Stable order by (energy, original index) — deterministic for ties.
        order = sorted(range(len(values)), key=lambda i: (float(energies[i]), i))
        return tuple(values[i] for i in order)


# --------------------------------------------------------------------------- #
# Losses (masked; multiple supported alternatives never forced to one-hot)
# --------------------------------------------------------------------------- #


def energy_regression_loss(
    energies: torch.Tensor,
    targets: torch.Tensor,
    observed_mask: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber regression applied only where cost is observed."""
    mask = observed_mask.bool()
    if int(mask.sum()) == 0:
        return energies.new_zeros(())
    return F.huber_loss(energies[mask], targets[mask], delta=delta)


def energy_pairwise_loss(
    energies: torch.Tensor,
    targets: torch.Tensor,
    group_ids: list,
    observed_mask: torch.Tensor,
    *,
    margin: float = 0.0,
) -> torch.Tensor:
    """Ranking loss over same-state/hole pairs with distinct observed costs.

    For a pair whose observed cost differs, the lower-cost candidate should get
    the lower energy. Pairs with equal observed cost are skipped, so several
    comparably-cheap supported alternatives are never forced into a strict order.
    """
    mask = observed_mask.bool()
    idx = [i for i in range(len(group_ids)) if bool(mask[i])]
    terms: list[torch.Tensor] = []
    for a_pos in range(len(idx)):
        for b_pos in range(a_pos + 1, len(idx)):
            i, j = idx[a_pos], idx[b_pos]
            if group_ids[i] != group_ids[j]:
                continue
            ti, tj = float(targets[i]), float(targets[j])
            if ti == tj:
                continue
            low, high = (i, j) if ti < tj else (j, i)
            # low cost should have lower energy: penalize energy[low] - energy[high] + margin
            terms.append(F.relu(energies[low] - energies[high] + margin))
    if not terms:
        return energies.new_zeros(())
    return torch.stack(terms).mean()


__all__ = [
    "DEFAULT_WORK_WEIGHTS",
    "WORK_TARGET_VERSION",
    "CandidateEnergyInput",
    "CandidateEnergyOutput",
    "CandidateEnergyRanker",
    "CandidateEnergyScorer",
    "cost_target_from_row",
    "energy_pairwise_loss",
    "energy_regression_loss",
    "work_scalar",
]
