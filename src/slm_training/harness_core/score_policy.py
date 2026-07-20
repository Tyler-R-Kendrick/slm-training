"""Eval-only score policies for grammar-constrained candidate ranking.

EFS1-03 (SLM-110): compare principled score corrections for the valid-but-empty
length bias on durable frontier checkpoints. These policies operate on a
per-candidate trace of token-level log probabilities and optional removed-mass
estimates; they do not own legality and do not mutate the model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class CandidatePath:
    """Sufficient statistics for scoring one grammar-valid candidate."""

    candidate_id: str
    token_ids: tuple[int, ...]
    log_probs: tuple[float, ...]
    # Optional per-step removed probability mass (e.g., from legal masking).
    removed_mass: tuple[float, ...] | None = None
    # Optional mask: 1.0 for semantic decisions, 0.0 for forced syntax/surface.
    semantic_mask: tuple[float, ...] | None = None
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if len(self.log_probs) != len(self.token_ids):
            raise ValueError("log_probs length must match token_ids length")
        if self.removed_mass is not None and len(self.removed_mass) != len(self.token_ids):
            raise ValueError("removed_mass length must match token_ids length")
        if self.semantic_mask is not None and len(self.semantic_mask) != len(self.token_ids):
            raise ValueError("semantic_mask length must match token_ids length")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "token_ids": list(self.token_ids),
            "log_probs": list(self.log_probs),
            "removed_mass": (
                list(self.removed_mass) if self.removed_mass is not None else None
            ),
            "semantic_mask": (
                list(self.semantic_mask) if self.semantic_mask is not None else None
            ),
            "metadata": dict(self.metadata or {}),
        }


class ScorePolicy(Protocol):
    """Policy that turns a candidate path into a scalar score (higher = better)."""

    @property
    def name(self) -> str: ...

    def score(self, path: CandidatePath) -> float: ...

    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RawCumulativePolicy:
    """Raw cumulative log-probability — what a greedy constrained decoder ranks."""

    name: str = "raw_cumulative"

    def score(self, path: CandidatePath) -> float:
        return sum(path.log_probs)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name}


@dataclass(frozen=True)
class SemanticLengthNormPolicy:
    """Normalize cumulative log-prob by the number of non-forced semantic decisions.

    Tuning ``alpha`` lets the policy interpolate between raw cumulative
    (alpha=0.0) and a strict per-decision average (alpha=1.0).
    """

    alpha: float = 1.0
    name: str = "semantic_length_norm"

    def score(self, path: CandidatePath) -> float:
        total = sum(path.log_probs)
        if path.semantic_mask is None:
            n_semantic = len(path.log_probs)
        else:
            n_semantic = max(1.0, sum(path.semantic_mask))
        if n_semantic == 0.0:
            return total
        per_decision = total / n_semantic
        return (1.0 - self.alpha) * total + self.alpha * per_decision

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "alpha": self.alpha}


@dataclass(frozen=True)
class GrammarAlignedMassPolicy:
    """Reward candidates that retain probability mass after legal masking.

    Uses per-step removed mass (or a token-level complement estimate) as a
    Grammar-Aligned Decoding / ASAp-style correction. The score is the raw
    cumulative log-prob minus a penalty proportional to removed mass.
    """

    beta: float = 1.0
    name: str = "grammar_aligned_mass"

    def score(self, path: CandidatePath) -> float:
        total = sum(path.log_probs)
        if path.removed_mass is None:
            # Fallback: estimate removed mass as 1 - p(gold_token).
            removed = [
                max(0.0, 1.0 - math.exp(max(lp, -50.0))) for lp in path.log_probs
            ]
        else:
            removed = list(path.removed_mass)
        if not removed:
            return total
        penalty = self.beta * sum(math.log(max(m, 1e-12)) for m in removed)
        return total - penalty

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "beta": self.beta}


@dataclass(frozen=True)
class MinimumMassRemaskPolicy:
    """Soft content-pressure policy: boost candidates with supported semantic mass.

    This is a wiring-level policy only; it does not implement a full remask
    loop. It scores a candidate by its cumulative log-prob plus a small bonus
    for retained mass on semantic positions.
    """

    gamma: float = 0.5
    name: str = "minimum_mass_remask"

    def score(self, path: CandidatePath) -> float:
        total = sum(path.log_probs)
        if path.removed_mass is None:
            return total
        mask = path.semantic_mask
        masses = path.removed_mass
        bonus = 0.0
        for i, m in enumerate(masses):
            weight = mask[i] if mask is not None else 1.0
            if weight > 0.0:
                bonus += math.log(max(m, 1e-12))
        return total + self.gamma * bonus

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "gamma": self.gamma}


@dataclass(frozen=True)
class ContentFloorPolicy:
    """Hard content-floor control: reject empty/minimal-shell candidates.

    This policy returns negative infinity for candidates with fewer than
    ``min_semantic_decisions`` semantic positions. It is a diagnostic policy,
    not a promotion candidate, because it requires a structured contract that
    proves the floor.
    """

    min_semantic_decisions: int = 1
    name: str = "content_floor"

    def score(self, path: CandidatePath) -> float:
        if path.semantic_mask is None:
            n_semantic = len(path.log_probs)
        else:
            n_semantic = sum(1.0 for v in path.semantic_mask if v > 0.0)
        if n_semantic < self.min_semantic_decisions:
            return float("-inf")
        return sum(path.log_probs)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "min_semantic_decisions": self.min_semantic_decisions}


POLICY_REGISTRY: dict[str, type[ScorePolicy]] = {
    "raw_cumulative": RawCumulativePolicy,
    "semantic_length_norm": SemanticLengthNormPolicy,
    "grammar_aligned_mass": GrammarAlignedMassPolicy,
    "minimum_mass_remask": MinimumMassRemaskPolicy,
    "content_floor": ContentFloorPolicy,
}


def build_policy(name: str, **kwargs: Any) -> ScorePolicy:
    """Construct a registered score policy by name."""
    if name not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown score policy {name!r}; choose from {sorted(POLICY_REGISTRY)}"
        )
    return POLICY_REGISTRY[name](**kwargs)  # type: ignore[return-value]


def rank_candidates(paths: Sequence[CandidatePath], policy: ScorePolicy) -> list[tuple[str, float]]:
    """Return candidate ids and scores sorted by score descending."""
    scored = [(path.candidate_id, policy.score(path)) for path in paths]
    return sorted(scored, key=lambda x: x[1], reverse=True)


def compare_policies(
    paths: Sequence[CandidatePath], policies: Sequence[ScorePolicy]
) -> dict[str, Any]:
    """Score every path under every policy and report rank changes."""
    results: dict[str, Any] = {}
    rankings: dict[str, list[str]] = {}
    for policy in policies:
        ranked = rank_candidates(paths, policy)
        rankings[policy.name] = [cid for cid, _ in ranked]
        results[policy.name] = [
            {"candidate_id": cid, "score": score} for cid, score in ranked
        ]
    return {"rankings": rankings, "scores": results}
