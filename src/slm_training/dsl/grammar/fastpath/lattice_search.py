"""Bounded search state over compiler-valid completion paths."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Callable, Generic, Mapping, TypeVar

from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionForest,
    CompletionPath,
)

PathKey = tuple[int, ...]
Nogood = tuple[tuple[int, ...], PathKey]
T = TypeVar("T")


def path_key(path: CompletionPath) -> PathKey:
    return tuple(int(token_id) for token_id in path.token_ids)


@dataclass(frozen=True)
class RankedForest:
    """Hard compiler candidates with an independent soft ordering."""

    paths: tuple[CompletionPath, ...]
    scores: tuple[float, ...]
    coverage: str

    @property
    def is_bottom(self) -> bool:
        return not self.paths

    @property
    def signature(self) -> str:
        payload = repr(
            (self.coverage, tuple(sorted(path_key(path) for path in self.paths)))
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def rank_forest(
    forest: CompletionForest,
    scores: Mapping[PathKey, float] | None = None,
    *,
    prefix: tuple[int, ...] = (),
    nogoods: frozenset[Nogood] = frozenset(),
) -> RankedForest:
    """Filter only explicit nogoods, then order by soft score."""
    scores = scores or {}
    live = [path for path in forest.paths if (prefix, path_key(path)) not in nogoods]
    live.sort(
        key=lambda path: (-float(scores.get(path_key(path), 0.0)), path_key(path))
    )
    return RankedForest(
        tuple(live),
        tuple(float(scores.get(path_key(path), 0.0)) for path in live),
        forest.coverage,
    )


def refine_hard_paths(
    previous: tuple[CompletionPath, ...],
    projected: tuple[CompletionPath, ...],
) -> tuple[CompletionPath, ...]:
    """Enforce monotone hard refinement at one stable prefix."""
    allowed = {path_key(path) for path in previous}
    if any(path_key(path) not in allowed for path in projected):
        raise ValueError("hard refinement cannot add candidates")
    return projected


@dataclass
class StagnationTracker:
    """Detect repeated hard states without prefix progress."""

    patience: int = 2
    last: tuple[str, int] | None = None
    repeats: int = 0

    def observe(self, signature: str, progress: int) -> bool:
        current = (signature, int(progress))
        self.repeats = self.repeats + 1 if current == self.last else 0
        self.last = current
        return self.repeats >= max(1, int(self.patience))


def trajectory_orders(
    ranked: RankedForest,
    *,
    width: int,
    noise: float,
    seed: int,
) -> tuple[tuple[CompletionPath, ...], ...]:
    """Return seeded soft permutations without changing hard membership."""
    width = max(1, int(width))
    orders: list[tuple[CompletionPath, ...]] = []
    for trajectory in range(width):
        rng = random.Random(f"{seed}:{ranked.signature}:{trajectory}")
        decorated = [
            (
                -(float(score) + (rng.uniform(-noise, noise) if noise > 0 else 0.0)),
                path_key(path),
                path,
            )
            for path, score in zip(ranked.paths, ranked.scores, strict=True)
        ]
        order = tuple(row[2] for row in sorted(decorated))
        if order not in orders:
            orders.append(order)
    return tuple(orders)


def deduplicate_semantic_candidates(
    candidates: tuple[T, ...], fingerprint: Callable[[T], str]
) -> tuple[T, ...]:
    """Keep the first validated candidate for each semantic fingerprint."""
    seen: set[str] = set()
    unique: list[T] = []
    for candidate in candidates:
        key = str(fingerprint(candidate))
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return tuple(unique)


@dataclass(frozen=True)
class TrajectoryCandidate(Generic[T]):
    """One terminal continuation ranked without access to gold output."""

    value: T
    valid: bool
    contract_satisfied: bool
    model_score: float
    simplicity: int
    fingerprint: str = ""


def select_trajectory_candidate(
    candidates: tuple[TrajectoryCandidate[T], ...], *, semantic_dedup: bool
) -> tuple[TrajectoryCandidate[T] | None, int]:
    """Select valid > contract-satisfying > model score > simple, never gold."""
    if semantic_dedup:
        seen: set[str] = set()
        live: list[TrajectoryCandidate[T]] = []
        for candidate in candidates:
            key = candidate.fingerprint if candidate.valid else ""
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            live.append(candidate)
    else:
        live = list(candidates)
    if not live:
        return None, 0
    selected = max(
        live,
        key=lambda row: (
            row.valid,
            row.contract_satisfied,
            row.model_score,
            -row.simplicity,
        ),
    )
    return selected, len(
        {row.fingerprint for row in live if row.valid and row.fingerprint}
    )


@dataclass(frozen=True)
class SearchDecision:
    prefix: tuple[int, ...]
    chosen: CompletionPath
    alternatives: tuple[CompletionPath, ...]


@dataclass
class LatticeSearchState:
    """Decision trail and local conflict memory for bounded compiler search."""

    backtrack_limit: int = 8
    decisions: list[SearchDecision] = field(default_factory=list)
    nogoods: set[Nogood] = field(default_factory=set)
    backtracks: int = 0

    def clone(self) -> "LatticeSearchState":
        """Independent branch state without deepcopy.

        choose/rollback only append/pop decisions, add nogoods, and bump
        counters; SearchDecision and CompletionPath are frozen, so copying
        the two containers fully isolates a trajectory branch.
        """
        return LatticeSearchState(
            backtrack_limit=self.backtrack_limit,
            decisions=list(self.decisions),
            nogoods=set(self.nogoods),
            backtracks=self.backtracks,
        )

    def choose(self, prefix: list[int], ranked: RankedForest) -> CompletionPath | None:
        if ranked.is_bottom:
            return None
        chosen = ranked.paths[0]
        if len(ranked.paths) > 1:
            self.decisions.append(
                SearchDecision(tuple(prefix), chosen, tuple(ranked.paths[1:]))
            )
        return chosen

    def rollback(
        self, *, local_nogoods: bool = True
    ) -> tuple[list[int], CompletionPath | None] | None:
        """Reject the latest conflicting choice and restore its stable prefix."""
        if self.backtracks >= self.backtrack_limit:
            return None
        while self.decisions:
            decision = self.decisions.pop()
            if local_nogoods:
                self.nogoods.add((decision.prefix, path_key(decision.chosen)))
            self.backtracks += 1
            alternative = next(
                (
                    path
                    for path in decision.alternatives
                    if (decision.prefix, path_key(path)) not in self.nogoods
                ),
                None,
            )
            if alternative is not None:
                if local_nogoods:
                    # Reproject at the stable prefix so the compiler, not this
                    # trail, proves that the remaining alternative is live.
                    return list(decision.prefix), None
                remaining = tuple(
                    path for path in decision.alternatives if path != alternative
                )
                if remaining:
                    self.decisions.append(
                        SearchDecision(decision.prefix, alternative, remaining)
                    )
                return list(decision.prefix), alternative
            if self.backtracks >= self.backtrack_limit:
                return None
        return None


__all__ = [
    "LatticeSearchState",
    "Nogood",
    "PathKey",
    "RankedForest",
    "SearchDecision",
    "StagnationTracker",
    "TrajectoryCandidate",
    "deduplicate_semantic_candidates",
    "path_key",
    "rank_forest",
    "refine_hard_paths",
    "select_trajectory_candidate",
    "trajectory_orders",
]
