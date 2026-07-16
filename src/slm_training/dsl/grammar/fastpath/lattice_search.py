"""Bounded search state over compiler-valid completion paths."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping

from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionForest,
    CompletionPath,
)

PathKey = tuple[int, ...]
Nogood = tuple[tuple[int, ...], PathKey]


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
    live = [
        path
        for path in forest.paths
        if (prefix, path_key(path)) not in nogoods
    ]
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

    def choose(self, prefix: list[int], ranked: RankedForest) -> CompletionPath | None:
        if ranked.is_bottom:
            return None
        chosen = ranked.paths[0]
        if len(ranked.paths) > 1:
            self.decisions.append(
                SearchDecision(tuple(prefix), chosen, tuple(ranked.paths[1:]))
            )
        return chosen

    def rollback(self) -> tuple[list[int], CompletionPath] | None:
        """Reject the latest choice and return its next live alternative."""
        if self.backtracks >= self.backtrack_limit:
            return None
        while self.decisions:
            decision = self.decisions.pop()
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
                remaining = tuple(
                    path
                    for path in decision.alternatives
                    if path != alternative
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
    "path_key",
    "rank_forest",
    "refine_hard_paths",
]
