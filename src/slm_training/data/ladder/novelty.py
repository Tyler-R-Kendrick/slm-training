"""Measured novelty budgets for frozen prompt variants."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class NoveltyDimension(str, Enum):
    FORM = "form"
    ORDERING = "ordering"
    OMISSION = "omission"
    PERSONA = "persona"
    REFERENTIAL_STRUCTURE = "referential_structure"
    AMBIGUITY = "ambiguity"
    MODALITY = "modality"


@dataclass(frozen=True)
class NoveltyCandidate:
    id: str
    text: str
    dimensions: tuple[NoveltyDimension, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimensions", tuple(sorted(set(self.dimensions), key=lambda x: x.value)))
        if not self.id or not self.text.strip():
            raise ValueError("novelty candidate id and text must be non-empty")


@dataclass(frozen=True)
class NoveltyDecision:
    candidate: NoveltyCandidate
    accepted: bool
    reason: str


@dataclass(frozen=True)
class NoveltyReport:
    decisions: tuple[NoveltyDecision, ...]

    @property
    def accepted(self) -> tuple[NoveltyCandidate, ...]:
        return tuple(d.candidate for d in self.decisions if d.accepted)

    @property
    def dropped_count(self) -> int:
        return sum(not decision.accepted for decision in self.decisions)

    @property
    def near_duplicate_count(self) -> int:
        return sum(decision.reason == "near_duplicate" for decision in self.decisions)

    def to_metrics(self) -> dict[str, int]:
        return {
            "candidates": len(self.decisions),
            "accepted": len(self.accepted),
            "dropped": self.dropped_count,
            "near_duplicates": self.near_duplicate_count,
        }


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a or b else 1.0


@dataclass(frozen=True)
class NoveltyBudget:
    max_variants: int = 8
    max_per_signature: int = 2
    near_duplicate_threshold: float = 0.85

    def __post_init__(self) -> None:
        if self.max_variants < 0 or self.max_per_signature < 1:
            raise ValueError("novelty caps must be non-negative")
        if not 0.0 <= self.near_duplicate_threshold <= 1.0:
            raise ValueError("near_duplicate_threshold must be in [0, 1]")

    def select(self, candidates: tuple[NoveltyCandidate, ...]) -> NoveltyReport:
        decisions: list[NoveltyDecision] = []
        accepted: list[NoveltyCandidate] = []
        signature_counts: dict[tuple[str, ...], int] = {}
        for candidate in sorted(candidates, key=lambda item: item.id):
            signature = tuple(dimension.value for dimension in candidate.dimensions)
            reason = "accepted"
            if not signature:
                reason = "no_novelty_dimension"
            elif NoveltyDimension.MODALITY in candidate.dimensions:
                reason = "online_modality_only"
            elif len(accepted) >= self.max_variants:
                reason = "total_cap"
            elif signature_counts.get(signature, 0) >= self.max_per_signature:
                reason = "signature_cap"
            elif any(
                _similarity(candidate.text, prior.text) >= self.near_duplicate_threshold
                for prior in accepted
            ):
                reason = "near_duplicate"
            is_accepted = reason == "accepted"
            decisions.append(NoveltyDecision(candidate, is_accepted, reason))
            if is_accepted:
                accepted.append(candidate)
                signature_counts[signature] = signature_counts.get(signature, 0) + 1
        return NoveltyReport(tuple(decisions))
