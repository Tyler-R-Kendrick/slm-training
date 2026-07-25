"""Legality-neutral scheduling prefixes over dynamic legal-edit candidates.

The exact compiler owns membership.  A proposal only changes which cheap
candidate descriptors are projected or verified first; omitted candidates are
restored by the mandatory fallback before an exact decision is returned.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


def _digest(parts: Sequence[str]) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateFeatureObject:
    """Cheap, inference-visible descriptor for one compiler-enumerated edit."""

    candidate_id: str
    family: str
    feature_digest: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.family or not self.feature_digest:
            raise ValueError("candidate identity, family, and feature digest are required")
        if not self.values or not all(math.isfinite(value) for value in self.values):
            raise ValueError("candidate features must be non-empty and finite")


@dataclass(frozen=True)
class ProposalTrainingRowV1:
    """Leakage-safe multi-positive supervision over one exact dynamic set."""

    row_id: str
    state_fingerprint: str
    hole_id: str
    complete_candidate_ids: tuple[str, ...]
    target_candidate_ids: tuple[str, ...]
    acceptable_candidate_ids: tuple[str, ...]
    supported_candidate_ids: tuple[str, ...]
    unsupported_candidate_ids: tuple[str, ...]
    unknown_candidate_ids: tuple[str, ...]
    candidate_feature_digests: tuple[tuple[str, str], ...]
    split: str
    lineage_digest: str
    checkpoint_digest: str | None
    config_digest: str
    bridge_version: str

    schema: str = "ProposalTrainingRowV1"

    def __post_init__(self) -> None:
        complete = set(self.complete_candidate_ids)
        partitions = {
            "target": set(self.target_candidate_ids),
            "acceptable": set(self.acceptable_candidate_ids),
            "supported": set(self.supported_candidate_ids),
            "unsupported": set(self.unsupported_candidate_ids),
            "unknown": set(self.unknown_candidate_ids),
        }
        if len(complete) != len(self.complete_candidate_ids):
            raise ValueError("complete candidate IDs must be unique")
        if any(not values <= complete for values in partitions.values()):
            raise ValueError("proposal labels must be members of the complete candidate set")
        if not partitions["target"] <= partitions["acceptable"]:
            raise ValueError("target edits must be acceptable")
        if not partitions["acceptable"] <= partitions["supported"]:
            raise ValueError("acceptable edits must be supported")
        if partitions["unknown"] & partitions["unsupported"]:
            raise ValueError("UNKNOWN candidates cannot be hard negatives")
        if set(dict(self.candidate_feature_digests)) != complete:
            raise ValueError("every complete candidate needs exactly one feature digest")
        if self.split not in {"train", "dev", "test"}:
            raise ValueError("split must be train, dev, or test")

    @property
    def digest(self) -> str:
        return _digest(
            (
                self.schema,
                self.row_id,
                self.state_fingerprint,
                *self.complete_candidate_ids,
                *self.target_candidate_ids,
                *self.acceptable_candidate_ids,
                *self.unknown_candidate_ids,
                self.split,
                self.lineage_digest,
                self.config_digest,
                self.bridge_version,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "row_id": self.row_id,
            "state_fingerprint": self.state_fingerprint,
            "hole_id": self.hole_id,
            "complete_candidate_ids": list(self.complete_candidate_ids),
            "target_candidate_ids": list(self.target_candidate_ids),
            "acceptable_candidate_ids": list(self.acceptable_candidate_ids),
            "supported_candidate_ids": list(self.supported_candidate_ids),
            "unsupported_candidate_ids": list(self.unsupported_candidate_ids),
            "unknown_candidate_ids": list(self.unknown_candidate_ids),
            "candidate_feature_digests": dict(self.candidate_feature_digests),
            "split": self.split,
            "lineage_digest": self.lineage_digest,
            "checkpoint_digest": self.checkpoint_digest,
            "config_digest": self.config_digest,
            "bridge_version": self.bridge_version,
            "digest": self.digest,
            "contains_final_source": False,
            "contains_future_witness_text": False,
        }


@dataclass(frozen=True)
class CandidateProposalDecision:
    """One deterministic scheduling decision with exact fallback metadata."""

    policy_name: str
    state_fingerprint: str
    complete_candidate_ids: tuple[str, ...]
    proposed_candidate_ids: tuple[str, ...]
    scheduled_candidate_ids: tuple[str, ...]
    scores: tuple[tuple[str, float], ...]
    calibrated_coverage_probability: float
    fallback_required: bool
    fallback_reason: str | None

    @property
    def exact_membership_preserved(self) -> bool:
        return set(self.scheduled_candidate_ids) == set(self.complete_candidate_ids)


ScoreFunction = Callable[[str, CandidateFeatureObject], float]


@dataclass(frozen=True)
class CandidateProposalPolicy:
    """Rank a cheap dynamic candidate superset without defining legality."""

    name: str
    k: int | None
    coverage_threshold: float = 0.95
    score_threshold: float | None = None
    mandatory_fallback: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("policy name is required")
        if self.k is not None and self.k < 1:
            raise ValueError("k must be positive or None for all")
        if not 0.0 <= self.coverage_threshold <= 1.0:
            raise ValueError("coverage threshold must be in [0, 1]")

    def propose(
        self,
        *,
        state_fingerprint: str,
        candidates: Sequence[CandidateFeatureObject],
        score: ScoreFunction,
        calibrated_coverage: Mapping[int, float] | None = None,
    ) -> CandidateProposalDecision:
        if not state_fingerprint:
            raise ValueError("exact state fingerprint is required")
        ordered = tuple(sorted(candidates, key=lambda item: item.candidate_id))
        ids = tuple(item.candidate_id for item in ordered)
        if len(set(ids)) != len(ids):
            raise ValueError("candidate IDs must be unique")
        scored: list[tuple[CandidateFeatureObject, float]] = []
        corrupt = False
        for item in ordered:
            value = float(score(state_fingerprint, item))
            if not math.isfinite(value):
                value = float("-inf")
                corrupt = True
            scored.append((item, value))
        ranked = sorted(scored, key=lambda pair: (-pair[1], pair[0].candidate_id))
        limit = len(ranked) if self.k is None else min(self.k, len(ranked))
        proposed = [
            item.candidate_id
            for item, value in ranked[:limit]
            if self.score_threshold is None or value >= self.score_threshold
        ]
        coverage = (
            float((calibrated_coverage or {}).get(limit, 0.0))
            if limit < len(ranked)
            else 1.0
        )
        coverage = max(0.0, min(1.0, coverage))
        reason: str | None = None
        if corrupt:
            reason = "non_finite_score"
        elif len(proposed) < len(ids) and coverage < self.coverage_threshold:
            reason = "coverage_below_threshold"
        elif not proposed and ids:
            reason = "empty_prefix"
        fallback = bool(reason) or (self.mandatory_fallback and len(proposed) < len(ids))
        if fallback and reason is None:
            reason = "mandatory_exact_fallback"
        proposed_set = set(proposed)
        remaining = [candidate_id for candidate_id in ids if candidate_id not in proposed_set]
        scheduled = tuple(proposed + remaining if fallback else proposed)
        return CandidateProposalDecision(
            policy_name=self.name,
            state_fingerprint=state_fingerprint,
            complete_candidate_ids=ids,
            proposed_candidate_ids=tuple(proposed),
            scheduled_candidate_ids=scheduled,
            scores=tuple((item.candidate_id, value) for item, value in ranked),
            calibrated_coverage_probability=coverage,
            fallback_required=fallback,
            fallback_reason=reason,
        )
