"""Evaluator-only permutation checks for typed operator-policy candidates.

The policy view intentionally hides opaque IDs, descriptor fingerprints, and
semantic action IDs.  This module keeps those identities *outside* model input
so a caller can compare a freshly re-enumerated permutation without creating a
feature channel or treating unequal PARTIAL membership as comparable.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from slm_training.dsl.operators.legal_set import LegalSetCoverage, OperatorLegalSetV1
from slm_training.dsl.operators.references import ReferenceTableV1

__all__ = [
    "OperatorPermutationAlignmentV1",
    "OperatorPermutationCacheV1",
    "OperatorPermutationComparisonV1",
    "OperatorPermutationReportV1",
    "align_semantic_scores",
    "build_operator_permutation_alignment",
    "compare_permuted_scores",
]


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _softmax(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        raise ValueError("scores must be non-empty")
    ceiling = max(values)
    weights = [math.exp(value - ceiling) for value in values]
    total = sum(weights)
    return tuple(weight / total for weight in weights)


@dataclass(frozen=True)
class OperatorPermutationAlignmentV1:
    """Evaluator-only identities for one fresh table/legal-set pair.

    ``semantic_action_ids`` and ``descriptor_fingerprints`` are deliberately
    absent from ``OperatorPolicyInputV1``.  They exist only to prove that two
    independently enumerated candidate sets have the same semantics before
    their model scores may be compared.
    """

    request_id: str
    state_digest: str
    branch_digest: str
    registry_fingerprint: str
    coverage: LegalSetCoverage
    semantic_action_ids: tuple[str, ...]
    descriptor_fingerprints: tuple[str, ...]
    schema: str = "operator_permutation_alignment/v1"

    def __post_init__(self) -> None:
        if not self.semantic_action_ids:
            raise ValueError("semantic_action_ids must be non-empty")
        if len(set(self.semantic_action_ids)) != len(self.semantic_action_ids):
            raise ValueError("semantic_action_ids must be unique")
        if len(set(self.descriptor_fingerprints)) != len(self.descriptor_fingerprints):
            raise ValueError("descriptor_fingerprints must be unique")

    @property
    def action_set_digest(self) -> str:
        return _digest(self.semantic_action_ids)


def build_operator_permutation_alignment(
    table: ReferenceTableV1, legal_set: OperatorLegalSetV1
) -> OperatorPermutationAlignmentV1:
    """Build external alignment metadata from a fresh enumeration only."""
    if legal_set.reference_table_fingerprint != table.fingerprint:
        raise ValueError("permutation.stale_reference_table")
    semantic_action_ids = tuple(
        sorted(action.semantic_id for action in legal_set.operator_actions)
    )
    if not semantic_action_ids:
        # Ordinary grammar actions are outside this typed-operator comparison.
        raise ValueError("permutation.no_operator_actions")
    return OperatorPermutationAlignmentV1(
        request_id=table.request_id,
        state_digest=table.state_digest,
        branch_digest=table.branch_digest,
        registry_fingerprint=legal_set.registry_fingerprint,
        coverage=legal_set.coverage,
        semantic_action_ids=semantic_action_ids,
        descriptor_fingerprints=tuple(
            sorted(entry.descriptor.fingerprint for entry in table.entries)
        ),
    )


class OperatorPermutationCacheV1:
    """Small in-process cache for safe, deterministic reference permutations."""

    def __init__(self) -> None:
        self._tables: dict[tuple[str, str, str, str, str, int], ReferenceTableV1] = {}
        self.hits = 0
        self.misses = 0

    def get_or_build(
        self,
        table: ReferenceTableV1,
        alignment: OperatorPermutationAlignmentV1,
        *,
        seed: int,
    ) -> ReferenceTableV1:
        """Cache only within the same request/state/branch/action-set boundary."""
        if (
            table.request_id != alignment.request_id
            or table.state_digest != alignment.state_digest
            or table.branch_digest != alignment.branch_digest
        ):
            raise ValueError("permutation.cross_boundary_cache")
        key = (
            alignment.request_id,
            alignment.state_digest,
            alignment.branch_digest,
            alignment.registry_fingerprint,
            alignment.action_set_digest,
            seed,
        )
        cached = self._tables.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        permuted = table.permuted(seed)
        self._tables[key] = permuted
        return permuted

    def evidence(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "entries": len(self._tables)}


def align_semantic_scores(
    semantic_action_ids: Sequence[str], scores: Sequence[float]
) -> dict[str, float]:
    """Associate evaluator-only semantic action IDs with raw model scores."""
    if len(semantic_action_ids) != len(scores):
        raise ValueError("permutation.score_count_mismatch")
    if len(set(semantic_action_ids)) != len(semantic_action_ids):
        raise ValueError("permutation.duplicate_semantic_action")
    return dict(zip(semantic_action_ids, map(float, scores)))


@dataclass(frozen=True)
class OperatorPermutationComparisonV1:
    """One compatible original/permuted score comparison."""

    comparable: bool
    top1_flip: bool
    accepted_set_mass_drift: float
    calibration_confidence_drift: float
    score_disagreement: float
    symmetric_kl: float | None
    js_divergence: float | None
    reason: str | None = None
    schema: str = "operator_permutation_comparison/v1"


def compare_permuted_scores(
    original: OperatorPermutationAlignmentV1,
    permuted: OperatorPermutationAlignmentV1,
    *,
    original_scores: Sequence[float],
    permuted_scores: Sequence[float],
    accepted_semantic_action_ids: Sequence[str],
) -> OperatorPermutationComparisonV1:
    """Compare scores only when the witnessed semantic membership is identical.

    In particular, unequal PARTIAL sets return ``comparable=False`` rather
    than applying a misleading KL/JS over different domains.
    """
    boundary = (
        original.request_id,
        original.state_digest,
        original.branch_digest,
        original.registry_fingerprint,
    )
    other_boundary = (
        permuted.request_id,
        permuted.state_digest,
        permuted.branch_digest,
        permuted.registry_fingerprint,
    )
    same_membership = set(original.semantic_action_ids) == set(
        permuted.semantic_action_ids
    )
    if boundary != other_boundary:
        return OperatorPermutationComparisonV1(
            False, False, 0.0, 0.0, 0.0, None, None, "cross_boundary"
        )
    if original.coverage is not permuted.coverage or not same_membership:
        return OperatorPermutationComparisonV1(
            False, False, 0.0, 0.0, 0.0, None, None, "unequal_witnessed_membership"
        )
    accepted = set(accepted_semantic_action_ids)
    if not accepted or not accepted.issubset(original.semantic_action_ids):
        raise ValueError("permutation.accepted_set_outside_membership")
    first = align_semantic_scores(original.semantic_action_ids, original_scores)
    second = align_semantic_scores(permuted.semantic_action_ids, permuted_scores)
    keys = tuple(sorted(first))
    first_probs = _softmax([first[key] for key in keys])
    second_probs = _softmax([second[key] for key in keys])
    first_by_key = dict(zip(keys, first_probs))
    second_by_key = dict(zip(keys, second_probs))
    mean = [(left + right) / 2.0 for left, right in zip(first_probs, second_probs)]
    kl_first = sum(
        left * math.log(left / middle) for left, middle in zip(first_probs, mean)
    )
    kl_second = sum(
        right * math.log(right / middle) for right, middle in zip(second_probs, mean)
    )
    direct_kl_first = sum(
        left * math.log(left / right) for left, right in zip(first_probs, second_probs)
    )
    direct_kl_second = sum(
        right * math.log(right / left) for left, right in zip(first_probs, second_probs)
    )
    accepted_first = sum(first_by_key[key] for key in accepted)
    accepted_second = sum(second_by_key[key] for key in accepted)
    return OperatorPermutationComparisonV1(
        comparable=True,
        top1_flip=max(first, key=first.get) != max(second, key=second.get),
        accepted_set_mass_drift=abs(accepted_first - accepted_second),
        calibration_confidence_drift=abs(max(first_probs) - max(second_probs)),
        score_disagreement=sum(
            abs(left - right) for left, right in zip(first_probs, second_probs)
        )
        / len(keys),
        symmetric_kl=(direct_kl_first + direct_kl_second) / 2.0,
        js_divergence=(kl_first + kl_second) / 2.0,
    )


@dataclass(frozen=True)
class OperatorPermutationReportV1:
    """Compact evaluator-side probe summary; never a model input or ship claim."""

    comparisons: tuple[OperatorPermutationComparisonV1, ...]
    cache: Mapping[str, int]
    schema: str = "operator_permutation_report/v1"

    def to_dict(self) -> dict[str, object]:
        comparable = [item for item in self.comparisons if item.comparable]
        n = len(comparable)
        return {
            "schema": self.schema,
            "comparisons": len(self.comparisons),
            "comparable": n,
            "top1_flips": sum(item.top1_flip for item in comparable),
            "accepted_set_mass_drift": sum(
                item.accepted_set_mass_drift for item in comparable
            )
            / n
            if n
            else None,
            "calibration_confidence_drift": sum(
                item.calibration_confidence_drift for item in comparable
            )
            / n
            if n
            else None,
            "score_disagreement": sum(item.score_disagreement for item in comparable)
            / n
            if n
            else None,
            "cache": dict(self.cache),
            "incomparable_reasons": sorted(
                {item.reason for item in self.comparisons if item.reason}
            ),
        }
