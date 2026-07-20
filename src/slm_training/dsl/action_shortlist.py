"""SLM-176: description-based retrieve-then-rerank over live legal action sets.

Default-off wiring/fixture harness.  The retrieval encoder is a deterministic
hash surrogate (`FixtureDescriptionEncoder`); production query vectors would
come from a learned controller state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from slm_training.dsl.action_descriptions import (
    ActionDescriptionCatalog,
    FixtureDescriptionEncoder,
)

__all__ = [
    "ActionShortlistPolicy",
    "ActionShortlistTrace",
    "retrieve_then_rerank",
    "build_query_vector",
]


@dataclass(frozen=True)
class ActionShortlistPolicy:
    """Description-retrieval shortlist policy for compiler legal action sets."""

    mode: str = "off"  # off | description_retrieval
    k: int = 8
    min_legal_size: int = 16
    score_margin: float = 0.0
    fallback_policy: str = "confidence_and_coverage"
    shadow_full_score: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"off", "description_retrieval"}:
            raise ValueError(f"unknown action_shortlist mode: {self.mode!r}")
        if self.k < 0:
            raise ValueError("k must be non-negative")
        if self.min_legal_size < 0:
            raise ValueError("min_legal_size must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionShortlistPolicy":
        return cls(
            mode=str(data.get("mode", "off")),
            k=int(data.get("k", 8)),
            min_legal_size=int(data.get("min_legal_size", 16)),
            score_margin=float(data.get("score_margin", 0.0)),
            fallback_policy=str(data.get("fallback_policy", "confidence_and_coverage")),
            shadow_full_score=bool(data.get("shadow_full_score", False)),
        )


@dataclass(frozen=True)
class ActionShortlistTrace:
    """One shortlist decision recorded during constrained decode."""

    legal_action_ids: tuple[str, ...]
    shortlist_ids: tuple[str, ...]
    retrieval_scores: dict[str, float]
    fallback_reason: str | None
    shadow_full_selected_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "legal_action_ids": list(self.legal_action_ids),
            "shortlist_ids": list(self.shortlist_ids),
            "retrieval_scores": {k: float(v) for k, v in self.retrieval_scores.items()},
            "fallback_reason": self.fallback_reason,
            "shadow_full_selected_id": self.shadow_full_selected_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionShortlistTrace":
        return cls(
            legal_action_ids=tuple(data.get("legal_action_ids", [])),
            shortlist_ids=tuple(data.get("shortlist_ids", [])),
            retrieval_scores={
                str(k): float(v) for k, v in data.get("retrieval_scores", {}).items()
            },
            fallback_reason=data.get("fallback_reason"),
            shadow_full_selected_id=data.get("shadow_full_selected_id"),
        )


def build_query_vector(
    state_context: str,
    action_catalog: ActionDescriptionCatalog,
    encoder: FixtureDescriptionEncoder,
) -> torch.Tensor:
    """Return a deterministic fixture query vector for ``state_context``.

    The query is a SHA-256 projection over a canonical state-context string.
    This is a wiring placeholder; a production system would derive the query
    from the learned controller hidden state.
    """
    _ = action_catalog  # reserved for future catalog-aware query normalization
    canonical = _canonical_state_string(state_context)
    return encoder.encode(canonical)


def _canonical_state_string(state_context: str) -> str:
    """Collapse whitespace and hash to a canonical state fingerprint."""
    return " ".join(str(state_context).split())


def _sha256_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def retrieve_then_rerank(
    legal_action_ids: tuple[str, ...],
    query_vector: torch.Tensor,
    action_vectors: Mapping[str, torch.Tensor],
    policy: ActionShortlistPolicy,
    mandatory_ids: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], dict[str, float], str | None]:
    """Return a shortlist subset of ``legal_action_ids`` plus per-id scores.

    Scoring is a dot product over only the legal action ids.  The returned
    shortlist includes the top-``k`` ids, all ``mandatory_ids``, and any ids
    within ``policy.score_margin`` of the ``k``-th score.  If the legal set is
    smaller than ``policy.min_legal_size`` or a simple confidence/entropy check
    fails, the full legal set is returned and a fallback reason is provided.
    """
    if policy.mode == "off":
        return legal_action_ids, {}, "mode_off"

    if len(legal_action_ids) < policy.min_legal_size:
        return legal_action_ids, {}, "legal_set_below_min_legal_size"

    present = {aid for aid in legal_action_ids if aid in action_vectors}
    if not present:
        return legal_action_ids, {}, "no_action_vectors_for_legal_set"

    ids = sorted(present)
    matrix = torch.stack([action_vectors[aid] for aid in ids])
    q = query_vector.to(dtype=matrix.dtype, device=matrix.device)
    scores_tensor = torch.matmul(matrix, q)
    scores: dict[str, float] = {
        aid: float(scores_tensor[i].item()) for i, aid in enumerate(ids)
    }

    # Simple confidence/entropy fallback: if the max score is barely above the
    # mean, the retrieval distribution is too flat to be trustworthy.
    score_values = torch.tensor(list(scores.values()), dtype=torch.float32)
    if score_values.numel() > 0:
        max_score = float(score_values.max().item())
        mean_score = float(score_values.mean().item())
        if max_score - mean_score < 0.01:
            return legal_action_ids, scores, "flat_score_distribution"

    # Sort descending by retrieval score.
    ranked = sorted(ids, key=lambda aid: scores[aid], reverse=True)
    k = max(0, min(policy.k, len(ranked)))
    if k == 0:
        # With k=0 we still honor mandatory ids and margin ties.
        kth_score = float("inf")
    else:
        kth_score = scores[ranked[k - 1]]

    selected = set(ranked[:k])
    selected.update(str(m) for m in mandatory_ids if m in present)

    if policy.score_margin > 0.0:
        lower_bound = kth_score - policy.score_margin
        for aid in ids:
            if scores[aid] >= lower_bound:
                selected.add(aid)

    shortlist = tuple(aid for aid in legal_action_ids if aid in selected)
    if not shortlist:
        return legal_action_ids, scores, "shortlist_empty"

    return shortlist, scores, None


def _json_fingerprint(obj: object) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()[:32]
