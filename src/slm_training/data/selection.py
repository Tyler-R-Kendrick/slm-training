"""DEITA-style build-time curation scoring: quality × diversity.

Each admitted record gets a ``curation_score`` in [0, 1] combining its
``assess_record`` quality score with a diversity weight that discounts
members of large semantic clusters (inverse-sqrt cluster size, the same
``semantic_cluster_key`` used by the exposure caps). The score is stamped
into ``meta`` so the web browser can sort by it and ``--derive-from``
curation jobs can filter on it.

A model-difficulty factor (Superfiltering-style NLL percentile from the
current checkpoint) is a deliberate follow-up: it requires a trained model
at build time, which the data build must not depend on by default.
"""

from __future__ import annotations

import math

from slm_training.data.dedup import semantic_cluster_key
from slm_training.dsl.schema import ExampleRecord

CURATION_SCORE_VERSION = 1


def curation_scores(records: list[ExampleRecord]) -> dict[str, float]:
    """Per-record ``quality × 1/sqrt(cluster_size)`` scores, keyed by id."""
    cluster_sizes: dict[tuple[str, str, str], int] = {}
    keys: dict[str, tuple[str, str, str]] = {}
    for record in records:
        key = semantic_cluster_key(record)
        keys[record.id] = key
        cluster_sizes[key] = cluster_sizes.get(key, 0) + 1
    scores: dict[str, float] = {}
    for record in records:
        quality = float((record.meta or {}).get("quality", {}).get("score") or 0.0)
        diversity = 1.0 / math.sqrt(cluster_sizes[keys[record.id]])
        scores[record.id] = round(max(0.0, min(1.0, quality)) * diversity, 4)
    return scores


def attach_curation_scores(records: list[ExampleRecord]) -> list[ExampleRecord]:
    """Stamp ``meta.curation_score`` (+ version) onto every record in place."""
    scores = curation_scores(records)
    for record in records:
        meta = dict(record.meta or {})
        meta["curation_score"] = scores[record.id]
        meta["curation_score_version"] = CURATION_SCORE_VERSION
        record.meta = meta
    return records


__all__ = ["CURATION_SCORE_VERSION", "attach_curation_scores", "curation_scores"]
