"""DEITA-style build-time curation scoring: quality × diversity × difficulty.

Each admitted record gets a ``curation_score`` in [0, 1] combining its
``assess_record`` quality score with a diversity weight that discounts
members of large semantic clusters (inverse-sqrt cluster size, the same
``semantic_cluster_key`` used by the exposure caps). The score is stamped
into ``meta`` so the web browser can sort by it and ``--derive-from``
curation jobs can filter on it.

The optional Superfiltering-style difficulty factor consumes
``record_nll.jsonl`` written by a trained run (`train_model
--emit-record-nll`): trivially easy records (lowest NLL percentiles —
likely memorized or near-duplicated) are discounted, while mid/high
difficulty keeps full weight. Data builds never load a model themselves;
the NLL file is the interface between the train and data phases.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from slm_training.data.dedup import semantic_cluster_key
from slm_training.dsl.schema import ExampleRecord

CURATION_SCORE_VERSION = 2
# Easy-tail discount: records below this NLL percentile are down-weighted
# linearly (floor 0.5 at percentile 0); everything harder keeps weight 1.0.
EASY_PERCENTILE = 0.2


def load_record_nll(path: Path | str) -> dict[str, float]:
    """Read a run's record_nll.jsonl into {record_id: nll} (None rows skipped)."""
    scores: dict[str, float] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("nll") is None:
            continue
        scores[str(row["id"])] = float(row["nll"])
    return scores


def difficulty_weights(
    records: list[ExampleRecord], nll_by_id: dict[str, float]
) -> dict[str, float]:
    """Per-record weight from NLL percentile; 1.0 when no NLL is known."""
    known = sorted(
        nll_by_id[record.id] for record in records if record.id in nll_by_id
    )
    weights: dict[str, float] = {}
    for record in records:
        nll = nll_by_id.get(record.id)
        if nll is None or not known:
            weights[record.id] = 1.0
            continue
        rank = sum(1 for value in known if value <= nll)
        percentile = rank / len(known)
        if percentile >= EASY_PERCENTILE:
            weights[record.id] = 1.0
        else:
            weights[record.id] = round(0.5 + 0.5 * (percentile / EASY_PERCENTILE), 4)
    return weights


def curation_scores(
    records: list[ExampleRecord],
    *,
    nll_by_id: dict[str, float] | None = None,
) -> dict[str, float]:
    """Per-record ``quality × diversity × difficulty`` scores, keyed by id."""
    cluster_sizes: dict[tuple[str, str, str], int] = {}
    keys: dict[str, tuple[str, str, str]] = {}
    for record in records:
        key = semantic_cluster_key(record)
        keys[record.id] = key
        cluster_sizes[key] = cluster_sizes.get(key, 0) + 1
    difficulty = (
        difficulty_weights(records, nll_by_id)
        if nll_by_id
        else {record.id: 1.0 for record in records}
    )
    scores: dict[str, float] = {}
    for record in records:
        quality = float((record.meta or {}).get("quality", {}).get("score") or 0.0)
        diversity = 1.0 / math.sqrt(cluster_sizes[keys[record.id]])
        scores[record.id] = round(
            max(0.0, min(1.0, quality)) * diversity * difficulty[record.id], 4
        )
    return scores


def attach_curation_scores(
    records: list[ExampleRecord],
    *,
    nll_by_id: dict[str, float] | None = None,
) -> list[ExampleRecord]:
    """Stamp ``meta.curation_score`` (+ version, + difficulty evidence) in place."""
    scores = curation_scores(records, nll_by_id=nll_by_id)
    difficulty = (
        difficulty_weights(records, nll_by_id)
        if nll_by_id
        else None
    )
    for record in records:
        meta = dict(record.meta or {})
        meta["curation_score"] = scores[record.id]
        meta["curation_score_version"] = CURATION_SCORE_VERSION
        if difficulty is not None:
            meta["difficulty_weight"] = difficulty[record.id]
            if nll_by_id and record.id in nll_by_id:
                meta["record_nll"] = nll_by_id[record.id]
        record.meta = meta
    return records


__all__ = [
    "CURATION_SCORE_VERSION",
    "EASY_PERCENTILE",
    "attach_curation_scores",
    "curation_scores",
    "difficulty_weights",
    "load_record_nll",
]
