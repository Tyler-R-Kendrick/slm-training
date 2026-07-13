"""Clustered train/validation splits keyed by structure fingerprint."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from slm_training.data.leakage import fingerprint_openui_structure
from slm_training.dsl.schema import ExampleRecord


@dataclass(frozen=True)
class ClusteredSplit:
    train: tuple[ExampleRecord, ...]
    val: tuple[ExampleRecord, ...]
    clusters: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def val_fraction(self) -> float:
        total = len(self.train) + len(self.val)
        if total == 0:
            return 0.0
        return len(self.val) / total


def structure_fingerprint(openui: str) -> str:
    """Structural layout fingerprint (style + namespace normalized)."""
    return fingerprint_openui_structure(openui)


def cluster_by_structure(records: Iterable[ExampleRecord]) -> dict[str, list[ExampleRecord]]:
    """Group records by ``fingerprint_openui_structure``."""
    clusters: dict[str, list[ExampleRecord]] = defaultdict(list)
    for record in records:
        clusters[structure_fingerprint(record.openui)].append(record)
    return dict(clusters)


def clustered_train_val_split(
    records: list[ExampleRecord],
    *,
    val_fraction: float = 0.1,
    seed: int = 0,
    min_val_clusters: int = 1,
) -> ClusteredSplit:
    """
    Assign whole structure clusters to train or val.

    Keeps isomorphic layouts out of both splits to avoid structural leakage.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if not records:
        return ClusteredSplit(train=(), val=(), clusters=())

    clusters = cluster_by_structure(records)
    cluster_items = sorted(
        clusters.items(),
        key=lambda item: (len(item[1]), item[0]),
        reverse=True,
    )
    rng = random.Random(seed)

    target_val = max(min_val_clusters, int(round(len(records) * val_fraction)))
    target_val = min(target_val, len(records))

    shuffled = list(cluster_items)
    rng.shuffle(shuffled)

    val: list[ExampleRecord] = []
    train: list[ExampleRecord] = []
    val_clusters = 0

    for fp, group in shuffled:
        group_sorted = sorted(group, key=lambda r: r.id)
        if len(val) < target_val or (
            val_clusters < min_val_clusters and len(train) + len(group) > len(records) - 1
        ):
            val.extend(group_sorted)
            val_clusters += 1
        else:
            train.extend(group_sorted)

    if not val and train:
        last_fp, last_group = shuffled[-1]
        moved = sorted(last_group, key=lambda r: r.id)
        for item in moved:
            train.remove(item)
        val.extend(moved)
        val_clusters = 1
        _ = last_fp

    train.sort(key=lambda r: r.id)
    val.sort(key=lambda r: r.id)
    cluster_report = tuple(
        (fp, tuple(sorted(r.id for r in group)))
        for fp, group in sorted(clusters.items(), key=lambda x: x[0])
    )
    return ClusteredSplit(
        train=tuple(train),
        val=tuple(val),
        clusters=cluster_report,
    )


__all__ = [
    "ClusteredSplit",
    "cluster_by_structure",
    "clustered_train_val_split",
    "structure_fingerprint",
]
