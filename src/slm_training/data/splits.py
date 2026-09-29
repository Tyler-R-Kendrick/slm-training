"""Clustered train/validation splits keyed by structure fingerprint."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from slm_training.data.leakage import fingerprint_openui, fingerprint_openui_structure
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


# Same family contract for certified sampling and actual readiness admission.
_FAMILY_LINK_KEYS = ("root_parent_id", "split_group_id", "parent_id")


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, key: str) -> str:
        parent = self._parent
        parent.setdefault(key, key)
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != root:
            parent[key], key = root, parent[key]
        return root

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            # Deterministic: the lexicographically smaller root survives.
            if root_right < root_left:
                root_left, root_right = root_right, root_left
            self._parent[root_right] = root_left


def _family_links(record: ExampleRecord) -> list[str]:
    meta = record.meta or {}
    links = [record.id]
    for key in _FAMILY_LINK_KEYS:
        value = meta.get(key)
        if value:
            links.append(str(value))
    return links


def root_family_index(
    records: Iterable[ExampleRecord], *, close_under_program_text: bool = True
) -> dict[str, str]:
    """Map every record id to its canonical root-family id.

    Families are the connected components of the id / root_parent_id /
    split_group_id / parent_id link graph, additionally closed under identical
    normalized program text when ``close_under_program_text`` is set (the
    default; see harnesses.test_data.certified for the evidence). The canonical id is
    the smallest ``root_parent_id`` seen in the component, else the smallest
    record id.
    """

    rows = list(records)
    forest = _UnionFind()
    for record in rows:
        links = _family_links(record)
        for link in links[1:]:
            forest.union(links[0], link)
    if close_under_program_text:
        first_with_program: dict[str, str] = {}
        for record in rows:
            program = fingerprint_openui(record.openui)
            anchor = first_with_program.setdefault(program, record.id)
            if anchor != record.id:
                forest.union(anchor, record.id)
    roots: dict[str, set[str]] = defaultdict(set)
    ids: dict[str, set[str]] = defaultdict(set)
    for record in rows:
        component = forest.find(record.id)
        ids[component].add(record.id)
        root = (record.meta or {}).get("root_parent_id")
        if root:
            roots[component].add(str(root))
    canonical: dict[str, str] = {}
    for component, members in ids.items():
        family = min(roots[component]) if roots[component] else min(members)
        for member in members:
            canonical[member] = family
    return canonical


__all__ = [
    "root_family_index",
    "ClusteredSplit",
    "cluster_by_structure",
    "clustered_train_val_split",
    "structure_fingerprint",
]
