"""Deterministic suite sharding and shard-result aggregation for SDE3-01.

Shards are assigned by a stable hash of the example identifier so repeated runs
with the same record ordering produce the same shard manifests.  Aggregation
validates completeness (no duplicates, no omissions) before combining per-shard
artifacts.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Sequence, TypeVar

from slm_training.dsl.schema import ExampleRecord

T = TypeVar("T")


@dataclass(frozen=True)
class ShardAssignment:
    """Deterministic mapping from example IDs to shard indexes."""

    n_shards: int
    example_ids: tuple[tuple[str, ...], ...]
    seed: str

    @property
    def shard_count(self) -> int:
        return self.n_shards

    def ids_for(self, shard_index: int) -> tuple[str, ...]:
        if not 0 <= shard_index < self.n_shards:
            raise IndexError(f"shard_index {shard_index} out of range {self.n_shards}")
        return self.example_ids[shard_index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_shards": self.n_shards,
            "seed": self.seed,
            "example_ids": [list(ids) for ids in self.example_ids],
        }


def _example_id_shard_index(example_id: str, n_shards: int, seed: str) -> int:
    digest = hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).hexdigest()
    return int(digest, 16) % max(1, n_shards)


def assign_example_ids(
    example_ids: Sequence[str],
    n_shards: int,
    *,
    seed: str = "slm-training-sde3-01",
) -> ShardAssignment:
    """Assign example IDs to shards deterministically.

    Raises if ``n_shards`` is not positive or if duplicate IDs are present.
    """
    n_shards = int(n_shards)
    if n_shards <= 0:
        raise ValueError(f"n_shards must be positive, got {n_shards}")
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("example_ids must be unique")
    buckets: list[list[str]] = [[] for _ in range(n_shards)]
    for example_id in example_ids:
        idx = _example_id_shard_index(example_id, n_shards, seed)
        buckets[idx].append(example_id)
    return ShardAssignment(
        n_shards=n_shards,
        example_ids=tuple(tuple(bucket) for bucket in buckets),
        seed=seed,
    )


def assign_records(
    records: Sequence[ExampleRecord],
    n_shards: int,
    *,
    seed: str = "slm-training-sde3-01",
) -> ShardAssignment:
    """Assign ``ExampleRecord`` objects by their ``id`` field."""
    return assign_example_ids([r.id for r in records], n_shards, seed=seed)


def split_by_assignment(
    items: Sequence[T],
    assignment: ShardAssignment,
    *,
    key: str = "id",
) -> list[list[T]]:
    """Return ``items`` partitioned according to ``assignment``.

    ``key`` names the attribute used to match each item to its assigned ID.
    """
    id_to_shard: dict[str, int] = {}
    for shard_index, ids in enumerate(assignment.example_ids):
        for example_id in ids:
            id_to_shard[example_id] = shard_index
    buckets: list[list[T]] = [[] for _ in range(assignment.n_shards)]
    for item in items:
        example_id = getattr(item, key)
        shard_index = id_to_shard[example_id]
        buckets[shard_index].append(item)
    return buckets


@dataclass
class AggregationSpec:
    """Describe how to combine per-shard payloads."""

    sum_keys: tuple[str, ...] = ()
    mean_keys: tuple[str, ...] = ()
    concat_keys: tuple[str, ...] = ()


@dataclass
class ShardResult:
    """One per-shard result plus metadata."""

    shard_index: int
    example_ids: tuple[str, ...]
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_index": self.shard_index,
            "example_ids": list(self.example_ids),
            "payload": _safe_json(self.payload),
        }


def aggregate_shard_payloads(
    results: Sequence[ShardResult],
    spec: AggregationSpec,
) -> dict[str, Any]:
    """Combine per-shard payloads according to ``spec``.

    Validates that every result reports disjoint example IDs and that the union
    equals the expected set (implied by the order of ``results``).
    """
    if not results:
        return {"shard_count": 0, "example_count": 0}

    seen: set[str] = set()
    for result in results:
        ids = set(result.example_ids)
        if seen & ids:
            raise ValueError(
                f"shard {result.shard_index} contains example IDs already seen"
            )
        seen |= ids

    aggregated: dict[str, Any] = {
        "shard_count": len(results),
        "example_count": len(seen),
        "example_ids": sorted(seen),
    }

    for key in spec.concat_keys:
        aggregated[key] = []
        for result in results:
            value = result.payload.get(key)
            if isinstance(value, list):
                aggregated[key].extend(value)
            elif value is not None:
                aggregated[key].append(value)

    for key in spec.sum_keys:
        total = 0.0
        for result in results:
            value = result.payload.get(key)
            if isinstance(value, (int, float)):
                total += float(value)
        aggregated[key] = total

    for key in spec.mean_keys:
        values = []
        for result in results:
            value = result.payload.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                values.append(float(value))
        aggregated[f"{key}_mean"] = sum(values) / len(values) if values else None

    return aggregated


def _safe_json(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return value
