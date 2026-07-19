"""Regression tests for SDE3-01 deterministic suite sharding."""

from __future__ import annotations

import pytest

from slm_training.evals.suite_sharding import (
    AggregationSpec,
    ShardResult,
    aggregate_shard_payloads,
    assign_example_ids,
    split_by_assignment,
)


def test_assign_example_ids_is_complete_and_disjoint() -> None:
    ids = [f"ex{i}" for i in range(20)]
    assignment = assign_example_ids(ids, n_shards=4)
    assert assignment.n_shards == 4
    union = set()
    for shard_ids in assignment.example_ids:
        assert not (union & set(shard_ids))
        union |= set(shard_ids)
    assert sorted(union) == sorted(ids)


def test_assign_example_ids_is_deterministic() -> None:
    ids = [f"ex{i}" for i in range(50)]
    a = assign_example_ids(ids, n_shards=5, seed="x")
    b = assign_example_ids(ids, n_shards=5, seed="x")
    assert a.example_ids == b.example_ids


def test_assign_example_ids_rejects_zero_shards() -> None:
    with pytest.raises(ValueError):
        assign_example_ids(["a"], n_shards=0)


def test_assign_example_ids_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError):
        assign_example_ids(["a", "a"], n_shards=2)


def test_split_by_assignment_matches_attribute() -> None:
    class Item:
        def __init__(self, item_id: str) -> None:
            self.item_id = item_id

    items = [Item(f"ex{i}") for i in range(10)]
    assignment = assign_example_ids([i.item_id for i in items], n_shards=2)
    buckets = split_by_assignment(items, assignment, key="item_id")
    assert len(buckets) == 2
    assert sum(len(b) for b in buckets) == 10


def test_aggregate_concat_and_sum() -> None:
    results = [
        ShardResult(0, ("a", "b"), {"details": [1, 2], "total": 3}),
        ShardResult(1, ("c",), {"details": [3], "total": 4}),
    ]
    spec = AggregationSpec(sum_keys=("total",), concat_keys=("details",))
    agg = aggregate_shard_payloads(results, spec)
    assert agg["example_count"] == 3
    assert agg["total"] == 7
    assert agg["details"] == [1, 2, 3]


def test_aggregate_mean_ignores_missing() -> None:
    results = [
        ShardResult(0, ("a",), {"score": 1.0}),
        ShardResult(1, ("b",), {"score": 3.0}),
        ShardResult(2, ("c",), {"score": None}),
    ]
    spec = AggregationSpec(mean_keys=("score",))
    agg = aggregate_shard_payloads(results, spec)
    assert agg["score_mean"] == 2.0


def test_aggregate_rejects_overlapping_ids() -> None:
    results = [
        ShardResult(0, ("a", "b"), {}),
        ShardResult(1, ("b", "c"), {}),
    ]
    with pytest.raises(ValueError):
        aggregate_shard_payloads(results, AggregationSpec())


def test_empty_results_yield_empty_aggregate() -> None:
    agg = aggregate_shard_payloads([], AggregationSpec())
    assert agg["shard_count"] == 0
    assert agg["example_count"] == 0
