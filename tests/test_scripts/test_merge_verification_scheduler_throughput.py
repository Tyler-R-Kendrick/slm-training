"""Large suites get small replayable shards and duration-limited slices."""

from scripts import merge_verification as owner


def test_large_source_suite_is_split_into_bounded_initial_shards(monkeypatch):
    monkeypatch.setattr(owner.check_changed, "_test_file_durations", lambda: {})
    nodes = [f"tests/test_{n % 300}.py::test_{n}" for n in range(11_979)]

    shards = owner.plan_shards(nodes, 170)

    assert len(shards) == 512
    assert max(map(len, shards)) <= 24
    assert sorted(node for shard in shards for node in shard) == sorted(nodes)


def test_shard_allowance_uses_estimate_instead_of_all_available_time(monkeypatch):
    node = "tests/test_slow.py::test_one"
    state = {
        "binding": {"max_attempts_per_obligation": 3},
        "nodes": [node],
        "attempts": [{"kind": "collection", "seconds": 2.0}],
        "workload_budget_seconds": 100.0,
    }
    monkeypatch.setattr(
        owner.check_changed, "_test_file_durations", lambda: {"tests/test_slow.py": 7.0}
    )

    allowance = owner._allowance(state, "shard", [node], 80.0)

    assert allowance == 16.0


def test_unmeasured_shard_gets_small_bounded_bootstrap_slice(monkeypatch):
    node = "tests/new_test.py::test_one"
    state = {
        "binding": {"max_attempts_per_obligation": 3},
        "nodes": [node],
        "attempts": [{"kind": "collection", "seconds": 2.0}],
        "workload_budget_seconds": 100.0,
        "shard_budget_seconds": 80.0,
    }
    monkeypatch.setattr(owner.check_changed, "_test_file_durations", lambda: {})

    assert owner._allowance(state, "shard", [node], 70.0) == 15.0
    assert owner._allowance(state, "shard", [node], 12.0) == 0.0
