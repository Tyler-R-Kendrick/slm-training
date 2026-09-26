"""Collection startup shares the remaining invocation budget, not a 20s cap."""

from copy import deepcopy

import pytest

from scripts import merge_verification as owner
from scripts.merge_verification_collection import collect


def _state():
    return {
        "binding": {
            "targets": ["tests/test_a.py", "tests/test_b.py"],
            "isolation_enforced": True,
            "runtime_roots": [],
            "max_attempts_per_obligation": 3,
        },
        "attempts": [],
        "workload_budget_seconds": 119.0,
        "shard_budget_seconds": 60.0,
    }


@pytest.mark.parametrize("full, available, succeeds", [(119, 60, True), (40, 60, True), (119, 60, False)])
def test_collection_uses_available_budget_and_preserves_timeout(tmp_path, full, available, succeeds):
    state = _state()
    state["workload_budget_seconds"] = full
    observed = []
    persisted = []

    def workload(root, targets, *, seconds, **kwargs):
        observed.append(seconds)
        # Preparation plus collection needs 26s, independent of batch size.
        ok = succeeds and seconds >= 26
        return {
            "status": "ok" if ok else "timeout",
            "seconds": 26 if ok else seconds,
            "nodes": [f"{target}::test_case" for target in targets] if ok else targets,
            "output_tail": '{"collected": 2, "exit_code": 0}',
        }

    complete = collect(
        state, tmp_path, tmp_path, lambda: available,
        lambda: persisted.append(deepcopy(state)),
        allowance=owner._allowance, run_workload=workload,
        plan_shards=lambda nodes, budget: [nodes],
    )
    assert observed == [min(full, available)]
    assert complete is succeeds
    assert len(state["attempts"]) == 1 and len(persisted) == 1
    if succeeds:
        assert state["collection_batch_index"] == 1
        assert state["shards"] == [state["nodes"]]
        assert len(state["collection_batches"]) == 1
    else:
        assert state["collection_batch_index"] == 0
        assert "nodes" not in state and "shards" not in state
        assert state["attempts"][0]["status"] == "timeout"
        assert [len(batch) for batch in state["collection_batches"]] == [1, 1]


@pytest.mark.parametrize("available, exhausted", [(-6, False), (0, False), (20, False), (60, True)])
def test_collection_tail_and_ancestor_exhaustion_cannot_launch(tmp_path, available, exhausted):
    state = _state()
    if exhausted:
        state["attempts"] = [
            {"kind": "collection", "nodes": [*state["binding"]["targets"], "tests/test_c.py"],
             "status": "timeout", "seconds": 60}
            for _ in range(3)
        ]
    attempts = deepcopy(state["attempts"])
    assert not collect(
        state, tmp_path, tmp_path, lambda: available, lambda: None,
        allowance=owner._allowance,
        run_workload=lambda *args, **kwargs: pytest.fail("inadmissible collection launched"),
        plan_shards=lambda *args: pytest.fail("incomplete collection certified"),
    )
    assert state["attempts"] == attempts
    assert state["collection_batch_index"] == 0 and "nodes" not in state
    waiting = next(iter(state["waiting"].values()))
    assert waiting["reason"] == ("retry_exhausted" if exhausted else "insufficient_budget")
    assert 0 < waiting["required_seconds"] <= state["workload_budget_seconds"]
