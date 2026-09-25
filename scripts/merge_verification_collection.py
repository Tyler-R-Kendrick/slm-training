"""Bounded, exact test-node collection for the merge verifier."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from scripts.merge_verification_evidence import digest

COLLECTION_BATCH_SIZE = 64
# Isolated pytest startup alone measured 11.5s; keep admission above that cost.
COLLECTION_MIN_ATTEMPT_SECONDS = 20.0

def _finish_if_ready(state, batches, plan_shards) -> bool:
    if state.setdefault("collection_batch_index", 0) < len(batches):
        return False
    nodes = state.get("collection_nodes", [])
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("collection batches did not produce an exact node set")
    state["nodes"] = nodes
    state["shards"] = plan_shards(nodes, state["shard_budget_seconds"])
    return True


def _split_timeout(state, index, targets, status):
    if status == "timeout" and len(targets) > 1:
        midpoint = len(targets) // 2
        state["collection_batches"][index : index + 1] = (
            targets[:midpoint],
            targets[midpoint:],
        )


def collect(
    state: dict,
    root,
    directory,
    budget: Callable[[], float],
    persist: Callable[[], None],
    *,
    allowance,
    run_workload,
    plan_shards,
) -> bool:
    if "nodes" in state:
        return True
    targets = []
    for target in state["binding"]["targets"]:
        path = root / target.split("::", 1)[0]
        if path.is_dir():
            targets.extend(
                str(child.relative_to(root))
                for child in sorted(path.rglob("*.py"))
                if child.is_file() and (
                    child.name.startswith("test_") or child.name.endswith("_test.py")
                )
            )
        else:
            targets.append(target)
    batches = state.setdefault(
        "collection_batches",
        [
            targets[i : i + COLLECTION_BATCH_SIZE]
            for i in range(0, len(targets), COLLECTION_BATCH_SIZE)
        ],
    )
    if _finish_if_ready(state, batches, plan_shards):
        return True
    index = state["collection_batch_index"]
    targets = batches[index]
    seconds = allowance(state, "collection", targets, budget())
    if not seconds:
        return False
    record = run_workload(
        root,
        targets,
        collect_only=True,
        seconds=seconds,
        directory=directory,
        isolated=state["binding"]["isolation_enforced"],
        runtimes=tuple(Path(path) for path in state["binding"]["runtime_roots"]),
    )
    state["attempts"].append({**record, "kind": "collection"})
    _split_timeout(state, index, targets, record["status"])
    if record["status"] != "ok":
        allowance(state, "collection", targets, budget())
    if record["status"] == "ok":
        collected = state.setdefault("collection_nodes", [])
        collected.extend(record["nodes"])
        if len(collected) != len(set(collected)):
            raise ValueError("collection batches overlap")
        state["collection_batch_index"] = index + 1
        if state["collection_batch_index"] == len(batches):
            state["nodes"] = collected
            state["shards"] = plan_shards(collected, state["shard_budget_seconds"])
        else:
            state.setdefault("waiting", {})[
                digest(["collection", batches[index + 1]])
            ] = {
                "kind": "collection",
                "target_digest": digest(batches[index + 1]),
                "reason": "insufficient_budget",
                # The next invocation only needs the next batch allowance;
                # advertising the whole remaining collection makes a
                # resumable collection appear permanently unaffordable.
                "required_seconds": min(
                    COLLECTION_MIN_ATTEMPT_SECONDS,
                    state.get("workload_budget_seconds", COLLECTION_MIN_ATTEMPT_SECONDS),
                ),
                "available_seconds": 0.0,
                "wake_source": "fresh_bounded_invocation",
            }
    persist()
    return record["status"] == "ok" and "nodes" in state
