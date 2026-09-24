"""Fair resumable shard scheduling with inherited finite retry charges."""

from pathlib import Path

from slm_training.levers import KILL_GRACE_SECONDS


def run_shards(state, root, directory, budget, persist, *, allowance, run_workload) -> None:
    shards = state["shards"]
    cursor = state.get("shard_cursor", 0)
    completed = set(state["passed_nodes"])
    for _ in range(len(shards)):
        index = cursor % len(shards)
        nodes = shards[index]
        if set(nodes) <= completed:
            cursor = index + 1
            continue
        available = budget()
        if available <= 2 * KILL_GRACE_SECONDS:
            # Resume at the next unfinished shard, not after scanning the
            # entire exhausted tail and wrapping to the same starting point.
            state["shard_cursor"] = index
            allowance(state, "shard", nodes, available)
            persist()
            break
        seconds = allowance(state, "shard", nodes, available)
        if not seconds:
            cursor = index + 1
            state["shard_cursor"] = cursor % len(state["shards"])
            continue
        record = run_workload(
            root,
            nodes,
            collect_only=False,
            seconds=seconds,
            directory=directory,
            isolated=state["binding"]["isolation_enforced"],
            runtimes=tuple(Path(path) for path in state["binding"]["runtime_roots"]),
        )
        state["attempts"].append({**record, "kind": "shard"})
        cursor = index + 1
        if record["status"] == "ok":
            state["passed_nodes"].extend(nodes)
            completed.update(nodes)
        else:
            allowance(state, "shard", nodes, budget())
        if (
            record["status"] == "timeout"
            and len(nodes) > 1
            and not attempts_exhausted(state, "shard", nodes)
        ):
            midpoint = len(nodes) // 2
            state["shards"][index : index + 1] = [nodes[:midpoint], nodes[midpoint:]]
            cursor = index + 2
        state["shard_cursor"] = cursor % len(state["shards"])
        persist()


def attempts_exhausted(state: dict, kind: str, nodes: list[str]) -> bool:
    attempts = sum(
        row.get("kind") == kind
        and (
            set(nodes) <= set(row["nodes"])
            if kind == "collection"
            else row["nodes"] == nodes
        )
        for row in state["attempts"]
    )
    return attempts >= state["binding"]["max_attempts_per_obligation"]
