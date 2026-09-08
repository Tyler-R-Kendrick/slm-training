"""Bounded verification status projection; full proof stays in the signed journal."""

from collections import Counter

from scripts.merge_verification_evidence import digest


def summarize(state: dict, *, reason: str = "", full: bool = False) -> dict:
    current = list(state["static"].values()) + state["attempts"]
    records = state.get("static_history", []) + current
    completed = set(state["passed_nodes"])
    nodes = state.get("nodes", [])
    pending = sorted(set(nodes) - completed)
    static_names = [row[0] for row in state["binding"]["static_commands"]]
    statics_ok = all(
        state["static"].get(name, {}).get("status") == "ok" for name in static_names
    )
    complete = bool(nodes) and set(nodes) == completed and statics_ok and not reason
    progress = phase_progress(state)
    action = next_action(state, complete=complete, reason=reason)
    fields = ("name", "kind", "status", "exit_code", "seconds", "reason")
    waits = list(state.get("waiting", {}).values())
    result = {
        "schema": "merge_verification/v2",
        "identity": state["identity"],
        "binding_sha256": digest(state["binding"]),
        "verification_complete": complete,
        "release_authorized": False,
        "evidence_class": "isolated_process"
        if state["binding"].get("isolation_enforced")
        else "local_process",
        "status": action["status"],
        "phase_progress": progress,
        "next_action": action,
        "required_seconds": action["required_seconds"],
        "reason": reason,
        "node_counts": {
            "required": len(nodes),
            "passed": len(completed),
            "pending": len(pending),
        },
        "pending_sample": pending[:10],
        "shard_counts": {
            "required": len(state.get("shards", [])),
            "pending": sum(
                not set(batch) <= completed for batch in state.get("shards", [])
            ),
        },
        "steps": [
            {key: row[key] for key in fields if key in row} for row in records[-20:]
        ],
        "attempt_count": len(records),
        "waiting_count": len(waits),
        "waiting_reasons": dict(Counter(row["reason"] for row in waits)),
        "waiting_sample": waits[:10],
        "journal_sha256": digest(state),
        "spent_seconds": sum(record.get("seconds", 0) for record in records),
        "rule": "local evidence needs independent isolated verification before autonomous release",
    }
    if full:
        result.update(
            binding=state["binding"],
            nodes=nodes,
            completed_nodes=sorted(completed),
            pending_nodes=pending,
            steps=records,
        )
    return result


def phase_progress(state):
    """Counts of verified obligations, not receipt/log activity."""
    names = [row[0] for row in state["binding"]["static_commands"]]
    collected = bool(state.get("nodes"))
    counts = {
        "static": (
            len(names),
            sum(state["static"].get(n, {}).get("status") == "ok" for n in names),
        ),
        "collection": (1, int(collected)),
        "tests": (
            len(state.get("nodes", [])) if collected else None,
            len(state["passed_nodes"]),
        ),
    }
    return {
        kind: {
            "required": required,
            "completed": completed,
            "pending": required - completed if required is not None else None,
        }
        for kind, (required, completed) in counts.items()
    }


def outstanding(state):
    for name, _ in state["binding"]["static_commands"]:
        if state["static"].get(name, {}).get("status") != "ok":
            yield "static", [name]
    if not state.get("nodes"):
        yield "collection", state["binding"].get("targets", [])
    passed = set(state["passed_nodes"])
    for nodes in state.get("shards", []):
        if not set(nodes) <= passed:
            yield "shard", nodes


def next_action(state, *, complete=False, reason=""):
    waits = [
        state.get("waiting", {}).get(digest([kind, targets]), {})
        for kind, targets in outstanding(state)
    ]
    resumable = [
        row
        for row in waits
        if not row or row.get("wake_source") == "fresh_bounded_invocation"
    ]
    kinds = {row.get("reason") for row in waits}
    status, action = "pending", "resume_verification"
    if complete:
        status, action = "complete", "none"
    elif reason or not waits:
        status, action = "invalid_evidence", "reconcile_verification_contract"
    elif not resumable:
        status, action = {
            frozenset({"retry_exhausted"}): (
                "waiting_repair",
                "dispatch_verification_repair",
            ),
            frozenset({"insufficient_budget"}): (
                "waiting_budget",
                "replan_verification_budget",
            ),
        }.get(
            frozenset(kinds), ("waiting_prerequisite", "dispatch_verification_remedies")
        )
    selected = resumable or waits
    requirements = [
        row.get("required_seconds", state.get("workload_budget_seconds"))
        for row in selected
    ]
    required = (
        min(requirements)
        if requirements and all(x is not None for x in requirements)
        else None
    )
    return {
        "status": status,
        "kind": action,
        "required_seconds": required,
        "units": "workload_interrupt_seconds",
        "resume_in_new_invocation": bool(resumable),
        "wake_source": "fresh_bounded_invocation" if resumable else action,
        "budget_semantics": "minimum next eligible workload; excludes controller and kill/finalization overhead",
    }
