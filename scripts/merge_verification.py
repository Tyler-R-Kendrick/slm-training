"""Resumable merge obligations; the signed cache is not release authority."""

from __future__ import annotations

import functools
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from scripts import check_changed
from scripts.merge_verification_evidence import (
    ReceiptCache,
    digest,
    environment_identity,
    file_digest,
    runtime_identity,
    source_identity,
    validate_cached_state,
)
from scripts.merge_verification_isolation import run_workload
from scripts.merge_verification_summary import summarize as _summary
from slm_training.levers import INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS

PYTEST_OPTIONS = ["-o", "addopts=", "-m", "", "-q", "-p", "no:cacheprovider"]
FINALIZATION_SECONDS = 10.0


def changed_paths(root: Path, base_ref: str) -> tuple[str, list[str]]:
    def git(*argv: str) -> str:
        return subprocess.run(
            ["git", *argv],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout

    base_tree = git("rev-parse", f"{base_ref}^{{tree}}").strip()
    paths = git("diff", "--name-only", "--no-renames", "-z", base_ref, "--")
    paths += git("ls-files", "--others", "--exclude-standard", "-z")
    return base_tree, sorted(set(paths.split("\0")) - {""})


def verification_binding(
    root: Path,
    base_ref: str,
    steps: tuple,
    *,
    isolated: bool = True,
    runtimes: tuple[Path, ...] = (),
) -> dict:
    base_tree, paths = changed_paths(root, base_ref)
    rule_paths = (
        "check_changed.py",
        "merge_verification.py",
        "merge_test_worker.py",
        "merge_verification_evidence.py",
        "merge_verification_isolation.py",
        "merge_verification_summary.py",
        "merge_verification_controller.py",
        "merge_verification_runtime.py",
        "verify_merge_ready.py",
    )
    return {
        "schema": "merge_verification_binding/v3",
        "base_tree": base_tree,
        "candidate_tree_sha256": source_identity(root),
        "changed_paths": paths,
        "targets": check_changed.select_tests(paths, root=root),
        "environment": environment_identity(),
        "pytest_options": PYTEST_OPTIONS,
        "selection_rules": {
            name: file_digest(Path(__file__).with_name(name)) for name in rule_paths
        },
        "static_commands": [
            [step.name, list(step.cmd)] for step in steps if step.static
        ],
        "run_cap": [INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS],
        "isolation_enforced": isolated,
        "runtime_roots": [str(path.resolve()) for path in runtimes],
        "runtime_identity": runtime_identity(runtimes),
        "max_attempts_per_obligation": 3,
    }


def plan_shards(nodes: list[str], budget_seconds: float) -> list[list[str]]:
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("cannot shard an empty or duplicate collection")
    table = check_changed._test_file_durations()
    counts = Counter(node.split("::", 1)[0] for node in nodes)
    total = sum(max(count, table.get(path, 0)) for path, count in counts.items())
    if not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("invalid shard budget")
    count = min(len(nodes), max(1, math.ceil(total * 2 / budget_seconds)))
    return [batch for batch in check_changed._shard_test_nodes(nodes, count) if batch]


def _execute_pending(
    state: dict,
    *,
    root: Path,
    cache: ReceiptCache,
    steps: tuple,
    run_step,
    deadline: float,
    step_seconds: float,
) -> dict:
    def budget() -> float:
        remaining = (
            deadline - time.monotonic() - KILL_GRACE_SECONDS - FINALIZATION_SECONDS
        )
        return min(step_seconds, remaining)

    # Unmeasured work gets a fresh invocation's actual allowance, never its tail.
    fresh = max(0.0, budget())
    state["workload_budget_seconds"] = fresh - min(1.0, fresh / 10)
    state.setdefault("shard_budget_seconds", max(1.0, state["workload_budget_seconds"]))
    state["waiting"] = {}

    def persist() -> None:
        reason = ""
        if source_identity(root) != state["binding"]["candidate_tree_sha256"]:
            reason = "source_changed_during_verification"
        if environment_identity() != state["binding"]["environment"]:
            reason = "environment_changed_during_verification"
        runtimes = tuple(Path(path) for path in state["binding"]["runtime_roots"])
        if runtime_identity(runtimes) != state["binding"]["runtime_identity"]:
            reason = "runtime_changed_during_verification"
        if reason:
            state["invalidated"] = reason
        cache.save(state)
        if reason:
            raise ValueError(reason)

    if state["binding"]["isolation_enforced"]:
        from scripts.merge_verification_isolation import isolated_static

        run_step = functools.partial(
            isolated_static,
            runtimes=tuple(Path(path) for path in state["binding"]["runtime_roots"]),
        )
    _run_statics(state, steps, run_step, root, budget, persist)
    if _collect(state, root, cache.directory, budget, persist):
        _run_shards(state, root, cache.directory, budget, persist)
    persist()
    return {
        **_summary(state),
        "journal_path": str(cache.directory / f"{state['identity']}.json"),
    }


def _allowance(state, kind, targets, available, *, exhausted=False, prior=None):
    if kind != "static":
        exhausted = _attempts_exhausted(state, kind, targets)
    fallback = (
        state.get("shard_budget_seconds", available) if kind == "shard" else available
    )
    full = state.get("workload_budget_seconds", fallback)
    required = full if not prior else max(1.0, prior.get("seconds", full) * 2)
    if kind == "shard":
        required = _shard_seconds(state, targets, full)
    if exhausted or available <= 0 or available + 1e-6 < required:
        state.setdefault("waiting", {})[digest([kind, targets])] = {
            "kind": kind,
            "target_digest": digest(targets),
            "reason": "retry_exhausted" if exhausted else "insufficient_budget",
            "required_seconds": required,
            "available_seconds": max(0.0, available),
            "wake_source": "verified_repair"
            if exhausted
            else "budget_or_schedule_change"
            if required > full
            else "fresh_bounded_invocation",
        }
        return 0.0
    return available


def _shard_seconds(state, nodes, full):
    table = check_changed._test_file_durations()
    counts = Counter(node.split("::", 1)[0] for node in state.get("nodes", nodes))
    weights = [table.get(node.split("::", 1)[0]) for node in nodes]
    if any(
        value is None or not math.isfinite(value) or value <= 0 for value in weights
    ):
        return min(full, state.get("shard_budget_seconds", full))
    collections = [
        row.get("seconds", 0)
        for row in state["attempts"]
        if row.get("kind") == "collection"
    ]
    startup = max(collections, default=1.0)
    return startup + 2 * sum(
        value / counts[node.split("::", 1)[0]] for node, value in zip(nodes, weights)
    )


def _run_statics(state, steps, run_step, root, budget, persist) -> bool:
    for step in steps:
        prior = state["static"].get(step.name)
        attempts = sum(
            row["name"] == step.name for row in state.get("static_history", [])
        )
        if not step.static or (prior and prior["status"] == "ok"):
            continue
        seconds = _allowance(
            state,
            "static",
            [step.name],
            budget(),
            prior=prior,
            exhausted=bool(prior)
            and attempts + 1 >= state["binding"]["max_attempts_per_obligation"],
        )
        if not seconds:
            continue
        record = run_step(step, budget_seconds=seconds, root=root)
        if prior:
            state.setdefault("static_history", []).append(prior)
        state["static"][step.name] = record
        if record["status"] != "ok":
            _allowance(
                state,
                "static",
                [step.name],
                budget(),
                prior=record,
                exhausted=attempts + int(bool(prior)) + 1
                >= state["binding"]["max_attempts_per_obligation"],
            )
        persist()
    return all(
        state["static"].get(step.name, {}).get("status") == "ok"
        for step in steps
        if step.static
    )


def _collect(state, root, directory, budget, persist) -> bool:
    if "nodes" in state:
        return True
    seconds = _allowance(state, "collection", state["binding"]["targets"], budget())
    if not seconds:
        return False
    record = run_workload(
        root,
        state["binding"]["targets"],
        collect_only=True,
        seconds=seconds,
        directory=directory,
        isolated=state["binding"]["isolation_enforced"],
        runtimes=tuple(Path(path) for path in state["binding"]["runtime_roots"]),
    )
    state["attempts"].append({**record, "kind": "collection"})
    if record["status"] != "ok":
        _allowance(state, "collection", state["binding"]["targets"], budget())
    if record["status"] == "ok":
        state["nodes"] = record["nodes"]
        state["shards"] = plan_shards(record["nodes"], state["shard_budget_seconds"])
    persist()
    return record["status"] == "ok"


def _run_shards(state, root, directory, budget, persist) -> None:
    for nodes in tuple(state["shards"]):
        if set(nodes) <= set(state["passed_nodes"]):
            continue
        seconds = _allowance(state, "shard", nodes, budget())
        if not seconds:
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
        if record["status"] == "ok":
            state["passed_nodes"].extend(nodes)
        else:
            _allowance(state, "shard", nodes, budget())
        if (
            record["status"] == "timeout"
            and len(nodes) > 1
            and not _attempts_exhausted(state, "shard", nodes)
        ):
            index = state["shards"].index(nodes)
            midpoint = len(nodes) // 2
            state["shards"][index : index + 1] = [nodes[:midpoint], nodes[midpoint:]]
        persist()


def _attempts_exhausted(state: dict, kind: str, nodes: list[str]) -> bool:
    attempts = sum(
        row.get("kind") == kind
        and (
            set(nodes) <= set(row["nodes"])
            if kind == "shard"
            else row["nodes"] == nodes
        )
        for row in state["attempts"]
    )
    return attempts >= state["binding"]["max_attempts_per_obligation"]


def run_release_gate(
    steps: tuple,
    *,
    root: Path,
    base_ref: str,
    state_dir: Path,
    step_seconds: float,
    run_step,
    local_feedback: bool = False,
    runtime_roots: tuple[Path, ...] = (),
) -> dict:
    return run_locked_release_gate(None, locals())


def run_locked_release_gate(expected_identity: str | None, invocation: dict) -> dict:
    """One capped invocation; a matching authenticated journal resumes its work."""
    import fcntl

    steps, root = invocation["steps"], invocation["root"]
    base_ref, state_dir = invocation["base_ref"], invocation["state_dir"]
    step_seconds, run_step = invocation["step_seconds"], invocation["run_step"]
    local_feedback, runtime_roots = (
        invocation["local_feedback"],
        invocation["runtime_roots"],
    )
    if (
        not math.isfinite(step_seconds)
        or not 0 < step_seconds <= INTERRUPT_AFTER_SECONDS
    ):
        raise ValueError("step budget exceeds canonical interrupt cap")
    deadline = time.monotonic() + min(INTERRUPT_AFTER_SECONDS, step_seconds + 30)
    if not local_feedback:
        from slm_training.autoresearch.heal.isolation import (
            IsolationUnavailable,
            probe_isolation,
        )

        capability = probe_isolation()
        if not capability.available:
            raise IsolationUnavailable(capability.reason)
    cache = ReceiptCache(state_dir, root)
    with (cache.directory / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = verification_binding(
            root,
            base_ref,
            steps,
            isolated=not local_feedback,
            runtimes=runtime_roots or (Path(sys.prefix),),
        )
        identity = digest(binding)
        if expected_identity is not None and identity != expected_identity:
            raise ValueError("locked_verification_identity_mismatch")
        state = cache.load(identity) or {
            "schema": "merge_verification_state/v1",
            "identity": identity,
            "binding": binding,
            "static": {},
            "attempts": [],
            "passed_nodes": [],
        }
        if state.get("schema") != "merge_verification_state/v1":
            raise ValueError("legacy_cache_not_reusable")
        validate_cached_state(state, binding)
        if not binding["targets"]:
            return _summary(state, reason="zero_required_test_collection")
        return _execute_pending(
            state,
            root=root,
            cache=cache,
            steps=steps,
            run_step=run_step,
            deadline=deadline,
            step_seconds=step_seconds,
        )
