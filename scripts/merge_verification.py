"""Resumable merge obligations; the signed cache is not release authority."""
from __future__ import annotations

import functools
import math
import os
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
from scripts.merge_verification_isolation import run_isolated_phase, run_workload
from scripts.merge_verification_shards import attempts_exhausted as _attempts_exhausted
from scripts.merge_verification_summary import summarize as _summary
from slm_training.levers import INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS
PYTEST_OPTIONS = [
    "-o",
    "addopts=",
    "-m",
    "not training and not slow",
    "-q",
    "-p",
    "no:cacheprovider",
]
FINALIZATION_SECONDS = 10.0
# Keep a complete source gate fundable by the controller grant. Timeouts split
# only the slow shards, instead of charging hundreds of process startups first.
MAX_INITIAL_SHARDS = 512

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
    runtime_digest_value: str | None = None,
) -> dict:
    if runtime_digest_value is not None and (
        len(runtime_digest_value) != 64
        or any(char not in "0123456789abcdef" for char in runtime_digest_value)
    ):
        raise ValueError("invalid controller runtime identity")
    base_tree, paths = changed_paths(root, base_ref)
    rule_paths = (
        "check_changed.py",
        "merge_verification.py",
        "merge_verification_collection.py",
        "merge_verification_shards.py",
        "merge_verification_git.py",
        "merge_test_worker.py",
        "merge_verification_evidence.py",
        "merge_verification_isolation.py",
        "merge_verification_summary.py",
        "merge_verification_controller.py",
        "merge_verification_runtime.py",
        "verify_merge_ready.py",
    )
    targets = check_changed.select_tests(paths, root=root) if paths else ["tests"]
    return {
        "schema": "merge_verification_binding/v3",
        "base_ref": base_ref,
        "base_tree": base_tree,
        "candidate_tree_sha256": source_identity(root),
        "changed_paths": paths,
        "targets": targets,
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
        "runtime_identity": runtime_digest_value or runtime_identity(runtimes),
        "max_attempts_per_obligation": 8,
    }

def plan_shards(nodes: list[str], budget_seconds: float) -> list[list[str]]:
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("cannot shard an empty or duplicate collection")
    table = check_changed._test_file_durations()
    counts = Counter(node.split("::", 1)[0] for node in nodes)
    total = sum(max(count * 5.0, table.get(path, 0)) for path, count in counts.items())
    if not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError("invalid shard budget")
    count = min(
        len(nodes), MAX_INITIAL_SHARDS, max(1, math.ceil(total * 4 / budget_seconds))
    )
    return [batch for batch in check_changed._shard_test_nodes(nodes, count) if batch]

def _identity_mismatch(state, root: Path) -> str:
    runtime_digest = os.environ.get("MERGE_VERIFICATION_RUNTIME_IDENTITY")
    checks = (
        (source_identity(root), state["binding"]["candidate_tree_sha256"], "source_changed_during_verification"),
        (environment_identity(), state["binding"]["environment"], "environment_changed_during_verification"),
        (
            runtime_digest
            or runtime_identity(
                tuple(Path(path) for path in state["binding"]["runtime_roots"])
            ),
            state["binding"]["runtime_identity"],
            "runtime_changed_during_verification",
        ),
    )
    return next((reason for actual, expected, reason in checks if actual != expected), "")
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
    fresh = max(0.0, budget())
    state["workload_budget_seconds"] = fresh - min(1.0, fresh / 10)
    state.setdefault("shard_budget_seconds", max(1.0, state["workload_budget_seconds"]))
    state["waiting"] = {}

    def persist(*, validate=True) -> None:
        reason = _identity_mismatch(state, root) if validate else ""
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
    def fast_persist() -> None:
        persist(validate=False)
    phase = functools.partial(run_isolated_phase, state, fast_persist=fast_persist)
    phase("static", lambda: _run_statics(state, steps, run_step, root, budget, fast_persist))
    if not state["binding"]["targets"]:
        state["no_tests_required"] = True
        persist()
    elif phase("collection", lambda: _collect(state, root, cache.directory, budget, persist, fast_persist)):
        state["shard_budget_seconds"] = min(
            state.get("shard_budget_seconds", budget()), max(1.0, budget() - 1.0)
        )
        phase("shard", lambda: _run_shards(state, root, cache.directory, budget, fast_persist))
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
    required = min(full, 10.0) if kind == "collection" and not prior else (available if kind == "static" and not prior else full)
    if kind == "static" and not prior and available <= 0:
        required = max(1.0, full)
    if prior:
        required = min(full, max(1.0, prior.get("seconds", full) * 2))
    if kind == "shard":
        # Estimates guide initial packing, but are not an admission floor:
        # run a bounded slice and split the shard if that slice times out.
        required = min(_shard_estimate_seconds(state, targets, full), available)
    wait_key = digest([kind, targets])
    # A remaining tail at-or-under two kill-graces is never spent on workload
    # obligations; static first-runs are exempt (their requirement is the
    # available slice itself — the floor would skip the static phase forever).
    if (
        exhausted
        or available <= 0
        or (kind != "static" and available <= 2 * KILL_GRACE_SECONDS)
        or available + 1e-6 < required
    ):
        state.setdefault("waiting", {})[wait_key] = {
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
    state.setdefault("waiting", {}).pop(wait_key, None)
    return required
def _shard_seconds(state, nodes, full):
    return min(full, _shard_estimate_seconds(state, nodes, full))

def _shard_estimate_seconds(state, nodes, full):
    table = check_changed._test_file_durations()
    counts = Counter(node.split("::", 1)[0] for node in state.get("nodes", nodes))
    weights = [table.get(node.split("::", 1)[0]) for node in nodes]
    if any(
        value is None or not math.isfinite(value) or value <= 0 for value in weights
    ):
        return state.get("shard_budget_seconds", full)
    collections = [
        row.get("seconds", 0)
        for row in state["attempts"]
        if row.get("kind") == "collection"
    ]
    startup = max(collections, default=1.0)
    estimate = startup + 2 * sum(
        value / counts[node.split("::", 1)[0]] for node, value in zip(nodes, weights)
    )
    return estimate

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

def _collect(state, root, directory, budget, persist, fast_persist=None) -> bool:
    from scripts.merge_verification_collection import collect

    return collect(
        state,
        root,
        directory,
        budget,
        fast_persist or persist,
        allowance=_allowance,
        run_workload=run_workload,
        plan_shards=plan_shards,
    )

def _run_shards(state, root, directory, budget, persist) -> None:
    from scripts.merge_verification_shards import run_shards

    run_shards(
        state, root, directory, budget, persist,
        allowance=_allowance, run_workload=run_workload,
    )


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
    runtime_digest = os.environ.get("MERGE_VERIFICATION_RUNTIME_IDENTITY")
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
            runtime_digest_value=invocation.get("runtime_digest"),
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
        return _execute_pending(
            state,
            root=root,
            cache=cache,
            steps=steps,
            run_step=run_step,
            deadline=deadline,
            step_seconds=step_seconds,
        )
