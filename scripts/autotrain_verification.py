"""One bounded source-verifier pass over controller-owned repair dependencies.

No grant is inferred, no repair is acknowledged here, and a successful journal
only wakes ISO's independent acceptance path. The existing runtime owns retry
timers, finite resource charges, leases and fairness with other activities.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from scripts.merge_verification import _summary, runtime_identity, verification_binding
from scripts.merge_verification_controller import execute_release_attempt
from scripts.merge_verification_evidence import (
    ReceiptCache,
    digest,
    validate_cached_state,
)
from scripts.verify_merge_ready import merge_gate_steps
from slm_training.autoresearch.storage import _sha
from slm_training.autoresearch.heal.repair_acceptance import source_verification_activity_id
from slm_training.harness_core.activity_contract import (
    ActivitySpec,
    ResourceGrant,
    WakeCondition,
)
from slm_training.levers import KILL_GRACE_SECONDS


class VerificationCapabilityUnavailable(ValueError):
    """A scoped missing controller grant/configuration, not a code failure."""


def _runtime_digest(dependency):
    pinned = dependency.get("runtime_identity")
    if pinned is not None:
        return pinned
    path = Path(dependency.get("manifest_path", ""))
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 1_048_576:
        return None
    manifest = json.loads(path.read_text())
    binding = manifest.get("binding")
    identity = dependency["verification_identity"]
    if (
        manifest.get("verification_identity") != identity
        or not isinstance(binding, dict)
        or digest(binding) != identity
    ):
        raise ValueError("source_verification_manifest_identity_mismatch")
    return binding.get("runtime_identity")


def load_dependency(store, event):
    identity = event["detail"]["dependency_digest"]
    if len(identity) != 64 or any(char not in "0123456789abcdef" for char in identity):
        raise ValueError("invalid_verification_dependency_digest")
    path = (
        store.root / "artifacts" / "source_verification_requests" / f"{identity}.json"
    )
    dependency = json.loads(path.read_text())
    if _sha(dependency) != identity:
        raise ValueError("verification_dependency_content_mismatch")
    if event["detail"]["repair_activity_id"] != event["experiment_id"]:
        raise ValueError("verification_dependency_repair_mismatch")
    return dependency


def dependency_plan(dependency):
    required = (
        "root",
        "state_dir",
        "base_ref",
        "verification_identity",
        "activity_id",
        "grant",
    )
    if any(not dependency.get(key) for key in required):
        raise VerificationCapabilityUnavailable(
            "source_verifier_configuration_or_grant_missing"
        )
    explicit = dependency["grant"]
    if not {"total_seconds", "max_attempts", "interrupt_seconds"} <= explicit.keys():
        raise VerificationCapabilityUnavailable(
            "source_verifier_resource_grant_incomplete"
        )
    grant = ResourceGrant.model_validate(explicit)
    # The inner gate keeps its own monotonic deadline, leaving five seconds for
    # child exit/summary publication before the independently enforced watchdog.
    step_seconds = grant.interrupt_seconds - 35
    if step_seconds <= 2 * KILL_GRACE_SECONDS:
        raise VerificationCapabilityUnavailable(
            "source_verifier_grant_below_startup_reserve"
        )
    identity = dependency["verification_identity"]
    if dependency["activity_id"] != source_verification_activity_id(identity, grant):
        raise ValueError("source_verification_activity_identity_mismatch")
    wake = WakeCondition.model_validate(dependency["wake"])
    if wake != WakeCondition(
        predicate="complete_current_source_verification",
        source="source_verification_completed",
        identity_digest=identity,
    ):
        raise ValueError("source_verification_wake_mismatch")
    root = Path(dependency["root"]).resolve()
    roots = tuple(
        Path(path).resolve() for path in dependency.get("runtime_roots", ())
    ) or (Path(sys.prefix),)
    state_dir = Path(dependency["state_dir"]).resolve()
    for exposed in (root, *roots):
        if state_dir.is_relative_to(exposed) or exposed.is_relative_to(state_dir):
            raise ValueError("source_verification_issuer_exposed_to_workload")
    pinned_runtime = _runtime_digest(dependency)
    current_runtime = runtime_identity(roots)
    binding = verification_binding(
        root, dependency["base_ref"], merge_gate_steps(), isolated=True, runtimes=roots,
        runtime_digest_value=current_runtime,
    )
    if pinned_runtime is not None and pinned_runtime != current_runtime:
        raise ValueError("source_verification_binding_changed")
    if digest(binding) != identity:
        raise ValueError("source_verification_binding_changed")
    return {
        "identity": identity,
        "activity_id": dependency["activity_id"],
        "source": str(root),
        "state_dir": str(state_dir),
        "source_digest": binding["candidate_tree_sha256"],
        "environment_digest": digest(binding["environment"]),
        "runtime_digest": binding["runtime_identity"],
        "runtime_roots": [str(path) for path in roots],
        "base_ref": dependency["base_ref"],
        "step_seconds": step_seconds,
        "local_feedback": False,
        "binding": binding,
        "grant": grant.model_dump(),
    }


def authenticated_completion(plan):
    """Recheck the exact signed journal, including on controller crash replay."""
    directory = Path(plan["state_dir"])
    if not (directory / "issuer.key").exists():
        return False
    state = ReceiptCache(directory, Path(plan["source"])).load(plan["identity"])
    if state is None:
        return False
    validate_cached_state(state, plan["binding"])
    if not _summary(state)["verification_complete"]:
        return False
    roots = tuple(Path(path) for path in plan["runtime_roots"])
    runtime_digest = runtime_identity(roots)
    current = verification_binding(
        Path(plan["source"]),
        plan["base_ref"],
        merge_gate_steps(),
        isolated=True,
        runtimes=roots,
        runtime_digest_value=runtime_digest,
    )
    if digest(current) != plan["identity"]:
        raise ValueError("source_verification_binding_changed_before_wake")
    return True


def register_dependency(runtime, plan):
    return runtime.register(
        ActivitySpec(
            activity_id=plan["activity_id"],
            family="source-verification",
            kind="verify",
            source_digest=plan["source_digest"],
            environment_digest=plan["environment_digest"],
            input_digest=digest(plan),
            output_namespace="attempts/" + plan["activity_id"],
            capabilities=("local_process", "isolated_verifier"),
            grant=ResourceGrant.model_validate(plan["grant"]),
        )
    )


def wake_repair(runtime, event, dependency, plan):
    repair_id = event["experiment_id"]
    repair = runtime.snapshot().get(repair_id)
    # Dependency persistence deliberately precedes finish. Never interrupt a
    # still-running repair, or consume a wake belonging to a successor request.
    if repair is None or repair.status != "waiting_dependency":
        return False
    try:
        if _active_dependency(runtime, event, repair) != (event, dependency):
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if not authenticated_completion(plan):
        return False
    runtime.wake(repair_id, evidence=repair.wake)
    runtime.store.append_event(
        "source_verification_completed",
        experiment_id=repair_id,
        detail={
            "verification_identity": plan["identity"],
            "activity_id": plan["activity_id"],
        },
    )
    return True


def _active_dependency(runtime, event, repair):
    """Follow authenticated activations from the parked wake, never mint authority."""
    from scripts.autotrain_verification_successor import _remaining_grant

    rows = [row for row in runtime.store.verify_event_chain()
            if row["experiment_id"] == event["experiment_id"]]
    requests = {row["detail"]["dependency_digest"]: row for row in rows
                if row["event_type"] == "source_verification_requested"}
    dependencies = {key: load_dependency(runtime.store, row) for key, row in requests.items()}
    roots = [key for key, value in dependencies.items()
             if WakeCondition.model_validate(value["wake"]) == repair.wake]
    if not roots:
        return None
    links = [row["detail"] for row in rows
             if row["event_type"] == "source_verification_successor_activated"]
    # Runtime rollback can repeat a wake identity with a smaller residual grant.
    # Anchor at activation ancestry; dependency digests distinguish those jobs.
    activated = {link["successor_dependency_digest"] for link in links}
    roots = [key for key in roots if key not in activated]
    if len(roots) != 1:
        raise ValueError("source_verification_successor_ambiguous_root")
    key, seen = roots[0], set()
    while key not in seen:
        seen.add(key)
        current = dependencies[key]
        outgoing = [link for link in links if link["predecessor_dependency_digest"] == key]
        if not outgoing:
            return requests[key], current
        if len(outgoing) != 1:
            raise ValueError("source_verification_successor_ambiguous_link")
        link = outgoing[0]
        target = link["successor_dependency_digest"]
        if target in seen:
            raise ValueError("source_verification_successor_cycle")
        successor = dependencies[target]
        if sum(item["successor_dependency_digest"] == target for item in links) != 1:
            raise ValueError("source_verification_successor_divergent_link")
        expected = {"predecessor_identity": current["verification_identity"],
                    "successor_identity": successor["verification_identity"],
                    "successor_activity_id": successor["activity_id"],
                    "request_digest": current["request_digest"],
                    "proposal_digest": current["proposal_digest"]}
        stable = ("request_digest", "proposal_digest", "candidate_snapshot_digest",
                  "source_snapshot_digest", "campaign_id", "blocked_activity_id")
        if (any(link.get(name) != value for name, value in expected.items())
                or any(current.get(name) != successor.get(name) for name in stable)):
            raise ValueError("source_verification_successor_identity_mismatch")
        state = runtime.snapshot().get(current["activity_id"])
        if (state is None or state.status == "running"
                or state.spec.grant != ResourceGrant.model_validate(current["grant"])
                or _remaining_grant(state) != ResourceGrant.model_validate(successor["grant"])):
            raise ValueError("source_verification_successor_grant_mismatch")
        key = target
    raise ValueError("source_verification_successor_cycle")


def drain_source_verification(runtime, common, log_event, *, cycle: int = 0):
    """Parent pre-cycle hook; at most one actual bounded pass, no busy retry loop."""
    events = [
        event
        for event in runtime.store.verify_event_chain()
        if event["event_type"] == "source_verification_requested"
    ]
    states = runtime.snapshot()
    # Rotate a bounded window before loading per-request dependencies. A full
    # sort here would still walk an unbounded blocked backlog before the limit.
    if len(events) > 32:
        start = (max(1, cycle) - 1) * 32 % len(events)
        events = (events[start:] + events[:start])[:32]
    # Least-attempted runnable verification first within this bounded window.
    events.sort(key=lambda event: _attempt_count(runtime.store, states, event))
    scan_deadline = time.monotonic() + KILL_GRACE_SECONDS
    for event in events:
        if time.monotonic() >= scan_deadline:
            break
        repair = runtime.snapshot().get(event["experiment_id"])
        if repair is None or repair.status != "waiting_dependency":
            continue
        result = _run_source_verification(runtime, event, common, log_event, repair)
        if result is not None:
            return result
    return None


def _run_source_verification(runtime, event, common, log_event, repair):
    dependency = None
    try:
        resolved = _active_dependency(runtime, event, repair)
        if resolved is None:
            return None
        event, dependency = resolved
        plan = dependency_plan(dependency)
        if wake_repair(runtime, event, dependency, plan):
            return None
        register_dependency(runtime, plan)
        from slm_training.autoresearch.heal.isolation import probe_isolation

        capabilities = {"local_process"}
        if probe_isolation().available:
            capabilities.add("isolated_verifier")
        lease = runtime.claim_next(capabilities=capabilities, activity_id=plan["activity_id"])
        if lease is None:
            return None
        summary = execute_release_attempt(runtime, plan, lease)
        woke = wake_repair(runtime, event, dependency, plan)
        log_event({"event": "source_verification_pass", "activity_id": plan["activity_id"],
                   "status": summary["status"], "repair_woken": woke,
                   "phase_progress": summary.get("phase_progress"),
                   "next_action": summary.get("next_action")})
        return summary
    except (OSError, ValueError, KeyError, TypeError) as exc:
        successor, observation_error = _successor_after_drift(
            runtime, event, dependency, common, exc
        )
        if successor is not None:
            log_event({"event": "source_verification_successor", **successor})
            return successor
        _record_source_verification_wait(runtime, event, observation_error, log_event)
        return None


def _record_source_verification_wait(runtime, event, error, log_event):
    unavailable = isinstance(error, VerificationCapabilityUnavailable)
    observation = {
        "event": "source_verification_wait",
        "repair_activity_id": event["experiment_id"],
        "status": "waiting_capability" if unavailable else "waiting_verification_evidence",
        "reason": str(error),
        "wake_source": "controller_policy" if unavailable
        else "source_verification_dependency_reconciled",
    }
    runtime.store.append_event(
        "source_verification_wait",
        experiment_id=event["experiment_id"],
        idempotency_key="verification-wait:" + digest(observation),
        detail=observation,
    )
    log_event(observation)


def _successor_after_drift(runtime, event, dependency, common, error):
    if (
        str(error) != "source_verification_binding_changed"
        or dependency is None
        or not common.get("repair_config")
        or not common.get("repair_config_digest")
        or not common.get("loop_id")
    ):
        return None, error
    from scripts.autotrain_verification_successor import plan_successor

    try:
        return plan_successor(runtime, event, dependency, common), error
    except (OSError, ValueError, KeyError, TypeError) as successor_error:
        return None, successor_error


def _attempt_count(store, states, event):
    try:
        activity_id = load_dependency(store, event).get("activity_id")
        state = states.get(activity_id)
        return state.attempts if state else 0
    except (OSError, ValueError, KeyError, TypeError):
        return 0
