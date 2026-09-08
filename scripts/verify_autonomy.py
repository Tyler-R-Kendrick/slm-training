"""Finite, resumable operational acceptance through actual campaign/runtime owners.

No training, promotion, provider access or persistent service is activated here.
Run multiple bounded invocations against the same locked plan to reach 100 real
workload attempts. Changed tested source refuses continuation of old evidence.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

from scripts.autonomy_fixture_worker import fixture_spec, run_fixture_attempt, validate_repair_proof
from slm_training.autoresearch.runtime.activity_process import verify_outputs
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.harness_core.activity_contract import (
    ActivityLease,
    contract_digest,
)
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.bounded_process import run_bounded_process
from slm_training.levers import INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS
from slm_training.versioning import build_version_stamp

_CAMPAIGN = "autonomy-adversary-finite-100"
_OWNERS = (
    "scripts/verify_autonomy.py",
    "scripts/autonomy_fixture_worker.py",
    "src/slm_training/autoresearch/runtime/__init__.py",
    "src/slm_training/autoresearch/runtime/activity_runtime.py",
    "src/slm_training/harness_core/activity_contract.py",
    "src/slm_training/autoresearch/runtime/activity_projection.py",
    "src/slm_training/autoresearch/runtime/activity_process.py",
    "src/slm_training/autoresearch/runtime/activity_publication.py",
    "src/slm_training/autoresearch/storage.py",
    "src/slm_training/autoresearch/campaign_events.py",
    "src/slm_training/autoresearch/campaign_formal_evidence.py",
    "src/slm_training/autoresearch/experiment_campaign.py",
    "src/slm_training/autoresearch/schemas.py",
    "src/slm_training/harness_core/bounded_process.py",
    "src/slm_training/harness_core/process_tree.py",
    "src/slm_training/levers.py",
    "src/slm_training/resources/versions.json",
)


def _plan(root: Path, *, version: int = 5) -> dict:
    faults = (
        "code_crash",
        "missing_output",
        "malformed_output",
        "hang",
        "stale_fence",
        "controller_lost",
        "verified_before_crash",
    )
    plan = {
        "schema_version": "autonomy_adversary_plan/v2",
        "successor_reason": "v1_finalization_reserve_exhausted_under_real_store_cost",
        "attempts": 100,
        "faults": [
            faults[i % 20] if i % 20 < len(faults) else "complete" for i in range(100)
        ],
        "source_files": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in _OWNERS
        },
        "environment_digest": contract_digest(
            {"python": sys.version, "platform": platform.platform()}
        ),
        "expected_succeeded": 70,
        "expected_cancelled": 5,
        "expected_waiting_capability": 1,
        "expected_actual_attempts": 100,
        "logical_worker_seconds_grant": 100 * (15 + KILL_GRACE_SECONDS),
        "scientific_claim": "none_operational_fixture_only",
    }
    if version == 2:
        return plan
    if version not in (3, 4, 5):
        raise ValueError("unsupported finite fault plan")
    faults = (*faults, "repair_fixture", "repair_replay", "storage_after_verify", "storage_before_verify")
    plan.update(
        schema_version="autonomy_adversary_plan/v3",
        successor_reason="v2_parked_code_crashes_without_repair_replay_or_storage_interruption",
        faults=[faults[i % 20] if i % 20 < len(faults) else "complete" for i in range(100)],
        expected_succeeded=65, expected_fake_agent_predicate_repairs=5,
        expected_verified_replays=5, expected_storage_interruptions=10,
        workload_count_unit="actual_runtime_run_attempt_including_planned_replays",
        repair_evidence_scope="fake_adapter_real_isolated_predicate_checks_no_source_release_authorization",
    )
    plan["controller_interrupt_seconds"] = [65 if fault == "repair_fixture" else 30 for fault in plan["faults"]]
    plan["logical_worker_seconds_grant"] = sum(seconds + KILL_GRACE_SECONDS for seconds in plan["controller_interrupt_seconds"])
    plan["source_files"].update({
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (root / "src/slm_training/autoresearch/heal").glob("*.py")
    })
    if version >= 4:
        plan.update(schema_version="autonomy_adversary_plan/v4",
                    successor_reason="v3_healthy_workload_startup_exhausted_two_second_lease",
                    ordinary_workload_interrupt_seconds=10,
                    ordinary_finalization_reserve_seconds=10,
                    ordinary_total_seconds=31)
        plan["controller_interrupt_seconds"] = [65 if fault == "repair_fixture" else 45 for fault in plan["faults"]]
        plan["logical_worker_seconds_grant"] = sum(seconds + KILL_GRACE_SECONDS for seconds in plan["controller_interrupt_seconds"])
    if version == 5:
        plan.update(schema_version="autonomy_adversary_plan/v5",
                    successor_reason="v4_isolated_check_preparation_exhausted_two_second_budget",
                    repair_check_seconds=10, repair_workload_interrupt_seconds=60)
        plan["controller_interrupt_seconds"] = [90 if fault == "repair_fixture" else 45 for fault in plan["faults"]]
        plan["logical_worker_seconds_grant"] = sum(seconds + KILL_GRACE_SECONDS for seconds in plan["controller_interrupt_seconds"])
    return plan


def _lock(store: CampaignStore, plan: dict) -> Path:
    events = store.verify_event_chain()
    locked = [
        row for row in events if row["event_type"] == "autonomy_fault_plan_locked"
    ]
    if locked:
        if locked[0]["detail"]["plan_digest"] != contract_digest(plan):
            raise ValueError(
                "locked acceptance source/plan changed; preserve old evidence"
            )
    artifact = store.write_artifact("autonomy_fault_plan", plan)
    if not locked:
        store.append_event(
            "autonomy_fault_plan_locked",
            idempotency_key="autonomy-plan",
            detail={"plan_digest": contract_digest(plan), "artifact": str(artifact)},
        )
    return artifact


def _summary(store: CampaignStore, plan: dict) -> dict:
    with ActivityRuntime(store) as runtime:
        states = runtime.snapshot()
    events = store.verify_event_chain()
    _check_artifacts(store, states, events)
    if states["external-authority"].status != "waiting_capability":
        raise AssertionError("external authority wait was lost or invented")
    observations = [
        row["detail"]
        for row in store.verify_event_chain()
        if row["event_type"] == "autonomy_fixture_workload_observed"
    ]
    indices = {row["index"] for row in observations}
    if len(indices) != len(observations):
        raise AssertionError("retry counted as an independent workload attempt")
    expected_charge = sum(
        row["attempt_reservation"]
        if row["fault"] in {"controller_lost", "stale_fence", "storage_before_verify"}
        else row["seconds"]
        for row in observations
    )
    actual_charge = sum(state.charged_seconds for state in states.values())
    if abs(expected_charge - actual_charge) > 1e-6:
        raise AssertionError(
            f"incorrect resource charges: {expected_charge} != {actual_charge}"
        )
    completed = sum(state.status == "succeeded" for state in states.values())
    coverage = _repair_coverage(store, plan, states, observations, events)
    if len(observations) == plan["attempts"]:
        if completed != plan["expected_succeeded"]:
            raise AssertionError(f"unexpected accepted work count: {completed}")
        if (
            sum(s.status == "cancelled" for s in states.values())
            != plan["expected_cancelled"]
        ):
            raise AssertionError("cancelled owner became a success")
    return {
        "schema_version": "autonomy_adversary_evidence/v1",
        "version_stamp": build_version_stamp("autoresearch.heal"),
        "plan_digest": contract_digest(plan),
        "source_files": plan["source_files"],
        "actual_workload_attempts": len(observations),
        "complete_operational_artifacts": completed,
        "charged_workload_seconds": actual_charge,
        "observed_workload_seconds": sum(row["seconds"] for row in observations),
        "observed_controller_seconds_including_workloads": sum(
            row["detail"]["seconds"]
            for row in events
            if row["event_type"] == "autonomy_fixture_controller_observed"
        ),
        "controller_contexts": sum(
            row["event_type"] == "activity_controller_started"
            for row in store.verify_event_chain()
        ),
        "states": {name: state.status for name, state in states.items()},
        "complete": len(observations) == plan["attempts"],
        "evidence_class": ("real_processes_fake_agent_isolated_predicate_checks_injected_storage_errors"
                           if "repair_fixture" in plan["faults"] else "real_workload_and_controller_processes_with_locked_faults"),
        "real_model_trials": 0,
        "live_agent_repairs": 0,
        "scientific_promotions": 0,
        "observations": observations,
        **coverage,
    }


def _repair_coverage(store, plan, states, observations, events):
    repairs = [row for row in observations if row["fault"] == "repair_fixture"
               and row["returncode"] == 0 and states[row["activity_id"]].status == "succeeded"]
    for row in repairs:
        validate_repair_proof(store, states[row["activity_id"]], row["index"] - 7)
    replays = [row for row in observations if row["fault"] == "repair_replay"]
    storage = [row for row in events if row["event_type"] == "autonomy_fixture_storage_interrupted"]
    for replay in replays:
        original = next(row for row in observations if row["index"] == replay["index"] - 8)
        if (original["returncode"] != 1 or replay["returncode"] != 0
                or original["activity_id"] != replay["activity_id"]
                or original["attempt_id"] == replay["attempt_id"]
                or states[replay["activity_id"]].status != "succeeded"):
            raise AssertionError("planned repair replay did not restore the original activity")
    if len(observations) == plan["attempts"] and "repair_fixture" in plan["faults"]:
        actual = (len(repairs), len(replays), len(storage))
        expected = tuple(plan[key] for key in ("expected_fake_agent_predicate_repairs", "expected_verified_replays", "expected_storage_interruptions"))
        if actual != expected:
            raise AssertionError("incomplete repair or storage interruption coverage")
    return {"fake_agent_predicate_repairs": len(repairs), "verified_activity_replays": len(replays),
            "storage_interruptions": len(storage), "source_releases_authorized": 0}


def _check_artifacts(store, states, events):
    for state in states.values():
        if state.status != "succeeded":
            continue
        matches = [
            event
            for event in events
            if event["event_type"] == "activity_outputs_verified"
            and event["experiment_id"] == state.spec.activity_id
        ]
        if len(matches) != 1:
            raise AssertionError("verified artifact lost or accepted twice")
        detail = matches[0]["detail"]
        verify_outputs(
            store.root,
            state,
            ActivityLease.model_validate(detail["lease"]),
            detail["outputs"],
        )
        if state.outputs != detail["outputs"]:
            raise AssertionError("committed output differs from verified artifact")
    scientific = [
        event["event_type"]
        for event in events
        if any(word in event["event_type"] for word in ("promot", "champion", "ship_"))
    ]
    if scientific:
        raise AssertionError(
            f"operational fixture emitted scientific claims: {scientific}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/autoresearch"))
    parser.add_argument("--batch-size", type=int, default=20, choices=range(1, 21))
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    source = Path(__file__).resolve().parents[1]
    store = CampaignStore(_CAMPAIGN, args.root.resolve())
    plan = _plan(source)
    _lock(store, plan)
    if args.worker_index is not None:
        return run_fixture_attempt(store, plan, args.worker_index)
    with (store.root / ".autonomy-batch.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("finite_verification_already_owned") from exc
        from slm_training.autoresearch.heal.isolation import probe_isolation

        capability = probe_isolation()
        if not capability.available:
            payload = {"status": "waiting_capability", "capability": "rootless_isolated_verifier",
                       "reason": capability.reason, "plan_digest": contract_digest(plan),
                       "complete": False, "live_agent_repairs": 0}
            artifact = store.write_artifact("autonomy_fixture_capability_wait", payload)
            store.append_event("autonomy_fixture_capability_wait", artifact_sha256=artifact.stem,
                               idempotency_key="fixture-capability:" + artifact.stem)
            print(json.dumps(payload, sort_keys=True))
            return 10
        return _run_batch(args, store, plan, source)


def _run_batch(args, store, plan, source):
    with ActivityRuntime(store) as runtime:
        blocked = fixture_spec(100, plan).model_copy(
            update={
                "activity_id": "external-authority",
                "capabilities": ("unavailable_provider",),
                "output_namespace": "runs/external-authority",
            }
        )
        runtime.register(blocked)
    observed = {
        row["detail"]["index"]
        for row in store.verify_event_chain()
        if row["event_type"] == "autonomy_fixture_workload_observed"
    }
    started = time.monotonic()
    for index in [i for i in range(plan["attempts"]) if i not in observed][
        : args.batch_size
    ]:
        seconds = plan.get("controller_interrupt_seconds", [15] * plan["attempts"])[index]
        if time.monotonic() - started > INTERRUPT_AFTER_SECONDS - seconds - KILL_GRACE_SECONDS - 15:
            break
        result = run_bounded_process(
            [
                sys.executable,
                "-m",
                "scripts.verify_autonomy",
                "--root",
                str(args.root.resolve()),
                "--worker-index",
                str(index),
            ],
            cwd=source,
            interrupt_after_seconds=seconds,
            kill_grace_seconds=KILL_GRACE_SECONDS,
        )
        expected = {"controller_lost": 73, "verified_before_crash": 74,
                    "storage_before_verify": 75, "storage_after_verify": 76}.get(
            plan["faults"][index], 0
        )
        if result.returncode != expected or result.timed_out:
            raise AssertionError(f"controller attempt {index} failed: {result}")
        store.append_event(
            "autonomy_fixture_controller_observed",
            experiment_id=f"attempt-{index:03d}",
            idempotency_key=f"controller-observation:{index}",
            detail={
                "index": index,
                "seconds": result.duration_seconds,
                "returncode": result.returncode,
            },
        )
    if _plan(source) != plan:
        raise ValueError("locked acceptance source/plan changed; preserve old evidence")
    summary = _summary(store, plan)
    store.write_artifact("autonomy_adversary_evidence", summary)
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in {"states", "observations", "source_files"}
            },
            sort_keys=True,
        )
    )
    return 0 if summary["complete"] else 10


if __name__ == "__main__":
    raise SystemExit(main())
