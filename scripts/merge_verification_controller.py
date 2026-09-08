"""Finite release-verification activities on the existing CampaignStore runtime.

A call drains the runnable prefix within the command cap. The same registered
activity is available to the existing controller on its next bounded tick;
this module never installs a service or spawns an unbounded guardian.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from scripts.merge_verification import _summary
from scripts.merge_verification_evidence import (
    ReceiptCache,
    digest,
    environment_identity,
    file_digest,
    source_identity,
    validate_cached_state,
    runtime_identity,
)
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harness_core.activity_contract import (
    ActivityOutcome,
    ActivitySpec,
    ResourceGrant,
    WakeCondition,
)
from slm_training.levers import INTERRUPT_AFTER_SECONDS


def release_grant(plan):
    if not 0 < plan["step_seconds"] <= INTERRUPT_AFTER_SECONDS - 80:
        raise ValueError(
            "finite controller step allowance must fit startup and finalization reserves"
        )
    return ResourceGrant(
        interrupt_seconds=plan["step_seconds"] + 35,
        total_seconds=plan["total_seconds"],
        finalization_reserve_seconds=10,
        max_attempts=plan["max_invocations"],
    )


def register_release(runtime, plan):
    grant = release_grant(plan)
    identity = digest(plan)
    activity_id = plan.get("activity_id", "release-verification")
    locks = [
        event["detail"]["plan_digest"]
        for event in runtime.store.verify_event_chain()
        if event["event_type"] == "release_verification_locked"
        and event.get("experiment_id") == activity_id
    ]
    if locks and locks != [identity]:
        raise ValueError(
            "locked verification source/config/grant changed; use an explicit successor job"
        )
    artifact = runtime.store.write_artifact("release_verification_plan", plan)
    runtime.store.append_event(
        "release_verification_locked",
        experiment_id=activity_id,
        idempotency_key="release-plan:" + activity_id,
        detail={"plan_digest": identity, "artifact": str(artifact)},
    )
    return runtime.register(
        ActivitySpec(
            activity_id=activity_id,
            family="release-verification",
            kind="verify",
            source_digest=plan["source_digest"],
            environment_digest=plan["environment_digest"],
            input_digest=identity,
            output_namespace="attempts/" + digest(activity_id),
            capabilities=("local_process",)
            if plan["local_feedback"]
            else ("local_process", "isolated_verifier"),
            grant=grant,
        )
    )


def verification_argv(plan):
    trusted = Path(__file__).resolve().parents[1]
    imports = [str(trusted), str(trusted / "src")]
    imports.extend(
        map(os.path.abspath, os.environ.get("PYTHONPATH", "").split(os.pathsep))
    )
    argv = [
        sys.executable,
        "-I",
        "-c",
        "import sys; sys.path[:0] = "
        + repr(imports)
        + "; from scripts.verify_merge_ready import main; raise SystemExit(main())",
        "--source",
        plan["source"],
        "--json",
        "--state-dir",
        plan["state_dir"],
        "--base-ref",
        plan["base_ref"],
        "--max-step-seconds",
        str(plan["step_seconds"]),
    ]
    if plan["local_feedback"]:
        argv.append("--local-feedback")
    if plan.get("identity"):
        argv.extend(("--identity", plan["identity"]))
    if plan.get("require_js_runtime"):
        argv.append("--require-js-runtime")
    for root in plan["runtime_roots"]:
        argv.extend(("--runtime-root", root))
    return argv


def validate_observation(result, plan):
    if (
        result.timed_out
        or result.cancelled
        or result.progress_stalled
        or result.returncode not in {0, 1, 10, 20}
    ):
        raise ValueError(
            "verification child did not produce a complete operational observation"
        )
    summary = json.loads(result.stdout)
    if result.returncode == 1:
        if summary.get("status") != "invalid_evidence":
            raise ValueError("failed verification child asserted a nonfailure outcome")
        return {
            "status": "invalid_evidence",
            "reason": str(summary.get("reason", "invalid_verification_evidence"))[
                :1000
            ],
            "verification_complete": False,
            "release_authorized": False,
        }
    if (
        summary.get("status")
        in {"waiting_capability", "waiting_dependency", "waiting_environment"}
        and result.returncode == 20
    ):
        return {**summary, "verification_complete": False, "release_authorized": False}
    if plan.get("identity") and summary.get("identity") != plan["identity"]:
        raise ValueError("verification result identity mismatch")
    cache = ReceiptCache(Path(plan["state_dir"]), Path(plan["source"]))
    state = cache.load(summary["identity"])
    if (
        state is None
        or state["binding"]["candidate_tree_sha256"] != plan["source_digest"]
    ):
        raise ValueError("verification source/journal mismatch")
    if (
        digest(state["binding"]["environment"]) != plan["environment_digest"]
        or state["binding"]["runtime_identity"] != plan["runtime_digest"]
    ):
        raise ValueError("verification environment/runtime mismatch")
    validate_cached_state(state, state["binding"])
    expected = _summary(state)
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("verification stdout differs from authenticated journal")
    return summary


def execute_release_attempt(runtime, plan, lease):
    started = time.monotonic()
    env = dict(os.environ)
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = os.pathsep.join(
            map(os.path.abspath, env["PYTHONPATH"].split(os.pathsep))
        )
    result = runtime.run(
        lease, verification_argv(plan), cwd=Path(plan["source"]), env=env
    )
    try:
        summary = validate_observation(result, plan)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        summary = {
            "status": "invalid_evidence",
            "reason": str(exc),
            "verification_complete": False,
            "release_authorized": False,
        }
    status = summary["status"]
    outcome = {
        "complete": ActivityOutcome.SUCCEEDED,
        "pending": ActivityOutcome.YIELDED,
        "waiting_capability": ActivityOutcome.CAPABILITY,
        "waiting_dependency": ActivityOutcome.RETRY,
        "waiting_environment": ActivityOutcome.ENVIRONMENT_FAILURE,
    }.get(status, ActivityOutcome.UNKNOWN_FAILURE)
    wake = (
        None
        if outcome in {ActivityOutcome.SUCCEEDED, ActivityOutcome.YIELDED}
        else WakeCondition(
            predicate=summary.get("next_action", {}).get(
                "kind", "verified_verification_prerequisite"
            ),
            source="independent_verifier_or_capability_probe",
            identity_digest=digest(plan),
        )
    )
    output = runtime.attempt_dir(lease) / "result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime.store._replace_durable(
        output, json.dumps(summary, sort_keys=True, allow_nan=False)
    )
    runtime.store.append_event(
        "release_verification_observed",
        experiment_id=lease.activity_id,
        idempotency_key=f"release-observation:{lease.attempt_id}",
        detail={"attempt_id": lease.attempt_id, "summary": summary},
    )
    runtime.finish(
        lease,
        outcome=outcome,
        outputs={"result.json": file_digest(output)},
        spent_seconds=time.monotonic() - started,
        wake=wake,
    )
    return summary


def drain_release(runtime, plan, *, deadline):
    """Existing controllers may call this directly; no nested controller lock."""
    state = register_release(runtime, plan)
    capabilities = {"local_process"}
    if not plan["local_feedback"]:
        from slm_training.autoresearch.heal.isolation import probe_isolation

        if probe_isolation().available:
            capabilities.add("isolated_verifier")
    observed = None
    while (
        not runtime.cancel_event.is_set()
        and time.monotonic() + state.spec.grant.attempt_seconds + 15 < deadline
    ):
        state = runtime.snapshot()[state.spec.activity_id]
        if state.status == "waiting_retry":
            delay = max(0, state.retry_at - runtime.clock())
            if delay > 0:
                runtime.cancel_event.wait(min(delay, 1))
                continue
        lease = runtime.claim_next(
            capabilities=capabilities, activity_id=state.spec.activity_id
        )
        if lease is None:
            break
        observed = execute_release_attempt(runtime, plan, lease)
        if observed["status"] != "pending":
            break
    state = runtime.snapshot()[state.spec.activity_id]
    if observed is None and state.status == "succeeded":
        observed = next(
            event["detail"]["summary"]
            for event in reversed(runtime.store.verify_event_chain())
            if event["event_type"] == "release_verification_observed"
            and event.get("experiment_id") == state.spec.activity_id
        )
    return {
        "schema": "release_verification_controller/v1",
        "activity_id": state.spec.activity_id,
        "status": state.status,
        "attempts": state.attempts,
        "charged_activity_seconds": state.charged_seconds,
        "total_seconds_grant": state.spec.grant.total_seconds,
        "retry_at": state.retry_at,
        "wake": state.wake.model_dump() if state.wake else None,
        "verification_complete": state.status == "succeeded",
        "release_authorized": False,
        "observation": observed,
    }


def verify_release(args):
    started = time.monotonic()
    source = args.source.resolve()
    roots = args.runtime_root or [Path(sys.prefix)]
    plan = {
        "schema": "release_verification_plan/v1",
        "activity_id": args.activity_id,
        "identity": args.identity,
        "source": str(source),
        "source_digest": source_identity(source),
        "environment_digest": digest(environment_identity()),
        "state_dir": str(args.state_dir.resolve()),
        "base_ref": args.base_ref,
        "step_seconds": args.max_step_seconds,
        "total_seconds": args.total_seconds,
        "max_invocations": args.max_invocations,
        "local_feedback": args.local_feedback,
        "require_js_runtime": args.require_js_runtime,
        "runtime_roots": [str(root.resolve()) for root in roots],
        "runtime_digest": runtime_identity(tuple(roots)),
    }
    store = CampaignStore(args.job_id, args.root.resolve())
    with ActivityRuntime(store) as runtime:
        summary = drain_release(
            runtime, plan, deadline=started + INTERRUPT_AFTER_SECONDS - 10
        )
        store.append_event(
            "release_controller_observed",
            detail={
                "seconds_including_workloads": time.monotonic() - started,
                "controller_epoch": runtime.epoch,
                "status": summary["status"],
            },
        )
    return summary


def add_release_parser(sub):
    command = sub.add_parser(
        "verify-release",
        help="Finite controller-owned resumable local verification; never promotion",
    )
    command.add_argument("--source", type=Path, default=Path.cwd())
    command.add_argument("--identity", help="locked independent verifier identity")
    command.add_argument("--activity-id", default="release-verification")
    command.add_argument("--state-dir", type=Path, required=True)
    command.add_argument("--job-id", default="release-verification")
    command.add_argument("--base-ref", default="HEAD")
    command.add_argument("--total-seconds", type=float, required=True)
    command.add_argument("--max-invocations", type=int, required=True)
    command.add_argument("--max-step-seconds", type=float, default=30)
    command.add_argument("--runtime-root", type=Path, action="append", default=[])
    command.add_argument("--local-feedback", action="store_true")
    command.add_argument(
        "--require-js-runtime",
        action="store_true",
        help="Require explicit complete bridge/SDK runtime grants before workloads",
    )
    command.set_defaults(func=cmd_verify_release)


def cmd_verify_release(args):
    result = verify_release(args)
    print(json.dumps(result, sort_keys=True))
    return (
        0
        if result["verification_complete"]
        else 10
        if result["status"] in {"runnable", "waiting_retry"}
        else 20
    )
