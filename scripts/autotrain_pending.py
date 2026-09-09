"""Typed driver yields in the existing runtime journal, never completed cycles."""

from pathlib import Path

from slm_training.autoresearch.storage import CampaignStore, _sha
from slm_training.harness_core.activity_contract import ActivityOutcome, WakeCondition


def validate_pending(payload):
    if not isinstance(payload, dict) or payload.get("schema_version") != "driver_pending/v1":
        raise ValueError("unknown driver pending contract")
    outcome = ActivityOutcome(payload["outcome"])
    if outcome not in {ActivityOutcome.CAPABILITY, ActivityOutcome.DEPENDENCY, ActivityOutcome.YIELDED}:
        raise ValueError("pending driver output cannot authorize a terminal outcome")
    if not payload.get("reason") or payload.get("measurement_complete") is not False:
        raise ValueError("driver pending requires an unmet predicate, not a measurement")
    return outcome, WakeCondition.model_validate(payload["wake"])


def publish_pending(root: Path, loop_id: str, payload: dict) -> None:
    validate_pending(payload)
    store = CampaignStore("runtime", root / "loops" / loop_id)
    artifact = store.write_artifact("driver_pending", payload)
    store.append_event("driver_yielded", artifact_sha256=artifact.stem,
                       detail={"pending_digest": artifact.stem})


def pending_since(store, event_ids):
    import json

    rows = [row for row in store.verify_event_chain()
            if row["event_type"] == "driver_yielded" and row["event_id"] not in event_ids]
    if len(rows) != 1:
        raise ValueError("driver yield omitted or duplicated its current pending output")
    row = rows[0]
    path = store.root / "artifacts/driver_pending" / f"{row['artifact_sha256']}.json"
    payload = json.loads(path.read_text())
    if _sha(payload) != row["artifact_sha256"] or row["detail"]["pending_digest"] != _sha(payload):
        raise ValueError("driver pending content changed")
    validate_pending(payload)
    return payload


def data_pending(result):
    payload = {"schema_version": "driver_pending/v1", "measurement_complete": False,
               "outcome": "dependency" if result["status"] == "waiting_dependency" else "capability",
               "reason": result["reason"], "wake": result["wake"], "readiness": result}
    if result.get("blocker"):
        payload["blocker"] = result["blocker"]
    validate_pending(payload)
    return payload


def screening_constraint_pending(report, *, inputs=None, root=None):
    """A wall deficit cannot be repaired by manufacturing more evaluation rows."""
    from slm_training.harness_core.activity_contract import contract_digest

    identity = contract_digest(report)
    code = "screening_wall_budget" if "wall_budget" in report.get("binding_constraints", ()) else "screening_constraint_unknown"
    payload = {"schema_version": "driver_pending/v1", "measurement_complete": False,
            "outcome": "dependency", "reason": code,
            "blocker": {"kind": "repair_harness", "blocker_code": code, "reason": code,
                        "required_capability": "bounded_measurement_repair", "report": report,
                        "executor": "configured_source_repair", "unmet_predicate": "locked_screening_range_feasible"},
            "wake": {"predicate": "bounded measurement plan covers the locked case requirement",
                     "source": "bounded_measurement_plan_verified", "identity_digest": identity}}
    if inputs is not None:
        import sys
        frozen = {**inputs, "screening_constraint": report}
        store = CampaignStore(frozen["matrix"]["campaign_id"], root)
        path = store.write_artifact("matrix_readiness_inputs", frozen)
        payload["readiness"] = {"input_path": str(path), "input_artifact_sha256": path.stem,
            "input_campaign_id": store.campaign_id, "request_digest": path.stem,
            "readiness_campaign_id": None}
        payload["wake"]["identity_digest"] = path.stem
        payload["blocker"].update(campaign_id=store.campaign_id, original_reproducer={
            "argv": [sys.executable, "-m", "scripts.autotrain_readiness_probe", "--input", str(path),
                     "--input-digest", path.stem, "--root", str(root)], "cwd": frozen["cwd"]})
    return payload


def screening_deficit_report(smoke_n, report, suite_records, *, automatic=False):
    """Unknown/empty actual selection is never a runnable or fabricated data plan."""
    resolved = dict(report) if isinstance(report, dict) else {}
    minimum = resolved.get("n_min", smoke_n)
    if (type(smoke_n) is not int or type(suite_records) is not int or suite_records < 0
            or type(minimum) is not int or minimum <= 0):
        return {**resolved, "binding_constraints": ["unknown"], "reason": "unresolved_screening_subset"}
    if automatic and (not resolved or (smoke_n > 0 and resolved.get("chosen_n") != smoke_n)):
        return {**resolved, "binding_constraints": ["unknown"], "reason": "unresolved_automatic_screening_selection"}
    binding = resolved.get("binding_constraints") or []
    if binding or resolved.get("must_generate") or smoke_n <= 0 or suite_records < minimum:
        return {**resolved, "n_min": minimum,
                "binding_constraints": binding or ["suite_volume" if suite_records < minimum else "unknown"]}
    return None


def selected_readiness_matrix(matrix, *, fitted_candidates):
    """Run the same pure selector used after hypothesize, without fake artifacts."""
    from scripts.autotrain_screening import screening_multi_arm_ids

    control = matrix["hypotheses"][0]["experiment"]["experiment_id"]
    candidates, _ = screening_multi_arm_ids(matrix=matrix, control_id=control,
        recommended_id=matrix["recommended_experiment_id"], fitted_candidates=fitted_candidates,
        by_id={row["experiment"]["experiment_id"]: None for row in matrix["hypotheses"]})
    return {**matrix, "selected_experiment_ids": [control, *candidates]}


def resolve_screening_matrix(matrix, matrix_inputs, deficit, *, context):
    """Pre-lock driver boundary: actual selected arms, remeasure n, recompile/recheck."""
    from scripts import run_autotrain_continuous as driver
    from scripts.autotrain_readiness import resolve_matrix_readiness

    cwd, root, policy = context["cwd"], context["root"], context["policy"]
    selected = selected_readiness_matrix(matrix, fitted_candidates=context["fitted_candidates"])
    frozen = {"matrix": selected, "cwd": str(cwd), "loop_id": context["loop_id"],
              "policy": policy.identity_dict(), "eval_version": matrix_inputs["eval_version"],
              "expected_selected_experiment_ids": selected["selected_experiment_ids"]}
    if set(deficit.get("binding_constraints") or ()) != {"suite_volume"}:
        return matrix, screening_constraint_pending(deficit, inputs=frozen, root=root)
    args = {"cwd": cwd, "root": root, "loop_id": context["loop_id"], "minimum": deficit["n_min"]}
    ready = resolve_matrix_readiness(selected, **args)
    if ready["status"] != "ready":
        return matrix, data_pending(ready)
    version = ready["eval_version"]
    rebuilt = driver._matrix(**{**matrix_inputs, "eval_version": version})
    n, report = driver._screening_n_report(policy, eval_version=version)
    remaining = screening_deficit_report(n, report, driver._screening_suite_records(version),
        automatic=getattr(policy, "measurement", {}).get("screening_smoke_n_mode") == "auto")
    final = selected_readiness_matrix(rebuilt, fitted_candidates=context["fitted_candidates"])
    frozen.update(matrix=final, eval_version=version)
    if final["selected_experiment_ids"] != selected["selected_experiment_ids"]:
        remaining = {"binding_constraints": ["unknown"], "reason": "successor_changed_scheduled_arms"}
    if remaining is not None:
        return matrix, screening_constraint_pending(remaining, inputs=frozen, root=root)
    final_ready = resolve_matrix_readiness(final, **args)
    if final_ready["status"] != "ready":
        return matrix, data_pending(final_ready)
    if final_ready["eval_version"] != version:
        return matrix, screening_constraint_pending({"binding_constraints": ["unknown"],
            "reason": "successor_readiness_changed_again"}, inputs=frozen, root=root)
    resolved = context["resolved_data"]
    resolved["eval_version"] = version
    resolved["successions"].append({"kind": "eval", "original": matrix_inputs["eval_version"],
        "successor": version, "readiness": ready, "resolved_readiness": final_ready,
        "selected_experiment_ids": selected["selected_experiment_ids"]})
    return rebuilt, None


def unresolved_driver_pending(runtime):
    """Join content-verified yield events to their exact committed driver attempt."""
    import hashlib
    import json
    from scripts.autotrain_supervisor_operations import _operation_payload

    events = runtime.store.verify_event_chain()
    yields = {e["artifact_sha256"]: e for e in events if e["event_type"] == "driver_yielded"}
    jobs = []
    for activity_id, state in runtime.snapshot().items():
        if state.status not in {"waiting_dependency", "waiting_capability"}:
            continue
        finishes = [e for e in events if e["event_type"] == "activity_transition"
                    and e["experiment_id"] == activity_id and e["detail"]["operation"] in {"finish", "reconcile_finish"}]
        if not finishes or "result.json" not in finishes[-1]["detail"].get("outputs", {}):
            continue
        attempt = finishes[-1]["detail"]["lease"]["attempt_id"]
        directory = runtime.store.root / state.spec.output_namespace / attempt
        request = json.loads((directory / "request.json").read_text())
        if request.get("operation") != "driver":
            continue
        path = directory / "result.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != finishes[-1]["detail"]["outputs"]["result.json"]:
            raise ValueError("pending driver result changed")
        payload = _operation_payload(path, request).get("pending")
        if not payload or _sha(payload) not in yields:
            raise ValueError("pending driver has no matching yield event")
        row = yields[_sha(payload)]
        current = pending_since(runtime.store, {e["event_id"] for e in events if e != row})
        if validate_pending(current)[1] != state.wake:
            raise ValueError("pending driver wake changed")
        jobs.append({"activity_id": activity_id, "pending_digest": _sha(payload),
                     "payload": current, "driver_request": request})
    return jobs


def next_driver_pending(runtime):
    """Least-recently serviced wait, persisted before dispatch so restart is fair."""
    served = {(e["experiment_id"], e["detail"]["pending_digest"]): i
              for i, e in enumerate(runtime.store.verify_event_chain())
              if e["event_type"] == "driver_pending_serviced"}
    jobs = sorted(unresolved_driver_pending(runtime),
        key=lambda job: served.get((job["activity_id"], job["pending_digest"]), -1))[:1]
    for job in jobs:
        runtime.store.append_event("driver_pending_serviced", experiment_id=job["activity_id"],
            detail={"pending_digest": job["pending_digest"]})
    return jobs


def drain_driver_pending(runtime, common, cycle, log_event, run_operation):
    """Parent pre_cycle seam: dispatch one scoped remedy, then independently verify.

    Repair answers never wake a driver. A fresh bounded controller probe must
    restore its frozen predicate; missing recipes remain capability waits.
    """
    for job in next_driver_pending(runtime):
        payload = job["payload"]
        blocker = payload.get("blocker") or payload.get("readiness", {}).get("blocker")
        if blocker and blocker.get("required_capability") == "driver_continuation_reconciliation":
            blocker = bind_continuation_probe(common, job, blocker)
        repaired = None
        if blocker and blocker.get("kind") in {"rebuild_data", "repair_harness"}:
            blocker = {**blocker, "affected_activity_id": job["activity_id"],
                "unmet_predicate": blocker.get("unmet_predicate") or payload["wake"]["predicate"],
                "reason": blocker.get("reason") or payload["reason"],
                "campaign_id": blocker.get("campaign_id") or payload.get("campaign_id")}
            repaired = run_operation(runtime, {**common, "operation": "repair", "hard_pending": [blocker],
                "campaign_id": blocker.get("campaign_id") or "repair-" + common["loop_id"],
                "driver_pending_digest": job["pending_digest"], "affected_activity_id": job["activity_id"],
                "max_heal_attempts": common.get("max_heal_attempts", 1),
                "playbooks_enabled": common.get("playbooks_enabled", True)}, sequence=0, log_event=log_event)
        if payload.get("readiness") or job.get("probe"):
            from scripts.autotrain_readiness_wake import recheck_driver_pending
            recheck_driver_pending(runtime, common, job, repaired=repaired, log_event=log_event)


def bind_continuation_probe(common, job, blocker):
    """Freeze a read-only cursor reproducer; this never clears a repair flag."""
    import sys

    pending = job["payload"]
    store = CampaignStore(pending["campaign_id"], Path(common["root"]))
    inputs = {"continuation": {"campaign_id": store.campaign_id, "input_digest": pending["input_digest"],
                              "blocked_state_digest": pending["wake"]["identity_digest"]},
              "cwd": common["cwd"], "loop_id": common["loop_id"]}
    path = store.write_artifact("driver_pending_inputs", inputs)
    job["probe"] = {"input_kind": "driver_pending_inputs", "input_path": str(path),
        "input_artifact_sha256": path.stem, "input_campaign_id": store.campaign_id,
        "readiness_campaign_id": None, "request_digest": pending["input_digest"]}
    return {**blocker, "original_reproducer": {"argv": [sys.executable, "-m", "scripts.autotrain_readiness_probe",
        "--input", str(path), "--input-digest", path.stem, "--root", common["root"]], "cwd": common["cwd"]}}


def verification_wait(runtime, lease, payload):
    """Queue the verifier's exact dependency without relaunching the repair agent."""
    dependencies = [row["verification_dependency"] for row in (payload or {}).get("agent_repairs", [])
                    if row.get("status") == "waiting_verification" and row.get("verification_dependency")]
    if not dependencies:
        return None
    if len(dependencies) != 1:
        raise ValueError("one repair operation must yield one verification dependency")
    dependency = dependencies[0]
    if dependency.get("schema_version") != "repair_verification_dependency/v1":
        raise ValueError("unknown source verification dependency")
    wake = WakeCondition.model_validate(dependency["wake"])
    identity = dependency.get("verification_identity")
    if identity is not None and (wake.identity_digest != identity
                                or dependency.get("activity_id") != "source-verification-" + identity):
        raise ValueError("source verification dependency identity mismatch")
    artifact = runtime.store.write_artifact("source_verification_requests", dependency)
    runtime.store.append_event("source_verification_requested", experiment_id=lease.activity_id,
        artifact_sha256=artifact.stem, detail={"dependency_digest": artifact.stem,
                                             "repair_activity_id": lease.activity_id},
        idempotency_key=f"source-verification:{lease.activity_id}:{artifact.stem}")
    return (ActivityOutcome.DEPENDENCY if identity and dependency.get("grant") else ActivityOutcome.CAPABILITY, wake)
