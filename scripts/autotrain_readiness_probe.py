"""Read-only, executable original readiness predicate over frozen controller inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.autoresearch.storage import CampaignStore
from slm_training.data.readiness_contract import evidence_digest, load_locked_readiness_request


def probe_inputs(inputs, *, root, request_digest=None, readiness_campaign_id=None):
    """Recompile actual arms and recheck data; never dispatch, train or publish."""
    from scripts.autotrain_readiness import _matrix_contexts
    from slm_training.harnesses.train_data.readiness import build_screening_request, check_readiness, snapshot_input

    cwd = Path(inputs["cwd"]).resolve()
    store = CampaignStore(inputs["matrix"]["campaign_id"], root)
    context, version, train_version = _matrix_contexts(inputs["matrix"], store, cwd, root)
    if readiness_campaign_id:
        activity = CampaignStore(readiness_campaign_id, root)
        request = load_locked_readiness_request(activity, wanted=request_digest)
        payload = request.model_dump(mode="json")
        if any(payload[key] != value for key, value in context.items()):
            raise ValueError("current compiled readiness context differs from locked request")
    else:
        request = build_screening_request(root=cwd, campaign_id=store.campaign_id,
            action_id=evidence_digest(inputs), train_version=train_version, eval_version=version,
            minimum=inputs["minimum"], context=context)
    observed = check_readiness(request, root=cwd)
    if observed["ready"]:
        return {"ready": True, "eval_version": version, "observation": observed}
    if readiness_campaign_id:
        destination = cwd / "outputs/data" / request.original.kind / request.successor_id
        if destination.is_dir():
            candidate = snapshot_input(cwd, destination, exposure=request.original.exposure)
            repaired = check_readiness(request, root=cwd, candidate=candidate)
            if repaired["ready"]:
                return {"ready": True, "eval_version": candidate.dataset_id,
                        "observation": repaired, "candidate": candidate.model_dump(mode="json")}
    return {"ready": False, "observation": observed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-digest", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--request-digest")
    parser.add_argument("--readiness-campaign-id")
    args = parser.parse_args(argv)
    try:
        inputs = json.loads(args.input.read_text())
        if evidence_digest(inputs) != args.input_digest:
            raise ValueError("frozen readiness input digest mismatch")
        if inputs.get("continuation"):
            result = probe_continuation(inputs, root=args.root)
        elif inputs.get("screening_constraint"):
            result = probe_screening(inputs)
        else:
            result = probe_inputs(inputs, root=args.root, request_digest=args.request_digest,
                                  readiness_campaign_id=args.readiness_campaign_id)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        result = {"ready": False, "failure_type": type(exc).__name__, "reason": str(exc)}
    print(json.dumps({"schema_version": "matrix_readiness_probe/v1",
                      "input_digest": args.input_digest, **result}, sort_keys=True))
    return 0 if result["ready"] else 1


def probe_screening(inputs):
    from scripts.run_autotrain_continuous import _screening_n_report, _screening_suite_records
    from scripts.autotrain_pending import screening_deficit_report
    from slm_training.autoresearch.climb_policy import load_climb_policy

    policy = load_climb_policy(Path(inputs["policy"]["path"]))
    if policy.identity_dict() != inputs["policy"]:
        raise ValueError("locked screening policy changed; no automatic authority migration")
    version = inputs["eval_version"]
    selected = inputs["matrix"].get("selected_experiment_ids")
    if selected is not None and selected != inputs.get("expected_selected_experiment_ids", selected):
        return {"ready": False, "reason": "locked selected arms changed"}
    if inputs["screening_constraint"].get("reason") == "successor_readiness_changed_again":
        return {"ready": False, "reason": "successor readiness requires a new independently verified context"}
    n, report = _screening_n_report(policy, eval_version=version)
    deficit = screening_deficit_report(n, report, _screening_suite_records(version),
        automatic=policy.measurement.get("screening_smoke_n_mode") == "auto")
    return {"ready": deficit is None, "eval_version": version, "observation": {"n": n, "report": report}}


def probe_continuation(inputs, *, root):
    """Use the cursor owner's input checks; never reset state or enlarge its grant."""
    from types import SimpleNamespace
    from scripts.autotrain_cycle_context import CycleJournal, load_context, read_artifact, verify_inputs
    from slm_training.levers import HARNESS_FINALIZATION_RESERVE_SECONDS

    frozen = inputs["continuation"]
    store = CampaignStore(frozen["campaign_id"], root)
    events = store.verify_event_chain()
    if not any(e["event_type"] == "driver_cycle_locked" for e in events):
        return {"ready": False, "reason": "missing canonical driver context lock"}
    value = load_context(store, frozen["input_digest"])
    verify_inputs(store, Path(inputs["cwd"]), value)
    states = [e for e in events if e["event_type"] == "driver_cycle_checkpoint"]
    if not states:
        return {"ready": False, "reason": "missing canonical driver cursor"}
    state_digest = states[-1]["artifact_sha256"]
    state = read_artifact(store, "driver_cycle_state", state_digest)
    CycleJournal._validate(SimpleNamespace(state=state, value=value))
    blocked = read_artifact(store, "driver_cycle_state", frozen["blocked_state_digest"])
    restored = (state["input_digest"] == frozen["input_digest"]
        and state_digest != frozen["blocked_state_digest"]
        and state["phase"] in {"arms", "finalizing"}
        and state.get("inflight") is None and not state.get("repair_required")
        and value["total_seconds"] - state["spent_seconds"] > HARNESS_FINALIZATION_RESERVE_SECONDS)
    if restored and blocked.get("repair_required") in {"driver_pending_no_progress", "continuation_no_progress"}:
        restored = _verified_cursor_advance(store, value, events, frozen, blocked, state)
    return {"ready": restored, "state_digest": state_digest,
            "reason": "current canonical cursor restored" if restored else "original continuation predicate remains unmet"}


def _verified_cursor_advance(store, value, events, frozen, blocked, state):
    """Clearing a flag is not evidence that the no-progress predicate changed."""
    from scripts.autotrain_cycle_execution import _new_outcome

    boundary = next(i for i, row in enumerate(events) if row["event_type"] == "driver_cycle_checkpoint"
                    and row["artifact_sha256"] == frozen["blocked_state_digest"])
    prior = {row["event_id"] for row in events[:boundary + 1]}
    if state["index"] == blocked["index"]:
        return _verified_pending_advance(store, value, prior, blocked, state)
    if state["index"] < blocked["index"]:
        return False
    for eid in value["order"][blocked["index"]:state["index"]]:
        outcome = _new_outcome(store, prior, eid, value["arms"][eid]["manifest_digest"])
        if outcome.exit_code != state["arm_exits"][eid]:
            return False
    return True


def _verified_pending_advance(store, value, prior, blocked, state):
    from scripts.autoresearch_command_cursor import yielded_outcome_since
    from scripts.autoresearch_continuation import _advanced, _pending_stage
    from slm_training.autoresearch.schemas import ExperimentOutcome

    if not blocked.get("last_yield") or not state.get("last_yield"):
        return False
    eid = value["order"][state["index"]]
    current = yielded_outcome_since(store, prior, eid, value["arms"][eid]["manifest_digest"])
    if current.model_dump(mode="json") != state["last_yield"]:
        return False
    before = _pending_stage(ExperimentOutcome.model_validate(blocked["last_yield"]))
    after = _pending_stage(current)
    return before is not None and after is not None and _advanced(before, after)


if __name__ == "__main__":
    raise SystemExit(main())
