"""Reconcile committed driver effects, never infer completion from loose files."""

from scripts.autoresearch_continuation import is_continuation_pending
from scripts.autotrain_cycle_context import read_artifact
from scripts.autotrain_cycle_finalize import stages, skipped
from scripts.autotrain_ledgers import publish_cycle_delivery, validate_cycle_delivery
from slm_training.autoresearch.preflight.compiled_treatment import begin_attempt
from slm_training.autoresearch.campaign_events import publish_cycle_handoff
from slm_training.autoresearch.schemas import AutotrainCycleHandoffV1, ExperimentOutcome


def new_outcome(store, before, eid, manifest):
    rows = [
        row
        for row in store.verify_event_chain()
        if row["event_type"] == "experiment_finished"
        and row["experiment_id"] == eid
        and row["event_id"] not in before
    ]
    if len(rows) != 1:
        raise ValueError("driver arm lacks one current terminal outcome")
    result = ExperimentOutcome.model_validate(
        read_artifact(store, "outcomes", rows[0]["artifact_sha256"])
    )
    if (
        result.campaign_id != store.campaign_id
        or result.experiment_id != eid
        or result.campaign_manifest_sha256 != manifest
        or is_continuation_pending(result)
    ):
        raise ValueError("driver terminal outcome identity/state mismatch")
    return result


def start_attempts(journal, eid):
    # One workload may supply a shared control to several locked comparisons.
    # Each reference has zero independent-sample increment; it is not a rerun.
    designs = [
        p for p in journal.value["locked_designs"].values() if eid in p["arm_ids"]
    ]
    if not designs:
        raise ValueError("driver arm lacks a locked comparison")
    return [begin_attempt(journal.store, pair, eid) for pair in designs]


def return_attempts(store, eid, attempts, code, *, reconciled=False):
    shared = attempts[0]["attempt_id"]
    for attempt in attempts:
        store.append_event(
            "experiment_attempt_returned",
            experiment_id=eid,
            detail={
                **attempt,
                "exit_code": code,
                "scientific_completion": False,
                "shared_execution_id": shared,
                "reconciled": reconciled,
            },
            idempotency_key=f"attempt-returned:{attempt['attempt_id']}",
        )


def accept_arm(journal, continuous, eid, outcome, code):
    repair = next(
        (
            signal.code
            for signal in outcome.harness_signals
            if signal.code
            in {
                "continuation_no_progress",
                "continuation_reconciliation_required",
                "continuation_total_budget_insufficient",
                "continuation_total_attempts_exhausted",
                "continuation_required_commands_missing",
            }
        ),
        None,
    )
    state = journal.state
    if repair is not None:
        state.update(repair_required=repair, last_yield=outcome.model_dump(mode="json"))
        return False
    if code == 0 and outcome.status != "completed":
        raise ValueError("driver zero exit contradicts its terminal outcome")
    # Attaching is idempotent and never occurs for a merely yielded outcome.
    continuous._attach_screening_eval_nll(
        journal.store.root / "runs" / eid, exit_code=code
    )
    state["arm_exits"][eid] = code
    state["seen"].append(eid)
    state["index"] += 1
    state.pop("last_yield", None)
    return True


def _after_inflight(journal):
    events = journal.store.verify_event_chain()
    checkpoints = [
        i
        for i, event in enumerate(events)
        if event["event_type"] == "driver_cycle_checkpoint"
    ]
    if not checkpoints:
        return []
    # pending() may checkpoint the unresolved attempt again. Locate its FIRST
    # reservation, not a later status projection, and refuse older evidence.
    for index in checkpoints:
        state = read_artifact(
            journal.store, "driver_cycle_state", events[index]["artifact_sha256"]
        )
        if (
            state["attempt"] == journal.state["attempt"]
            and state["inflight"] is not None
        ):
            return events[index + 1 :]
    return []


def _recover_arm(journal, continuous, fresh):
    if journal.state["index"] >= len(journal.value["order"]):
        return False
    eid = journal.value["order"][journal.state["index"]]
    terminal = [
        e
        for e in fresh
        if e["event_type"] == "experiment_finished" and e["experiment_id"] == eid
    ]
    attempts = [
        e["detail"]
        for e in fresh
        if e["event_type"] == "experiment_attempt_started" and e["experiment_id"] == eid
    ]
    if not terminal or not attempts:
        return False
    postprocessing = {
        event["event_type"] for event in fresh if event["experiment_id"] == eid
    }
    if not {"outcome_diagnosed", "hypothesizer_feedback_recorded"} <= postprocessing:
        return False  # The child may have died before its trailing obligations.
    expected = {
        p["design_digest"]
        for p in journal.value["locked_designs"].values()
        if eid in p["arm_ids"]
    }
    if {p["design_digest"] for p in attempts} != expected:
        return False
    before = {e["event_id"] for e in journal.store.verify_event_chain()} - {
        e["event_id"] for e in fresh
    }
    result = new_outcome(
        journal.store, before, eid, journal.value["arms"][eid]["manifest_digest"]
    )
    # The canonical cmd_run maps completed to 0 and other terminal outcomes to
    # 2. The model subprocess's exit code is a separate field, not this mapping.
    code = 0 if result.status == "completed" else 2
    returns = [
        e["detail"]
        for e in fresh
        if e["event_type"] == "experiment_attempt_returned"
        and e["experiment_id"] == eid
    ]
    returned = {row["attempt_id"] for row in returns}
    if not returned:
        return_attempts(journal.store, eid, attempts, code, reconciled=True)
    elif returned != {a["attempt_id"] for a in attempts}:
        return False
    elif len({row["exit_code"] for row in returns}) != 1:
        raise ValueError("reconciled attempt return codes disagree")
    else:
        code = returns[0]["exit_code"]
    return accept_arm(journal, continuous, eid, result, code)


def _recover_publication(journal, fresh):
    plan = stages(journal.value)
    index = journal.state["final_index"]
    if index >= len(plan) or plan[index] not in {"delivery", "handoff"}:
        return False
    name = plan[index]
    kind, event_type = (
        ("cycle_deliveries", "cycle_delivery_published")
        if name == "delivery"
        else ("handoff_revisions", "cycle_handoff_published")
    )
    events = [event for event in fresh if event["event_type"] == event_type]
    if len(events) != 1:
        return False
    record = read_artifact(journal.store, kind, events[0]["artifact_sha256"])
    if name == "delivery":
        validate_cycle_delivery(
            record,
            campaign_id=journal.store.campaign_id,
            loop_id=journal.value["loop_id"],
        )
        # The existing publisher reconciles the same committed pointer/ledger.
        record = publish_cycle_delivery(journal.store.root.parent, record)
        journal.state["delivery"] = {
            **record,
            "arm_exits": journal.state["arm_exits"],
            "arm_skipped": skipped(journal.value),
        }
        if journal.state["promotion_chunks"] is not None:
            return False  # Its derivative needs the existing chunk finalizer.
    else:
        handoff = AutotrainCycleHandoffV1.model_validate(record)
        if (
            handoff.campaign_id != journal.store.campaign_id
            or handoff.loop_id != journal.value["loop_id"]
        ):
            raise ValueError("reconciled handoff differs from locked campaign/loop")
        publish_cycle_handoff(journal.store, handoff)
    journal.state["final_index"] += 1
    return True


def reconcile_inflight(journal, continuous):
    if journal.state["inflight"] is None:
        return True
    fresh = _after_inflight(journal)
    recovered = (
        _recover_arm(journal, continuous, fresh)
        if journal.state["phase"] == "arms"
        else _recover_publication(journal, fresh)
    )
    if not recovered:
        return False
    # No restart-comparable monotonic end time exists. Keep the entire prior
    # bounded reservation charged, rather than refunding unknown spent compute.
    journal.state.update(inflight=None, settled_attempt=True)
    journal.save()
    return True
