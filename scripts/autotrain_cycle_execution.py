"""Same-campaign driver reentry; the existing cmd_run owns full command cursors."""

from pathlib import Path
import hashlib
import json
import sys
import time
from types import SimpleNamespace

from scripts.autoresearch_command_cursor import yielded_outcome_since
from scripts.autoresearch_continuation import (
    _pending_stage,
    _progress,
    is_continuation_pending,
)
from scripts.autotrain_cycle_context import (
    CycleJournal,
    active_reference,
    load_context,
    retire,
    verify_inputs,
    writer,
)
from scripts.autotrain_cycle_finalize import execute_stage, stages
from scripts.autotrain_cycle_reconcile import (
    accept_arm,
    new_outcome as _new_outcome,
    reconcile_inflight,
    return_attempts,
    start_attempts,
)
from slm_training.autoresearch.schemas import ExperimentOutcome
from slm_training.autoresearch.storage import CampaignStore
from slm_training.levers import (
    HARNESS_FINALIZATION_RESERVE_SECONDS,
    INTERRUPT_AFTER_SECONDS,
)


def continuous_owner(namespace):
    """Preserve dynamic-import callers and their patched canonical owner symbols."""
    return sys.modules.get(namespace["__name__"]) or SimpleNamespace(**namespace)


def _run_arm(journal, continuous, cwd, deadline):
    value, state, store = journal.value, journal.state, journal.store
    eid = value["order"][state["index"]]
    arm = value["arms"][eid]
    cmd = list(arm["cmd"])
    allowance_index = cmd.index("--experiment-wall-seconds") + 1
    # Invocation allowance is incidental; cmd_run's persisted logical grant,
    # manifest, experiment and compiled command plan remain exactly unchanged.
    cmd[allowance_index] = str(
        min(
            float(cmd[allowance_index]),
            deadline - time.monotonic() - HARNESS_FINALIZATION_RESERVE_SECONDS,
        )
    )
    before = {event["event_id"] for event in store.verify_event_chain()}
    attempts = start_attempts(journal, eid)
    result = continuous._stage_command(
        cmd,
        cwd=cwd,
        deadline=deadline,
        root=store.root.parent,
        loop_id=value["loop_id"],
        stage=f"experiment:{eid}",
    )
    if result.stdout:
        print(
            result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True
        )
    if result.stderr:
        print(
            result.stderr,
            file=sys.stderr,
            end="" if result.stderr.endswith("\n") else "\n",
            flush=True,
        )
    code = (
        124
        if result.timed_out
        else int(result.returncode if result.returncode is not None else 127)
    )
    return_attempts(store, eid, attempts, code)
    if code == 10:
        outcome = yielded_outcome_since(store, before, eid, arm["manifest_digest"])
        if not is_continuation_pending(outcome):
            raise ValueError("driver exit 10 lacks a valid current pending outcome")
        previous = state.get("last_yield")
        if previous is not None:
            old = _pending_stage(ExperimentOutcome.model_validate(previous)) or {}
            new = _pending_stage(outcome) or {}
            if _progress(old) == _progress(new):
                state["repair_required"] = "driver_pending_no_progress"
        state["last_yield"] = outcome.model_dump(mode="json")
        return False
    outcome = _new_outcome(store, before, eid, arm["manifest_digest"])
    return accept_arm(journal, continuous, eid, outcome, code)


def _budget_pending(journal, deadline):
    grant = journal.store.load_campaign().budget.continuation_grant
    if grant and journal.state["attempt"] >= grant.max_attempts:
        return journal.pending("driver_total_attempts_exhausted", capability=True)
    reserve = HARNESS_FINALIZATION_RESERVE_SECONDS * (
        3 if journal.state["phase"] == "arms" else 1
    )
    if journal.remaining <= reserve:
        return journal.pending("driver_logical_grant_exhausted", capability=True)
    if deadline - time.monotonic() <= reserve:
        return journal.pending("driver_invocation_yielded")
    return None


def _advance(journal, continuous, cwd, deadline):
    value, state = journal.value, journal.state
    if state["phase"] == "arms" and state["index"] == len(value["order"]):
        state["phase"] = "finalizing"
    if state["phase"] == "arms":
        complete = _run_arm(journal, continuous, cwd, deadline)
    else:
        plan = stages(value)
        if state["final_index"] == len(plan):
            state["outputs"] = _outputs(journal.store)
            state["phase"] = "completed"
            return True
        complete = (
            execute_stage(
                plan[state["final_index"]], journal, continuous, cwd, deadline
            )
            is not False
        )
        if complete:
            state["final_index"] += 1
    return complete


def _outputs(store):
    from scripts.autotrain_ledgers import validate_cycle_delivery
    from slm_training.autoresearch.schemas import AutotrainCycleHandoffV1

    value = load_context(store)
    result = {}
    for name in ("sdlc_delivery.json", "cycle_handoff.json"):
        path = store.root / name
        raw = path.read_bytes()
        payload = json.loads(raw)
        if (
            not isinstance(payload, dict)
            or payload.get("campaign_id") != store.campaign_id
        ):
            raise ValueError("driver finalization omitted its current outputs")
        if name == "cycle_handoff.json":
            handoff = AutotrainCycleHandoffV1.model_validate(payload)
            if handoff.loop_id != value["loop_id"]:
                raise ValueError("driver handoff differs from locked loop")
        else:
            validate_cycle_delivery(
                payload, campaign_id=store.campaign_id, loop_id=value["loop_id"]
            )
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def execute_pending(journal, continuous, cwd, deadline):
    if journal.state.get("repair_required"):
        return journal.pending(journal.state["repair_required"], capability=True)
    if not reconcile_inflight(journal, continuous):
        return journal.pending(
            "driver_attempt_requires_reconciliation", capability=True
        )
    while journal.state["phase"] != "completed":
        pending = _budget_pending(journal, deadline)
        if pending is not None:
            return pending
        available = operation_allowance(journal, deadline)
        journal.start(journal.state["phase"], available)
        complete = _advance(
            journal,
            continuous,
            cwd,
            time.monotonic() + available - HARNESS_FINALIZATION_RESERVE_SECONDS,
        )
        journal.settle()
        if journal.state.get("repair_required"):
            return journal.pending(journal.state["repair_required"], capability=True)
        if not complete:
            return journal.pending("locked_arm_or_evaluation_yielded")
    continuous._clear_active_stage(journal.store.root.parent, journal.value["loop_id"])
    print(
        f"CYCLE_COMPLETE {journal.store.campaign_id} role={journal.value['role']} "
        f"intent={journal.value['cycle_intent']}",
        flush=True,
    )
    return journal.store.campaign_id


def operation_allowance(journal, deadline):
    available = min(journal.remaining, deadline - time.monotonic())
    state, value = journal.state, journal.value
    if state["phase"] == "arms" and state["index"] < len(value["order"]):
        cmd = value["arms"][value["order"][state["index"]]]["cmd"]
        available = min(
            available,
            float(cmd[cmd.index("--experiment-wall-seconds") + 1])
            + 2 * HARNESS_FINALIZATION_RESERVE_SECONDS,
        )
    elif state["final_index"] < len(stages(value)) and stages(value)[
        state["final_index"]
    ] in {"delivery", "handoff"}:
        available = min(available, 3 * HARNESS_FINALIZATION_RESERVE_SECONDS)
    return available


def resume_cycle(cwd, root, loop_id, continuous, deadline=None):
    """Call before campaign-init/research/matrix; None alone means no saved work."""
    started = time.monotonic()
    deadline = min(
        deadline if deadline is not None else float("inf"),
        started + INTERRUPT_AFTER_SECONDS,
    )
    with writer(root, loop_id) as runtime:
        reference = active_reference(runtime)
        if reference is None:
            return None
        campaign_id = reference["campaign_id"]
        if Path(campaign_id).name != campaign_id or campaign_id in {".", ".."}:
            raise ValueError("invalid referenced driver campaign")
        store = CampaignStore(campaign_id, root)
        value = load_context(store, reference["input_digest"])
        if value["loop_id"] != loop_id or value["cwd"] != str(Path(cwd).resolve()):
            raise ValueError("driver context workspace/loop mismatch")
        verify_inputs(store, cwd, value)
        journal = CycleJournal(store, value, started=started)
        if journal.state["phase"] == "completed":
            if journal.state.get("outputs") != _outputs(store):
                raise ValueError("completed driver outputs changed")
            retire(runtime, reference["input_digest"])
            return store.campaign_id
        result = execute_pending(journal, continuous, cwd, deadline)
        if isinstance(result, str):
            retire(runtime, reference["input_digest"])
        return result


def completed_cycle_since(cwd, root, loop_id, campaign_id, event_ids):
    """Prove this invocation completed the same locked campaign, not a new slug.

    The supervisor still owns its source check, operation lease and output
    envelope. A prior completion, a retirement log alone or changed artifacts
    cannot discharge its terminal-output predicate.
    """
    runtime = CampaignStore("runtime", Path(root) / "loops" / loop_id)
    events = runtime.verify_event_chain()
    registered = {
        e["detail"]["input_digest"]: e["detail"]["campaign_id"]
        for e in events
        if e["event_type"] == "driver_cycle_registered"
    }
    retired = [
        e
        for e in events
        if e["event_type"] == "driver_cycle_retired"
        and e["event_id"] not in event_ids
        and registered.get(e["detail"]["input_digest"]) == campaign_id
    ]
    if len(retired) != 1:
        raise ValueError("same-campaign completion lacks one current retirement")
    store = CampaignStore(campaign_id, root)
    value = load_context(store, retired[0]["detail"]["input_digest"])
    if value["loop_id"] != loop_id or value["cwd"] != str(Path(cwd).resolve()):
        raise ValueError("completed driver context workspace/loop mismatch")
    verify_inputs(store, cwd, value)
    journal = CycleJournal(store, value)
    if journal.state["phase"] != "completed" or journal.state.get(
        "outputs"
    ) != _outputs(store):
        raise ValueError("same-campaign completion lacks validated terminal outputs")
    return {
        "input_digest": journal.digest,
        "retirement_event_id": retired[0]["event_id"],
        "outputs": journal.state["outputs"],
    }


def cycle_event_ids(root, loop_id):
    runtime = CampaignStore("runtime", Path(root) / "loops" / loop_id)
    return frozenset(e["event_id"] for e in runtime.verify_event_chain())


def pass_completion(root, loop_id, campaign_id, boundary):
    """Legacy/no-op passes have no proof; malformed new retirement fails closed."""
    if boundary is None or campaign_id is None:
        return None
    cwd, prior = boundary
    runtime = CampaignStore("runtime", Path(root) / "loops" / loop_id)
    if not any(
        e["event_type"] == "driver_cycle_retired" and e["event_id"] not in prior
        for e in runtime.verify_event_chain()
    ):
        return None
    return completed_cycle_since(cwd, root, loop_id, campaign_id, prior)


def driver_operation(request, continuous, cwd, root, loop_id):
    from scripts.autotrain_pending import pending_since
    from scripts.autotrain_supervisor_operations import operation_publication_scope

    journal = CampaignStore("runtime", root / "loops" / loop_id)
    prior_events = {row["event_id"] for row in journal.verify_event_chain()}
    previous_campaign_id = continuous._latest_cycle(root, loop_id)[1]
    with operation_publication_scope(request, root, loop_id):
        returncode = continuous.main(request["driver_argv"])
    if type(returncode) is int and returncode == 10:
        return {
            "returncode": 10,
            "campaign_id": previous_campaign_id,
            "pending": pending_since(journal, prior_events),
        }
    if type(returncode) is not int or returncode != 0:
        raise ValueError(
            f"driver operation returned unsuccessful status: {returncode!r}"
        )
    campaign_id = continuous._latest_cycle(root, loop_id)[1]
    # Exit zero alone is not a completed experiment. The canonical handoff
    # remains authoritative; this receipt records only operational output.
    handoff = root / str(campaign_id) / "cycle_handoff.json"
    if returncode == 0 and not handoff.is_file():
        raise ValueError("driver exited zero without its required handoff")
    completion = None
    if campaign_id == previous_campaign_id:
        completion = completed_cycle_since(
            cwd, root, loop_id, campaign_id, prior_events
        )
    return {
        "returncode": returncode,
        "campaign_id": campaign_id,
        "completion": completion,
        "handoff_digest": hashlib.sha256(handoff.read_bytes()).hexdigest()
        if handoff.is_file()
        else None,
    }
