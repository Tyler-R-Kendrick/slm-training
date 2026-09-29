"""Explicit host CLI authorization; ordinary retries reuse the recorded successor."""

import hashlib
import json

from scripts.autotrain_controller_execution import journal_authority
from slm_training.harness_core.activity_contract import contract_digest
from slm_training.autoresearch.heal.operation_recovery import wake_verified_operation
from slm_training.autoresearch.heal.operation_timeout import verified_timeout_handoff


def recover_host_timeout(runtime, args, common, controller):
    path, expected = getattr(args, "timeout_recovery", None), getattr(args, "timeout_recovery_sha256", None)
    if path is None and expected is None:
        return
    if path is None or expected is None or controller is None:
        raise ValueError("timeout recovery requires explicit host proof and pinned controller")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("timeout authorization file changed")
    proof = json.loads(raw)
    identity = contract_digest(proof)
    with journal_authority(runtime.store):
        if any(e["event_type"] == "operation_successor_activated" and
               e["detail"]["handoff"].get("activation_id") == identity
               for e in runtime.store.verify_event_chain()):
            raise ValueError("timeout authorization already consumed")
        handoff = verified_timeout_handoff(runtime, (proof, identity), cwd=common["cwd"])
        wake_verified_operation(runtime, handoff, cwd=common["cwd"],
                                controller_execution=controller, timeout_authorization=(proof, identity))


def retained_timeout_successor(runtime, request):
    """Follow authenticated activations from the original CLI request, not one hop."""
    from scripts.autotrain_execution_transition import _operation_edge

    events = runtime.store.verify_event_chain()
    plans = {contract_digest(e["detail"]): e["detail"] for e in events
             if e["event_type"] == "operation_successor_planned"}
    stable = logical_driver_request(request)
    originals = {e["experiment_id"] for e in events
                 if e["event_type"] == "operation_repair_requested"
                 and logical_driver_request(e["detail"]["request"]) == stable}
    roots = [p for p in plans.values()
             if p["handoff"]["resume_activity_id"] in originals
             and p["handoff"].get("schema_version") == "operation_timeout_handoff/v1"]
    if not roots:
        return None
    if len(roots) != 1:
        raise ValueError("timeout successor root ambiguous")
    plan, seen, previous = roots[0], set(), None
    states = runtime.snapshot()
    with journal_authority(runtime.store):
        while True:
            activity = plan["request"]["successor_activity_id"]
            if activity in seen:
                raise ValueError("timeout successor cycle")
            seen.add(activity)
            predecessor = plan["handoff"]["resume_activity_id"]
            siblings = [p for p in plans.values() if p["handoff"]["resume_activity_id"] == predecessor]
            if len(siblings) != 1:
                raise ValueError("timeout successor branch")
            children = [p for p in plans.values() if p["handoff"]["resume_activity_id"] == activity]
            original, successor, _ = _operation_edge(
                runtime.store, plan, events, states, terminal=not children)
            if len(seen) > 1 and original != previous:
                raise ValueError("timeout successor ancestor request changed")
            if not children:
                return retain_driver_configuration(runtime, request, successor)
            if len(children) != 1:
                raise ValueError("timeout successor branch")
            previous, plan = successor, children[0]


def logical_driver_request(request):
    """Host configuration is execution authority, never a new scientific budget."""
    from scripts.autotrain_controller_execution import CONFIGURATION_KEYS

    return {k: v for k, v in request.items()
            if k not in {*CONFIGURATION_KEYS, "predecessor_campaign_id"}}


def retain_driver_configuration(runtime, request, original):
    """Authenticate host rebinding separately; preserve recorded request bytes."""
    from scripts.autotrain_controller_execution import CONFIGURATION_KEYS, authorize_configuration_rebinding

    if any(request.get(key) != original.get(key) for key in CONFIGURATION_KEYS):
        authorize_configuration_rebinding(runtime, request)
    return original
