"""Verified implementation transitions preserve logical locks, never source identity."""

from pathlib import Path

from slm_training.autoresearch.storage import CampaignStore, _sha
from slm_training.harness_core.activity_contract import contract_digest


def _operation_edge(journal, plan, events, states, *, terminal):
    from slm_training.autoresearch.heal.repair_release import verified_activation_handoff

    request, handoff = plan["request"], plan["handoff"]
    timeout = handoff.get("schema_version") == "operation_timeout_handoff/v1"
    checked = verified_activation_handoff(journal, handoff) if terminal and not timeout else handoff
    authority = {"plan_digest": contract_digest(plan),
                 "controller_execution": request.get("controller_execution")}
    if not authority["controller_execution"] or not any(
            e["event_type"] == "operation_controller_activation_verified"
            and e["experiment_id"] == request["successor_activity_id"]
            and e["detail"] == authority for e in events):
        raise ValueError("execution_transition_controller_receipt_missing")
    logical = request["logical_continuation"]
    previous = logical["predecessor_activity_id"]
    originals = [e["detail"]["request"] for e in events
                 if e["event_type"] == "operation_repair_requested" and e["experiment_id"] == previous
                 and contract_digest(e["detail"]["request"]) == logical["predecessor_request_digest"]]
    if not originals or any(row != originals[0] for row in originals):
        raise ValueError("execution_transition_original_request_missing")
    original = originals[0]
    before, after = states[previous], states[request["successor_activity_id"]]
    if timeout:
        _timeout_binding(journal, handoff, original, before, events)
    activation = checked.get("activation_id", checked["publication_id"])
    expected_event = {"handoff": checked, "successor_request_digest": contract_digest(request)}
    changed = {"cwd", "source_digest", "environment_digest", "successor_activity_id",
               "resource_grant", "logical_continuation", "controller_execution"}
    if (checked["resume_activity_id"] != previous or logical["activation_id"] != activation
            or logical["schema_version"] != "operation_continuation/v1"
            or logical["scientific_replicate_increment"] != 0
            or before.status != "cancelled" or (terminal and after.status == "cancelled")
            or before.spec.input_digest != contract_digest(original)
            or before.spec.source_digest != original["source_digest"]
            or before.spec.environment_digest != original["environment_digest"]
            or after.spec.model_dump(mode="json") != plan["spec"]
            or after.spec.input_digest != contract_digest(request)
            or after.spec.source_digest != checked["source_digest"]
            or request["source_digest"] != checked["source_digest"]
            or request["cwd"] != checked["successor_execution"]
            or request["environment_digest"] != after.spec.environment_digest
            or {k: v for k, v in request.items() if k not in changed}
               != {k: v for k, v in original.items() if k not in changed}
            or not any(e["event_type"] == "operation_successor_activated"
                       and e["experiment_id"] == previous and e["detail"] == expected_event for e in events)):
        raise ValueError("execution_transition_operation_binding_mismatch")
    grant = before.spec.grant.model_dump(mode="json")
    spec_changes = {"activity_id", "source_digest", "environment_digest", "input_digest", "output_namespace", "grant"}
    if ({k: v for k, v in before.spec.model_dump(mode="json").items() if k not in spec_changes}
            != {k: v for k, v in after.spec.model_dump(mode="json").items() if k not in spec_changes}):
        raise ValueError("execution_transition_operation_contract_changed")
    grant.update(total_seconds=grant["total_seconds"] - before.charged_seconds,
                 max_attempts=grant["max_attempts"] - before.attempts)
    prior = original.get("logical_continuation", {})
    if (request["resource_grant"] != grant or after.spec.grant.model_dump(mode="json") != grant
            or logical["logical_activity_id"] != prior.get("logical_activity_id", previous)
            or logical["logical_request_digest"] != prior.get("logical_request_digest", contract_digest(original))
            or logical["prior_attempts"] != prior.get("prior_attempts", 0) + before.attempts
            or logical["prior_charged_seconds"] != prior.get("prior_charged_seconds", 0) + before.charged_seconds
            or logical["logical_resource_grant"] != prior.get("logical_resource_grant", before.spec.grant.model_dump(mode="json"))):
        raise ValueError("execution_transition_grant_changed")
    transition = checked.get("environment_transition")
    expected_environment = original["environment_digest"]
    if transition is not None:
        if transition["predecessor_digest"] != expected_environment:
            raise ValueError("execution_transition_environment_lineage_changed")
        expected_environment = transition["successor_digest"]
    if request["environment_digest"] != expected_environment:
        raise ValueError("execution_transition_environment_lineage_changed")
    return original, request, checked


def _timeout_binding(journal, handoff, original, before, events):
    """Consumed host proof is bound by the live controller activation receipt."""
    proof = handoff["timeout_authorization"]
    identity = contract_digest(proof)
    cancellations = [e for e in events if e["event_id"] == proof["cancel_event_id"]]
    if (proof["schema_version"] != "host_timeout_recovery/v1"
            or proof["cause"] != "host_run_timeout"
            or Path(proof["journal_root"]).resolve() != journal.root.resolve()
            or proof["state_digest"] != contract_digest(before)
            or proof["request_digest"] != contract_digest(original)
            or handoff["activation_id"] != identity or handoff["publication_id"] != identity
            or len(cancellations) != 1
            or cancellations[0]["experiment_id"] != before.spec.activity_id
            or cancellations[0]["detail"].get("sequence") != before.sequence):
        raise ValueError("execution_transition_timeout_binding_changed")


def _unchanged_science(original, request, handoff):
    from slm_training.harness_core.execution_release import _runtime_manifest
    from slm_training.autoresearch.heal.repair_scope import worker_changes_with_valid_overlay, repair_classification

    base, current = Path(original["cwd"]), Path(request["cwd"])
    before, after = _runtime_manifest(base), _runtime_manifest(current)
    if (before is None or after is None or before["source_digest"] != original["source_digest"]
            or after["source_digest"] != request["source_digest"]):
        raise ValueError("execution_transition_materialization_changed")
    if handoff.get("schema_version") == "operation_timeout_handoff/v1":
        if (base.resolve() != current.resolve() or before != after
                or original["source_digest"] != request["source_digest"]
                or original["environment_digest"] != request["environment_digest"]
                or handoff.get("environment_transition") is not None):
            raise ValueError("execution_transition_timeout_source_changed")
        return
    changed = tuple(sorted(name for name in before["files"].keys() | after["files"].keys()
                           if before["files"].get(name) != after["files"].get(name)))
    worker = worker_changes_with_valid_overlay(base, current, changed, handoff["original_request_digest"])
    # Only orchestration repairs can retain previously executed scientific work.
    # Source acceptance alone does not prove arbitrary implementation equivalence.
    if (not worker or repair_classification(worker) != "implementation_repair"
            or any(not (name.startswith("scripts/autotrain_")
                        or name in {"scripts/autoresearch_continuation.py", "scripts/autoresearch_command_cursor.py"}
                        or name.startswith("tests/") and name not in before["files"])
                   for name in worker)):
        raise ValueError("execution_transition_scientific_implementation_changed")


def _verify_locked_execution(store, physical_cwd, locked_inputs):
    """Authorize only an activated repair of this exact existing driver context."""
    from scripts.merge_verification_evidence import digest
    from slm_training.autoresearch.runtime.activity_projection import ActivityProjection
    from scripts.autotrain_cycle_lock import load_context

    if load_context(store) != locked_inputs:
        raise ValueError("execution_transition_driver_lock_changed")
    loop_id = locked_inputs["loop_id"]
    journal = CampaignStore("runtime", store.root.parent / "loops" / loop_id)
    events = journal.verify_event_chain()
    states = ActivityProjection(journal).read()
    plans = {_sha(e["detail"]): e["detail"] for e in events
             if e["event_type"] == "operation_successor_planned"}
    endpoints = [p for p in plans.values()
                 if Path(p["request"]["cwd"]).resolve() == Path(physical_cwd).resolve()
                 and states[p["request"]["successor_activity_id"]].status != "cancelled"]
    if len(endpoints) != 1:
        raise ValueError("execution_transition_missing_or_ambiguous")
    request = endpoints[0]["request"]
    from slm_training.harness_core.controller_execution import validated_controller_environment

    _, environment = validated_controller_environment(request["controller_execution"])
    _verify_ancestry(journal, endpoints[0], plans.values(), events, states, store, locked_inputs)
    if digest(environment) != request["environment_digest"]:
        raise ValueError("execution_transition_campaign_or_environment_mismatch")
    return request


def _verify_ancestry(journal, endpoint, plans, events, states, store, locked_inputs):
    plan, seen = endpoint, set()
    while True:
        successor = plan["request"]["successor_activity_id"]
        if successor in seen:
            raise ValueError("execution_transition_cycle")
        seen.add(successor)
        original, request, handoff = _operation_edge(
            journal, plan, events, states, terminal=plan == endpoint)
        predecessor = request["logical_continuation"]["predecessor_activity_id"]
        branches = [p for p in plans if p["request"]["logical_continuation"]["predecessor_activity_id"] == predecessor]
        if len(branches) != 1:
            raise ValueError("execution_transition_branch")
        if (original["operation"] != "driver" or original["loop_id"] != locked_inputs["loop_id"]
                or Path(original["root"]).resolve() != store.root.parent.resolve()
                or (handoff["pending"]["campaign_id"]
                    if handoff.get("schema_version") == "operation_timeout_handoff/v1"
                    else handoff["resume_campaign_id"]) != store.campaign_id):
            raise ValueError("execution_transition_campaign_mismatch")
        if (handoff.get("schema_version") == "operation_timeout_handoff/v1"
                and handoff["pending"]["input_digest"] != contract_digest(locked_inputs)):
            raise ValueError("execution_transition_timeout_driver_lock_changed")
        _unchanged_science(original, request, handoff)
        if not original.get("logical_continuation"):
            source = locked_inputs.get("publication_source")
            if (Path(original["cwd"]).resolve() != Path(locked_inputs["cwd"]).resolve()
                    or source is not None and source["source_digest"] != original["source_digest"]):
                raise ValueError("execution_transition_original_lock_mismatch")
            return
        parents = [p for p in plans if p["request"] == original]
        if len(parents) != 1:
            raise ValueError("execution_transition_ancestor_missing_or_ambiguous")
        plan = parents[0]


def logical_cursor_inputs(store, inputs):
    """Keep historical cursor hash; physical execution remains the caller's cwd."""
    from scripts.autotrain_cycle_lock import load_context
    from scripts.autoresearch_continuation_identity import resolved_continuation_grant

    if store is None:
        return inputs
    value = load_context(store)
    if value is None or inputs["cwd"] == value["cwd"]:
        return inputs
    verify_locked_execution(store, inputs["cwd"], value)
    actual = resolved_continuation_grant(Path(inputs["cwd"]), inputs["total"]).execution_identity
    arm = value["arms"].get(inputs["experiment"]["experiment_id"])
    grant = store.load_campaign().budget.continuation_grant
    if (actual != inputs["identity"] or arm is None or arm["commands"] != inputs["commands"]
            or arm["manifest_digest"] != inputs["manifest"] or inputs["total"] != value["total_seconds"]
            or inputs.get("max_attempts") != (grant.max_attempts if grant else None)):
        raise ValueError("execution_transition_cursor_contract_changed")
    return {**inputs, "cwd": value["cwd"], "identity": value["execution_identity"]}


def preregistration_source(store, physical_cwd, plan):
    """Validate original evidence at its original release, then current authority."""
    from scripts.autotrain_cycle_lock import load_context

    source = Path(plan["source_path"]).resolve()
    if source != Path(physical_cwd).resolve():
        value = load_context(store)
        if value is None or Path(value["cwd"]).resolve() != source:
            raise ValueError("execution_transition_preregistration_context_missing")
        verify_locked_execution(store, physical_cwd, value)
    return source


def verify_locked_execution(store, physical_cwd, locked_inputs):
    """Only a live controller fence may select the authoritative chain."""
    from scripts.autotrain_controller_execution import journal_authority

    journal = CampaignStore("runtime", store.root.parent / "loops" / locked_inputs["loop_id"])
    with journal_authority(journal):
        return _verify_locked_execution(store, physical_cwd, locked_inputs)
