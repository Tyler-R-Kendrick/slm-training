"""Locked diagnostic comparison and read-only view of canonical prerequisites."""

import json
import math
from pathlib import Path


def locked_startup_commit(cwd):
    """Use authenticated release provenance; an unmarked copy fails closed."""
    from slm_training.harness_core.execution_release import runtime_git_provenance

    provenance = runtime_git_provenance(Path(cwd))
    if not provenance or provenance["code_dirty"] or (
        provenance["upstream_commit"] != provenance["integration_commit"]
    ):
        raise ValueError("locked diagnostic requires a clean authenticated execution release")
    return provenance["integration_commit"]


def require_preregistered_plan(store, plan):
    """Bind the current plan to its single pre-outcome content-addressed event."""
    from slm_training.autoresearch.storage import _sha

    events = store.verify_event_chain()
    locks = [(i, row) for i, row in enumerate(events)
             if row["event_type"] == "science_lab_preregistered"]
    digest = _sha(plan)
    if len(locks) != 1 or locks[0][1]["artifact_sha256"] != digest:
        raise ValueError("locked preregistration differs from preregistered event")
    if any(row["event_type"] in {"experiment_started", "experiment_attempt_started",
                                  "experiment_finished", "experiment_yielded"}
           for row in events[:locks[0][0]]):
        raise ValueError("locked preregistration was recorded after outcomes")
    artifact = store.root / "artifacts" / "science_lab_preregistration" / f"{digest}.json"
    if not artifact.is_file() or json.loads(artifact.read_text()) != plan:
        raise ValueError("locked preregistration artifact differs from event")


def locked_prerequisite_report(root, loop_id, campaign_id):
    """Project receipt-checked canonical actions without scientific soft-heals."""
    from slm_training.autoresearch.action_dependencies import campaign_prerequisites
    from slm_training.autoresearch.heal.classify import classify_blocker
    from slm_training.autoresearch.schemas import AutotrainCycleHandoffV1
    from slm_training.autoresearch.storage import autotrain_action_sha256

    path = Path(root) / campaign_id / "cycle_handoff.json"
    pending, waits = (), []
    if path.is_file():
        handoff = AutotrainCycleHandoffV1.model_validate_json(path.read_text())
        if (handoff.loop_id, handoff.campaign_id) != (loop_id, campaign_id):
            raise ValueError("locked handoff identity differs from campaign")
        pending, waits = campaign_prerequisites(root, handoff)
    blockers = []
    for index, action in pending:
        row = {"campaign_id": campaign_id, "index": index,
               "kind": action.kind, "reason": action.reason,
               "action_sha256": autotrain_action_sha256(action),
               "evidence_ids": list(action.evidence_ids)}
        for field in ("blocker_code", "unmet_predicate", "required_capability",
                      "frozen_manifest_sha256"):
            value = getattr(action, field)
            if value is not None:
                row[field] = value
        row["blocker_class"] = classify_blocker(action.kind, action.reason,
                                                 code=action.blocker_code)
        blockers.append(row)
    return {"hard_pending": blockers, "soft_healed": [], "delivery_waits": waits,
            "predecessor_campaign_id": campaign_id,
            "blocker_cleared": not blockers}


def require_locked_repair(blocker):
    """Only evidence-bound source/environment repairs may precede the frozen pair."""
    from slm_training.autoresearch.heal.classify import classify_blocker

    if blocker.get("kind") not in {"repair_harness", "repair_formal",
                                   "heal_postcondition_failed"}:
        raise ValueError("locked diagnostic repair would mutate its scientific plan")
    if not (blocker.get("action_sha256") or blocker.get("affected_activity_id")):
        raise ValueError("locked diagnostic repair lacks action or activity evidence")
    kind = classify_blocker(str(blocker.get("kind") or ""),
                            str(blocker.get("reason") or ""),
                            code=blocker.get("blocker_code"))
    if kind not in {"environment", "code", "formal_infra", "unknown"}:
        raise ValueError("locked diagnostic repair requires a new preregistration")
    return True


def locked_repair_rows(blockers):
    """Document handoff remains a delivery duty, not a repair or new arm."""
    rows = [row for row in blockers if row.get("kind") != "document"]
    for row in rows:
        require_locked_repair(row)
    return rows


def require_locked_completion(driver, campaign_id):
    if driver["returncode"] != 0:
        raise ValueError("locked diagnostic driver did not exit successfully")
    if driver["campaign_id"] != campaign_id or not driver.get("completion"):
        raise ValueError("locked diagnostic lacks same-campaign completion proof")


def require_locked_eval_selection(store, arm_ids, inputs):
    """Both measured arms must use the exact preregistered six-case selection."""
    expected = inputs["selected_record_ids"]
    if len(expected) != inputs["eval_cases"] or len(set(expected)) != len(expected):
        raise ValueError("locked diagnostic selection is incomplete")
    for arm_id in arm_ids:
        path = store.root / "runs" / arm_id / "eval_nll_records.json"
        if not path.is_file():
            raise ValueError("locked diagnostic paired comparison incomplete")
        record = json.loads(path.read_text())
        selection = record.get("selection") or {}
        if (record.get("schema") != "eval_nll_records/v1"
            or selection.get("selected_record_ids") != expected
            or selection.get("selected_root_ids") != inputs["selected_root_ids"]
            or selection.get("input_sha256s") != inputs["input_sha256s"]
            or selection.get("selection_sha256") != inputs["selection_sha256"]
            or set(record.get("records") or {}) != set(expected)):
            raise ValueError("locked diagnostic eval selection differs from preregistration")


def finalize_diagnostic(journal, continuous):
    """Compare the exact completed pair, then publish diagnostic-only evidence."""
    from scripts.autotrain_ledgers import publish_cycle_delivery
    from scripts.autotrain_measurement import measurement_is_complete
    from slm_training.autoresearch.campaign_events import publish_cycle_handoff
    from slm_training.autoresearch.climb_policy import load_climb_policy, primary_for_role
    from slm_training.autoresearch.schemas import AutotrainActionV1, AutotrainCycleHandoffV1
    from slm_training.versioning import build_version_stamp

    value, state, store = journal.value, journal.state, journal.store
    ids = value["order"]
    if len(ids) != 2 or set(state["arm_exits"]) != set(ids) or any(
        type(state["arm_exits"][eid]) is not int or state["arm_exits"][eid] != 0
        for eid in ids
    ):
        raise ValueError("locked diagnostic requires both successful arm exits")
    plan = json.loads(Path(value["preregistration_path"]).read_text())
    primary = plan["primary"]
    policy = primary_for_role(load_climb_policy(), "screening")
    if (primary["metric"], primary["direction"], primary["minimum_effect"]) != (
        policy["metric"], policy["direction"], policy["minimum_effect"]
    ) or value["effective_primary"] != primary["metric"]:
        raise ValueError("locked diagnostic primary differs from canonical comparison policy")
    require_locked_eval_selection(store, ids, plan["inputs"])
    decision = continuous._classify_positive(
        camp_dir=store.root, primary_metric=primary["metric"],
        control_id=ids[0], candidate_id=ids[1], role="screening",
        observed_sd_path=store.root / "observed_paired_sd.json",
    )
    paired = decision.get("paired_test") or {}
    effect = paired.get("median_delta")
    if (not measurement_is_complete(decision) or not paired.get("diagnostic_complete")
        or paired.get("n_pairs") != plan["inputs"]["eval_cases"]
        or not isinstance(effect, (int, float)) or not math.isfinite(effect)):
        raise ValueError("locked diagnostic paired comparison incomplete")
    stamp = build_version_stamp("harness.autoresearch.experiment_campaign")
    stamp.update(code_commit=value["integration"], code_dirty=False,
                 stamped_at=store.load_campaign().created_at)
    record = publish_cycle_delivery(store.root.parent, {
        **decision, "schema": "autotrain_sdlc_delivery/v1",
        "loop_id": value["loop_id"], "campaign_id": store.campaign_id,
        "cycle_role": "screening", "cycle_intent": "locked_pair_diagnostic",
        "claim_class": "diagnostic", "promotion_allowed": False,
        "stack_layer": False, "measurement_complete": True,
        "arm_order": ids, "arm_exits": state["arm_exits"],
        "measured_effect": effect, "minimum_effect": primary["minimum_effect"],
        "version_stamp": stamp,
        "effect_gate_pass": bool(paired.get("win")),
        "reasons": [*decision.get("reasons", []),
                    "Locked diagnostic only; no promotion or main delivery authority"],
    })
    state["delivery"] = record
    checkpoints = continuous._created_checkpoint_paths(store.root)
    handoff = AutotrainCycleHandoffV1(
        loop_id=value["loop_id"], campaign_id=store.campaign_id,
        cycle_index=value["cycle"], upstream_commit=value["upstream"],
        integration_commit=value["integration"], cycle_role="screening",
        cycle_intent="locked_pair_diagnostic", evidence_class="scratch",
        climb_state="inconclusive",
        ship_state="blocked", primary_metric=primary["metric"],
        reasons=tuple(record["reasons"]),
        actions=(AutotrainActionV1(
            kind="document", owner="documenting-experiment-results",
            reason="Document locked diagnostic results without promotion or shipment claims",
            evidence_ids=(f"campaign:{store.campaign_id}",),
        ),), checkpoint_paths=checkpoints,
        checkpoint_documentation_required=bool(checkpoints),
    )
    publish_cycle_handoff(store, handoff)
