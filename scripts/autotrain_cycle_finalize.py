"""Resume existing delivery, selection, champion gates and handoff consumers.

No scientific thresholds or new promotion authority live in this module.
Each returned result is checkpointed by the campaign driver before advancing.
"""

from pathlib import Path
import sys

from slm_training.autoresearch.climb_policy import load_climb_policy


def common(journal):
    value = journal.value
    return {
        "root": journal.store.root.parent,
        "loop_id": value["loop_id"],
        "campaign_id": journal.store.campaign_id,
        "cycle_index": value["cycle"],
    }


def stages(value):
    if (
        value["promotion_chunk_plan"] is not None
        and value["promoting_champion"] is not None
    ):
        return ("chunks", "promotion")
    return (
        "chunks",
        "delivery",
        "multiarm",
        "replay",
        "resolution",
        "handoff",
        "status",
        "retire",
    )


def chunks(journal, continuous, cwd, deadline):
    value, state = journal.value, journal.state
    if value["promotion_chunk_plan"] is None or not state["seen"]:
        return True
    pair = {value["control_eid"], value["candidate_eid"]}
    if value["promoting_champion"] is not None and set(state["seen"]) == pair:
        from scripts.autotrain_promotion_finalize import lock_finalization

        lock_finalization(
            journal.store,
            cwd,
            {
                **{k: v for k, v in common(journal).items() if k != "root"},
                "upstream_commit": value["upstream"],
                "integration_commit": value["integration"],
                "role": value["role"],
                "cycle_intent": value["cycle_intent"],
                "primary_metric": value["effective_primary"],
                "matrix": value["matrix"],
                "entry": value["promoting_champion"],
                "control_id": value["control_eid"],
                "candidate_id": value["candidate_eid"],
                "arm_order": value["scheduled_order"],
                "arm_seed": value["arm_seed"],
                "arm_exits": state["arm_exits"],
                "arm_skipped": skipped(value),
                "formal_status": value["promote_formal_status"],
                "skip_slugs": value["skip_slugs"],
            },
        )
    result = continuous._run_promotion_eval_chunks(
        cwd=cwd,
        root=journal.store.root.parent,
        loop_id=value["loop_id"],
        campaign_id=journal.store.campaign_id,
        plan=value["promotion_chunk_plan"],
        experiment_paths={eid: Path(value["by_id"][eid]) for eid in state["seen"]},
        arm_order=[eid for eid in value["order"] if eid in state["seen"]],
        deadline=deadline,
    )
    state["promotion_chunks"] = result
    return bool(result.get("arms")) and all(
        row["status"] == "complete" for row in result["arms"].values()
    )


def skipped(value):
    return {row["arm_id"]: {"reason": row["reason"]} for row in value["multi_arm_skip"]}


def delivery(journal, continuous, cwd, deadline):
    from scripts.autotrain_search import record_cycle_credit

    record_cycle_credit(journal)
    value, state = journal.value, journal.state
    result = continuous._phase_a_delivery(
        **common(journal),
        cwd=cwd,
        deadline=deadline,
        primary_metric=value["effective_primary"],
        role=value["role"],
        cycle_intent=value["cycle_intent"],
        arm_order=value["scheduled_order"],
        arm_seed=value["arm_seed"],
        control_id=value["control_eid"],
        candidate_id=value["candidate_eid"],
        arm_exits=state["arm_exits"],
        arm_skipped=skipped(value),
    )
    result = {**result, "arm_exits": state["arm_exits"], "arm_skipped": skipped(value)}
    if state["promotion_chunks"] is not None:
        result = continuous._attach_promotion_chunks(result, state["promotion_chunks"])
    state["delivery"] = result


def _winner(journal, continuous):
    value, per_arm, scored = journal.value, {}, []
    metric = value["effective_primary"]
    direction = str(value["role_primary"].get("direction") or "increase")
    for eid in value["screening_candidate_ids"]:
        metrics = continuous._run_metrics(journal.store.root, eid)
        val = metrics.get(metric, metrics.get(metric.rsplit(".", 1)[-1]))
        per_arm[eid] = float(val) if isinstance(val, (int, float)) else None
        if per_arm[eid] is not None:
            scored.append(
                (
                    eid,
                    per_arm[eid],
                    continuous._arm_trainable_params(journal.store.root, eid),
                )
            )
    winner = (
        continuous.select_best_by_primary_then_smallest(scored, direction=direction)
        if scored
        else None
    )
    control = continuous._run_metrics(journal.store.root, value["control_eid"])
    control_value = control.get(metric, control.get(metric.rsplit(".", 1)[-1]))
    win_value = per_arm.get(winner)
    if isinstance(control_value, (int, float)) and isinstance(win_value, (int, float)):
        effect = (
            control_value - win_value
            if direction == "decrease"
            else win_value - control_value
        )
        if effect >= float(value["role_primary"].get("minimum_effect") or 0):
            return winner, per_arm, direction
    return None, per_arm, direction


def multiarm(journal, continuous, cwd, deadline):
    value, state = journal.value, journal.state
    if not value["screening_multi"]:
        return
    winner, per_arm, direction = _winner(journal, continuous)
    if winner:
        state["delivery"]["candidate_id"] = winner
    continuous._exhaust_screening_losers(
        root=journal.store.root.parent,
        loop_id=value["loop_id"],
        policy=load_climb_policy(),
        matrix=value["matrix"],
        control_id=value["control_eid"],
        loser_ids=[eid for eid in value["screening_candidate_ids"] if eid != winner],
        claim_class=value["claim_for_role"],
        primary_metric=value["effective_primary"],
        direction=direction,
        train_version=value["train_version"],
        eval_version=value["eval_version"],
    )
    state["delivery"]["multi_arm"] = {
        "max_arms_per_cycle": int(value["multi_arm_cfg"]["max_arms_per_cycle"]),
        "fitted_candidates": value["fitted_candidate_count"],
        "scheduled_candidates": value["screening_candidate_ids"],
        "constraint": value["multi_arm_constraint"],
        "selection_rule": value["selection_rule_locked"],
        "winner_id": winner,
        "per_arm_primary": per_arm,
        "size_skipped": value["multi_arm_skip"],
    }


def replay(journal, continuous, cwd, deadline):
    from slm_training.autoresearch.schemas import (
        AutotrainActionReceiptV1,
        AutotrainActionV1,
        AutotrainCycleHandoffV1,
    )
    from slm_training.autoresearch.storage import (
        append_autotrain_action_receipt,
        autotrain_action_sha256,
        bind_autotrain_action_evidence,
    )

    value, state = journal.value, journal.state
    original = value["replay"]
    if (
        original is None
        or set(state["arm_exits"]) != set(value["replay_manifests"])
        or not all(code == 0 for code in state["arm_exits"].values())
        or state["delivery"].get("measurement_complete") is not True
    ):
        return
    handoff = AutotrainCycleHandoffV1.model_validate(original["handoff"])
    action = AutotrainActionV1.model_validate(original["action"])
    paths = [
        journal.store.root / "campaign.json",
        journal.store.root / "sdlc_delivery.json",
        *(
            journal.store.root / "manifests" / f"{eid}.json"
            for eid in sorted(state["arm_exits"])
        ),
    ]
    evidence = tuple(
        str(p.relative_to(cwd) if p.is_relative_to(cwd) else p) for p in paths
    )
    append_autotrain_action_receipt(
        journal.store.root.parent,
        AutotrainActionReceiptV1(
            loop_id=value["loop_id"],
            campaign_id=handoff.campaign_id,
            action_index=original["action_index"],
            action_sha256=autotrain_action_sha256(action),
            action_kind="retry_measurement",
            status="completed",
            evidence_uris=evidence,
            evidence=bind_autotrain_action_evidence(
                journal.store.root.parent, handoff, action, evidence
            ),
        ),
    )


def resolution(journal, continuous, cwd, deadline):
    value, state = journal.value, journal.state
    entry = value["open_champion"] or value["replayed_confirmation"]
    if value["open_champion"] is not None or (
        value["promoting_champion"] is None and entry is not None
    ):
        state["resolution"] = continuous._resolve_confirm_result(
            **common(journal), entry=entry, delivery=state["delivery"]
        )
    elif value["promoting_champion"] is not None:
        state["delivery"], state["resolution"] = continuous._resolve_promote_delivery(
            {
                **common(journal),
                "entry": value["promoting_champion"],
                "formal_status": value["promote_formal_status"],
                "arm_exits": state["arm_exits"],
            },
            state["delivery"],
            deadline,
        )
    elif continuous._screening_enqueue_allowed(
        cycle_intent=value["cycle_intent"], replay=value["replay"]
    ):
        state["resolution"] = continuous._enqueue_champion(
            root=journal.store.root.parent,
            loop_id=value["loop_id"],
            delivery=state["delivery"],
            camp_dir=journal.store.root,
        )


def handoff(journal, continuous, cwd, deadline):
    value, state = journal.value, journal.state
    continuous._write_cycle_handoff(
        **common(journal),
        cwd=cwd,
        upstream_commit=value["upstream"],
        integration_commit=value["integration"],
        role=value["role"],
        cycle_intent=value["cycle_intent"],
        primary_metric=value["effective_primary"],
        matrix=value["matrix"],
        delivery=state["delivery"],
        resolution=state["resolution"],
        formal_status=value["promote_formal_status"],
        skip_slugs=set(value["skip_slugs"]),
    )


def status(journal, continuous, cwd, deadline):
    value = journal.value
    continuous._run(
        [
            *value["ar"],
            "status",
            "--loop-id",
            value["loop_id"],
            "--matrix",
            "--last",
            "5",
        ],
        cwd=cwd,
        deadline=deadline,
        root=journal.store.root.parent,
        loop_id=value["loop_id"],
        stage="campaign-status",
    )


def retire(journal, continuous, cwd, deadline):
    value, result = journal.value, journal.state["delivery"]
    try:
        exits = result.get("arm_exits") or {}
        if (
            continuous._HEAL_RESUME_SLUG in str(result.get("candidate_id") or "")
            and exits
            and all(int(code) == 0 for code in exits.values())
        ):
            continuous._retire_i10_heal_arm(
                journal.store.root.parent,
                value["loop_id"],
                reason=f"complete_measurement:{journal.store.campaign_id}",
            )
    except Exception as exc:  # Optional retirement never asserts scientific success.
        print(f"HEAL_RESUME_RETIRE_WARN err={exc!r}", file=sys.stderr, flush=True)


def promotion(journal, continuous, cwd, deadline):
    from scripts.autotrain_promotion_finalize import finalize_promotion

    journal.state["promotion_result"] = finalize_promotion(
        journal.store, cwd, continuous, journal.state["promotion_chunks"], deadline
    )


def execute_stage(name, journal, continuous, cwd, deadline):
    functions = {
        "chunks": chunks,
        "delivery": delivery,
        "multiarm": multiarm,
        "replay": replay,
        "resolution": resolution,
        "handoff": handoff,
        "status": status,
        "retire": retire,
        "promotion": promotion,
    }
    return functions[name](journal, continuous, cwd, deadline)


def closeout_driver(continuous, cwd, root, loop_id):
    """Queue content-bound local evidence; never sync or edit the running release.

    Canonical handoff generation already materializes documents in the campaign
    delivery workspace. Any external publication needs its controller capability;
    environment credentials alone never authorize a closeout side effect.
    """
    import hashlib
    from slm_training.autoresearch.storage import _sha
    from scripts.autotrain_controller_repair import preserve_workspace

    try:
        campaign_id = continuous._latest_cycle(root, loop_id)[1]
        if not campaign_id:
            return None
        directory = Path(root) / campaign_id
        paths = [
            directory / name for name in ("cycle_handoff.json", "sdlc_delivery.json")
        ]
        if not all(path.is_file() for path in paths):
            return None
        evidence = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        }
        return preserve_workspace(
            cwd=cwd,
            root=root,
            loop_id=loop_id,
            reason=f"driver_local_evidence_delivery:{campaign_id}:{_sha(evidence)}",
            paths=tuple(evidence),
        )
    except Exception as exc:
        print(f"DRIVER_DELIVERY_WAIT_WARN {exc!r}", flush=True)
        return None
