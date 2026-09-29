"""Bounded continuation of the existing promotion evaluator.

Campaign events own the chunk ledger; promotion_chunks.json is a recoverable
projection. A launch is charged before execution, including a lost worker.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import time
from pathlib import Path

from slm_training.autoresearch.storage import CampaignStore
from slm_training.levers import INTERRUPT_AFTER_SECONDS
from slm_training.lineage.records import canonical_json

LEDGER_SCHEMA = "autotrain_promotion_chunks/v2"
LEDGER_NAME = "promotion_chunks.json"
FINALIZATION_SECONDS = 5.0


def _digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _save(store, ledger):
    artifact = store.write_artifact("promotion_chunks", ledger)
    sha = artifact.stem
    store.append_event(
        "promotion_chunk_state",
        artifact_sha256=sha,
        detail={"ledger_sha256": sha},
        idempotency_key=f"promotion-chunks:{sha}",
    )
    store._replace_durable(
        store.root / LEDGER_NAME,
        json.dumps(ledger, sort_keys=True, allow_nan=False) + "\n",
    )


def load_ledger(store):
    events = [
        row
        for row in store.verify_event_chain()
        if row["event_type"] == "promotion_chunk_state"
    ]
    if not events:
        if (store.root / LEDGER_NAME).exists():
            raise ValueError(
                "legacy chunk ledger requires explicit budget reconciliation"
            )
        return None
    sha = events[-1]["artifact_sha256"]
    path = store.root / "artifacts" / "promotion_chunks" / f"{sha}.json"
    ledger = json.loads(path.read_text())
    if _digest(ledger) != sha:
        raise ValueError("promotion chunk ledger integrity mismatch")
    if ledger.get("schema") != LEDGER_SCHEMA:
        raise ValueError("unsupported promotion chunk ledger schema")
    return ledger


def _initialize(context, command_factory):
    plan = context["plan"]
    for key in ("run_n", "records_per_run"):
        if type(plan.get(key)) is not int or plan[key] <= 0:
            raise ValueError(f"invalid promotion chunk {key}")
    arms = {}
    for eid in context["arm_order"]:
        run_dir = context["camp_dir"] / "runs" / eid
        cmd = command_factory(
            root=context["root"],
            campaign_id=context["campaign_id"],
            experiment_path=context["experiment_paths"][eid],
            run_dir=run_dir,
            plan=plan,
        )
        checkpoint = (
            Path(cmd[cmd.index("--checkpoint") + 1])
            if "--checkpoint" in cmd
            else run_dir / "checkpoints" / "last.pt"
        )
        arms[eid] = {
            "run_dir": str(run_dir),
            "run_budget": plan["run_n"],
            "runs_used": 0,
            "runs": [],
            "status": "pending",
            "command": list(cmd),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if checkpoint.is_file()
            else None,
        }
    return {
        "schema": LEDGER_SCHEMA,
        "campaign_id": context["campaign_id"],
        "plan": dict(plan),
        "arms": arms,
        "arm_order": list(context["arm_order"]),
    }


def _launch(store, context, arm, eid, *, deadline, stage_runner, scoreboard):
    index = arm["runs_used"] + 1
    before = scoreboard(Path(arm["run_dir"]))
    reserved = min(float(INTERRUPT_AFTER_SECONDS), deadline - time.monotonic())
    row = {
        "index": index,
        "state": "launched",
        "reserved_seconds": reserved,
        "pending_before": before["pending"],
    }
    arm["runs"].append(row)
    arm["runs_used"] = index
    arm["status"] = "running"
    _save(store, context["ledger"])
    result = stage_runner(
        arm["command"],
        cwd=context["cwd"],
        deadline=deadline,
        root=context["root"],
        loop_id=context["loop_id"],
        stage=f"promotion-chunk:{eid}:{index}",
    )
    code = (
        124
        if result.timed_out
        else (127 if result.returncode is None else result.returncode)
    )
    after = scoreboard(Path(arm["run_dir"]))
    row.update(
        state="finished",
        exit_code=code,
        timed_out=bool(result.timed_out),
        duration_seconds=result.duration_seconds,
        pending_after=after["pending"],
        decoded_this_run_n=after["decoded"],
        measurement_complete=after["complete"],
    )
    # evaluate_model uses 2..8 for measured threshold/gate rejection and 10
    # for incomplete resumable work; none of these says the model improved.
    if result.timed_out or code in (124, 130, 143, -2, -9, -15):
        arm["status"] = "pending"
        row["interrupted"] = True
    elif code not in (0, 2, 3, 4, 5, 6, 7, 8, 10) or not after["exists"]:
        arm.update(
            status="harness_failure",
            error=f"chunk {index} exit={code} scoreboard={after['exists']}",
        )
    else:
        arm["status"] = "complete" if after["complete"] else "pending"
    _save(store, context["ledger"])


def _advance(store, context, *, deadline, stage_runner, scoreboard):
    ledger = context["ledger"]
    # ponytail: sequential arms; the controller schedules other families between invocations.
    for eid in ledger["arm_order"]:
        arm = ledger["arms"][eid]
        checkpoint = Path(arm["checkpoint"])
        if not checkpoint.is_file():
            arm["status"] = "no_checkpoint"
            continue
        if (
            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            != arm["checkpoint_sha256"]
        ):
            raise ValueError("promotion chunk checkpoint changed")
        while arm["status"] not in {"harness_failure", "complete"}:
            state = scoreboard(Path(arm["run_dir"]))
            if state["complete"]:
                arm["status"] = "complete"
                break
            if arm["runs_used"] >= arm["run_budget"]:
                arm["status"] = "chunk_budget_exhausted"
                break
            available = deadline - time.monotonic()
            if available < float(ledger["plan"]["chunk_wall_seconds"]):
                arm["status"] = "invocation_yield"
                break
            _launch(
                store,
                context,
                arm,
                eid,
                deadline=deadline,
                stage_runner=stage_runner,
                scoreboard=scoreboard,
            )
        if arm["status"] == "complete":
            _bind_completion(arm, scoreboard)
    _save(store, ledger)
    return ledger


def _bind_completion(arm, scoreboard):
    run_dir = Path(arm["run_dir"])
    if not scoreboard(run_dir)["complete"]:
        raise ValueError("completed promotion scoreboard is no longer complete")
    sha = hashlib.sha256((run_dir / "scoreboard.json").read_bytes()).hexdigest()
    if arm.get("completed_scoreboard_sha256", sha) != sha:
        raise ValueError("completed promotion scoreboard changed")
    arm["completed_scoreboard_sha256"] = sha


def completed_chunk_evidence(camp_dir, experiment_id, outcome):
    """A current completed eval can supersede only an eval-stage interruption.

    Historical outcomes stay untouched. Training/data failures never disappear
    merely because a checkpoint or scoreboard exists.
    """
    stages = outcome.get("stage_telemetry") or []
    training = [row for row in stages if "scripts.train_model" in row.get("command", [])]
    prerequisites = [row for row in stages if "scripts.evaluate_model" not in row.get("command", [])]
    if not training or any(row.get("exit_code") != 0 or row.get("measurement_complete") is False
                           or row.get("timed_out") for row in prerequisites):
        return False
    store = CampaignStore(camp_dir.name, camp_dir.parent)
    ledger = load_ledger(store)
    if ledger is None:
        return False
    arm = ledger["arms"].get(experiment_id, {})
    if arm.get("status") != "complete" or not arm.get("completed_scoreboard_sha256"):
        return False
    board = camp_dir / "runs" / experiment_id / "scoreboard.json"
    checkpoint = Path(arm["checkpoint"])
    return (board.is_file() and checkpoint.is_file()
            and hashlib.sha256(board.read_bytes()).hexdigest() == arm["completed_scoreboard_sha256"]
            and hashlib.sha256(checkpoint.read_bytes()).hexdigest() == arm["checkpoint_sha256"])


def run_chunks(context, *, command_factory, stage_runner, scoreboard, deadline=None):
    """Execute only work fitting this invocation; never pause its accounting clock."""
    started = time.monotonic()
    deadline = (
        min(
            deadline if deadline is not None else float("inf"),
            started + INTERRUPT_AFTER_SECONDS,
        )
        - FINALIZATION_SECONDS
    )
    store = CampaignStore(context["campaign_id"], context["root"])
    store.root.mkdir(parents=True, exist_ok=True)
    with (store.root / ".promotion-chunks.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = load_ledger(store)
        if ledger is None:
            ledger = _initialize(context, command_factory)
            _save(store, ledger)
        if ledger["plan"] != context["plan"] or ledger["arm_order"] != list(
            context["arm_order"]
        ):
            raise ValueError("promotion continuation changed its locked plan or arms")
        return _advance(
            store,
            {**context, "ledger": ledger},
            deadline=deadline,
            stage_runner=stage_runner,
            scoreboard=scoreboard,
        )


def resume_chunks(context, *, stage_runner, scoreboard, deadline=None):
    store = CampaignStore(context["campaign_id"], context["root"])
    ledger = load_ledger(store)
    if ledger is None:
        raise ValueError("no promotion continuation")
    return run_chunks(
        {
            **context,
            "camp_dir": store.root,
            "plan": ledger["plan"],
            "arm_order": ledger["arm_order"],
        },
        command_factory=None,
        stage_runner=stage_runner,
        scoreboard=scoreboard,
        deadline=deadline,
    )
