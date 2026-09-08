"""Bounded continuation; campaign history owns the cursor, not old telemetry."""

from __future__ import annotations

import math
import shlex
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from scripts.autoresearch_command_cursor import CommandCursor, ContinuationGrant
from slm_training.autoresearch.engine import is_resumable_eval_command
from slm_training.autoresearch.schemas import (
    ExperimentOutcome,
    ExperimentSpec,
    HarnessSignalV1,
)
from slm_training.levers import (
    HARNESS_FINALIZATION_RESERVE_SECONDS,
    INTERRUPT_AFTER_SECONDS,
)


def _merged(previous, current):
    if (previous.experiment_id, previous.campaign_id) != (
        current.experiment_id,
        current.campaign_id,
    ) or previous.campaign_manifest_sha256 != current.campaign_manifest_sha256:
        raise ValueError("continuation outcome identity mismatch")
    return current.model_copy(
        update={
            "metrics": {**previous.metrics, **current.metrics},
            "data_metrics": {**previous.data_metrics, **current.data_metrics},
            "stage_telemetry": (*previous.stage_telemetry, *current.stage_telemetry),
            "artifact_uris": tuple(
                dict.fromkeys((*previous.artifact_uris, *current.artifact_uris))
            ),
            "telemetry_uris": tuple(
                dict.fromkeys((*previous.telemetry_uris, *current.telemetry_uris))
            ),
            "harness_signals": current.harness_signals or previous.harness_signals,
            "started_at": previous.started_at or current.started_at,
            "command": previous.command or current.command,
            "wall_time_budget_seconds": previous.wall_time_budget_seconds,
        }
    )


def _pending_stage(outcome):
    if outcome is None or outcome.status != "stopped" or not outcome.stage_telemetry:
        return None
    stage = outcome.stage_telemetry[-1]
    command = list(stage.get("command") or ())
    evaluation = is_resumable_eval_command(command) and stage.get("exit_code") in (
        None,
        10,
    )
    training = (
        "scripts.train_model" in command
        and stage.get("exit_code") == 0
        and stage.get("resume_kind") == "training"
        and stage.get("resume_command")
        and stage.get("resume_validation") == "required_by_trainer_before_updates"
    )
    if (
        stage.get("resume_pending") is True
        and not any(
            stage.get(key) for key in ("timed_out", "killed", "interrupted", "skipped")
        )
        and (evaluation or training)
    ):
        return stage
    return None


def is_continuation_pending(outcome: ExperimentOutcome) -> bool:
    """Normal bounded yield only; no-progress/reconciliation need typed repair."""
    if outcome.status != "stopped":
        return False
    error = outcome.error or ""
    if error.startswith(
        (
            "continuation_no_progress",
            "continuation_reconciliation_required",
            "continuation_total_budget",
            "continuation_total_attempts",
            "continuation_required_commands",
        )
    ):
        return False
    return error == "continuation_budget_pending" or _pending_stage(outcome) is not None


def _progress(stage):
    if stage.get("resume_kind") == "training":
        completed = stage.get("completed_optimizer_updates")
        total = stage.get("requested_optimizer_updates")
        if (
            type(completed) is int
            and type(total) is int
            and 0 <= completed < total
            and isinstance(stage.get("progress_identity"), str)
            and stage["progress_identity"]
        ):
            return f"training:{total}", {"remaining": total - completed}
        return None
    progress = stage.get("progress")
    if isinstance(progress, dict):
        completed, total = progress.get("completed"), progress.get("total")
        if (
            type(completed) is int
            and type(total) is int
            and 0 <= completed <= total
            and total > 0
            and isinstance(progress.get("identity"), str)
            and progress["identity"]
        ):
            return progress["identity"], {"remaining": total - completed}
    parsed = stage.get("parsed_output") or {}
    counts = (parsed.get("resume") or {}).get("pending_record_n")
    if (
        not isinstance(counts, dict)
        or not counts
        or any(type(n) is not int or n < 0 for n in counts.values())
    ):
        return None
    return "eval", counts


def _advanced(before, after):
    old, new = _progress(before), _progress(after)
    if before.get("resume_kind") == "training" and before.get(
        "progress_identity"
    ) == after.get("progress_identity"):
        return False
    return (
        old is not None
        and new is not None
        and old[0] == new[0]
        and old[1].keys() == new[1].keys()
        and all(new[1][key] <= old[1][key] for key in old[1])
        and sum(new[1].values()) < sum(old[1].values())
    )


def _canonical(command):
    """Only the engine-injected invocation deadline is incidental to argv."""
    result = list(command)
    for flag in ("--evaluation-wall-seconds", "--max-wall-minutes"):
        if flag in result:
            index = result.index(flag)
            del result[index : index + 2]
    return result


def _stage_complete(stage):
    if stage.get("measurement_complete") is False:
        return False
    return (
        stage.get("exit_code") == 0
        or stage.get("expected_gate_rejection") is True
        or stage.get("measurement_complete") is True
    ) and not any(
        stage.get(key)
        for key in ("resume_pending", "timed_out", "killed", "interrupted", "skipped")
    )


def _position(outcome, commands):
    position = 0
    for stage in outcome.stage_telemetry:
        if position >= len(commands) or _canonical(
            stage.get("command", ())
        ) != _canonical(commands[position]):
            raise ValueError(
                "continuation stage does not match the outstanding command"
            )
        if _stage_complete(stage):
            position += 1
            continue
        break
    return position


def _stop(outcome, code):
    signal = HarnessSignalV1(
        family="model_build",
        code=code.split(":", 1)[0],
        evidence_uri=f"campaign:{outcome.campaign_id}/experiment:{outcome.experiment_id}/command_cursor",
        reproduced_on_frozen_input=True,
        primary=not any(s.primary for s in outcome.harness_signals),
    )
    return outcome.model_copy(
        update={
            "status": "stopped",
            "error": code,
            "harness_signals": (*outcome.harness_signals, signal),
        }
    )


def _resume_commands(commands, position, stage):
    suffix = [list(command) for command in commands[position:]]
    override = stage.get("resume_command")
    if override is not None:
        if (
            not isinstance(override, list)
            or not override
            or not all(isinstance(x, str) for x in override)
        ):
            raise ValueError("invalid controller-validated resume command")
        suffix[0] = list(override)
    elif "--resume-run" not in suffix[0]:
        suffix[0].append("--resume-run")
    return suffix


def continue_pending_evaluation(
    experiment: ExperimentSpec,
    outcome: ExperimentOutcome,
    *,
    deadline: float,
    campaign_manifest_sha256: str,
    execute_commands: Callable,
    cwd: Path,
    commands: Sequence[Sequence[str]] | None = None,
) -> ExperimentOutcome:
    """Compatibility entrypoint; compiled-plan callers must pass all commands."""
    if not math.isfinite(deadline):
        raise ValueError("continuation deadline must be finite")
    deadline = min(
        deadline,
        time.monotonic()
        + INTERRUPT_AFTER_SECONDS
        - HARNESS_FINALIZATION_RESERVE_SECONDS,
    )
    stage = _pending_stage(outcome)
    if stage is None:
        return outcome
    plan = (
        [list(command) for command in commands]
        if commands is not None
        else [shlex.split(command) for command in outcome.command]
    )
    if not plan:
        plan = [list(stage["command"])]  # Legacy eval-only API, no invented successors.
    position = _position(outcome, plan)
    while stage is not None and deadline > time.monotonic():
        suffix = _resume_commands(plan, position, stage)
        current = execute_commands(
            experiment,
            suffix,
            cwd=cwd,
            timeout_seconds=deadline - time.monotonic(),
            campaign_manifest_sha256=campaign_manifest_sha256,
        )
        advanced = _position(current, suffix)
        position += advanced
        outcome = _merged(outcome, current)
        next_stage = _pending_stage(current)
        if (
            next_stage is not None
            and advanced == 0
            and not _advanced(stage, next_stage)
        ):
            return _stop(
                outcome, "continuation_no_progress:repair_measurement_required"
            )
        if current.status == "completed" and position != len(plan):
            return _stop(outcome, "continuation_required_commands_missing")
        stage = next_stage
    return outcome


def execute_with_continuation(
    experiment,
    commands: Sequence[Sequence[str]],
    *,
    wall_seconds: float,
    campaign_manifest_sha256: str,
    execute_commands: Callable,
    cwd: Path,
    store=None,
    grant: ContinuationGrant | None = None,
) -> ExperimentOutcome:
    """Execute one invocation; an explicit total grant spans journaled restarts.

    The controller supplies the release/environment/input identity in ``grant``
    and retains activity leases. Uncommitted starts require reconciliation.
    """
    invocation_started = time.monotonic()
    interrupt_limit = (
        INTERRUPT_AFTER_SECONDS
        if grant is None or grant.interrupt_seconds is None
        else grant.interrupt_seconds
    )
    reserve = (
        HARNESS_FINALIZATION_RESERVE_SECONDS
        if grant is None or grant.finalization_reserve_seconds is None
        else grant.finalization_reserve_seconds
    )
    if not math.isfinite(interrupt_limit) or not 0 < interrupt_limit <= INTERRUPT_AFTER_SECONDS:
        raise ValueError("continuation interrupt allowance exceeds canonical invocation cap")
    if not math.isfinite(reserve) or not 0 <= reserve < interrupt_limit:
        raise ValueError("continuation finalization reserve is outside interrupt allowance")
    if not math.isfinite(wall_seconds) or not 0 < wall_seconds <= interrupt_limit:
        raise ValueError("continuation wall allowance exceeds canonical invocation cap")
    if wall_seconds <= reserve:
        raise ValueError(
            "continuation invocation allowance cannot cover finalization reserve"
        )
    plan = [list(command) for command in commands]
    if not plan or any(not command for command in plan):
        raise ValueError("continuation requires a nonempty compiled command plan")
    total = wall_seconds if grant is None else grant.total_wall_seconds
    identity = None if grant is None else grant.execution_identity
    with CommandCursor(
        store, experiment, plan, campaign_manifest_sha256, identity, total, cwd=cwd,
        max_attempts=grant.max_attempts if grant else None,
    ) as cursor:
        deadline = invocation_started + wall_seconds
        cursor.track_invocation(invocation_started, time.monotonic)
        return _execute_cursor(cursor, execute_commands, cwd, deadline, wall_seconds, reserve)


def _cursor_allowance(cursor, deadline, reserve=HARNESS_FINALIZATION_RESERVE_SECONDS):
    if cursor.attempt >= cursor.inputs.get("max_attempts", math.inf):
        return 0, "continuation_total_attempts_exhausted"
    remaining = min(deadline - time.monotonic(), cursor.remaining) - reserve
    code = ("continuation_total_budget_insufficient"
            if cursor.remaining <= HARNESS_FINALIZATION_RESERVE_SECONDS
            else "continuation_budget_pending")
    return remaining, code


def _execute_cursor(cursor, execute_commands, cwd, deadline, wall_seconds, reserve=HARNESS_FINALIZATION_RESERVE_SECONDS):
    outcome, position = cursor.outcome, cursor.position
    if cursor.unresolved or (outcome is not None and outcome.status != "stopped"):
        return cursor.result()
    if outcome is not None and (outcome.error or "").startswith(
        "continuation_no_progress"
    ):
        return outcome
    plan = cursor.inputs["commands"]
    while position < len(plan):
        remaining, code = _cursor_allowance(cursor, deadline, reserve)
        if remaining <= 0:
            return cursor.result(code)
        stage = _pending_stage(outcome)
        suffix = (
            _resume_commands(plan, position, stage)
            if stage
            else [list(c) for c in plan[position:]]
        )
        if (
            outcome is not None
            and stage is None
            and not (
                outcome.stage_telemetry and _stage_complete(outcome.stage_telemetry[-1])
            )
        ):
            return cursor.result("continuation_reconciliation_required")
        started = time.monotonic()
        # Reserve unaccounted invocation time, including finalization and overhead.
        cursor.start(min(cursor.remaining, wall_seconds - cursor.observed_seconds))
        previous, offset = outcome, position

        def stage_callback(partial):
            observed = _merged(previous, partial) if previous is not None else partial
            cursor.checkpoint(observed, offset + _position(partial, suffix))

        options = {"stage_callback": stage_callback} if cursor.store is not None else {}
        current = execute_commands(
            cursor.experiment,
            suffix,
            cwd=cwd,
            timeout_seconds=remaining,
            campaign_manifest_sha256=cursor.inputs["manifest"],
            **options,
        )
        advance = _position(current, suffix)
        position += advance
        outcome = _merged(outcome, current) if outcome is not None else current
        next_stage = _pending_stage(current)
        if stage and next_stage and advance == 0 and not _advanced(stage, next_stage):
            outcome = _stop(
                outcome, "continuation_no_progress:repair_measurement_required"
            )
        if current.status == "completed" and position != len(plan):
            outcome = _stop(outcome, "continuation_required_commands_missing")
        cursor.commit(outcome, position, time.monotonic() - started)
        if next_stage is None or (outcome.error or "").startswith(
            "continuation_no_progress"
        ):
            return outcome
    return cursor.result()
