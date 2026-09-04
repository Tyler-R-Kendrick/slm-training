"""Tracking which stage of a cycle is currently running.

One responsibility: the active-stage marker and the process bound to it, so a
killed or interrupted cycle can be attributed to the stage it died in.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from scripts.autotrain_paths import loop_state_path
from scripts.autotrain_records import write_loop_state_unlocked
from slm_training.autoresearch.schemas import (
    AutotrainLoopStateV1,
    utc_now,
)
from slm_training.autoresearch.storage import (
    autotrain_loop_state_lock,
)
from slm_training.harness_core.bounded_process import (
    BoundedProcessResult,
    ProcessOutcome,
)


def stage_process_callbacks(
    *, root: Path | None, loop_id: str | None, stage: str | None
) -> tuple[Callable[[int], None] | None, Callable[[int], None] | None]:
    if root is None and loop_id is None:
        return None, None
    if root is None or loop_id is None or stage is None:
        raise ValueError("root, loop_id, and stage must be supplied together")
    set_active_stage(root, loop_id, stage)

    def update(pid: int) -> None:
        set_stage_process(root, loop_id, stage, pid)

    return update, update


def raise_for_bounded_result(result: BoundedProcessResult) -> None:
    if result.timed_out:
        raise subprocess.TimeoutExpired(
            result.command,
            result.duration_seconds,
            output=result.stdout,
            stderr=result.stderr,
        )
    if result.outcome is ProcessOutcome.LAUNCH_FAILED:
        raise OSError(result.launch_error or "subprocess launch failed")
    if result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.command,
            output=result.stdout,
            stderr=result.stderr,
        )


def set_active_stage(root: Path, loop_id: str, stage: str) -> None:
    path = loop_state_path(root, loop_id)
    with autotrain_loop_state_lock(root, loop_id):
        if not path.is_file():
            return
        try:
            state = AutotrainLoopStateV1.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return
        write_loop_state_unlocked(
            root,
            state.model_copy(
                update={
                    "active_stage": stage,
                    "child_pid": None,
                    "stage_started_at": None,
                    "heartbeat_at": utc_now(),
                }
            ),
        )


def set_stage_process(root: Path, loop_id: str, stage: str, child_pid: int) -> None:
    path = loop_state_path(root, loop_id)
    with autotrain_loop_state_lock(root, loop_id):
        if not path.is_file():
            return
        try:
            state = AutotrainLoopStateV1.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return
        now = utc_now()
        started_at = state.stage_started_at if state.active_stage == stage else None
        write_loop_state_unlocked(
            root,
            state.model_copy(
                update={
                    "active_stage": stage,
                    "child_pid": child_pid,
                    "stage_started_at": started_at or now,
                    "heartbeat_at": now,
                }
            ),
        )


def clear_active_stage(root: Path, loop_id: str) -> None:
    path = loop_state_path(root, loop_id)
    with autotrain_loop_state_lock(root, loop_id):
        if not path.is_file():
            return
        try:
            state = AutotrainLoopStateV1.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return
        write_loop_state_unlocked(
            root,
            state.model_copy(
                update={
                    "active_stage": None,
                    "child_pid": None,
                    "stage_started_at": None,
                    "heartbeat_at": utc_now(),
                }
            ),
        )
