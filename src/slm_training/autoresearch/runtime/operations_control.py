"""Explicit user-owned supervisor start/stop. Importing never activates a service."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from slm_training.autoresearch.runtime.activity_process import (
    controller_status,
    process_identity,
    stop_previous_worker,
)
from slm_training.autoresearch.runtime.operations_doctor import storage_health
from slm_training.harness_core.execution_release import runtime_source_identity
from slm_training.autoresearch.runtime.operations_status import runtime_store
from slm_training.levers import KILL_GRACE_SECONDS


class StartConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: str = Field(pattern="^local_supervisor_start/v1$")
    execution: str
    loop_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    train_version: str = Field(min_length=1)
    steps: int = Field(ge=1)
    max_cycles: int = Field(ge=0)
    local_execution_authorized: bool
    recovery_config: str | None = None


def load_start_config(path: Path) -> StartConfig:
    if path.is_symlink() or path.stat().st_size > 65536:
        raise ValueError("unsafe_start_config")
    return StartConfig.model_validate_json(path.read_text())


def start_supervisor(root: Path, config: StartConfig) -> dict:
    if not config.local_execution_authorized:
        raise ValueError("local_execution_grant_missing")
    execution = Path(config.execution).resolve(strict=True)
    identity = runtime_source_identity(execution)
    if identity is None:
        raise ValueError("immutable_execution_release_required")
    health = storage_health(root)
    if not health["ready"]:
        return {"started": False, "waiting": "storage_maintenance", "health": health}
    store = runtime_store(root, config.loop_id)
    store.root.mkdir(parents=True, exist_ok=True)
    with (store.root / ".operations-start.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = controller_status(store, now=time.time())
        if status.get("lock_owned"):
            return {"started": False, "already_owned": True, "controller": status}
        return _launch(root, config, execution, identity, store)


def _launch(root, config, execution, identity, store) -> dict:
    argv = [
        sys.executable,
        "-m",
        "scripts.run_autotrain_supervisor",
        "--root",
        str(root.resolve()),
        "--loop-id",
        config.loop_id,
        "--train-version",
        config.train_version,
        "--steps",
        str(config.steps),
        "--max-cycles",
        str(config.max_cycles),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(execution / "src")
    # Reuse readable dependency bytecode; do not recompile the environment per start.
    # The immutable release and shared runtimes must not acquire new cache writes.
    env.pop("PYTHONPYCACHEPREFIX", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Private Git-free releases must not inherit an enclosing checkout's HEAD.
    env["GIT_CEILING_DIRECTORIES"] = str(execution.parent)
    if config.recovery_config is not None:
        argv.extend(
            ("--repair-config", str(Path(config.recovery_config).resolve(strict=True)))
        )
    child = subprocess.Popen(
        argv,
        cwd=execution,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        owned = process_identity(child.pid)
        store.append_event(
            "operations_start_requested",
            idempotency_key="start:" + owned,
            detail={
                "process_identity": owned,
                "source_digest": identity,
                "max_cycles": config.max_cycles,
                "at": time.time(),
            },
        )
        return _await_launch(child, owned, store, identity)
    except BaseException:
        # Publication failure must not leave an untracked background service.
        if child.poll() is None:
            stop_previous_worker(process_identity(child.pid), KILL_GRACE_SECONDS)
        child.wait(timeout=KILL_GRACE_SECONDS)
        raise


def _await_launch(child, owned, store, identity):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and child.poll() is None:
        status = controller_status(store, now=time.time())
        if status.get("alive") and status.get("process_identity") == owned:
            return {"started": True, "controller": status, "source_digest": identity}
        time.sleep(0.05)
    if child.poll() is None:
        stop_previous_worker(owned, KILL_GRACE_SECONDS)
        child.wait(timeout=KILL_GRACE_SECONDS)
    return {
        "started": False,
        "waiting": "supervisor_startup_failed",
        "exit_code": child.poll(),
    }


def stop_supervisor(root: Path, loop_id: str) -> dict:
    store = runtime_store(root, loop_id)
    status = controller_status(store, now=time.time())
    identity = status.get("process_identity")
    if not identity or not status.get("lock_owned"):
        return {"stopped": False, "reason": "no_owned_controller", "controller": status}
    pid = int(identity.split(":")[1])
    try:
        matches = (
            pid > 1 and process_identity(pid) == identity and os.getpgid(pid) == pid
        )
    except (ProcessLookupError, FileNotFoundError):
        matches = False
    if not matches:
        return {"stopped": False, "reason": "controller_identity_or_session_mismatch"}
    store.append_event(
        "operations_stop_requested",
        idempotency_key="stop:" + identity,
        detail={"process_identity": identity, "at": time.time()},
    )
    stop_previous_worker(identity, KILL_GRACE_SECONDS)
    after = controller_status(store, now=time.time())
    return {
        "stopped": not after.get("lock_owned"),
        "controller": after,
        "checkpoint_claim": "only_committed_runtime_artifacts_survive",
        "reconciliation": "canonical_supervisor_resume",
    }


def service_template(config: Path, root: Path) -> str:
    """Output a user service recipe only; never install or enable systemd."""
    # JSON argv is an unambiguous, portable operating recipe, not a shell unit.
    return json.dumps(
        {
            "installed": False,
            "activated": False,
            "command": [
                sys.executable,
                "-m",
                "scripts.autoresearch",
                "--root",
                str(root),
                "start",
                "--config",
                str(config),
            ],
            "stop_command": [
                sys.executable,
                "-m",
                "scripts.autoresearch",
                "--root",
                str(root),
                "stop",
                "--loop-id",
                "<configured-loop>",
            ],
            "lifecycle": "explicit_user_owned_background_supervisor",
        },
        indent=2,
    )
