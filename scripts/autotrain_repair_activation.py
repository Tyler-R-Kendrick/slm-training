"""Controller-owned source handoff: close the old lease before fresh imports."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from slm_training.autoresearch.heal.operation_recovery import wake_verified_operation
from slm_training.autoresearch.heal.repair_release import verified_activation_handoff


class VerifiedRestart(Exception):
    def __init__(self, handoff):
        super().__init__("verified successor requires fresh controller imports")
        self.handoff = handoff


def recover_release(runtime, common, *, sequence, log_event, run_operation):
    """Recover only controller acceptance, including a crash before activation.

    Worker output and its suggested command never authorize an executable. An
    activated but unfinished successor reuses its registered remaining grant.
    """
    if runtime.cancel_event.is_set():
        return
    accepted = [
        row for row in runtime.store.verify_event_chain()
        if row["event_type"] == "repair_release_accepted"
    ]
    if not accepted:
        return
    checked = verified_activation_handoff(
        runtime.store, accepted[-1]["detail"]["handoff"]
    )
    if Path(common["cwd"]).resolve() != Path(checked["successor_execution"]).resolve():
        raise VerifiedRestart(checked)
    request = wake_verified_operation(runtime, checked, cwd=Path(common["cwd"]))
    state = runtime.snapshot()[request["successor_activity_id"]]
    if state.status in {"succeeded", "cancelled"}:
        return
    run_operation(runtime, request, sequence=sequence, log_event=log_event)


def restart_supervisor(args, handoff):
    """Called only after leaving ActivityRuntime and restoring signal handlers.

    Rebuild argv from parsed controller options, never the worker's command.
    Keep the absolute finite pass ceiling and clear old Python import paths.
    """
    execution = Path(handoff["successor_execution"]).resolve(strict=True)
    argv = [sys.executable, "-m", "scripts.run_autotrain_supervisor"]
    for name, value in vars(args).items():
        if name in {"operation_request", "operation_output"} or value is None:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.extend((flag, str(value.resolve() if isinstance(value, Path) else value)))
    env = {**os.environ, "PYTHONPATH": str(execution / "src")}
    env.pop("PYTHONHOME", None)
    os.chdir(execution)
    os.execve(sys.executable, argv, env)
