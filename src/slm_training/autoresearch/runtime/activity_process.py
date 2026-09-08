"""Linux process identity and bounded execution side effects for activities."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

from slm_training.harness_core.bounded_process import run_bounded_process
from slm_training.harness_core.process_tree import OwnedProcessTree

from slm_training.harness_core.activity_contract import ActivityOutcome, contract_digest


def verify_outputs(store_root: Path, state, lease, outputs: dict[str, str]) -> None:
    root = store_root / state.spec.output_namespace / lease.attempt_id
    if not root.resolve().is_relative_to(store_root.resolve()) or not outputs:
        raise ValueError("success requires confined nonempty output manifest")
    for name, digest in outputs.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("missing or escaping output artifact")
        if path.is_symlink() or path.stat().st_nlink != 1:
            raise ValueError("output content/link validation failed")
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != digest or path.stat().st_size == 0:
            raise ValueError("output content/link validation failed")
        if path.suffix == ".json":
            json.loads(path.read_text())


def prepare_outputs(runtime, state, lease, outputs, spent_seconds):
    """Durable verification event precedes terminal state; replay never reruns work."""
    verify_outputs(runtime.store.root, state, lease, outputs)
    detail = {
        "schema_version": "activity_outputs/v1",
        "lease": lease.model_dump(mode="json"),
        "input_digest": contract_digest(state.spec),
        "outputs": outputs,
        "spent_seconds": spent_seconds,
    }
    runtime.store.append_event(
        "activity_outputs_verified",
        experiment_id=lease.activity_id,
        idempotency_key=f"activity-outputs:{lease.attempt_id}",
        detail=detail,
    )


def reconcile_verified(runtime, state) -> bool:
    matches = [
        r
        for r in runtime.store.verify_event_chain()
        if r["event_type"] == "activity_outputs_verified"
        and r["experiment_id"] == state.spec.activity_id
        and r["detail"]["lease"] == state.lease.model_dump(mode="json")
    ]
    if not matches:
        return False
    detail = matches[-1]["detail"]
    if detail["schema_version"] != "activity_outputs/v1" or detail[
        "input_digest"
    ] != contract_digest(state.spec):
        raise ValueError("incompatible verified completion")
    verify_outputs(runtime.store.root, state, state.lease, detail["outputs"])
    runtime._append(
        state,
        runtime._event(
            state,
            "reconcile_finish",
            lease=state.lease,
            outcome=ActivityOutcome.SUCCEEDED,
            outputs=detail["outputs"],
            spent_seconds=detail["spent_seconds"],
        ),
    )
    return True


def dispatch_repairs(runtime, factory):
    """Materialize approved repair jobs; the factory is controller code, not an agent."""
    if factory is None:
        return []
    registered = []
    for state in runtime.snapshot().values():
        if state.status != "waiting_repair":
            continue
        job = factory(state)
        if (
            job.kind not in ("repair", "verify")
            or job.activity_id == state.spec.activity_id
        ):
            raise ValueError(
                "repair factory must return a distinct approved repair activity"
            )
        runtime.register(job)
        registered.append(job.activity_id)
    return registered


def controller_status(store, *, now: float, max_age: float = 60) -> dict:
    events = store.verify_event_chain()
    starts = [r for r in events if r["event_type"] == "activity_controller_started"]
    if not starts:
        return {"alive": False, "reason": "no_controller_identity"}
    start = starts[-1]
    identity = start["detail"]["owner_identity"]
    pid = int(identity.split(":")[1])
    try:
        current = process_identity(pid) == identity
    except FileNotFoundError:
        current = False
    fd = os.open(store.root / ".activity-controller.lock", os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = False
        except BlockingIOError:
            locked = True
    finally:
        os.close(fd)
    relevant = [
        r
        for r in events[events.index(start) :]
        if r["event_type"] in ("activity_controller_started", "activity_transition")
    ]
    age = max(0, now - max(r["detail"]["at"] for r in relevant))
    return {
        "alive": current and locked and age <= max_age,
        "process_identity": identity,
        "lock_owned": locked,
        "heartbeat_age_seconds": age,
        "epoch": start["detail"]["epoch"],
        "scientific_progress": False,
    }


def process_identity(pid: int | None = None) -> str:
    """Bind a PID to this boot and kernel start ticks (Linux/WSL only)."""
    pid = os.getpid() if pid is None else pid
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return f"{boot}:{pid}:{fields[19]}"


def stop_previous_worker(identity: str, grace: float) -> None:
    """Stop only a still-matching owned session after its controller loses lease."""
    if not identity:
        return
    pid = int(identity.split(":")[1])
    tree = None
    try:
        if process_identity(pid) != identity or os.getpgid(pid) != pid:
            return
        tree = OwnedProcessTree(pid)
        os.killpg(pid, signal.SIGINT)
        tree.signal(signal.SIGINT, excluding_group=pid)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z":
                break
            time.sleep(0.01)
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, FileNotFoundError):
        return
    finally:
        if tree is not None:
            tree.signal(signal.SIGKILL)


def run_activity(
    runtime,
    lease,
    argv,
    *,
    cwd,
    env=None,
    progress_probe=None,
    progress_timeout_seconds=None,
):
    """Execute only; the controller separately verifies output/predicate completion."""
    state = runtime._owned(lease)
    runtime.attempt_dir(lease).mkdir(parents=True, exist_ok=True)
    remaining = (
        lease.expires_at
        - runtime.clock()
        - state.spec.grant.kill_grace_seconds
        - state.spec.grant.finalization_reserve_seconds
    )
    last_heartbeat = [time.monotonic()]

    def on_start(pid):
        runtime.heartbeat(lease, worker_pid=pid)

    if "controller_publication" in state.spec.capabilities:
        gate = runtime.attempt_dir(lease) / ".controller-start.json"
        if gate.exists():
            raise ValueError("controller attempt already launched")

        def on_start(pid):
            runtime.heartbeat(lease, worker_pid=pid)
            runtime.store._replace_durable(gate, lease.token)

        # The PID survives exec. The bounded parent still owns the full clock;
        # an observer that stalls cannot let this barrier run indefinitely.
        entry = (
            "import os,sys,time\nfrom pathlib import Path\n"
            "gate,token=Path(sys.argv[1]),sys.argv[2]\n"
            "deadline=time.monotonic()+float(sys.argv[3])\n"
            "while not gate.is_file():\n"
            " if time.monotonic()>=deadline: raise TimeoutError('controller start not registered')\n"
            " time.sleep(0.005)\n"
            "if gate.read_text()!=token: raise ValueError('controller start token mismatch')\n"
            "os.execvpe(sys.argv[4],sys.argv[4:],os.environ)\n"
        )
        argv = [
            sys.executable,
            "-c",
            entry,
            str(gate),
            lease.token,
            str(max(0, remaining)),
            *argv,
        ]

    def heartbeat(_pid):
        if time.monotonic() - last_heartbeat[0] >= 15:
            runtime.heartbeat(lease)
            last_heartbeat[0] = time.monotonic()

    return run_bounded_process(
        argv,
        cwd=cwd,
        env=env,
        interrupt_after_seconds=min(state.spec.grant.interrupt_seconds, remaining),
        kill_grace_seconds=state.spec.grant.kill_grace_seconds,
        on_start=on_start,
        on_heartbeat=heartbeat,
        cancel_event=runtime.cancel_event,
        progress_probe=progress_probe,
        progress_timeout_seconds=progress_timeout_seconds,
    )
