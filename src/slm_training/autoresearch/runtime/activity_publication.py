"""One fence validator for controller and explicitly delegated control children.

This is a trusted-principal API, not a sandbox. Only controller-operation
children with the explicit capability may use it; repair workloads never get
the store mounted. It confers no claim, completion, receipt or policy authority.
"""

from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .activity_process import process_identity
from .activity_projection import ActivityProjection
from slm_training.harness_core.activity_contract import ActivityLease


class StaleLease(RuntimeError):
    """An expired, superseded or wrong-controller writer cannot publish."""


@dataclass(frozen=True)
class DelegatedPublisher:
    """Publication capability for a pinned controller child, never a repair worker."""

    store: object
    source_digest: str

    def publication(self, lease):
        return delegated_publication(
            self.store, lease, source_digest=self.source_digest
        )


def validate_activity_fence(states, lease, *, epoch, owner, now):
    lease = ActivityLease.model_validate(lease.model_dump())
    state = states.get(lease.activity_id)
    if (
        state is None
        or state.status != "running"
        or state.lease != lease
        or lease.epoch != epoch
        or lease.owner_identity != owner
        or lease.expires_at <= now
    ):
        raise StaleLease("stale, expired, or foreign activity fence")
    return state


def _parent_holds_lock(path: Path, pid: int) -> bool:
    """Linux kernel lock identity; a locked file alone does not identify its owner."""
    stat = path.stat()
    wanted = (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino)
    for line in Path("/proc/locks").read_text().splitlines():
        parts = line.split()
        if (
            len(parts) < 8
            or parts[1] != "FLOCK"
            or parts[3] != "WRITE"
            or parts[4] != str(pid)
        ):
            continue
        major, minor, inode = parts[5].split(":")
        if (int(major, 16), int(minor, 16), int(inode)) == wanted:
            return True
    return False


@contextmanager
def delegated_publication(store, lease: ActivityLease, *, source_digest: str):
    """Serialize a trusted child pointer CAS with claims, cancellation and restart.

    The parent must declare ``controller_publication`` in ActivitySpec.capabilities
    and ``kind=control``. ``run`` records this child's boot/PID/start identity.
    Source release equality supplements OS isolation; it does not authenticate
    arbitrary same-user Python code. Unsupported procfs fails closed.
    """
    fd = os.open(
        store.root / ".activities.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        events = store.verify_event_chain()
        starts = [
            row for row in events if row["event_type"] == "activity_controller_started"
        ]
        if not starts:
            raise StaleLease("missing controller epoch")
        parent = starts[-1]["detail"]
        state = validate_activity_fence(
            ActivityProjection(store).read(),
            lease,
            epoch=parent["epoch"],
            owner=parent["owner_identity"],
            now=time.time(),
        )
        pid = int(parent["owner_identity"].split(":")[1])
        if (
            state.spec.kind != "control"
            or "controller_publication" not in state.spec.capabilities
            or state.spec.source_digest != source_digest
            or os.getppid() != pid
            or state.worker_identity != process_identity()
        ):
            raise StaleLease("publication not delegated to this controller child")
        try:
            alive = process_identity(pid) == parent["owner_identity"]
            locked = _parent_holds_lock(store.root / ".activity-controller.lock", pid)
        except (OSError, ValueError) as exc:
            raise StaleLease("controller ownership cannot be verified") from exc
        if not alive or not locked:
            raise StaleLease("parent controller no longer owns its lock")
        yield state
    finally:
        os.close(fd)
