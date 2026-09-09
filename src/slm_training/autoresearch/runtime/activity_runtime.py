"""Local controller over CampaignStore; trust/FS limits: runtime-activities.md."""

from __future__ import annotations

import fcntl
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from ..storage import CampaignStore
from .activity_process import (
    dispatch_repairs,
    prepare_outputs,
    reconcile_verified,
    run_activity,
    stop_previous_worker,
)
from .activity_process import process_identity as process_identity
from .activity_projection import ActivityProjection
from .activity_publication import StaleLease as StaleLease
from .activity_publication import validate_activity_fence
from slm_training.harness_core.activity_contract import (
    ActivityEvent,
    ActivityLease,
    ActivityOutcome,
    ActivitySpec,
    ActivityState,
    ResourceCapacity,
    WakeCondition,
    contract_digest,
    pending_actions,
    reduce_activity,
)


class ControllerBusy(RuntimeError):
    """Another supervisor owns this campaign's controller lock."""


class ActivityRuntime:
    """Controller-owned operational projection, replayed before each mutation."""

    def __init__(
        self,
        store: CampaignStore,
        *,
        capacity: ResourceCapacity | None = None,
        controller_clock: Callable[[], float] = time.time,
        repair_factory: Callable[[ActivityState], ActivitySpec] | None = None,
    ):
        self.store = store
        self._projection = ActivityProjection(store)
        self.capacity = capacity or ResourceCapacity()
        self.clock = controller_clock
        self.repair_factory = repair_factory
        self.epoch = ""
        self.owner = process_identity()
        self._fd: int | None = None
        self._mutex = threading.RLock()
        self._transaction_local = threading.local()
        self.cancel_event = threading.Event()

    def __enter__(self):
        self.store.root.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            self.store.root / ".activity-controller.lock", os.O_CREAT | os.O_RDWR, 0o600
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise ControllerBusy("another controller is active") from exc
        self._fd = fd
        self.epoch = uuid.uuid4().hex
        try:
            with self._transaction():
                self.store.append_event(
                    "activity_controller_started",
                    idempotency_key=self.epoch,
                    detail={
                        "schema_version": "activity_controller/v1",
                        "epoch": self.epoch,
                        "owner_identity": self.owner,
                        "at": self.clock(),
                    },
                )
            self.reconcile()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        self.cancel_event.set()
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    @contextmanager
    def _transaction(self):
        if self._fd is None or process_identity() != self.owner:
            raise StaleLease("controller context not owned by this process")
        if getattr(self._transaction_local, "depth", 0):
            # ponytail: reuse the process-local flock owner for nested mutations.
            yield
            return
        with self._mutex:
            fd = os.open(
                self.store.root / ".activities.lock", os.O_CREAT | os.O_RDWR, 0o600
            )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                self._transaction_local.depth = 1
                yield
            finally:
                self._transaction_local.depth = 0
                os.close(fd)

    def snapshot(self) -> dict[str, ActivityState]:
        """Legacy campaign events carry no invented activity/completion semantics."""
        return self._projection.read()

    def _append(
        self, state: ActivityState | None, event: ActivityEvent
    ) -> ActivityState:
        updated = reduce_activity(state, event)
        chain = self.store.verify_event_chain()
        self.store.append_event(
            "activity_transition",
            experiment_id=event.activity_id,
            status=updated.status,
            detail=event.model_dump(mode="json"),
            idempotency_key=f"activity:{event.activity_id}:{event.sequence}",
            expected_parent=chain[-1]["event_id"] if chain else "",
        )
        return updated

    def _event(self, state: ActivityState, operation: str, **kwargs) -> ActivityEvent:
        return ActivityEvent(
            operation=operation,
            activity_id=state.spec.activity_id,
            sequence=state.sequence + 1,
            at=self.clock(),
            **kwargs,
        )

    def register(self, spec: ActivitySpec) -> ActivityState:
        with self._transaction():
            states = self.snapshot()
            old = states.get(spec.activity_id)
            if old is not None:
                if old.spec != spec:
                    raise ValueError(
                        "activity identity binds different immutable inputs"
                    )
                return old
            if any(dep not in states for dep in spec.dependencies):
                raise ValueError(
                    "register dependencies first (cycles cannot be registered)"
                )
            return self._append(
                None,
                ActivityEvent(
                    operation="register",
                    spec=spec,
                    activity_id=spec.activity_id,
                    sequence=0,
                    at=self.clock(),
                ),
            )

    def reconcile(self) -> None:
        """Lost attempts remain fully charged; no unverified artifact is adopted."""
        with self._transaction():
            for state in self.snapshot().values():
                lease = state.lease
                if lease is not None and (
                    lease.epoch != self.epoch or lease.expires_at <= self.clock()
                ):
                    if reconcile_verified(self, state):
                        continue
                    stop_previous_worker(
                        state.worker_identity, state.spec.grant.kill_grace_seconds
                    )
                    self._append(state, self._event(state, "recover", lease=lease))
        self.dispatch_repairs()

    def _refresh_waits(self, states, capabilities):
        for state in states.values():
            missing = sorted(set(state.spec.capabilities) - capabilities)
            dependencies = [
                d for d in state.spec.dependencies if states[d].status != "succeeded"
            ]
            if state.status in ("runnable", "waiting_retry") and (
                missing or dependencies
            ):
                outcome = (
                    ActivityOutcome.CAPABILITY
                    if missing
                    else ActivityOutcome.DEPENDENCY
                )
                wake = WakeCondition(
                    predicate=",".join(missing or dependencies),
                    source="capability_probe" if missing else "activity_terminal_event",
                    identity_digest=contract_digest(state.spec),
                )
                self._append(
                    state, self._event(state, "park", outcome=outcome, wake=wake)
                )
            elif state.action.startswith("auto_") and not missing and not dependencies:
                self._append(state, self._event(state, "wake", wake=state.wake))

    def claim_next(
        self, *, capabilities: set[str], activity_id: str | None = None
    ) -> ActivityLease | None:
        self.reconcile()
        with self._transaction():
            states = self.snapshot()
            self._refresh_waits(states, capabilities)
            states = self.snapshot()
            running = [s for s in states.values() if s.status == "running"]
            cpu = sum(s.spec.grant.cpu_slots for s in running)
            memory = sum(s.spec.grant.memory_mb for s in running)
            # Least attempted first prevents repeated retries starving independent work.
            candidates = sorted(
                states.values(),
                key=lambda s: (
                    s.attempts,
                    s.spec.kind not in ("repair", "verify"),
                    s.sequence,
                ),
            )
            for state in candidates:
                if activity_id is not None and state.spec.activity_id != activity_id:
                    continue
                if self._ready(state, states, capabilities, cpu, memory):
                    lease = ActivityLease(
                        activity_id=state.spec.activity_id,
                        attempt_id=uuid.uuid4().hex,
                        epoch=self.epoch,
                        generation=state.attempts + 1,
                        token=uuid.uuid4().hex,
                        owner_identity=self.owner,
                        expires_at=self.clock()
                        + state.spec.grant.attempt_seconds
                        + state.spec.grant.finalization_reserve_seconds,
                    )
                    self._append(state, self._event(state, "claim", lease=lease))
                    return lease
        return None

    def cancel(self, activity_id: str, *, reason: str) -> ActivityState:
        """Revoke one owned activity; charged lost work remains visible."""
        with self._transaction():
            state = self.snapshot()[activity_id]
            if state.status in ("succeeded", "cancelled"):
                return state
            stop_previous_worker(
                state.worker_identity, state.spec.grant.kill_grace_seconds
            )
            return self._append(
                state,
                self._event(state, "cancel", lease=state.lease, cancel_reason=reason),
            )

    def cancel_all(self, *, reason: str) -> None:
        self.cancel_event.set()
        for activity_id in self.snapshot():
            self.cancel(activity_id, reason=reason)

    def controller_status(self, *, heartbeat_max_age_seconds: float = 60) -> dict:
        from .activity_process import controller_status

        return controller_status(
            self.store, now=self.clock(), max_age=heartbeat_max_age_seconds
        )

    def _ready(self, state, states, capabilities, cpu, memory) -> bool:
        grant = state.spec.grant
        return (
            not self.cancel_event.is_set()
            and state.status in ("runnable", "waiting_retry")
            and state.retry_at <= self.clock()
            and set(state.spec.capabilities) <= capabilities
            and all(states[d].status == "succeeded" for d in state.spec.dependencies)
            and cpu + grant.cpu_slots <= self.capacity.cpu_slots
            and memory + grant.memory_mb <= self.capacity.memory_mb
            and state.attempts < grant.max_attempts
            and state.charged_seconds
            + grant.attempt_seconds
            + grant.finalization_reserve_seconds
            <= grant.total_seconds
        )

    def _owned(self, lease: ActivityLease) -> ActivityState:
        return validate_activity_fence(
            self.snapshot(), lease, epoch=self.epoch, owner=self.owner, now=self.clock()
        )

    @contextmanager
    def publication(self, lease: ActivityLease) -> Iterator[ActivityState]:
        """Fence local CAS under lock; publisher owns intent/event reconciliation."""
        with self._transaction():
            yield self._owned(lease)

    def attempt_dir(self, lease: ActivityLease) -> Path:
        state = self._owned(lease)
        return self.store.root / state.spec.output_namespace / lease.attempt_id

    def heartbeat(
        self,
        lease: ActivityLease,
        *,
        progress_digest: str = "",
        worker_pid: int | None = None,
    ) -> None:
        try:
            worker = process_identity(worker_pid) if worker_pid is not None else ""
        except FileNotFoundError:
            worker = ""  # A short child can exit before the start observer runs.
        with self._transaction():
            state = self._owned(lease)
            self._append(
                state,
                self._event(
                    state,
                    "heartbeat",
                    lease=lease,
                    progress_digest=progress_digest,
                    worker_identity=worker,
                ),
            )

    def finish(
        self,
        lease: ActivityLease,
        *,
        outcome: ActivityOutcome,
        outputs: dict[str, str] | None = None,
        spent_seconds: float,
        wake: WakeCondition | None = None,
    ) -> ActivityState:
        with self._transaction():
            state = self._owned(lease)
            outputs = outputs or {}
            event = self._event(
                state,
                "finish",
                lease=lease,
                outcome=outcome,
                outputs=outputs,
                spent_seconds=spent_seconds,
                wake=wake,
            )
            reduce_activity(
                state, event
            )  # Reject invalid charges before any evidence append.
            if outcome == ActivityOutcome.SUCCEEDED:
                prepare_outputs(self, state, lease, outputs, spent_seconds)
            updated = self._append(state, event)
        self.dispatch_repairs()
        return updated

    def dispatch_repairs(self) -> list[str]:
        return dispatch_repairs(self, self.repair_factory)

    def wake(self, activity_id: str, *, evidence: WakeCondition) -> ActivityState:
        with self._transaction():
            state = self.snapshot()[activity_id]
            if state.wake != evidence:
                raise ValueError("wake predicate/source/identity does not match")
            return self._append(state, self._event(state, "wake", wake=evidence))

    def run(
        self,
        lease: ActivityLease,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
        progress_probe: Callable[[], str | None] | None = None,
        progress_timeout_seconds: float | None = None,
    ):
        return run_activity(
            self,
            lease,
            argv,
            cwd=cwd,
            env=env,
            progress_probe=progress_probe,
            progress_timeout_seconds=progress_timeout_seconds,
        )

    def actions(self, *, capabilities: set[str]) -> list[dict]:
        """Typed wait/action projection; absent capability parks only its activity."""
        return pending_actions(self.snapshot(), capabilities)
