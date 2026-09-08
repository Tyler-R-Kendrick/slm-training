"""Shared operational contracts and deterministic campaign-event reducer.

These states describe execution, never scientific acceptance or promotion.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slm_training.levers import INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ResourceCapacity(Contract):
    cpu_slots: int = Field(default=1, gt=0, strict=True)
    memory_mb: int = Field(default=2048, gt=0, strict=True)


class ResourceGrant(ResourceCapacity):
    interrupt_seconds: float = Field(
        default=INTERRUPT_AFTER_SECONDS, gt=0, le=INTERRUPT_AFTER_SECONDS
    )
    kill_grace_seconds: float = Field(
        default=KILL_GRACE_SECONDS, ge=0, le=KILL_GRACE_SECONDS
    )
    total_seconds: float = Field(default=541, gt=0)
    finalization_reserve_seconds: float = Field(default=1, ge=0)
    max_attempts: int = Field(default=3, gt=0, strict=True)

    @property
    def attempt_seconds(self) -> float:
        return self.interrupt_seconds + self.kill_grace_seconds

    @model_validator(mode="after")
    def sufficient_budget(self):
        if (
            self.total_seconds
            < self.attempt_seconds + self.finalization_reserve_seconds
        ):
            raise ValueError("grant cannot fund one invocation and finalization")
        return self


class ActivitySpec(Contract):
    schema_version: Literal["activity_spec/v1"] = "activity_spec/v1"
    activity_id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    kind: Literal["train", "eval", "repair", "verify", "delivery", "control"]
    source_digest: str = Field(pattern=r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$")
    environment_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    output_namespace: str = Field(min_length=1)
    dependencies: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ("local_process",)
    grant: ResourceGrant = Field(default_factory=ResourceGrant)

    @model_validator(mode="after")
    def safe_namespace(self):
        from pathlib import PurePosixPath

        path = PurePosixPath(self.output_namespace)
        if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
            raise ValueError("output_namespace must be a confined relative path")
        if self.activity_id in self.dependencies or len(set(self.dependencies)) != len(
            self.dependencies
        ):
            raise ValueError("self or duplicate dependency")
        return self


class ActivityOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    YIELDED = "yielded"
    RETRY = "retry"
    CODE_FAILURE = "code_failure"
    DATA_FAILURE = "data_failure"
    ENVIRONMENT_FAILURE = "environment_failure"
    FORMAL_INFRASTRUCTURE = "formal_infrastructure"
    FORMAL_CONTRADICTION = "formal_contradiction"
    DELIVERY_FAILURE = "delivery_failure"
    WALL_BUDGET = "wall_budget"
    SUITE_VOLUME = "suite_volume"
    MIXED_BUDGET = "mixed_budget"
    UNKNOWN_FAILURE = "unknown_failure"
    CAPABILITY = "capability"
    DEPENDENCY = "dependency"
    CANCELLED = "cancelled"


REMEDIES = {
    ActivityOutcome.CODE_FAILURE: "repair_harness",
    ActivityOutcome.DATA_FAILURE: "rebuild_data",
    ActivityOutcome.ENVIRONMENT_FAILURE: "repair_environment",
    ActivityOutcome.FORMAL_INFRASTRUCTURE: "repair_formal",
    ActivityOutcome.FORMAL_CONTRADICTION: "review_hypothesis",
    ActivityOutcome.DELIVERY_FAILURE: "deliver_stack",
    ActivityOutcome.WALL_BUDGET: "resume_evaluation",
    ActivityOutcome.SUITE_VOLUME: "rebuild_data",
    ActivityOutcome.MIXED_BUDGET: "calibrate_budget",
    ActivityOutcome.UNKNOWN_FAILURE: "diagnose",
}


class WakeCondition(Contract):
    predicate: str = Field(min_length=1)
    source: str = Field(min_length=1)
    identity_digest: str = Field(pattern=r"^[a-f0-9]{64}$")


class ActivityLease(Contract):
    activity_id: str
    attempt_id: str
    epoch: str
    generation: int = Field(gt=0, strict=True)
    token: str = Field(min_length=1)
    owner_identity: str = Field(min_length=1)
    expires_at: float = Field(gt=0)


class ActivityState(Contract):
    spec: ActivitySpec
    status: Literal[
        "runnable",
        "running",
        "waiting_dependency",
        "waiting_retry",
        "waiting_repair",
        "waiting_capability",
        "succeeded",
        "cancelled",
    ] = "runnable"
    sequence: int = Field(default=0, ge=0, strict=True)
    attempts: int = Field(default=0, ge=0, strict=True)
    charged_seconds: float = Field(default=0, ge=0)
    lease: ActivityLease | None = None
    wake: WakeCondition | None = None
    retry_at: float = 0
    action: str = ""
    last_progress_digest: str = ""
    outputs: dict[str, str] = Field(default_factory=dict)
    worker_identity: str = ""


class ActivityEvent(Contract):
    schema_version: Literal["activity_event/v1"] = "activity_event/v1"
    operation: Literal[
        "register",
        "claim",
        "heartbeat",
        "finish",
        "wake",
        "recover",
        "park",
        "cancel",
        "reconcile_finish",
    ]
    activity_id: str
    sequence: int = Field(ge=0, strict=True)
    at: float = Field(ge=0)
    spec: ActivitySpec | None = None
    lease: ActivityLease | None = None
    outcome: ActivityOutcome | None = None
    wake: WakeCondition | None = None
    outputs: dict[str, str] = Field(default_factory=dict)
    spent_seconds: float = Field(default=0, ge=0)
    progress_digest: str = ""
    worker_identity: str = ""
    cancel_reason: str = ""


def contract_digest(value: BaseModel | dict) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _updated(state: ActivityState, fields: dict) -> ActivityState:
    return ActivityState.model_validate({**state.model_dump(), **fields})


def _finished(state: ActivityState, event: ActivityEvent) -> dict:
    outcome = event.outcome
    if outcome is None:
        raise ValueError("finish requires typed outcome")
    grant = state.spec.grant
    fields = {
        "lease": None,
        "wake": event.wake,
        "action": "",
        "charged_seconds": state.charged_seconds
        - grant.attempt_seconds
        + event.spent_seconds,
    }
    if outcome == ActivityOutcome.SUCCEEDED:
        if not event.outputs:
            raise ValueError("success requires validated nonempty output manifest")
        return {**fields, "status": "succeeded", "outputs": event.outputs}
    if outcome == ActivityOutcome.CANCELLED:
        return {**fields, "status": "cancelled"}
    if outcome in (ActivityOutcome.RETRY, ActivityOutcome.YIELDED):
        funded = (
            fields["charged_seconds"]
            + grant.attempt_seconds
            + grant.finalization_reserve_seconds
            <= grant.total_seconds
        )
        if state.attempts < grant.max_attempts and funded:
            return {
                **fields,
                "status": "waiting_retry",
                "action": "retry",
                "retry_at": event.at + min(60, 2 ** min(state.attempts, 6)),
            }
        return {
            **fields,
            "status": "waiting_repair",
            "action": "diagnose_exhausted",
            "wake": WakeCondition(
                predicate="successor_resource_grant",
                source="controller_policy",
                identity_digest=contract_digest(state.spec),
            ),
        }
    if event.wake is None:
        raise ValueError(
            "wait requires an identified predicate and durable wake source"
        )
    status = {
        ActivityOutcome.CAPABILITY: "waiting_capability",
        ActivityOutcome.DEPENDENCY: "waiting_dependency",
    }.get(outcome, "waiting_repair")
    return {**fields, "status": status, "action": REMEDIES.get(outcome, outcome.value)}


def _claimed(state: ActivityState, event: ActivityEvent) -> dict:
    if state.status not in ("runnable", "waiting_retry") or event.lease is None:
        raise ValueError("activity is not claimable")
    if event.at < state.retry_at or event.lease.generation != state.attempts + 1:
        raise ValueError("retry timer or lease generation mismatch")
    if (
        event.lease.activity_id != event.activity_id
        or event.lease.expires_at <= event.at
    ):
        raise ValueError("lease identity/deadline mismatch")
    grant = state.spec.grant
    if (
        state.attempts >= grant.max_attempts
        or state.charged_seconds
        + grant.attempt_seconds
        + grant.finalization_reserve_seconds
        > grant.total_seconds
    ):
        raise ValueError("activity grant exhausted")
    return {
        "status": "running",
        "lease": event.lease,
        "attempts": state.attempts + 1,
        "charged_seconds": state.charged_seconds + grant.attempt_seconds,
    }


def _queued(state: ActivityState, event: ActivityEvent) -> dict:
    if event.operation == "wake":
        if (
            not state.status.startswith("waiting_")
            or state.action == "diagnose_exhausted"
        ):
            raise ValueError("activity cannot wake without successor contract")
        if event.wake != state.wake:
            raise ValueError("wrong wake predicate identity")
        return {"status": "runnable", "wake": None}
    if state.status not in ("runnable", "waiting_retry") or event.wake is None:
        raise ValueError("only queued work can park with an explicit wake predicate")
    statuses = {
        ActivityOutcome.CAPABILITY: "waiting_capability",
        ActivityOutcome.DEPENDENCY: "waiting_dependency",
    }
    if event.outcome not in statuses:
        raise ValueError("invalid queued wait kind")
    return {
        "status": statuses[event.outcome],
        "wake": event.wake,
        "action": "auto_" + event.outcome.value,
    }


def _running(state: ActivityState, event: ActivityEvent) -> dict:
    if event.operation == "cancel":
        if not event.cancel_reason or event.lease != state.lease:
            raise ValueError("cancellation requires a reason and current lease")
        return {"status": "cancelled", "lease": None, "wake": None, "action": ""}
    if state.lease is None or event.lease != state.lease:
        raise ValueError("stale or missing lease fence")
    if event.operation == "recover":
        charged = ActivityEvent.model_validate(
            {
                **event.model_dump(),
                "outcome": ActivityOutcome.RETRY,
                "spent_seconds": state.spec.grant.attempt_seconds,
            }
        )
        return _finished(state, charged)
    if event.at >= state.lease.expires_at and event.operation != "reconcile_finish":
        raise ValueError("expired lease event")
    if event.operation == "heartbeat":
        return {
            "last_progress_digest": event.progress_digest or state.last_progress_digest,
            "worker_identity": event.worker_identity or state.worker_identity,
        }
    if event.operation in ("finish", "reconcile_finish"):
        if (
            event.operation == "reconcile_finish"
            and event.outcome != ActivityOutcome.SUCCEEDED
        ):
            raise ValueError("reconciliation requires previously verified output")
        return _finished(state, event)
    raise ValueError(f"invalid activity transition: {event.operation}")


def reduce_activity(state: ActivityState | None, event: ActivityEvent) -> ActivityState:
    """Pure transition dispatch; reject invalid history, including model_copy bypasses."""
    event = ActivityEvent.model_validate(event.model_dump())
    if state is None:
        if event.operation != "register" or event.sequence != 0 or event.spec is None:
            raise ValueError("activity history must begin with registration")
        if event.spec.activity_id != event.activity_id:
            raise ValueError("activity identity mismatch")
        return ActivityState(spec=event.spec)
    state = ActivityState.model_validate(state.model_dump())
    if (
        event.sequence != state.sequence + 1
        or event.activity_id != state.spec.activity_id
    ):
        raise ValueError("activity sequence/identity mismatch")
    if state.status in ("succeeded", "cancelled"):
        raise ValueError("terminal activity is immutable")
    if event.operation == "claim":
        fields = _claimed(state, event)
    elif event.operation in ("park", "wake"):
        fields = _queued(state, event)
    else:
        fields = _running(state, event)
    return _updated(state, {**fields, "sequence": event.sequence})


def pending_actions(
    states: dict[str, ActivityState], capabilities: set[str]
) -> list[dict]:
    """Presentation derived only from operational projection and current capability."""
    actions = []
    for state in states.values():
        missing = sorted(set(state.spec.capabilities) - capabilities)
        dependencies = [
            d for d in state.spec.dependencies if states[d].status != "succeeded"
        ]
        if state.status in ("succeeded", "cancelled", "running"):
            continue
        actions.append(
            {
                "activity_id": state.spec.activity_id,
                "status": "waiting_capability"
                if missing
                else "waiting_dependency"
                if dependencies
                else state.status,
                "action": state.action,
                "missing_capabilities": missing,
                "dependencies": dependencies,
                "wake": state.wake.model_dump() if state.wake else None,
                "retry_at": state.retry_at,
                "input_digest": contract_digest(state.spec),
            }
        )
    return actions
