"""Default-off allocation experiments, not another scheduler or promotion owner.

Li et al. arXiv:1603.06560 motivates the successive-halving comparator. All
methods share a locked resource ceiling. Unknown fidelity disables pruning;
cost actually consumed must still be reported rather than calling ceilings cost.
"""

from __future__ import annotations

import math
import random
from typing import Literal

from pydantic import Field, model_validator

from .evidence import Contract, Digest, contract_digest


class AllocationPlan(Contract):
    schema_version: Literal["search_allocation/v1"] = "search_allocation/v1"
    algorithm: Literal["rotation", "random", "successive_halving"]
    enabled: bool = False
    treatment_ids: tuple[Digest, ...] = Field(min_length=2)
    checkpoints: tuple[int, ...] = Field(min_length=2)
    total_updates: int = Field(gt=0, strict=True)
    random_seed: int = Field(strict=True)
    direction: Literal["maximize", "minimize"]
    endpoint_identity: Digest
    # Explicit authority-controlled adoption identity, NOT a computed p-value.
    fidelity_adoption_digest: Digest | None = None
    exploration_survivors: int = Field(default=1, ge=1, strict=True)

    @model_validator(mode="after")
    def locked_resources(self):
        if len(set(self.treatment_ids)) != len(self.treatment_ids):
            raise ValueError("duplicate treatment")
        if any(type(n) is not int or n <= 0 for n in self.checkpoints):
            raise ValueError("checkpoints must be positive optimizer updates")
        if tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("checkpoints must strictly increase")
        if self.total_updates < len(self.treatment_ids) * self.checkpoints[0]:
            raise ValueError("budget cannot cover the initial screen")
        return self


class FidelityObservation(Contract):
    plan_digest: Digest
    treatment_id: Digest
    updates: int = Field(gt=0, strict=True)
    value: float = Field(strict=True)
    cursor_digest: Digest
    attempt_id: str = Field(min_length=1)
    # Observations are committed COMPLETE endpoints only. Pending work is an activity.
    endpoint_identity: Digest


class Allocation(Contract):
    plan_digest: Digest
    treatment_id: Digest
    from_updates: int = Field(ge=0, strict=True)
    to_updates: int = Field(gt=0, strict=True)
    parent_cursor_digest: Digest | None = None

    def activity(
        self,
        *,
        factory,
        source_digest: str,
        environment_digest: str,
        grant,
        dependencies: tuple[str, ...] = (),
    ):
        digest = contract_digest(self)
        return factory(
            activity_id=f"allocation-{digest}",
            family=self.treatment_id,
            kind="train",
            source_digest=source_digest,
            environment_digest=environment_digest,
            input_digest=digest,
            output_namespace=f"search/{self.plan_digest}/{digest}",
            grant=grant,
            dependencies=dependencies,
        )


def checked_observations(plan: AllocationPlan, observations) -> dict:
    """Replayed attempts do not create more resource or statistical units."""
    rows = {}
    for raw in observations:
        row = FidelityObservation.model_validate(
            raw.model_dump() if isinstance(raw, FidelityObservation) else raw
        )
        if (
            row.plan_digest != contract_digest(plan)
            or row.endpoint_identity != plan.endpoint_identity
            or row.treatment_id not in plan.treatment_ids
            or row.updates not in plan.checkpoints
        ):
            raise ValueError("incompatible calibration observation")
        key = row.treatment_id, row.updates
        if key in rows and rows[key].model_dump(
            exclude={"attempt_id"}
        ) != row.model_dump(exclude={"attempt_id"}):
            raise ValueError("conflicting retry observation")
        rows[key] = row
    for treatment, update in rows:
        earlier = plan.checkpoints[: plan.checkpoints.index(update)]
        if any((treatment, checkpoint) not in rows for checkpoint in earlier):
            raise ValueError("observation omitted a locked continuation checkpoint")
    consumed = sum(
        max((n for key, n in rows if key == treatment), default=0)
        for treatment in plan.treatment_ids
    )
    if consumed > plan.total_updates:
        raise ValueError("observations exceed the locked update budget")
    return rows


def _survivors(plan, active, rows, checkpoint, rung):
    if plan.algorithm != "successive_halving" or plan.fidelity_adoption_digest is None:
        return active
    ordered = sorted(
        active,
        key=lambda key: (
            rows[key, checkpoint].value * (1 if plan.direction == "minimize" else -1),
            key,
        ),
    )
    keep = max(1, math.ceil(len(ordered) / 2))
    remainder = ordered[keep:]
    random.Random(plan.random_seed + rung).shuffle(remainder)
    return ordered[:keep] + remainder[: plan.exploration_survivors]


def next_allocations(plan: AllocationPlan, observations=()) -> list[Allocation]:
    """Pure proposal function. Runtime owns leases, queues, retries and wall charges."""
    plan = AllocationPlan.model_validate(plan.model_dump())
    if not plan.enabled:
        return []
    rows = checked_observations(plan, observations)
    active = list(plan.treatment_ids)
    if plan.algorithm == "random":
        random.Random(plan.random_seed).shuffle(active)
    for rung, checkpoint in enumerate(plan.checkpoints):
        missing = [key for key in active if (key, checkpoint) not in rows]
        if not missing:
            active = _survivors(plan, active, rows, checkpoint, rung)
            continue
        return _pending_allocations(plan, rows, missing, rung, checkpoint)
    return []


def _pending_allocations(plan, rows, missing, rung, checkpoint):
    consumed = sum(
        max((n for key, n in rows if key == treatment), default=0)
        for treatment in plan.treatment_ids
    )
    pending = []
    previous = plan.checkpoints[rung - 1] if rung else 0
    for key in missing:
        if previous and (key, previous) not in rows:
            raise ValueError("continuation missing its committed cursor")
        delta = checkpoint - previous
        if consumed + delta > plan.total_updates:
            break
        pending.append(
            Allocation(
                plan_digest=contract_digest(plan),
                treatment_id=key,
                from_updates=previous,
                to_updates=checkpoint,
                parent_cursor_digest=rows[key, previous].cursor_digest
                if previous
                else None,
            )
        )
        consumed += delta
    return pending


def _rank_sign(left: float, right: float) -> int:
    """Compare finite observations without subtraction overflow or underflow."""
    return (left > right) - (left < right)


def fidelity_report(plan: AllocationPlan, observations) -> dict:
    rows = checked_observations(plan, observations)
    short, long = plan.checkpoints[0], plan.checkpoints[-1]
    keys = [
        key
        for key in plan.treatment_ids
        if (key, short) in rows and (key, long) in rows
    ]
    signs = [
        _rank_sign(rows[a, short].value, rows[b, short].value)
        * _rank_sign(rows[a, long].value, rows[b, long].value)
        for i, a in enumerate(keys)
        for b in keys[i + 1 :]
    ]
    comparable = [x for x in signs if x != 0]
    agreement = sum(x > 0 for x in comparable) / len(comparable) if comparable else None
    direction = 1 if plan.direction == "minimize" else -1
    best_short = (
        min(keys, key=lambda k: (direction * rows[k, short].value, k)) if keys else None
    )
    best_long = (
        min(keys, key=lambda k: (direction * rows[k, long].value, k)) if keys else None
    )
    regret = (
        direction * (rows[best_short, long].value - rows[best_long, long].value)
        if keys
        else None
    )
    if regret is not None and not math.isfinite(regret):
        raise ValueError("nonfinite selection regret: finite observations overflow")
    return {
        "schema_version": "fidelity_report/v1",
        "plan_digest": contract_digest(plan),
        "complete": len(keys) == len(plan.treatment_ids),
        "paired_configurations": len(keys),
        "pairwise_rank_agreement": agreement,
        "tied_config_pairs": len(signs) - len(comparable),
        "selection_regret": regret,
        "short_selected": best_short,
        "long_selected": best_long,
        "consumed_updates": sum(
            max((n for key, n in rows if key == treatment), default=0)
            for treatment in plan.treatment_ids
        ),
        "uncertainty": "descriptive_only_no_independent_replicate_interval",
        "pruning_authorized": plan.fidelity_adoption_digest is not None,
        "claim_class": "diagnostic_only",
        "promotion_eligible": False,
    }


def lock_allocation_plan(store, plan: AllocationPlan):
    """Commit immutable allocation before any workload, in CampaignStore's chain."""
    artifact = store.write_artifact("search_allocations", plan)
    return store.append_event(
        "search_allocation_locked",
        artifact_sha256=artifact.stem,
        idempotency_key=f"allocation-plan:{contract_digest(plan)}",
        detail={
            "default_off": True,
            "algorithm": plan.algorithm,
            "total_updates": plan.total_updates,
        },
    )


def record_fidelity_observation(store, observation: FidelityObservation):
    observation = FidelityObservation.model_validate(observation.model_dump())
    artifact = store.write_artifact("fidelity_observations", observation)
    return store.append_event(
        "fidelity_observation_recorded",
        artifact_sha256=artifact.stem,
        idempotency_key=f"fidelity:{observation.plan_digest}:{observation.treatment_id}:{observation.updates}:{observation.attempt_id}",
        detail={"claim_class": "diagnostic_only"},
    )
