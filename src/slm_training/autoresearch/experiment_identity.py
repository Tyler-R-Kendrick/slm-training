"""Versioned treatment, randomized-unit and retry identities (not verdicts).

Legacy step-excluding fingerprints cannot be upgraded without resolved inputs.
These contracts bind future work; historical artifacts keep their old meaning.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

SCHEMA = "experiment_identity/v1"
_INCIDENTAL = frozenset(
    {
        "seed",
        "mask_seed",
        "loss_mask_seed",
        "run_id",
        "run_root",
        "campaign_id",
        "experiment_id",
        "attempt_id",
        "activity_id",
        "replicate_id",
        "hypothesis_id",
        "decode_timeout_seconds",
        "checkpoint_every_steps",
        "resume_from",
        "initialize_from",  # Content identity and role are required in bindings.
        "train_dir",  # Snapshot/preprocessing identities are required in bindings.
        "telemetry",
        "telemetry_sample_interval_ms",
        "sync_checkpoints",
        "checkpoint_bucket",
        "checkpoint_bucket_dry_run",
    }
)
_BINDINGS = frozenset(
    {
        "architecture",
        "tokenizer_layout",
        "training_snapshot",
        "preprocessing",
        "starting_checkpoint",
        "starting_checkpoint_role",
        "endpoint",
        "resource_contract",
    }
)


def digest(value: Any) -> str:
    """Strict canonical JSON: no stringification of objects or NaN acceptance."""
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode()).hexdigest()


class InterventionContract(BaseModel):
    """An explicit intervention, not permission to change arbitrary knobs."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["mechanism", "data", "duration", "capacity", "randomness"]
    varied_fields: tuple[str, ...] = Field(min_length=1)
    resource_basis: Literal["updates", "examples", "tokens", "wall_seconds"]
    capacity_charged: bool = False
    randomness_distribution: str = "locked_seed_design"

    @model_validator(mode="after")
    def unique_fields(self) -> InterventionContract:
        if len(set(self.varied_fields)) != len(self.varied_fields):
            raise ValueError("duplicate intervention field")
        if any(not field or field.startswith("_") for field in self.varied_fields):
            raise ValueError("invalid intervention field")
        if self.kind == "capacity" and not self.capacity_charged:
            raise ValueError("capacity intervention requires explicit charging")
        return self


def treatment_identity(
    resolved_config: Mapping[str, Any],
    *,
    bindings: Mapping[str, Any],
    intervention: InterventionContract,
) -> str:
    """Hash the complete resolved recipe, not an unresolved candidate delta.

    Caller supplies canonical immutable identities, not path names as substitutes
    for data/checkpoint digests. Execution timeouts do not replace locked product
    deadlines in ``resource_contract``. Seed realizations belong to the replicate.
    """
    missing = _BINDINGS - bindings.keys()
    if missing or any(bindings[key] in (None, "", {}) for key in _BINDINGS):
        raise ValueError(f"incomplete scientific bindings: {sorted(missing)}")
    for key in ("steps", "batch_size"):
        value = resolved_config.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"resolved recipe requires positive integer {key}")
    config = {k: v for k, v in resolved_config.items() if k not in _INCIDENTAL}
    return digest(
        {
            "schema": SCHEMA,
            "config": config,
            "bindings": dict(bindings),
            "intervention": intervention.model_dump(mode="json"),
        }
    )


def replicate_identity(
    *,
    hypothesis_id: str,
    design_digest: str,
    unit_id: str,
    seeds: Mapping[str, int],
    ancestor_digest: str,
    independence: str,
) -> str:
    """A planned unit can be shared across paired treatments, never across retries.

    ``independence`` must say conditional-on-ancestor when learned ancestry is
    shared; identity alone is not a statistical independence certificate.
    """
    if not all((hypothesis_id, design_digest, unit_id, ancestor_digest, independence)):
        raise ValueError("replicate requires locked design, unit and ancestry claim")
    if not seeds or any(type(v) is not int or v < 0 for v in seeds.values()):
        raise ValueError("replicate requires nonnegative integer seed realizations")
    return digest({"schema": SCHEMA, "role": "replicate", **locals()})


def attempt_identity(*, treatment_id: str, replicate_id: str, ordinal: int) -> str:
    if not treatment_id or not replicate_id or type(ordinal) is not int or ordinal < 0:
        raise ValueError(
            "attempt requires treatment, replicate and nonnegative ordinal"
        )
    return digest({"schema": SCHEMA, "role": "attempt", **locals()})


def legacy_identity_reference(value: str) -> dict[str, str]:
    """Explicit weak migration: never mint a treatment from a legacy short hash."""
    if not value:
        raise ValueError("missing legacy identity")
    return {
        "schema": "legacy_knobs_fingerprint/v1",
        "fingerprint": value,
        "claim": "unresolved_recipe_not_scientific_identity",
    }


def validate_matrix_role(role: str, count: int) -> None:
    """Shared schema hook: default search keeps its floor; confirmation is paired."""
    if type(count) is not int or role not in {
        "search",
        "screen",
        "confirm",
        "promotion",
    }:
        raise ValueError("invalid matrix role/count")
    if (role == "search" and count < 5) or (role != "search" and count != 2):
        raise ValueError(
            f"{role} requires {'at least five' if role == 'search' else 'two'} arms"
        )


def validate_intervention_resources(intervention, totals) -> None:
    """Schema hook; optional legacy fields never invent a declared intervention."""
    if intervention is not None:
        InterventionContract.model_validate(intervention)
    if totals is None:
        return
    if (
        intervention is None
        or len(totals) != 2
        or any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in totals
        )
    ):
        raise ValueError(
            "planned resource totals require a declared intervention and two finite positive quantities"
        )


def serialize_declared_fields(model, handler, fields):
    """New optional identity fields do not change old sealed payloads on read."""
    payload = handler(model)
    for name in fields:
        if name not in model.model_fields_set:
            payload.pop(name, None)
    return payload


class ExperimentIdentityFields(BaseModel):
    """Identity declaration/migration fields, composed with the strict spec owner."""

    hypothesis_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    intervention: dict[str, Any] | None = None
    planned_resource_totals: tuple[Any, Any] | None = None

    @model_validator(mode="after")
    def validate_identity_resources(self):
        validate_intervention_resources(self.intervention, self.planned_resource_totals)
        return self

    @model_serializer(mode="wrap")
    def serialize_declared_identity(self, handler):
        return serialize_declared_fields(
            self, handler, ("hypothesis_id", "intervention", "planned_resource_totals")
        )


class MatrixRoleFields(BaseModel):
    """Role cardinality/migration only; membership remains in HypothesisMatrix."""

    matrix_role: Literal["search", "screen", "confirm", "promotion"] = "search"

    @model_validator(mode="after")
    def validate_role_cardinality(self):
        validate_matrix_role(self.matrix_role, len(self.hypotheses))
        return self

    @model_serializer(mode="wrap")
    def serialize_declared_role(self, handler):
        return serialize_declared_fields(self, handler, ("matrix_role",))
