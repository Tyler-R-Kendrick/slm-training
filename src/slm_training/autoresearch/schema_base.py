"""Shared strict model base and harness-family identity; no verdict authority."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_serializer
from slm_training.harness_core.activity_contract import ResourceGrant
from slm_training.levers import MAX_RUN_MINUTES
from slm_training.harness_core.lineage.store import utc_now as utc_now


HarnessFamily = Literal[
    "autoresearch",
    "annotations",
    "distill",
    "experiments",
    "model_build",
    "preference",
    "quality",
    "rl",
    "test_data",
    "train_data",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CampaignBudget(StrictModel):
    max_experiments: int = Field(default=12, ge=1, le=1000)
    max_gpu_hours: float = Field(default=0.0, ge=0)
    # Historical walls remain readable; each invocation is still capped.
    max_wall_minutes: float = Field(default=float(MAX_RUN_MINUTES), gt=0, le=60.0)
    continuation_grant: ResourceGrant | None = None

    @model_serializer(mode="wrap")
    def legacy_budget_payload(self, handler):
        value = handler(self)
        if self.continuation_grant is None:
            value.pop("continuation_grant", None)
        return value

    @property
    def logical_seconds(self) -> float:
        return (self.continuation_grant.total_seconds if self.continuation_grant
                else min(self.max_wall_minutes, MAX_RUN_MINUTES) * 60)
