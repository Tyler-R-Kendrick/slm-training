"""Evidence-bound supervisor action and receipt contracts.

Public imports remain available from schemas; serialization and validation are unchanged.
"""
from __future__ import annotations

from typing import Literal
from pydantic import Field, model_validator, model_serializer
from .schema_base import HarnessFamily, StrictModel, utc_now


class AutotrainActionV1(StrictModel):
    """One evidence-bound action for the agent supervisor between cycles."""

    schema_version: Literal["AutotrainActionV1"] = "AutotrainActionV1"
    kind: Literal[
        "stop_campaign",
        "repair_harness",
        "repair_formal",
        "rebuild_data",
        "document",
        "deliver_stack",
        "retry_measurement",
        "next_experiment",
        "monitor",
    ]
    owner: Literal[
        "autotrain",
        "improve-openui-harnesses",
        "improve-lean-optimums",
        "synthesis-feedback",
        "documenting-experiment-results",
        "sdlc",
    ]
    reason: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    harness_family: HarnessFamily | None = None
    frozen_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # Absent on legacy handoffs: prose must not invent a typed diagnosis.
    blocker_code: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]*$")
    unmet_predicate: str | None = Field(default=None, min_length=1)
    required_capability: str | None = Field(default=None, min_length=1)
    dependency_scope: Literal["delivery"] | None = None

    @model_serializer(mode="wrap")
    def serialize_legacy_scope(self, handler):
        value = handler(self)
        if self.dependency_scope is None:
            value.pop("dependency_scope", None)
        return value

    @model_validator(mode="after")
    def validate_harness_action(self) -> AutotrainActionV1:
        if self.dependency_scope is not None and self.kind != "document":
            raise ValueError("only document publication may use delivery dependency scope")
        if self.kind == "repair_harness" and self.harness_family is None:
            raise ValueError("repair_harness action requires harness_family")
        if self.kind != "repair_harness" and self.harness_family is not None:
            raise ValueError("harness_family is only valid for repair_harness")
        if self.kind not in {"repair_harness", "retry_measurement"} and (
            self.frozen_manifest_sha256 is not None
        ):
            raise ValueError(
                "frozen_manifest_sha256 is only valid for repair/retry actions"
            )
        return self


class AutotrainActionEvidenceV1(StrictModel):
    """Content identity for one durable action-receipt evidence item."""

    uri: str = Field(min_length=1)
    kind: Literal["git_commit", "repo_file", "campaign_artifact"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class AutotrainActionReceiptV1(StrictModel):
    """Append-only evidence that a supervisor executed one handoff action."""

    schema_version: Literal["AutotrainActionReceiptV1"] = "AutotrainActionReceiptV1"
    loop_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    action_index: int = Field(ge=0)
    action_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    action_kind: str = Field(min_length=1)
    # ``superseded`` is a driver-owned terminal state for an obsolete
    # execution prerequisite.  It is not an operator ack or a waiver: the
    # replacement policy evidence must prove the action is no longer needed.
    status: Literal["completed", "blocked", "superseded"]
    evidence_uris: tuple[str, ...] = Field(min_length=1)
    evidence: tuple[AutotrainActionEvidenceV1, ...] = ()
    recorded_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_evidence_identity(self) -> AutotrainActionReceiptV1:
        if (
            self.evidence
            and tuple(item.uri for item in self.evidence) != self.evidence_uris
        ):
            raise ValueError("receipt evidence identities must match evidence_uris")
        return self
