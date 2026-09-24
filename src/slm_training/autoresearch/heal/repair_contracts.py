"""Source-repair contracts. Worker proposals carry no acceptance authority."""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator

from slm_training.levers import INTERRUPT_AFTER_SECONDS
from slm_training.lineage.records import canonical_json

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, Field(min_length=1, max_length=512)]
InstructionText = Annotated[str, Field(min_length=1, max_length=262144)]


class RepairModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def digest(self) -> str:
        return hashlib.sha256(
            canonical_json(self.model_dump(mode="json")).encode()
        ).hexdigest()


class ProviderEndpoint(RepairModel):
    """Controller-approved host loopback proxy; contains no provider credentials."""

    port: Annotated[int, Field(ge=1, le=65535)] | None = None
    model: Name
    authentication: Literal["host_proxy", "codex_subscription"] = "host_proxy"
    base_path: str = "/v1"
    paths: tuple[str, ...] = ("/v1/responses",)
    methods: tuple[Literal["GET", "POST"], ...] = ("POST",)

    @model_validator(mode="after")
    def canonical_routes(self):
        import re

        if (self.authentication == "host_proxy") != (self.port is not None):
            raise ValueError("only host_proxy authentication takes a loopback port")
        if self.authentication == "codex_subscription" and (
            self.base_path != "/v1" or self.paths != ("/v1/responses",) or self.methods != ("POST",)
        ):
            raise ValueError("subscription grant permits only POST /v1/responses")
        paths = (self.base_path, *self.paths)
        if not self.paths or not self.methods or any(
            not re.fullmatch(r"(?:/[A-Za-z0-9_-]+)+", path) for path in paths
        ):
            raise ValueError("provider routes must be nonempty canonical absolute paths")
        if any(not path.startswith(self.base_path + "/") for path in self.paths):
            raise ValueError("provider routes must be below the granted base path")
        return self

    @model_serializer(mode="wrap")
    def preserve_proxy_contract(self, handler):
        value = handler(self)
        if self.authentication == "host_proxy":
            value.pop("authentication", None)
        return value


class RepairGrant(RepairModel):
    grant_id: Name
    provider: Name
    executable: Name
    executable_sha256: Digest
    expires_at: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    max_attempts: Annotated[int, Field(ge=1, le=10)]
    total_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    interrupt_seconds: Annotated[int, Field(ge=1, le=INTERRUPT_AFTER_SECONDS)]
    network: Literal["none", "approved_provider_only"] = "none"
    provider_endpoint: ProviderEndpoint | None = None
    repair_classes: tuple[
        Literal["code", "environment", "data", "formal_infra"], ...
    ] = ("code",)

    @model_validator(mode="after")
    def endpoint_requires_network_grant(self):
        if self.provider_endpoint is not None and self.network != "approved_provider_only":
            raise ValueError("provider endpoint requires approved_provider_only")
        return self

    def accounting_digest(self) -> str:
        """Stable grant identity for budget accounting across expiry refreshes."""
        payload = self.model_dump(mode="json")
        payload.pop("expires_at")
        return hashlib.sha256(canonical_json(payload).encode()).hexdigest()

    @model_serializer(mode="wrap")
    def omit_absent_endpoint(self, handler):
        value = handler(self)
        if self.provider_endpoint is None:
            value.pop("provider_endpoint", None)
        return value


class RepairBlocker(RepairModel):
    code: Name
    blocker_class: Literal[
        "code",
        "environment",
        "data",
        "formal_infra",
        "formal_contradiction",
        "delivery",
        "authority",
        "unknown",
    ]
    owner: Name
    source_digest: Digest
    environment_digest: Digest
    input_digest: Digest
    reproducer: tuple[Name, ...] = Field(min_length=1)
    predicate: Name
    needed_capability: Name
    evidence: tuple[Name, ...] = Field(min_length=1)

    def fingerprint(self) -> str:
        # Human logs and incidental campaign/attempt ids never mint new budgets.
        fields = self.model_dump(mode="json", exclude={"evidence"})
        return hashlib.sha256(canonical_json(fields).encode()).hexdigest()


class RepairRequest(RepairModel):
    blocked_activity_id: Name | None = None
    schema_version: Literal["source_repair_request/v1"] = "source_repair_request/v1"
    activity_id: Name
    attempt_id: Name
    campaign_id: Name
    fence: Name
    parent_event: Name
    blocker: RepairBlocker
    grant: RepairGrant | None
    allowed_paths: tuple[Name, ...] = Field(min_length=1)
    verification_manifest_digest: Digest
    project_instructions: InstructionText
    owner_contract: InstructionText
    existing_tests: tuple[Name, ...] = Field(min_length=1)
    failure_returncode: Annotated[int, Field(gt=0, lt=256)]
    failure_stdout_sha256: Digest
    failure_stderr_sha256: Digest
    semantics_preserving_paths: tuple[str, ...] = ()

    @field_validator("allowed_paths")
    @classmethod
    def scoped_paths(cls, paths: tuple[str, ...]) -> tuple[str, ...]:
        for value in paths:
            path = PurePosixPath(value)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in value
                or value != path.as_posix()
                or value in {".", ""}
            ):
                raise ValueError("repair paths must be canonical relative paths")
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate allowed path")
        return paths


class RepairProposal(RepairModel):
    schema_version: Literal["source_repair_proposal/v1"] = "source_repair_proposal/v1"
    request_digest: Digest
    tree_digest: Digest
    patch_digest: Digest
    root_cause: Name
    regression_test: Name
    reproduction_artifacts: tuple[Digest, ...] = Field(min_length=1)
    classification: Literal["implementation", "measurement_semantic", "trust_policy"]


class VerificationBinding(RepairModel):
    """Fresh controller authority for verifying an immutable prior proposal."""

    request_digest: Digest
    proposal_digest: Digest
    grant_digest: Digest
    fence: Name
    attempt_id: Name
    parent_event: Name


class RepairVerification(RepairModel):
    """Only accepted through an authenticated controller-owned verifier channel."""

    request_digest: Digest
    proposal_digest: Digest
    verifier_release: Digest
    manifest_digest: Digest
    source_digest: Digest
    environment_digest: Digest
    input_digest: Digest
    grant_id: Name
    fence: Name
    authority_binding_digest: Digest | None = None
    original_failure_reproduced: bool
    original_predicate_restored: bool
    required_checks_passed: bool
    protected_surfaces_unchanged: bool
    release_digest: Digest
    evidence_digest: Digest


class RepairDispatchResult(RepairModel):
    status: Literal[
        "waiting_capability",
        "waiting_verification",
        "waiting_diagnosis",
        "rejected",
        "verified",
        "cancelled",
    ]
    request_digest: Digest
    reason: Name
    proposal: RepairProposal | None = None
    verification: RepairVerification | None = None
    spent_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
