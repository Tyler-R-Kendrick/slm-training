"""Versioned, controller-supplied data action inputs; legacy counts are not proof."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

READINESS_CONTEXT_FIELDS = frozenset({"trainer_config", "scored_suites", "training_ancestors",
    "learned_tables", "initialization", "lineage_root", "starting_run_id", "preprocessing_identity"})
READINESS_OPTIONAL_CONTEXT_FIELDS = frozenset({"target_kind", "additional_trainer_configs", "starting_checkpoint_bundle"})


def evidence_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scoped_path(root: Path, relative: str) -> Path:
    """No absolute paths, aliases, symlinks, or traversal in action inputs."""
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError("unsafe data action path")
    result = root.resolve() / value
    if any(part.is_symlink() for part in (result, *result.parents)):
        raise ValueError("symlink in data action path")
    result.resolve().relative_to(root.resolve())
    return result


class SnapshotInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    directory: str
    dataset_id: str
    kind: Literal["train", "eval", "trajectory"]
    records: str = "records.jsonl"
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exposure: Literal["train_only", "public_regression", "access_controlled", "unknown"]

    @field_validator("directory", "records")
    @classmethod
    def relative_path(cls, value: str) -> str:
        if not value or Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError("snapshot paths must be relative and confined")
        return value


class LearnedTableInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_sources: list[SnapshotInput] = Field(min_length=1)


class DataReadinessRequest(BaseModel):
    """Frozen original predicate; no inference of purpose from status prose.

    The controller derives scored suites and ancestry from the locked campaign
    and lineage. Worker-supplied requests are not authority to change either.
    """
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal["data_readiness_request/v1"] = "data_readiness_request/v1"
    action_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    purpose: Literal["validity", "volume", "screening_volume"]
    original: SnapshotInput
    successor_id: str
    target_kind: str = "document"
    minimum_unique_cases: int = Field(gt=0)
    minimum_unique_families: int = Field(gt=0)
    scored_suites: list[SnapshotInput] = Field(min_length=1)
    training_ancestors: list[SnapshotInput]
    learned_tables: list[LearnedTableInput]
    initialization: Literal["scratch", "parent"]
    lineage_root: str | None = None
    starting_run_id: str | None = None
    trainer_config: dict[str, Any]
    additional_trainer_configs: list[dict[str, Any]] = Field(default_factory=list)
    starting_checkpoint_bundle: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    preprocessing_identity: str = Field(min_length=1)
    sampling_policy_digest: str | None = None
    generation_knobs: dict[str, Any] | None = None

    @field_validator("additional_trainer_configs")
    @classmethod
    def explicit_additional_configs(cls, value):
        if any(not config for config in value):
            raise ValueError("additional trainer configurations must be explicit nonempty mappings")
        return value

    @model_validator(mode="after")
    def check_action(self):
        from slm_training.data.store import DataStore
        from slm_training.dsl.schema import OUTPUT_KINDS

        DataStore.validate_id(self.successor_id)
        if self.target_kind not in OUTPUT_KINDS:
            raise ValueError("unknown target kind")
        if self.successor_id == self.original.dataset_id:
            raise ValueError("repair requires an immutable successor identity")
        if any(item.kind != "eval" for item in self.scored_suites):
            raise ValueError("scored suites must be evaluation inputs")
        if self.original.kind == "eval" and self.purpose != "screening_volume":
            raise ValueError("routine repair cannot replace confirmation targets")
        if self.purpose == "screening_volume" and not self.sampling_policy_digest:
            raise ValueError("screening growth needs a locked sampling policy")
        if self.initialization == "parent" and not (
            self.lineage_root and self.starting_run_id and self.training_ancestors
        ):
            raise ValueError("parent initialization needs verifiable training ancestry")
        if self.original.exposure == "access_controlled":
            raise ValueError("routine repair cannot replace sealed data")
        return self

    @property
    def sha256(self) -> str:
        payload = self.model_dump(mode="json")
        # Optional obligations extend v1 without changing historical one-config
        # request identities merely because a newer reader supplied defaults.
        for key in ("additional_trainer_configs", "starting_checkpoint_bundle"):
            if not payload[key]:
                payload.pop(key)
        return evidence_digest(payload)



def load_locked_readiness_request(store, *, wanted=None, action_id=None):
    """Read the unique controller-locked request; no executor or latest-file dependency."""
    matches = {}
    for event in store.verify_event_chain():
        if event["event_type"] != "data_readiness_request_locked":
            continue
        detail = event["detail"]
        if wanted and detail["request_sha256"] != wanted:
            continue
        request = _request_from_event(store, event)
        if action_id not in (None, request.action_id):
            continue
        matches[request.sha256] = request
    if len(matches) != 1:
        raise ValueError("missing or ambiguous committed data readiness request")
    return next(iter(matches.values()))


def _request_from_event(store, event):
    from slm_training.lineage.records import content_sha

    detail = event["detail"]
    if event.get("experiment_id") is not None:
        lock = store.load_experiment_campaign(event["experiment_id"])
        if lock.manifest_sha256 != detail["manifest_sha256"]:
            raise ValueError("data request campaign manifest changed")
    sha = event["artifact_sha256"]
    path = store.root / "artifacts/data_readiness_requests" / f"{sha}.json"
    payload = json.loads(path.read_text())
    request = DataReadinessRequest.model_validate(payload)
    if request.campaign_id != store.campaign_id:
        raise ValueError("wrong campaign data readiness request")
    if content_sha(payload) != sha or request.sha256 != detail["request_sha256"]:
        raise ValueError("data readiness request integrity mismatch")
    if event.get("experiment_id") is None:
        input_sha = detail.get("activity_input_sha256", "")
        if not isinstance(input_sha, str) or len(input_sha) != 64 or any(c not in "0123456789abcdef" for c in input_sha):
            raise ValueError("missing bootstrap readiness input identity")
        inputs = json.loads((store.root / "artifacts/data_readiness_inputs" / f"{input_sha}.json").read_text())
        if content_sha(inputs) != input_sha or evidence_digest(inputs) != request.action_id:
            raise ValueError("bootstrap readiness input identity mismatch")
        validate_bootstrap_inputs(request, inputs)
    return request


def validate_bootstrap_inputs(request: DataReadinessRequest, inputs: dict) -> None:
    """Bootstrap authority is readiness-only and binds the actual controller input."""
    if set(inputs) != {"loop_id", "train_version", "eval_version", "minimum", "context"}:
        raise ValueError("invalid bootstrap readiness inputs")
    context = inputs["context"]
    if (not isinstance(context, dict) or not READINESS_CONTEXT_FIELDS.issubset(context)
            or context.keys() - READINESS_CONTEXT_FIELDS - READINESS_OPTIONAL_CONTEXT_FIELDS
            or not isinstance(inputs["loop_id"], str)
            or not inputs["loop_id"] or request.purpose != "screening_volume" or request.original.kind != "eval"
            or type(inputs["minimum"]) is not int
            or request.minimum_unique_cases != inputs["minimum"]
            or request.minimum_unique_families != inputs["minimum"]
            or request.original.dataset_id != inputs["eval_version"]
            or not any(ref.dataset_id == inputs["train_version"] for ref in request.training_ancestors)):
        raise ValueError("bootstrap readiness request/input mismatch")
    payload = request.model_dump(mode="json")
    if any(key not in payload or payload[key] != value for key, value in context.items()):
        raise ValueError("bootstrap readiness context mismatch")
