"""Pinned post-training inputs for the existing promotion decision and handoff.

Campaign events own this continuation. It never selects arms, trains a model,
changes policy or issues a promotion verdict itself.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scripts.merge_verification_evidence import environment_identity
from scripts.run_autotrain_supervisor import _source_identity as source_identity
from slm_training.autoresearch.climb_policy import load_climb_policy
from slm_training.lineage.records import content_sha as digest
from slm_training.levers import INTERRUPT_AFTER_SECONDS


class PromotionFinalizationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["promotion_finalization/v1"] = "promotion_finalization/v1"
    campaign_id: str
    loop_id: str
    cycle_index: int = Field(ge=1)
    upstream_commit: str
    integration_commit: str
    role: Literal["promotion"]
    cycle_intent: Literal["promote"]
    primary_metric: str
    matrix: dict
    entry: dict
    control_id: str
    candidate_id: str
    arm_order: list[str]
    arm_seed: int
    arm_exits: dict[str, int]
    arm_skipped: dict[str, dict]
    formal_status: str | None
    skip_slugs: list[str]
    publication_repository: str | None = Field(default=None, exclude_if=lambda value: value is None)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifests: dict[str, str]

    @model_validator(mode="after")
    def validate_binding(self):
        names = [self.campaign_id, self.loop_id, self.control_id, self.candidate_id]
        if any(not name or Path(name).name != name or name in {".", ".."} or "\\" in name for name in names):
            raise ValueError("invalid continuation identity")
        arms = {self.control_id, self.candidate_id}
        if (len(arms) != 2 or len(self.arm_order) != 2 or set(self.arm_order) != arms
            or set(self.manifests) != arms or not self.entry.get("entry_id")):
            raise ValueError("promotion continuation requires the exact locked pair and entry")
        return self


def _manifests(store, arms):
    if any(not isinstance(arm, str) or Path(arm).name != arm or arm in {".", ".."} or "\\" in arm for arm in arms):
        raise ValueError("invalid manifest identity")
    return {arm: hashlib.sha256((store.root / "manifests" / f"{arm}.json").read_bytes()).hexdigest()
            for arm in arms}


def lock_finalization(store, cwd, payload):
    """Called by the driver before chunk dispatch, never inferred after a restart."""
    from scripts.autotrain_source_publication import _REPOSITORY

    value = PromotionFinalizationInput.model_validate({
        **payload, "source_sha256": source_identity(cwd),
        "publication_repository": _REPOSITORY.get(),
        "environment_sha256": digest(environment_identity()),
        "policy_sha256": load_climb_policy().sha256,
        "manifests": _manifests(store, payload["arm_order"]),
    }).model_dump(mode="json")
    existing = load_finalization(store)
    if existing is not None and existing != value:
        raise ValueError("promotion finalization inputs changed")
    artifact = store.write_artifact("promotion_finalization", value)
    store.append_event("promotion_finalization_locked", artifact_sha256=artifact.stem,
                       idempotency_key="promotion-finalization:locked")
    return value


def load_finalization(store):
    events = [row for row in store.verify_event_chain() if row["event_type"] == "promotion_finalization_locked"]
    if not events:
        return None  # Legacy campaigns do not acquire invented continuation inputs.
    sha = events[-1]["artifact_sha256"]
    value = json.loads((store.root / "artifacts/promotion_finalization" / f"{sha}.json").read_text())
    if digest(value) != sha:
        raise ValueError("promotion finalization artifact changed")
    return PromotionFinalizationInput.model_validate(value).model_dump(mode="json")


def finalization_pending(store, ledger):
    if not ledger or not ledger["arms"] or not all(row["status"] == "complete" for row in ledger["arms"].values()):
        return False
    return load_finalization(store) is not None and _completed_outputs(store) is None


def _completed_outputs(store):
    events = [row for row in store.verify_event_chain() if row["event_type"] == "promotion_finalized"]
    if not events:
        return None
    sha = events[-1]["artifact_sha256"]
    value = json.loads((store.root / "artifacts/promotion_finalization" / f"{sha}.json").read_text())
    if digest(value) != sha or value["input_sha256"] != digest(load_finalization(store)):
        raise ValueError("completed finalization artifact changed")
    for name, expected in value["outputs"].items():
        if name not in {"sdlc_delivery.json", "cycle_handoff.json"}:
            raise ValueError("invalid finalization output path")
        if hashlib.sha256((store.root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("completed finalization output changed")
    return value


def _verify_inputs(store, cwd, value, ledger):
    from scripts.autotrain_promotion_chunks import load_ledger

    if (value["campaign_id"] != store.campaign_id
        or value["source_sha256"] != source_identity(cwd)
        or value["environment_sha256"] != digest(environment_identity())
        or value["policy_sha256"] != load_climb_policy().sha256
        or value["manifests"] != _manifests(store, value["arm_order"])):
        raise ValueError("promotion finalization source/environment/policy/input mismatch")
    if set(ledger["arms"]) != set(value["arm_order"]):
        raise ValueError("promotion continuation is missing a locked arm")
    if ledger != load_ledger(store):
        raise ValueError("promotion continuation differs from its committed ledger")


def _completion_exits(value, ledger):
    exits = {}
    for arm_id, arm in ledger["arms"].items():
        if arm["status"] != "complete":
            raise ValueError("promotion finalization requires complete measurement")
        runs = arm["runs"]
        code = runs[-1].get("exit_code") if runs else value["arm_exits"].get(arm_id)
        if type(code) is not int or code not in {0, 2, 3, 4, 5, 6, 7, 8}:
            raise ValueError("completed artifact needs independent interrupted-publication reconciliation")
        exits[arm_id] = code  # Actual final evaluation exit, not a rewritten training attempt.
    return exits


def finalize_promotion(store, cwd, continuous, ledger, deadline=None):
    """Resume only the original decision/receipt consumers; never re-enter training."""
    value = load_finalization(store)
    if value is None:
        raise ValueError("legacy promotion lacks locked finalization inputs")
    _verify_inputs(store, cwd, value, ledger)
    from scripts.autotrain_source_publication import require_source_publication
    require_source_publication(store, value["loop_id"], measurement_complete=bool(ledger["arms"])
        and all(arm["status"] == "complete" for arm in ledger["arms"].values()))
    if not finalization_pending(store, ledger):
        return {"campaign_id": store.campaign_id, "already_finalized": True}
    exits = _completion_exits(value, ledger)
    deadline = min(deadline if deadline is not None else float("inf"), time.monotonic() + INTERRUPT_AFTER_SECONDS)
    common = {"root": store.root.parent, "loop_id": value["loop_id"], "campaign_id": store.campaign_id,
              "cycle_index": value["cycle_index"]}
    delivery = continuous._phase_a_delivery(
        **common, cwd=cwd, deadline=deadline,
        **{key: value[key] for key in ("primary_metric", "role", "cycle_intent", "arm_order", "arm_seed",
                                      "control_id", "candidate_id", "arm_skipped")}, arm_exits=exits,
    )
    delivery = continuous._attach_promotion_chunks(delivery, ledger)
    delivery["original_arm_exits"] = value["arm_exits"]
    delivery, resolution = continuous._resolve_promote_delivery(
        {**common, "entry": value["entry"], "formal_status": value["formal_status"], "arm_exits": exits},
        delivery, deadline,
    )
    continuous._write_cycle_handoff(
        **common, cwd=cwd, delivery=delivery, resolution=resolution,
        **{key: value[key] for key in ("upstream_commit", "integration_commit", "role", "cycle_intent",
                                      "primary_metric", "matrix", "formal_status")},
        skip_slugs=frozenset(value["skip_slugs"]),
    )
    outputs = {name: hashlib.sha256((store.root / name).read_bytes()).hexdigest()
               for name in ("sdlc_delivery.json", "cycle_handoff.json")}
    artifact = store.write_artifact("promotion_finalization", {"input_sha256": digest(value), "outputs": outputs})
    store.append_event("promotion_finalized", artifact_sha256=artifact.stem,
                       idempotency_key="promotion-finalization:completed")
    return {"campaign_id": store.campaign_id, "resolution": resolution, "outputs": outputs}
