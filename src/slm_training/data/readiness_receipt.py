"""Current data predicate evidence, independent of repair execution ownership."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from slm_training.data.readiness_contract import SnapshotInput, load_locked_readiness_request
from slm_training.harnesses.train_data.readiness import check_readiness
from slm_training.lineage.records import content_sha


class DataActionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal["autotrain_data_action_evidence/v1"] = "autotrain_data_action_evidence/v1"
    loop_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    action_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_root: str = Field(min_length=1)
    ready_event_id: str = Field(min_length=1)
    candidate: SnapshotInput


def current_successors(store, *, cwd, loop_id, action_sha=None, kind=None):
    """Return only currently restored predicates from the authoritative history."""

    results = []
    for event in store.verify_event_chain():
        if event["event_type"] not in {"data_successor_ready", "screening_successor_ready"}:
            continue
        detail = event["detail"]
        if detail.get("loop_id") != loop_id or detail.get("data_root") != str(Path(cwd).resolve()):
            continue
        request = load_locked_readiness_request(store, wanted=detail["request_sha256"])
        if action_sha is not None and request.action_id != action_sha:
            continue
        if kind is not None and request.original.kind != kind:
            continue
        candidate = SnapshotInput.model_validate(detail["candidate"])
        artifact = store.root / "artifacts/data_readiness" / f"{event['artifact_sha256']}.json"
        observed = json.loads(artifact.read_text())
        if (content_sha(observed) != event["artifact_sha256"]
                or observed.get("request_sha256") != request.sha256
                or observed.get("snapshot") != candidate.model_dump() or observed.get("ready") is not True):
            raise ValueError("data successor observation integrity mismatch")
        expected = Path("outputs/data") / request.original.kind / request.successor_id
        if Path(candidate.directory) != expected:
            raise ValueError("data successor publication destination mismatch")
        if check_readiness(request, root=Path(cwd), candidate=candidate)["ready"]:
            results.append((request, candidate, event))
    return results


def validate_data_action_evidence(store, handoff, action_sha, path):
    """Used at receipt ingestion AND replay; a stale receipt cannot heal again."""

    evidence = DataActionEvidence.model_validate_json(path.read_text())
    if (evidence.campaign_id != handoff.campaign_id or evidence.loop_id != handoff.loop_id
            or evidence.action_sha256 != action_sha):
        raise ValueError("wrong data action evidence identity")
    if (path.parent.resolve() != (store.root / "artifacts/data_action_evidence").resolve()
            or content_sha(evidence.model_dump(mode="json")) != path.stem):
        raise ValueError("data action evidence artifact identity mismatch")
    matches = current_successors(store, cwd=Path(evidence.data_root),
                                 loop_id=handoff.loop_id, action_sha=evidence.action_sha256)
    if len(matches) != 1:
        raise ValueError("missing or ambiguous current data predicate")
    request, candidate, event = matches[0]
    if (request.sha256 != evidence.request_sha256 or candidate != evidence.candidate
            or event["event_id"] != evidence.ready_event_id):
        raise ValueError("stale data action evidence")
    return evidence, request
