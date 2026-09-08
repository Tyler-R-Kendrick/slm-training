"""Default-off offline comparative analysis, with locked request and evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from slm_training.autoresearch.paired_stats import compare_fixed_sequential_signs
from slm_training.harness_core.activity_contract import contract_digest
from slm_training.autoresearch.storage import CampaignStore, _sha
from slm_training.versioning import build_version_stamp


class SignComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    schema_version: Literal["offline_sign_request/v1"]
    campaign_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    locked_design_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    endpoint: Literal["candidate_better_probability_conditional_on_non_tie"]
    unit_ids: tuple[str, ...]
    deltas: tuple[float, ...]
    iid_units_declared: Literal[True]
    design_law: str = Field(min_length=20, max_length=16384)
    ancestor_claim: str = Field(min_length=1, max_length=4096)
    alpha: str = "1/20"


def compare_sign_request(path: Path, root: Path, docs: Path) -> dict:
    if path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise ValueError("unsafe_analysis_request")
    request = SignComparisonRequest.model_validate_json(path.read_text())
    store = CampaignStore(request.campaign_id, root)
    payload = request.model_dump(mode="json")
    request_digest = contract_digest(payload)
    prior = [
        event
        for event in store.verify_event_chain()
        if event["event_type"] == "offline_sign_request_locked"
    ]
    if prior and prior[0]["detail"]["request_digest"] != request_digest:
        raise ValueError("immutable_analysis_request_changed")
    artifact = store.write_artifact("offline_sign_request", payload)
    store.append_event(
        "offline_sign_request_locked",
        idempotency_key="offline-sign-request",
        detail={
            "request_digest": request_digest,
            "artifact": str(artifact),
            "claim": "offline_reanalysis_not_prospective_preregistration",
        },
    )
    result = _completed_result(store, request_digest) or _measure(
        store, request, request_digest
    )
    _publish_docs(docs, result, request_digest)
    return result


def _completed_result(store: CampaignStore, request_digest: str) -> dict | None:
    completed = [
        event
        for event in store.verify_event_chain()
        if event["event_type"] == "offline_sign_analysis_completed"
        and event["detail"]["request_digest"] == request_digest
    ]
    if not completed:
        return None
    path = Path(completed[-1]["detail"]["artifact"])
    expected_parent = store.root / "artifacts/offline_sign_result"
    if path.is_symlink() or path.resolve().parent != expected_parent.resolve():
        raise ValueError("analysis_artifact_outside_namespace")
    result = json.loads(path.read_text())
    if _sha(result) != path.stem or result.get("request_digest") != request_digest:
        raise ValueError("analysis_artifact_integrity_failure")
    return result


def _measure(store, request, request_digest):
    result = compare_fixed_sequential_signs(
        request.deltas,
        unit_ids=request.unit_ids,
        iid_units_declared=request.iid_units_declared,
        alpha=request.alpha,
    )
    result.update(
        request_digest=request_digest,
        locked_design_digest=request.locked_design_digest,
        design_law=request.design_law,
        ancestor_claim=request.ancestor_claim,
        version_stamp=build_version_stamp("autoresearch.heal"),
    )
    output = store.write_artifact("offline_sign_result", result)
    store.append_event(
        "offline_sign_analysis_completed",
        idempotency_key="offline-sign-result:" + request_digest,
        detail={
            "request_digest": request_digest,
            "artifact": str(output),
            "promotion_authority": False,
        },
    )
    return result


def _publish_docs(docs, result, request_digest):
    docs.mkdir(parents=True, exist_ok=True)
    stem = "autonomy-sign-comparison-" + request_digest[:16]
    for destination, text in (
        (docs / (stem + ".json"), json.dumps(result, indent=2, sort_keys=True) + "\n"),
        (
            docs / (stem + ".md"),
            "# Offline probability-sign comparison\n\n"
            + "Request: `"
            + request_digest
            + "`. Existing fixed analysis and promotion are unchanged.\n\n"
            + "Independent-unit law is declared, not proven by distinct IDs. No mean-NLL inference, "
            + "holdout-reuse guarantee, model training or confirmation claim.\n\n"
            + "Fixed-horizon verdict: `"
            + result["fixed_horizon_verdict"]
            + "`; see JSON for intervals.\n",
        ),
    ):
        if destination.exists() and destination.read_text() != text:
            raise ValueError("immutable_analysis_document_conflict")
        if not destination.exists():
            CampaignStore._atomic_new(destination, text)
