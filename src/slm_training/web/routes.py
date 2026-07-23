"""Observability (read) and action (exec) routers for the control-plane app.

Read endpoints are pure filesystem reads (safe everywhere, including Vercel).
The gate/promotion evaluation endpoints are pure math (no subprocess, no FS
writes) so the Checkpoints gate editor is fully live even in read-only mode.
Exec endpoints (jobs, comparisons) require ``capabilities.execution`` and 403
otherwise. State (readers / capabilities / jobs registry) lives on ``app.state``.

``otel_ingest_router`` has no ``/api`` prefix: it serves the standard OTLP/HTTP
paths (``/v1/traces``, ``/v1/logs``) that ``RunTrace._mirror`` derives from a
base endpoint URL, so any app instance can act as a telemetry peer.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from slm_training.web.otel_hub import MAX_INGEST_BYTES

from slm_training.autoresearch.run_insights import RunInsightSubmission
from slm_training.harnesses.experiments.promotion import (
    PromotionCriteria,
    evaluate_promotion,
)
from slm_training.harnesses.model_build.ship_gates import (
    DEFAULT_SHIP_GATES,
    evaluate_ship_gates,
)
from slm_training.features.levers import feature_flag_registry_payload

observability_router = APIRouter(prefix="/api")
actions_router = APIRouter(prefix="/api")
otel_ingest_router = APIRouter()


def _readers(request: Request):
    return request.app.state.readers


def _otel(request: Request):
    return request.app.state.otel


def _require_execution(request: Request):
    caps = request.app.state.capabilities
    if not caps.execution:
        raise HTTPException(
            status_code=403, detail="execution disabled (read-only deployment)"
        )
    return request.app.state.jobs


# --------------------------------------------------------------------------- #
# Read / observability
# --------------------------------------------------------------------------- #
@observability_router.get("/capabilities")
def capabilities(request: Request) -> dict[str, Any]:
    from slm_training.web.jobs import catalog

    caps = request.app.state.capabilities.to_dict()
    caps["jobs"] = catalog()
    caps["run_insights"] = {
        "browser": True,
        "openai_available": bool(os.getenv("OPENAI_API_KEY")),
    }
    hub = getattr(request.app.state, "otel", None)
    caps["otel"] = {
        "hub": bool(hub and hub.enabled),
        "peers_configured": bool(hub and hub.peers),
        "auth_mode": hub.auth_mode if hub else "open",
    }
    features = getattr(request.app.state, "features", None)
    caps["features"] = {
        "openfeature": bool(features),
        "provider": features.provider if features else None,
    }
    return caps


@observability_router.get("/features/bootstrap")
def features_bootstrap(
    request: Request,
    targeting_key: str = Query(default="anonymous", max_length=320),
) -> dict[str, Any]:
    features = getattr(request.app.state, "features", None)
    if features is None:
        raise HTTPException(status_code=503, detail="feature runtime unavailable")
    return features.bootstrap_payload(targeting_key=targeting_key)


@observability_router.get("/features/levers")
def features_levers() -> dict[str, Any]:
    return feature_flag_registry_payload()


@observability_router.get("/overview")
def overview(request: Request) -> dict[str, Any]:
    return _readers(request).overview()


@observability_router.get("/scoreboards")
def scoreboards(request: Request) -> dict[str, Any]:
    return {"scoreboards": _readers(request).scoreboards()}


@observability_router.get("/experiment-flags")
def experiment_flags(request: Request) -> dict[str, Any]:
    return _readers(request).experiment_flags()


@observability_router.get("/experiment-flags/{key}")
def experiment_flag(request: Request, key: str) -> dict[str, Any]:
    detail = _readers(request).experiment_flag(key)
    if detail is None:
        raise HTTPException(status_code=404, detail="unknown experiment feature flag")
    return detail


@observability_router.get("/scoreboards/{kind}")
def scoreboard(request: Request, kind: str) -> dict[str, Any]:
    board = _readers(request).scoreboard(kind)
    if board["provenance"] == "unknown":
        raise HTTPException(status_code=404, detail=f"unknown scoreboard: {kind}")
    return board


@observability_router.get("/runs")
def runs(request: Request) -> dict[str, Any]:
    return _readers(request).runs()


@observability_router.get("/runs/{run_id}")
def run(request: Request, run_id: str) -> dict[str, Any]:
    return _readers(request).run(run_id)


@observability_router.get("/runs/{run_id}/data")
def run_training_data(request: Request, run_id: str) -> dict[str, Any]:
    return _readers(request).run_training_data(run_id)


@observability_router.get("/runs/{run_id}/rl-traces")
def rl_traces(
    request: Request,
    run_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
) -> dict[str, Any]:
    return _readers(request).rl_traces(run_id, offset=offset, limit=limit)


@observability_router.get("/lineage/champions")
def champions(request: Request) -> dict[str, Any]:
    return _readers(request).champions()


@observability_router.get("/lineage/deployments")
def deployments(request: Request) -> dict[str, Any]:
    return _readers(request).deployment_state()


@observability_router.get("/checkpoints")
def checkpoints(request: Request) -> dict[str, Any]:
    return _readers(request).checkpoints()


@observability_router.get("/checkpoints/{run_id}/gates")
def checkpoint_gates(request: Request, run_id: str) -> dict[str, Any]:
    return _readers(request).gates_for_run(run_id)


@observability_router.get("/data/train")
def data_train(request: Request, version: str | None = Query(default=None)) -> dict[str, Any]:
    return _readers(request).train_data(version)


@observability_router.get("/data/train/{version}/records")
def data_train_records(
    request: Request,
    version: str,
    split: str | None = Query(default=None),
    source: str | None = Query(default=None),
    q: str | None = Query(default=None, max_length=200),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    return _readers(request).train_records(
        version, split=split, source=source, query=q, offset=offset, limit=limit
    )


@observability_router.get("/data/train/{version}/quality")
def data_train_quality(request: Request, version: str) -> dict[str, Any]:
    return _readers(request).train_quality(version)


@observability_router.get("/data/train/{version}/rejected")
def data_train_rejected(
    request: Request,
    version: str,
    stage: str | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    return _readers(request).train_rejected(
        version, stage=stage, offset=offset, limit=limit
    )


@observability_router.get("/data/preference")
def data_preference(request: Request) -> dict[str, Any]:
    return _readers(request).preference_data()


@observability_router.get("/data/test")
def data_test(request: Request) -> dict[str, Any]:
    return _readers(request).test_data()


@observability_router.get("/data/test/records")
def data_test_records(
    request: Request,
    suite: str | None = Query(default=None),
    q: str | None = Query(default=None, max_length=200),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    return _readers(request).test_records(suite, query=q, offset=offset, limit=limit)


@observability_router.get("/annotations/summary")
def annotations_summary(request: Request) -> dict[str, Any]:
    return _readers(request).annotations_summary()


@observability_router.get("/comparisons/metrics")
def comparison_metrics(
    request: Request, candidate_run_id: str = Query(...)
) -> dict[str, Any]:
    return _readers(request).comparison_metrics(candidate_run_id)


@observability_router.get("/system")
def system(request: Request) -> dict[str, Any]:
    return _readers(request).system()


@observability_router.get("/dispatches")
def dispatches(request: Request) -> dict[str, Any]:
    return _readers(request).dispatches()


# --------------------------------------------------------------------------- #
# OTel hub: active runs across peers + lazy per-run event streams
# --------------------------------------------------------------------------- #
@observability_router.get("/otel/runs")
def otel_runs(request: Request, local: int = Query(default=0)) -> dict[str, Any]:
    # Sync handler on purpose: peer fetches are blocking urllib calls and must
    # run on the threadpool, never on the event loop. ``local=1`` is the
    # federation contract — peers only ever serve their own ingested state.
    return _otel(request).merged_runs(local_only=bool(local))


@observability_router.get("/otel/runs/{run_id}/events")
def otel_events(
    request: Request,
    run_id: str,
    since: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=500),
) -> dict[str, Any]:
    return _otel(request).events(run_id, since=since, limit=limit)


@observability_router.get("/otel/runs/{run_id}/stream")
async def otel_stream(
    request: Request, run_id: str, since: int = Query(default=0, ge=0)
) -> StreamingResponse:
    hub = _otel(request)
    if hub.enabled and hub.has_local(run_id):
        generator = hub.stream(run_id, since=since)
    else:
        loop = asyncio.get_running_loop()
        peer = await loop.run_in_executor(None, hub.find_peer_for, run_id)
        if peer is not None:
            generator = hub.stream_remote(run_id, peer, since=since)
        elif hub.enabled:
            # Not seen anywhere yet: subscribe locally so a run that starts
            # after the viewer opens the page streams from its first event.
            generator = hub.stream(run_id, since=since)
        else:
            raise HTTPException(
                status_code=503,
                detail="otel streaming unavailable on this deployment",
            )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@otel_ingest_router.post("/v1/{signal}")
async def otel_ingest(signal: str, request: Request) -> dict[str, Any]:
    if signal not in {"traces", "logs"}:
        raise HTTPException(status_code=404, detail=f"unknown OTLP signal: {signal}")
    hub = _otel(request)
    if not hub.enabled:
        raise HTTPException(
            status_code=503, detail="otel hub disabled on this deployment"
        )
    ok, user = await hub.authorize(request.headers.get("authorization"))
    if not ok:
        raise HTTPException(status_code=401, detail="valid bearer token required")
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > MAX_INGEST_BYTES:
        raise HTTPException(status_code=413, detail="payload too large")
    # Stream with an incremental cap instead of request.body(): an oversized
    # or lying upload 413s at the threshold rather than buffering fully first.
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_INGEST_BYTES:
            raise HTTPException(status_code=413, detail="payload too large")
    try:
        payload = json.loads(bytes(body))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc
    return hub.ingest(signal, payload, user=user)


# --------------------------------------------------------------------------- #
# Pure-compute gate/promotion evaluation (read-only-safe; powers gate editor)
# --------------------------------------------------------------------------- #
class GateEvalRequest(BaseModel):
    suites: dict[str, dict[str, Any]]
    thresholds: dict[str, dict[str, float]] | None = None


@observability_router.get("/gates/policy")
def gates_policy() -> dict[str, Any]:
    return {"policy": DEFAULT_SHIP_GATES}


@observability_router.post("/gates/evaluate")
def gates_evaluate(payload: GateEvalRequest) -> dict[str, Any]:
    return evaluate_ship_gates(payload.suites, thresholds=payload.thresholds)


class PromotionEvalRequest(BaseModel):
    ship_suites: dict[str, dict[str, Any]] | None = None
    integrity: dict[str, Any] | None = None
    rankings: dict[str, list[str]] | None = None
    eg_time_by_seed: list[float] | None = None
    category_regression_tolerance: float = 0.02
    eg_time_lcb_min: float = 1.0
    require_rank_stable_top2: bool = True
    campaign_manifest: dict[str, Any] | None = None
    campaign_result: dict[str, Any] | None = None
    campaign_store_root: Path | None = None
    campaign_artifact_root: Path | None = None


@observability_router.post("/promotion/evaluate")
def promotion_evaluate(payload: PromotionEvalRequest) -> dict[str, Any]:
    criteria = PromotionCriteria(
        category_regression_tolerance=payload.category_regression_tolerance,
        require_rank_stable_top2=payload.require_rank_stable_top2,
        eg_time_lcb_min=payload.eg_time_lcb_min,
    )
    campaign_store = None
    if payload.campaign_manifest is not None and payload.campaign_store_root is not None:
        from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
        from slm_training.autoresearch.storage import CampaignStore

        manifest = ExperimentCampaignV1.model_validate(payload.campaign_manifest)
        campaign_store = CampaignStore(
            manifest.campaign_id, payload.campaign_store_root
        )
    return evaluate_promotion(
        integrity=payload.integrity,
        rankings=payload.rankings,
        eg_time_by_seed=payload.eg_time_by_seed,
        ship_suites=payload.ship_suites,
        criteria=criteria,
        campaign_manifest=payload.campaign_manifest,
        campaign_result=payload.campaign_result,
        campaign_store=campaign_store,
        artifact_root=payload.campaign_artifact_root,
    )


# --------------------------------------------------------------------------- #
# Action / exec (require execution)
# --------------------------------------------------------------------------- #
class JobRequest(BaseModel):
    job: str = Field(..., min_length=1, max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)


@actions_router.post("/runs/{run_id}/insights")
def persist_run_insights(
    request: Request, run_id: str, payload: RunInsightSubmission
) -> dict[str, Any]:
    _require_execution(request)
    try:
        return _readers(request).save_run_insights(run_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=503, detail="run insights are not writable") from exc


@actions_router.post("/runs/{run_id}/insights/openai")
def enrich_run_insights_openai(request: Request, run_id: str) -> dict[str, Any]:
    _require_execution(request)
    try:
        return _readers(request).enrich_run_with_openai(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=503, detail="run insights are not writable") from exc


@actions_router.post("/jobs")
def submit_job(request: Request, payload: JobRequest) -> dict[str, Any]:
    registry = _require_execution(request)
    try:
        job = registry.submit(payload.job, payload.params)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"unknown job: {payload.job}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return job.to_dict()


@actions_router.get("/jobs")
def list_jobs(request: Request) -> dict[str, Any]:
    caps = request.app.state.capabilities
    if not caps.execution:
        return {"execution": False, "jobs": []}
    return {"execution": True, "jobs": [j.to_dict() for j in request.app.state.jobs.list()]}


@actions_router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> dict[str, Any]:
    registry = _require_execution(request)
    job = registry.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return {**job.to_dict(), "tail": registry.tail(job_id, lines=40)}


@actions_router.get("/jobs/{job_id}/logs")
def job_logs(request: Request, job_id: str) -> StreamingResponse:
    registry = _require_execution(request)
    if job_id not in registry.jobs:
        raise HTTPException(status_code=404, detail="unknown job")
    return StreamingResponse(
        registry.stream(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@actions_router.post("/jobs/{job_id}/cancel")
def cancel_job(request: Request, job_id: str) -> dict[str, Any]:
    registry = _require_execution(request)
    if job_id not in registry.jobs:
        raise HTTPException(status_code=404, detail="unknown job")
    return registry.cancel(job_id).to_dict()


class ComparisonCreate(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    champion_run_id: str = Field(..., max_length=128)
    candidate_run_id: str = Field(..., max_length=128)
    champion_openui: str = Field(..., max_length=20000)
    candidate_openui: str = Field(..., max_length=20000)


@actions_router.post("/comparisons")
def create_comparison(request: Request, payload: ComparisonCreate) -> dict[str, Any]:
    _require_execution(request)
    return request.app.state.readers.comparisons.create(
        prompt=payload.prompt,
        champion_run_id=payload.champion_run_id,
        candidate_run_id=payload.candidate_run_id,
        champion_openui=payload.champion_openui,
        candidate_openui=payload.candidate_openui,
    )


class ComparisonVote(BaseModel):
    winner: str = Field(..., pattern="^(left|right|tie)$")
    reviewer_id: str = Field(default="anon", max_length=128)


@actions_router.post("/comparisons/{pair_id}/vote")
def vote_comparison(
    request: Request, pair_id: str, payload: ComparisonVote
) -> dict[str, Any]:
    _require_execution(request)
    try:
        return request.app.state.readers.comparisons.vote(
            pair_id, payload.winner, reviewer_id=payload.reviewer_id  # type: ignore[arg-type]
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown comparison")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
