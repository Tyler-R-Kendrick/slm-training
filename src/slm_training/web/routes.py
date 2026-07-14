"""Observability (read) and action (exec) routers for the control-plane app.

Read endpoints are pure filesystem reads (safe everywhere, including Vercel).
The gate/promotion evaluation endpoints are pure math (no subprocess, no FS
writes) so the Checkpoints gate editor is fully live even in read-only mode.
Exec endpoints (jobs, comparisons) require ``capabilities.execution`` and 403
otherwise. State (readers / capabilities / jobs registry) lives on ``app.state``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from slm_training.harnesses.experiments.promotion import (
    PromotionCriteria,
    evaluate_promotion,
)
from slm_training.harnesses.model_build.ship_gates import (
    DEFAULT_SHIP_GATES,
    evaluate_ship_gates,
)

observability_router = APIRouter(prefix="/api")
actions_router = APIRouter(prefix="/api")


def _readers(request: Request):
    return request.app.state.readers


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
    return caps


@observability_router.get("/overview")
def overview(request: Request) -> dict[str, Any]:
    return _readers(request).overview()


@observability_router.get("/scoreboards")
def scoreboards(request: Request) -> dict[str, Any]:
    return {"scoreboards": _readers(request).scoreboards()}


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
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    return _readers(request).train_records(version, split=split, limit=limit)


@observability_router.get("/data/test")
def data_test(request: Request) -> dict[str, Any]:
    return _readers(request).test_data()


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


@observability_router.post("/promotion/evaluate")
def promotion_evaluate(payload: PromotionEvalRequest) -> dict[str, Any]:
    criteria = PromotionCriteria(
        category_regression_tolerance=payload.category_regression_tolerance,
        require_rank_stable_top2=payload.require_rank_stable_top2,
        eg_time_lcb_min=payload.eg_time_lcb_min,
    )
    return evaluate_promotion(
        integrity=payload.integrity,
        rankings=payload.rankings,
        eg_time_by_seed=payload.eg_time_by_seed,
        ship_suites=payload.ship_suites,
        criteria=criteria,
    )


# --------------------------------------------------------------------------- #
# Action / exec (require execution)
# --------------------------------------------------------------------------- #
class JobRequest(BaseModel):
    job: str = Field(..., min_length=1, max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)


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
