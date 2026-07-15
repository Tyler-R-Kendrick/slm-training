"""FastAPI app: TwoTower OpenUI annotate playground."""

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from slm_training.harnesses.annotations.store import (
    AnnotationStorageError,
    AnnotationStore,
)
from slm_training.web.capabilities import resolve_capabilities
from slm_training.web.comparisons import BlindedComparisonStore
from slm_training.web.deployments import DeploymentRegistry
from slm_training.web.observability import Readers
from slm_training.web.routes import actions_router, observability_router
from slm_training.web.service import PlaygroundService

STATIC_DIR = Path(__file__).resolve().parent / "static"
SPA_DIR = STATIC_DIR / "app"


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    grammar_constrained: bool = True
    design_md: str | None = Field(default=None, max_length=12000)


class SampleRequest(BaseModel):
    prompt: str | None = Field(default=None, max_length=2000)
    session_id: str | None = Field(default=None, max_length=128)
    grammar_constrained: bool = True
    design_md: str | None = Field(default=None, max_length=12000)
    auto_prompt: bool = True


class IdentityPayload(BaseModel):
    kind: Literal["model", "user", "system"]
    provider: str = Field(..., min_length=1, max_length=160)
    id: str = Field(..., min_length=1, max_length=320)
    model: str | None = Field(default=None, max_length=320)
    display_name: str | None = Field(default=None, max_length=320)
    runtime: str | None = Field(default=None, max_length=160)


class AnnotateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    openui: str = Field(..., min_length=1, max_length=20000)
    rating: Literal["up", "down"]
    description: str | None = Field(default=None, max_length=4000)
    design_md: str | None = Field(default=None, max_length=12000)
    valid: bool | None = None
    session_id: str | None = Field(default=None, max_length=128)
    generation_id: str | None = Field(default=None, max_length=160)
    original_openui: str | None = Field(default=None, max_length=20000)
    human_corrected: bool = False
    identities: dict[str, IdentityPayload] = Field(default_factory=dict)
    meta: dict[str, Any] | None = None


class BrowserGenerationAttemptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    openui: str = Field(default="", max_length=20000)
    attempt: int = Field(..., ge=1, le=3)
    error: str | None = Field(default=None, max_length=4000)
    prior_failures: list[str] = Field(default_factory=list, max_length=6)
    design_md: str | None = Field(default=None, max_length=12000)
    session_id: str | None = Field(default=None, max_length=128)
    meta: dict[str, Any] | None = None


class ServerGenerationAttemptRequest(BaseModel):
    prompt: str | None = Field(default=None, max_length=2000)
    session_id: str | None = Field(default=None, max_length=128)
    grammar_constrained: bool = True
    design_md: str | None = Field(default=None, max_length=12000)
    auto_prompt: bool = True
    attempt: int = Field(..., ge=1, le=3)
    prior_failures: list[str] = Field(default_factory=list, max_length=6)
    request_identity: IdentityPayload | None = None


class BrowserGenerationReviewRequest(BaseModel):
    generation_id: str = Field(..., min_length=1, max_length=160)
    prompt: str = Field(..., min_length=1, max_length=2000)
    openui: str = Field(..., min_length=1, max_length=20000)
    attempt: int = Field(..., ge=1, le=3)
    passed: bool
    score: float = Field(..., ge=0, le=1)
    reasons: list[str] = Field(default_factory=list, max_length=8)
    prior_failures: list[str] = Field(default_factory=list, max_length=6)
    session_id: str | None = Field(default=None, max_length=128)
    meta: dict[str, Any] | None = None


class ComparisonPairRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    champion_run_id: str = Field(..., min_length=1, max_length=160)
    candidate_run_id: str = Field(..., min_length=1, max_length=160)
    champion_openui: str = Field(..., min_length=1, max_length=20000)
    candidate_openui: str = Field(..., min_length=1, max_length=20000)


class ComparisonVoteRequest(BaseModel):
    pair_id: str = Field(..., min_length=1, max_length=160)
    winner: Literal["left", "right", "tie"]
    reviewer_id: str = Field(..., min_length=1, max_length=320)


def create_app(
    checkpoint: Path | None = None,
    device: str = "cpu",
    annotations_path: Path | None = None,
    human_train_path: Path | None = None,
    human_pairs_path: Path | None = None,
    bad_outputs_path: Path | None = None,
    generation_attempts_path: Path | None = None,
    annotation_token: str | None = None,
    execution: bool = False,
    root: Path | None = None,
    annotation_token_required: bool = False,
    annotation_store: AnnotationStore | None = None,
    model_factory: Callable[[Path, str], Any] | None = None,
    deployment_root: Path | None = None,
    comparisons_path: Path | None = None,
) -> FastAPI:
    service = PlaygroundService(
        checkpoint=checkpoint,
        device=device,
        annotations_path=annotations_path,
        human_train_path=human_train_path,
        human_pairs_path=human_pairs_path,
        bad_outputs_path=bad_outputs_path,
        generation_attempts_path=generation_attempts_path,
        annotation_store=annotation_store,
        model_factory=model_factory,
    )
    deployments = DeploymentRegistry(deployment_root or Path("outputs/lineage/deployments"))
    comparisons = BlindedComparisonStore(
        comparisons_path or Path("outputs/annotations/comparisons.jsonl")
    )

    # Control-plane capability + job runner (only wired when execution is allowed).
    app_root = Path(root) if root is not None else Path(".")
    capabilities = resolve_capabilities(execution, outputs_dir=app_root / "outputs")
    registry = None
    if capabilities.execution:
        from slm_training.web.jobs import JobRegistry

        registry = JobRegistry(app_root)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if registry is not None:
            await registry.start()
        yield
        if registry is not None:
            await registry.shutdown()

    app = FastAPI(
        title="TwoTower OpenUI Playground", version="0.2.0", lifespan=lifespan
    )
    app.state.readers = Readers(
        app_root, persist_insights=capabilities.execution
    )
    app.state.capabilities = capabilities
    if registry is not None:
        app.state.jobs = registry

    @app.middleware("http")
    async def _browser_acceleration_headers(request, call_next):
        response = await call_next(request)
        # Chromium requires cross-origin isolation for SharedArrayBuffer-backed
        # multi-threaded WASM. `credentialless` keeps CDN-hosted model assets
        # usable while allowing the local app to consume all available cores.
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        response.headers.setdefault("Cross-Origin-Embedder-Policy", "credentialless")
        return response

    def _require_annotation_token(authorization: str | None) -> None:
        if annotation_token is None:
            if annotation_token_required:
                raise HTTPException(
                    status_code=503,
                    detail="annotation authorization is not configured",
                )
            return
        scheme, _, supplied = (authorization or "").partition(" ")
        if (
            scheme.lower() != "bearer"
            or not supplied
            or not secrets.compare_digest(supplied, annotation_token)
        ):
            raise HTTPException(status_code=401, detail="valid bearer token required")

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, **service.info()}

    @app.get("/api/examples")
    def examples() -> dict:
        return {"examples": service.info()["examples"]}

    @app.get("/api/deployments")
    async def deployment_manifests() -> dict:
        return deployments.payload()

    @app.post("/api/comparisons/pair")
    async def comparison_pair(
        payload: ComparisonPairRequest,
        authorization: str | None = Header(default=None),
    ) -> dict:
        _require_annotation_token(authorization)
        return comparisons.create(**payload.model_dump())

    @app.post("/api/comparisons/vote")
    async def comparison_vote(
        payload: ComparisonVoteRequest,
        authorization: str | None = Header(default=None),
    ) -> dict:
        _require_annotation_token(authorization)
        try:
            return comparisons.vote(
                payload.pair_id, payload.winner, reviewer_id=payload.reviewer_id
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="comparison not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/comparisons/metrics/{candidate_run_id}")
    async def comparison_metrics(candidate_run_id: str) -> dict:
        return comparisons.metrics(candidate_run_id)

    @app.get("/api/prompt/next")
    def prompt_next(session_id: str | None = Query(default=None)) -> dict:
        return service.next_prompt(session_id)

    @app.post("/api/generate")
    def generate(payload: GenerateRequest) -> dict:
        try:
            result = service.generate(
                payload.prompt,
                grammar_constrained=payload.grammar_constrained,
                design_md=payload.design_md,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "prompt": result.prompt,
            "openui": result.openui,
            "valid": result.valid,
            "error": result.error,
            "stream": result.stream,
            "serialized": result.serialized,
        }

    @app.post("/api/sample")
    def sample(payload: SampleRequest) -> dict:
        try:
            return service.sample(
                prompt=(payload.prompt or "").strip() or None,
                session_id=payload.session_id,
                grammar_constrained=payload.grammar_constrained,
                design_md=payload.design_md,
                auto_prompt=payload.auto_prompt,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/server-attempt")
    def server_attempt(payload: ServerGenerationAttemptRequest) -> dict:
        try:
            return service.server_attempt(
                prompt=(payload.prompt or "").strip() or None,
                session_id=payload.session_id,
                grammar_constrained=payload.grammar_constrained,
                design_md=payload.design_md,
                auto_prompt=payload.auto_prompt,
                attempt=payload.attempt,
                prior_failures=payload.prior_failures,
                request_identity=(
                    payload.request_identity.model_dump(exclude_none=True)
                    if payload.request_identity
                    else None
                ),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except AnnotationStorageError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/annotate")
    def annotate(
        payload: AnnotateRequest,
        authorization: str | None = Header(default=None),
    ) -> dict:
        _require_annotation_token(authorization)
        try:
            return service.annotate(
                prompt=payload.prompt,
                openui=payload.openui,
                rating=payload.rating,
                description=payload.description,
                design_md=payload.design_md,
                valid=payload.valid,
                session_id=payload.session_id,
                generation_id=payload.generation_id,
                original_openui=payload.original_openui,
                human_corrected=payload.human_corrected,
                identities={
                    role: identity.model_dump(exclude_none=True)
                    for role, identity in payload.identities.items()
                },
                meta=payload.meta,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except AnnotationStorageError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/generation-attempt")
    def generation_attempt(payload: BrowserGenerationAttemptRequest) -> dict:
        try:
            return service.record_browser_attempt(
                prompt=payload.prompt,
                openui=payload.openui,
                attempt=payload.attempt,
                error=payload.error,
                prior_failures=payload.prior_failures,
                design_md=payload.design_md,
                session_id=payload.session_id,
                meta=payload.meta,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except AnnotationStorageError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/generation-review")
    def generation_review(payload: BrowserGenerationReviewRequest) -> dict:
        try:
            return service.record_browser_review(
                generation_id=payload.generation_id,
                prompt=payload.prompt,
                openui=payload.openui,
                attempt=payload.attempt,
                passed=payload.passed,
                score=payload.score,
                reasons=payload.reasons,
                prior_failures=payload.prior_failures,
                session_id=payload.session_id,
                meta=payload.meta,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except AnnotationStorageError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/annotations/recent")
    def annotations_recent(
        limit: int = Query(default=20, ge=1, le=200),
        authorization: str | None = Header(default=None),
    ) -> dict:
        _require_annotation_token(authorization)
        try:
            return {
                "annotations": service.list_recent(limit=limit),
                "count": service.annotation_count(),
                "path": str(service.annotations_path),
                "storage": service.annotation_store.backend,
            }
        except AnnotationStorageError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    app.include_router(observability_router)
    app.include_router(actions_router)

    def _spa() -> FileResponse:
        spa_index = SPA_DIR / "index.html"
        if spa_index.exists():
            return FileResponse(spa_index)
        raise HTTPException(
            status_code=503,
            detail="dashboard bundle missing; run npm run dashboard:build",
        )

    @app.get("/")
    def index() -> FileResponse:
        return _spa()

    @app.get("/playground/classic")
    def playground_classic() -> RedirectResponse:
        # Preserve old bookmarks while keeping one supported playground.
        return RedirectResponse(url="/playground", status_code=308)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str) -> FileResponse:
        # Client-side routing: serve the SPA shell for unknown non-API paths.
        if full_path.startswith(("api/", "static/")):
            raise HTTPException(status_code=404, detail="not found")
        return _spa()

    app.state.service = service
    return app
