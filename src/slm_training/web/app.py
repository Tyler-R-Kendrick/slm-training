"""FastAPI app: TwoTower OpenUI annotate playground."""

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from slm_training.web.capabilities import resolve_capabilities
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


class AnnotateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    openui: str = Field(..., min_length=1, max_length=20000)
    rating: Literal["up", "down"]
    description: str | None = Field(default=None, max_length=4000)
    design_md: str | None = Field(default=None, max_length=12000)
    valid: bool | None = None
    session_id: str | None = Field(default=None, max_length=128)
    meta: dict[str, Any] | None = None


def create_app(
    checkpoint: Path | None = None,
    device: str = "cpu",
    annotations_path: Path | None = None,
    human_train_path: Path | None = None,
    human_pairs_path: Path | None = None,
    bad_outputs_path: Path | None = None,
    annotation_token: str | None = None,
    execution: bool = False,
    root: Path | None = None,
) -> FastAPI:
    service = PlaygroundService(
        checkpoint=checkpoint,
        device=device,
        annotations_path=annotations_path,
        human_train_path=human_train_path,
        human_pairs_path=human_pairs_path,
        bad_outputs_path=bad_outputs_path,
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
    app.state.readers = Readers(app_root)
    app.state.capabilities = capabilities
    if registry is not None:
        app.state.jobs = registry

    def _require_annotation_token(authorization: str | None) -> None:
        if annotation_token is None:
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
                meta=payload.meta,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/annotations/recent")
    def annotations_recent(
        limit: int = Query(default=20, ge=1, le=200),
        authorization: str | None = Header(default=None),
    ) -> dict:
        _require_annotation_token(authorization)
        return {
            "annotations": service.list_recent(limit=limit),
            "count": service.annotation_count(),
            "path": str(service.annotations_path),
        }

    app.include_router(observability_router)
    app.include_router(actions_router)

    def _spa_or_classic() -> FileResponse:
        spa_index = SPA_DIR / "index.html"
        if spa_index.exists():
            return FileResponse(spa_index)
        # Cold path (SPA bundle not built): serve the classic playground so the
        # app is never broken.
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/")
    def index() -> FileResponse:
        return _spa_or_classic()

    @app.get("/playground")
    def playground() -> FileResponse:
        # Classic annotate playground stays reachable as a standalone page.
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str) -> FileResponse:
        # Client-side routing: serve the SPA shell for unknown non-API paths.
        if full_path.startswith(("api/", "static/")):
            raise HTTPException(status_code=404, detail="not found")
        return _spa_or_classic()

    app.state.service = service
    return app
