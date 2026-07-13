"""FastAPI app: TwoTower OpenUI annotate playground."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from slm_training.web.service import PlaygroundService

STATIC_DIR = Path(__file__).resolve().parent / "static"


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
) -> FastAPI:
    service = PlaygroundService(
        checkpoint=checkpoint,
        device=device,
        annotations_path=annotations_path,
        human_train_path=human_train_path,
        human_pairs_path=human_pairs_path,
    )
    app = FastAPI(title="TwoTower OpenUI Playground", version="0.2.0")

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
            prompt = (payload.prompt or "").strip()
            session_id = payload.session_id
            if not prompt and payload.auto_prompt:
                nxt = service.next_prompt(session_id)
                prompt = nxt["prompt"]
                session_id = nxt["session_id"]
            if not prompt:
                raise ValueError("prompt must be non-empty (or enable auto_prompt)")
            result = service.generate(
                prompt,
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
            "session_id": session_id,
        }

    @app.post("/api/annotate")
    def annotate(payload: AnnotateRequest) -> dict:
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
    def annotations_recent(limit: int = Query(default=20, ge=1, le=200)) -> dict:
        return {
            "annotations": service.list_recent(limit=limit),
            "count": service.annotation_count(),
            "path": str(service.annotations_path),
        }

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.state.service = service
    return app
