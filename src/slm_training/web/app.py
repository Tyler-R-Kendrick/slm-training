"""FastAPI app: TwoTower OpenUI playground."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from slm_training.web.service import PlaygroundService

STATIC_DIR = Path(__file__).resolve().parent / "static"


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    grammar_constrained: bool = True
    design_md: str | None = Field(default=None, max_length=12000)


def create_app(
    checkpoint: Path | None = None,
    device: str = "cpu",
) -> FastAPI:
    service = PlaygroundService(checkpoint=checkpoint, device=device)
    app = FastAPI(title="TwoTower OpenUI Playground", version="0.1.0")

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, **service.info()}

    @app.get("/api/examples")
    def examples() -> dict:
        return {"examples": service.info()["examples"]}

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

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.state.service = service
    return app
