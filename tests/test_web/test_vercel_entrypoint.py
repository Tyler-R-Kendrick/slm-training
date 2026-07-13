"""Vercel FastAPI entrypoint smoke tests."""

from __future__ import annotations

from fastapi import FastAPI


def test_api_index_exports_fastapi_app() -> None:
    from api.index import app

    assert isinstance(app, FastAPI)
    assert app.title == "TwoTower OpenUI Playground"
    routes = {getattr(route, "path", None) for route in app.routes}
    assert "/api/health" in routes
    assert "/" in routes


def test_playground_service_imports_without_torch() -> None:
    """Cold-start path must not require torch (Vercel entrypoint constraint)."""
    import sys

    # Ensure a clean check: service module should not have pulled twotower yet.
    sys.modules.pop("slm_training.models.twotower", None)
    from slm_training.web import service as service_mod

    assert "slm_training.models.twotower" not in sys.modules
    svc = service_mod.PlaygroundService(checkpoint=service_mod.DEFAULT_CHECKPOINT)
    assert svc.ready is True or svc.ready is False  # exists() bool
    assert svc._model is None
