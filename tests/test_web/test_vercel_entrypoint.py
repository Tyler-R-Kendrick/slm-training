"""Vercel FastAPI entrypoint smoke tests."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from fastapi import FastAPI


def test_vercel_entrypoint_points_to_a_module_file() -> None:
    root = Path(__file__).parents[2]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    module, separator, name = config["tool"]["vercel"]["entrypoint"].partition(":")

    assert (separator, name) == (":", "app")
    assert root.joinpath(*module.split(".")).with_suffix(".py").is_file()


def test_api_index_exports_fastapi_app() -> None:
    from slm_training.web.vercel import app

    assert isinstance(app, FastAPI)
    assert app.title == "TwoTower OpenUI Playground"
    routes = {getattr(route, "path", None) for route in app.routes}
    assert "/api/health" in routes
    assert "/api/generation-attempt" in routes
    assert "/api/generation-review" in routes
    assert "/api/server-attempt" in routes
    assert "/" in routes


def test_vercel_entrypoint_imports_without_torch() -> None:
    """Cold-start path must not require torch (Vercel entrypoint constraint)."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['torch'] = None; import slm_training.web.vercel",
        ],
        check=True,
    )
