"""Vercel FastAPI entrypoint for the OpenUI playground.

Vercel discovers ``app`` under ``api/index.py`` (or ``tool.vercel.entrypoint``).
"""

from __future__ import annotations

from slm_training.web.app import create_app

app = create_app()
