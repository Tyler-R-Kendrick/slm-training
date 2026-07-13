"""Playground API smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from slm_training.web.app import create_app

CKPT = Path("outputs/runs/playground_demo/checkpoints/last.pt")


@pytest.mark.skipif(not CKPT.exists(), reason="playground demo checkpoint missing")
def test_playground_health_and_generate() -> None:
    app = create_app(checkpoint=CKPT, device="cpu")
    client = TestClient(app)
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["exists"] is True

    page = client.get("/")
    assert page.status_code == 200
    assert "TwoTower" in page.text

    res = client.post(
        "/api/generate",
        json={"prompt": "Hero card with title and body", "grammar_constrained": True},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["valid"] is True
    assert "Stack" in payload["openui"]
    assert ":" in payload["openui"]  # placeholder-augmented
