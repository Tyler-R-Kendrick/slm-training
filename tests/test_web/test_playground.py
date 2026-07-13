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
    assert "root" in payload["openui"]
    # Grammar-constrained playground must never emit invalid OpenUI.
    for _ in range(2):
        again = client.post(
            "/api/sample",
            json={"auto_prompt": True, "grammar_constrained": True, "session_id": "test"},
        )
        assert again.status_code == 200
        assert again.json()["valid"] is True


@pytest.mark.skipif(not CKPT.exists(), reason="playground demo checkpoint missing")
def test_annotate_static_has_tab_toggle() -> None:
    app = create_app(checkpoint=CKPT, device="cpu")
    client = TestClient(app)
    js = client.get("/static/app.js")
    assert js.status_code == 200
    assert 'event.key === "Tab"' in js.text
    assert 'activeView === "render" ? "dsl" : "render"' in js.text
    # Tab must not steal focus from buttons / view tabs.
    assert "focus === cardEl" in js.text
