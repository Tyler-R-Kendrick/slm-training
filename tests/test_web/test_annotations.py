"""Annotation store + playground annotate API tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from slm_training.harnesses.annotations import (
    AnnotationRecord,
    append_annotation,
    export_to_preference_pairs,
    export_to_train_seeds,
    load_annotations,
    maybe_append_preference_pair,
    new_annotation_id,
    upsert_human_train_seed,
    utc_now_iso,
)
from slm_training.dsl.schema import load_jsonl
from slm_training.models.paths import PLAYGROUND_DEMO_CHECKPOINT
from slm_training.web.app import create_app
from slm_training.web.prompts import compose_prompt

CKPT = PLAYGROUND_DEMO_CHECKPOINT
HERO = 'root = Stack([hero], "column")\nhero_title = TextContent(":hero.title")\nhero = Card([hero_title])\n'
BAD = 'root = TextContent(":broken.x")\n'


def test_vary_prompt_keeps_substance() -> None:
    from slm_training.web.prompts import PromptCursor

    import random

    out = compose_prompt(random.Random(3))
    assert isinstance(out, str)
    assert len(out) > 20
    assert "OpenUI" in out or "openui" in out.lower() or "layout" in out.lower() or "generate" in out.lower() or "Compose" in out or "Write" in out or "Produce" in out or "For " in out

    cursor = PromptCursor(session_id="unit-test-prompts")
    a = cursor.next()
    b = cursor.next()
    assert a != b
    # Must not recycle classic fixture wording verbatim.
    blocked = {
        "hero card with title and body",
        "primary call to action button",
        "create a vertical hero card with a title and body.",
    }
    assert a.strip().lower() not in blocked
    assert b.strip().lower() not in blocked


def test_annotation_append_and_promote(tmp_path: Path) -> None:
    feedback = tmp_path / "feedback.jsonl"
    human = tmp_path / "human_train.jsonl"
    pairs = tmp_path / "pairs.jsonl"

    up = AnnotationRecord(
        id=new_annotation_id(),
        ts=utc_now_iso(),
        prompt="Hero card with title and body",
        openui=HERO,
        rating="up",
        description="clean hierarchy",
        valid=True,
    )
    down = AnnotationRecord(
        id=new_annotation_id(),
        ts=utc_now_iso(),
        prompt="Hero card with title and body",
        openui=BAD,
        rating="down",
        description="too thin",
        valid=True,
    )
    append_annotation(feedback, up)
    append_annotation(feedback, down)
    assert len(load_annotations(feedback)) == 2

    assert upsert_human_train_seed(up, human) == human
    seeds = load_jsonl(human)
    assert len(seeds) == 1
    assert seeds[0].source == "human"
    assert seeds[0].meta["annotation_id"] == up.id

    pair = maybe_append_preference_pair(down, feedback_path=feedback, pairs_path=pairs)
    assert pair is not None
    assert pair.chosen.strip() == HERO.strip()
    assert pair.rejected.strip() == BAD.strip()

    train = export_to_train_seeds(feedback, human)
    pref = export_to_preference_pairs(feedback, pairs)
    assert train["count"] == 1
    assert pref["count"] >= 1


def test_annotate_api_persists(tmp_path: Path) -> None:
    feedback = tmp_path / "feedback.jsonl"
    human = tmp_path / "human.jsonl"
    pairs = tmp_path / "pairs.jsonl"
    app = create_app(
        checkpoint=CKPT if CKPT.exists() else Path("/nonexistent.pt"),
        device="cpu",
        annotations_path=feedback,
        human_train_path=human,
        human_pairs_path=pairs,
    )
    client = TestClient(app)

    page = client.get("/playground")
    assert page.status_code == 200
    assert "TwoTower" in page.text
    assert "btnUp" in page.text or "Thumbs up" in page.text or "grade" in page.text.lower()

    prompt = client.get("/api/prompt/next", params={"session_id": "unit"})
    assert prompt.status_code == 200
    assert prompt.json()["prompt"]

    up = client.post(
        "/api/annotate",
        json={
            "prompt": "Hero card with title and body",
            "openui": HERO,
            "rating": "up",
            "description": "nice",
            "session_id": "unit",
            "valid": True,
        },
    )
    assert up.status_code == 200
    assert up.json()["ok"] is True
    assert feedback.exists()
    rows = [json.loads(line) for line in feedback.read_text().splitlines() if line.strip()]
    assert rows[-1]["rating"] == "up"
    assert human.exists()

    down = client.post(
        "/api/annotate",
        json={
            "prompt": "Hero card with title and body",
            "openui": BAD,
            "rating": "down",
            "session_id": "unit",
            "valid": True,
        },
    )
    assert down.status_code == 200
    assert down.json()["preference_pair"] is not None

    recent = client.get("/api/annotations/recent")
    assert recent.status_code == 200
    assert recent.json()["count"] >= 2


def test_sample_api_generates() -> None:
    app = create_app(checkpoint=CKPT, device="cpu")
    client = TestClient(app)
    res = client.post(
        "/api/sample",
        json={"session_id": "gen", "auto_prompt": True, "grammar_constrained": True},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["prompt"]
    assert "openui" in payload
