"""RICO conversion unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.data.rico import screen_to_record
from slm_training.data.rico.convert import RicoElement, screen_to_openui
from slm_training.dsl import bridge_available
from slm_training.dsl.parser import validate


def test_screen_to_openui_builds_stack() -> None:
    elements = [
        RicoElement("Toolbar", resource_id="app:id/toolbar", bounds=[0, 0, 100, 40]),
        RicoElement("Card", resource_id="app:id/hero", bounds=[0, 50, 100, 200]),
        RicoElement(
            "Text Button",
            resource_id="app:id/submit",
            clickable=True,
            bounds=[0, 220, 100, 260],
        ),
    ]
    openui, placeholders, meta = screen_to_openui(elements, namespace="train0")
    assert openui.startswith("root = Stack(")
    assert "Card(" in openui
    assert "Button(" in openui
    assert "TextContent(" in openui
    assert meta["n_children"] == 3
    assert placeholders
    assert all(p.startswith(":") for p in placeholders)


@pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd tools/openui_bridge && npm ci",
)
def test_rico_fixture_screens_validate() -> None:
    path = Path("fixtures/rico/semantic_train.jsonl")
    if not path.exists():
        pytest.skip("RICO fixtures missing")
    ok = 0
    for line in path.read_text(encoding="utf-8").splitlines()[:15]:
        screen = json.loads(line)
        record = screen_to_record(screen, split="train")
        validate(record.openui)
        assert record.source == "rico"
        assert record.placeholders
        ok += 1
    assert ok >= 10
