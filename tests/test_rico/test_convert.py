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


def test_screen_to_openui_reserves_root_binder() -> None:
    elements = [
        RicoElement(
            "List Item",
            resource_id="app:id/root",
            bounds=[0, 0, 100, 40],
        ),
        RicoElement(
            "List Item",
            resource_id="app:id/root",
            bounds=[0, 50, 100, 90],
        ),
    ]

    openui, _, meta = screen_to_openui(elements, namespace="test")

    assert openui.startswith("root = Stack([root_1, root_2]")
    assert openui.count("\nroot = ") == 0
    assert meta["n_children"] == 2
    validate(openui)


def test_screen_prompt_keeps_provenance_in_metadata_only() -> None:
    record = screen_to_record(
        {
            "split_src": "train",
            "screen_index": 42,
            "elements": [
                {
                    "component_label": "Text Button",
                    "resource_id": "app:id/submit",
                    "clickable": True,
                    "bounds": [0, 0, 100, 40],
                }
            ],
        }
    )

    assert "RICO" not in record.prompt
    assert "screen 42" not in record.prompt
    assert record.meta["rico_split"] == "train"
    assert record.meta["rico_screen_index"] == 42


@pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)
def test_rico_fixture_screens_validate() -> None:
    path = Path("src/slm_training/resources/rico/semantic_train.jsonl")
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
