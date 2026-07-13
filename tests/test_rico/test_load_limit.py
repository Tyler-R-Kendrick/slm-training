"""RICO loader limit / HF cache behavior."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.data.rico.load import load_rico_screens


def test_load_rico_screens_fills_limit_from_cache(tmp_path: Path) -> None:
    local = tmp_path / "local.jsonl"
    cache = tmp_path / "cache.jsonl"
    local_rows = [
        {
            "split_src": "test",
            "screen_index": i,
            "elements": [
                {"component_label": "Text", "bounds": [0, 0, 10, 10]},
                {"component_label": "Text Button", "bounds": [0, 10, 10, 20]},
            ],
            "n_elements": 2,
        }
        for i in range(3)
    ]
    cache_rows = [
        {
            "split_src": "test",
            "screen_index": 100 + i,
            "elements": [
                {"component_label": "Text", "bounds": [0, 0, 10, 10]},
                {"component_label": "Card", "bounds": [0, 10, 10, 20]},
            ],
            "n_elements": 2,
        }
        for i in range(10)
    ]
    local.write_text("\n".join(json.dumps(r) for r in local_rows) + "\n")
    cache.write_text("\n".join(json.dumps(r) for r in cache_rows) + "\n")

    # limit=8 => keep 3 local + 5 from cache (no live HF needed)
    screens = load_rico_screens(
        path=local,
        hf_split="test",
        limit=8,
        hf_cache_path=cache,
    )
    assert len(screens) == 8
    assert screens[0]["screen_index"] == 0
    assert screens[3]["screen_index"] == 100
