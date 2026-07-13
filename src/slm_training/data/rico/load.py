"""Load RICO semantic screens from fixtures or Hugging Face."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from slm_training.data.rico.labels import COMPONENT_LABELS, MAPPABLE_LABELS


def load_rico_jsonl(path: Path | str) -> list[dict]:
    path = Path(path)
    screens: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                screens.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return screens


def _explode_semantic_row(row: dict, *, split_src: str, screen_index: int) -> dict | None:
    elems: list[dict] = []
    for layer in row.get("children") or []:
        if not isinstance(layer, dict):
            continue
        labels = layer.get("component_label") or []
        n = len(labels)
        for j in range(n):
            raw = labels[j]
            name = COMPONENT_LABELS.get(int(raw), str(raw))
            if name not in MAPPABLE_LABELS:
                continue
            elems.append(
                {
                    "component_label": name,
                    "klass": (layer.get("klass") or [None] * n)[j],
                    "resource_id": (layer.get("resource_id") or [None] * n)[j],
                    "clickable": bool((layer.get("clickable") or [False] * n)[j]),
                    "bounds": (layer.get("bounds") or [None] * n)[j],
                    "icon_class": (layer.get("icon_class") or [None] * n)[j],
                }
            )
    if len(elems) < 2:
        return None
    return {
        "split_src": split_src,
        "screen_index": screen_index,
        "root_klass": row.get("klass"),
        "root_bounds": row.get("bounds"),
        "elements": elems[:50],
        "n_elements": len(elems),
    }


def iter_rico_huggingface(
    *,
    split: str = "train",
    limit: int = 200,
    config_name: str = "ui-screenshots-and-hierarchies-with-semantic-annotations",
) -> Iterator[dict]:
    """Stream semantic RICO screens from Hugging Face (requires `datasets`)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "RICO live download requires the `datasets` package. "
            "Install with: pip install datasets pillow"
        ) from exc

    ds = load_dataset("shunk031/Rico", name=config_name, split=split, streaming=True)
    produced = 0
    for i, row in enumerate(ds):
        screen = _explode_semantic_row(dict(row), split_src=split, screen_index=i)
        if screen is None:
            continue
        yield screen
        produced += 1
        if produced >= limit:
            break


def load_rico_screens(
    *,
    path: Path | str | None = None,
    hf_split: str | None = None,
    limit: int | None = None,
    hf_cache_path: Path | str | None = None,
) -> list[dict]:
    """Load screens from a local JSONL fixture and/or live HF split.

    When both a local path and ``hf_split`` are provided, ``limit`` is the
    total desired count: local screens are kept first, then HF fills the
    remainder (instead of truncating away the HF pull).
    """
    screens: list[dict] = []
    if path is not None:
        screens.extend(load_rico_jsonl(path))

    if hf_split is not None:
        remaining = None if limit is None else max(0, limit - len(screens))
        if remaining is None or remaining > 0:
            live_limit = 200 if remaining is None else remaining
            # Prefer a local HF cache when present (deterministic / offline).
            cache = Path(hf_cache_path) if hf_cache_path else None
            if cache is not None and cache.exists():
                cached = load_rico_jsonl(cache)
                need = live_limit
                screens.extend(cached[:need])
                live_limit = max(0, need - min(need, len(cached)))
            if live_limit > 0:
                live = list(iter_rico_huggingface(split=hf_split, limit=live_limit))
                screens.extend(live)
                if cache is not None:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    # Append-only merge into cache for reuse.
                    existing_idx = {
                        (s.get("split_src"), s.get("screen_index"))
                        for s in (load_rico_jsonl(cache) if cache.exists() else [])
                    }
                    with cache.open("a", encoding="utf-8") as handle:
                        for screen in live:
                            key = (screen.get("split_src"), screen.get("screen_index"))
                            if key in existing_idx:
                                continue
                            handle.write(json.dumps(screen, ensure_ascii=False) + "\n")
                            existing_idx.add(key)

    if limit is not None:
        screens = screens[:limit]
    return screens
