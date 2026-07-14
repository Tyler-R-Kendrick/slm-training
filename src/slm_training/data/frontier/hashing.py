"""Hashes binding frozen teacher artifacts to exact train golds."""

from __future__ import annotations

import hashlib

from slm_training.data.structure import strip_style_literals


def _normalized_gold(openui: str, prompt: str) -> bytes:
    skeleton = strip_style_literals(openui or "").strip()
    return skeleton.encode("utf-8") + b"\0" + (prompt or "").strip().encode("utf-8")


def gold_content_hash(openui: str, prompt: str) -> str:
    """Return the 16-hex identity of style-free gold bytes plus prompt."""
    return hashlib.sha256(_normalized_gold(openui, prompt)).hexdigest()[:16]


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256((prompt or "").strip().encode("utf-8")).hexdigest()
