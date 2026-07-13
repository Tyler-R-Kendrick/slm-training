"""Placeholder helpers for placeholder-augmented OpenUI Lang."""

from __future__ import annotations

import re
from typing import Iterable

PLACEHOLDER_RE = re.compile(r":[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")

# User-facing string props in official openuiLibrary that must be placeholders.
CONTENT_PROPS = frozenset(
    {
        "text",
        "label",
        "title",
        "body",
        "content",
        "placeholder",
        "alt",
        "hint",
        "description",
        "trigger",
    }
)


def is_placeholder(value: str) -> bool:
    return bool(PLACEHOLDER_RE.fullmatch(value.strip().strip('"')))


def extract_placeholders(source: str) -> list[str]:
    """Return unique placeholders in source order of first appearance."""
    seen: set[str] = set()
    ordered: list[str] = []
    for match in PLACEHOLDER_RE.finditer(source):
        token = match.group(0)
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def merge_placeholders(*groups: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for item in group:
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out
