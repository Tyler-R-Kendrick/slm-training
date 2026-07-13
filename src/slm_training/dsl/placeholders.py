"""Placeholder helpers for placeholder-augmented OpenUI Lang."""

from __future__ import annotations

import re

PLACEHOLDER_RE = re.compile(r":[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")

# Content-bearing props in our @openuidev/lang-core library (library.mjs).
CONTENT_PROPS = frozenset({"title", "body", "content", "label"})


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
