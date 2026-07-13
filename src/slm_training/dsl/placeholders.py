"""Placeholder helpers for the OpenUI subset."""

from __future__ import annotations

import re

PLACEHOLDER_RE = re.compile(r":[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")

# Props that must be placeholders (content-bearing).
CONTENT_PROPS = frozenset({"text", "label", "title", "body"})


def is_placeholder(value: str) -> bool:
    return bool(PLACEHOLDER_RE.fullmatch(value.strip()))


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
