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

# String-property roles shared by synthesis validation and constrained decode.
# Content roles own request-local placeholders; structural roles own opaque
# identifier atoms such as ``$0``.  A string property outside both sets is not
# allowed to borrow either namespace.
TEMPLATIZABLE_PROPS = CONTENT_PROPS | {
    "codeString",
    "data",
    "details",
    "subtitle",
    "tags",
    "textMarkdown",
}
STRUCTURAL_ID_PROPS = frozenset({"category", "language", "name", "src", "value"})


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
