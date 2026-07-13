"""Helpers for quality experiments: soft rejects, schema snippets, curriculum tags."""

from __future__ import annotations

import re
from typing import Iterable

from slm_training.dsl.schema import ExampleRecord
from slm_training.models.grammar import PREFERRED_COMPONENT_NAMES, STRUCTURAL_TOKENS


def compact_schema_snippet(*, budget: int = 600) -> str:
    """Offline schema prior (no Node) for context conditioning."""
    names = sorted(set(PREFERRED_COMPONENT_NAMES) | set(STRUCTURAL_TOKENS))
    # Keep common layout primitives first.
    priority = [
        "Stack",
        "Card",
        "CardHeader",
        "TextContent",
        "Button",
        "Buttons",
        "Input",
        "Form",
        "ImageBlock",
        "Separator",
    ]
    ordered = [n for n in priority if n in names] + [n for n in names if n not in priority]
    body = ", ".join(ordered[:40])
    text = f"components: {body}"
    return text[:budget]


_COMPONENT_CALL = re.compile(r"\b([A-Z][A-Za-z0-9]*)\s*\(")


def soft_corrupt_openui(openui: str) -> str:
    """
    Produce a grammar-likely-valid but worse layout (valid-but-worse reject).

    Prefer structural degradations over BrokenText syntax corruption.
    """
    text = openui
    # Rename a placeholder namespace first (hurts fidelity, keeps grammar).
    ph = re.search(r'":([A-Za-z_][A-Za-z0-9_]*)\.', text)
    if ph:
        renamed = text.replace(f'":{ph.group(1)}.', '":wrong.', 1)
        if renamed != text:
            return renamed
    # Downgrade a Button to TextContent (keeps placeholders).
    if "Button(" in text:
        return text.replace("Button(", "TextContent(", 1)
    # Drop one child from a Stack([a, b, ...]) list if possible.
    m = re.search(r"Stack\(\[([^\]]+)\]", text)
    if m:
        parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
        if len(parts) >= 2:
            new_list = ", ".join(parts[:-1])
            return text[: m.start(1)] + new_list + text[m.end(1) :]
    # Last resort: wrap in an extra Stack (still often valid).
    if text.strip().startswith("root ="):
        return text.replace(
            "root = ",
            'root = Stack([inner], "column", "m")\ninner = ',
            1,
        )
    return text


def tag_curriculum_stage(record: ExampleRecord) -> str:
    """
    A = shape-heavy / RICO screens
    B = binding-heavy fixtures & awwwards with rich placeholders
    C = hard / adversarial-style prompts (meta or id markers)
    """
    meta = dict(record.meta or {})
    if meta.get("curriculum") in {"A", "B", "C"}:
        return str(meta["curriculum"])
    rid = (record.id or "").lower()
    source = (record.source or "").lower()
    if "adversarial" in rid or meta.get("suite") == "adversarial" or "hard" in rid:
        return "C"
    if source.startswith("rico") or rid.startswith("rico"):
        return "A"
    return "B"


def apply_curriculum_tags(records: Iterable[ExampleRecord]) -> list[ExampleRecord]:
    out: list[ExampleRecord] = []
    for record in records:
        stage = tag_curriculum_stage(record)
        meta = {**dict(record.meta or {}), "curriculum": stage}
        out.append(
            ExampleRecord(
                id=record.id,
                prompt=record.prompt,
                openui=record.openui,
                placeholders=record.placeholders,
                split=record.split,
                source=record.source,
                meta=meta,
                design_md=record.design_md,
            )
        )
    return out


def curriculum_schedule(step: int, total_steps: int) -> str:
    """Early shape (A), mid binding (B), late hard (C)."""
    if total_steps <= 0:
        return "B"
    frac = step / max(1, total_steps)
    if frac < 0.35:
        return "A"
    if frac < 0.75:
        return "B"
    return "C"
