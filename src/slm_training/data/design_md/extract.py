"""Heuristic DESIGN.md extraction (offline-safe; VLM optional)."""

from __future__ import annotations

from typing import Any

from slm_training.design_md import bridge_available, lint, load_default_design_md


def extract_design_md(
    *,
    title: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    html: str | None = None,
    screenshot_path: str | None = None,
    variant: str = "strict",
    provider: str | None = None,
) -> str:
    """
    Produce a DESIGN.md for a UI sample.

    Offline default: specialize the fixture DESIGN.md with metadata.
    ``provider`` is reserved for a future VLM hook (never called here).
    """
    _ = (html, screenshot_path, provider)
    base = load_default_design_md()
    bits: list[str] = []
    if title:
        bits.append(f"Source title: {title}")
    if description:
        bits.append(f"Source description: {description[:400]}")
    if tags:
        bits.append("Tags: " + ", ".join(tags[:12]))
    bits.append(f"Extraction variant: {variant}")
    injection = "\n".join(f"- {b}" for b in bits)
    marker = "## Overview\n"
    if marker in base:
        head, tail = base.split(marker, 1)
        return f"{head}{marker}\nSource notes:\n{injection}\n\n{tail.lstrip()}"
    return base + "\n\n## Source notes\n" + injection + "\n"


def extract_and_filter(**kwargs: Any) -> tuple[str | None, dict[str, Any]]:
    """Extract DESIGN.md and keep the highest-scoring lint-passing variant."""
    variants = list(kwargs.pop("variants", None) or ["strict", "creative", "a11y"])
    if not bridge_available():
        text = extract_design_md(variant=variants[0], **kwargs)
        return text, {"score": 1.0, "summary": {"errors": 0}, "offline": True}

    best: tuple[float, str, dict[str, Any]] | None = None
    for variant in variants:
        text = extract_design_md(variant=variant, **kwargs)
        report = lint(text)
        score = float(report.get("score") or 0.0)
        if report.get("ok") and (best is None or score > best[0]):
            best = (score, text, report)
    if best is None:
        return None, {"ok": False, "score": 0.0}
    return best[1], best[2]
