"""Prompt / layout synthesis plugins for the training-data harness."""

from __future__ import annotations

import re
from typing import Protocol

from slm_training.dsl.schema import ExampleRecord

_ROOT_STACK_RE = re.compile(
    r'^root\s*=\s*Stack\(\[(?P<children>[^\]]*)\](?P<rest>(?:,\s*"[^"]*")*)\)\s*$',
    re.M,
)


class PromptSynthesizer(Protocol):
    def expand(self, record: ExampleRecord) -> list[ExampleRecord]:
        """Return zero or more additional records derived from ``record``."""


class NoopSynthesizer:
    def expand(self, record: ExampleRecord) -> list[ExampleRecord]:
        return []


class TemplateSynthesizer:
    """Deterministic prompt paraphrases (same OpenUI target)."""

    TEMPLATES = (
        "Please generate UI for: {prompt}",
        "OpenUI layout request: {prompt}",
        "Design a screen that does the following — {prompt}",
    )

    def expand(self, record: ExampleRecord) -> list[ExampleRecord]:
        out: list[ExampleRecord] = []
        for i, template in enumerate(self.TEMPLATES):
            out.append(
                ExampleRecord(
                    id=f"{record.id}_syn_{i}",
                    prompt=template.format(prompt=record.prompt),
                    openui=record.openui,
                    placeholders=list(record.placeholders),
                    split=record.split,
                    source=f"{record.source}+template",
                    meta={
                        **record.meta,
                        "synth": "template",
                        "parent_id": record.id,
                        "synth_index": i,
                    },
                    design_md=record.design_md,
                )
            )
        return out


class LayoutAugmentSynthesizer:
    """
    Deterministic structural augmentations.

    Produces at most two variants:
    1. Flip Stack direction column<->row when a single root Stack exists
    2. Append a secondary CTA button sibling when the root has fewer than 4 children
    """

    def expand(self, record: ExampleRecord) -> list[ExampleRecord]:
        out: list[ExampleRecord] = []
        flipped = self._flip_direction(record)
        if flipped:
            out.append(flipped)
        with_cta = self._append_cta(record)
        if with_cta:
            out.append(with_cta)
        return out

    def _flip_direction(self, record: ExampleRecord) -> ExampleRecord | None:
        match = _ROOT_STACK_RE.search(record.openui)
        if not match:
            return None
        rest = match.group("rest") or ""
        if '"column"' in rest:
            new_rest = rest.replace('"column"', '"row"', 1)
            direction = "row"
        elif '"row"' in rest:
            new_rest = rest.replace('"row"', '"column"', 1)
            direction = "column"
        elif rest.strip() == "":
            # Bare Stack([...]) — treat as column default, make explicit row.
            new_rest = ', "row", "m"'
            direction = "row"
        else:
            return None
        children = match.group("children")
        new_root = f"root = Stack([{children}]{new_rest})"
        openui = _ROOT_STACK_RE.sub(new_root, record.openui, count=1)
        if openui == record.openui:
            return None
        return ExampleRecord(
            id=f"{record.id}_aug_dir",
            prompt=f"{record.prompt} Prefer a {direction} Stack layout.",
            openui=openui,
            placeholders=list(record.placeholders),
            split=record.split,
            source=f"{record.source}+aug",
            meta={
                **record.meta,
                "synth": "layout_augment",
                "aug": "flip_direction",
                "parent_id": record.id,
            },
            design_md=record.design_md,
        )

    def _append_cta(self, record: ExampleRecord) -> ExampleRecord | None:
        match = _ROOT_STACK_RE.search(record.openui)
        if not match:
            return None
        children = [c.strip() for c in match.group("children").split(",") if c.strip()]
        if len(children) >= 4 or "cta_aug" in children:
            return None
        if any("Button(" in line for line in record.openui.splitlines()):
            # Already has a button — skip to avoid redundant CTAs.
            return None
        rest = match.group("rest") or ', "column", "m"'
        new_children = ", ".join([*children, "cta_aug"])
        new_root = f"root = Stack([{new_children}]{rest})"
        openui = _ROOT_STACK_RE.sub(new_root, record.openui, count=1)
        openui = openui.rstrip() + '\ncta_aug = Button(":cta_aug.label")'
        placeholders = list(record.placeholders)
        if ":cta_aug.label" not in placeholders:
            placeholders.append(":cta_aug.label")
        return ExampleRecord(
            id=f"{record.id}_aug_cta",
            prompt=f"{record.prompt} Include a clear call-to-action button.",
            openui=openui,
            placeholders=placeholders,
            split=record.split,
            source=f"{record.source}+aug",
            meta={
                **record.meta,
                "synth": "layout_augment",
                "aug": "append_cta",
                "parent_id": record.id,
            },
            design_md=record.design_md,
        )


class QualitySynthesizer:
    """Compose template paraphrases + layout augments (deterministic, ordered)."""

    def __init__(self) -> None:
        self._template = TemplateSynthesizer()
        self._layout = LayoutAugmentSynthesizer()

    def expand(self, record: ExampleRecord) -> list[ExampleRecord]:
        # Layout first (new OpenUI), then light prompt paraphrases of the *original* only.
        return [*self._layout.expand(record), *self._template.expand(record)]


def get_synthesizer(name: str) -> PromptSynthesizer:
    if name in {"none", "noop", "off"}:
        return NoopSynthesizer()
    if name in {"template", "templates"}:
        return TemplateSynthesizer()
    if name in {"layout", "layout_augment", "aug"}:
        return LayoutAugmentSynthesizer()
    if name in {"quality", "full", "default"}:
        return QualitySynthesizer()
    raise ValueError(f"unknown synthesizer {name!r}")
