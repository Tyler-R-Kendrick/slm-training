"""Prompt synthesis plugins for the training-data harness."""

from __future__ import annotations

from typing import Protocol

from slm_training.dsl.schema import ExampleRecord


class PromptSynthesizer(Protocol):
    def expand(self, record: ExampleRecord) -> list[ExampleRecord]:
        """Return zero or more additional records derived from ``record``."""


class NoopSynthesizer:
    def expand(self, record: ExampleRecord) -> list[ExampleRecord]:
        return []


class TemplateSynthesizer:
    """Rule-based prompt paraphrases (offline, no LLM)."""

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
                    meta={**record.meta, "synth": "template", "parent_id": record.id},
                    design_md=record.design_md,
                )
            )
        return out


def get_synthesizer(name: str) -> PromptSynthesizer:
    if name in {"none", "noop", "off"}:
        return NoopSynthesizer()
    if name in {"template", "templates"}:
        return TemplateSynthesizer()
    raise ValueError(f"unknown synthesizer {name!r}")
