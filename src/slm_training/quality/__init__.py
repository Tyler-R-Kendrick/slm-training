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


def curriculum_schedule(step: int, total_steps: int) -> str:
    """
    Primary stage for a step (used when mix_curriculum=False).

    Early shape (A), mid binding (B), late hard (C).
    """
    if total_steps <= 0:
        return "B"
    frac = step / max(1, total_steps)
    if frac < 0.35:
        return "A"
    if frac < 0.75:
        return "B"
    return "C"


def curriculum_mix_weights(step: int, total_steps: int) -> dict[str, float]:
    """
    Soft stage mixture — always keep ≥30% binding (B) to prevent C leakage.

    Late training emphasizes C but never goes C-only (E9 smoke `:adv.*` failure).
    """
    if total_steps <= 0:
        return {"A": 0.2, "B": 0.6, "C": 0.2}
    frac = step / max(1, total_steps)
    if frac < 0.35:
        return {"A": 0.50, "B": 0.40, "C": 0.10}
    if frac < 0.75:
        return {"A": 0.25, "B": 0.55, "C": 0.20}
    # Late: still ≥30% B.
    return {"A": 0.15, "B": 0.35, "C": 0.50}


_ADV_PLACEHOLDER = re.compile(r'":adv\.')


def strip_adv_placeholders(openui: str) -> str:
    """Remap adversarial placeholder namespaces so they cannot leak into smoke."""
    return _ADV_PLACEHOLDER.sub('":item.', openui)


def sanitize_curriculum_record(record: ExampleRecord, *, stage: str | None = None) -> ExampleRecord:
    """Tag stage and strip `:adv.*` placeholders from non-C records."""
    stage = stage or tag_curriculum_stage(record)
    openui = record.openui
    placeholders = list(record.placeholders or [])
    if stage != "C":
        openui = strip_adv_placeholders(openui)
        placeholders = [
            p.replace(":adv.", ":item.") if p.startswith(":adv.") else p
            for p in placeholders
        ]
    meta = {**dict(record.meta or {}), "curriculum": stage}
    return ExampleRecord(
        id=record.id,
        prompt=record.prompt,
        openui=openui,
        placeholders=placeholders,
        split=record.split,
        source=record.source,
        meta=meta,
        design_md=record.design_md,
    )


def apply_curriculum_tags(
    records: Iterable[ExampleRecord],
    *,
    sanitize: bool = True,
) -> list[ExampleRecord]:
    out: list[ExampleRecord] = []
    for record in records:
        stage = tag_curriculum_stage(record)
        if sanitize:
            out.append(sanitize_curriculum_record(record, stage=stage))
        else:
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


def sample_curriculum_batch(
    records: list[ExampleRecord],
    *,
    batch_size: int,
    step: int,
    total_steps: int,
    rng,
    mix: bool = True,
) -> list[ExampleRecord]:
    """Draw a batch using soft mix weights (or hard stage when mix=False)."""
    if not records:
        return []
    by_stage: dict[str, list[ExampleRecord]] = {"A": [], "B": [], "C": []}
    for r in records:
        stage = str((r.meta or {}).get("curriculum") or "B")
        by_stage.setdefault(stage, []).append(r)
    if not mix:
        primary = curriculum_schedule(step, total_steps)
        pool = by_stage.get(primary) or records
        shuffled = list(pool)
        rng.shuffle(shuffled)
        return shuffled[:batch_size]

    weights = curriculum_mix_weights(step, total_steps)
    # Fallback empty stages into B then any.
    def _pool(stage: str) -> list[ExampleRecord]:
        return by_stage.get(stage) or by_stage.get("B") or records

    out: list[ExampleRecord] = []
    for _ in range(batch_size):
        roll = rng.random()
        cum = 0.0
        chosen = "B"
        for stage, w in weights.items():
            cum += w
            if roll <= cum:
                chosen = stage
                break
        pool = _pool(chosen)
        out.append(pool[rng.randrange(len(pool))])
    return out
