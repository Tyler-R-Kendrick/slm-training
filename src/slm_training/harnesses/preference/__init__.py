"""Composite reward + preference pair builder for DPO-style training."""

from __future__ import annotations

import os
import tempfile
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    design_md: str | None = None
    chosen_score: float = 0.0
    rejected_score: float = 0.0
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "design_md": self.design_md,
            "chosen_score": self.chosen_score,
            "rejected_score": self.rejected_score,
            "meta": self.meta or {},
        }


def layout_metrics(openui: str) -> float:
    """Cheap structural aesthetic proxy in [0, 1]."""
    lines = [ln for ln in openui.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    has_root = any(ln.startswith("root =") for ln in lines)
    n_components = sum(1 for ln in lines if " = " in ln and "(" in ln)
    depth_proxy = openui.count("[") + openui.count("(")
    # Prefer a root + a few components without extreme nesting.
    score = 0.2 * float(has_root)
    score += min(0.5, 0.08 * n_components)
    score += 0.3 if 2 <= depth_proxy <= 24 else 0.1
    return min(1.0, score)


def grammar_score(openui: str) -> float:
    try:
        from slm_training.dsl.parser import validate

        program = validate(openui)
        serialized = (program.serialized or openui).strip()
        compact = serialized.replace(" ", "")
        if "Stack([])" in compact or "Stack([]," in compact:
            return 0.0
        # Require at least one non-Stack component + a placeholder.
        from slm_training.dsl.placeholders import extract_placeholders
        import re

        comps = re.findall(r"\b([A-Z][A-Za-z0-9]*)\s*\(", serialized)
        if not any(c != "Stack" for c in comps):
            return 0.0
        if not extract_placeholders(serialized):
            return 0.0
        return 1.0
    except Exception:  # noqa: BLE001
        return 0.0


def placeholder_score(openui: str, gold: ExampleRecord | None = None) -> float:
    preds = set(extract_placeholders(openui))
    if gold is None:
        return 1.0 if preds else 0.5
    gold_set = set(gold.placeholders or extract_placeholders(gold.openui))
    if not gold_set:
        return 1.0 if not preds else 0.5
    return len(preds & gold_set) / len(gold_set)


def design_lint_score(design_md: str | None) -> float:
    """
    Score a DESIGN.md document.

    Warnings (e.g. unused color tokens) must not tank the score — only errors
    drive hard failures. Used for preference training when design_md is passed
    explicitly; eval never calls this with gold DESIGN.md.
    """
    if not design_md:
        return 0.5
    try:
        from slm_training.dsl.design_md import bridge_available, lint

        if not bridge_available():
            return 0.8
        report = lint(design_md)
        summary = report.get("summary") or {}
        errors = int(summary.get("errors") or 0)
        if errors:
            return float(report.get("score") or 0.0)
        # Warnings/infos only → treat as clean enough for structure training.
        return max(float(report.get("score") or 0.0), 0.9)
    except Exception:  # noqa: BLE001
        return 0.5


def composite_reward(
    openui: str,
    *,
    gold: ExampleRecord | None = None,
    design_md: str | None = None,
) -> float:
    """
    Ordered composite on *structure* (grammar + placeholders + layout).

    When ``design_md`` is ``None`` (eval / ship reward_score), DESIGN.md style
    lint is excluded entirely — colors/typography cannot move the score.

    Preference training may pass an explicit ``design_md`` to optionally blend
    a small lint term; warnings-only docs still score high.
    """
    from slm_training.data.structure import strip_style_literals

    openui = strip_style_literals(openui)
    g = grammar_score(openui)
    if g <= 0.0:
        return 0.0
    ph = placeholder_score(openui, gold)
    layout = layout_metrics(openui)
    if design_md is None:
        # Structure-only (eval path).
        score = 0.55 * g + 0.30 * ph + 0.15 * layout
    else:
        lint = design_lint_score(design_md)
        score = 0.45 * g + 0.25 * ph + 0.20 * lint + 0.10 * layout
    return round(float(score), 4)


def build_pairs_from_candidates(
    prompt: str,
    candidates: list[str],
    *,
    gold: ExampleRecord | None = None,
    design_md: str | None = None,
    prefer_valid_rejects: bool = True,
) -> PreferencePair | None:
    scored = [
        (composite_reward(c, gold=gold, design_md=design_md), c) for c in candidates
    ]
    if prefer_valid_rejects:
        valid = [(s, c) for s, c in scored if grammar_score(c) > 0.0]
        if len(valid) >= 2:
            scored = valid
    scored.sort(key=lambda x: x[0], reverse=True)
    if len(scored) < 2:
        return None
    if scored[0][0] <= scored[-1][0]:
        return None
    return PreferencePair(
        prompt=prompt,
        chosen=scored[0][1],
        rejected=scored[-1][1],
        design_md=design_md
        if design_md is not None
        else (gold.design_md if gold else None),
        chosen_score=scored[0][0],
        rejected_score=scored[-1][0],
    )


def write_pairs(path: Path | str, pairs: list[PreferencePair]) -> int:
    """Atomically replace a preference-pair JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            for pair in pairs:
                handle.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return len(pairs)


def load_pairs(path: Path | str) -> list[PreferencePair]:
    path = Path(path)
    out: list[PreferencePair] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        out.append(
            PreferencePair(
                prompt=data["prompt"],
                chosen=data["chosen"],
                rejected=data["rejected"],
                design_md=data.get("design_md"),
                chosen_score=float(data.get("chosen_score") or 0),
                rejected_score=float(data.get("rejected_score") or 0),
                meta=data.get("meta") or {},
            )
        )
    return out


def collect_pairs_with_generator(
    records: list[ExampleRecord],
    generate_fn: Callable[[ExampleRecord], list[str]],
    *,
    prefer_valid_rejects: bool = True,
    structure_only: bool = True,
    include_gold: bool = True,
    generator_checkpoint: str | None = None,
) -> list[PreferencePair]:
    """Build ranked preference pairs from generated candidates.

    Corpus separation (keep these distinct — conflating them makes it
    impossible to tell whether a climb learned from its own exploration or
    just received more supervision):

    * ``include_gold=True`` (default) — the gold target is injected as a
      candidate when missing. Pairs are tagged ``pair_corpus=gold_correction``.
    * ``include_gold=False`` — pairs come only from policy candidates; any
      exact-gold candidate the generator produced by copying is also dropped.
      Pairs are tagged ``pair_corpus=self_distilled``.
    """
    pairs: list[PreferencePair] = []
    for record in records:
        cands = generate_fn(record)
        gold_injected = False
        if include_gold:
            if record.openui not in cands:
                cands = [record.openui, *cands]
                gold_injected = True
        else:
            cands = [c for c in cands if c != record.openui]
        design = None if structure_only else record.design_md
        pair = build_pairs_from_candidates(
            record.prompt,
            cands,
            gold=record,
            design_md=design,
            prefer_valid_rejects=prefer_valid_rejects,
        )
        if pair:
            gold_used = record.openui in {pair.chosen, pair.rejected}
            pair.meta = {
                **(pair.meta or {}),
                "record_id": record.id,
                "pair_corpus": "gold_correction" if include_gold else "self_distilled",
                "gold_injected": gold_injected,
                "gold_used": gold_used,
                "generator_checkpoint": generator_checkpoint,
            }
            pairs.append(pair)
    return pairs
