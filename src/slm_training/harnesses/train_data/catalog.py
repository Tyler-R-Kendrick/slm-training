"""Source-family catalog: lineage, exposure accounting, and parent caps.

The corpus is a mixture of source *families* (real data, human feedback, and
named deterministic transformations). Mixture search and exposure caps need:

* a normalized ``source_family`` per record (free-form ``source`` strings such
  as ``"rico+template"`` are not a taxonomy),
* ``root_parent_id`` / ``transformation_lineage`` so three paraphrases of one
  parent are never counted as three independent examples,
* per-family unique/exposure statistics in the manifest.
"""

from __future__ import annotations

import math
from typing import Any

from slm_training.dsl.schema import ExampleRecord

# Known family names (self_* families arrive with the distillation stage).
KNOWN_FAMILIES = (
    "human_curated",
    "human_feedback",
    "rico_real",
    "awwwards_real",
    "prompt_paraphrase",
    "layout_augment",
    "namespace_augment",
    "stress_adversarial",
    "self_distilled_success",
    "self_distilled_repair",
    "gold_correction",
    # OpenUI-SLM program-first families (ProgramSpec derivatives).
    "programspec_generated",
    "language_contract",
    "schema_primer",
    "repair_taxonomy",
    "hard_negative_repair",
    "edit_patch",
    "edit_trajectory",
    "state_behavior",
    "render_grounded",
    "visual_edit",
    "web_projection",
    "frontier_semantic",
    "frontier_product",
    "frontier_user",
    "frontier_simplified",
    "design_md_contrastive",
    "adversarial_injection",
    "vision_bridged",
)

_SYNTH_TO_FAMILY = {
    "template": "prompt_paraphrase",
    "layout_augment": "layout_augment",
    "namespace_augment": "namespace_augment",
    # OpenUI-SLM per-gold synthesizers (meta["synth"] label -> family).
    "repair": "repair_taxonomy",
    "hard_negative": "hard_negative_repair",
    "edit": "edit_patch",
    "edit_trajectory": "edit_trajectory",
    "design_md_contrastive": "design_md_contrastive",
    "frontier_semantic": "frontier_semantic",
    "frontier_product": "frontier_product",
    "frontier_user": "frontier_user",
    "frontier_simplified": "frontier_simplified",
    "schema_primer": "schema_primer",
}

_BASE_SOURCE_TO_FAMILY = {
    "fixture": "human_curated",
    "fixtures": "human_curated",
    "human": "human_feedback",
    "rico": "rico_real",
    "awwwards": "awwwards_real",
    # OpenUI-SLM source loaders (source base -> family).
    "programspec": "programspec_generated",
    "language_contract": "language_contract",
    "schema_primer": "schema_primer",
    "deconstruct": "web_projection",
    "render": "render_grounded",
    "vision": "vision_bridged",
}


def classify_source_family(record: ExampleRecord) -> str:
    """Normalize a record's provenance to a mixture family name.

    The *outermost* transformation wins: a namespace augment of a template
    paraphrase belongs to ``namespace_augment`` (its lineage records the rest).
    """
    meta = dict(record.meta or {})
    synth = str(meta.get("synth") or "")
    if synth in _SYNTH_TO_FAMILY:
        return _SYNTH_TO_FAMILY[synth]
    source = (record.source or "").lower()
    if synth == "stress" or "stress" in source:
        return "stress_adversarial"
    base = source.split("+", 1)[0].strip()
    if base in _BASE_SOURCE_TO_FAMILY:
        return _BASE_SOURCE_TO_FAMILY[base]
    if base in KNOWN_FAMILIES:
        return base
    return base or "unknown"


# id -> (parent_id | None, synth label | None); built over *all* candidates so
# lineage survives even when an intermediate variant was filtered out later.
LineageIndex = dict[str, tuple[str | None, str | None]]


def lineage_entry(record: ExampleRecord) -> tuple[str | None, str | None]:
    meta = dict(record.meta or {})
    parent = meta.get("parent_id")
    synth = meta.get("synth")
    return (str(parent) if parent else None, str(synth) if synth else None)


def resolve_lineage(
    record_id: str, index: LineageIndex, *, max_depth: int = 16
) -> tuple[str, list[str]]:
    """Walk parent pointers to the root; return (root_parent_id, lineage).

    ``lineage`` lists transformation labels innermost-first (root side first).
    """
    lineage: list[str] = []
    current = record_id
    for _ in range(max_depth):
        parent, synth = index.get(current, (None, None))
        if synth:
            lineage.append(synth)
        if not parent or parent == current:
            return current, list(reversed(lineage))
        current = parent
    return current, list(reversed(lineage))


def annotate_lineage(
    records: list[ExampleRecord], index: LineageIndex
) -> list[ExampleRecord]:
    """Set meta.source_family / root_parent_id / transformation_lineage in place."""
    for record in records:
        meta = dict(record.meta or {})
        root, lineage = resolve_lineage(record.id, index)
        meta["source_family"] = classify_source_family(record)
        meta["root_parent_id"] = root
        meta["transformation_lineage"] = lineage
        record.meta = meta
    return records


def _percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    idx = max(0, min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[idx]


def _exposure_stats(counts: list[int]) -> dict[str, int]:
    ordered = sorted(counts)
    return {
        "max": ordered[-1] if ordered else 0,
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
    }


def family_stats(records: list[ExampleRecord]) -> dict[str, Any]:
    """Per-family unique / token / parent-exposure accounting for the manifest."""
    from slm_training.models.tokenizer import tokenize_text

    by_family: dict[str, list[ExampleRecord]] = {}
    for record in records:
        family = str((record.meta or {}).get("source_family") or "unknown")
        by_family.setdefault(family, []).append(record)

    families: dict[str, Any] = {}
    all_parent_counts: dict[str, int] = {}
    for family, members in sorted(by_family.items()):
        parents: dict[str, int] = {}
        prompt_chars = 0
        target_chars = 0
        target_tokens = 0
        for record in members:
            root = str((record.meta or {}).get("root_parent_id") or record.id)
            parents[root] = parents.get(root, 0) + 1
            all_parent_counts[root] = all_parent_counts.get(root, 0) + 1
            prompt_chars += len(record.prompt)
            target_chars += len(record.openui)
            target_tokens += len(tokenize_text(record.openui))
        families[family] = {
            "unique_records": len(members),
            "unique_root_parents": len(parents),
            "prompt_chars": prompt_chars,
            "target_chars": target_chars,
            "target_tokens": target_tokens,
            "records_per_root_parent": _exposure_stats(list(parents.values())),
        }
    return {
        "families": families,
        "total_records": len(records),
        "unique_root_parents": len(all_parent_counts),
        "records_per_root_parent": _exposure_stats(list(all_parent_counts.values())),
    }


def apply_parent_cap(
    records: list[ExampleRecord], max_records_per_parent: int | None
) -> tuple[list[ExampleRecord], list[dict[str, str]]]:
    """Deterministically cap records per root parent (exposure control).

    The root record itself (id == root_parent_id) is always preferred; the
    remaining slots go to variants in sorted-id order.
    """
    if not max_records_per_parent or max_records_per_parent <= 0:
        return list(records), []
    groups: dict[str, list[ExampleRecord]] = {}
    for record in sorted(records, key=lambda r: r.id):
        root = str((record.meta or {}).get("root_parent_id") or record.id)
        groups.setdefault(root, []).append(record)

    kept_ids: set[str] = set()
    dropped: list[dict[str, str]] = []
    for root, members in groups.items():
        members = sorted(members, key=lambda r: (r.id != root, r.id))
        for record in members[:max_records_per_parent]:
            kept_ids.add(record.id)
        for record in members[max_records_per_parent:]:
            dropped.append(
                {
                    "id": record.id,
                    "root_parent_id": root,
                    "source_family": str(
                        (record.meta or {}).get("source_family") or "unknown"
                    ),
                    "reason": "max_records_per_parent",
                }
            )
    kept = [r for r in records if r.id in kept_ids]
    return kept, dropped
