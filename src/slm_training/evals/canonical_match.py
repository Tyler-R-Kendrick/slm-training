"""Canonical exact-match eval over (prediction, gold) OpenUI pairs (D2).

Surface exact-match under-counts correct layouts that differ only in binder
names, statement order, or style literals. Canonical exact-match compares the
D2 canonical form (`dsl.canonicalize`) instead, so alpha-equivalent correct
programs score as matches. Reported *beside* surface exact-match, never as a
replacement — a diagnostic, not a ship gate.
"""

from __future__ import annotations

from typing import Any, Iterable

from slm_training.dsl.canonicalize import canonical_equal, canonicalize


def canonical_exact_match_rate(
    pairs: Iterable[tuple[str, str]], *, dsl: str | None = None
) -> dict[str, Any]:
    """Canonical vs surface exact-match rate over ``(pred, gold)`` pairs."""
    total = 0
    canonical_hits = 0
    surface_hits = 0
    rescued = 0  # canonical match that surface exact-match missed
    for pred, gold in pairs:
        total += 1
        surface = str(pred).strip() == str(gold).strip()
        canonical = canonical_equal(pred, gold, dsl=dsl)
        surface_hits += int(surface)
        canonical_hits += int(canonical)
        if canonical and not surface:
            rescued += 1
    return {
        "n": total,
        "surface_exact_match": surface_hits / total if total else None,
        "canonical_exact_match": canonical_hits / total if total else None,
        # Correct layouts the surface metric under-counted (alpha-renaming,
        # reordering, style) — the value canonicalization adds.
        "canonicalization_rescued": rescued,
        "canonicalization_rescued_rate": rescued / total if total else None,
    }


def to_canonical(source: str, *, dsl: str | None = None) -> str | None:
    """Canonical form of ``source``, or ``None`` if it does not parse."""
    try:
        return canonicalize(source, dsl=dsl)
    except Exception:  # noqa: BLE001
        return None
