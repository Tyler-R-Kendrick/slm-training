"""D1 simplification-consistent forward corruption.

Tree simplification belongs in the FORWARD diffusion process, not bolted into
the reverse loop. If training targets are pre-canonicalized (D2
`dsl.canonicalize`), then every noised intermediate the denoiser sees is a noised
version of the *canonical* tree — the reverse model learns to reconstruct one
canonical form per layout equivalence class. Simplifying in the reverse loop
instead would shift mid-trajectory states off-distribution: the model's
predictions were trained for the unsimplified tree.

This module is the data-side transform: canonicalize `ExampleRecord.openui`
(and placeholder-preserving accepted outputs) before the corruption pipeline
tokenizes and masks it. It never drops a record — a target that fails to
canonicalize is passed through unchanged and counted.

Diagnostic/data-prep only; running the comparison (forward vs post-hoc vs none)
is an X-row on `scripts/run_grammar_matrix.py`, not claimed here.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

from slm_training.dsl.canonicalize import canonicalize
from slm_training.dsl.schema import ExampleRecord


def simplify_target(source: str, *, dsl: str | None = None) -> str:
    """Canonical (simplified) form of ``source``; original on any failure."""
    try:
        return canonicalize(source, dsl=dsl)
    except Exception:  # noqa: BLE001 - never drop a target; pass it through
        return source


def simplify_record(record: ExampleRecord, *, dsl: str | None = None) -> ExampleRecord:
    """Return ``record`` with its ``openui`` target canonicalized.

    Placeholders and all other fields are preserved (canonicalization keeps
    placeholder identities; it only normalizes binder names, order, and style).
    """
    simplified = simplify_target(record.openui, dsl=dsl)
    if simplified == record.openui:
        return record
    return replace(record, openui=simplified)


def simplify_records(
    records: Iterable[ExampleRecord], *, dsl: str | None = None
) -> tuple[list[ExampleRecord], dict[str, Any]]:
    """Canonicalize every record's target; return (records, stats).

    Stats report how many targets changed and how many distinct canonical forms
    the corpus collapsed to (the equivalence-class count the reverse model must
    learn — lower is a smaller, cleaner learning target).
    """
    out: list[ExampleRecord] = []
    changed = 0
    canonical_forms: set[str] = set()
    for record in records:
        simplified = simplify_record(record, dsl=dsl)
        if simplified.openui != record.openui:
            changed += 1
        canonical_forms.add(simplified.openui)
        out.append(simplified)
    stats = {
        "n": len(out),
        "changed": changed,
        "distinct_canonical_targets": len(canonical_forms),
    }
    return out, stats
