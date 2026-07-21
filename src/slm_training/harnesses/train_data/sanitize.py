"""Deterministic sanitization pass for generated document targets.

Record-level orchestration of the two ``dsl.analysis`` transforms:

1. :func:`slm_training.dsl.analysis.optimize.optimize` — D2 canonicalization
   plus schema-checked rewrites (default elision, dead-binding removal,
   guarded single-child Stack flattening);
2. :func:`slm_training.dsl.analysis.templatize.templatize` — content-literal
   → ``:binder.slot`` placeholder repair (the official parser's content
   policy rejects literal content props, so this step *rescues* candidates
   that today die at the normalize stage).

The pass runs inside ``_normalize_record`` on the style-stripped source
*before* the official ``validate`` call, so the semantic contract, prompt
remediation, judge, verification stamp, dedup and manifest fingerprints all
see the sanitized bytes. Failure is never fatal: any exception falls back to
the unchanged input with a recorded reason, and the record then faces every
existing gate exactly as it does today — sanitize never drops a record.

Modes (mirrors the ``integrity_gate_mode`` enum shape):

- ``off``     — the pass never runs;
- ``audit``   — outcomes are computed and reported, stored bytes unchanged;
- ``enforce`` — sanitized bytes replace the target.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any

from slm_training.data.contract import canonical_slot_contract
from slm_training.data.structure import strip_style_literals
from slm_training.dsl.analysis.optimize import (
    OptimizeOptions,
    optimize,
)
from slm_training.dsl.analysis.templatize import templatize
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord

SANITIZE_MODES = ("off", "audit", "enforce")

@dataclass(frozen=True)
class SanitizeOptions:
    mode: str = "off"
    optimize: OptimizeOptions = field(default_factory=OptimizeOptions)
    templatize: bool = True

    def __post_init__(self) -> None:
        if self.mode not in SANITIZE_MODES:
            raise ValueError(
                f"unknown sanitize mode {self.mode!r}; expected one of "
                f"{SANITIZE_MODES}"
            )

    @property
    def enabled(self) -> bool:
        return self.mode != "off"


@dataclass(frozen=True)
class SanitizeOutcome:
    openui: str
    placeholders: tuple[str, ...]
    applied: bool
    fallback: bool
    reasons: tuple[str, ...]
    rewrites: dict[str, int]
    flatten_opportunities: int
    literals_templatized: int
    template_fills: dict[str, str]
    templatize_skipped: dict[str, int]
    changed: bool

    def to_meta(self, mode: str) -> dict[str, Any]:
        """Provenance block for ``meta['sanitize']`` (never prompt-visible)."""
        return {
            "mode": mode,
            "applied": self.applied,
            "changed": self.changed,
            "fallback": self.fallback,
            "reasons": list(self.reasons),
            "rewrites": dict(self.rewrites),
            "flatten_opportunities": self.flatten_opportunities,
            "literals_templatized": self.literals_templatized,
            "template_fills": dict(self.template_fills),
            "templatize_skipped": dict(self.templatize_skipped),
        }


def should_sanitize(record: ExampleRecord) -> tuple[bool, str]:
    """Whether the pass may touch this record's target, with the skip reason.

    Every document target is eligible. Symbol-only output is stronger than
    historical verbatim, repair, edit, and scope-pair byte identity.
    """
    if record.target_kind != "document":
        return False, "non_document"
    return True, ""


def _protected_components(prompt: str) -> frozenset[str]:
    from slm_training.data.quality import _prompt_component_mentions

    return _prompt_component_mentions(prompt or "")


@lru_cache(maxsize=4096)
def _sanitize_cached(
    scrubbed: str,
    optimize_options: OptimizeOptions,
    apply_templatize: bool,
) -> SanitizeOutcome:
    # Keyed on the transform inputs only — audit and enforce share entries.
    reasons: list[str] = []
    source = scrubbed

    try:
        optimized = optimize(source, options=optimize_options, validate=False)
        rewrites = dict(optimized.rewrites)
        flatten_opportunities = optimized.flatten_opportunities
        source = optimized.source

        template_fills: dict[str, str] = {}
        templatize_skipped: dict[str, int] = {}
        if apply_templatize:
            templatized = templatize(source)
            template_fills = dict(templatized.replacements)
            templatize_skipped = dict(templatized.skipped)
            source = templatized.source

        # The official parser is the only authority on the final form: it
        # re-checks grammar, schema, and the placeholder content policy, and
        # its serializer output is the stored fixed point (G8-compatible).
        from slm_training.dsl.parser import validate
        from slm_training.dsl.language_contract import assert_symbol_only_output

        program = validate(source)
        final = strip_style_literals(program.serialized or source).strip()
        if not final:
            raise ValueError("sanitized program serialized to empty source")
        assert_symbol_only_output(final)
    except Exception as exc:  # noqa: BLE001 - fallback must never drop a record
        reason = f"{type(exc).__name__}: {exc}"
        return SanitizeOutcome(
            openui=scrubbed,
            placeholders=tuple(extract_placeholders(scrubbed)),
            applied=False,
            fallback=True,
            reasons=(reason[:300],),
            rewrites={},
            flatten_opportunities=0,
            literals_templatized=0,
            template_fills={},
            templatize_skipped={},
            changed=False,
        )

    return SanitizeOutcome(
        openui=final,
        placeholders=tuple(canonical_slot_contract(final)),
        applied=True,
        fallback=False,
        reasons=tuple(reasons),
        rewrites=rewrites,
        flatten_opportunities=flatten_opportunities,
        literals_templatized=len(template_fills),
        template_fills=template_fills,
        templatize_skipped=templatize_skipped,
        changed=final != scrubbed.strip(),
    )


def sanitize_openui(
    openui: str,
    *,
    prompt: str = "",
    options: SanitizeOptions,
) -> SanitizeOutcome:
    """Sanitize one document target; cached per (target, prompt-mentions).

    Template/paraphrase synthesis emits several rows per seed with identical
    targets, so the cache collapses the pass to roughly one extra official
    parse per *unique* document target.
    """
    scrubbed = strip_style_literals(openui or "").strip()
    optimize_options = replace(
        options.optimize, protected_components=_protected_components(prompt)
    )
    return _sanitize_cached(scrubbed, optimize_options, options.templatize)


def sanitize_cache_info() -> dict[str, int]:
    info = _sanitize_cached.cache_info()
    return {"hits": info.hits, "misses": info.misses}


def sanitized_reserved_structures(
    seed_path: object, options: SanitizeOptions
) -> set[str]:
    """Structural fingerprints of the reserved test fixtures *after* sanitize.

    Reserved-structure decontamination compares structural fingerprints, and
    sanitization (canonical statement order, default elision, flattening)
    moves admitted records to a different fingerprint family than the raw
    fixtures. Augmenting the reserved set with each fixture's sanitized form
    keeps the test-structure firewall intact — strictly more is rejected,
    never less.
    """
    from pathlib import Path

    from slm_training.data.leakage import fingerprint_openui_structure
    from slm_training.dsl.schema import load_jsonl

    path = Path(str(seed_path)) if seed_path is not None else None
    if path is None or not path.exists() or not options.enabled:
        return set()
    fingerprints: set[str] = set()
    for record in load_jsonl(path):
        if record.target_kind != "document":
            continue
        outcome = sanitize_openui(record.openui, options=options)
        fingerprints.add(fingerprint_openui_structure(outcome.openui))
    return fingerprints


def aggregate_sanitization(
    records: list[ExampleRecord],
    *,
    mode: str,
) -> dict[str, Any]:
    """Fold per-record ``meta['sanitize']`` blocks into the report section.

    Records without a block (verbatim / non-document branches never reach the
    seam) are classified through :func:`should_sanitize` so every record is
    accounted for.
    """
    totals = {
        "defaults_elided": 0,
        "dead_bindings_removed": 0,
        "containers_flattened": 0,
        "flatten_opportunities": 0,
    }
    skipped: dict[str, int] = {}
    sanitized = 0
    changed = 0
    fallbacks = 0
    fallback_reasons: dict[str, int] = {}
    literals_templatized = 0
    records_templatized = 0
    for record in records:
        block = (record.meta or {}).get("sanitize")
        if not isinstance(block, dict):
            eligible, reason = should_sanitize(record)
            key = reason if not eligible else "unprocessed"
            skipped[key] = skipped.get(key, 0) + 1
            continue
        skip_reason = block.get("skip_reason")
        if skip_reason:
            skipped[str(skip_reason)] = skipped.get(str(skip_reason), 0) + 1
            continue
        sanitized += 1
        if block.get("changed"):
            changed += 1
        if block.get("fallback"):
            fallbacks += 1
            for reason in block.get("reasons") or []:
                key = str(reason).split(":", 1)[0]
                fallback_reasons[key] = fallback_reasons.get(key, 0) + 1
        for key in ("defaults_elided", "dead_bindings_removed", "containers_flattened"):
            totals[key] += int((block.get("rewrites") or {}).get(key) or 0)
        totals["flatten_opportunities"] += int(block.get("flatten_opportunities") or 0)
        count = int(block.get("literals_templatized") or 0)
        literals_templatized += count
        if count:
            records_templatized += 1
    return {
        "mode": mode,
        "sanitized": sanitized,
        "changed": changed,
        "skipped": dict(sorted(skipped.items())),
        "rewrites": totals,
        "literals_templatized": literals_templatized,
        "records_templatized": records_templatized,
        "fallbacks": fallbacks,
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "cache": sanitize_cache_info(),
    }


__all__ = [
    "SANITIZE_MODES",
    "SanitizeOptions",
    "SanitizeOutcome",
    "aggregate_sanitization",
    "sanitize_cache_info",
    "sanitize_openui",
    "should_sanitize",
]
