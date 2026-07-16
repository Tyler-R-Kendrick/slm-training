"""Scope-graded data families: identity anchors, canonical-form pairs,
scoped repairs, and typed lexical mappings.

Every family is emitted at each AST-derived lexical scope (document /
statement / expression / lexical) via the grammar-generic
:mod:`slm_training.data.scope_extract` walker:

* ``scope_identity_{scope}`` — echo pairs (input == output, byte-identical)
  that intentionally overfit the model on the DSL's own surface;
* ``scope_canonical_{scope}`` — the *same inputs*, but the target is the
  deterministic canonical form (``validate(...).serialized``), plus a
  chosen=canonical / rejected=verbatim preference pair for ranking bias;
* ``scope_repair_{scope}`` — meaningful typos/mistakes as inputs with the
  corrected fragment as target (document scope stays with the existing
  ``corruption_repair`` family);
* ``lexical_typed_map`` — surface token -> typed AST-node rendering
  (``true`` -> ``Boolean(true)``).

All emission is deterministic (no model calls, no RNG) and every record ties
back to its root program's lineage/split group so split isolation holds.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from slm_training.data.scope_extract import (
    SCOPES,
    TERMINAL_CATEGORIES,
    ScopeSlice,
    extract_scope_slices,
    typed_render,
)
from slm_training.dsl.parser import ParseError, validate, validate_output
from slm_training.dsl.schema import ExampleRecord, OutputKind

TYPED_FAMILY = "lexical_typed_map"
REPAIR_SCOPES = ("statement", "expression", "lexical")

_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')

IDENTITY_PROMPT = "Emit the OpenUI {scope} for this input.\n---INPUT---\n{text}"
REPAIR_PROMPT = "Repair this OpenUI {scope}.\n---BROKEN---\n{text}"
TYPED_PROMPT = "Type this OpenUI token: {text}"


def scope_families() -> tuple[str, ...]:
    """All source-family names owned by this module."""
    return (
        *(f"scope_identity_{scope}" for scope in SCOPES),
        *(f"scope_canonical_{scope}" for scope in SCOPES),
        *(f"scope_repair_{scope}" for scope in REPAIR_SCOPES),
        TYPED_FAMILY,
    )


@dataclass(frozen=True)
class ScopeCorpusConfig:
    scopes: tuple[str, ...] = SCOPES
    identity_per_scope: int = 3
    canonical_pairs_per_scope: int = 3
    repairs_per_scope: int = 2
    typed_per_program: int = 4
    dsl: str = "openui"


@dataclass(frozen=True)
class ScopePreferencePair:
    """Canonical-bias ranking pair; projected to PreferencePair at write time."""

    prompt: str
    chosen: str
    rejected: str
    scope: str
    root_id: str
    canonical_pair_id: str
    variant: str


@dataclass(frozen=True)
class _Root:
    root_id: str
    split: str
    split_group_id: str
    program_family_id: str
    lineage_id: str


def _record_id(*parts: str) -> str:
    raw = ":".join(parts)
    return f"{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}_scope"


def _kind(scope: str) -> OutputKind:
    return scope  # scope names deliberately match OutputKind values


def _category(slice_: ScopeSlice) -> str | None:
    if slice_.scope != "lexical":
        return None
    return TERMINAL_CATEGORIES.get(slice_.category)


def _valid_fragment(slice_: ScopeSlice) -> bool:
    if slice_.scope == "document":
        return True
    try:
        validate_output(slice_.text, _kind(slice_.scope), _category(slice_))
    except ParseError:
        return False
    return True


def _base_meta(root: _Root, task: str, slice_: ScopeSlice) -> dict[str, Any]:
    from slm_training.dsl.language_contract import contract_id as current_contract_id

    return {
        "task": task,
        "contract_id": current_contract_id(),
        "determinacy": "deterministic",
        "tier": "Silver",
        "source_kind": "deterministic",
        "program_family_id": root.program_family_id,
        "lineage_id": root.lineage_id,
        "split_group_id": root.split_group_id,
        "parent_id": root.root_id,
        "scope_slice": slice_.to_meta(),
        "scope_kind": slice_.scope,
    }


def _record(
    root: _Root,
    slice_: ScopeSlice,
    *,
    family: str,
    prompt: str,
    openui: str,
    task: str,
    target_kind: OutputKind | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> ExampleRecord:
    meta = _base_meta(root, task, slice_)
    if extra_meta:
        meta.update(extra_meta)
    kind = target_kind if target_kind is not None else _kind(slice_.scope)
    if kind == "document" and task == "identity":
        meta["preserve_verbatim"] = True
    return ExampleRecord(
        id=_record_id(root.root_id, family, slice_.scope, str(slice_.span), openui),
        prompt=prompt,
        openui=openui,
        split=root.split,
        source=family,
        target_kind=kind,
        target_category=_category(slice_) if kind == slice_.scope else None,
        meta=meta,
    )


# --------------------------------------------------------------------------- #
# Deterministic de-canonicalization variants (fail-closed round-trip check)
# --------------------------------------------------------------------------- #


def _rotate_statements(canonical: str) -> str:
    lines = canonical.splitlines()
    if len(lines) < 2:
        return canonical
    return "\n".join([*lines[1:], lines[0]])


def _tighten_whitespace(canonical: str) -> str:
    return canonical.replace(" = ", "=").replace(", ", ",")


def _single_quote_strings(canonical: str) -> str:
    def flip(match: re.Match[str]) -> str:
        inner = match.group()[1:-1]
        if "'" in inner or "\\" in inner:
            return match.group()
        return f"'{inner}'"

    return _STRING_RE.sub(flip, canonical)


VARIANT_TRANSFORMS = (
    ("rotate_statements", _rotate_statements),
    ("tighten_whitespace", _tighten_whitespace),
    ("single_quote_strings", _single_quote_strings),
)


def decanonicalize_variants(canonical: str) -> list[tuple[str, str]]:
    """Deterministic non-canonical rewrites that round-trip to ``canonical``.

    Fail-closed: a variant is only returned when it parses AND its canonical
    serialization is byte-identical to the input.
    """
    variants: list[tuple[str, str]] = []
    seen = {canonical}
    for name, transform in VARIANT_TRANSFORMS:
        candidate = transform(canonical)
        if candidate in seen:
            continue
        try:
            program = validate(candidate)
        except (ParseError, RuntimeError, ValueError):
            continue
        if (program.serialized or candidate.strip()) != canonical:
            continue
        seen.add(candidate)
        variants.append((name, candidate))
    return variants


# --------------------------------------------------------------------------- #
# Emitters
# --------------------------------------------------------------------------- #


def _slices_by_scope(
    source: str, config: ScopeCorpusConfig
) -> dict[str, list[ScopeSlice]]:
    out: dict[str, list[ScopeSlice]] = {scope: [] for scope in config.scopes}
    for slice_ in extract_scope_slices(source, dsl=config.dsl, scopes=config.scopes):
        if _valid_fragment(slice_):
            out[slice_.scope].append(slice_)
    return out


def _identity_records(
    root: _Root,
    by_scope: dict[str, list[ScopeSlice]],
    config: ScopeCorpusConfig,
) -> list[ExampleRecord]:
    records: list[ExampleRecord] = []
    for scope, slices in by_scope.items():
        for slice_ in slices[: config.identity_per_scope]:
            records.append(
                _record(
                    root,
                    slice_,
                    family=f"scope_identity_{scope}",
                    prompt=IDENTITY_PROMPT.format(scope=scope, text=slice_.text),
                    openui=slice_.text,
                    task="identity",
                )
            )
    return records


def _align_key(slice_: ScopeSlice) -> tuple[str, str, tuple[int, ...]]:
    # Statement rotation shifts the top-level index; anchor + intra-statement
    # path stays stable, so align on that instead of absolute position.
    return (slice_.scope, slice_.statement_anchor, slice_.ast_path[1:])


def _canonical_records(
    root: _Root,
    canonical: str,
    canonical_by_scope: dict[str, list[ScopeSlice]],
    config: ScopeCorpusConfig,
) -> tuple[list[ExampleRecord], list[ScopePreferencePair]]:
    from slm_training.data.verify import VerificationContext, verify_record

    records: list[ExampleRecord] = []
    pairs: list[ScopePreferencePair] = []
    emitted_per_scope: dict[str, int] = {}
    for variant_name, variant in decanonicalize_variants(canonical):
        # A variant is always usable as *input*; as an identity *target* it
        # must also clear the layered verifier (fail-closed: some surface
        # forms parse but are quarantined as training targets).
        probe = ExampleRecord(
            id="scope-variant-probe", prompt="scope variant probe", openui=variant
        )
        identity_safe = verify_record(
            probe, VerificationContext(source_kind="deterministic")
        ).ok
        variant_by_scope = _slices_by_scope(variant, config)
        for scope in config.scopes:
            canonical_index = {
                _align_key(item): item for item in canonical_by_scope.get(scope, [])
            }
            for slice_ in variant_by_scope.get(scope, []):
                if emitted_per_scope.get(scope, 0) >= config.canonical_pairs_per_scope:
                    break
                if scope == "document":
                    target = canonical
                else:
                    match = canonical_index.get(_align_key(slice_))
                    if match is None:
                        continue  # fail-closed: drop unmatched slices
                    target = match.text
                if target == slice_.text:
                    continue
                prompt = IDENTITY_PROMPT.format(scope=scope, text=slice_.text)
                pair_id = _record_id(root.root_id, "canonical_pair", scope, slice_.text)
                records.append(
                    _record(
                        root,
                        slice_,
                        family=f"scope_canonical_{scope}",
                        prompt=prompt,
                        openui=target,
                        task="edit",
                        extra_meta={
                            "canonical_pair_id": pair_id,
                            "variant_transform": variant_name,
                            "quality_bias": "high",
                        },
                    )
                )
                # The identity twin shares the exact prompt but echoes the
                # verbatim input — the ranking bias decides between them.
                if scope != "document" or identity_safe:
                    records.append(
                        _record(
                            root,
                            slice_,
                            family=f"scope_identity_{scope}",
                            prompt=prompt,
                            openui=slice_.text,
                            task="identity",
                            extra_meta={
                                "canonical_pair_id": pair_id,
                                "variant_transform": variant_name,
                            },
                        )
                    )
                pairs.append(
                    ScopePreferencePair(
                        prompt=prompt,
                        chosen=target,
                        rejected=slice_.text,
                        scope=scope,
                        root_id=root.root_id,
                        canonical_pair_id=pair_id,
                        variant=variant_name,
                    )
                )
                emitted_per_scope[scope] = emitted_per_scope.get(scope, 0) + 1
    return records, pairs


def _repair_records(
    root: _Root,
    by_scope: dict[str, list[ScopeSlice]],
    config: ScopeCorpusConfig,
) -> list[ExampleRecord]:
    from slm_training.data.corrupt import build_scoped_corruptions

    records: list[ExampleRecord] = []
    for scope in REPAIR_SCOPES:
        if scope not in config.scopes:
            continue
        emitted = 0
        for slice_ in by_scope.get(scope, []):
            if emitted >= config.repairs_per_scope:
                break
            corruptions = build_scoped_corruptions(
                slice_.text,
                _kind(scope),
                category=_category(slice_),
                limit=1,
            )
            for case in corruptions:
                records.append(
                    _record(
                        root,
                        slice_,
                        family=f"scope_repair_{scope}",
                        prompt=REPAIR_PROMPT.format(
                            scope=scope, text=case.broken_text
                        ),
                        openui=case.clean_text,
                        task="repair",
                        extra_meta={
                            "repair": {
                                "family": "scope_repair",
                                "operator": case.operator.value,
                                "operator_family": case.family.value,
                                "location": case.location,
                                "broken": case.broken_text,
                            }
                        },
                    )
                )
                emitted += 1
    return records


def _typed_records(
    root: _Root,
    by_scope: dict[str, list[ScopeSlice]],
    config: ScopeCorpusConfig,
) -> list[ExampleRecord]:
    records: list[ExampleRecord] = []
    seen: set[str] = set()
    for slice_ in by_scope.get("lexical", []):
        if len(records) >= config.typed_per_program:
            break
        if not slice_.typed or slice_.text in seen:
            continue
        rendering = typed_render(slice_.category, slice_.text)
        if rendering is None:
            continue
        seen.add(slice_.text)
        records.append(
            _record(
                root,
                slice_,
                family=TYPED_FAMILY,
                prompt=TYPED_PROMPT.format(text=slice_.text),
                openui=rendering,
                task="generation",
                target_kind="typed_node",
                extra_meta={
                    "lexical_map": {
                        "surface": slice_.text,
                        "terminal": slice_.category,
                    }
                },
            )
        )
    return records


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_scope_corpus(
    *,
    root_id: str,
    openui: str,
    split: str = "train",
    split_group_id: str,
    program_family_id: str,
    lineage_id: str,
    config: ScopeCorpusConfig | None = None,
) -> tuple[list[ExampleRecord], list[ScopePreferencePair]]:
    """All four scope-graded families for one root program.

    ``openui`` is canonicalized first (identity anchors memorize the
    canonical surface; non-canonical variants are derived deterministically
    from it so every canonical pair round-trips by construction).
    """
    cfg = config or ScopeCorpusConfig()
    program = validate(openui)
    canonical = program.serialized or openui.strip()
    root = _Root(
        root_id=root_id,
        split=split,
        split_group_id=split_group_id,
        program_family_id=program_family_id,
        lineage_id=lineage_id,
    )
    by_scope = _slices_by_scope(canonical, cfg)
    records = _identity_records(root, by_scope, cfg)
    canonical_records, pairs = _canonical_records(root, canonical, by_scope, cfg)
    records.extend(canonical_records)
    records.extend(_repair_records(root, by_scope, cfg))
    records.extend(_typed_records(root, by_scope, cfg))
    return records, pairs


__all__ = [
    "REPAIR_SCOPES",
    "TYPED_FAMILY",
    "ScopeCorpusConfig",
    "ScopePreferencePair",
    "build_scope_corpus",
    "decanonicalize_variants",
    "scope_families",
]
