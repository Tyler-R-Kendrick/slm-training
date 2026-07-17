"""Shared types for DSL packs (F1 / SLM-34).

A pack is the full per-DSL bundle the training/eval stack needs:
grammar backend + canonicalizer + validity oracle + typed corpus generator +
scope check + placeholder policy. The grammar/decode half was already
pluggable via `dsl.grammar.backends`; the pack lifts the remaining
OpenUI-hard-wired pieces behind one explicit, registrable contract so F2
(GraphQL), F3 (patterns), and F4 (nomenclatures) implement a bundle instead
of editing call sites.

Design constraints carried from the F3/F4 scoping docs:
- `grammar` and `validity_oracle` are independent fields — an ontology pack
  (F4) validates by consistency check, not by CFG parse.
- `canonicalize` must be a normal form (idempotent); identity is legal for
  DSLs without a codec round-trip but must be stated in `notes`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from slm_training.dsl.grammar.backends import GrammarBackend, get_backend
from slm_training.dsl.schema import ExampleRecord
from slm_training.dsl.stream_types import StreamStatus


@dataclass(frozen=True)
class PlaceholderPolicy:
    """How a DSL routes content: which strings are placeholders and where
    they may appear. `content_props` is empty for DSLs without prop routing."""

    is_placeholder: Callable[[str], bool]
    extract: Callable[[str], list[str]]
    content_props: frozenset[str] = frozenset()


@dataclass(frozen=True)
class DSLPack:
    """One DSL's complete bundle. All callables operate on surface source."""

    id: str
    description: str
    # Id in the grammar-backend registry (parse / serialize / decode masks).
    grammar: str
    # Normal form: idempotent, semantics-preserving. Identity allowed (notes).
    canonicalize: Callable[[str], str]
    canonical_equal: Callable[[str, str], bool]
    # (source, output_kind) -> parsed program object; raises on invalid.
    # Independent of `grammar` so non-CFG oracles (F4) plug in.
    validity_oracle: Callable[[str, str], object]
    # Deterministic typed-AST corpus generator: (count, seed) -> records.
    # None = corpus is externally curated (must be stated in notes).
    corpus_generator: Callable[[int, int], list[ExampleRecord]] | None
    # Streaming scope/legality check for partial sources.
    scope_check: Callable[[str], StreamStatus]
    placeholders: PlaceholderPolicy
    # Honest boundaries of this pack (identity canonicalizer, no generator…).
    notes: tuple[str, ...] = field(default=())

    def backend(self) -> GrammarBackend:
        return get_backend(self.grammar)
