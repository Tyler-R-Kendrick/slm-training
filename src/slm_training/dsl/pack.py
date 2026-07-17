"""F1 (SLM-34): the DSL pack contract.

A **DSL pack** bundles everything the training / decode stack needs to target
one DSL: ``{grammar, canonicalizer, validity oracle, typed-AST corpus
generator, scope rules, placeholder policy, language-contract id}``. The
grammar-backend protocol (`dsl/grammar/backends`) already carries the
parse / validate / serialize / stream-check / token surfaces and is pluggable
via ``SLM_GRAMMAR_DSL``; the pack contract composes a backend with the five
responsibilities that were previously OpenUI-hardwired module imports, so a
second DSL (F2 GraphQL, F3 patterns DSL, F4 nomenclatures, G3 latent packs)
is a *registration*, not a codebase fork.

Layering note: corpus generation lives in ``harnesses/train_data`` (above this
package), so packs hold **lazy providers** — dotted-path strings resolved on
first use — rather than direct imports. Every other member is a plain callable
on strings, which keeps the contract trivially testable.

Deliberately *not* moved: the existing OpenUI component owners
(``dsl/canonicalize.py``, ``dsl/parser.py``, ``dsl/placeholders.py``,
``dsl/language_contract.py``, ``harnesses/train_data/scope_corpus.py``) keep
their paths — the contract wires them by reference. The unblocking value for
F2/G3 is the explicit interface; relocating stable owners would churn every
import site for zero behavioral gain (see `docs/repository-organization.md`:
extend the existing owner).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable

from slm_training.dsl.grammar.backends import GrammarBackend, get_backend


def _resolve(dotted: str) -> Callable[..., Any]:
    """Import ``pkg.module:attr`` lazily (layering-safe provider)."""
    module_name, _, attr = dotted.partition(":")
    if not module_name or not attr:
        raise ValueError(f"provider must be 'module:attr', got {dotted!r}")
    obj = import_module(module_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"provider {dotted!r} resolved to non-callable {obj!r}")
    return obj


@dataclass(frozen=True)
class ScopeRules:
    """How the DSL binds and references symbols.

    ``bind_encodings`` are the reference representations the codec supports
    (OpenUI: absolute ``<BIND_j>`` and C1 relative ``<BINDDEF>``/``<BINDREL_±k>``);
    ``reference_legality`` names the component that enforces scope legality —
    per the program's externalization principle this must be a verifier, not
    the model.
    """

    bind_encodings: tuple[str, ...]
    reference_legality: str
    scope_families_provider: str | None = None

    def scope_families(self) -> tuple[str, ...]:
        if not self.scope_families_provider:
            return ()
        return tuple(_resolve(self.scope_families_provider)())


@dataclass(frozen=True)
class PlaceholderPolicy:
    """Content routing: identity semantics stay out of the model's scope."""

    is_placeholder: Callable[[str], bool]
    extract: Callable[[str], list[str]]
    merge: Callable[..., list[str]]


@dataclass(frozen=True)
class DslPack:
    """One DSL, fully described for the training / decode stack."""

    id: str
    grammar: str  # grammar-backend id (dsl/grammar/backends registry)
    canonicalize: Callable[[str], str]
    canonical_fingerprint: Callable[[str], str]
    validity_oracle: Callable[[str], Any]  # raises on invalid source
    scope_rules: ScopeRules
    placeholder_policy: PlaceholderPolicy
    contract_id: Callable[[], str]
    # Lazy 'module:attr' providers (resolve above-layer components on use).
    corpus_generator_provider: str | None = None
    extras: dict[str, str] = field(default_factory=dict)

    def backend(self) -> GrammarBackend:
        return get_backend(self.grammar)

    def corpus_generator(self) -> Callable[..., Any]:
        if not self.corpus_generator_provider:
            raise LookupError(f"pack {self.id!r} declares no corpus generator")
        return _resolve(self.corpus_generator_provider)


_REGISTRY: dict[str, DslPack] = {}
_DEFAULT_ID = "openui"


def register_pack(pack: DslPack) -> DslPack:
    _REGISTRY[pack.id] = pack
    return pack


def list_packs() -> list[DslPack]:
    _ensure_builtins()
    return list(_REGISTRY.values())


def available_packs() -> list[str]:
    return [pack.id for pack in list_packs()]


def get_pack(dsl: str | None = None) -> DslPack:
    """Resolve a pack: explicit id > ``SLM_DSL_PACK`` > ``SLM_GRAMMAR_DSL``."""
    _ensure_builtins()
    key = (
        dsl
        or os.getenv("SLM_DSL_PACK")
        or os.getenv("SLM_GRAMMAR_DSL")
        or _DEFAULT_ID
    ).strip().lower()
    if key in {"default", "auto"}:
        key = _DEFAULT_ID
    if key not in _REGISTRY:
        raise KeyError(f"unknown DSL pack {dsl!r}; known={sorted(_REGISTRY)}")
    return _REGISTRY[key]


def _ensure_builtins() -> None:
    if _REGISTRY:
        return
    from slm_training.dsl.packs.openui import build_openui_pack

    register_pack(build_openui_pack())


__all__ = [
    "DslPack",
    "PlaceholderPolicy",
    "ScopeRules",
    "available_packs",
    "get_pack",
    "list_packs",
    "register_pack",
]
