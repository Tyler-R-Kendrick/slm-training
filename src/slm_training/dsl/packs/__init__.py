"""DSL pack registry (F1 / SLM-34).

A pack bundles everything one DSL needs to ride the training/eval stack:
{grammar backend, canonicalizer, validity oracle, corpus generator, scope
check, placeholder policy}. Mirrors the `dsl.grammar.backends` registry
conventions (lazy builtin registration, `default`/`auto` aliases).
"""

from __future__ import annotations

from slm_training.dsl.packs.types import DSLPack, PlaceholderPolicy

_REGISTRY: dict[str, DSLPack] = {}
_DEFAULT_ID = "openui"


def register_pack(pack: DSLPack) -> DSLPack:
    _REGISTRY[pack.id] = pack
    return pack


def list_packs() -> list[DSLPack]:
    _ensure_builtins()
    return list(_REGISTRY.values())


def available_packs() -> list[str]:
    return [pack.id for pack in list_packs()]


def get_pack(dsl: str | None = None) -> DSLPack:
    _ensure_builtins()
    key = (dsl or _DEFAULT_ID).strip().lower()
    if key in {"default", "auto"}:
        key = _DEFAULT_ID
    if key not in _REGISTRY:
        raise KeyError(f"unknown DSL pack {dsl!r}; known={sorted(_REGISTRY)}")
    return _REGISTRY[key]


def _ensure_builtins() -> None:
    if _REGISTRY:
        return
    from slm_training.dsl.packs.arith_sketch import build_pack as _arith
    from slm_training.dsl.packs.graphql import build_pack as _graphql
    from slm_training.dsl.packs.openui import build_pack as _openui
    from slm_training.dsl.packs.toy_layout import build_pack as _toy

    register_pack(_openui())
    register_pack(_toy())
    register_pack(_arith())
    # Registered unconditionally; its oracle needs the optional graphql-core
    # dependency, gated by `backend().available()` at use sites.
    register_pack(_graphql())


__all__ = [
    "DSLPack",
    "PlaceholderPolicy",
    "available_packs",
    "get_pack",
    "list_packs",
    "register_pack",
]
