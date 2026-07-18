"""Pluggable grammar backends for DSL-agnostic parse / train / constrained decode.

Backends expose a common contract so OpenUI (official lang-core or Lark) and
future DSLs can drive the same training + decode stack.
"""

from __future__ import annotations

import logging
from typing import Callable

from slm_training.dsl.stream_types import StreamStatus
from slm_training.dsl.grammar.backends.types import (
    GRAMMARS_DIR,
    GrammarBackend,
    GrammarInfo,
    REPO_ROOT,
)

_LOGGER = logging.getLogger(__name__)

_REGISTRY: dict[str, GrammarBackend] = {}
_DEFAULT_ID = "openui"
# True only once the default id resolves; a partial load retries on next call.
_BUILTINS_LOADED = False


def register_backend(backend: GrammarBackend) -> GrammarBackend:
    _REGISTRY[backend.info.id] = backend
    return backend


def list_backends() -> list[GrammarInfo]:
    _ensure_builtins()
    return [b.info for b in _REGISTRY.values()]


def available_backends() -> list[str]:
    return [info.id for info in list_backends()]


def get_backend(dsl: str | None = None) -> GrammarBackend:
    _ensure_builtins()
    key = (dsl or _DEFAULT_ID).strip().lower()
    if key in {"default", "auto"}:
        key = _DEFAULT_ID
    if key not in _REGISTRY:
        # The default OpenUI id must keep resolving even if a stale or
        # partially-populated registry only carries a concrete implementation.
        if key == "openui":
            for fallback in ("openui-langcore", "openui-lark"):
                if fallback in _REGISTRY:
                    return _REGISTRY[fallback]
        raise KeyError(
            f"unknown grammar backend {dsl!r}; known={sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def set_default_backend(dsl: str) -> GrammarBackend:
    global _DEFAULT_ID
    backend = get_backend(dsl)
    _DEFAULT_ID = backend.info.id
    return backend


def _try_register(name: str, loader: Callable[[], GrammarBackend]) -> None:
    if name in _REGISTRY:
        return
    try:
        register_backend(loader())
    except Exception:  # noqa: BLE001 — one broken backend must not poison the rest
        _LOGGER.warning("grammar backend %r failed to register", name, exc_info=True)


def _load_langcore() -> GrammarBackend:
    from slm_training.dsl.grammar.backends.openui_langcore import OpenUILangCoreBackend

    return OpenUILangCoreBackend()


def _load_lark() -> GrammarBackend:
    from slm_training.dsl.grammar.backends.openui_lark import OpenUILarkBackend

    return OpenUILarkBackend()


def _load_hybrid() -> GrammarBackend:
    from slm_training.dsl.grammar.backends.openui_hybrid import OpenUIHybridBackend

    return OpenUIHybridBackend()


def _load_toy_layout() -> GrammarBackend:
    from slm_training.dsl.grammar.backends.toy_layout import ToyLayoutBackend

    return ToyLayoutBackend()


def _load_arith_sketch() -> GrammarBackend:
    from slm_training.dsl.grammar.backends.arith_sketch import ArithSketchBackend

    return ArithSketchBackend()


def _load_graphql() -> GrammarBackend:
    from slm_training.dsl.grammar.backends.graphql_js import GraphQLJsBackend

    return GraphQLJsBackend()


def _load_lark_alias() -> GrammarBackend:
    from slm_training.dsl.grammar.backends.lark_backend import LarkFileBackend

    # Alias: generic Lark loader stays available for ad-hoc grammars.
    return LarkFileBackend(
        dsl_id="lark-openui",
        grammar_path=GRAMMARS_DIR / "openui.lark",
        description="Alias of openui-lark via generic LarkFileBackend",
        prop_order_path=GRAMMARS_DIR / "openui_prop_order.json",
    )


_BUILTIN_LOADERS: tuple[tuple[str, Callable[[], GrammarBackend]], ...] = (
    ("openui-langcore", _load_langcore),
    ("openui-lark", _load_lark),
    ("openui", _load_hybrid),
    ("toy-layout", _load_toy_layout),
    ("arith-sketch", _load_arith_sketch),
    ("graphql", _load_graphql),
    ("lark-openui", _load_lark_alias),
)


def _ensure_builtins() -> None:
    """Idempotently register builtins; never leave a poisoned partial registry.

    Every builtin registers independently, so one failing constructor cannot
    abort the rest (a mid-loop exception once stranded the registry with only
    ``openui-langcore`` + ``openui-lark`` and the default ``openui`` id
    permanently unresolvable). The loaded flag is only set once the default id
    resolves, so a transient failure is retried on the next call instead of
    being cached forever.
    """
    global _BUILTINS_LOADED, _DEFAULT_ID
    if _BUILTINS_LOADED:
        return
    for name, loader in _BUILTIN_LOADERS:
        _try_register(name, loader)
    if "openui" not in _REGISTRY:
        for fallback in ("openui-langcore", "openui-lark"):
            if fallback in _REGISTRY:
                _REGISTRY["openui"] = _REGISTRY[fallback]
                _LOGGER.warning(
                    "grammar backend 'openui' unavailable; aliased to %r", fallback
                )
                break
    if "openui" in _REGISTRY:
        _BUILTINS_LOADED = True
        _DEFAULT_ID = "openui"


__all__ = [
    "GRAMMARS_DIR",
    "REPO_ROOT",
    "GrammarBackend",
    "GrammarInfo",
    "StreamStatus",
    "available_backends",
    "get_backend",
    "list_backends",
    "register_backend",
    "set_default_backend",
]
