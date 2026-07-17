"""Pluggable grammar backends for DSL-agnostic parse / train / constrained decode.

Backends expose a common contract so OpenUI (official lang-core or Lark) and
future DSLs can drive the same training + decode stack.
"""

from __future__ import annotations

from slm_training.dsl.stream_types import StreamStatus
from slm_training.dsl.grammar.backends.types import (
    GRAMMARS_DIR,
    GrammarBackend,
    GrammarInfo,
    REPO_ROOT,
)

_REGISTRY: dict[str, GrammarBackend] = {}
_DEFAULT_ID = "openui"


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
        raise KeyError(
            f"unknown grammar backend {dsl!r}; known={sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def set_default_backend(dsl: str) -> GrammarBackend:
    global _DEFAULT_ID
    backend = get_backend(dsl)
    _DEFAULT_ID = backend.info.id
    return backend


def _ensure_builtins() -> None:
    if _REGISTRY:
        return
    from slm_training.dsl.grammar.backends.graphql_js import GraphQLJsBackend
    from slm_training.dsl.grammar.backends.lark_backend import LarkFileBackend
    from slm_training.dsl.grammar.backends.openui_hybrid import OpenUIHybridBackend
    from slm_training.dsl.grammar.backends.openui_langcore import OpenUILangCoreBackend
    from slm_training.dsl.grammar.backends.openui_lark import OpenUILarkBackend
    from slm_training.dsl.grammar.backends.toy_layout import ToyLayoutBackend

    register_backend(OpenUILangCoreBackend())
    register_backend(OpenUILarkBackend())
    register_backend(OpenUIHybridBackend())
    register_backend(ToyLayoutBackend())
    register_backend(GraphQLJsBackend())
    # Alias: generic Lark loader stays available for ad-hoc grammars.
    register_backend(
        LarkFileBackend(
            dsl_id="lark-openui",
            grammar_path=GRAMMARS_DIR / "openui.lark",
            description="Alias of openui-lark via generic LarkFileBackend",
            prop_order_path=GRAMMARS_DIR / "openui_prop_order.json",
        )
    )
    global _DEFAULT_ID
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
