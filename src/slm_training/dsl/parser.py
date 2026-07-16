"""DSL parse/validate/serialize — routed through GrammarBackend.

Default DSL is OpenUI (hybrid: official @openuidev/lang-core when available,
otherwise the Lark grammar in ``src/slm_training/dsl/grammars/openui.lark``). Pass ``dsl=`` or set
``SLM_GRAMMAR_DSL`` to switch grammars for training.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from lark import Lark, UnexpectedInput

from slm_training.dsl.lang_core import ParseError, Program
from slm_training.dsl.schema import OUTPUT_KINDS, OutputKind

_GRAMMAR = Path(__file__).with_name("grammars") / "openui.lark"
_NUMBER = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")


def _backend(dsl: str | None = None):
    from slm_training.dsl.grammar.backends import get_backend

    return get_backend(dsl or os.getenv("SLM_GRAMMAR_DSL") or "openui")


def parse(source: str, *, dsl: str | None = None) -> Program:
    return _backend(dsl).parse(source)


def validate(source: str, *, dsl: str | None = None) -> Program:
    return _backend(dsl).validate(source)


def serialize(program: Program, *, dsl: str | None = None) -> str:
    return _backend(dsl).serialize(program)


def stream_check(source: str, *, dsl: str | None = None) -> Any:
    return _backend(dsl).stream_check(source)


@lru_cache(maxsize=1)
def _fragment_parser() -> Lark:
    return Lark(
        _GRAMMAR.read_text(encoding="utf-8"),
        start=["expr", "statement"],
        parser="lalr",
        maybe_placeholders=False,
    )


def lexical_tokens(source: str) -> list[str]:
    """Return the compiler-derived output symbols, excluding whitespace/comments."""
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    return [token for token in DSLNativeTokenizer.lex_surface(source) if token != "NL"]


def validate_output(
    source: str,
    kind: OutputKind = "document",
    category: str | None = None,
) -> str:
    """Validate one document or compact OpenUI output target."""
    if kind not in OUTPUT_KINDS:
        raise ValueError(f"invalid output kind {kind!r}")
    text = source.strip()
    if not text:
        raise ParseError("output must be non-empty")
    if kind == "document":
        return serialize(validate(text))
    if kind in {"statement", "expression"}:
        try:
            _fragment_parser().parse(text, start="statement" if kind == "statement" else "expr")
        except UnexpectedInput as exc:
            raise ParseError(str(exc)) from exc
        return text

    tokens = lexical_tokens(text)
    if len(tokens) != 1:
        raise ParseError(f"lexical output must contain exactly one symbol; got {tokens!r}")
    token = tokens[0]
    if category == "boolean" and token not in {"true", "false"}:
        raise ParseError("expected a boolean lexical symbol")
    if category == "number" and _NUMBER.fullmatch(token) is None:
        raise ParseError("expected a numeric lexical symbol")
    if category in {"enum", "string"} and not (
        len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}
    ):
        raise ParseError(f"expected a quoted {category} lexical symbol")
    return token


__all__ = [
    "ParseError",
    "Program",
    "lexical_tokens",
    "parse",
    "serialize",
    "stream_check",
    "validate",
    "validate_output",
]
