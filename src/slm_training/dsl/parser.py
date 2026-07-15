"""DSL parse/validate/serialize — routed through GrammarBackend.

Default DSL is OpenUI (hybrid: official @openuidev/lang-core when available,
otherwise the Lark grammar in ``src/slm_training/dsl/grammars/openui.lark``). Pass ``dsl=`` or set
``SLM_GRAMMAR_DSL`` to switch grammars for training.
"""

from __future__ import annotations

import os
from typing import Any

from slm_training.dsl.lang_core import ParseError, Program


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


__all__ = ["ParseError", "Program", "parse", "serialize", "stream_check", "validate"]
