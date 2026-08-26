"""Shared OpenUI subset DSL: lang-core bridge, schema, and grammar stack.

Subpackages:

- ``dsl.design_md`` — DESIGN.md lint bridge
- ``dsl.grammar.backends`` — lang-core / Lark / hybrid / toy-layout backends
- ``dsl.grammar.fastpath`` — DFA force-emit, MaskGIT admit, FastPathGate
"""

from importlib import import_module
from typing import Any

from slm_training.dsl.lang_core import (
    ParseError,
    Program,
    bridge_available,
    generate_system_prompt,
    library_schema,
    parse,
    serialize,
    stream_check,
    validate,
)
from slm_training.dsl.placeholders import extract_placeholders, is_placeholder
from slm_training.dsl.schema import ExampleRecord, load_jsonl, write_jsonl

_PRODUCTION_EXPORTS = frozenset(
    {
        "ProductionCodec",
        "ProductionProgram",
        "ProductionVocab",
        "build_vocab_from_corpus",
        "decode_productions",
        "emit_statement_bindings",
        "encode_openui",
        "parse_statement_bindings",
        "roundtrip_openui",
        "statement_binding_order",
    }
)

__all__ = [
    "ExampleRecord",
    "ParseError",
    "Program",
    "ProductionCodec",
    "ProductionProgram",
    "ProductionVocab",
    "bridge_available",
    "build_vocab_from_corpus",
    "decode_productions",
    "emit_statement_bindings",
    "encode_openui",
    "extract_placeholders",
    "generate_system_prompt",
    "is_placeholder",
    "library_schema",
    "load_jsonl",
    "parse",
    "parse_statement_bindings",
    "roundtrip_openui",
    "statement_binding_order",
    "serialize",
    "stream_check",
    "validate",
    "write_jsonl",
]


def __getattr__(name: str) -> Any:
    """Load the data-dependent production codec after data contracts settle."""
    if name not in _PRODUCTION_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("slm_training.dsl.production_codec"), name)
    globals()[name] = value
    return value
