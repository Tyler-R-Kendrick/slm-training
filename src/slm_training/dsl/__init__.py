"""Shared OpenUI subset DSL: lang-core bridge, schema, and grammar stack.

Subpackages:

- ``dsl.design_md`` — DESIGN.md lint bridge
- ``dsl.grammar.backends`` — lang-core / Lark / hybrid / toy-layout backends
- ``dsl.grammar.fastpath`` — DFA force-emit, MaskGIT admit, FastPathGate
"""

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
from slm_training.dsl.production_codec import (
    ProductionCodec,
    ProductionProgram,
    ProductionVocab,
    build_vocab_from_corpus,
    decode_productions,
    encode_openui,
    roundtrip_openui,
)
from slm_training.dsl.schema import ExampleRecord, load_jsonl, write_jsonl

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
    "encode_openui",
    "extract_placeholders",
    "generate_system_prompt",
    "is_placeholder",
    "library_schema",
    "load_jsonl",
    "parse",
    "roundtrip_openui",
    "serialize",
    "stream_check",
    "validate",
    "write_jsonl",
]
