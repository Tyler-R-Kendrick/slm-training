"""Shared OpenUI subset DSL: official lang-core bridge + record schema."""

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

__all__ = [
    "ExampleRecord",
    "ParseError",
    "Program",
    "bridge_available",
    "extract_placeholders",
    "generate_system_prompt",
    "is_placeholder",
    "library_schema",
    "load_jsonl",
    "parse",
    "serialize",
    "stream_check",
    "validate",
    "write_jsonl",
]
