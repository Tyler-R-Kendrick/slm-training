"""Shared DSL package."""

from slm_training.dsl.placeholders import extract_placeholders, is_placeholder
from slm_training.dsl.parser import ParseError, parse, serialize, validate
from slm_training.dsl.schema import ExampleRecord, load_jsonl, write_jsonl

__all__ = [
    "ExampleRecord",
    "ParseError",
    "extract_placeholders",
    "is_placeholder",
    "load_jsonl",
    "parse",
    "serialize",
    "validate",
    "write_jsonl",
]
