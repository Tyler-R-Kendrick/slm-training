"""slm-training: OpenUI harnesses and adapters."""

from slm_training.dsl import (
    ExampleRecord,
    bridge_available,
    extract_placeholders,
    generate_system_prompt,
    is_placeholder,
    library_schema,
    load_jsonl,
    parse,
    serialize,
    validate,
    write_jsonl,
)

__all__ = [
    "ExampleRecord",
    "bridge_available",
    "extract_placeholders",
    "generate_system_prompt",
    "is_placeholder",
    "library_schema",
    "load_jsonl",
    "parse",
    "serialize",
    "validate",
    "write_jsonl",
]
