"""slm-training: OpenUI harnesses and adapters."""

from importlib import import_module
from typing import Any

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


def __getattr__(name: str) -> Any:
    """Load legacy DSL exports only when callers request them."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("slm_training.dsl"), name)
    globals()[name] = value
    return value
