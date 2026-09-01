"""slm-training: OpenUI harnesses and adapters."""

from __future__ import annotations

import importlib
import os
from typing import Any

if os.environ.get("_SLM_TRAINING_LIGHT_IMPORT") != "1":
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


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module("slm_training.dsl"), name)
    globals()[name] = value
    return value
