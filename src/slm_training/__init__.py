"""slm-training: OpenUI harnesses and adapters."""

import os

# Native ORT 1.29 initializes its external telemetry on import. The API opt-out
# is too late to prevent its persistent identifier or initialization event.
# Local evidence stays in the canonical SLM/AgentV owners, not vendor telemetry.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"

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
