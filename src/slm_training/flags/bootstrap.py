"""Load an in-process ruleset from env / JSON file (zero remote deps)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from slm_training.flags.api import FlagClient
from slm_training.flags.in_memory import InMemoryProvider, ruleset_from_mapping

# OPENUI_FLAGS_JSON='{"verified_solver_decode": true}'
# OPENUI_FLAGS_PATH=path/to/flags.json
ENV_FLAGS_JSON = "OPENUI_FLAGS_JSON"
ENV_FLAGS_PATH = "OPENUI_FLAGS_PATH"


def load_ruleset_mapping(
    *,
    json_text: str | None = None,
    path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = environ if environ is not None else os.environ
    text = json_text
    if text is None:
        text = env.get(ENV_FLAGS_JSON)
    if text:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"{ENV_FLAGS_JSON} must be a JSON object")
        return dict(data)

    flag_path = path if path is not None else env.get(ENV_FLAGS_PATH)
    if flag_path:
        payload = json.loads(Path(flag_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{flag_path}: must contain a JSON object")
        return dict(payload)
    return {}


def client_from_environ(
    *,
    json_text: str | None = None,
    path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> FlagClient | None:
    """Return a FlagClient when a ruleset is configured; else None."""
    mapping = load_ruleset_mapping(json_text=json_text, path=path, environ=environ)
    if not mapping:
        return None
    return FlagClient(InMemoryProvider(ruleset_from_mapping(mapping)))
