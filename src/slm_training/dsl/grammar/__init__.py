"""DSL grammar stack: pluggable backends + incremental fast-path decode."""

from typing import Any

from slm_training.dsl.grammar.backends import (
    available_backends,
    get_backend,
    set_default_backend,
)
from slm_training.dsl.grammar.fastpath import (
    OpenUIIncrementalEngine,
    admit_fill,
    draft_forced_ids,
    engine_for_dsl,
    force_next_token_id,
)


def __getattr__(name: str) -> Any:
    """Load the optional PyTorch gate only when explicitly requested."""
    if name == "FastPathGate":
        from slm_training.dsl.grammar.fastpath import FastPathGate

        return FastPathGate
    raise AttributeError(name)

__all__ = [
    "FastPathGate",
    "OpenUIIncrementalEngine",
    "admit_fill",
    "available_backends",
    "draft_forced_ids",
    "engine_for_dsl",
    "force_next_token_id",
    "get_backend",
    "set_default_backend",
]
