"""DSL grammar stack: pluggable backends + incremental fast-path decode."""

from slm_training.dsl.grammar.backends import (
    available_backends,
    get_backend,
    set_default_backend,
)
from slm_training.dsl.grammar.fastpath import (
    FastPathGate,
    OpenUIIncrementalEngine,
    admit_fill,
    draft_forced_ids,
    engine_for_dsl,
    force_next_token_id,
)

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
