"""Grammar-baked deterministic fast-path for OpenUI decode / train aux."""

from typing import Any

from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine, engine_for_dsl
from slm_training.dsl.grammar.fastpath.force_emit import draft_forced_ids, force_next_token_id
from slm_training.dsl.grammar.fastpath.maskgit_constrain import admit_fill


def __getattr__(name: str) -> Any:
    """Load the optional PyTorch gate only when explicitly requested."""
    if name == "FastPathGate":
        from slm_training.dsl.grammar.fastpath.gate import FastPathGate

        return FastPathGate
    raise AttributeError(name)

__all__ = [
    "FastPathGate",
    "OpenUIIncrementalEngine",
    "admit_fill",
    "draft_forced_ids",
    "engine_for_dsl",
    "force_next_token_id",
]
