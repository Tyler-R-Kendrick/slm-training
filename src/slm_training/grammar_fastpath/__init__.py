"""Grammar-baked deterministic fast-path for OpenUI decode / train aux."""

from slm_training.grammar_fastpath.engine import OpenUIIncrementalEngine, engine_for_dsl
from slm_training.grammar_fastpath.force_emit import draft_forced_ids, force_next_token_id
from slm_training.grammar_fastpath.gate import FastPathGate
from slm_training.grammar_fastpath.maskgit_constrain import admit_fill

__all__ = [
    "FastPathGate",
    "OpenUIIncrementalEngine",
    "admit_fill",
    "draft_forced_ids",
    "engine_for_dsl",
    "force_next_token_id",
]
