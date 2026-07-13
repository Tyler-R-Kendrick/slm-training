"""Self-distillation substrate: rollout / trajectory persistence.

The append-only trace store converts successful decode trajectories into a
durable data asset: the prerequisite for offline self-distillation SFT and
trajectory-aligned diffusion RL (E64).
"""

from slm_training.distill.trace_store import (
    DecodeTraceRecorder,
    TraceStore,
    checkpoint_sha,
    decode_config_hash,
)

__all__ = [
    "DecodeTraceRecorder",
    "TraceStore",
    "checkpoint_sha",
    "decode_config_hash",
]
