"""Self-distillation: trajectory store, selection, SFT, and repair corpora."""

from __future__ import annotations

from typing import Any

from slm_training.harnesses.distill.repair import (
    extract_failure_cone,
    repair_records_from_traces,
)
from slm_training.harnesses.distill.select import (
    SelectConfig,
    corpus_label,
    select_traces,
)
from slm_training.harnesses.distill.trace_store import (
    DecodeTraceRecorder,
    TraceStore,
    checkpoint_sha,
    decode_config_hash,
)


def __getattr__(name: str) -> Any:
    """Load the PyTorch self-distill SFT trainer only when explicitly requested."""
    if name in {"DistillSFTConfig", "train_self_distill", "traces_to_records"}:
        from slm_training.harnesses.distill.sft import (
            DistillSFTConfig,
            train_self_distill,
            traces_to_records,
        )

        return {
            "DistillSFTConfig": DistillSFTConfig,
            "train_self_distill": train_self_distill,
            "traces_to_records": traces_to_records,
        }[name]
    raise AttributeError(name)


__all__ = [
    "DecodeTraceRecorder",
    "DistillSFTConfig",
    "SelectConfig",
    "TraceStore",
    "checkpoint_sha",
    "corpus_label",
    "decode_config_hash",
    "extract_failure_cone",
    "repair_records_from_traces",
    "select_traces",
    "traces_to_records",
    "train_self_distill",
]
