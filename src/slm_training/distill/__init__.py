"""Self-distillation: trajectory store, selection, SFT, and repair corpora."""

from slm_training.distill.repair import extract_failure_cone, repair_records_from_traces
from slm_training.distill.select import SelectConfig, corpus_label, select_traces
from slm_training.distill.sft import DistillSFTConfig, train_self_distill, traces_to_records
from slm_training.distill.trace_store import (
    DecodeTraceRecorder,
    TraceStore,
    checkpoint_sha,
    decode_config_hash,
)

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
