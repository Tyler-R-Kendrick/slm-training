"""Rights, safety, and dataset-metadata gates for external training data."""

from slm_training.data.governance.core import (
    AssetRights,
    ContentScan,
    SourceGovernance,
    emit_dataset_metadata,
    govern_record,
    record_content_hash,
    scan_record,
)

__all__ = [
    "AssetRights",
    "ContentScan",
    "SourceGovernance",
    "emit_dataset_metadata",
    "govern_record",
    "record_content_hash",
    "scan_record",
]
