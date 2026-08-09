"""Durable evidence store: committed local mirror + optional Supabase sync.

See ``docs/design/evidence-store.md``. Public surface:

- :class:`~slm_training.evidence_store.records.EvidenceRecordV1`
- :func:`~slm_training.evidence_store.records.compute_config_fingerprint`
- :func:`~slm_training.evidence_store.client.find_prior_attempts`
"""

from __future__ import annotations

from slm_training.evidence_store.client import (
    EvidenceStoreClient,
    SupabaseConfig,
    find_prior_attempts,
)
from slm_training.evidence_store.local_index import (
    DEFAULT_LOCAL_INDEX_PATH,
    dedup_records,
    load_local_index,
    merge_into_local_index,
    write_local_index,
)
from slm_training.evidence_store.records import (
    EVIDENCE_RECORD_SCHEMA,
    EvidenceRecordV1,
    compute_config_fingerprint,
)

__all__ = [
    "DEFAULT_LOCAL_INDEX_PATH",
    "EVIDENCE_RECORD_SCHEMA",
    "EvidenceRecordV1",
    "EvidenceStoreClient",
    "SupabaseConfig",
    "compute_config_fingerprint",
    "dedup_records",
    "find_prior_attempts",
    "load_local_index",
    "merge_into_local_index",
    "write_local_index",
]
