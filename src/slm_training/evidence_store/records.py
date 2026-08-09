"""Normalized durable evidence records for experiment outcomes.

One :class:`EvidenceRecordV1` captures one experiment outcome in a
source-agnostic shape so the 1,500+ result JSONs under ``docs/design``, the
committed autotrain-climb ledger, and live loop delivery records all land in a
single queryable index (committed local mirror + optional Supabase table).

Fingerprinting
--------------
``compute_config_fingerprint`` mirrors the canonicalization already used by
``slm_training.autoresearch.evidence_ledger.build_ledger`` for its payload
digest — ``sha256`` over ``json.dumps(obj, sort_keys=True, default=str)``.
That module exposes no importable fingerprint helper (the digest is computed
inline inside ``build_ledger``), so the canonical-JSON convention is mirrored
here and pinned by tests against the same expression.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

EVIDENCE_RECORD_SCHEMA = "evidence_record/v1"

Outcome = Literal[
    "screen_positive",
    "screen_negative",
    "confirm_failed",
    "confirmed",
    "ship_rejected",
    "promoted",
]


def canonical_json(config: Mapping[str, Any]) -> str:
    """Canonical JSON serialization used for fingerprinting.

    Identical canonicalization to the inline payload digest in
    ``slm_training.autoresearch.evidence_ledger.build_ledger``:
    ``json.dumps(payload, sort_keys=True, default=str)``.
    """
    return json.dumps(config, sort_keys=True, default=str)


def compute_config_fingerprint(config: Mapping[str, Any]) -> str:
    """sha256 hex digest of the canonical-JSON-serialized config dict."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


class EvidenceRecordV1(BaseModel):
    """One normalized experiment outcome (durable, source-agnostic)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = EVIDENCE_RECORD_SCHEMA
    experiment_id: str
    campaign_id: str = ""
    hypothesis_text: str = ""
    lever_keys: list[str] = Field(default_factory=list)
    config_fingerprint: str
    endpoint_metric: str
    effect_size: float | None = None
    n_seeds: int = 1
    steps: int = 0
    p_value: float | None = None
    outcome: Outcome
    blocker: str | None = None
    version_stamp: dict[str, Any] = Field(default_factory=dict)
    source_path: str = ""
    run_date: str | None = None

    def dedup_key(self) -> tuple[str, str]:
        """Identity used by the local index and the Supabase unique index."""
        return (self.config_fingerprint, self.endpoint_metric)

    def canonical_line(self) -> str:
        """Stable single-line JSON form for the committed JSONL mirror."""
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, default=str
        )
