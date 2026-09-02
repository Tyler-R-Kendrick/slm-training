"""Production write path for hypothesis-family closures (WP-4 / FP11).

``conclusions.conclude()`` — the append-only, idempotent, policy-thresholded
family-closure writer — existed with a read-side enforcement gate
(``preflight/concluded_family.py``) but **no production caller**: the
committed ``closed_approaches.v1.json`` ledger stayed empty, so families
never actually concluded and dead approaches kept consuming cycles (the
"engineered never to stop, therefore never to conclude" RC4 gap, WP-4's
missing half).

This module is that caller. It maps durable evidence-store rows
(``EvidenceRecordV1`` in the committed ``local_index.jsonl``) into
``ConclusionEvidenceRecord`` and lets ``conclude()`` apply the committed
climb-policy thresholds. Mapping is conservative (fail-open toward *not*
closing):

- ``decidable`` is True only when the record carries a finite p-value — a
  measurement that could not have rejected anything can never help close a
  family (the power-feasibility law applied at conclusion time).
- ``n_seeds`` and outcomes pass through unchanged; the policy's
  ``adequate_power_requires`` floor does the gating.

Runs fail-soft from the supervisor's post-cycle step: a writer bug must never
take down a training cycle.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from slm_training.autoresearch.conclusions import (
    ClosedApproachRecord,
    ConclusionEvidenceRecord,
    conclude,
)

__all__ = [
    "conclusion_record_from_evidence",
    "write_family_closures",
]


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def conclusion_record_from_evidence(
    raw: Mapping[str, Any],
) -> ConclusionEvidenceRecord | None:
    """Map one ``EvidenceRecordV1`` row into conclusion-criteria form."""
    record_id = str(raw.get("experiment_id") or raw.get("config_fingerprint") or "")
    lever_keys = tuple(str(k) for k in (raw.get("lever_keys") or ()) if str(k).strip())
    if not record_id or not lever_keys:
        return None
    version_stamp = raw.get("version_stamp")
    harness_versions = (
        {str(k): str(v) for k, v in version_stamp.items()}
        if isinstance(version_stamp, Mapping)
        else {}
    )
    return ConclusionEvidenceRecord(
        record_id=record_id,
        lever_keys=lever_keys,
        outcome=str(raw.get("outcome") or ""),
        n_seeds=int(raw.get("n_seeds") or 0),
        decidable=_finite(raw.get("p_value")),
        hypothesis_text=str(raw.get("hypothesis_text") or "") or None,
        reasons=tuple(
            str(r) for r in ((raw.get("blocker"),) if raw.get("blocker") else ())
        ),
        harness_versions=harness_versions,
        run_date=str(raw.get("run_date") or "") or None,
    )


def write_family_closures(
    *,
    records: Iterable[Mapping[str, Any]] | None = None,
    path: Path | None = None,
) -> list[ClosedApproachRecord]:
    """Fold the durable evidence store into family closures (idempotent).

    ``records`` defaults to the committed local evidence index. Returns the
    newly appended closures (empty on every re-run with unchanged evidence,
    by ``conclude()``'s own append-only contract).
    """
    if records is None:
        from slm_training.evidence_store.local_index import load_local_index

        records = [r.model_dump(mode="json") for r in load_local_index()]
    mapped = [
        record
        for record in (conclusion_record_from_evidence(raw) for raw in records)
        if record is not None
    ]
    if not mapped:
        return []
    return conclude(mapped, path=path)
