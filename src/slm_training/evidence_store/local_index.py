"""Committed local mirror of the durable evidence store.

The mirror is a deduplicated JSONL file at
``src/slm_training/resources/evidence_store/local_index.jsonl`` — committed so
a fresh container inherits the full cross-session memory even when Supabase
credentials are absent. Records are keyed by
``(config_fingerprint, endpoint_metric)`` and written in stable sorted order
so regeneration produces clean diffs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

from slm_training.evidence_store.records import EvidenceRecordV1

_LOGGER = logging.getLogger(__name__)

DEFAULT_LOCAL_INDEX_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "evidence_store"
    / "local_index.jsonl"
)


def load_local_index(path: Path | None = None) -> list[EvidenceRecordV1]:
    """Load the committed mirror; empty list when missing or unreadable.

    Individually malformed lines are skipped with a single summary warning so
    one bad merge never blinds every consumer.
    """
    target = path or DEFAULT_LOCAL_INDEX_PATH
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return []
    records: list[EvidenceRecordV1] = []
    skipped = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(EvidenceRecordV1.model_validate_json(line))
        except ValueError:
            skipped += 1
    if skipped:
        _LOGGER.warning(
            "evidence_store: skipped %d malformed line(s) in %s", skipped, target
        )
    return records


def dedup_records(records: Iterable[EvidenceRecordV1]) -> list[EvidenceRecordV1]:
    """Deduplicate on ``(config_fingerprint, endpoint_metric)``.

    Later records win (callers merge existing-index records first, then new
    sync output, so a re-run refreshes in place), and the survivors are
    returned in stable ``(config_fingerprint, endpoint_metric)`` order.
    """
    by_key: dict[tuple[str, str], EvidenceRecordV1] = {}
    for record in records:
        by_key[record.dedup_key()] = record
    return [by_key[key] for key in sorted(by_key)]


def write_local_index(
    records: Sequence[EvidenceRecordV1], path: Path | None = None
) -> list[EvidenceRecordV1]:
    """Write the deduplicated, stably sorted mirror; returns what was written."""
    target = path or DEFAULT_LOCAL_INDEX_PATH
    deduped = dedup_records(records)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(record.canonical_line() + "\n" for record in deduped)
    target.write_text(body, encoding="utf-8")
    return deduped


def merge_into_local_index(
    new_records: Sequence[EvidenceRecordV1], path: Path | None = None
) -> list[EvidenceRecordV1]:
    """Merge ``new_records`` over the existing mirror and rewrite it."""
    target = path or DEFAULT_LOCAL_INDEX_PATH
    merged = list(load_local_index(target)) + list(new_records)
    return write_local_index(merged, target)
