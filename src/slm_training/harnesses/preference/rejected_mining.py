"""Mine preference pairs from a dataset's rejected-record ledger.

The strict build persists every rejected candidate with its stage, reason,
and (for quality/verification stages) full payload. Those payloads are
exactly the "valid-but-worse" / "invalid" negatives preference training
wants: each is paired against the best admitted record sharing its root
parent, so the model learns to prefer the surviving twin over the rejected
variant. Complements the deliberate ``soft_corrupt`` negatives with
negatives the gates actually caught.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.harnesses.preference import PreferencePair

# Stages whose payloads make honest preference negatives: quality (parseable
# but below the bar) and verification (quarantined by a hard gate).
MINABLE_STAGES = frozenset({"quality", "verification"})


def _root_of(meta: dict[str, Any], record_id: str) -> str:
    return str(
        meta.get("root_parent_id")
        or meta.get("parent_id")
        or meta.get("split_group_id")
        or record_id
    )


def _best_admitted_by_root(records: list[ExampleRecord]) -> dict[str, ExampleRecord]:
    best: dict[str, ExampleRecord] = {}

    def score(record: ExampleRecord) -> float:
        meta = record.meta or {}
        if meta.get("curation_score") is not None:
            return float(meta["curation_score"])
        return float((meta.get("quality") or {}).get("score") or 0.0)

    for record in records:
        root = _root_of(record.meta or {}, record.id)
        current = best.get(root)
        if current is None or score(record) > score(current):
            best[root] = record
    return best


def mine_rejected_pairs(dataset_dir: Path) -> list[PreferencePair]:
    dataset_dir = Path(dataset_dir)
    rejected_path = dataset_dir / "rejected.jsonl"
    records_path = dataset_dir / "records.jsonl"
    if not rejected_path.is_file() or not records_path.is_file():
        return []
    admitted = load_jsonl(records_path)
    best_by_root = _best_admitted_by_root(admitted)

    pairs: list[PreferencePair] = []
    for line in rejected_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if entry.get("stage") not in MINABLE_STAGES:
            continue
        payload = entry.get("record")
        if not isinstance(payload, dict) or not payload.get("openui"):
            continue
        payload_meta = payload.get("meta") or {}
        root = _root_of(payload_meta, str(payload.get("id")))
        twin = best_by_root.get(root)
        if twin is None or twin.openui.strip() == str(payload["openui"]).strip():
            continue
        detail = entry.get("detail") or {}
        chosen_meta = twin.meta or {}
        pairs.append(
            PreferencePair(
                prompt=str(payload.get("prompt") or twin.prompt),
                chosen=twin.openui,
                rejected=str(payload["openui"]),
                design_md=twin.design_md,
                chosen_score=float(
                    chosen_meta.get("curation_score")
                    or (chosen_meta.get("quality") or {}).get("score")
                    or 0.0
                ),
                rejected_score=float(detail.get("score") or 0.0),
                meta={
                    "pair_corpus": "rejected_ledger",
                    "source_stage": entry.get("stage"),
                    "rejection_reason": entry.get("reason"),
                    "rejected_id": payload.get("id"),
                    "chosen_id": twin.id,
                    "root_parent_id": root,
                },
            )
        )
    pairs.sort(key=lambda pair: (str(pair.meta or {}).lower()))
    return pairs


def pairs_fingerprint(pairs: list[PreferencePair]) -> str:
    digest = hashlib.sha256()
    for pair in pairs:
        digest.update(
            f"{pair.prompt}\n{pair.chosen}\n{pair.rejected}\n".encode("utf-8")
        )
    return digest.hexdigest()


__all__ = ["MINABLE_STAGES", "mine_rejected_pairs", "pairs_fingerprint"]
