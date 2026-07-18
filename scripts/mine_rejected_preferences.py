#!/usr/bin/env python3
"""Mine preference pairs from a train dataset's rejected-record ledger.

Turns the strict build's persisted rejects (quality fails, quarantines) into
DPO-style negatives paired against their best admitted twins, written as a
versioned preference dataset (`pairs.jsonl` + manifest) and registered as a
lineage DataSnapshot.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from slm_training.data.store import DataStore, write_common_manifest
from slm_training.harnesses.preference import write_pairs
from slm_training.harnesses.preference.rejected_mining import (
    mine_rejected_pairs,
    pairs_fingerprint,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Train dataset id (resolved via the DataStore) or dataset path.",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Output preference dataset id (default: <dataset>_rejmine).",
    )
    parser.add_argument(
        "--out-root", type=Path, default=Path("outputs/data/preference")
    )
    parser.add_argument(
        "--register-lineage",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lineage-root", type=Path, default=Path("outputs/lineage"))
    args = parser.parse_args(argv)

    store = DataStore()
    dataset_dir = Path(store.resolve_path("train", args.dataset))
    if not dataset_dir.is_dir():
        raise SystemExit(f"train dataset not found: {args.dataset}")
    pairs = mine_rejected_pairs(dataset_dir)
    version = args.version or f"{dataset_dir.name}_rejmine"
    DataStore.validate_id(version)
    out_dir = args.out_root / version
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "pairs.jsonl"
    write_pairs(pairs_path, pairs)
    manifest = {
        "version": version,
        "kind": "preference",
        "pair_corpus": "rejected_ledger",
        "source_dataset": str(dataset_dir.as_posix()),
        "records": str(pairs_path.as_posix()),
        "record_count": len(pairs),
        "content_fingerprint": pairs_fingerprint(pairs),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    write_common_manifest(out_dir, kind="preference", dataset_id=version)
    print(json.dumps({"version": version, "pairs": len(pairs)}, indent=2))
    print(f"wrote {out_dir}")
    if args.register_lineage and pairs:
        from slm_training.lineage.data_cycle import register_dataset_snapshot
        from slm_training.lineage.store import LineageStore

        snapshot, snapshot_path, created = register_dataset_snapshot(
            LineageStore(args.lineage_root), dataset_dir=out_dir, kind="preference"
        )
        state = "registered" if created else "already-registered"
        print(f"lineage_snapshot={snapshot.sha} ({state}: {snapshot_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
