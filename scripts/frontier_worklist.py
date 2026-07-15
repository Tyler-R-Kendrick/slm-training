#!/usr/bin/env python3
"""Write the frontier-artifact worklist: the train-split golds a skill must describe.

Lists **train-split golds only** (never test/held) with their ``gold_content_hash``
so the ``frontier-describe`` agent-skill knows which bundles to (re)generate. A gold
whose artifact already exists at the current hash is marked ``has_fresh_artifact`` so
re-runs fill only the missing / stale ones (idempotent).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.data.frontier import FRONTIER_DIR, artifact_path, gold_content_hash
from slm_training.dsl.schema import load_jsonl


def build_worklist(records_path: Path, *, frontier_root: Path) -> list[dict]:
    rows: list[dict] = []
    for record in load_jsonl(records_path):
        if record.split != "train":
            continue
        gold_hash = gold_content_hash(record.openui, record.prompt)
        rows.append(
            {
                "gold_id": record.id,
                "gold_content_hash": gold_hash,
                "prompt": record.prompt,
                "skeleton_openui": record.openui,
                "has_fresh_artifact": artifact_path(
                    record.id, gold_hash, root=frontier_root
                ).exists(),
            }
        )
    rows.sort(key=lambda r: r["gold_id"])
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("src/slm_training/resources/train_seeds.jsonl"))
    parser.add_argument("--out", type=Path, default=FRONTIER_DIR / "worklist.jsonl")
    parser.add_argument("--frontier-root", type=Path, default=FRONTIER_DIR)
    args = parser.parse_args(argv)

    rows = build_worklist(args.records, frontier_root=args.frontier_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    pending = sum(1 for r in rows if not r["has_fresh_artifact"])
    print(
        json.dumps(
            {"total_train_golds": len(rows), "pending": pending, "out": str(args.out)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
