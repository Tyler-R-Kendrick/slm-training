#!/usr/bin/env python3
"""Cross-corpus redundancy + garbage audit over every committed train snapshot.

Reads all `DataStore.versions("train")` snapshots plus the legacy
`src/slm_training/resources/train_data/` home, then reports:

- pairwise snapshot overlap (exact prompt⊕openui pairs and structural
  fingerprints) — how much the committed corpora duplicate each other;
- global near-duplicate clusters (MinHash banding + Jaccard verification,
  cross-snapshot aware);
- what a strict semantic-dedup pass would drop on the union
  (deterministic lexical engine unless the embeddings extra is pinned);
- per-snapshot garbage rates re-scored under the current
  `assess_record` quality contract.

Durable results land in `docs/design/data-corpus-audit.json` + `.md`
(documenting-experiment-results convention).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from slm_training.data.dedup import (
    fuzzy_payload,
    jaccard_from_signatures,
    minhash_signature,
)
from slm_training.data.leakage import fingerprint_openui_structure, fingerprint_pair
from slm_training.data.quality import assess_record
from slm_training.data.semantic_dedup import apply_semantic_dedup, similarity_engine
from slm_training.data.store import DataStore
from slm_training.dsl.schema import ExampleRecord, load_jsonl

LEGACY_TRAIN_DATA_ROOT = Path("src/slm_training/resources/train_data")
_BANDS = 16
_ROWS_PER_BAND = 4  # 16 * 4 == the 64 MinHash permutations


def _load_snapshots(include_local: bool) -> dict[str, list[ExampleRecord]]:
    store = DataStore()
    snapshots: dict[str, list[ExampleRecord]] = {}
    for ref in store.versions("train"):
        if not include_local and ref.storage not in {"git", "legacy"}:
            continue
        records_path = ref.path / "records.jsonl"
        if records_path.is_file():
            snapshots[ref.dataset_id] = load_jsonl(records_path)
    if LEGACY_TRAIN_DATA_ROOT.is_dir():
        for records_path in sorted(LEGACY_TRAIN_DATA_ROOT.glob("*/records.jsonl")):
            name = f"legacy:{records_path.parent.name}"
            if records_path.parent.name not in snapshots:
                snapshots[name] = load_jsonl(records_path)
    return snapshots


def _overlap_matrix(
    snapshots: dict[str, list[ExampleRecord]],
) -> list[dict[str, object]]:
    pair_sets = {
        name: {fingerprint_pair(r.prompt, r.openui) for r in records}
        for name, records in snapshots.items()
    }
    structure_sets = {
        name: {fingerprint_openui_structure(r.openui) for r in records}
        for name, records in snapshots.items()
    }
    names = sorted(snapshots)
    rows: list[dict[str, object]] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pair_shared = len(pair_sets[a] & pair_sets[b])
            structure_shared = len(structure_sets[a] & structure_sets[b])
            if not pair_shared and not structure_shared:
                continue
            smaller = min(len(pair_sets[a]), len(pair_sets[b])) or 1
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "exact_pair_shared": pair_shared,
                    "structure_shared": structure_shared,
                    "exact_pair_containment": round(pair_shared / smaller, 4),
                }
            )
    rows.sort(key=lambda row: (-int(row["exact_pair_shared"]), str(row["a"])))
    return rows


def _near_dup_clusters(
    snapshots: dict[str, list[ExampleRecord]], *, threshold: float
) -> list[dict[str, object]]:
    """MinHash LSH banding + Jaccard verification + union-find clustering."""
    items: list[tuple[str, str, tuple[int, ...]]] = []  # (snapshot, id, signature)
    for name, records in sorted(snapshots.items()):
        for record in records:
            items.append((name, record.id, minhash_signature(fuzzy_payload(record))))

    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for index, (_, _, signature) in enumerate(items):
        for band in range(_BANDS):
            key = signature[band * _ROWS_PER_BAND : (band + 1) * _ROWS_PER_BAND]
            buckets[(band, key)].append(index)

    parent = list(range(len(items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    checked: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i, x in enumerate(members):
            for y in members[i + 1 :]:
                key = (min(x, y), max(x, y))
                if key in checked:
                    continue
                checked.add(key)
                if jaccard_from_signatures(items[x][2], items[y][2]) >= threshold:
                    union(x, y)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(items)):
        clusters[find(index)].append(index)
    rows: list[dict[str, object]] = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        by_snapshot: dict[str, list[str]] = defaultdict(list)
        for index in members:
            name, record_id, _ = items[index]
            by_snapshot[name].append(record_id)
        rows.append(
            {
                "size": len(members),
                "snapshots": sorted(by_snapshot),
                "cross_snapshot": len(by_snapshot) > 1,
                "members": {
                    name: sorted(ids)[:5] for name, ids in sorted(by_snapshot.items())
                },
            }
        )
    rows.sort(key=lambda row: (-int(row["size"]), str(row["snapshots"])))
    return rows


def _garbage_rates(
    snapshots: dict[str, list[ExampleRecord]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, records in sorted(snapshots.items()):
        scores: list[float] = []
        hard_fails = 0
        reasons: dict[str, int] = defaultdict(int)
        for record in records:
            # Lenient on DESIGN.md so pre-design-md snapshots are scored on
            # content quality, not on artifact-era differences.
            report = assess_record(record, require_design_md=False)
            scores.append(report.score)
            if not report.ok:
                hard_fails += 1
                for reason in report.reasons:
                    reasons[reason] += 1
        total = len(records) or 1
        below = sum(1 for score in scores if score < 0.55)
        rows.append(
            {
                "snapshot": name,
                "records": len(records),
                "hard_fail": hard_fails,
                "hard_fail_rate": round(hard_fails / total, 4),
                "below_threshold": below,
                "below_threshold_rate": round(below / total, 4),
                "mean_score": round(sum(scores) / total, 4) if records else None,
                "top_reasons": dict(
                    sorted(reasons.items(), key=lambda item: -item[1])[:5]
                ),
            }
        )
    rows.sort(key=lambda row: -float(row["hard_fail_rate"]))
    return rows


def _semantic_union_redundancy(
    snapshots: dict[str, list[ExampleRecord]], *, threshold: float | None
) -> dict[str, object]:
    union: list[ExampleRecord] = []
    owner: dict[str, str] = {}
    for name, records in sorted(snapshots.items()):
        for record in records:
            key = f"{name}:{record.id}"
            clone = ExampleRecord(
                id=key,
                prompt=record.prompt,
                openui=record.openui,
                placeholders=list(record.placeholders),
                split=record.split,
                source=record.source,
                meta=dict(record.meta or {}),
                design_md=record.design_md,
                target_kind=record.target_kind,
                target_category=record.target_category,
                accepted_outputs=list(record.accepted_outputs),
            )
            union.append(clone)
            owner[key] = name
    kept, dropped = apply_semantic_dedup(union, threshold=threshold)
    cross = sum(
        1
        for drop in dropped
        if owner.get(str(drop["id"])) != owner.get(str(drop["duplicate_of"]))
    )
    return {
        "engine": similarity_engine(),
        "union_records": len(union),
        "kept": len(kept),
        "dropped": len(dropped),
        "dropped_cross_snapshot": cross,
        "samples": dropped[:20],
    }


def _markdown(report: dict[str, object]) -> str:
    lines = [
        "# Data corpus audit",
        "",
        f"Generated {report['generated_at']} by `scripts/audit_data_corpora.py` "
        f"(semantic engine: {report['semantic_redundancy']['engine']}).",  # type: ignore[index]
        "",
        "## Snapshots",
        "",
        "| snapshot | records | hard-fail rate | below-0.55 rate | mean score |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["garbage"]:  # type: ignore[union-attr]
        lines.append(
            f"| {row['snapshot']} | {row['records']} | {row['hard_fail_rate']} "
            f"| {row['below_threshold_rate']} | {row['mean_score']} |"
        )
    lines += [
        "",
        "## Cross-snapshot exact overlap (top rows)",
        "",
        "| a | b | shared pairs | shared structures | containment |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in list(report["overlap"])[:15]:  # type: ignore[arg-type]
        lines.append(
            f"| {row['a']} | {row['b']} | {row['exact_pair_shared']} "
            f"| {row['structure_shared']} | {row['exact_pair_containment']} |"
        )
    semantic = report["semantic_redundancy"]  # type: ignore[index]
    clusters = report["near_dup_clusters"]  # type: ignore[index]
    cross_clusters = sum(1 for c in clusters if c["cross_snapshot"])  # type: ignore[union-attr]
    lines += [
        "",
        "## Redundancy",
        "",
        f"- MinHash near-dup clusters (≥2 members): {len(clusters)} "  # type: ignore[arg-type]
        f"({cross_clusters} span multiple snapshots).",
        f"- Semantic dedup on the union would drop {semantic['dropped']} of "
        f"{semantic['union_records']} records "
        f"({semantic['dropped_cross_snapshot']} cross-snapshot).",
        "",
        "Full detail: `docs/design/data-corpus-audit.json`.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--near-dup-jaccard", type=float, default=0.9)
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=None,
        help="Cosine cutoff for the union pass (default: engine default).",
    )
    parser.add_argument(
        "--include-local",
        action="store_true",
        help="Also audit uncommitted outputs/data/train versions.",
    )
    parser.add_argument("--out-json", type=Path, default=Path("docs/design/data-corpus-audit.json"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/design/data-corpus-audit.md"))
    args = parser.parse_args(argv)

    snapshots = _load_snapshots(args.include_local)
    if not snapshots:
        raise SystemExit("no committed train snapshots found")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine_pin": os.getenv("SLM_SEMANTIC_DEDUP_ENGINE") or "auto",
        "snapshots": {name: len(records) for name, records in sorted(snapshots.items())},
        "overlap": _overlap_matrix(snapshots),
        "near_dup_clusters": _near_dup_clusters(
            snapshots, threshold=args.near_dup_jaccard
        ),
        "semantic_redundancy": _semantic_union_redundancy(
            snapshots, threshold=args.semantic_threshold
        ),
        "garbage": _garbage_rates(snapshots),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("snapshots",)}, indent=2))
    semantic = report["semantic_redundancy"]
    print(
        f"near-dup clusters: {len(report['near_dup_clusters'])}; "
        f"semantic union drop: {semantic['dropped']}/{semantic['union_records']} "
        f"({semantic['engine']})"
    )
    print(f"wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
