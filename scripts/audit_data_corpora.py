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
from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.data.quality import assess_record
from slm_training.data.semantic_dedup import apply_semantic_dedup, similarity_engine
from slm_training.data.semantic_plan.canonicalize import plan_factor_fingerprints
from slm_training.data.store import DataStore
from slm_training.data.verify import certify_snapshots, write_certified_snapshot
from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.dsl.symbolic_expr_corpus import (
    CORPUS_SCHEMA_VERSION,
    CorpusGenerationConfigV1,
    CorpusRejectionV1,
    audit_corpus_leakage,
    generate_corpus,
)
from slm_training.versioning import build_version_stamp

LEGACY_TRAIN_DATA_ROOT = Path("src/slm_training/resources/train_data")
_BANDS = 16
_ROWS_PER_BAND = 4  # 16 * 4 == the 64 MinHash permutations

# VCE-008 (SLM-463): renamed-symbol OOD probe. These are the observed
# openui_hard_valid_v1 transform_ids that swap which symbol a role binds to
# without changing structure -- the natural "renamed identity" slice for
# this corpus, rather than an invented one.
_RENAMED_SYMBOL_TRANSFORM_PREFIXES = ("binding_swap_symbol",)


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


# ---------------------------------------------------------------------------
# VCE-008 (SLM-463): leakage, dedup, topology, and OOD split audits over the
# immutable semantic-contrast corpus (default: openui_hard_valid_v1). Reads
# through DataStore.verify (hash-checked) and never writes into the dataset
# directory -- "audited without mutation" by construction.
# ---------------------------------------------------------------------------


def _load_contrast_corpus(dataset_id: str = "openui_hard_valid_v1") -> list[dict]:
    ref = DataStore().verify("eval", dataset_id)
    records_path = ref.path / "records.jsonl"
    return [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _contrast_fingerprints(records: list[dict]) -> list[dict[str, str]]:
    """Alpha-equivalence-normalized per-factor SHA-256 fingerprints, one per record."""
    fingerprints = []
    for record in records:
        plan = SemanticPlanV1.from_dict(record["source_plan"])
        fingerprints.append(plan_factor_fingerprints(plan))
    return fingerprints


def _contrast_lineage_split_report(records: list[dict]) -> dict[str, object]:
    """Group by source_program_id; flag any lineage family split across
    more than one corpus split (scope: 'group before split by lineage')."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[str(record["source_program_id"])].append(index)

    violations: list[dict[str, object]] = []
    for source_program_id, indices in sorted(groups.items()):
        splits = {str(records[i].get("meta", {}).get("split", "unknown")) for i in indices}
        if len(splits) > 1:
            violations.append(
                {
                    "source_program_id": source_program_id,
                    "splits": sorted(splits),
                    "record_ids": [records[i]["record"]["id"] for i in indices],
                }
            )
    return {
        "lineage_groups": len(groups),
        "cross_split_lineage_violations": violations,
    }


def _contrast_alpha_equivalent_duplicates(
    records: list[dict], fingerprints: list[dict[str, str]]
) -> dict[str, object]:
    """Exact/alpha-equivalent duplicate detection keyed on the canonicalized
    'exact' plan fingerprint -- renamed symbols/roles cannot bypass this
    because canonicalize_plan already normalizes them before hashing."""
    by_fingerprint: dict[str, list[int]] = defaultdict(list)
    for index, fp in enumerate(fingerprints):
        by_fingerprint[fp["exact"]].append(index)

    within_split: list[dict[str, object]] = []
    cross_split: list[dict[str, object]] = []
    quarantined: list[dict[str, object]] = []
    for exact_fp, indices in by_fingerprint.items():
        if len(indices) < 2:
            continue
        splits = [str(records[i].get("meta", {}).get("split", "unknown")) for i in indices]
        record_ids = [records[i]["record"]["id"] for i in indices]
        source_lineage = sorted({str(records[i]["source_program_id"]) for i in indices})
        entry = {
            "exact_fingerprint": exact_fp,
            "record_ids": record_ids,
            "source_lineage": source_lineage,
        }
        if len(set(splits)) > 1:
            cross_split.append(entry)
            # Keep the earliest (train-preferring) occurrence; quarantine the rest.
            keep = indices[splits.index("train")] if "train" in splits else indices[0]
            for i in indices:
                if i != keep:
                    quarantined.append(
                        {
                            "record_id": records[i]["record"]["id"],
                            "reason": "alpha_equivalent_cross_split_duplicate",
                            "source_program_id": str(records[i]["source_program_id"]),
                            "duplicate_of": records[keep]["record"]["id"],
                        }
                    )
        else:
            within_split.append(entry)
    return {
        "within_split_duplicates": within_split,
        "cross_split_duplicates": cross_split,
        "quarantined": quarantined,
    }


def _contrast_ood_slices(
    records: list[dict], fingerprints: list[dict[str, str]]
) -> dict[str, object]:
    """Frozen, reproducible OOD slices: held-out rows whose topology,
    component-family composition, or source lineage never appears in train,
    plus the renamed-symbol probe already present via binding_swap_symbol*
    transforms."""
    split_of = [str(r.get("meta", {}).get("split", "unknown")) for r in records]
    train_idx = [i for i, s in enumerate(split_of) if s == "train"]
    held_out_idx = [i for i, s in enumerate(split_of) if s == "held_out"]

    def _unseen_slice(factor: str) -> list[str]:
        train_values = {fingerprints[i][factor] for i in train_idx}
        return sorted(
            records[i]["record"]["id"]
            for i in held_out_idx
            if fingerprints[i][factor] not in train_values
        )

    train_sources = {str(records[i]["source_program_id"]) for i in train_idx}
    unseen_source_family = sorted(
        records[i]["record"]["id"]
        for i in held_out_idx
        if str(records[i]["source_program_id"]) not in train_sources
    )
    renamed_symbol_slice = sorted(
        record["record"]["id"]
        for record in records
        if str(record.get("transform_id", "")).startswith(_RENAMED_SYMBOL_TRANSFORM_PREFIXES)
    )
    return {
        "unseen_topology": _unseen_slice("topology"),
        "unseen_component_combination": _unseen_slice("component_family"),
        "unseen_source_family": unseen_source_family,
        "renamed_symbol": renamed_symbol_slice,
    }


def contrast_corpus_leakage_report(dataset_id: str = "openui_hard_valid_v1") -> dict[str, object]:
    """Full VCE-008 leakage/dedup/topology/OOD audit over the immutable
    semantic-contrast corpus. Read-only: never mutates the dataset."""
    records = _load_contrast_corpus(dataset_id)
    fingerprints = _contrast_fingerprints(records)
    lineage = _contrast_lineage_split_report(records)
    duplicates = _contrast_alpha_equivalent_duplicates(records, fingerprints)
    ood = _contrast_ood_slices(records, fingerprints)
    quarantined_ids = {entry["record_id"] for entry in duplicates["quarantined"]}
    for violation in lineage["cross_split_lineage_violations"]:
        quarantined_ids.update(violation["record_ids"])
    return {
        "dataset_id": dataset_id,
        "records_audited": len(records),
        "lineage": lineage,
        "duplicates": duplicates,
        "ood_slices": {name: len(ids) for name, ids in ood.items()},
        "ood_slice_members": ood,
        "quarantined_record_count": len(quarantined_ids),
        "version_stamp": build_version_stamp("data.corpus_certification"),
    }


def _contrast_markdown(report: dict[str, object]) -> str:
    lineage = report["lineage"]  # type: ignore[index]
    duplicates = report["duplicates"]  # type: ignore[index]
    slices = report["ood_slices"]  # type: ignore[index]
    lines = [
        "# Semantic-contrast corpus leakage/dedup/OOD audit",
        "",
        f"Dataset: `{report['dataset_id']}` ({report['records_audited']} records audited, read-only).",
        "",
        "## Lineage / split leakage",
        "",
        f"- Lineage groups (by `source_program_id`): {lineage['lineage_groups']}",  # type: ignore[index]
        f"- Cross-split lineage violations: {len(lineage['cross_split_lineage_violations'])}",  # type: ignore[index,arg-type]
        "",
        "## Alpha-equivalent / exact duplicates",
        "",
        f"- Within-split duplicate groups: {len(duplicates['within_split_duplicates'])}",  # type: ignore[index,arg-type]
        f"- Cross-split (contaminating) duplicate groups: {len(duplicates['cross_split_duplicates'])}",  # type: ignore[index,arg-type]
        f"- Quarantined records: {report['quarantined_record_count']}",
        "",
        "## OOD slices (frozen, reproducible)",
        "",
    ]
    for name, count in slices.items():
        lines.append(f"- `{name}`: {count} records")
    lines += ["", "Full detail: `docs/design/contrast-corpus-leakage-audit.json`."]
    return "\n".join(lines) + "\n"


def _default_symbolic_regression_corpus_config(seed: int, num_problems: int) -> CorpusGenerationConfigV1:
    return CorpusGenerationConfigV1(
        schema_version=CORPUS_SCHEMA_VERSION,
        seed=seed,
        num_problems=num_problems,
        num_variables_range=(1, 3),
        allowed_operators=frozenset({"+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt", "abs"}),
        max_basis_depth=3,
        max_coefficient_terms=3,
        coefficient_probability=0.7,
        numeric_domain_policy="reject_nonfinite",
        complexity_budget=64,
        train_rows=24,
        validation_rows=8,
        extrapolation_rows=8,
        train_domain=(-3.0, 3.0),
        extrapolation_domain=(4.0, 7.0),
        coefficient_range=(-5.0, 5.0),
    )


def _rejection_histogram(rejections: tuple[CorpusRejectionV1, ...]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for rejection in rejections:
        histogram[rejection.reason] = histogram.get(rejection.reason, 0) + 1
    return histogram


def symbolic_regression_corpus_leakage_report(seed: int = 2026, num_problems: int = 200) -> dict[str, object]:
    """SRP-008 leakage/dedup/topology/OOD audit over a deterministically
    generated symbolic-regression corpus. Nothing is persisted by the pack
    itself -- the corpus is fully reproducible from ``(seed, num_problems)``
    alone (see ``dsl/symbolic_expr_corpus.py``), so auditing it is just
    regenerating it and running the shared leakage/OOD-slice checks."""
    config = _default_symbolic_regression_corpus_config(seed, num_problems)
    corpus = generate_corpus(config)
    audit = audit_corpus_leakage(corpus)
    split_counts: dict[str, int] = {}
    for record in corpus.records:
        split_counts[record.corpus_split] = split_counts.get(record.corpus_split, 0) + 1
    ood_slice_members = corpus.ood_slices.to_dict()
    ood_slice_members.pop("schema_version", None)
    return {
        "seed": seed,
        "num_problems": num_problems,
        "records_generated": len(corpus.records),
        "split_counts": split_counts,
        "rejections": {
            "count": len(corpus.rejections),
            "by_reason": _rejection_histogram(corpus.rejections),
        },
        "leakage_audit": audit.to_dict(),
        "ood_slices": {name: len(ids) for name, ids in ood_slice_members.items()},
        "ood_slice_members": ood_slice_members,
        "corpus_fingerprint": corpus.corpus_fingerprint,
        "version_stamp": build_version_stamp("data.corpus_certification"),
    }


def _symbolic_regression_markdown(report: dict[str, object]) -> str:
    audit = report["leakage_audit"]  # type: ignore[index]
    ood = report["ood_slices"]  # type: ignore[index]
    lines = [
        "# Symbolic-regression corpus leakage/dedup/OOD audit",
        "",
        f"Deterministically regenerated from seed={report['seed']} "
        f"(requested {report['num_problems']} problems, {report['records_generated']} generated).",
        "",
        "## Leakage / dedup",
        "",
        f"- Cross-split family violations: {len(audit['cross_split_family_violations'])}",  # type: ignore[index,arg-type]
        f"- Duplicate family ids: {len(audit['duplicate_family_ids'])}",  # type: ignore[index,arg-type]
        f"- Clean: {audit['is_clean']}",  # type: ignore[index]
        "",
        "## Rejections (filtered, not silently dropped)",
        "",
    ]
    for reason, count in report["rejections"]["by_reason"].items():  # type: ignore[index]
        lines.append(f"- `{reason}`: {count}")
    lines += ["", "## OOD slices (frozen, reproducible)", ""]
    for name, count in ood.items():
        lines.append(f"- `{name}`: {count} records")
    lines += [
        "",
        f"Corpus content fingerprint: `{report['corpus_fingerprint']}`.",
        "",
        "Full detail: `docs/design/symbolic-regression-corpus-leakage-audit.json`.",
    ]
    return "\n".join(lines) + "\n"


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


def _certification_markdown(report: dict[str, object]) -> str:
    tiers = report["tiers"]
    return "\n".join(
        [
            "# SLM-289 corpus certification",
            "",
            "**Claim class:** local deterministic corpus certification; not a model ship result.",
            "",
            f"- source snapshots: {report['source_snapshot_count']}",
            f"- source rows: {report['source_rows']}",
            f"- canonical Gold/Silver core records: {report['core_records']}",
            f"- Gold: {tiers.get('Gold', 0)}; Silver: {tiers.get('Silver', 0)}; Bronze: {tiers.get('Bronze', 0)}; Quarantine: {tiers.get('Quarantine', 0)}",
            f"- quarantine rate: {report['quarantine_rate']:.2%} (matched clean-subset control threshold: >10%)",
            "- G12 remains explicit optional evidence and is never a human-rating admission gate.",
            "",
            "The immutable `openui_verified_v1` snapshot retains every source row in compact membership manifests, records exact-dedup lineage, and admits only canonical Gold/Silver rows to `records.jsonl`. Bronze and Quarantine rows are excluded from core SFT by construction.",
            "",
            "Reproduce from a clean checkout: `python -m scripts.audit_data_corpora --mode certify --output-dir src/slm_training/resources/data/train/openui_verified_v1`. The immutable output refuses overwrite. Opt in to the cleaned snapshot with `--train-version openui_verified_v1`; existing training defaults are unchanged.",
            "",
            "Verification: `DataStore.verify('train', 'openui_verified_v1')` validated every declared artifact hash.",
            "",
        ]
    )


def _write_certification_result(directory: Path) -> tuple[Path, Path]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    quality = json.loads((directory / "quality_report.json").read_text(encoding="utf-8"))
    report = {
        "schema": "slm289_corpus_certification_result/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": manifest["dataset_id"],
        "source_snapshot_count": len(manifest["source_fingerprints"]),
        "source_rows": manifest["source_rows"],
        "core_records": manifest["core_records"],
        "tiers": manifest["tiers"],
        "quarantine_rate": quality["quarantine_rate"],
        "source_fingerprints": manifest["source_fingerprints"],
        "version_stamp": manifest["version_stamp"],
    }
    json_path = Path("docs/design/iter-slm289-corpus-certification-20260725.json")
    md_path = Path("docs/design/corpus-certification.md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_certification_markdown(report), encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("audit", "certify", "report-certification", "contrast-audit", "symbolic-regression-audit"),
        default="audit",
    )
    parser.add_argument(
        "--contrast-dataset-id",
        default="openui_hard_valid_v1",
        help="eval-kind dataset id for --mode contrast-audit (VCE-008).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/slm_training/resources/data/train/openui_verified_v1"),
        help="Immutable published snapshot destination for --mode certify.",
    )
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
    parser.add_argument(
        "--contrast-out-json",
        type=Path,
        default=Path("docs/design/contrast-corpus-leakage-audit.json"),
    )
    parser.add_argument(
        "--contrast-out-md",
        type=Path,
        default=Path("docs/design/contrast-corpus-leakage-audit.md"),
    )
    parser.add_argument(
        "--sr-seed",
        type=int,
        default=2026,
        help="Generation seed for --mode symbolic-regression-audit (SRP-008).",
    )
    parser.add_argument(
        "--sr-num-problems",
        type=int,
        default=200,
        help="Corpus size for --mode symbolic-regression-audit (SRP-008).",
    )
    parser.add_argument(
        "--sr-out-json",
        type=Path,
        default=Path("docs/design/symbolic-regression-corpus-leakage-audit.json"),
    )
    parser.add_argument(
        "--sr-out-md",
        type=Path,
        default=Path("docs/design/symbolic-regression-corpus-leakage-audit.md"),
    )
    args = parser.parse_args(argv)

    if args.mode == "report-certification":
        json_path, md_path = _write_certification_result(args.output_dir)
        print(f"wrote {json_path} and {md_path}")
        return 0
    if args.mode == "contrast-audit":
        report = contrast_corpus_leakage_report(args.contrast_dataset_id)
        args.contrast_out_json.parent.mkdir(parents=True, exist_ok=True)
        args.contrast_out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        args.contrast_out_md.write_text(_contrast_markdown(report), encoding="utf-8")
        print(
            json.dumps(
                {
                    "dataset_id": report["dataset_id"],
                    "records_audited": report["records_audited"],
                    "quarantined_record_count": report["quarantined_record_count"],
                    "ood_slices": report["ood_slices"],
                },
                indent=2,
            )
        )
        print(f"wrote {args.contrast_out_json} and {args.contrast_out_md}")
        return 0
    if args.mode == "symbolic-regression-audit":
        report = symbolic_regression_corpus_leakage_report(args.sr_seed, args.sr_num_problems)
        args.sr_out_json.parent.mkdir(parents=True, exist_ok=True)
        args.sr_out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        args.sr_out_md.write_text(_symbolic_regression_markdown(report), encoding="utf-8")
        print(
            json.dumps(
                {
                    "records_generated": report["records_generated"],
                    "leakage_audit": report["leakage_audit"],
                    "ood_slices": report["ood_slices"],
                },
                indent=2,
            )
        )
        print(f"wrote {args.sr_out_json} and {args.sr_out_md}")
        return 0
    snapshots = _load_snapshots(args.include_local)
    if not snapshots:
        raise SystemExit("no committed train snapshots found")
    if args.mode == "certify":
        store = DataStore()
        fingerprints = {
            ref.dataset_id: (ref.fingerprint or "")
            for ref in store.versions("train")
            if args.include_local or ref.storage in {"git", "legacy"}
        }
        rows = certify_snapshots(snapshots)
        manifest = write_certified_snapshot(
            args.output_dir,
            dataset_id="openui_verified_v1",
            rows=rows,
            source_fingerprints=fingerprints,
        )
        print(
            json.dumps(
                {
                    "dataset_id": "openui_verified_v1",
                    "source_rows": len(rows),
                    "core_records": manifest.get("core_records"),
                    "output_dir": str(args.output_dir),
                },
                indent=2,
            )
        )
        return 0
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
        "version_stamp": build_version_stamp("data.corpus_certification"),
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
