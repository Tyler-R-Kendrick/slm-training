"""Certified eval candidates: a root-family split of ``openui_verified_v1``.

The screening climb used to grow its smoke suite from a hand-written tuple of
21 fixtures that were already all in use, so every deficit request returned
nothing while 1,682 certified records sat unused. This module is the sampler
that replaces that tuple:

* **Root families.** Every certified record is linked to its family through
  ``meta.root_parent_id`` / ``meta.split_group_id`` / ``meta.parent_id``; a
  union-find over those links makes the family closed under every recorded
  derivation, and the canonical family id is the smallest ``root_parent_id``
  in the component (the record id when no root is recorded).
* **Program-text closure.** The first split of this corpus lost 39 of its
  eval-bucket candidates to decontamination: their program text, once
  normalized, also sat in the train bucket under a *different* recorded
  family. Closing families under raw text alone recovered almost none of
  them (38 were still discarded) because the collisions appear only after
  style-literal stripping and sanitizing. Families are therefore closed
  under identical program text twice: on the raw corpus at family
  assignment (cheap, before any record can be rejected) and again on the
  *normalized* programs of every admitted record. Records that share a
  program share a bucket, so the eval pool is never silently shrunk by
  cross-family duplicates; the quality report records how many
  link-families the closure merged.
* **Buckets.** ``RootFamilySplitPolicyV1`` hashes the family id into
  ``train`` (0-79), ``validation`` (80-89) or ``test`` (90-99). Eval
  candidates come only from the validation/test buckets; the train bucket is
  materialized as its own dataset so the leakage fingerprints exist.
* **Normalization.** Every record goes through the same ``_normalize`` as the
  fixture path (structure-only strip, deterministic sanitize, symbol-only
  output contract, opaque ``:slot_<n>`` markers). Records that fail the
  contract are rejected and recorded, never patched.
* **Deduplication and re-identification.** The certified corpus carries 700
  distinct ids for 1,682 rows: the same id recurs with distinct prompt/program
  variants. Exact normalized pairs are dropped as duplicates; distinct
  variants sharing an id are kept under a deterministic
  ``<id>__<pair-sha8>`` identity so no program is silently lost and every
  suite id is unique.
* **Decontamination.** Eval candidates are checked with the same
  ``find_leakage`` fingerprints the fixture build uses (id, split group,
  prompt, program text, structure, pair) against the train bucket and any
  extra train manifest the caller names.
* **Selection.** Deterministic, seeded, stratified by ``source`` with a
  round-robin allocation across strata; inside a stratum the greedy order
  prefers unseen program structures, then unseen source ids, then a seeded
  hash, so a suite covers as many distinct programs as its size allows.

Nothing here is a gate: rejections are evidence for the corpus certification
harness (see ``synthesis_feedback.json`` of the materialized train bucket).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from slm_training.data.leakage import (
    fingerprint_design_md,
    fingerprint_openui,
    fingerprint_openui_structure,
    fingerprint_pair,
    fingerprint_prompt,
    find_leakage,
    load_train_fingerprints,
    norm_text,
)
from slm_training.dsl.schema import ExampleRecord, load_jsonl, write_jsonl
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1
from slm_training.versioning import build_version_stamp

CERTIFIED_SCHEMA = "certified_eval_candidates/v1"
CERTIFIED_CORPUS_ID = "openui_verified_v1"
CERTIFIED_TRAIN_BUCKET_ID = "openui_verified_train_v2"
TRAIN_RESOURCE_ROOT = Path("src/slm_training/resources/data/train")
CERTIFIED_CORPUS_DIR = TRAIN_RESOURCE_ROOT / CERTIFIED_CORPUS_ID
CERTIFIED_TRAIN_BUCKET_DIR = TRAIN_RESOURCE_ROOT / CERTIFIED_TRAIN_BUCKET_ID
# Eval-bucket split -> the suite it feeds. Validation feeds screening smoke,
# test feeds the locked held-out suite; the two never share a family.
SUITE_FOR_SPLIT: Mapping[str, str] = {"validation": "smoke", "test": "held_out"}
SPLIT_FOR_SUITE: Mapping[str, str] = {v: k for k, v in SUITE_FOR_SPLIT.items()}
DEFAULT_SEED = 0
_STAMP_COMPONENT = "data.test_build"

# Family links recorded on certified records, in precedence order.
_FAMILY_LINK_KEYS = ("root_parent_id", "split_group_id", "parent_id")
# Certified meta carried onto eval records (the rest is build-time provenance).
_EVAL_META_KEYS = (
    "split_group_id",
    "root_parent_id",
    "source_family",
    "contract_id",
    "verification_tier",
    "program_family_id",
)


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, key: str) -> str:
        parent = self._parent
        parent.setdefault(key, key)
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != root:
            parent[key], key = root, parent[key]
        return root

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            # Deterministic: the lexicographically smaller root survives.
            if root_right < root_left:
                root_left, root_right = root_right, root_left
            self._parent[root_right] = root_left


def _family_links(record: ExampleRecord) -> list[str]:
    meta = record.meta or {}
    links = [record.id]
    for key in _FAMILY_LINK_KEYS:
        value = meta.get(key)
        if value:
            links.append(str(value))
    return links


def root_family_index(
    records: Iterable[ExampleRecord], *, close_under_program_text: bool = True
) -> dict[str, str]:
    """Map every record id to its canonical root-family id.

    Families are the connected components of the id / root_parent_id /
    split_group_id / parent_id link graph, additionally closed under identical
    normalized program text when ``close_under_program_text`` is set (the
    default; see the module docstring for the evidence). The canonical id is
    the smallest ``root_parent_id`` seen in the component, else the smallest
    record id.
    """

    rows = list(records)
    forest = _UnionFind()
    for record in rows:
        links = _family_links(record)
        for link in links[1:]:
            forest.union(links[0], link)
    if close_under_program_text:
        first_with_program: dict[str, str] = {}
        for record in rows:
            program = fingerprint_openui(record.openui)
            anchor = first_with_program.setdefault(program, record.id)
            if anchor != record.id:
                forest.union(anchor, record.id)
    roots: dict[str, set[str]] = defaultdict(set)
    ids: dict[str, set[str]] = defaultdict(set)
    for record in rows:
        component = forest.find(record.id)
        ids[component].add(record.id)
        root = (record.meta or {}).get("root_parent_id")
        if root:
            roots[component].add(str(root))
    canonical: dict[str, str] = {}
    for component, members in ids.items():
        family = min(roots[component]) if roots[component] else min(members)
        for member in members:
            canonical[member] = family
    return canonical


def bucket_of(family_id: str, policy: RootFamilySplitPolicyV1 | None = None) -> int:
    policy = policy or RootFamilySplitPolicyV1()
    return (
        int.from_bytes(hashlib.sha256(family_id.encode("utf-8")).digest()[:8], "big")
        % policy.modulus
    )


@dataclass
class CertifiedPartition:
    """Normalized, deduplicated, re-identified corpus split by root family."""

    schema: str = CERTIFIED_SCHEMA
    corpus_id: str = CERTIFIED_CORPUS_ID
    corpus_records: int = 0
    distinct_source_ids: int = 0
    # Families after program-text closure (what the buckets hash on) and the
    # link-only count before it; the difference is what the closure merged.
    families: int = 0
    link_families: int = 0
    by_split: dict[str, list[ExampleRecord]] = field(
        default_factory=lambda: {"train": [], "validation": [], "test": []}
    )
    family_of: dict[str, str] = field(default_factory=dict)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    # rejection reason -> count, and per (split, reason).
    rejection_histogram: Counter = field(default_factory=Counter)
    rejection_by_split: dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    reidentified: int = 0

    def records(self, split: str) -> list[ExampleRecord]:
        return list(self.by_split[split])

    def family_counts(self) -> dict[str, int]:
        return {
            split: len({self.family_of[r.id] for r in rows})
            for split, rows in self.by_split.items()
        }

    def source_counts(self) -> dict[str, dict[str, int]]:
        return {
            split: dict(sorted(Counter(r.source for r in rows).items()))
            for split, rows in self.by_split.items()
        }


def _normalize_certified(record: ExampleRecord) -> ExampleRecord:
    from slm_training.harnesses.test_data.pipeline import _normalize
    from slm_training.harnesses.train_data.sanitize import SanitizeOptions

    return _normalize(record, sanitize=SanitizeOptions(mode="enforce"))


#: Output kind the trainer applies. ``TwoTowerModel.from_records`` calls
#: ``assert_role_safe_output(record.openui, output_kind=record.target_kind)``
#: and certified records carry no ``target_kind``, so the model resolves the
#: document contract. Admission must apply the same contract or it admits
#: records the trainer refuses.
_ROLE_SAFE_OUTPUT_KIND = "document"


def _assert_certified_role_safe(record: ExampleRecord) -> None:
    """Raise unless the normalized program satisfies the trainer's contract.

    A record that fails here cannot be trained on: ``from_records`` raises on
    the first violation and takes the whole arm down with it (measured on
    2026-09-02, 29 of 1,083 admitted records carried a placeholder in a
    non-content property and every screening arm exited non-zero). The
    contract itself is never relaxed — the record is refused at admission and
    recorded in the rejected ledger like any other refusal.
    """

    from slm_training.dsl.analysis.templatize import assert_role_safe_output

    assert_role_safe_output(record.openui, output_kind=_ROLE_SAFE_OUTPUT_KIND)


def partition_certified_corpus(
    corpus_path: Path | str = CERTIFIED_CORPUS_DIR / "records.jsonl",
    *,
    policy: RootFamilySplitPolicyV1 | None = None,
) -> CertifiedPartition:
    """Split the certified corpus into train / validation / test buckets.

    Family assignment happens on the raw corpus (before any record can be
    rejected) so a rejected variant never moves its siblings between buckets;
    admitted records are then re-closed under identical *normalized* program
    text. Every record is normalized (no bucket-restricted fast path) so the
    sampler and the train-bucket build always see one identical partition.
    """

    policy = policy or RootFamilySplitPolicyV1()
    raw = load_jsonl(corpus_path)
    link_family_of = root_family_index(raw)
    link_only = root_family_index(raw, close_under_program_text=False)
    part = CertifiedPartition(
        corpus_records=len(raw),
        distinct_source_ids=len({r.id for r in raw}),
        link_families=len(set(link_only.values())),
    )
    # Bucket label at rejection time (before the normalized closure).
    split_of_record = {r.id: policy.assign(link_family_of[r.id]) for r in raw}

    normalized: list[tuple[ExampleRecord, ExampleRecord]] = []
    for record in raw:
        split = split_of_record[record.id]
        try:
            candidate = _normalize_certified(record)
        except Exception as exc:  # noqa: BLE001 — recorded, never patched
            reason = type(exc).__name__
            part.rejection_histogram[reason] += 1
            part.rejection_by_split[split][reason] += 1
            part.rejected.append(
                {
                    "id": record.id,
                    "split": split,
                    "source": record.source,
                    "stage": "normalize",
                    "reason": reason,
                    "detail": str(exc)[:200],
                }
            )
            continue
        try:
            _assert_certified_role_safe(candidate)
        except Exception as exc:  # noqa: BLE001 — recorded, never patched
            part.rejection_histogram["role_unsafe_output"] += 1
            part.rejection_by_split[split]["role_unsafe_output"] += 1
            part.rejected.append(
                {
                    "id": record.id,
                    "split": split,
                    "source": record.source,
                    "stage": "role_safety",
                    "reason": "role_unsafe_output",
                    "detail": str(exc)[:200],
                }
            )
            continue
        normalized.append((record, candidate))

    seen_pairs: set[str] = set()
    pairs_per_id: dict[str, set[str]] = defaultdict(set)
    kept: list[tuple[ExampleRecord, ExampleRecord, str]] = []
    for original, record in normalized:
        split = split_of_record[original.id]
        pair = fingerprint_pair(record.prompt, record.openui)
        if pair in seen_pairs:
            part.rejection_histogram["duplicate_pair"] += 1
            part.rejection_by_split[split]["duplicate_pair"] += 1
            part.rejected.append(
                {
                    "id": original.id,
                    "split": split,
                    "source": original.source,
                    "stage": "dedup",
                    "reason": "duplicate_pair",
                    "detail": pair,
                }
            )
            continue
        seen_pairs.add(pair)
        pairs_per_id[original.id].add(pair)
        kept.append((original, record, pair))

    # Normalized program-text closure over admitted records: families whose
    # normalized programs coincide are merged; the surviving canonical id is
    # the smallest merged family id, so the closure is deterministic.
    forest = _UnionFind()
    anchor_by_program: dict[str, str] = {}
    for original, record, _pair in kept:
        family = link_family_of[original.id]
        anchor = anchor_by_program.setdefault(fingerprint_openui(record.openui), family)
        if anchor != family:
            forest.union(anchor, family)
    family_after = {r.id: forest.find(link_family_of[r.id]) for r in raw}
    part.families = len(set(family_after.values()))

    for original, record, pair in kept:
        family = family_after[original.id]
        split = policy.assign(family)
        bucket = bucket_of(family, policy)
        certified_id = original.id
        if len(pairs_per_id[original.id]) > 1:
            certified_id = f"{original.id}__{pair[:8]}"
            part.reidentified += 1
        meta = dict(record.meta)
        meta.update(
            {
                "certified_corpus": part.corpus_id,
                "certified_source_id": original.id,
                "root_family_id": family,
                "root_family_bucket": bucket,
                "root_family_split": split,
            }
        )
        out = ExampleRecord(
            id=certified_id,
            prompt=record.prompt,
            openui=record.openui,
            placeholders=list(record.placeholders),
            split=record.split,
            source=record.source,
            meta=meta,
            design_md=record.design_md,
            target_kind=record.target_kind,
            target_category=record.target_category,
            accepted_outputs=list(record.accepted_outputs),
        )
        part.family_of[out.id] = family
        part.by_split[split].append(out)
    return part


_PARTITION_CACHE: dict[tuple[str, int, int], CertifiedPartition] = {}


def _cached_partition(corpus_path: Path | str) -> CertifiedPartition:
    """Partition keyed by corpus path, size and mtime (normalization is costly)."""

    path = Path(corpus_path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    part = _PARTITION_CACHE.get(key)
    if part is None:
        part = partition_certified_corpus(path)
        _PARTITION_CACHE.clear()
        _PARTITION_CACHE[key] = part
    return part


def _fingerprints_for(records: Iterable[ExampleRecord]) -> dict[str, set[str]]:
    fps: dict[str, set[str]] = {
        "ids": set(),
        "split_group_ids": set(),
        "prompts": set(),
        "openuis": set(),
        "structures": set(),
        "pairs": set(),
        "design_mds": set(),
    }
    for record in records:
        fps["ids"].add(record.id)
        group = (record.meta or {}).get("split_group_id")
        if group:
            fps["split_group_ids"].add(str(group))
        fps["prompts"].add(fingerprint_prompt(record.prompt))
        fps["openuis"].add(fingerprint_openui(record.openui))
        fps["structures"].add(fingerprint_openui_structure(record.openui))
        fps["pairs"].add(fingerprint_pair(record.prompt, record.openui))
        design = fingerprint_design_md(record.design_md)
        if design:
            fps["design_mds"].add(design)
    return fps


def _merge_fingerprints(*sets: Mapping[str, set[str]]) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = defaultdict(set)
    for fps in sets:
        for key, values in fps.items():
            merged[key] |= set(values)
    return dict(merged)


def train_bucket_fingerprints(
    partition: CertifiedPartition,
    *,
    train_bucket_manifest: Path | str | None = CERTIFIED_TRAIN_BUCKET_DIR
    / "manifest.json",
    extra_train_manifests: Iterable[Path | str] = (),
) -> dict[str, set[str]]:
    """Leakage fingerprints of the certified train bucket (+ extra manifests).

    The partition's own train bucket is always fingerprinted; the materialized
    bucket manifest (when present) and any extra manifests are unioned in, so
    the sampler never silently skips decontamination.
    """

    manifest = Path(train_bucket_manifest) if train_bucket_manifest else None
    sets = [_fingerprints_for(partition.by_split["train"])]
    if manifest is not None and manifest.is_file():
        sets.append(load_train_fingerprints(manifest))
    for extra in extra_train_manifests:
        path = Path(extra)
        if path.is_file():
            sets.append(load_train_fingerprints(path))
    return _merge_fingerprints(*sets)


@dataclass
class CertifiedSample:
    records: list[ExampleRecord]
    suite: str
    splits: tuple[str, ...]
    seed: int
    need: int
    eligible: int
    excluded_existing: int
    decontaminated: Counter
    stratification: dict[str, int]
    distinct_source_ids: int
    distinct_structures: int
    families: int

    def report(self) -> dict[str, Any]:
        return {
            "schema": CERTIFIED_SCHEMA,
            "suite": self.suite,
            "splits": list(self.splits),
            "seed": self.seed,
            "requested": self.need,
            "selected": len(self.records),
            "eligible": self.eligible,
            "excluded_existing_ids": self.excluded_existing,
            "decontaminated": dict(sorted(self.decontaminated.items())),
            "stratification": dict(sorted(self.stratification.items())),
            "distinct_source_ids": self.distinct_source_ids,
            "distinct_structures": self.distinct_structures,
            "families": self.families,
        }


def _seeded_hash(seed: int, record_id: str) -> str:
    return hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest()


def select_stratified(
    pool: Iterable[ExampleRecord], *, need: int, seed: int
) -> list[ExampleRecord]:
    """Deterministic diversity-first round-robin over ``source`` strata."""

    by_source: dict[str, list[ExampleRecord]] = defaultdict(list)
    for record in pool:
        by_source[record.source].append(record)
    order = sorted(by_source)
    for source in order:
        by_source[source].sort(key=lambda r: _seeded_hash(seed, r.id))
    structure_used: Counter = Counter()
    source_id_used: Counter = Counter()
    out: list[ExampleRecord] = []
    while len(out) < max(0, need):
        progressed = False
        for source in order:
            candidates = by_source[source]
            if not candidates:
                continue
            best = min(
                candidates,
                key=lambda r: (
                    structure_used[fingerprint_openui_structure(r.openui)],
                    source_id_used[str(r.meta.get("certified_source_id") or r.id)],
                    _seeded_hash(seed, r.id),
                ),
            )
            candidates.remove(best)
            structure_used[fingerprint_openui_structure(best.openui)] += 1
            source_id_used[str(best.meta.get("certified_source_id") or best.id)] += 1
            out.append(best)
            progressed = True
            if len(out) >= need:
                break
        if not progressed:
            break
    return out


def _as_eval_record(record: ExampleRecord, *, suite: str) -> ExampleRecord:
    meta = {
        key: record.meta[key]
        for key in (
            "certified_corpus",
            "certified_source_id",
            "root_family_id",
            "root_family_bucket",
            "root_family_split",
            *_EVAL_META_KEYS,
        )
        if record.meta.get(key) is not None
    }
    meta["suite"] = suite
    return ExampleRecord(
        id=record.id,
        prompt=record.prompt,
        openui=record.openui,
        placeholders=list(record.placeholders),
        split=suite,
        source=record.source,
        meta=meta,
        design_md=record.design_md,
        target_kind=record.target_kind,
        target_category=record.target_category,
        accepted_outputs=list(record.accepted_outputs),
    )


def sample_certified_candidates(
    *,
    existing_ids: Iterable[str],
    need: int,
    suite: str = "smoke",
    splits: tuple[str, ...] | None = None,
    seed: int = DEFAULT_SEED,
    partition: CertifiedPartition | None = None,
    corpus_path: Path | str = CERTIFIED_CORPUS_DIR / "records.jsonl",
    train_bucket_manifest: Path | str | None = CERTIFIED_TRAIN_BUCKET_DIR
    / "manifest.json",
    extra_train_manifests: Iterable[Path | str] = (),
) -> CertifiedSample:
    """Draw up to ``need`` decontaminated eval records for ``suite``.

    ``splits`` defaults to the eval bucket that feeds ``suite``
    (``validation`` for smoke, ``test`` for held_out). Ids listed in
    ``existing_ids`` (the target suite's current ids, plus any ids reserved by
    sibling suites) are never returned. Selection is deterministic for a
    given corpus, seed, and exclusion set.
    """

    if suite not in SPLIT_FOR_SUITE:
        raise ValueError(f"certified sampling supports {sorted(SPLIT_FOR_SUITE)}, not {suite!r}")
    splits = tuple(splits) if splits else (SPLIT_FOR_SUITE[suite],)
    for split in splits:
        if split not in SUITE_FOR_SPLIT:
            raise ValueError(f"eval candidates come from validation/test buckets, not {split!r}")
    if partition is None:
        # The whole corpus is partitioned (the train bucket is derived here
        # too, so decontamination never depends on a manifest being present).
        partition = _cached_partition(corpus_path)
    part = partition
    fps = train_bucket_fingerprints(
        part,
        train_bucket_manifest=train_bucket_manifest,
        extra_train_manifests=extra_train_manifests,
    )
    excluded = {str(item) for item in existing_ids}
    decontaminated: Counter = Counter()
    pool: list[ExampleRecord] = []
    excluded_existing = 0
    for split in splits:
        for record in part.by_split[split]:
            if record.id in excluded or str(record.meta.get("certified_source_id")) in excluded:
                excluded_existing += 1
                continue
            reasons = find_leakage(record, fps)
            if reasons:
                decontaminated["+".join(reasons)] += 1
                continue
            pool.append(record)
    chosen = select_stratified(pool, need=need, seed=seed)
    records = [_as_eval_record(r, suite=suite) for r in chosen]
    return CertifiedSample(
        records=records,
        suite=suite,
        splits=splits,
        seed=seed,
        need=need,
        eligible=len(pool),
        excluded_existing=excluded_existing,
        decontaminated=decontaminated,
        stratification=dict(Counter(r.source for r in records)),
        distinct_source_ids=len({r.meta.get("certified_source_id") for r in records}),
        distinct_structures=len({fingerprint_openui_structure(r.openui) for r in records}),
        families=len({part.family_of[r.id] for r in records}),
    )


def certified_eval_candidates(
    *,
    existing_ids: Iterable[str],
    need: int,
    suite: str = "smoke",
    seed: int = DEFAULT_SEED,
    extra_train_manifests: Iterable[Path | str] = (),
) -> list[ExampleRecord]:
    """Convenience wrapper returning only the sampled records."""

    return sample_certified_candidates(
        existing_ids=existing_ids,
        need=need,
        suite=suite,
        seed=seed,
        extra_train_manifests=extra_train_manifests,
    ).records


# --------------------------------------------------------------------------
# Train-bucket materialization (the leakage-fingerprint side of the split).
# --------------------------------------------------------------------------


def _content_fingerprint(records: Iterable[ExampleRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(f"{record.id}\n{record.prompt}\n{record.openui}\n".encode("utf-8"))
    return digest.hexdigest()


def _synthesis_feedback(part: CertifiedPartition) -> dict[str, Any]:
    """Evidence for the corpus-certification harness, in the feedback shape."""

    recommendations: list[dict[str, Any]] = []
    contract = part.rejection_histogram.get("OutputContractError", 0)
    if contract:
        recommendations.append(
            {
                "code": "stale_output_contract",
                "family": "corpus_certification",
                "evidence": {"rejected": contract, "gate": "symbol_only/v2"},
                "action": (
                    "re-certify openui_verified_v1 under the current symbol-only "
                    "output contract; records carrying free-form string literals "
                    "were certified before symbol_only/v2 and are rejected here, "
                    "never patched"
                ),
            }
        )
    role_unsafe = part.rejection_histogram.get("role_unsafe_output", 0)
    if role_unsafe:
        recommendations.append(
            {
                "code": "role_unsafe_output",
                "family": "corpus_certification",
                "evidence": {
                    "rejected": role_unsafe,
                    "contract": "assert_role_safe_output",
                    "output_kind": _ROLE_SAFE_OUTPUT_KIND,
                },
                "action": (
                    "certified records place placeholders in non-content "
                    "properties (for example RadialChart.labels), which the "
                    "trainer's role-safe output contract refuses at "
                    "from_records; re-certify openui_verified_v1 so a "
                    "placeholder only ever occupies a content property. They "
                    "are refused at admission here, never patched"
                ),
            }
        )
    if part.reidentified:
        recommendations.append(
            {
                "code": "id_collision",
                "family": "corpus_certification",
                "evidence": {
                    "corpus_records": part.corpus_records,
                    "distinct_source_ids": part.distinct_source_ids,
                    "reidentified": part.reidentified,
                },
                "action": (
                    "certification merges several source snapshots that reuse "
                    "record ids for distinct prompt/program variants; namespace "
                    "ids by source snapshot so consumers stop re-identifying"
                ),
            }
        )
    dup = part.rejection_histogram.get("duplicate_pair", 0)
    if dup:
        recommendations.append(
            {
                "code": "redundant_expansion",
                "family": "corpus_certification",
                "evidence": {"duplicate_pair": dup},
                "action": "exact prompt/program duplicates survive certification; dedupe at admission",
            }
        )
    return {
        "schema": "synthesis_feedback/v1",
        "recommendations": recommendations,
        "experiment_candidates": [],
        "note": (
            "root-family split of a certified corpus; nothing was synthesized. "
            "Rejections are admission evidence for the certification harness."
        ),
    }


def materialize_certified_train_bucket(
    output_dir: Path | str,
    *,
    dataset_id: str = CERTIFIED_TRAIN_BUCKET_ID,
    partition: CertifiedPartition | None = None,
    corpus_path: Path | str = CERTIFIED_CORPUS_DIR / "records.jsonl",
) -> dict[str, Any]:
    """Write the train bucket as a train dataset with leakage fingerprints.

    The manifest mirrors the fields ``load_train_fingerprints`` reads
    (``records``, ``ids``, ``split_group_ids`` and the four fingerprint
    lists) so ``build_test_data --train-manifest`` and the leakage test can
    consume it exactly like a ``build_train_data`` snapshot.
    """

    part = partition or partition_certified_corpus(corpus_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = part.by_split["train"]
    records_path = out_dir / "records.jsonl"
    write_jsonl(records_path, records)
    rejected_path = out_dir / "rejected.jsonl"
    rejected_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in part.rejected),
        encoding="utf-8",
    )
    fps = _fingerprints_for(records)
    built_at = datetime.now(timezone.utc).isoformat()
    version_stamp = build_version_stamp(_STAMP_COMPONENT)
    eval_pool = {
        split: {
            "records": len(part.by_split[split]),
            "families": part.family_counts()[split],
            "by_source": part.source_counts()[split],
        }
        for split in ("validation", "test")
    }
    quality_report = {
        "schema": "certified_split_quality/v1",
        "dataset_id": dataset_id,
        "corpus_id": part.corpus_id,
        "corpus_records": part.corpus_records,
        "distinct_source_ids": part.distinct_source_ids,
        "families": part.families,
        "split_policy": {
            "schema_version": RootFamilySplitPolicyV1().schema_version,
            "modulus": RootFamilySplitPolicyV1().modulus,
            "validation_buckets": [80, 89],
            "test_buckets": [90, 99],
            "family_links": list(_FAMILY_LINK_KEYS),
            "program_text_closure": True,
        },
        "family_closure": {
            "link_families": part.link_families,
            "families": part.families,
            "merged_by_program_text": part.link_families - part.families,
        },
        "admitted": {
            "train": len(records),
            "validation": len(part.by_split["validation"]),
            "test": len(part.by_split["test"]),
        },
        "family_counts": part.family_counts(),
        "by_source": part.source_counts(),
        "rejection_histogram": dict(sorted(part.rejection_histogram.items())),
        "rejection_by_split": {
            split: dict(sorted(counter.items()))
            for split, counter in sorted(part.rejection_by_split.items())
        },
        "reidentified": part.reidentified,
        "eval_pool": eval_pool,
        "warnings": [
            f"{part.rejection_histogram['OutputContractError']} certified records "
            "violate symbol_only/v2 and were rejected"
        ]
        if part.rejection_histogram.get("OutputContractError")
        else [],
        "recommendations": ["keep strict leakage gates; do not weaken"],
        "version_stamp": version_stamp,
    }
    (out_dir / "quality_report.json").write_text(
        json.dumps(quality_report, indent=2) + "\n", encoding="utf-8"
    )
    feedback_path = out_dir / "synthesis_feedback.json"
    feedback_path.write_text(
        json.dumps(_synthesis_feedback(part), indent=2) + "\n", encoding="utf-8"
    )
    stats = {
        "version": dataset_id,
        "profile": "certified_split",
        "source": "certified",
        "derive_from": str(Path(corpus_path).as_posix()),
        "record_count": len(records),
        "rejected_total": len(part.rejected),
        "rejection_histogram": quality_report["rejection_histogram"],
        "component_histogram": dict(
            sorted(Counter(r.source for r in records).items())
        ),
        "with_design_md": sum(1 for r in records if r.design_md),
        "sanitize_mode": "enforce",
        "built_at": built_at,
        "version_stamp": version_stamp,
    }
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "version": dataset_id,
        "kind": "train_data",
        "profile": "certified_split",
        "source": "certified",
        "derive_from": str(Path(corpus_path).as_posix()),
        "split_policy": quality_report["split_policy"],
        "records": str(records_path.as_posix()),
        "stats": str(stats_path.as_posix()),
        "quality_report": str((out_dir / "quality_report.json").as_posix()),
        "rejected": str(rejected_path.as_posix()),
        "synthesis_feedback": str(feedback_path.as_posix()),
        "record_count": len(records),
        "ids": [r.id for r in records],
        "root_family_ids": sorted({part.family_of[r.id] for r in records}),
        "split_group_ids": sorted(fps["split_group_ids"]),
        "prompt_fingerprints": sorted(fps["prompts"]),
        "openui_fingerprints": sorted(fps["openuis"]),
        "structure_fingerprints": sorted(fps["structures"]),
        "pair_fingerprints": sorted(fps["pairs"]),
        "design_md_fingerprints": sorted(fps["design_mds"]),
        "content_fingerprint": _content_fingerprint(records),
        "source_families": dict(sorted(Counter(r.source for r in records).items())),
        "built_at": built_at,
        "version_stamp": version_stamp,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "output_dir": str(out_dir),
        "manifest": manifest,
        "quality_report": quality_report,
    }


# --------------------------------------------------------------------------
# Eval sidecars (quality report + certified screening-n range).
# --------------------------------------------------------------------------


def screening_sample_size_sidecar(
    *, eval_version: str, smoke_n: int, suite: str = "smoke", suite_records: int | None = None
) -> dict[str, Any]:
    """Certified screening-n range for a suite at the exact decidability floor.

    The power floor is deliberately not declared: no paired SD has been
    measured for the current screening primary, and inventing one would be a
    fabricated bound. ``chosen_n`` is therefore the exact sign-test floor
    clamped by the arm-wall and suite-volume ceilings. ``suite`` names the
    suite the range is computed for (``suite_records`` defaults to
    ``smoke_n``); ``smoke_n`` stays the smoke count the driver reads.
    """

    if suite_records is None:
        suite_records = smoke_n

    from slm_training.autoresearch.screening_sample_size import (
        ScreeningSampleSizeObservation,
        compute_screening_sample_size,
    )
    from slm_training.levers import (
        HARNESS_FINALIZATION_RESERVE_SECONDS,
        MAX_HARNESS_WALL_SECONDS,
    )

    alpha = "1/20"
    stage_minutes = 3
    decode_floor = 2
    min_train_floor = 20
    overhead = 8
    try:
        from slm_training.autoresearch.climb_policy import load_climb_policy

        policy = load_climb_policy()
        gate = policy.payload.get("power_gate") or {}
        alpha = str(gate.get("alpha") or alpha)
        measurement = policy.measurement
        stage_minutes = int(measurement.get("screening_stage_wall_minutes") or stage_minutes)
        block = measurement.get("screening_sample_size") or {}
        decode_floor = int(block.get("default_decode_floor_seconds") or decode_floor)
        thrash = measurement.get("thrash_timing") or {}
        min_train_floor = int(thrash.get("min_train_floor_seconds") or min_train_floor)
        overhead = int(thrash.get("eval_overhead_seconds") or overhead)
    except Exception:  # noqa: BLE001 — sidecar keeps the documented defaults
        pass
    # Mirrors the driver's symmetric two-stage arm wall (screening, no formal).
    symmetric = (MAX_HARNESS_WALL_SECONDS - HARNESS_FINALIZATION_RESERVE_SECONDS) / 2
    arm_wall_seconds = int(min(stage_minutes * 60, symmetric))
    report = compute_screening_sample_size(
        ScreeningSampleSizeObservation(
            alpha=alpha,
            suite_records=suite_records,
            arm_wall_seconds=arm_wall_seconds,
            min_train_floor_seconds=min_train_floor,
            suite_overhead_seconds=overhead,
            per_record_decode_floor_seconds=decode_floor,
        )
    )
    return {
        "eval_version": eval_version,
        "schema_version": report.schema_version,
        "suite": suite,
        "suite_records": suite_records,
        "smoke_n": smoke_n,
        "report": report.model_dump(mode="json"),
        "power_floor": (
            "not_declared: no measured paired SD for the screening primary; "
            "range is the exact decidability floor clamped by the ceilings"
        ),
        "arm_wall_seconds": arm_wall_seconds,
    }


def write_certified_eval_sidecars(
    output_dir: Path | str, *, eval_version: str, stats: Mapping[str, Any]
) -> dict[str, Path]:
    """Write quality_report.json and screening_sample_size.json beside a build."""

    out_dir = Path(output_dir)
    counts = dict(stats.get("suite_counts") or {})
    certified = dict(stats.get("certified") or {})
    quality = {
        "schema": "eval_quality_report/v1",
        "dataset_id": eval_version,
        "source": "certified",
        "suite_counts": counts,
        "leakage_rejected": int(stats.get("leakage_rejected") or 0),
        "error_count": int(stats.get("error_count") or 0),
        "errors": list(stats.get("errors") or []),
        "certified": certified,
        "warnings": [],
        "recommendations": ["keep strict leakage gates; do not weaken"],
        "version_stamp": stats.get("version_stamp") or build_version_stamp(_STAMP_COMPONENT),
    }
    quality_path = out_dir / "quality_report.json"
    quality_path.write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    # The range is computed for the suite the build actually carries: smoke
    # when present, else the held-out suite (a held-out-only build must not
    # report an empty smoke range as "must generate").
    smoke_n = int(counts.get("smoke") or 0)
    suite = "smoke" if smoke_n else "held_out"
    sidecar = screening_sample_size_sidecar(
        eval_version=eval_version,
        smoke_n=smoke_n,
        suite=suite,
        suite_records=int(counts.get(suite) or 0),
    )
    sidecar_path = out_dir / "screening_sample_size.json"
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"quality_report": quality_path, "screening_sample_size": sidecar_path}


def exact_program_texts(records: Iterable[ExampleRecord]) -> set[str]:
    """Whitespace-normalized program texts (for exact-overlap assertions)."""

    return {norm_text(record.openui) for record in records}


__all__ = [
    "CERTIFIED_CORPUS_DIR",
    "CERTIFIED_CORPUS_ID",
    "CERTIFIED_SCHEMA",
    "CERTIFIED_TRAIN_BUCKET_DIR",
    "CERTIFIED_TRAIN_BUCKET_ID",
    "DEFAULT_SEED",
    "SPLIT_FOR_SUITE",
    "SUITE_FOR_SPLIT",
    "CertifiedPartition",
    "CertifiedSample",
    "bucket_of",
    "certified_eval_candidates",
    "exact_program_texts",
    "materialize_certified_train_bucket",
    "partition_certified_corpus",
    "root_family_index",
    "sample_certified_candidates",
    "screening_sample_size_sidecar",
    "select_stratified",
    "train_bucket_fingerprints",
    "write_certified_eval_sidecars",
]
