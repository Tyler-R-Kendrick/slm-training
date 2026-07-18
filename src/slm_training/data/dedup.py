"""Fuzzy lexical + semantic-cluster dedup layers (P1a).

Applied after exact ``fingerprint_pair`` dedup. Cross-source collisions resolve
by versioned family priority — never input order.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any

from slm_training.data.leakage import fingerprint_openui_structure, norm_text
from slm_training.data.quality import component_counts
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.train_data.catalog import classify_source_family

# Higher index = higher keep priority when collapsing duplicates.
FAMILY_PRIORITY: tuple[str, ...] = (
    "stress_adversarial",
    "namespace_augment",
    "layout_augment",
    "prompt_paraphrase",
    "self_distilled_repair",
    "self_distilled_success",
    "gold_correction",
    "awwwards_real",
    "rico_real",
    "human_curated",
    "human_feedback",
)
_FAMILY_RANK = {name: i for i, name in enumerate(FAMILY_PRIORITY)}

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "for",
        "in",
        "on",
        "with",
        "from",
        "by",
        "is",
        "are",
        "be",
        "as",
        "at",
        "this",
        "that",
        "it",
        "into",
        "using",
        "use",
        "make",
        "create",
        "build",
        "show",
        "page",
        "ui",
        "layout",
        "screen",
        "component",
        "please",
    }
)
_WORD_RE = re.compile(r"[a-z0-9_]+")
_PLACEHOLDER_RE = re.compile(r":[A-Za-z0-9_.]+")
_SLOT_ARITY_RE = re.compile(
    r"([A-Z][A-Za-z0-9_]*)\s*\(([^)]*)\)",
)


def family_priority(family: str) -> int:
    return _FAMILY_RANK.get(family, -1)


def _record_family(record: ExampleRecord) -> str:
    meta = record.meta or {}
    return str(meta.get("source_family") or classify_source_family(record))


def _keep_key(record: ExampleRecord) -> tuple[int, int, str]:
    """Prefer higher-priority family, then root self, then sorted id."""
    family = _record_family(record)
    root = str((record.meta or {}).get("root_parent_id") or record.id)
    return (-family_priority(family), 0 if record.id == root else 1, record.id)


def char_ngrams(text: str, n: int = 4) -> list[str]:
    cleaned = norm_text(text)
    if len(cleaned) < n:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]


def _hash_to_uint64(payload: str) -> int:
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


@lru_cache(maxsize=8)
def _minhash_coefficients(seed: int, permutations: int) -> tuple[tuple[int, int], ...]:
    """Permutation coefficients depend only on (seed, i); one SHA pass, reused."""
    return tuple(
        (_hash_to_uint64(f"a|{seed}|{i}") | 1, _hash_to_uint64(f"b|{seed}|{i}"))
        for i in range(permutations)
    )


def minhash_signature(
    text: str,
    *,
    n: int = 4,
    permutations: int = 64,
    seed: int = 0,
) -> tuple[int, ...]:
    """Character n-gram MinHash (pure Python, deterministic)."""
    grams = char_ngrams(text, n=n)
    if not grams:
        return tuple(0 for _ in range(permutations))
    gram_hashes = [_hash_to_uint64(g) for g in grams]
    mask = (1 << 64) - 1
    return tuple(
        min(((a * h + b) & mask) for h in gram_hashes)
        for a, b in _minhash_coefficients(seed, permutations)
    )


def jaccard_from_signatures(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / len(a)


def fuzzy_payload(record: ExampleRecord) -> str:
    return norm_text(record.prompt) + "\n" + norm_text(record.openui)


def apply_fuzzy_dedup(
    records: list[ExampleRecord],
    *,
    threshold: float = 0.92,
    permutations: int = 64,
    seed: int = 0,
) -> tuple[list[ExampleRecord], list[dict[str, str]]]:
    """Collapse near-duplicates within the same structure cluster via MinHash."""
    if threshold <= 0:
        return list(records), []
    by_structure: dict[str, list[ExampleRecord]] = {}
    for record in records:
        key = fingerprint_openui_structure(record.openui)
        by_structure.setdefault(key, []).append(record)

    kept_ids: set[str] = set()
    dropped: list[dict[str, str]] = []
    for members in by_structure.values():
        ordered = sorted(members, key=_keep_key)
        signatures: list[tuple[ExampleRecord, tuple[int, ...]]] = [
            (rec, minhash_signature(fuzzy_payload(rec), permutations=permutations, seed=seed))
            for rec in ordered
        ]
        survivors: list[tuple[ExampleRecord, tuple[int, ...]]] = []
        for record, sig in signatures:
            duplicate_of = None
            for kept_rec, kept_sig in survivors:
                if jaccard_from_signatures(sig, kept_sig) >= threshold:
                    duplicate_of = kept_rec.id
                    break
            if duplicate_of is None:
                survivors.append((record, sig))
                kept_ids.add(record.id)
            else:
                dropped.append(
                    {
                        "id": record.id,
                        "duplicate_of": duplicate_of,
                        "source_family": _record_family(record),
                        "reason": "fuzzy_minhash",
                    }
                )
    kept = [r for r in records if r.id in kept_ids]
    return kept, dropped


def prompt_semantic_cluster(prompt: str) -> str:
    """Content-word bag hash (stopword- and template-prefix-stripped)."""
    text = norm_text(prompt).lower()
    # Drop leading template boilerplate like "Create a UI for:".
    text = re.sub(r"^(create|build|make|design|show)\b[^:]*:\s*", "", text)
    words = [
        w
        for w in _WORD_RE.findall(text)
        if w not in _STOPWORDS and len(w) > 1
    ]
    bag = " ".join(sorted(set(words)))
    return hashlib.sha256(bag.encode("utf-8")).hexdigest()[:16]


def binding_pattern_cluster(openui: str) -> str:
    """Multiset of (component → slot arity) with namespaces erased."""
    text = _PLACEHOLDER_RE.sub(":ph", openui or "")
    counts: Counter[str] = Counter()
    for match in _SLOT_ARITY_RE.finditer(text):
        name = match.group(1)
        args = match.group(2)
        arity = 0 if not args.strip() else args.count(",") + 1
        counts[f"{name}:{arity}"] += 1
    if not counts:
        # Fall back to component multiset when regex misses nested forms.
        for name, n in sorted(component_counts(openui).items()):
            counts[f"{name}:?"] = n
    payload = "|".join(f"{k}={v}" for k, v in sorted(counts.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def semantic_cluster_key(record: ExampleRecord) -> tuple[str, str, str]:
    return (
        prompt_semantic_cluster(record.prompt),
        fingerprint_openui_structure(record.openui),
        binding_pattern_cluster(record.openui),
    )


def apply_semantic_cluster_cap(
    records: list[ExampleRecord],
    *,
    max_per_cluster: int = 8,
) -> tuple[list[ExampleRecord], list[dict[str, str]]]:
    """Cap representatives per semantic cluster (root-parent-first, id order)."""
    if not max_per_cluster or max_per_cluster <= 0:
        return list(records), []
    groups: dict[tuple[str, str, str], list[ExampleRecord]] = {}
    for record in records:
        groups.setdefault(semantic_cluster_key(record), []).append(record)

    kept_ids: set[str] = set()
    dropped: list[dict[str, str]] = []
    for key, members in groups.items():
        ordered = sorted(members, key=_keep_key)
        for record in ordered[:max_per_cluster]:
            kept_ids.add(record.id)
        for record in ordered[max_per_cluster:]:
            dropped.append(
                {
                    "id": record.id,
                    "cluster": "|".join(key),
                    "source_family": _record_family(record),
                    "reason": "semantic_cluster_cap",
                }
            )
    kept = [r for r in records if r.id in kept_ids]
    return kept, dropped


def _percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    idx = max(0, min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[idx]


def cluster_exposure_stats(records: list[ExampleRecord]) -> dict[str, Any]:
    """p50/p95 cluster exposure next to parent-exposure stats."""
    counts: Counter[str] = Counter()
    for record in records:
        key = "|".join(semantic_cluster_key(record))
        counts[key] += 1
    values = sorted(counts.values())
    return {
        "unique_clusters": len(counts),
        "records_per_cluster": {
            "max": values[-1] if values else 0,
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
        },
    }


def memorization_diagnostic(
    per_record_nll: list[dict[str, Any]],
    *,
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Fraction of held-out positions/records with NLL < threshold, by family."""
    by_family: dict[str, list[float]] = {}
    for row in per_record_nll:
        family = str(row.get("source_family") or "unknown")
        mean = row.get("mean_nll")
        if mean is None:
            continue
        by_family.setdefault(family, []).append(float(mean))
    out: dict[str, Any] = {}
    for family, values in sorted(by_family.items()):
        near = sum(1 for v in values if v < threshold)
        out[family] = {
            "n_records": len(values),
            "near_certain_fraction": near / len(values) if values else None,
            "threshold": threshold,
        }
    return out
