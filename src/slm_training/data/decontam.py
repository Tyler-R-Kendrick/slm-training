"""Tülu-3-style n-gram decontamination of train candidates against eval data.

The exact/structural fingerprints in :mod:`slm_training.data.leakage` catch
byte- and layout-identical leaks; this pass catches *fuzzy textual* leakage:
a train candidate whose token 8-grams overlap an eval record beyond a
threshold fraction is treated as contaminated and rejected (strict profile)
so the model is never trained on near-copies of held-out material.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from slm_training.dsl.schema import ExampleRecord, load_jsonl

DEFAULT_NGRAM_SIZE = 8
DEFAULT_OVERLAP_THRESHOLD = 0.5

_TOKEN_RE = re.compile(r"[a-z0-9:$._]+")


def _tokens(record: ExampleRecord) -> list[str]:
    text = f"{record.prompt}\n{record.openui}".lower()
    return _TOKEN_RE.findall(text)


def _ngrams(tokens: list[str], n: int) -> Iterable[tuple[str, ...]]:
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def build_eval_ngram_index(
    eval_records: dict[str, list[ExampleRecord]],
    *,
    n: int = DEFAULT_NGRAM_SIZE,
) -> dict[tuple[str, ...], str]:
    """Map every eval n-gram to the (first) suite that contains it."""
    index: dict[tuple[str, ...], str] = {}
    for suite in sorted(eval_records):
        for record in eval_records[suite]:
            for gram in _ngrams(_tokens(record), n):
                index.setdefault(gram, suite)
    return index


def overlap_report(
    record: ExampleRecord,
    index: dict[tuple[str, ...], str],
    *,
    n: int = DEFAULT_NGRAM_SIZE,
) -> dict[str, object]:
    """Fraction of the record's n-grams found in the eval index, by suite."""
    tokens = _tokens(record)
    grams = list(_ngrams(tokens, n))
    if not grams:
        return {"overlap": 0.0, "suite": None, "ngrams": 0}
    matches: dict[str, int] = {}
    matched_total = 0
    for gram in grams:
        suite = index.get(gram)
        if suite is None:
            continue
        matched_total += 1
        matches[suite] = matches.get(suite, 0) + 1
    if not matched_total:
        return {"overlap": 0.0, "suite": None, "ngrams": len(grams)}
    top_suite = max(sorted(matches), key=lambda suite: matches[suite])
    return {
        "overlap": round(matched_total / len(grams), 4),
        "suite": top_suite,
        "ngrams": len(grams),
    }


def apply_ngram_decontam(
    records: list[ExampleRecord],
    eval_records: dict[str, list[ExampleRecord]],
    *,
    n: int = DEFAULT_NGRAM_SIZE,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> tuple[list[ExampleRecord], list[dict[str, object]]]:
    """Split records into (kept, flagged) by eval n-gram overlap."""
    if not eval_records:
        return list(records), []
    index = build_eval_ngram_index(eval_records, n=n)
    if not index:
        return list(records), []
    kept: list[ExampleRecord] = []
    flagged: list[dict[str, object]] = []
    for record in records:
        report = overlap_report(record, index, n=n)
        if float(report["overlap"] or 0.0) >= overlap_threshold:
            flagged.append(
                {
                    "id": record.id,
                    "reason": "ngram_overlap",
                    "overlap": report["overlap"],
                    "suite": report["suite"],
                    "ngram_size": n,
                }
            )
        else:
            kept.append(record)
    return kept, flagged


def load_eval_suites(
    eval_root: Path | None,
    *,
    test_seed_path: Path | None = None,
) -> dict[str, list[ExampleRecord]]:
    """Collect committed eval-suite records plus reserved test seeds.

    Layout walked: ``<eval_root>/<version>/suites/<suite>/records.jsonl``.
    Unreadable files are skipped — decontamination guards data we can read;
    the exact/structural leakage gates remain the fail-closed layer.
    """
    suites: dict[str, list[ExampleRecord]] = {}
    if eval_root is not None and Path(eval_root).is_dir():
        for records_path in sorted(Path(eval_root).glob("*/suites/*/records.jsonl")):
            suite_name = f"{records_path.parent.parent.parent.name}/{records_path.parent.name}"
            try:
                suites[suite_name] = load_jsonl(records_path)
            except (OSError, ValueError):
                continue
    if test_seed_path is not None and Path(test_seed_path).is_file():
        try:
            suites["test_seeds"] = load_jsonl(test_seed_path)
        except (OSError, ValueError):
            pass
    return suites


__all__ = [
    "DEFAULT_NGRAM_SIZE",
    "DEFAULT_OVERLAP_THRESHOLD",
    "apply_ngram_decontam",
    "build_eval_ngram_index",
    "load_eval_suites",
    "overlap_report",
]
