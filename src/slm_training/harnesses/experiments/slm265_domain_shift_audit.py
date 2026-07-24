"""Leakage-safe, outcome-blind domain-shift strata for SLM-265."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from slm_training.data.leakage import (
    find_leakage,
    fingerprint_openui,
    fingerprint_openui_structure,
    fingerprint_pair,
    fingerprint_prompt,
)
from slm_training.data.semantic_dedup import record_vectors
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.train_data.diversity import fingerprint_record

SCHEMA = "coverage_gap_manifest/v1"


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _fingerprints(records: Iterable[ExampleRecord]) -> dict[str, set[str]]:
    values = {key: set() for key in ("ids", "split_group_ids", "prompts", "openuis", "structures", "pairs", "design_mds")}
    for record in records:
        values["ids"].add(record.id)
        values["prompts"].add(fingerprint_prompt(record.prompt))
        values["openuis"].add(fingerprint_openui(record.openui))
        values["structures"].add(fingerprint_openui_structure(record.openui))
        values["pairs"].add(fingerprint_pair(record.prompt, record.openui))
        if group := record.meta.get("split_group_id"):
            values["split_group_ids"].add(str(group))
    return values


@dataclass(frozen=True)
class DomainShiftRowV1:
    record_id: str
    suite: str
    composite_distance: float
    prompt_distance: float
    program_distance: float
    stratum: str
    leakage_reasons: tuple[str, ...]
    feature_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "suite": self.suite,
            "composite_distance": self.composite_distance,
            "prompt_distance": self.prompt_distance,
            "program_distance": self.program_distance,
            "stratum": self.stratum,
            "leakage_reasons": list(self.leakage_reasons),
            "feature_hash": self.feature_hash,
        }


def _program_distance(record: ExampleRecord, train_features: set[tuple[str, str, str]]) -> tuple[float, str]:
    feature = fingerprint_record(record)
    key = (feature.canonical_root_id, feature.binding_aware_sketch, feature.topology_sketch)
    return (0.0 if key in train_features else 1.0), _sha(feature.to_dict())


def _strata(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Freeze quantile strata before outcome loading; ties resolve by record id."""
    ordered = sorted(rows, key=lambda row: (row["composite_distance"], row["record_id"]))
    if len(ordered) == 1:
        return {ordered[0]["record_id"]: "near"}
    result: dict[str, str] = {}
    for index, row in enumerate(ordered):
        quantile = index / (len(ordered) - 1)
        result[row["record_id"]] = "near" if quantile <= 0.25 else "mid" if quantile < 0.75 else "far"
    return result


def build_coverage_gap_manifest(
    train_records: Iterable[ExampleRecord], eval_suites: Mapping[str, Iterable[ExampleRecord]]
) -> dict[str, Any]:
    """Return a deterministic manifest without accepting model outcomes or labels."""
    train = list(train_records)
    flattened = [(suite, record) for suite, records in sorted(eval_suites.items()) for record in records]
    train_features = {
        (item.canonical_root_id, item.binding_aware_sketch, item.topology_sketch)
        for item in map(fingerprint_record, train)
    }
    vectors, engine = record_vectors(train + [record for _, record in flattened])
    train_vectors = vectors[: len(train)]
    rows: list[dict[str, Any]] = []
    train_fingerprints = _fingerprints(train)
    for offset, (suite, record) in enumerate(flattened):
        vector = vectors[len(train) + offset]
        prompt_distance = 1.0 if not len(train_vectors) else 1.0 - float(np.max(train_vectors @ vector))
        program_distance, feature_hash = _program_distance(record, train_features)
        rows.append({
            "record_id": record.id, "suite": suite,
            "prompt_distance": round(prompt_distance, 8),
            "program_distance": program_distance,
            "composite_distance": round((prompt_distance + program_distance) / 2.0, 8),
            "leakage_reasons": sorted(find_leakage(record, train_fingerprints)),
            "feature_hash": feature_hash,
        })
    strata = _strata(rows)
    result_rows = [DomainShiftRowV1(stratum=strata[row["record_id"]], **row).to_dict() for row in rows]
    gaps = []
    for suite in sorted(eval_suites):
        suite_rows = [row for row in result_rows if row["suite"] == suite]
        gaps.append({"suite": suite, "n": len(suite_rows), "far_n": sum(row["stratum"] == "far" for row in suite_rows), "leaked_n": sum(bool(row["leakage_reasons"]) for row in suite_rows)})
    payload = {
        "schema": SCHEMA,
        "outcome_firewall": "No generation, evaluator, rating, or success field is accepted by this manifest builder.",
        "feature_views": {"prompt": {"engine": engine, "payload": "semantic_dedup.prompt_plus_normalized_structure"}, "program": {"owner": "train_data.diversity.DiversityFingerprints", "axes": ["canonical_root", "binding", "topology"]}},
        "train_n": len(train), "eval_n": len(flattened), "rows": result_rows,
        "coverage_gaps": gaps,
        "status": "insufficient_power_or_provenance",
        "limitations": ["No durable checkpoint references for frozen generation evaluation.", "Near/mid/far strata are descriptive and do not replace full rico_held ship gates."],
    }
    payload["manifest_hash"] = _sha({key: value for key, value in payload.items() if key != "manifest_hash"})
    return payload
