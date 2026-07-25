"""Immutable, all-view-decontaminated promotion holdout manifests.

This is deliberately a data contract, not an evaluator: promotion code receives
only the content digest emitted here and fails closed when that digest changes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from slm_training.data.leakage import find_leakage, fingerprint_openui
from slm_training.data.progspec import ProgramSpec
from slm_training.data.semantic_plan import OpenUISemanticPlanExtractor, plan_factor_fingerprints
from slm_training.data.semantic_dedup import default_threshold, record_vectors
from slm_training.dsl.pack import get_pack
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.experiments.canonical_ast_dedup import (
    build_canonical_ast_fingerprint,
)

SCHEMA = "LockedPromotionManifestV1"
PARTITIONS = ("dev", "locked_test", "agentv_calibration", "human_audit")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _fingerprints(records: Iterable[ExampleRecord]) -> dict[str, set[str]]:
    from slm_training.evals.generalization import _train_fingerprints

    return _train_fingerprints(records)


def record_complexity(record: ExampleRecord) -> dict[str, int | str]:
    """Static, reproducible complexity strata; decoder entropy is not inferred."""
    from slm_training.evals.generalization import _component_combinations, _program_depth

    components = len({name for group in _component_combinations(record.openui, 1) for name in group})
    depth = _program_depth(record.openui)
    placeholders = len(record.placeholders)
    return {
        "ast_depth": depth,
        "component_cardinality": components,
        "binder_count": placeholders,
        "reference_fan_out": placeholders,
        "reference_diameter": max(depth - 1, 0),
        "legal_action_entropy": "requires_decode_trace",
        "source_domain": str((record.meta or {}).get("source_domain") or record.source),
    }


def complexity_stratum(record: ExampleRecord) -> str:
    row = record_complexity(record)
    components = int(row["component_cardinality"])
    return f"d{min(int(row['ast_depth']), 5)}-c{min(components, 6)}-b{min(int(row['binder_count']), 6)}"


# Kept for callers that imported the original private helpers.
_complexity = record_complexity
_stratum = complexity_stratum


def _semantic_plan_fingerprint(record: ExampleRecord) -> str | None:
    """Canonical gold-plan fingerprint, absent only when a source cannot parse."""
    try:
        spec = ProgramSpec.from_openui(
            id=record.id,
            openui=record.openui,
            facts={},
            program_family_id=str((record.meta or {}).get("program_family_id") or record.source),
            lineage_id=str((record.meta or {}).get("lineage_id") or record.id),
            split_group_id=str((record.meta or {}).get("split_group_id") or record.id),
            split=record.split,
        )
        plan = OpenUISemanticPlanExtractor().extract(spec, get_pack("openui"))
        return plan_factor_fingerprints(plan)["exact"]
    except (KeyError, ValueError):
        return None


def _all_view_rejections(
    candidates: list[ExampleRecord], source: list[ExampleRecord]
) -> tuple[list[ExampleRecord], list[dict[str, Any]], dict[str, Any]]:
    source_fps = _fingerprints(source)
    # Historical snapshots contain many exact target copies.  They remain in
    # the prompt-sim scan below, but cannot add AST/plan authority twice.
    structural_source = list(
        {fingerprint_openui(record.openui): record for record in source}.values()
    )
    rejected: list[dict[str, Any]] = []
    exact_clean: list[ExampleRecord] = []
    for record in candidates:
        reasons = [f"exact:{reason}" for reason in find_leakage(record, source_fps)]
        if reasons:
            rejected.append({"id": record.id, "views": reasons})
        else:
            exact_clean.append(record)

    candidate_ast: dict[str, str] = {}
    for record in exact_clean:
        try:
            candidate_ast[record.id] = build_canonical_ast_fingerprint(
                record.openui
            ).canonical_fingerprint
        except ValueError as exc:
            rejected.append({"id": record.id, "views": [f"canonical_ast_invalid:{type(exc).__name__}"]})
    ast_targets = set(candidate_ast.values())
    source_ast: set[str] = set()
    for record in structural_source:
        if not ast_targets - source_ast:
            break
        try:
            fingerprint = build_canonical_ast_fingerprint(record.openui).canonical_fingerprint
        except ValueError:
            continue
        if fingerprint in ast_targets:
            source_ast.add(fingerprint)
    ast_clean = [
        record for record in exact_clean
        if record.id in candidate_ast and candidate_ast[record.id] not in source_ast
    ]
    for record in exact_clean:
        if record.id in candidate_ast and candidate_ast[record.id] in source_ast:
            rejected.append({"id": record.id, "views": ["canonical_ast"]})

    candidate_plan = {record.id: _semantic_plan_fingerprint(record) for record in ast_clean}
    source_plan: set[str] = set()
    plan_targets = {plan for plan in candidate_plan.values() if plan}
    for record in structural_source:
        if not plan_targets - source_plan:
            break
        if (plan := _semantic_plan_fingerprint(record)) in plan_targets:
            source_plan.add(plan)
    kept = []
    for record in ast_clean:
        plan = candidate_plan[record.id]
        if plan is None:
            rejected.append({"id": record.id, "views": ["semantic_plan_invalid"]})
        elif plan in source_plan:
            rejected.append({"id": record.id, "views": ["semantic_plan"]})
        else:
            kept.append(record)

    # Prompt-sim is intentionally computed after hard views so no vector is
    # accepted merely because a lexical fallback happened not to collide.
    if kept and source:
        # ``record_vectors`` deliberately uses dense, deterministic lexical
        # vectors. Audit sources can be thousands of rows, so process bounded
        # chunks instead of materialising an O(source × vocabulary) matrix.
        similarities = np.zeros(len(kept), dtype=np.float64)
        engine: str | None = None
        cutoff: float | None = None
        for start in range(0, len(source), 512):
            chunk = source[start : start + 512]
            vectors, chunk_engine = record_vectors([*chunk, *kept])
            if engine is None:
                engine = chunk_engine
                cutoff = default_threshold(engine)
            elif engine != chunk_engine:
                raise ValueError("prompt-similarity engine changed during audit")
            similarities = np.maximum(
                similarities,
                np.max(vectors[: len(chunk)] @ vectors[len(chunk) :].T, axis=0),
            )
        assert engine is not None and cutoff is not None
        final: list[ExampleRecord] = []
        for record, similarity in zip(kept, similarities, strict=True):
            if float(similarity) >= cutoff:
                rejected.append(
                    {"id": record.id, "views": ["prompt_similarity"], "similarity": round(float(similarity), 6)}
                )
            else:
                final.append(record)
        kept = final
    else:
        engine, cutoff = "not_applicable", None
    return kept, rejected, {"prompt_similarity_engine": engine, "prompt_similarity_threshold": cutoff}


@dataclass(frozen=True)
class LockedManifest:
    rows: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]

    @property
    def sha256(self) -> str:
        return _sha({"schema": SCHEMA, "rows": self.rows, "metadata": self.metadata})

    def payload(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "manifest_sha256": self.sha256, "metadata": self.metadata, "rows": list(self.rows)}


def build_locked_manifest(
    candidates: Iterable[ExampleRecord],
    *,
    source_records: Iterable[ExampleRecord],
    min_locked_records: int = 200,
    partition_size: int = 20,
) -> LockedManifest:
    """Build an immutable manifest or fail before writing an undersized holdout."""
    candidate_rows = sorted(candidates, key=lambda record: record.id)
    source_rows = sorted(source_records, key=lambda record: record.id)
    clean, rejected, verifier = _all_view_rejections(candidate_rows, source_rows)
    grouped: dict[str, list[ExampleRecord]] = {}
    for record in clean:
        grouped.setdefault(complexity_stratum(record), []).append(record)
    ordered = [record for _, rows in sorted(grouped.items()) for record in rows]
    required = min_locked_records + 3 * partition_size
    if len(ordered) < required:
        raise ValueError(
            f"locked manifest needs {required} all-view-clean records; found {len(ordered)}"
        )
    assignments = ([("dev", partition_size), ("agentv_calibration", partition_size), ("human_audit", partition_size)])
    partitions: dict[str, str] = {}
    cursor = 0
    for partition, count in assignments:
        for record in ordered[cursor : cursor + count]:
            partitions[record.id] = partition
        cursor += count
    for record in ordered[cursor:]:
        partitions[record.id] = "locked_test"
    rows = tuple(
        {
            "record": record.to_dict(),
            "partition": partitions[record.id],
            "complexity": record_complexity(record),
        }
        for record in ordered
    )
    counts = Counter(partitions.values())
    metadata = {
        "candidate_sha256": _sha([record.to_dict() for record in candidate_rows]),
        "source_sha256": _sha([record.to_dict() for record in source_rows]),
        "verifier": {**verifier, "hard_views": ["exact", "normalized", "canonical_ast", "semantic_plan", "prompt_similarity"]},
        "partition_counts": dict(sorted(counts.items())),
        "stratum_counts": dict(sorted(Counter(complexity_stratum(record) for record in ordered).items())),
        "locked_labels_selection_forbidden": True,
        "human_audit_is_optional_not_a_promotion_gate": True,
    }
    if counts["locked_test"] < min_locked_records:
        raise ValueError("locked_test partition is below the required minimum")
    return LockedManifest(rows=rows, metadata=metadata)


def write_locked_manifest(path: Path, manifest: LockedManifest) -> str:
    """Write once; a path with different bytes is an immutable-contract error."""
    payload = manifest.payload()
    encoded = _canonical(payload) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"immutable manifest already exists with different content: {path}")
        return manifest.sha256
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")
    return manifest.sha256


def measure_stratified_legal_entropy(
    manifest_payload: dict[str, Any], *, max_records: int = 24
) -> dict[str, Any]:
    """Replay one locked record per observed stratum with exact grammar authority.

    This measures ``log2(|legal actions|)`` from compiler-owned sets, not a
    model posterior.  It is therefore valid without a learned decode rollout
    and makes the measurement recipe reproducible from the immutable manifest.
    """
    from slm_training.dsl.grammar.fastpath.compiler_draft import gold_compiler_decisions
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    if manifest_payload.get("schema") != SCHEMA:
        raise ValueError("unsupported locked manifest schema")
    rows = manifest_payload.get("rows") or []
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("partition") != "locked_test":
            continue
        complexity = row.get("complexity") or {}
        stratum = "-".join(
            f"{key}={complexity.get(key)}"
            for key in ("ast_depth", "component_cardinality", "binder_count")
        )
        groups.setdefault(stratum, []).append(row)
    selected = [
        sorted(group, key=lambda row: str((row.get("record") or {}).get("id")))[0]
        for _, group in sorted(groups.items())
    ][:max_records]
    tokenizer = DSLNativeTokenizer.build()
    measurements: list[dict[str, Any]] = []
    for row in selected:
        record = row["record"]
        decisions = gold_compiler_decisions(
            tokenizer,
            tokenizer.encode(record["openui"], add_special=True),
            slot_contract=list(record.get("placeholders") or ()),
        )
        arities = [len(decision.candidate_ids) for decision in decisions]
        measurements.append(
            {
                "id": record["id"],
                "stratum": row["complexity"],
                "decision_count": len(arities),
                "mean_legal_action_entropy_bits": (
                    sum(math.log2(max(arity, 1)) for arity in arities) / len(arities)
                    if arities
                    else 0.0
                ),
                "max_legal_action_entropy_bits": (
                    max((math.log2(max(arity, 1)) for arity in arities), default=0.0)
                ),
                "max_legal_action_count": max(arities, default=0),
            }
        )
    return {
        "schema": "LockedManifestLegalEntropyV1",
        "manifest_sha256": manifest_payload.get("manifest_sha256"),
        "partition": "locked_test",
        "selection": "one canonical record per observed complexity stratum",
        "tokenizer": "DSLNativeTokenizer",
        "authority": "gold_compiler_decisions",
        "records": measurements,
    }
