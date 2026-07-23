"""Lower staged artifact-graph rows into canonical train-data contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slm_training.data.contract import GenerationRequest, RuntimeSymbol
from slm_training.dsl.language_contract import SymbolicSurfacePolicyV1
from slm_training.dsl.pack import get_pack
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.preference import PreferencePair
from slm_training.harnesses.synthesis_plan import SynthesisPlanV1
from slm_training.harnesses.train_data.artifact_graph import (
    SCHEMA_VERSION as GRAPH_SCHEMA_VERSION,
    ArtifactGraphStore,
    ArtifactNodeV1,
)
from slm_training.harnesses.train_data.integrity import evaluate_integrity


_SPLIT_MAP = {"train": "train", "validation": "held_out", "test": "ood"}


@dataclass(frozen=True)
class StagedMaterialization:
    records: tuple[ExampleRecord, ...]
    preference_pairs: tuple[PreferencePair, ...]
    rejections: tuple[dict[str, Any], ...]
    nodes: dict[str, ArtifactNodeV1]
    store: ArtifactGraphStore


class StagedValidationError(ValueError):
    def __init__(self, reason: str, detail: dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def materialize_staged_graph(
    plan: SynthesisPlanV1,
    *,
    output_dir: Path,
    require_split: str,
) -> StagedMaterialization:
    """Load the plan destination graph and lower QA rows to ExampleRecord."""

    destination = output_dir.as_posix().rstrip("/")
    if destination not in {item.rstrip("/") for item in plan.destinations}:
        raise ValueError(
            "train-data output must be one of the synthesis plan destinations: "
            f"output={destination!r}, destinations={list(plan.destinations)!r}"
        )
    store = ArtifactGraphStore(output_dir)
    nodes = store.load_nodes()
    if not nodes:
        raise ValueError(
            f"synthesis plan destination has no artifact graph nodes: {output_dir}"
        )

    records: list[ExampleRecord] = []
    pairs: list[PreferencePair] = []
    rejections: list[dict[str, Any]] = []
    for node in sorted(nodes.values(), key=lambda item: item.artifact_id):
        if node.artifact_type != "qa_pair" or node.split != require_split:
            continue
        try:
            row_records, row_pairs = _materialize_qa(node, nodes, plan)
        except (KeyError, TypeError, ValueError) as exc:
            detail = {"error": str(exc), "artifact_type": node.artifact_type}
            store.quarantine_node(node, ("materialization_contract",), detail)
            rejections.append(
                {
                    "artifact_id": node.artifact_id,
                    "reason": "materialization_contract",
                    "detail": detail,
                }
            )
            continue
        records.extend(row_records)
        pairs.extend(row_pairs)
    return StagedMaterialization(
        records=tuple(records),
        preference_pairs=tuple(pairs),
        rejections=tuple(rejections),
        nodes=nodes,
        store=store,
    )


def validate_staged_record(
    record: ExampleRecord,
    plan: SynthesisPlanV1,
) -> dict[str, Any]:
    """Run plan-authoritative policy, pack, integrity, and tokenizer checks."""

    symbols = tuple(
        RuntimeSymbol.from_dict(item)
        for item in (record.meta.get("runtime_symbols") or ())
    )
    explicit = {item.surface for item in symbols}
    symbols = (
        *symbols,
        *(
            RuntimeSymbol(surface=slot, role="external_entity")
            for slot in record.placeholders
            if slot not in explicit
        ),
    )
    policy = SymbolicSurfacePolicyV1(
        pack_id=plan.dsl_pack_id,
        policy_version=plan.surface_policy_version,
    )
    try:
        policy_report = policy.require_admitted(record.openui, runtime_symbols=symbols)
        pack = get_pack(plan.dsl_pack_id)
        program = pack.backend.validate(record.openui)
        serialized = pack.backend.serialize(program)
        stream = pack.backend.stream_check(serialized)
        if not stream.complete_ok:
            raise ValueError(
                "pack static check failed: "
                f"errors={list(stream.error_codes)} unresolved={list(stream.unresolved)}"
            )
        canonicalize = pack.require("canonicalize")
        canonical = canonicalize(record.openui)
        if canonicalize(canonical) != canonical:
            raise ValueError("pack canonicalization is not idempotent")
        if canonicalize(serialized) != canonical:
            raise ValueError("pack parse/serialize changed canonical meaning")

        request = GenerationRequest(
            prompt=record.prompt,
            slot_contract=tuple(record.placeholders),
            design_md=record.design_md,
            runtime_symbols=tuple(symbols),
            output_kind=record.target_kind,
            output_category=record.target_category,
        )
        integrity = evaluate_integrity(record, request)
        if not integrity.passed:
            raise ValueError(
                f"synthetic integrity failed: {list(integrity.hard_fail_reasons)}"
            )

        from slm_training.models.dsl_tokenizer import DSLNativeTokenizer, SymbolTable

        tokenizer = DSLNativeTokenizer.build()
        table = SymbolTable.from_placeholders(record.placeholders)
        token_ids = tokenizer.encode(
            record.openui,
            add_special=False,
            table=table,
            placeholders=record.placeholders,
        )
        decoded = tokenizer.decode(token_ids, table=table)
        if canonicalize(decoded) != canonical:
            raise ValueError("DSL tokenizer round trip changed canonical meaning")
    except Exception as exc:
        detail = {"error": str(exc)}
        if "integrity" in locals():
            detail["integrity"] = integrity.to_dict()
        raise StagedValidationError("staged_target_validation", detail) from exc

    return {
        "symbolic_surface": {
            "policy_version": policy_report.policy_version,
            "pack_id": policy_report.pack_id,
            "pack_version": policy_report.pack_version,
            "source_sha256": policy_report.source_sha256,
        },
        "pack": {
            "id": pack.pack_id,
            "backend": pack.backend.info.id,
            "canonical_sha256": _sha(canonical),
        },
        "integrity": integrity.to_dict(),
        "tokenizer": {
            "kind": "dsl_native",
            "version": tokenizer.version,
            "token_count": len(token_ids),
            "round_trip_sha256": _sha(decoded),
        },
    }


def graph_publication(
    materialization: StagedMaterialization,
    *,
    accepted_record_ids: set[str],
) -> dict[str, Any]:
    """Return deterministic graph identity and materialization counts."""

    rows = [
        materialization.nodes[node_id].to_dict()
        for node_id in sorted(materialization.nodes)
    ]
    quarantine = sorted(materialization.store.quarantine.glob("*.json"))
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "path": materialization.store.root.as_posix(),
        "node_count": len(rows),
        "node_ids": [row["artifact_id"] for row in rows],
        "sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "materialized_count": len(materialization.records),
        "accepted_count": sum(
            record.id in accepted_record_ids for record in materialization.records
        ),
        "rejected_count": len(materialization.rejections)
        + len(materialization.records)
        - sum(record.id in accepted_record_ids for record in materialization.records),
        "quarantine_count": len(quarantine),
    }


def dataset_card_markdown(
    plan: SynthesisPlanV1,
    graph: dict[str, Any],
    *,
    version: str,
    version_stamp: dict[str, Any],
) -> str:
    """Render a stable plan-only dataset card beside the manifest."""

    component_versions = version_stamp.get("components") or {}
    rows = [
        "# OpenUI staged training dataset",
        "",
        f"- Dataset version: `{version}`",
        f"- Synthesis plan: `{plan.plan_id}` / `{plan.sha}`",
        f"- Artifact graph: `{graph['schema_version']}` / `{graph['sha256']}`",
        f"- Graph nodes: {graph['node_count']}",
        f"- Materialized: {graph['materialized_count']}",
        f"- Accepted: {graph['accepted_count']}",
        f"- Rejected: {graph['rejected_count']}",
        f"- Quarantined graph rows: {graph['quarantine_count']}",
        "",
        "## Validation",
        "",
        "Every accepted target traversed the canonical ExampleRecord pipeline, "
        "symbolic-surface policy, active DslPack parse/static and canonical "
        "round trips, synthetic-integrity checks, tokenizer round trip, and "
        "the existing quality, leakage, and dedup gates.",
        "",
        "## Version stamp",
        "",
    ]
    rows.extend(
        f"- `{component}`: `{value}`"
        for component, value in sorted(component_versions.items())
    )
    return "\n".join(rows) + "\n"


def _materialize_qa(
    pair: ArtifactNodeV1,
    nodes: dict[str, ArtifactNodeV1],
    plan: SynthesisPlanV1,
) -> tuple[list[ExampleRecord], list[PreferencePair]]:
    payload = pair.payload
    question_id = _required_text(payload, "question_id")
    answer_ids = _required_text_list(payload, "accepted_answer_ids")
    canonical_value = payload.get("canonical_preference_answer_id")
    if canonical_value is not None and (
        not isinstance(canonical_value, str) or not canonical_value
    ):
        raise ValueError(
            "canonical_preference_answer_id must be null or a non-empty string"
        )
    canonical_id = canonical_value
    if canonical_id is not None and canonical_id not in answer_ids:
        raise ValueError("canonical_preference_answer_id is not accepted")
    expected_parents = {question_id, *answer_ids}
    if set(pair.parent_ids) != expected_parents:
        raise ValueError("qa_pair parent_ids do not match typed source IDs")
    question = nodes[question_id]
    answers = [nodes[item] for item in answer_ids]
    if question.artifact_type != "question":
        raise ValueError("question_id does not resolve to a question node")
    if any(item.artifact_type != "answer" for item in answers):
        raise ValueError("accepted_answer_ids must resolve to answer nodes")
    if any(item.split != pair.split for item in (question, *answers)):
        raise ValueError("qa_pair sources do not share one split")

    prompt = _required_text(question.payload, "prompt")
    if _sha(prompt) != question.surface_sha256:
        raise ValueError("question prompt does not match surface_sha256")
    answer_by_id = {item.artifact_id: item for item in answers}
    surfaces: dict[str, str] = {}
    for answer in answers:
        surface = _required_text(answer.payload, "openui")
        if _sha(surface) != answer.surface_sha256:
            raise ValueError(
                f"answer {answer.artifact_id} does not match surface_sha256"
            )
        surfaces[answer.artifact_id] = surface

    records: list[ExampleRecord] = []
    for answer_id in answer_ids:
        answer = answer_by_id[answer_id]
        target_kind = str(answer.payload.get("target_kind") or "document")
        target_category = (
            None
            if answer.payload.get("target_category") is None
            else str(answer.payload["target_category"])
        )
        runtime_symbols = answer.payload.get("runtime_symbols") or []
        if not isinstance(runtime_symbols, list):
            raise TypeError("answer runtime_symbols must be a list")
        graph_sources = {
            "graph_node_id": answer_id,
            "qa_pair_artifact_id": pair.artifact_id,
            "question_artifact_id": question_id,
            "answer_artifact_ids": answer_ids,
            "canonical_answer_artifact_id": canonical_id,
        }
        records.append(
            ExampleRecord(
                id=f"staged-{pair.artifact_id}-{answer_id}",
                prompt=prompt,
                openui=surfaces[answer_id],
                placeholders=extract_placeholders(surfaces[answer_id]),
                split=_SPLIT_MAP[pair.split],
                source="staged",
                meta={
                    "task": "generation",
                    "determinacy": "deterministic",
                    "source_kind": "deterministic",
                    "parent_id": answer_id,
                    "root_parent_id": pair.root_family_id,
                    "program_family_id": pair.root_family_id,
                    "lineage_id": pair.artifact_id,
                    "split_group_id": pair.split_group_id,
                    "runtime_symbols": runtime_symbols,
                    "provenance": {
                        "schema_version": GRAPH_SCHEMA_VERSION,
                        "synthesis_plan_id": plan.plan_id,
                        "synthesis_plan_sha256": plan.sha,
                        **graph_sources,
                    },
                    "staged_sources": graph_sources,
                    "synthesis_plan": {"plan_id": plan.plan_id, "sha256": plan.sha},
                },
                design_md=(
                    None
                    if answer.payload.get("design_md") is None
                    else str(answer.payload["design_md"])
                ),
                target_kind=target_kind,  # type: ignore[arg-type]
                target_category=target_category,
            )
        )
    record_ids = [record.id for record in records]
    preference_pairs: list[PreferencePair] = []
    if canonical_id is not None:
        canonical_record = next(
            record
            for record in records
            if record.meta["staged_sources"]["graph_node_id"] == canonical_id
        )
        preference_pairs = [
            PreferencePair(
                prompt=prompt,
                chosen=surfaces[canonical_id],
                rejected=surfaces[item],
                design_md=canonical_record.design_md,
                chosen_score=1.0,
                rejected_score=0.75,
                meta={
                    "pair_corpus": "staged_canonical_preference",
                    "record_ids": record_ids,
                    "equivalent_accepted": True,
                    "synthesis_plan_sha256": plan.sha,
                    "qa_pair_artifact_id": pair.artifact_id,
                    "question_artifact_id": question_id,
                    "answer_artifact_ids": answer_ids,
                    "canonical_answer_artifact_id": canonical_id,
                },
            )
            for item in answer_ids
            if item != canonical_id
        ]
    return records, preference_pairs


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_text_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{key} must be a non-empty unique string list")
    return list(value)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


__all__ = [
    "StagedMaterialization",
    "StagedValidationError",
    "dataset_card_markdown",
    "graph_publication",
    "materialize_staged_graph",
    "validate_staged_record",
]
