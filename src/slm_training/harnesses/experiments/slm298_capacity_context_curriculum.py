"""SLM-298's preregistered, local capacity/context/curriculum factorial.

This owner freezes the 2 x 2 x 2 x 3 local matrix before a cell can train.  It
does not invent another trainer: execution uses ``ModelBuildConfig`` and the
shared evaluator, while this module owns only the matched-cell contract and
the AST/reference-derived curriculum view.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignBudget,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
)
from slm_training.data.locked_eval_manifest import record_complexity
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.model_build.data import load_train_records
from slm_training.levers import MAX_RUN_MINUTES
from slm_training.versioning import build_version_stamp

EXPERIMENT_ID = "slm298-capacity-context-curriculum"
CAMPAIGN_ID = "slm298-ap014-local"
SEEDS = (0, 1, 2)
WIDTHS = (32, 64)
CONTEXTS = ("scratch_trainable", "scratch_frozen")
CURRICULA = ("flat", "complexity_ordered")
DERIVED_TRAIN_DIR = Path("outputs/data/train/slm298_verified_symbolonly_v2")
LOCKED_MANIFEST = Path(
    "src/slm_training/resources/data/eval/manifests/abstract_planning_locked_v1.jsonl"
)
TARGET_TOKEN_BUDGET = 5_000


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class FactorialCell:
    d_model: int
    context: str
    curriculum: str
    seed: int

    @property
    def cell_id(self) -> str:
        return f"d{self.d_model}-{self.context}-{self.curriculum}-s{self.seed}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "cell_id": self.cell_id}


@dataclass(frozen=True)
class Slm298Protocol:
    train_manifest_sha256: str
    locked_eval_manifest_sha256: str
    train_record_count: int
    locked_record_ids: tuple[str, ...]
    target_token_budget: int
    cells: tuple[FactorialCell, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "Slm298CapacityContextCurriculumV1",
            "experiment_id": EXPERIMENT_ID,
            "train_manifest_sha256": self.train_manifest_sha256,
            "locked_eval_manifest_sha256": self.locked_eval_manifest_sha256,
            "train_record_count": self.train_record_count,
            "locked_record_ids": list(self.locked_record_ids),
            "target_token_budget": self.target_token_budget,
            "max_wall_minutes_per_cell": MAX_RUN_MINUTES,
            "output_tokenizer": "lexer",
            "grammar_ltr_primary": False,
            "cells": [cell.to_dict() for cell in self.cells],
        }
        return {**payload, "protocol_sha256": _sha(payload)}


def build_cells() -> tuple[FactorialCell, ...]:
    return tuple(
        FactorialCell(d_model, context, curriculum, seed)
        for d_model, context, curriculum, seed in product(
            WIDTHS, CONTEXTS, CURRICULA, SEEDS
        )
    )


def _locked_payload(path: Path) -> tuple[str, tuple[str, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = _sha(
        {
            "schema": payload.get("schema"),
            "rows": payload.get("rows"),
            "metadata": payload.get("metadata"),
        }
    )
    expected = str(payload.get("manifest_sha256") or "")
    if payload.get("schema") != "LockedPromotionManifestV1" or actual != expected:
        raise ValueError("locked promotion manifest digest mismatch")
    ids = tuple(
        str(row["record"]["id"])
        for row in payload.get("rows", ())
        if row.get("partition") == "locked_test"
    )
    if not ids:
        raise ValueError("SLM-298 requires locked_test records")
    return expected, ids


def load_protocol(
    *,
    train_dir: Path,
    locked_manifest: Path = LOCKED_MANIFEST,
    target_token_budget: int = TARGET_TOKEN_BUDGET,
) -> Slm298Protocol:
    """Bind the local certified train snapshot and immutable holdout once."""
    records = load_train_records(train_dir)
    locked_sha, locked_ids = _locked_payload(locked_manifest)
    overlap = {record.id for record in records} & set(locked_ids)
    if overlap:
        raise ValueError("locked evaluation records overlap the SLM-298 training snapshot")
    protocol = Slm298Protocol(
        train_manifest_sha256=_file_sha256(train_dir / "manifest.json"),
        locked_eval_manifest_sha256=locked_sha,
        train_record_count=len(records),
        locked_record_ids=locked_ids,
        target_token_budget=target_token_budget,
        cells=build_cells(),
    )
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: Slm298Protocol) -> None:
    """Fail closed unless every retained cell is exactly matched and triplicated."""
    expected = {
        (width, context, curriculum, seed)
        for width, context, curriculum, seed in product(
            WIDTHS, CONTEXTS, CURRICULA, SEEDS
        )
    }
    observed = {
        (cell.d_model, cell.context, cell.curriculum, cell.seed) for cell in protocol.cells
    }
    if observed != expected or len(protocol.cells) != len(expected):
        raise ValueError("SLM-298 requires exactly 2x2x2 cells with three seeds each")
    if protocol.train_record_count < 1 or not protocol.train_manifest_sha256:
        raise ValueError("SLM-298 requires a pinned nonempty training snapshot")
    if not protocol.locked_record_ids or not protocol.locked_eval_manifest_sha256:
        raise ValueError("SLM-298 requires a pinned locked evaluation manifest")
    if protocol.target_token_budget < 1:
        raise ValueError("SLM-298 requires a positive matched token budget")


def complexity_curriculum_records(records: list[ExampleRecord]) -> list[ExampleRecord]:
    """Attach A/B/C stages from AST/reference complexity, never token length."""
    ranked = sorted(
        records,
        key=lambda record: (
            int(record_complexity(record)["ast_depth"]),
            int(record_complexity(record)["component_cardinality"]),
            int(record_complexity(record)["binder_count"]),
            int(record_complexity(record)["reference_fan_out"]),
            int(record_complexity(record)["reference_diameter"]),
            record.id,
        ),
    )
    if not ranked:
        return []
    result: list[ExampleRecord] = []
    for index, record in enumerate(ranked):
        stage = "ABC"[min(2, (index * 3) // len(ranked))]
        payload = record.to_dict()
        payload["meta"] = {**dict(record.meta or {}), "curriculum": stage}
        result.append(ExampleRecord.from_dict(payload))
    return result


def write_train_view(
    protocol: Slm298Protocol,
    *,
    train_dir: Path,
    output_dir: Path,
    curriculum: str,
) -> Path:
    """Materialize an immutable local train view consumed by the shared trainer."""
    if curriculum not in CURRICULA:
        raise ValueError(f"unsupported curriculum: {curriculum}")
    records = load_train_records(train_dir)
    if len(records) != protocol.train_record_count:
        raise ValueError("training snapshot record count changed")
    if _file_sha256(train_dir / "manifest.json") != protocol.train_manifest_sha256:
        raise ValueError("training snapshot manifest digest changed")
    prepared = records if curriculum == "flat" else complexity_curriculum_records(records)
    destination = output_dir / "train_views" / protocol.to_dict()["protocol_sha256"] / curriculum
    destination.mkdir(parents=True, exist_ok=True)
    records_path = destination / "records.jsonl"
    records_path.write_text(
        "".join(_canonical(record.to_dict()) + "\n" for record in prepared),
        encoding="utf-8",
    )
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "Slm298DerivedTrainViewV1",
                "source_manifest_sha256": protocol.train_manifest_sha256,
                "curriculum": curriculum,
                "record_count": len(prepared),
                "ordering_authority": (
                    "original_snapshot" if curriculum == "flat" else "record_complexity_ast_reference"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def build_campaign(protocol: Slm298Protocol) -> ExperimentCampaignV1:
    """Lock AP-014's factorial endpoints and stop rules before local execution."""
    stamp = build_version_stamp(
        "harness.experiments", "data.locked_eval_manifest", "data.corpus_certification"
    )
    arms = tuple(
        CampaignArmV1(
            arm_id=f"d{width}-{context}-{curriculum}",
            role="control" if (width, context, curriculum) == (32, "scratch_trainable", "flat") else "candidate",
            config_sha256=_sha({"d_model": width, "context": context, "curriculum": curriculum, "output_tokenizer": "lexer", "grammar_ltr_primary": False}),
        )
        for width, context, curriculum in product(WIDTHS, CONTEXTS, CURRICULA)
    )
    return ExperimentCampaignV1(
        campaign_id=CAMPAIGN_ID,
        experiment_id=EXPERIMENT_ID,
        hypothesis="Capacity, local frozen context, and AST/reference complexity curriculum have separable effects on strict meaningful programs.",
        decision="Measure matched local controls before selecting any later plan baseline.",
        endpoints=(
            CampaignEndpointV1(endpoint_id="meaning_v2", metric="binding_aware_meaningful_v2", role="primary", direction="increase", minimum_effect=0.05),
            CampaignEndpointV1(endpoint_id="binder_f1", metric="binder_reference_f1", role="secondary", direction="increase", minimum_effect=0.02),
        ),
        arms=arms,
        seeds=SEEDS,
        budget=CampaignBudget(max_experiments=len(protocol.cells), max_wall_minutes=MAX_RUN_MINUTES),
        stopping_rules=("Reject any timed-out or incomplete cell; no best-seed selection.",),
        controls=(
            CampaignControlV1(control_id="scratch_flat", description="Matched scratch, flat-curriculum control.", kind="quality"),
            CampaignControlV1(control_id="grammar_on_off", description="Shared evaluator reports constrained/native and compiler variants for every cell.", kind="negative"),
        ),
        negative_controls=("grammar_on_off",),
        multiplicity_families=(MultiplicityFamilyV1(family_id="slm298_strict_semantics", hypothesis_ids=("meaning_v2", "binder_f1"), alpha=0.05),),
        promotion_gates=(CampaignGateV1(gate_id="diagnostic_only", endpoint_id="meaning_v2", operator="ge", threshold=-1.0),),
        rollback_gates=(CampaignGateV1(gate_id="incomplete_rejects", endpoint_id="meaning_v2", operator="lt", threshold=2.0),),
        artifact_requirements=(
            ArtifactRequirementV1(kind="version_stamp"),
            ArtifactRequirementV1(kind="seed_result", minimum_count=len(protocol.cells)),
            ArtifactRequirementV1(kind="paired_examples"),
            ArtifactRequirementV1(kind="endpoint_result"),
            ArtifactRequirementV1(kind="agentevals"),
            ArtifactRequirementV1(kind="agentv"),
        ),
        claim_class="diagnostic",
        source_commit=str(stamp["code_commit"]),
        source_dirty=bool(stamp["code_dirty"]),
        author="slm-training",
        locked_eval_manifest_sha256=protocol.locked_eval_manifest_sha256,
        created_at="2026-07-25T00:00:00Z",
    )
