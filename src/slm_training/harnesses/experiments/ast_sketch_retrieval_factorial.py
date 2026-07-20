"""SLM-133 EFS3-06 AST-sketch dedup × choice-native retrieval factorial wiring.

This module preregisters the fixed-budget 2×2 factorial and emits a torch-free
fixture plan. Real measurement requires the EFS1 exposure decision (SLM-109),
the corrected choice-native model (SLM-124), and a labeled semantic corpus
(SLM-105). No ship claim, no GPU train, no full factorial eval.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from slm_training.data.leakage import norm_text
from slm_training.dsl.canonicalize import canonical_fingerprint
from slm_training.dsl.grammar.backends.ast_utils import ast_fingerprint, component_multiset
from slm_training.dsl.parser import validate
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.production_codec import ProductionProgram, encode_choices
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.train_data.diversity import _binding_aware_sketch, _topology_sketch
from slm_training.versioning import build_version_stamp

__all__ = [
    "AST_SKETCH_RETRIEVAL_ID",
    "DATA_SAMPLING_ARMS",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "RETRIEVAL_MODES",
    "AstSketchRetrievalArm",
    "AstSketchRetrievalManifest",
    "AstSketchRetrievalReport",
    "AstSketchRetrievalRow",
    "AstTrainingSketchV1",
    "ChoiceRetrievalExemplarV1",
    "DataSampling",
    "RetrievalMode",
    "build_ast_sketch_retrieval_manifest",
    "build_ast_training_sketch",
    "build_choice_exemplar_bank",
    "build_choice_retrieval_exemplar",
    "format_choice_exemplar_context",
    "nearest_choice_exemplars",
    "random_choice_exemplars",
    "render_markdown",
    "run_fixture_matrix",
    "validate_manifest",
]

MATRIX_VERSION = "efs3-06-v1"
MATRIX_SET = "ast-sketch-retrieval"
AST_SKETCH_RETRIEVAL_ID = "efs-ast-sketch-retrieval"
CHOICE_CODEC_VERSION = "choice_native_v1"
SKETCH_VERSION = "ast_sketch_v1"


class DataSampling(str, Enum):
    RAW_STRATIFIED = "raw_stratified"
    AST_SKETCH_BALANCED = "ast_sketch_balanced"


class RetrievalMode(str, Enum):
    NONE = "none"
    CHOICE_EXEMPLAR = "choice_exemplar"
    RANDOM_CHOICE = "random_choice"
    SURFACE_SKELETON = "surface_skeleton"


DATA_SAMPLING_ARMS = (DataSampling.RAW_STRATIFIED, DataSampling.AST_SKETCH_BALANCED)
RETRIEVAL_MODES = (
    RetrievalMode.NONE,
    RetrievalMode.CHOICE_EXEMPLAR,
    RetrievalMode.RANDOM_CHOICE,
    RetrievalMode.SURFACE_SKELETON,
)


def _hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _root_parent_id(record: ExampleRecord) -> str:
    return str(record.meta.get("root_parent_id") or record.id)


def _leakage_group_ids(record: ExampleRecord) -> tuple[str, ...]:
    groups = record.meta.get("leakage_group_ids")
    if isinstance(groups, str):
        return (groups,)
    return tuple(groups or ())


@dataclass(frozen=True)
class AstTrainingSketchV1:
    """Versioned binding-aware AST sketch for training-corpus balancing."""

    schema: str
    sketch_hash: str
    topology_hash: str
    binding_hash: str
    component_topology_hash: str
    slot_contract_shape_hash: str
    placeholder_signature: str
    canonical_fingerprint: str
    structural_fingerprint: str
    sketch_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "sketch_hash": self.sketch_hash,
            "topology_hash": self.topology_hash,
            "binding_hash": self.binding_hash,
            "component_topology_hash": self.component_topology_hash,
            "slot_contract_shape_hash": self.slot_contract_shape_hash,
            "placeholder_signature": self.placeholder_signature,
            "canonical_fingerprint": self.canonical_fingerprint,
            "structural_fingerprint": self.structural_fingerprint,
            "sketch_version": self.sketch_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AstTrainingSketchV1:
        if value.get("schema") != "AstTrainingSketchV1":
            raise ValueError(f"unsupported sketch schema: {value.get('schema')!r}")
        return cls(
            schema=str(value["schema"]),
            sketch_hash=str(value["sketch_hash"]),
            topology_hash=str(value["topology_hash"]),
            binding_hash=str(value["binding_hash"]),
            component_topology_hash=str(value["component_topology_hash"]),
            slot_contract_shape_hash=str(value["slot_contract_shape_hash"]),
            placeholder_signature=str(value.get("placeholder_signature", "")),
            canonical_fingerprint=str(value["canonical_fingerprint"]),
            structural_fingerprint=str(value["structural_fingerprint"]),
            sketch_version=str(value.get("sketch_version", SKETCH_VERSION)),
        )


@dataclass(frozen=True)
class ChoiceRetrievalExemplarV1:
    """Train-only choice-native retrieval exemplar with stable symbolic labels."""

    schema: str
    record_id: str
    root_parent_hash: str
    prompt_hash: str
    normalized_prompt: str
    choice_sequence: tuple[str, ...]
    sequence_hash: str
    codec_version: str
    ast_sketch: AstTrainingSketchV1
    canonical_fingerprint: str
    structural_fingerprint: str
    component_summary: dict[str, Any]
    placeholder_summary: tuple[str, ...]
    source_family: str
    quality_tier: str
    leakage_group_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = dict(asdict(self))
        data["schema"] = self.schema
        data["choice_sequence"] = list(self.choice_sequence)
        data["ast_sketch"] = self.ast_sketch.to_dict()
        data["component_summary"] = dict(self.component_summary)
        data["placeholder_summary"] = list(self.placeholder_summary)
        data["leakage_group_ids"] = list(self.leakage_group_ids)
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ChoiceRetrievalExemplarV1:
        if value.get("schema") != "ChoiceRetrievalExemplarV1":
            raise ValueError(f"unsupported exemplar schema: {value.get('schema')!r}")
        return cls(
            schema=str(value["schema"]),
            record_id=str(value["record_id"]),
            root_parent_hash=str(value["root_parent_hash"]),
            prompt_hash=str(value["prompt_hash"]),
            normalized_prompt=str(value["normalized_prompt"]),
            choice_sequence=tuple(value.get("choice_sequence") or ()),
            sequence_hash=str(value["sequence_hash"]),
            codec_version=str(value.get("codec_version", CHOICE_CODEC_VERSION)),
            ast_sketch=AstTrainingSketchV1.from_dict(value["ast_sketch"]),
            canonical_fingerprint=str(value["canonical_fingerprint"]),
            structural_fingerprint=str(value["structural_fingerprint"]),
            component_summary=dict(value.get("component_summary") or {}),
            placeholder_summary=tuple(value.get("placeholder_summary") or ()),
            source_family=str(value.get("source_family", "fixture")),
            quality_tier=str(value.get("quality_tier", "unknown")),
            leakage_group_ids=tuple(value.get("leakage_group_ids") or ()),
        )


@dataclass(frozen=True)
class AstSketchRetrievalArm:
    data_sampling: DataSampling
    retrieval_mode: RetrievalMode
    seeds: tuple[int, ...]
    k: int
    context_budget: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_sampling": self.data_sampling.value,
            "retrieval_mode": self.retrieval_mode.value,
            "seeds": list(self.seeds),
            "k": self.k,
            "context_budget": self.context_budget,
        }

    @property
    def arm_name(self) -> str:
        return f"{self.data_sampling.value}__{self.retrieval_mode.value}"


@dataclass(frozen=True)
class AstSketchRetrievalRow:
    arm: str
    data_sampling: str
    retrieval_mode: str
    seed: int
    run_id: str
    d_model: int
    status: str
    k: int
    context_budget: int
    checkpoint_uri: str | None = None
    binding_aware_meaningful_v2_rate_strict: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class AstSketchRetrievalManifest:
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = AST_SKETCH_RETRIEVAL_ID
    hypothesis: str = (
        "At a fixed corpus and exposure budget, AST-sketch balancing increases "
        "held-out binding-aware semantic quality and sketch/component/binding "
        "coverage versus a stratified raw sample, and nearest train-only "
        "choice-native exemplars improve semantic selection over no retrieval "
        "and over random/surface-exemplar controls."
    )
    falsifier: str = (
        "After matched exposure and representation-native decoding, sketch "
        "balancing produces no quality or coverage gain, nearest choice "
        "retrieval is equivalent to random retrieval or surface skeletons, "
        "or the interaction is null/negative."
    )
    base_recipe: dict[str, Any] = field(
        default_factory=lambda: {
            "device": "cpu",
            "context_backend": "scratch",
            "denoiser_backend": "scratch",
            "batch_size": 2,
            "learning_rate": 0.0003,
            "mask_pattern": "diffusion",
            "grammar_ltr_primary": False,
            "grammar_constrained": True,
            "parallel_unmask": "adaptive",
            "gen_steps": 8,
            "best_of_n": 1,
            "train_version": "e218_schema_normalized_judge_v5",
            "eval_version": "remediated",
            "eval_suites": "smoke,held_out,adversarial,ood,rico_held",
            "base_token_budget": 50000,
            "max_wall_minutes": 3.0,
            "checkpoint_sync": False,
            "retrieval_k": 4,
            "retrieval_context_budget": 400,
            "representation": "choice",
            "d_model": 128,
        }
    )
    arms: tuple[AstSketchRetrievalArm, ...] = (
        AstSketchRetrievalArm(
            data_sampling=DataSampling.RAW_STRATIFIED,
            retrieval_mode=RetrievalMode.NONE,
            seeds=(0, 1, 2),
            k=4,
            context_budget=400,
        ),
        AstSketchRetrievalArm(
            data_sampling=DataSampling.AST_SKETCH_BALANCED,
            retrieval_mode=RetrievalMode.NONE,
            seeds=(0, 1, 2),
            k=4,
            context_budget=400,
        ),
        AstSketchRetrievalArm(
            data_sampling=DataSampling.RAW_STRATIFIED,
            retrieval_mode=RetrievalMode.CHOICE_EXEMPLAR,
            seeds=(0, 1, 2),
            k=4,
            context_budget=400,
        ),
        AstSketchRetrievalArm(
            data_sampling=DataSampling.AST_SKETCH_BALANCED,
            retrieval_mode=RetrievalMode.CHOICE_EXEMPLAR,
            seeds=(0, 1, 2),
            k=4,
            context_budget=400,
        ),
        AstSketchRetrievalArm(
            data_sampling=DataSampling.RAW_STRATIFIED,
            retrieval_mode=RetrievalMode.RANDOM_CHOICE,
            seeds=(0, 1, 2),
            k=4,
            context_budget=400,
        ),
        AstSketchRetrievalArm(
            data_sampling=DataSampling.RAW_STRATIFIED,
            retrieval_mode=RetrievalMode.SURFACE_SKELETON,
            seeds=(0, 1, 2),
            k=4,
            context_budget=400,
        ),
    )
    primary_metric: str = "binding_aware_meaningful_v2_rate_strict"
    claim_class: str = "frontier"
    status: str = "not_run"
    parent_checkpoint_uri: str | None = None
    checkpoint_bucket: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["base_recipe_hash"] = self.recipe_hash()
        data["arms"] = [a.to_dict() for a in self.arms]
        return data

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    def recipe_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.base_recipe, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class AstSketchRetrievalReport:
    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    manifest: AstSketchRetrievalManifest
    rows: list[AstSketchRetrievalRow]
    version_stamp: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "manifest": self.manifest.to_dict(),
            "rows": [r.to_dict() for r in self.rows],
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


def build_ast_training_sketch(
    source: str,
    *,
    dsl: str | None = None,
    sketch_version: str = SKETCH_VERSION,
) -> AstTrainingSketchV1:
    """Return a versioned binding-aware AST sketch for ``source``.

    The sketch collapses formatting, alpha-equivalent binder names, and free
    literal payloads while preserving component topology, binding roles, and
    placeholder identity.
    """
    program = validate(source, dsl=dsl)
    root = program.root
    topology = _topology_sketch(root)
    binding = _binding_aware_sketch(root)
    placeholders = sorted(set(extract_placeholders(source)))
    components: dict[str, Any] = component_multiset(root)
    canonical = canonical_fingerprint(source, dsl=dsl)
    structural = ast_fingerprint(root)
    payload = {
        "topology": topology,
        "binding": binding,
        "placeholders": placeholders,
        "components": components,
        "canonical_fingerprint": canonical,
    }
    return AstTrainingSketchV1(
        schema="AstTrainingSketchV1",
        sketch_hash=_hash(payload),
        topology_hash=_hash(topology),
        binding_hash=_hash(binding),
        component_topology_hash=_hash(topology),
        slot_contract_shape_hash=_hash(binding),
        placeholder_signature=":".join(placeholders),
        canonical_fingerprint=canonical,
        structural_fingerprint=structural,
        sketch_version=sketch_version,
    )


def build_choice_retrieval_exemplar(
    record: ExampleRecord,
    *,
    codec_version: str = CHOICE_CODEC_VERSION,
) -> ChoiceRetrievalExemplarV1:
    """Build a stable choice-native retrieval exemplar from ``record``."""
    source = record.openui.strip()
    program = validate(source)
    root = program.root

    try:
        encoded: ProductionProgram = encode_choices(source)
        choice_sequence = tuple(encoded.tokens)
    except Exception:
        choice_sequence = ()

    ast_sketch = build_ast_training_sketch(source)
    normalized_prompt = norm_text(record.prompt)
    components = component_multiset(root)
    placeholders = sorted(set(extract_placeholders(source)))

    return ChoiceRetrievalExemplarV1(
        schema="ChoiceRetrievalExemplarV1",
        record_id=record.id,
        root_parent_hash=_sha256_text(_root_parent_id(record)),
        prompt_hash=_sha256_text(normalized_prompt),
        normalized_prompt=normalized_prompt,
        choice_sequence=choice_sequence,
        sequence_hash=_hash(list(choice_sequence)),
        codec_version=codec_version,
        ast_sketch=ast_sketch,
        canonical_fingerprint=canonical_fingerprint(source),
        structural_fingerprint=ast_fingerprint(root),
        component_summary=dict(components),
        placeholder_summary=tuple(placeholders),
        source_family=record.source,
        quality_tier=str(record.meta.get("quality_tier", "unknown")),
        leakage_group_ids=_leakage_group_ids(record),
    )


def build_choice_exemplar_bank(
    records: Iterable[ExampleRecord],
) -> list[ChoiceRetrievalExemplarV1]:
    """Build a train-only choice-native exemplar bank."""
    return [build_choice_retrieval_exemplar(r) for r in records]


def _query_tokens(query: str) -> set[str]:
    return set(norm_text(query).split())


def _overlap_score(query_tokens: set[str], candidate_prompt: str) -> float:
    candidate_tokens = set(norm_text(candidate_prompt).split())
    if not query_tokens or not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(query_tokens | candidate_tokens)


def _must_exclude(
    query: ExampleRecord,
    candidate: ChoiceRetrievalExemplarV1,
    query_root_parent_hash: str,
    query_leakage_groups: set[str],
) -> bool:
    if candidate.record_id == query.id:
        return True
    if candidate.root_parent_hash == query_root_parent_hash:
        return True
    if query_leakage_groups and query_leakage_groups.intersection(candidate.leakage_group_ids):
        return True
    return False


def nearest_choice_exemplars(
    bank: Sequence[ChoiceRetrievalExemplarV1],
    query: ExampleRecord,
    *,
    k: int = 4,
    exclude_same_root: bool = True,
) -> list[tuple[ChoiceRetrievalExemplarV1, float, int]]:
    """Return top-K choice exemplars by deterministic prompt overlap.

    Each result is ``(exemplar, score, rank)``. Tie-breaking is stable by
    sequence hash and record id. Excludes the same record, root parent, and
    declared leakage groups.
    """
    query_tokens = _query_tokens(query.prompt)
    query_root_parent = _sha256_text(_root_parent_id(query))
    query_leakage_groups = set(_leakage_group_ids(query))

    scored: list[tuple[ChoiceRetrievalExemplarV1, float]] = []
    for candidate in bank:
        if exclude_same_root and _must_exclude(
            query, candidate, query_root_parent, query_leakage_groups
        ):
            continue
        score = _overlap_score(query_tokens, candidate.normalized_prompt)
        if score <= 0.0:
            continue
        scored.append((candidate, score))

    scored.sort(
        key=lambda item: (
            -item[1],
            item[0].sequence_hash,
            item[0].record_id,
        )
    )
    return [(ex, score, rank + 1) for rank, (ex, score) in enumerate(scored[: max(0, int(k))])]


def random_choice_exemplars(
    bank: Sequence[ChoiceRetrievalExemplarV1],
    query: ExampleRecord,
    *,
    k: int = 4,
    seed: int = 0,
    exclude_same_root: bool = True,
) -> list[tuple[ChoiceRetrievalExemplarV1, float, int]]:
    """Deterministic random-choice negative control matched in K.

    Random selection is reproducible from the query id hash and arm seed, and
    respects the same leakage exclusions as nearest retrieval.
    """
    import random

    query_root_parent = _sha256_text(_root_parent_id(query))
    query_leakage_groups = set(_leakage_group_ids(query))
    eligible = [
        ex
        for ex in bank
        if not exclude_same_root
        or not _must_exclude(query, ex, query_root_parent, query_leakage_groups)
    ]
    rng = random.Random(_hash([query.id, seed]))
    selected = rng.sample(eligible, min(max(0, int(k)), len(eligible)))
    return [(ex, 0.0, rank + 1) for rank, ex in enumerate(selected)]


def format_choice_exemplar_context(
    hits: Sequence[tuple[ChoiceRetrievalExemplarV1, float, int]],
    *,
    budget: int = 400,
) -> str:
    """Format top-K choice exemplars into a versioned context section."""
    lines = ["---RETRIEVED_CHOICE_EXEMPLARS v1---"]
    used = len(lines[0]) + 1
    for exemplar, score, rank in hits:
        block = (
            f"[rank={rank} score={score:.4f}]\n"
            f"intent: {exemplar.normalized_prompt[:120]}\n"
            f"sketch: {exemplar.ast_sketch.sketch_hash[:16]}\n"
            f"choices: {' '.join(exemplar.choice_sequence[:32])}\n"
        )
        if used + len(block) + 1 > budget and hits.index((exemplar, score, rank)) > 0:
            break
        lines.append(block.rstrip())
        used += len(block) + 1
    return "\n\n".join(lines)


def build_ast_sketch_retrieval_manifest(
    *,
    parent_checkpoint_uri: str | None = None,
    checkpoint_bucket: str | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
    include_controls: bool = True,
) -> AstSketchRetrievalManifest:
    """Return the preregistered SLM-133 factorial manifest."""

    def _arm(data_sampling: DataSampling, retrieval_mode: RetrievalMode) -> AstSketchRetrievalArm:
        return AstSketchRetrievalArm(
            data_sampling=data_sampling,
            retrieval_mode=retrieval_mode,
            seeds=seeds,
            k=4,
            context_budget=400,
        )

    arms = [
        _arm(DataSampling.RAW_STRATIFIED, RetrievalMode.NONE),
        _arm(DataSampling.AST_SKETCH_BALANCED, RetrievalMode.NONE),
        _arm(DataSampling.RAW_STRATIFIED, RetrievalMode.CHOICE_EXEMPLAR),
        _arm(DataSampling.AST_SKETCH_BALANCED, RetrievalMode.CHOICE_EXEMPLAR),
    ]
    if include_controls:
        arms.extend([
            _arm(DataSampling.RAW_STRATIFIED, RetrievalMode.RANDOM_CHOICE),
            _arm(DataSampling.RAW_STRATIFIED, RetrievalMode.SURFACE_SKELETON),
        ])

    if parent_checkpoint_uri is None:
        status = "not_run"
        claim_class = "wiring"
    else:
        status = "frontier_pending_gpu"
        claim_class = "frontier"

    return AstSketchRetrievalManifest(
        arms=tuple(arms),
        parent_checkpoint_uri=parent_checkpoint_uri,
        checkpoint_bucket=checkpoint_bucket,
        status=status,
        claim_class=claim_class,
    )


def validate_manifest(manifest: AstSketchRetrievalManifest) -> list[str]:
    errors: list[str] = []
    if not manifest.arms:
        errors.append("arms must not be empty")
    seen_arms: set[str] = set()
    for arm in manifest.arms:
        if arm.data_sampling not in DATA_SAMPLING_ARMS:
            errors.append(f"unsupported data_sampling: {arm.data_sampling}")
        if arm.retrieval_mode not in RETRIEVAL_MODES:
            errors.append(f"unsupported retrieval_mode: {arm.retrieval_mode}")
        if not arm.seeds:
            errors.append(f"{arm.arm_name}: seeds must not be empty")
        if arm.k <= 0:
            errors.append(f"{arm.arm_name}: k must be positive")
        if arm.context_budget <= 0:
            errors.append(f"{arm.arm_name}: context_budget must be positive")
        if arm.arm_name in seen_arms:
            errors.append(f"duplicate arm: {arm.arm_name}")
        seen_arms.add(arm.arm_name)
    if manifest.claim_class in {"frontier", "ship_candidate"}:
        if not manifest.parent_checkpoint_uri:
            errors.append("frontier/ship_candidate manifest requires parent_checkpoint_uri")
        if not manifest.checkpoint_bucket:
            errors.append("frontier/ship_candidate manifest requires checkpoint_bucket")
    return errors


def run_fixture_matrix(
    manifest: AstSketchRetrievalManifest,
    *,
    run_id: str = "slm133_fixture",
    output_dir: Path | None = None,
) -> AstSketchRetrievalReport:
    """Torch-free fixture that validates the manifest and emits a plan."""
    rows: list[AstSketchRetrievalRow] = []
    d_model = manifest.base_recipe.get("d_model", 128)
    for arm in manifest.arms:
        for seed in arm.seeds:
            notes = [
                f"planned {arm.data_sampling.value} × {arm.retrieval_mode.value} seed {seed}",
                "fixture-only: no model trained",
                f"retrieval k={arm.k} budget={arm.context_budget}",
            ]
            if arm.retrieval_mode in {RetrievalMode.CHOICE_EXEMPLAR, RetrievalMode.RANDOM_CHOICE}:
                notes.append("choice-native exemplar context (symbolic labels)")
            rows.append(
                AstSketchRetrievalRow(
                    arm=arm.arm_name,
                    data_sampling=arm.data_sampling.value,
                    retrieval_mode=arm.retrieval_mode.value,
                    seed=seed,
                    run_id=f"{arm.arm_name}__d{d_model}__s{seed}",
                    d_model=d_model,
                    status="fixture_planned",
                    k=arm.k,
                    context_budget=arm.context_budget,
                    checkpoint_uri=None,
                    binding_aware_meaningful_v2_rate_strict=None,
                    notes=notes,
                )
            )
    report = AstSketchRetrievalReport(
        matrix_set=manifest.matrix_set,
        matrix_version=manifest.matrix_version,
        experiment_id=manifest.experiment_id,
        run_id=run_id,
        status="fixture",
        manifest=manifest,
        rows=rows,
        version_stamp=build_version_stamp("harness.experiments"),
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "ast_sketch_retrieval_report.json")
    return report


def render_markdown(report: AstSketchRetrievalReport) -> str:
    lines = [
        f"# SLM-133 — AST-sketch dedup × choice-native retrieval factorial ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`  ",
        f"Version: `{report.matrix_version}`  ",
        f"Status: **{report.status}**  ",
        "",
        "## Hypothesis",
        "",
        report.manifest.hypothesis,
        "",
        "## Falsifier",
        "",
        report.manifest.falsifier,
        "",
        "## Frozen base recipe (SHA-256)",
        "",
        f"```\n{report.manifest.recipe_hash()}\n```",
        "",
        "## Arms",
        "",
        "| Arm | Data sampling | Retrieval mode | Seeds | K | Context budget |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm in report.manifest.arms:
        lines.append(
            f"| {arm.arm_name} | {arm.data_sampling.value} | {arm.retrieval_mode.value} | "
            f"{','.join(map(str, arm.seeds))} | {arm.k} | {arm.context_budget} |"
        )
    lines.extend(
        ["", "## Rows", "", "| Arm | Seed | d_model | Run id | Status |", "| --- | --- | --- | --- | --- |"]
    )
    for row in report.rows:
        lines.append(
            f"| {row.arm} | {row.seed} | {row.d_model} | `{row.run_id}` | {row.status} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "Fixture/plan only. The actual factorial requires a labeled semantic corpus, "
            "the EFS1 exposure decision (SLM-109), a trained choice-native checkpoint "
            "(SLM-124), and GPU hosts. No data-efficiency or retrieval claim is made "
            "from this artifact.",
            "",
        ]
    )
    return "\n".join(lines)
