"""SLM-147 SPV1-04: leakage-safe retrieved valid AST prototypes for X22 seeding.

Wiring/fixture harness only. The matrix initializes the Kapur tree-edit diffusion
baseline (X22) from retrieved hard-valid prototypes rather than from the generic
minimal seed, and compares retrieval strategies under matched (CPU-only, no model)
seed-distance and validity diagnostics.

Real end-to-end measurement requires a trained X22 checkpoint, a labeled semantic
corpus, GPU hosts, and AgentV evaluation. No ship claim is made here.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from slm_training.data.leakage import (
    fingerprint_openui_structure,
    fingerprint_prompt,
    norm_text,
)
from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.data.semantic_plan.canonicalize import plan_factor_fingerprints
from slm_training.data.semantic_plan.corpus import build_fixture_plan_corpus
from slm_training.dsl.canonicalize import canonical_fingerprint
from slm_training.dsl.grammar.backends.ast_utils import ast_fingerprint, component_multiset
from slm_training.dsl.parser import validate
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.experiments.ast_sketch_retrieval_factorial import (
    AstTrainingSketchV1,
    build_ast_training_sketch,
)
from slm_training.harnesses.experiments.slm146_semantic_plan_compiler import (
    _render_fixture_ast,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "X22_RETRIEVAL_ID",
    "PrototypeCandidate",
    "PrototypeIndexEntry",
    "RetrievalStrategy",
    "Slm147Arm",
    "Slm147Manifest",
    "Slm147Record",
    "Slm147Report",
    "Slm147Row",
    "ValidPrototypeIndex",
    "ValidPrototypeRetriever",
    "build_manifest",
    "build_prototype_index",
    "render_markdown",
    "run_fixture_matrix",
    "validate_manifest",
]

MATRIX_VERSION = "spv1-04-v1"
MATRIX_SET = "slm147_x22_retrieval"
X22_RETRIEVAL_ID = "slm147-x22-retrieval"

_PLACEHOLDER_RE = re.compile(r":[A-Za-z0-9_.]+")


class RetrievalStrategy(str, Enum):
    MINIMAL = "minimal"
    RANDOM = "random"
    PROMPT_SIMILARITY = "prompt_similarity"
    AST_SKETCH = "ast_sketch"
    SEMANTIC_PLAN = "semantic_plan"
    HYBRID = "hybrid"
    ORACLE_NEAREST = "oracle_nearest"
    RETRIEVAL_AS_CONTEXT = "retrieval_as_context"


@dataclass(frozen=True)
class PrototypeCandidate:
    """One retrieved, leakage-audited, and optionally adapted prototype."""

    source: str
    record_id: str
    score: float
    strategy: str
    canonical_fingerprint: str
    structural_fingerprint: str
    ast_sketch: AstTrainingSketchV1 | None
    plan_factors: dict[str, str] | None
    provenance: str
    leakage_pass: bool
    leakage_reasons: tuple[str, ...]
    remapped_source: str | None = None
    valid: bool = False
    adaptation_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = dict(asdict(self))
        data["ast_sketch"] = self.ast_sketch.to_dict() if self.ast_sketch else None
        data["plan_factors"] = dict(self.plan_factors or {})
        data["leakage_reasons"] = list(self.leakage_reasons)
        data["adaptation_notes"] = list(self.adaptation_notes)
        return data


@dataclass(frozen=True)
class PrototypeIndexEntry:
    """One train-only prototype index entry with precomputed fingerprints."""

    record: ExampleRecord
    source: str
    plan: SemanticPlanV1
    sketch: AstTrainingSketchV1
    plan_factors: dict[str, str]
    components: dict[str, int]
    canonical_fingerprint: str
    structural_fingerprint: str


@dataclass
class ValidPrototypeIndex:
    """Train-only local prototype index with manifest."""

    entries: list[PrototypeIndexEntry]
    manifest: dict[str, Any] = field(default_factory=dict)


class ValidPrototypeRetriever(Protocol):
    """Provider-neutral local prototype-retrieval interface."""

    def retrieve(
        self,
        request: Any,
        plan: SemanticPlanV1 | None,
        pack: Any,
        k: int,
        strategy: RetrievalStrategy,
    ) -> list[PrototypeCandidate]: ...


@dataclass(frozen=True)
class Slm147Arm:
    """One diagnostic arm of the SLM-147 prototype-seeding matrix."""

    arm_id: str
    strategy: RetrievalStrategy
    k: int
    seeds: tuple[int, ...]
    description: str
    promotable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "strategy": self.strategy.value,
            "k": self.k,
            "seeds": list(self.seeds),
            "description": self.description,
            "promotable": self.promotable,
        }


@dataclass(frozen=True)
class Slm147Record:
    """Per-record diagnostics for one arm/seed."""

    record_id: str
    strategy: str
    seed: int
    seed_valid: bool
    seed_to_gold_ratio: float | None
    component_coverage: float
    leakage_pass: bool
    adaptation_valid: bool
    retrieval_score: float | None
    initial_state: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class Slm147Row:
    """Aggregated row for one arm/seed."""

    arm_id: str
    strategy: str
    seed: int
    status: str
    promotable: bool
    n_records: int
    seed_valid_count: int
    mean_seed_to_gold_ratio: float | None
    mean_component_coverage: float
    leakage_pass_count: int
    adaptation_valid_count: int
    mean_retrieval_score: float | None
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class Slm147Manifest:
    """Preregistered manifest for the SLM-147 X22 prototype-retrieval matrix."""

    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = X22_RETRIEVAL_ID
    hypothesis: str = (
        "A leakage-safe retrieved hard-valid AST prototype initializes X22 closer "
        "to an acceptable target than the generic minimal seed. "
        "SemanticPlan-aware and AST-sketch retrieval select more editable prototypes "
        "than surface prompt or random controls."
    )
    falsifier: str = (
        "Retrieved prototypes are no closer to gold than the minimal seed, "
        "simple prompt/component retrieval matches or beats plan/sketch retrieval, "
        "or adaptation/remapping fails closed too often for the strategy to be useful."
    )
    arms: tuple[Slm147Arm, ...] = ()
    claim_class: str = "wiring"
    status: str = "not_run"

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["arms"] = [arm.to_dict() for arm in self.arms]
        return data

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class Slm147Report:
    """Full fixture report for SLM-147."""

    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    manifest: Slm147Manifest
    rows: list[Slm147Row]
    index_manifest: dict[str, Any]
    version_stamp: dict[str, Any] = field(default_factory=dict)
    claim_class: str = "wiring"

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "manifest": self.manifest.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
            "index_manifest": self.index_manifest,
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _derive_prompt(spec: ProgramSpec, source: str) -> str:
    """Derive a simple prompt from the spec AST/source."""
    components = component_multiset(spec.ast)
    parts = [f"Build a {spec.program_family_id} layout"]
    if components:
        parts.append("with " + ", ".join(components.keys()))
    return " ".join(parts)


def _spec_to_record(spec: ProgramSpec, source: str) -> ExampleRecord:
    return ExampleRecord(
        id=spec.id,
        prompt=_derive_prompt(spec, source),
        openui=source,
        placeholders=extract_placeholders(source),
        split=spec.split,
        source=spec.program_family_id,
        meta={
            "split_group_id": spec.split_group_id,
            "lineage_id": spec.lineage_id,
            "program_family_id": spec.program_family_id,
        },
        design_md=None,
    )


def _unique_placeholders(source: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in _PLACEHOLDER_RE.findall(source):
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _remap_placeholders(source: str, target_inventory: Sequence[str]) -> str | None:
    """Hygienically remap prototype placeholders onto the query inventory.

    Returns ``None`` when the query inventory is too short to cover every unique
    prototype placeholder without duplication.
    """
    source_placeholders = _unique_placeholders(source)
    if not source_placeholders:
        return source
    if len(target_inventory) < len(source_placeholders):
        return None
    mapping = {p: target_inventory[i] for i, p in enumerate(source_placeholders)}

    def _replace(match: re.Match[str]) -> str:
        return mapping[match.group(0)]

    return _PLACEHOLDER_RE.sub(_replace, source)


def _canonical_source(source: str) -> str | None:
    try:
        program = validate(source)
        return program.serialized or source.strip()
    except Exception:  # noqa: BLE001
        return None


def _token_ratio(a: str | None, b: str | None) -> float | None:
    if a is None or b is None:
        return None
    return SequenceMatcher(None, a.split(), b.split()).ratio()


def _component_coverage(seed_source: str, gold_source: str) -> float:
    try:
        seed_program = validate(seed_source)
        gold_program = validate(gold_source)
    except Exception:  # noqa: BLE001
        return 0.0
    seed_components = component_multiset(seed_program.root)
    gold_components = component_multiset(gold_program.root)
    if not gold_components:
        return 1.0
    total = sum(max(seed_components.get(k, 0), gold_components.get(k, 0)) for k in set(seed_components) | set(gold_components))
    if not total:
        return 1.0
    overlap = sum(min(seed_components.get(k, 0), gold_components.get(k, 0)) for k in set(seed_components) | set(gold_components))
    return overlap / total


def _minimal_seed(inventory: Sequence[str]) -> str:
    """Canonical minimal valid X22 seed (mirrors TreeEditDiffusionModel._seed_state)."""
    slot = inventory[0] if inventory else ":content.body"
    if not slot.startswith(":"):
        slot = f":{slot}"
    candidates = [
        f'root = Stack([n0], "column")\nn0 = TextContent({json.dumps(slot, ensure_ascii=False)})',
        'root = Stack([], "column")',
    ]
    for text in candidates:
        try:
            validate(text)
            return text
        except Exception:  # noqa: BLE001
            continue
    return candidates[-1]


def _must_exclude(query: ExampleRecord, candidate: ExampleRecord) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if candidate.id == query.id:
        reasons.append("same_record_id")
    query_group = (query.meta or {}).get("split_group_id")
    cand_group = (candidate.meta or {}).get("split_group_id")
    if query_group and cand_group and query_group == cand_group:
        reasons.append("same_split_group")
    if fingerprint_prompt(candidate.prompt) == fingerprint_prompt(query.prompt):
        reasons.append("prompt")
    if fingerprint_openui_structure(candidate.openui) == fingerprint_openui_structure(query.openui):
        reasons.append("structure")
    return bool(reasons), reasons


def _prompt_similarity(query: ExampleRecord, candidate: ExampleRecord) -> float:
    query_tokens = set(norm_text(query.prompt).split())
    cand_tokens = set(norm_text(candidate.prompt).split())
    if not query_tokens or not cand_tokens:
        return 0.0
    return len(query_tokens & cand_tokens) / len(query_tokens | cand_tokens)


def _ast_sketch_score(query: AstTrainingSketchV1, candidate: AstTrainingSketchV1) -> float:
    if query.canonical_fingerprint == candidate.canonical_fingerprint:
        return 1.0
    scores: list[float] = []
    weights: list[float] = []
    if query.topology_hash == candidate.topology_hash:
        scores.append(1.0)
        weights.append(0.5)
    if query.binding_hash == candidate.binding_hash:
        scores.append(1.0)
        weights.append(0.3)
    q_placeholders = set(query.placeholder_signature.split(":")) - {""}
    c_placeholders = set(candidate.placeholder_signature.split(":")) - {""}
    if q_placeholders or c_placeholders:
        union = q_placeholders | c_placeholders
        scores.append(len(q_placeholders & c_placeholders) / len(union) if union else 1.0)
        weights.append(0.2)
    if not scores:
        return 0.0
    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)


def _component_score(query: dict[str, int], candidate: dict[str, int]) -> float:
    if not query and not candidate:
        return 1.0
    keys = set(query) | set(candidate)
    if not keys:
        return 1.0
    total = sum(max(query.get(k, 0), candidate.get(k, 0)) for k in keys)
    if not total:
        return 1.0
    overlap = sum(min(query.get(k, 0), candidate.get(k, 0)) for k in keys)
    return overlap / total


def _plan_factor_score(query: dict[str, str], candidate: dict[str, str]) -> float:
    weights = {
        "exact": 0.35,
        "archetype": 0.25,
        "role_set": 0.20,
        "topology": 0.15,
        "bindings": 0.05,
    }
    matches = {k: (query.get(k) == candidate.get(k) and query.get(k) is not None) for k in weights}
    total = sum(weights.values())
    return sum(weights[k] for k, v in matches.items() if v) / total


def _score_entry(
    query: QueryContext,
    entry: PrototypeIndexEntry,
    strategy: RetrievalStrategy,
) -> float:
    if strategy is RetrievalStrategy.PROMPT_SIMILARITY:
        return _prompt_similarity(query.record, entry.record)
    if strategy is RetrievalStrategy.AST_SKETCH:
        return _ast_sketch_score(query.sketch, entry.sketch)
    if strategy is RetrievalStrategy.SEMANTIC_PLAN:
        return _plan_factor_score(query.plan_factors, entry.plan_factors)
    if strategy is RetrievalStrategy.HYBRID:
        s_prompt = _prompt_similarity(query.record, entry.record)
        s_ast = _ast_sketch_score(query.sketch, entry.sketch)
        s_plan = _plan_factor_score(query.plan_factors, entry.plan_factors)
        return (s_prompt + s_ast + s_plan) / 3.0
    if strategy is RetrievalStrategy.RANDOM:
        return 0.0
    if strategy is RetrievalStrategy.ORACLE_NEAREST:
        return _token_ratio(
            _canonical_source(query.source), _canonical_source(entry.source)
        ) or 0.0
    return 0.0


@dataclass(frozen=True)
class QueryContext:
    """Precomputed query-side features for retrieval scoring."""

    record: ExampleRecord
    source: str
    plan: SemanticPlanV1
    sketch: AstTrainingSketchV1
    plan_factors: dict[str, str]
    components: dict[str, int]
    inventory: list[str]


def _build_query_context(spec: ProgramSpec, plan: SemanticPlanV1) -> QueryContext:
    source = _render_fixture_ast(spec.ast)
    record = _spec_to_record(spec, source)
    program = validate(source)
    sketch = build_ast_training_sketch(source)
    return QueryContext(
        record=record,
        source=source,
        plan=plan,
        sketch=sketch,
        plan_factors=plan_factor_fingerprints(plan),
        components=component_multiset(program.root),
        inventory=[p if p.startswith(":") else f":{p}" for p in extract_placeholders(source)],
    )


def _build_index_entry(spec: ProgramSpec, plan: SemanticPlanV1) -> PrototypeIndexEntry | None:
    source = _render_fixture_ast(spec.ast)
    record = _spec_to_record(spec, source)
    try:
        program = validate(source)
        canonical = canonical_fingerprint(source)
        structural = ast_fingerprint(program.root)
        sketch = build_ast_training_sketch(source)
        factors = plan_factor_fingerprints(plan)
        components = component_multiset(program.root)
    except Exception:  # noqa: BLE001
        return None
    return PrototypeIndexEntry(
        record=record,
        source=source,
        plan=plan,
        sketch=sketch,
        plan_factors=factors,
        components=components,
        canonical_fingerprint=canonical,
        structural_fingerprint=structural,
    )


def build_prototype_index(
    corpus: dict[str, list[tuple[ProgramSpec, SemanticPlanV1]]] | None = None,
) -> ValidPrototypeIndex:
    """Build a train-only prototype index from the fixture plan corpus."""
    if corpus is None:
        corpus = build_fixture_plan_corpus(
            count=64,
            seed=0,
            root_containers=["Stack", "Card"],
            leaf_components=["TextContent", "Button"],
        )
    entries: list[PrototypeIndexEntry] = []
    for spec, plan in corpus.get("train", []):
        entry = _build_index_entry(spec, plan)
        if entry is not None:
            entries.append(entry)
    manifest = {
        "schema": "ValidPrototypeIndexV1",
        "source": "fixture_plan_corpus",
        "split": "train",
        "n_entries": len(entries),
        "corpus_train_count": len(corpus.get("train", [])),
    }
    return ValidPrototypeIndex(entries=entries, manifest=manifest)


def _retrieve_candidates(
    index: ValidPrototypeIndex,
    query: QueryContext,
    strategy: RetrievalStrategy,
    k: int,
    seed: int,
) -> list[PrototypeCandidate]:
    """Retrieve up to *k* candidates for *strategy*, leakage-audited."""
    k = max(1, int(k))
    eligible: list[tuple[PrototypeIndexEntry, float]] = []

    for entry in index.entries:
        excluded, reasons = _must_exclude(query.record, entry.record)
        if excluded:
            continue
        score = _score_entry(query, entry, strategy)
        eligible.append((entry, score))

    if strategy is RetrievalStrategy.RANDOM:
        rng = random.Random(_sha256_text(f"{query.record.id}:{seed}"))
        rng.shuffle(eligible)
        selected = eligible[:k]
    else:
        selected = sorted(
            eligible,
            key=lambda item: (-item[1], item[0].record.id),
        )[:k]

    candidates: list[PrototypeCandidate] = []
    for entry, score in selected:
        excluded, reasons = _must_exclude(query.record, entry.record)
        candidates.append(
            PrototypeCandidate(
                source=entry.source,
                record_id=entry.record.id,
                score=score,
                strategy=strategy.value,
                canonical_fingerprint=entry.canonical_fingerprint,
                structural_fingerprint=entry.structural_fingerprint,
                ast_sketch=entry.sketch,
                plan_factors=dict(entry.plan_factors),
                provenance="train_fixture_retrieval",
                leakage_pass=not excluded,
                leakage_reasons=tuple(reasons),
            )
        )
    return candidates


def _adapt_candidate(
    candidate: PrototypeCandidate,
    query: QueryContext,
) -> PrototypeCandidate:
    """Canonicalize, remap placeholders, and validate a retrieved prototype."""
    notes: list[str] = []
    canonical = _canonical_source(candidate.source)
    if canonical is None:
        notes.append("invalid_prototype_source")
        return PrototypeCandidate(
            **{
                **candidate.__dict__,
                "remapped_source": None,
                "valid": False,
                "adaptation_notes": tuple(notes),
            }
        )
    remapped = _remap_placeholders(candidate.source, query.inventory)
    if remapped is None:
        notes.append("placeholder_remap_failed")
        return PrototypeCandidate(
            **{
                **candidate.__dict__,
                "remapped_source": None,
                "valid": False,
                "adaptation_notes": tuple(notes),
            }
        )
    if _canonical_source(remapped) is None:
        notes.append("remapped_source_invalid")
        return PrototypeCandidate(
            **{
                **candidate.__dict__,
                "remapped_source": remapped,
                "valid": False,
                "adaptation_notes": tuple(notes),
            }
        )
    notes.append("remap_ok")
    return PrototypeCandidate(
        **{
            **candidate.__dict__,
            "remapped_source": remapped,
            "valid": True,
            "adaptation_notes": tuple(notes),
        }
    )


def _build_seed(
    arm: Slm147Arm,
    query: QueryContext,
    index: ValidPrototypeIndex,
    seed: int,
) -> tuple[str, PrototypeCandidate | None, list[str]]:
    """Return the initial X22 seed, the adapted candidate (if any), and notes."""
    notes: list[str] = []
    if arm.strategy is RetrievalStrategy.MINIMAL:
        return _minimal_seed(query.inventory), None, ["minimal_x22_seed"]

    candidates = _retrieve_candidates(index, query, arm.strategy, arm.k, seed)
    if not candidates:
        notes.append(f"no_eligible_{arm.strategy.value}_candidate")
        return _minimal_seed(query.inventory), None, notes

    if arm.strategy is RetrievalStrategy.RETRIEVAL_AS_CONTEXT:
        # Historical control: retrieval is attached as context, not used as seed.
        notes.append(f"retrieval_context_k={len(candidates)}")
        return _minimal_seed(query.inventory), candidates[0], notes

    best = _adapt_candidate(candidates[0], query)
    notes.append(f"strategy={arm.strategy.value} score={best.score:.4f}")
    if best.valid and best.leakage_pass and best.remapped_source:
        return best.remapped_source, best, notes

    notes.extend(["fallback_minimal", *best.adaptation_notes, *(best.leakage_reasons or [])])
    return _minimal_seed(query.inventory), best, notes


def _run_arm_on_record(
    arm: Slm147Arm,
    spec: ProgramSpec,
    plan: SemanticPlanV1,
    index: ValidPrototypeIndex,
    seed: int,
) -> Slm147Record:
    query = _build_query_context(spec, plan)
    gold_source = query.source
    initial_seed, candidate, notes = _build_seed(arm, query, index, seed)

    canonical_seed = _canonical_source(initial_seed)
    seed_valid = canonical_seed is not None
    ratio = _token_ratio(canonical_seed, _canonical_source(gold_source)) if seed_valid else None
    coverage = _component_coverage(initial_seed, gold_source) if seed_valid else 0.0

    retrieval_score = candidate.score if candidate is not None else None
    adaptation_valid = bool(candidate is not None and candidate.valid and candidate.leakage_pass)
    leakage_pass = candidate.leakage_pass if candidate is not None else True

    return Slm147Record(
        record_id=spec.id,
        strategy=arm.strategy.value,
        seed=seed,
        seed_valid=seed_valid,
        seed_to_gold_ratio=ratio,
        component_coverage=coverage,
        leakage_pass=leakage_pass,
        adaptation_valid=adaptation_valid,
        retrieval_score=retrieval_score,
        initial_state=initial_seed,
        notes=notes,
    )


def _aggregate_records(arm: Slm147Arm, seed: int, records: list[Slm147Record]) -> Slm147Row:
    n = len(records)
    if not n:
        return Slm147Row(
            arm_id=arm.arm_id,
            strategy=arm.strategy.value,
            seed=seed,
            status="empty",
            promotable=arm.promotable,
            n_records=0,
            seed_valid_count=0,
            mean_seed_to_gold_ratio=None,
            mean_component_coverage=0.0,
            leakage_pass_count=0,
            adaptation_valid_count=0,
            mean_retrieval_score=None,
            notes=["empty corpus"],
        )

    ratios = [r.seed_to_gold_ratio for r in records if r.seed_to_gold_ratio is not None]
    mean_ratio = sum(ratios) / len(ratios) if ratios else None
    scores = [r.retrieval_score for r in records if r.retrieval_score is not None]
    mean_score = sum(scores) / len(scores) if scores else None

    notes = [
        f"strategy={arm.strategy.value}",
        "k=1",
        "fixture-only: no X22 model trained or decoded",
    ]
    if not arm.promotable:
        notes.append("non-promotable diagnostic arm")

    return Slm147Row(
        arm_id=arm.arm_id,
        strategy=arm.strategy.value,
        seed=seed,
        status="fixture",
        promotable=arm.promotable,
        n_records=n,
        seed_valid_count=sum(1 for r in records if r.seed_valid),
        mean_seed_to_gold_ratio=mean_ratio,
        mean_component_coverage=sum(r.component_coverage for r in records) / n,
        leakage_pass_count=sum(1 for r in records if r.leakage_pass),
        adaptation_valid_count=sum(1 for r in records if r.adaptation_valid),
        mean_retrieval_score=mean_score,
        notes=notes,
    )


def build_manifest() -> Slm147Manifest:
    """Return the default SLM-147 diagnostic matrix manifest."""
    arms = (
        Slm147Arm(
            arm_id="A_minimal_seed",
            strategy=RetrievalStrategy.MINIMAL,
            k=1,
            seeds=(0, 1, 2),
            description="Canonical minimal X22 seed; baseline for search distance.",
        ),
        Slm147Arm(
            arm_id="B_random_prototype",
            strategy=RetrievalStrategy.RANDOM,
            k=1,
            seeds=(0, 1, 2),
            description="Random valid training prototype control.",
        ),
        Slm147Arm(
            arm_id="C_prompt_similarity",
            strategy=RetrievalStrategy.PROMPT_SIMILARITY,
            k=1,
            seeds=(0, 1, 2),
            description="Prototype retrieved by prompt-token overlap.",
        ),
        Slm147Arm(
            arm_id="D_ast_sketch",
            strategy=RetrievalStrategy.AST_SKETCH,
            k=1,
            seeds=(0, 1, 2),
            description="Prototype retrieved by binding-aware AST sketch.",
        ),
        Slm147Arm(
            arm_id="E_semantic_plan",
            strategy=RetrievalStrategy.SEMANTIC_PLAN,
            k=1,
            seeds=(0, 1, 2),
            description="Prototype retrieved by SemanticPlanV1 factor fingerprints.",
        ),
        Slm147Arm(
            arm_id="F_hybrid",
            strategy=RetrievalStrategy.HYBRID,
            k=1,
            seeds=(0, 1, 2),
            description="Weighted hybrid of prompt, AST-sketch, and plan similarity.",
        ),
        Slm147Arm(
            arm_id="G_oracle_nearest",
            strategy=RetrievalStrategy.ORACLE_NEAREST,
            k=1,
            seeds=(0, 1, 2),
            description="Oracle nearest training prototype; diagnostic ceiling only.",
            promotable=False,
        ),
        Slm147Arm(
            arm_id="H_retrieval_as_context",
            strategy=RetrievalStrategy.RETRIEVAL_AS_CONTEXT,
            k=1,
            seeds=(0, 1, 2),
            description="Historical control: retrieval added as context, seed remains minimal.",
        ),
    )
    return Slm147Manifest(arms=arms, claim_class="wiring", status="not_run")


def validate_manifest(manifest: Slm147Manifest) -> list[str]:
    errors: list[str] = []
    if not manifest.arms:
        errors.append("arms must not be empty")
    seen: set[str] = set()
    for arm in manifest.arms:
        if arm.arm_id in seen:
            errors.append(f"duplicate arm_id: {arm.arm_id}")
        seen.add(arm.arm_id)
        if not arm.seeds:
            errors.append(f"{arm.arm_id}: seeds must not be empty")
        if arm.k <= 0:
            errors.append(f"{arm.arm_id}: k must be positive")
    return errors


def run_fixture_matrix(
    corpus: dict[str, list[tuple[ProgramSpec, SemanticPlanV1]]] | None = None,
    *,
    run_id: str = "slm147_fixture",
    output_dir: Path | None = None,
) -> Slm147Report:
    """Run the SLM-147 prototype-seeding diagnostic matrix on the fixture corpus."""
    manifest = build_manifest()
    if corpus is None:
        corpus = build_fixture_plan_corpus(
            count=64,
            seed=0,
            root_containers=["Stack", "Card"],
            leaf_components=["TextContent", "Button"],
        )
    index = build_prototype_index(corpus)
    val_records = corpus.get("val", [])

    rows: list[Slm147Row] = []
    for arm in manifest.arms:
        for seed in arm.seeds:
            per_record: list[Slm147Record] = []
            for spec, plan in val_records:
                per_record.append(_run_arm_on_record(arm, spec, plan, index, seed))
            rows.append(_aggregate_records(arm, seed, per_record))

    report = Slm147Report(
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=X22_RETRIEVAL_ID,
        run_id=run_id,
        status="fixture",
        manifest=manifest,
        rows=rows,
        index_manifest=index.manifest,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm147_x22_retrieval",
        ),
        claim_class="wiring",
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm147_x22_retrieval_report.json")
    return report


def render_markdown(report: Slm147Report) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-147 / SPV1-04: X22 leakage-safe prototype retrieval ({report.run_id})",
        "",
        f"Matrix set: `{report.matrix_set}`",
        "",
        f"Version: `{report.matrix_version}`",
        "",
        f"Status: **{report.status}**",
        "",
        "**Claim class:** wiring / fixture only. No GPU was used, no production "
        "X22 checkpoint was loaded, and no ship-gate claim is made.",
        "",
        "## Hypothesis",
        "",
        report.manifest.hypothesis,
        "",
        "## Falsifier",
        "",
        report.manifest.falsifier,
        "",
        "## Manifest",
        "",
        "| Arm | Strategy | Seeds | Promotable | Description |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arm in report.manifest.arms:
        strategy_value = arm.strategy.value if isinstance(arm.strategy, RetrievalStrategy) else str(arm.strategy)
        lines.append(
            f"| {arm.arm_id} | {strategy_value} | {','.join(map(str, arm.seeds))} | "
            f"{arm.promotable} | {arm.description} |"
        )

    lines.extend(["", "## Results", ""])
    for row in report.rows:
        lines.append(f"### {row.arm_id} (seed={row.seed})")
        lines.append(f"- records: {row.n_records}")
        lines.append(f"- seed valid: {row.seed_valid_count}")
        if row.mean_seed_to_gold_ratio is not None:
            lines.append(f"- mean seed-to-gold token ratio: {row.mean_seed_to_gold_ratio:.3f}")
        lines.append(f"- mean component coverage: {row.mean_component_coverage:.3f}")
        lines.append(f"- leakage pass: {row.leakage_pass_count}")
        lines.append(f"- adaptation valid: {row.adaptation_valid_count}")
        if row.mean_retrieval_score is not None:
            lines.append(f"- mean retrieval score: {row.mean_retrieval_score:.3f}")
        for note in row.notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.extend(
        [
            "## Verdict",
            "",
            "This is a fixture wiring run. It validates that the retrieval index is "
            "train-only, leakage-audited, and that retrieved prototypes can be "
            "hygienically remapped into hard-valid X22 initial states. "
            "Any promotable arm reporting `seed_valid_count < n_records` or a "
            "leakage failure indicates a harness bug. The oracle-nearest arm is "
            "explicitly non-promotable.",
            "",
        ]
    )
    return "\n".join(lines)
