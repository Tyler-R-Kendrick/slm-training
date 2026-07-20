"""SLM-148 SPV1-05: plan-conditioned X22 × conflict-slice matched campaign.

Wiring/fixture harness only. The campaign registers the staged seed × recovery
factorial for X22 valid-state generation, exercises plan-conditioned seed
construction (frequency, learned, gold, retrieved), and applies the canonical
conflict-slice repair policies from SLM-113 on a synthetic corpus for
distance/validity/recovery diagnostics.

Real end-to-end measurement requires a trained X22 checkpoint, a labeled
semantic corpus, GPU hosts, and AgentV evaluation. No ship-gate claim is made
here.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.progspec.semantic_plan import (
    PlanArchetype,
    PlanBinding,
    PlanIdentity,
    PlanSymbol,
    PlanTopology,
    RoleSlot,
    SemanticPlanV1,
)
from slm_training.data.semantic_plan.compiler import (
    OpenUISemanticPlanCompiler,
    PlanSeedResult,
)
from slm_training.data.semantic_plan.corpus import build_fixture_plan_corpus
from slm_training.data.semantic_plan.oracle import PlanOracleSubstitutor
from slm_training.dsl.grammar.backends.ast_utils import component_multiset
from slm_training.dsl.pack import get_pack
from slm_training.dsl.parser import validate
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.experiments.conflict_slice_repair import (
    ConflictSliceV1,
    RepairPolicyName,
    RepairTrace,
    TopologyNode,
    apply_repair_policy,
)
from slm_training.harnesses.experiments.slm147_x22_retrieval import (
    RetrievalStrategy,
    Slm147Arm,
    ValidPrototypeIndex,
    _build_query_context,
    _build_seed,
    _minimal_seed,
    build_prototype_index,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "X22_CONFLICT_CAMPAIGN_ID",
    "SeedStrategy",
    "Slm148Manifest",
    "Slm148Record",
    "Slm148RecoveryArm",
    "Slm148Report",
    "Slm148Row",
    "Slm148SearchConfig",
    "Slm148SeedArm",
    "build_manifest",
    "render_markdown",
    "run_fixture_matrix",
    "validate_manifest",
]

MATRIX_VERSION = "spv1-05-v1"
MATRIX_SET = "slm148_x22_conflict_campaign"
X22_CONFLICT_CAMPAIGN_ID = "slm148-x22-conflict-campaign"

_CONTENT_PROP = {
    "TextContent": "text",
    "Button": "label",
    "Input": "placeholder",
    "CardHeader": "title",
    "Card": "children",
    "Stack": "children",
}


class SeedStrategy(str, Enum):
    """Plan/retrieval source for the initial X22 seed."""

    MINIMAL = "minimal"
    FREQUENCY_PRIOR = "frequency_prior"
    LEARNED_ARCHETYPE_ROLE_SET = "learned_archetype_role_set"
    LEARNED_FULL_PLAN = "learned_full_plan"
    GOLD_FACTOR = "gold_factor"
    GOLD_PLAN = "gold_plan"
    RETRIEVED_PROTOTYPE = "retrieved_prototype"
    PLAN_RERANKED_RETRIEVAL = "plan_reranked_retrieval"


@dataclass(frozen=True)
class Slm148SeedArm:
    """One seed-strategy arm in the staged factorial."""

    arm_id: str
    strategy: SeedStrategy
    seeds: tuple[int, ...]
    description: str
    promotable: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["strategy"] = self.strategy.value
        return data


@dataclass(frozen=True)
class Slm148RecoveryArm:
    """One recovery-policy arm crossed with surviving seed strategies."""

    arm_id: str
    policy: RepairPolicyName
    description: str
    diagnostic: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["policy"] = self.policy
        return data


@dataclass(frozen=True)
class Slm148SearchConfig:
    """Matched search budget placeholders from SLM-111."""

    beam_width: int = 4
    max_depth: int = 8
    equal_forward_budget: int | None = 64
    equal_wall_ms: int | None = None
    notes: str = "fixture-only: no live X22 beam search is executed"

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class Slm148Record:
    """Per-record diagnostics for one arm/seed."""

    record_id: str
    seed_strategy: str
    recovery_policy: str
    stage: str
    seed: int
    plan_source: str
    seed_source: str
    seed_valid: bool
    seed_to_gold_ratio: float | None
    component_coverage: float
    conflict_id: str
    recovered: bool
    remasked_nodes: int
    preserved_nodes: int
    forwards: int
    verifier_calls: int
    repeated_conflict: bool
    protected_mutations: int
    completeness_class: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class Slm148Row:
    """Aggregated row for one arm/seed."""

    arm_id: str
    seed_strategy: str
    recovery_policy: str
    stage: str
    seed: int
    promotable: bool
    n_records: int
    seed_valid_count: int
    mean_seed_to_gold_ratio: float | None
    mean_component_coverage: float
    recovery_rate: float
    mean_remasked_nodes: float
    mean_preserved_nodes: float
    mean_forwards: float
    mean_verifier_calls: float
    repeated_conflict_rate: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(frozen=True)
class Slm148Manifest:
    """Preregistered manifest for the SLM-148 campaign."""

    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = X22_CONFLICT_CAMPAIGN_ID
    hypothesis: str = (
        "Plan-conditioned initialization (gold or learned) and conflict-localized "
        "recovery together lift X22 closer to acceptable programs than minimal seed "
        "or coarse remask controls, under matched seed/edit/search/verifier budgets."
    )
    falsifier: str = (
        "No seed strategy reduces seed-to-gold distance, no recovery policy improves "
        "recovery rate while preserving more correct structure than full remask, or "
        "plan features silently alter legal candidate membership."
    )
    seed_arms: tuple[Slm148SeedArm, ...] = ()
    recovery_arms: tuple[Slm148RecoveryArm, ...] = ()
    search_config: Slm148SearchConfig = field(default_factory=Slm148SearchConfig)
    claim_class: str = "wiring"
    status: str = "not_run"

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["seed_arms"] = [arm.to_dict() for arm in self.seed_arms]
        data["recovery_arms"] = [arm.to_dict() for arm in self.recovery_arms]
        data["search_config"] = self.search_config.to_dict()
        return data

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class Slm148Report:
    """Full fixture report for SLM-148."""

    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    manifest: Slm148Manifest
    rows: list[Slm148Row]
    survivors: list[str]
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
            "survivors": self.survivors,
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


def _token_ratio(a: str | None, b: str | None) -> float | None:
    if a is None or b is None:
        return None
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a.split(), b.split()).ratio()


def _canonical_source(source: str) -> str | None:
    try:
        program = validate(source)
        return program.serialized or source.strip()
    except Exception:  # noqa: BLE001
        return None


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
    keys = set(seed_components) | set(gold_components)
    total = sum(max(seed_components.get(k, 0), gold_components.get(k, 0)) for k in keys)
    if not total:
        return 1.0
    overlap = sum(min(seed_components.get(k, 0), gold_components.get(k, 0)) for k in keys)
    return overlap / total


def _derive_prompt(spec: ProgramSpec, source: str) -> str:
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


def _role_to_family(train_records: list[tuple[ProgramSpec, SemanticPlanV1]]) -> dict[str, str]:
    """Map role ids to their most common component family in the train split."""
    counts: dict[str, Counter[str]] = {}
    for _spec, plan in train_records:
        for slot in plan.role_slots:
            counts.setdefault(slot.role_id, Counter())[slot.component_family or "Stack"] += 1
    return {rid: counter.most_common(1)[0][0] for rid, counter in counts.items()}


def _build_vocabs(
    corpus: dict[str, list[tuple[ProgramSpec, SemanticPlanV1]]],
) -> tuple[dict[str, int], dict[str, int], dict[str, int], int]:
    families: set[str] = set()
    roles: set[str] = set()
    archetypes: set[str] = set()
    max_roles = 0
    for split in corpus.values():
        for _spec, plan in split:
            if plan.archetype.id:
                archetypes.add(plan.archetype.id)
            role_ids = [slot.role_id for slot in plan.role_slots]
            roles.update(role_ids)
            max_roles = max(max_roles, len(role_ids))
            for slot in plan.role_slots:
                if slot.component_family:
                    families.add(slot.component_family)
    family_vocab = {f: i for i, f in enumerate(sorted(families))}
    role_vocab = {r: i for i, r in enumerate(sorted(roles))}
    archetype_vocab = {a: i for i, a in enumerate(sorted(archetypes))}
    return family_vocab, role_vocab, archetype_vocab, max_roles


def _plan_training_examples(
    split: list[tuple[ProgramSpec, SemanticPlanV1]],
    family_vocab: dict[str, int],
    role_vocab: dict[str, int],
    archetype_vocab: dict[str, int],
    max_len: int,
) -> list[Any]:
    """Build PlanTrainingExamples for the fixture predictor."""
    # Deferred import keeps this module torch-free at top level.
    from slm_training.models.semantic_plan_predictor import (
        PlanTrainingExample,
        build_role_set_target,
        featurize_program_spec,
    )
    import torch

    examples: list[Any] = []
    for spec, plan in split:
        if plan.archetype.id is None:
            continue
        archetype_label = archetype_vocab[plan.archetype.id]
        role_ids = [slot.role_id for slot in plan.role_slots]
        features = featurize_program_spec(spec, family_vocab)
        role_mask = build_role_set_target(role_ids, role_vocab, len(role_vocab))
        sorted_roles = sorted(
            {role_vocab[r] for r in role_ids if r in role_vocab},
            key=lambda i: i,
        )
        padded = (sorted_roles + [-1] * max_len)[:max_len]
        serialized = torch.tensor(padded, dtype=torch.long)
        examples.append(
            PlanTrainingExample(
                example_id=spec.id,
                input_features=features,
                archetype_label=archetype_label,
                role_set_mask=role_mask,
                serialized_roles=serialized,
                source_plan=plan,
                program_spec=spec,
            )
        )
    return examples


def _train_predictor_bundle(
    corpus: dict[str, list[tuple[ProgramSpec, SemanticPlanV1]]],
    *,
    epochs: int = 20,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Train the SLM-144 fixture predictor on the corpus."""
    from slm_training.models.semantic_plan_predictor import train_fixture_predictor

    family_vocab, role_vocab, archetype_vocab, max_roles = _build_vocabs(corpus)
    max_len = max(1, max_roles)
    train_examples = _plan_training_examples(
        corpus["train"], family_vocab, role_vocab, archetype_vocab, max_len
    )
    val_examples = _plan_training_examples(
        corpus["val"], family_vocab, role_vocab, archetype_vocab, max_len
    )
    bundle = train_fixture_predictor(
        train_examples,
        val_examples,
        epochs=epochs,
        batch_size=batch_size,
        seed=0,
        device="cpu",
    )
    bundle["family_vocab"] = family_vocab
    bundle["role_vocab"] = role_vocab
    bundle["archetype_vocab"] = archetype_vocab
    return bundle


def _predict_role_indices(bundle: dict[str, Any], spec: ProgramSpec) -> list[int]:
    """Return predicted role-set indices for ``spec``."""
    from slm_training.models.semantic_plan_predictor import (
        featurize_program_spec,
        predict_role_set_from_logits,
    )
    import torch

    features = featurize_program_spec(spec, bundle["family_vocab"]).unsqueeze(0)
    role_head = bundle["role_set_head"]
    role_head.eval()
    with torch.no_grad():
        logits = role_head(features)
    return predict_role_set_from_logits(logits[0], role_head.blank_role)


def _predict_archetype_id(bundle: dict[str, Any], spec: ProgramSpec) -> str | None:
    """Return predicted archetype id for ``spec``."""
    from slm_training.models.semantic_plan_predictor import featurize_program_spec
    import torch

    inverse = {v: k for k, v in bundle["archetype_vocab"].items()}
    features = featurize_program_spec(spec, bundle["family_vocab"]).unsqueeze(0)
    head = bundle["archetype_head"]
    head.eval()
    with torch.no_grad():
        idx = int(head(features).argmax(dim=-1)[0])
    return inverse.get(idx)


def _predict_serialized_role_indices(bundle: dict[str, Any], spec: ProgramSpec) -> list[int]:
    """Return serialized-inventory role order predicted for ``spec``."""
    from slm_training.models.semantic_plan_predictor import (
        featurize_program_spec,
        predict_serialized_inventory,
    )
    import torch

    features = featurize_program_spec(spec, bundle["family_vocab"]).unsqueeze(0)
    head = bundle["serialized_inventory_head"]
    head.eval()
    with torch.no_grad():
        logits = head(features)
    return predict_serialized_inventory(logits[0])


def _build_predicted_base_plan(
    spec: ProgramSpec,
    bundle: dict[str, Any],
    role_to_family: dict[str, str],
) -> SemanticPlanV1:
    """Build a predicted plan from archetype + learned role set."""
    inverse_role = {v: k for k, v in bundle["role_vocab"].items()}
    archetype_id = _predict_archetype_id(bundle, spec)
    role_indices = _predict_role_indices(bundle, spec)
    role_ids = [inverse_role[i] for i in role_indices if i in inverse_role]

    identity = PlanIdentity(
        pack_id="openui",
        contract_hash=spec.contract_id,
        source_program_fingerprint=None,
        prompt_context_hash=None,
        provenance="predicted",
    )
    archetype = PlanArchetype(id=archetype_id or "unknown", confidence=0.7)
    slots = tuple(
        RoleSlot(
            role_id=rid,
            component_family=role_to_family.get(rid, "Stack"),
        )
        for rid in role_ids
    )
    return SemanticPlanV1(
        identity=identity,
        archetype=archetype,
        role_slots=slots,
        topology=PlanTopology(),
        symbols=(),
        bindings=(),
    )


def _build_full_learned_plan(
    spec: ProgramSpec,
    bundle: dict[str, Any],
    role_to_family: dict[str, str],
) -> SemanticPlanV1:
    """Build a fully learned plan by chaining predicted roles and inventing symbols."""
    base = _build_predicted_base_plan(spec, bundle, role_to_family)
    role_ids = [slot.role_id for slot in base.role_slots]
    if not role_ids:
        return base

    edges: list[dict[str, str]] = []
    for i in range(1, len(role_ids)):
        edges.append(
            {
                "parent_role_id": role_ids[i - 1],
                "child_role_id": role_ids[i],
                "relation": "contains",
            }
        )

    symbols: list[PlanSymbol] = []
    bindings: list[PlanBinding] = []
    for rid in role_ids:
        family = role_to_family.get(rid, "Stack")
        prop = _CONTENT_PROP.get(family)
        if prop and prop != "children":
            sym_id = f"sym_{rid}"
            symbols.append(
                PlanSymbol(
                    symbol_id=sym_id,
                    semantic_role=prop,
                    allowed_pointer_targets=None,
                )
            )
            bindings.append(
                PlanBinding(
                    role_slot_id=rid,
                    candidate_symbols=(sym_id,),
                    placeholder_fallback=True,
                )
            )

    return base.model_copy(
        update={
            "topology": PlanTopology(parent_relation_candidates=tuple(edges) or None),
            "symbols": tuple(symbols),
            "bindings": tuple(bindings),
        }
    )


def _build_frequency_prior_plan(
    train_records: list[tuple[ProgramSpec, SemanticPlanV1]],
    role_to_family: dict[str, str],
) -> SemanticPlanV1:
    """Build a deterministic frequency/archetype prior plan."""
    archetype_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    for _spec, plan in train_records:
        if plan.archetype.id:
            archetype_counts[plan.archetype.id] += 1
        for slot in plan.role_slots:
            role_counts[slot.role_id] += 1

    archetype_id = archetype_counts.most_common(1)[0][0] if archetype_counts else "stack"
    roles_by_freq = [rid for rid, _ in role_counts.most_common()]

    identity = PlanIdentity(
        pack_id="openui",
        contract_hash=None,
        source_program_fingerprint=None,
        prompt_context_hash=None,
        provenance="merged",
    )
    slots = tuple(
        RoleSlot(
            role_id=rid,
            component_family=role_to_family.get(rid, "Stack"),
        )
        for rid in roles_by_freq
    )
    edges: list[dict[str, str]] = []
    for i in range(1, len(roles_by_freq)):
        edges.append(
            {
                "parent_role_id": roles_by_freq[i - 1],
                "child_role_id": roles_by_freq[i],
                "relation": "contains",
            }
        )

    symbols: list[PlanSymbol] = []
    bindings: list[PlanBinding] = []
    for rid in roles_by_freq:
        family = role_to_family.get(rid, "Stack")
        prop = _CONTENT_PROP.get(family)
        if prop and prop != "children":
            sym_id = f"sym_{rid}"
            symbols.append(PlanSymbol(symbol_id=sym_id, semantic_role=prop))
            bindings.append(
                PlanBinding(
                    role_slot_id=rid,
                    candidate_symbols=(sym_id,),
                    placeholder_fallback=True,
                )
            )

    return SemanticPlanV1(
        identity=identity,
        archetype=PlanArchetype(id=archetype_id, confidence=0.5),
        role_slots=slots,
        topology=PlanTopology(parent_relation_candidates=tuple(edges) or None),
        symbols=tuple(symbols),
        bindings=tuple(bindings),
    )


def _compile_seed(
    plan: SemanticPlanV1,
    pack: Any,
    *,
    honesty_mode: str = "production",
) -> PlanSeedResult:
    compiler = OpenUISemanticPlanCompiler(honesty_mode=honesty_mode)
    return compiler.build_valid_seed(None, plan, pack)


def _build_seed_for_arm(
    arm: Slm148SeedArm,
    spec: ProgramSpec,
    gold_plan: SemanticPlanV1,
    pack: Any,
    index: ValidPrototypeIndex,
    bundle: dict[str, Any],
    role_to_family: dict[str, str],
    seed: int,
) -> tuple[str, str, list[str]]:
    """Return (seed_source, plan_source, notes) for one seed arm."""
    inventory = [p if p.startswith(":") else f":{p}" for p in extract_placeholders(_render_gold_source(spec))]

    if arm.strategy is SeedStrategy.MINIMAL:
        return _minimal_seed(inventory), "none", ["minimal_x22_seed"]

    if arm.strategy is SeedStrategy.FREQUENCY_PRIOR:
        plan = _build_frequency_prior_plan(bundle["train_records"], role_to_family)
        result = _compile_seed(plan, pack)
        if result.ok and result.seed:
            return result.seed, "frequency_prior", ["frequency_prior_plan_seed"]
        return _minimal_seed(inventory), "frequency_prior", ["frequency_plan_failed", result.reason or "unknown"]

    if arm.strategy is SeedStrategy.LEARNED_ARCHETYPE_ROLE_SET:
        predicted_base = _build_predicted_base_plan(spec, bundle, role_to_family)
        subst = PlanOracleSubstitutor(
            plan_source="predicted",
            oracle_factors=("archetype", "roles"),
            use_mode="seed",
            honesty_mode="production",
        )
        plan = subst.apply(gold_plan, predicted_base)
        result = _compile_seed(plan, pack)
        if result.ok and result.seed:
            return result.seed, "predicted", ["learned_archetype_role_set_seed"]
        return _minimal_seed(inventory), "predicted", ["learned_plan_failed", result.reason or "unknown"]

    if arm.strategy is SeedStrategy.LEARNED_FULL_PLAN:
        plan = _build_full_learned_plan(spec, bundle, role_to_family)
        result = _compile_seed(plan, pack)
        if result.ok and result.seed:
            return result.seed, "predicted", ["learned_full_plan_seed"]
        return _minimal_seed(inventory), "predicted", ["learned_full_plan_failed", result.reason or "unknown"]

    if arm.strategy is SeedStrategy.GOLD_FACTOR:
        predicted_base = _build_predicted_base_plan(spec, bundle, role_to_family)
        subst = PlanOracleSubstitutor(
            plan_source="gold",
            oracle_factors=("bindings",),
            use_mode="seed",
            honesty_mode="oracle_diagnostic",
        )
        plan = subst.apply(predicted_base, gold_plan)
        result = _compile_seed(plan, pack, honesty_mode="oracle_diagnostic")
        if result.ok and result.seed:
            return result.seed, "gold", ["gold_factor_topology_seed"]
        return _minimal_seed(inventory), "gold", ["gold_factor_failed", result.reason or "unknown"]

    if arm.strategy is SeedStrategy.GOLD_PLAN:
        result = _compile_seed(gold_plan, pack, honesty_mode="oracle_diagnostic")
        if result.ok and result.seed:
            return result.seed, "gold", ["gold_plan_oracle_seed"]
        return _minimal_seed(inventory), "gold", ["gold_plan_failed", result.reason or "unknown"]

    if arm.strategy is SeedStrategy.RETRIEVED_PROTOTYPE:
        tmp_arm = Slm147Arm(
            arm_id="tmp_retrieved",
            strategy=RetrievalStrategy.HYBRID,
            k=1,
            seeds=(seed,),
            description="",
        )
        query = _build_query_context(spec, gold_plan)
        seed_source, _candidate, notes = _build_seed(tmp_arm, query, index, seed)
        return seed_source, "retrieved", notes

    if arm.strategy is SeedStrategy.PLAN_RERANKED_RETRIEVAL:
        tmp_arm = Slm147Arm(
            arm_id="tmp_plan_reranked",
            strategy=RetrievalStrategy.SEMANTIC_PLAN,
            k=1,
            seeds=(seed,),
            description="",
        )
        query = _build_query_context(spec, gold_plan)
        seed_source, _candidate, notes = _build_seed(tmp_arm, query, index, seed)
        return seed_source, "retrieved", notes

    return _minimal_seed(inventory), "none", ["unknown_strategy_fallback"]


def _render_gold_source(spec: ProgramSpec) -> str:
    """Render a canonical source string from a spec for placeholder extraction."""
    from slm_training.harnesses.experiments.slm146_semantic_plan_compiler import (
        _render_fixture_ast,
    )

    return _render_fixture_ast(spec.ast)


def _ast_to_topology(root: dict[str, Any], parent_id: int | None = None) -> TopologyNode:
    """Convert a validated OpenUI AST into a conflict-slice topology tree."""
    node_id = hash((id(root), root.get("statementId", "")))
    node_id = node_id % (2**31)
    node_type = str(root.get("typeName") or "MASK")
    children: list[TopologyNode] = []
    child_list = (root.get("props") or {}).get("children") or []
    if isinstance(child_list, list):
        for child in child_list:
            if isinstance(child, dict):
                children.append(_ast_to_topology(child, node_id))
    return TopologyNode(
        node_id=node_id,
        node_type=node_type,
        parent_id=parent_id,
        children=tuple(children),
        active=True,
        protected=False,
        certified=False,
        decision_level=0,
    )


def _build_conflict_slice(
    tree: TopologyNode,
    gold_program: Any,
    plan_source: str,
    record_id: str,
    seed: int,
) -> ConflictSliceV1:
    """Synthesize a conflict slice from the seed tree and gold program."""
    from slm_training.harnesses.experiments.conflict_slice_repair import (
        _tree_fingerprint,
    )

    gold_components = component_multiset(gold_program.root)
    failing_id: int | None = None
    # Prefer a node whose type is over-represented vs gold.
    seed_components: Counter[str] = Counter()
    for node in _walk_topology(tree):
        seed_components[node.node_type] += 1
    excess_types = [t for t, c in seed_components.items() if c > gold_components.get(t, 0)]

    for node in _walk_topology(tree):
        if node.node_type in excess_types:
            failing_id = node.node_id
            break
    if failing_id is None:
        # Fallback: last leaf.
        leaves = [n for n in _walk_topology(tree) if not n.children]
        if leaves:
            failing_id = leaves[-1].node_id
        else:
            failing_id = tree.node_id

    failing_node = next(n for n in _walk_topology(tree) if n.node_id == failing_id)
    frontier: set[int] = set()
    if failing_node.parent_id is not None:
        frontier.add(failing_node.parent_id)
    for child in failing_node.children:
        frontier.add(child.node_id)

    completeness: Any = "SOUND_OVERAPPROX"
    if plan_source in ("gold", "oracle"):
        completeness = "EXACT"

    original_fp = _tree_fingerprint(tree)
    return ConflictSliceV1(
        conflict_id=f"{record_id}-{seed}",
        stage="binding",
        reason_code="plan_component_mismatch",
        failing_node_ids=(failing_id,),
        dependency_frontier=tuple(sorted(frontier)),
        protected_node_ids=(),
        completeness_class=completeness,
        original_state_fingerprint=original_fp,
        source_provenance=plan_source,
        notes="synthetic fixture analyzer slice",
    )


def _walk_topology(root: TopologyNode):
    yield root
    for child in root.children:
        yield from _walk_topology(child)


def _run_recovery(
    seed_source: str,
    gold_program: Any,
    plan_source: str,
    policy: RepairPolicyName,
    record_id: str,
    seed: int,
    search_config: Slm148SearchConfig,
) -> RepairTrace:
    """Apply a repair policy to the seed and return the trace."""
    try:
        program = validate(seed_source)
    except Exception:  # noqa: BLE001
        return RepairTrace(
            trace_id=f"{record_id}-{policy}-s{seed}",
            conflict_id=f"{record_id}-{seed}",
            policy=policy,
            seed=seed,
            original_tree=TopologyNode(node_id=0, node_type="MASK"),
            repaired_tree=TopologyNode(node_id=0, node_type="MASK"),
            remasked_node_ids=(),
            protected_mutations=0,
            budget_forwards=search_config.equal_forward_budget or 64,
            budget_verifier_calls=0,
            recovered=False,
            repeated_conflict=False,
            notes="invalid_seed_no_recovery",
        )

    tree = _ast_to_topology(program.root)
    slice_ = _build_conflict_slice(tree, gold_program, plan_source, record_id, seed)
    return apply_repair_policy(
        tree,
        slice_,
        policy,
        seed=seed,
        budget_forwards=search_config.equal_forward_budget or 64,
        budget_verifier_calls=16,
    )


def _make_record(
    spec: ProgramSpec,
    gold_plan: SemanticPlanV1,
    seed_arm: Slm148SeedArm,
    recovery_arm: Slm148RecoveryArm | None,
    stage: str,
    seed: int,
    seed_source: str,
    plan_source: str,
    trace: RepairTrace,
    notes: list[str],
) -> Slm148Record:
    gold_source = _render_gold_source(spec)
    canonical_seed = _canonical_source(seed_source)
    seed_valid = canonical_seed is not None
    ratio = _token_ratio(canonical_seed, _canonical_source(gold_source)) if seed_valid else None
    coverage = _component_coverage(seed_source, gold_source) if seed_valid else 0.0

    return Slm148Record(
        record_id=spec.id,
        seed_strategy=seed_arm.strategy.value,
        recovery_policy=recovery_arm.policy if recovery_arm else "none",
        stage=stage,
        seed=seed,
        plan_source=plan_source,
        seed_source=seed_source,
        seed_valid=seed_valid,
        seed_to_gold_ratio=ratio,
        component_coverage=coverage,
        conflict_id=trace.conflict_id,
        recovered=trace.recovered,
        remasked_nodes=len(trace.remasked_node_ids),
        preserved_nodes=_node_count_topology(trace.repaired_tree),
        forwards=trace.budget_forwards,
        verifier_calls=trace.budget_verifier_calls,
        repeated_conflict=trace.repeated_conflict,
        protected_mutations=trace.protected_mutations,
        completeness_class=trace.notes.split("=")[-1].split()[0] if "completeness=" in trace.notes else "UNKNOWN",
        notes=notes + [trace.notes],
    )


def _node_count_topology(root: TopologyNode) -> int:
    return sum(1 for _ in _walk_topology(root))


def _aggregate_records(
    seed_arm: Slm148SeedArm,
    recovery_arm: Slm148RecoveryArm | None,
    stage: str,
    seed: int,
    records: list[Slm148Record],
) -> Slm148Row:
    n = len(records)
    if not n:
        return Slm148Row(
            arm_id=seed_arm.arm_id,
            seed_strategy=seed_arm.strategy.value,
            recovery_policy=recovery_arm.policy if recovery_arm else "none",
            stage=stage,
            seed=seed,
            promotable=seed_arm.promotable and not (recovery_arm.diagnostic if recovery_arm else False),
            n_records=0,
            seed_valid_count=0,
            mean_seed_to_gold_ratio=None,
            mean_component_coverage=0.0,
            recovery_rate=0.0,
            mean_remasked_nodes=0.0,
            mean_preserved_nodes=0.0,
            mean_forwards=0.0,
            mean_verifier_calls=0.0,
            repeated_conflict_rate=0.0,
            notes=["empty corpus"],
        )

    ratios = [r.seed_to_gold_ratio for r in records if r.seed_to_gold_ratio is not None]
    mean_ratio = sum(ratios) / len(ratios) if ratios else None

    notes = [
        f"seed_strategy={seed_arm.strategy.value}",
        f"recovery_policy={recovery_arm.policy if recovery_arm else 'none'}",
        f"stage={stage}",
        "fixture-only: no X22 model trained or decoded",
    ]
    if not seed_arm.promotable:
        notes.append("non-promotable seed arm")
    if recovery_arm and recovery_arm.diagnostic:
        notes.append("non-promotable diagnostic recovery arm")

    return Slm148Row(
        arm_id=seed_arm.arm_id,
        seed_strategy=seed_arm.strategy.value,
        recovery_policy=recovery_arm.policy if recovery_arm else "none",
        stage=stage,
        seed=seed,
        promotable=seed_arm.promotable and not (recovery_arm.diagnostic if recovery_arm else False),
        n_records=n,
        seed_valid_count=sum(1 for r in records if r.seed_valid),
        mean_seed_to_gold_ratio=mean_ratio,
        mean_component_coverage=sum(r.component_coverage for r in records) / n,
        recovery_rate=sum(1 for r in records if r.recovered) / n,
        mean_remasked_nodes=sum(r.remasked_nodes for r in records) / n,
        mean_preserved_nodes=sum(r.preserved_nodes for r in records) / n,
        mean_forwards=sum(r.forwards for r in records) / n,
        mean_verifier_calls=sum(r.verifier_calls for r in records) / n,
        repeated_conflict_rate=sum(1 for r in records if r.repeated_conflict) / n,
        notes=notes,
    )


def _strategy_value(strategy: SeedStrategy | str) -> str:
    return strategy.value if isinstance(strategy, SeedStrategy) else strategy


def build_manifest() -> Slm148Manifest:
    """Return the default SLM-148 staged factorial manifest."""
    seeds = (
        Slm148SeedArm(
            arm_id="S0_minimal",
            strategy=SeedStrategy.MINIMAL,
            seeds=(0, 1, 2),
            description="Canonical minimal X22 seed; baseline for search distance.",
        ),
        Slm148SeedArm(
            arm_id="S1_frequency_prior",
            strategy=SeedStrategy.FREQUENCY_PRIOR,
            seeds=(0, 1, 2),
            description="Deterministic train-set archetype/role frequency prior seed.",
        ),
        Slm148SeedArm(
            arm_id="S2_learned_archetype_role_set",
            strategy=SeedStrategy.LEARNED_ARCHETYPE_ROLE_SET,
            seeds=(0, 1, 2),
            description="Learned archetype + role set merged into gold topology/bindings.",
        ),
        Slm148SeedArm(
            arm_id="S3_learned_full_plan",
            strategy=SeedStrategy.LEARNED_FULL_PLAN,
            seeds=(0, 1, 2),
            description="Fully learned plan seed (chain topology; wiring approximation).",
        ),
        Slm148SeedArm(
            arm_id="S4_gold_factor_bindings",
            strategy=SeedStrategy.GOLD_FACTOR,
            seeds=(0, 1, 2),
            description="Gold binding factor substituted into predicted plan; diagnostic.",
            promotable=False,
        ),
        Slm148SeedArm(
            arm_id="S5_gold_plan_oracle",
            strategy=SeedStrategy.GOLD_PLAN,
            seeds=(0, 1, 2),
            description="Full gold plan oracle seed; diagnostic ceiling.",
            promotable=False,
        ),
        Slm148SeedArm(
            arm_id="S6_retrieved_prototype",
            strategy=SeedStrategy.RETRIEVED_PROTOTYPE,
            seeds=(0, 1, 2),
            description="Best leakage-safe retrieved valid prototype (SPV1-04 hybrid).",
        ),
        Slm148SeedArm(
            arm_id="S7_plan_reranked_retrieval",
            strategy=SeedStrategy.PLAN_RERANKED_RETRIEVAL,
            seeds=(0, 1, 2),
            description="Retrieved prototype reranked by learned plan factors.",
        ),
    )
    recoveries = (
        Slm148RecoveryArm(
            arm_id="R0_none",
            policy="none",
            description="Canonical X22 no additional remask/recovery.",
        ),
        Slm148RecoveryArm(
            arm_id="R1_full_remask",
            policy="full_remask",
            description="Full/coarse remask control from SLM-113.",
        ),
        Slm148RecoveryArm(
            arm_id="R2_suffix_rollback",
            policy="suffix_rollback",
            description="Suffix rollback control.",
        ),
        Slm148RecoveryArm(
            arm_id="R3_conflict_slice",
            policy="conflict_slice",
            description="Conflict-slice localized revision.",
        ),
        Slm148RecoveryArm(
            arm_id="R4_oracle_conflict_slice",
            policy="conflict_slice_expanded",
            description="Oracle conflict slice expanded; diagnostic.",
            diagnostic=True,
        ),
    )
    return Slm148Manifest(
        seed_arms=seeds,
        recovery_arms=recoveries,
        search_config=Slm148SearchConfig(),
        claim_class="wiring",
        status="not_run",
    )


def validate_manifest(manifest: Slm148Manifest) -> list[str]:
    """Validate manifest shape and honest constraints."""
    errors: list[str] = []
    if not manifest.seed_arms:
        errors.append("seed_arms must not be empty")
    if not manifest.recovery_arms:
        errors.append("recovery_arms must not be empty")
    seen: set[str] = set()
    for arm in manifest.seed_arms:
        if arm.arm_id in seen:
            errors.append(f"duplicate arm_id: {arm.arm_id}")
        seen.add(arm.arm_id)
        if not arm.seeds:
            errors.append(f"{arm.arm_id}: seeds must not be empty")
        if arm.strategy in (SeedStrategy.GOLD_PLAN, SeedStrategy.GOLD_FACTOR) and arm.promotable:
            errors.append(f"{arm.arm_id}: gold/oracle seed arm must be non-promotable")
    for arm in manifest.recovery_arms:
        if arm.arm_id in seen:
            errors.append(f"duplicate arm_id: {arm.arm_id}")
        seen.add(arm.arm_id)
    return errors


def run_fixture_matrix(
    corpus: dict[str, list[tuple[ProgramSpec, SemanticPlanV1]]] | None = None,
    *,
    run_id: str = "slm148_fixture",
    output_dir: Path | None = None,
    predictor_epochs: int = 20,
    predictor_batch_size: int = 8,
) -> Slm148Report:
    """Run the SLM-148 staged factorial fixture matrix on the fixture corpus."""
    manifest = build_manifest()
    if corpus is None:
        corpus = build_fixture_plan_corpus(
            count=64,
            seed=0,
            root_containers=["Stack", "Card"],
            leaf_components=["TextContent", "Button"],
        )

    pack = get_pack("openui")
    index = build_prototype_index(corpus)
    bundle = _train_predictor_bundle(
        corpus,
        epochs=predictor_epochs,
        batch_size=predictor_batch_size,
    )
    bundle["train_records"] = corpus["train"]
    role_to_family = _role_to_family(corpus["train"])

    val_records = corpus.get("val", [])
    rows: list[Slm148Row] = []
    screening: dict[str, list[Slm148Record]] = {}

    # Stage 1: screen all seed strategies with R0.
    for seed_arm in manifest.seed_arms:
        screening[seed_arm.arm_id] = []
        for seed in seed_arm.seeds:
            per_record: list[Slm148Record] = []
            for spec, gold_plan in val_records:
                seed_source, plan_source, notes = _build_seed_for_arm(
                    seed_arm,
                    spec,
                    gold_plan,
                    pack,
                    index,
                    bundle,
                    role_to_family,
                    seed,
                )
                trace = _run_recovery(
                    seed_source,
                    validate(_render_gold_source(spec)),
                    plan_source,
                    "none",
                    spec.id,
                    seed,
                    manifest.search_config,
                )
                per_record.append(
                    _make_record(
                        spec,
                        gold_plan,
                        seed_arm,
                        None,
                        "screening",
                        seed,
                        seed_source,
                        plan_source,
                        trace,
                        notes,
                    )
                )
            screening[seed_arm.arm_id].extend(per_record)
            rows.append(
                _aggregate_records(seed_arm, None, "screening", seed, per_record)
            )

    # Survivors: promotable seed arms whose every screening record produced a valid seed.
    survivors = [
        arm
        for arm in manifest.seed_arms
        if arm.promotable and all(r.seed_valid for r in screening[arm.arm_id])
    ]

    # Stage 2: cross survivors with recovery arms.
    for seed_arm in survivors:
        for seed in seed_arm.seeds:
            for recovery_arm in manifest.recovery_arms:
                per_record: list[Slm148Record] = []
                for spec, gold_plan in val_records:
                    seed_source, plan_source, notes = _build_seed_for_arm(
                        seed_arm,
                        spec,
                        gold_plan,
                        pack,
                        index,
                        bundle,
                        role_to_family,
                        seed,
                    )
                    trace = _run_recovery(
                        seed_source,
                        validate(_render_gold_source(spec)),
                        plan_source,
                        recovery_arm.policy,
                        spec.id,
                        seed,
                        manifest.search_config,
                    )
                    per_record.append(
                        _make_record(
                            spec,
                            gold_plan,
                            seed_arm,
                            recovery_arm,
                            "cross",
                            seed,
                            seed_source,
                            plan_source,
                            trace,
                            notes,
                        )
                    )
                rows.append(
                    _aggregate_records(
                        seed_arm, recovery_arm, "cross", seed, per_record
                    )
                )

    report = Slm148Report(
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=X22_CONFLICT_CAMPAIGN_ID,
        run_id=run_id,
        status="fixture",
        manifest=manifest,
        rows=rows,
        survivors=[arm.arm_id for arm in survivors],
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm148_x22_conflict_campaign",
        ),
        claim_class="wiring",
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm148_x22_conflict_campaign_report.json")
    return report


def render_markdown(report: Slm148Report) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-148 / SPV1-05: plan-conditioned X22 × conflict-slice campaign ({report.run_id})",
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
        "## Seed arms",
        "",
        "| Arm | Strategy | Seeds | Promotable | Description |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arm in report.manifest.seed_arms:
        lines.append(
            f"| {arm.arm_id} | {_strategy_value(arm.strategy)} | {','.join(map(str, arm.seeds))} | "
            f"{arm.promotable} | {arm.description} |"
        )

    lines.extend(["", "## Recovery arms", "", "| Arm | Policy | Diagnostic | Description |", "| --- | --- | --- | --- |"])
    for arm in report.manifest.recovery_arms:
        lines.append(
            f"| {arm.arm_id} | {arm.policy} | {arm.diagnostic} | {arm.description} |"
        )

    lines.extend(["", "## Survivors", ""])
    if report.survivors:
        lines.append(", ".join(f"`{s}`" for s in report.survivors))
    else:
        lines.append("No seed arm survived screening.")

    lines.extend(["", "## Results", ""])
    for row in report.rows:
        lines.append(f"### {row.arm_id} / {row.recovery_policy} / seed={row.seed} ({row.stage})")
        lines.append(f"- records: {row.n_records}")
        lines.append(f"- seed valid: {row.seed_valid_count}")
        if row.mean_seed_to_gold_ratio is not None:
            lines.append(f"- mean seed-to-gold ratio: {row.mean_seed_to_gold_ratio:.3f}")
        lines.append(f"- mean component coverage: {row.mean_component_coverage:.3f}")
        lines.append(f"- recovery rate: {row.recovery_rate:.3f}")
        lines.append(f"- mean remasked nodes: {row.mean_remasked_nodes:.1f}")
        lines.append(f"- mean preserved nodes: {row.mean_preserved_nodes:.1f}")
        lines.append(f"- mean forwards: {row.mean_forwards:.1f}")
        lines.append(f"- mean verifier calls: {row.mean_verifier_calls:.1f}")
        lines.append(f"- repeated conflict rate: {row.repeated_conflict_rate:.3f}")
        for note in row.notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.extend(
        [
            "## Verdict",
            "",
            "This is a fixture wiring run. It validates that the staged factorial "
            "manifest is honest (gold/oracle arms non-promotable), that plan-conditioned "
            "and retrieved seeds compile to hard-valid initial states, that the "
            "conflict-slice policy from SLM-113 can be applied to those states, and "
            "that recovery bookkeeping is deterministic and replayable. "
            "Real quality/cost claims require the trained X22 model, SLM-111 beam/depth "
            "points, and AgentV evaluation on held-out suites.",
            "",
        ]
    )
    return "\n".join(lines)
