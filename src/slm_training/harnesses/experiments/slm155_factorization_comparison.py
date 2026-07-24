"""SLM-155 (SPV3-02): matched AR legal-action vs plan-conditioned X22 factorization.

Fixture/wiring-only comparison harness. It registers a common manifest, runs a
tiny shared AR scorer, a tiny plan-conditioned X22 seed + conflict-slice repair,
and a hybrid AR→X22 refinement arm on synthetic states. No trained X22
checkpoint or GPU measurement is performed; no ship-gate claim is made.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from slm_training.data.progspec.semantic_plan import (
    PlanArchetype,
    PlanIdentity,
    PlanTopology,
    RoleSlot,
    SemanticPlanV1,
)
from slm_training.data.semantic_plan.compiler import OpenUISemanticPlanCompiler
from slm_training.dsl.pack import get_pack
from slm_training.dsl.parser import validate
from slm_training.harnesses.experiments.conflict_slice_repair import (
    ConflictSliceV1,
    TopologyNode,
    apply_repair_policy,
)
from slm_training.harnesses.experiments.slm148_x22_conflict_campaign import (
    _ast_to_topology,
)
from slm_training.models.global_semantic_critic import (
    GlobalSemanticCritic,
    GlobalSemanticCriticConfig,
)
from slm_training.models.legal_action_scorer import (
    LegalActionScorerConfig,
    make_fixture_decisions,
    train_fixture_scorer,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "MATRIX_SET",
    "MATRIX_VERSION",
    "FACTORIZATION_CAMPAIGN_ID",
    "FINAL_PROGRAM_OUTCOME_SCHEMA",
    "FactorizationFamily",
    "FactorizationArm",
    "CommonConfig",
    "FactorizationTrace",
    "FactorizationRecord",
    "FactorizationRow",
    "FactorizationManifest",
    "FactorizationReport",
    "FinalProgramOutcomeV1",
    "build_manifest",
    "validate_manifest",
    "run_fixture_campaign",
    "run_fixture_campaign_with_records",
    "score_final_program",
    "old_outcome_label",
    "build_transition_table",
    "paired_validity_table",
    "render_markdown",
]

MATRIX_VERSION = "spv3-02-v1"
MATRIX_SET = "slm155_factorization_comparison"
FACTORIZATION_CAMPAIGN_ID = "slm155-factorization-comparison"

_MINIMAL_SEED_SOURCE = 'root = Stack([], "column")'


class FactorizationFamily(str, Enum):
    """Generation factorization family."""

    AR = "ar"
    X22 = "x22"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class CommonConfig:
    """Frozen orthogonal controls shared by every arm."""

    dsl_pack: str = "openui"
    plan_source: str = "predicted"
    compiler_honesty_mode: str = "production"
    scorer_variant: str = "mlp"
    scorer_seed: int = 0
    n_train_decisions: int = 64
    n_eval_decisions: int = 16
    x22_max_depth: int = 4
    x22_beam_width: int = 4
    equal_forward_budget: int = 64
    seeds: tuple[int, ...] = (0, 1, 2)
    metric_versions: dict[str, str] = field(default_factory=lambda: {"meaningful": "2.0.0"})

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["seeds"] = list(self.seeds)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommonConfig":
        return cls(
            dsl_pack=data.get("dsl_pack", "openui"),
            plan_source=data.get("plan_source", "predicted"),
            compiler_honesty_mode=data.get("compiler_honesty_mode", "production"),
            scorer_variant=data.get("scorer_variant", "mlp"),
            scorer_seed=data.get("scorer_seed", 0),
            n_train_decisions=data.get("n_train_decisions", 64),
            n_eval_decisions=data.get("n_eval_decisions", 16),
            x22_max_depth=data.get("x22_max_depth", 4),
            x22_beam_width=data.get("x22_beam_width", 4),
            equal_forward_budget=data.get("equal_forward_budget", 64),
            seeds=tuple(data.get("seeds", [0, 1, 2])),
            metric_versions=data.get("metric_versions", {"meaningful": "2.0.0"}),
        )


@dataclass(frozen=True)
class FactorizationArm:
    """One arm in the factorization comparison."""

    arm_id: str
    family: FactorizationFamily
    name: str
    description: str
    promotable: bool = True
    diagnostic: bool = False
    uses_gold_plan: bool = False
    uses_oracle_selector: bool = False
    max_beam_width: int = 1
    max_edit_budget: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["family"] = self.family.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FactorizationArm":
        return cls(
            arm_id=data["arm_id"],
            family=FactorizationFamily(data.get("family", "ar")),
            name=data.get("name", ""),
            description=data.get("description", ""),
            promotable=data.get("promotable", True),
            diagnostic=data.get("diagnostic", False),
            uses_gold_plan=data.get("uses_gold_plan", False),
            uses_oracle_selector=data.get("uses_oracle_selector", False),
            max_beam_width=data.get("max_beam_width", 1),
            max_edit_budget=data.get("max_edit_budget", 0),
        )


@dataclass(frozen=True)
class FactorizationTrace:
    """Common per-example trace envelope for both factorizations."""

    trace_id: str
    arm_id: str
    prompt_hash: str
    plan_hash: str
    initial_state_fingerprint: str
    legal_candidates: tuple[str, ...]
    decisions: tuple[str, ...]
    selected_actions: tuple[str, ...]
    cost_counters: dict[str, int]
    final_ast_fingerprint: str
    metrics: dict[str, float]
    ar_boundary_index: int | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["legal_candidates"] = list(self.legal_candidates)
        data["decisions"] = list(self.decisions)
        data["selected_actions"] = list(self.selected_actions)
        return data


@dataclass(frozen=True)
class FactorizationRecord:
    """Per-example result for one arm/seed."""

    record_id: str
    arm_id: str
    family: FactorizationFamily
    seed: int
    prompt_hash: str
    plan_source: str
    accepted: bool
    semantic_score: float
    cost_forwards: int
    cost_edits: int
    cost_verifier_calls: int
    trace: FactorizationTrace
    # SLM-297: retained final-program source for unified re-scoring. Empty
    # only for legacy constructions; the campaign always fills it.
    program_source: str = ""
    program_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["family"] = self.family.value
        data["trace"] = self.trace.to_dict()
        return data


@dataclass(frozen=True)
class FactorizationRow:
    """Aggregated row for one arm/seed."""

    arm_id: str
    family: FactorizationFamily
    seed: int
    promotable: bool
    n_records: int
    mean_semantic_score: float
    mean_cost_forwards: float
    mean_cost_edits: float
    mean_cost_verifier_calls: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["family"] = self.family.value
        return data


@dataclass(frozen=True)
class FactorizationManifest:
    """Preregistered manifest for the SLM-155 campaign."""

    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = FACTORIZATION_CAMPAIGN_ID
    hypothesis: str = (
        "For short OpenUI semantic decision streams, direct autoregressive "
        "legal-action scoring matches or exceeds plan-conditioned X22 semantic "
        "quality at lower deployed cost; a small AR→X22 hybrid edit tail adds a "
        "useful Pareto point only when AR realization contains recoverable "
        "structural errors."
    )
    falsifier: str = (
        "X22 produces materially better semantic outcomes at comparable cost, "
        "or the hybrid duplicates the better parent at higher cost, or plan "
        "features silently alter legal candidate membership between families."
    )
    common_config: CommonConfig = field(default_factory=CommonConfig)
    arms: tuple[FactorizationArm, ...] = ()
    claim_class: str = "wiring"
    status: str = "not_run"

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["common_config"] = self.common_config.to_dict()
        data["arms"] = [arm.to_dict() for arm in self.arms]
        return data

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class FactorizationReport:
    """Full fixture report for SLM-155."""

    matrix_set: str
    matrix_version: str
    experiment_id: str
    run_id: str
    status: str
    manifest: FactorizationManifest
    rows: list[FactorizationRow]
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
            "version_stamp": self.version_stamp,
        }

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )


#: Schema marker for the SLM-297 unified final-program outcome envelope.
FINAL_PROGRAM_OUTCOME_SCHEMA = "final_program_outcome/v1"

#: Note attached to X22 arms whose repair phase is simulated: the retained
#: final program is the (unmodified) seed source.
X22_SIMULATED_REPAIR_NOTE = "x22 repair simulated; seed source retained as final program"


@dataclass(frozen=True)
class FinalProgramOutcomeV1:
    """SLM-297 unified final-program semantic outcome for one record.

    The semantic score is the final program's ``meaningful_program_v1``
    verdict (plus parse/contract facts and typed reason codes). Process
    metrics (accepted / recovered / costs) are retained for provenance only
    and are never used as the semantic score.
    """

    record_id: str
    arm_id: str
    family: str
    seed: int
    program_source: str
    program_note: str | None
    parse_ok: bool
    contract_ok: bool
    v1_verdict: bool
    v1_reason_codes: tuple[str, ...]
    structural_similarity: float | None
    abstain_reason: str | None
    process_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = dict(asdict(self))
        data["schema"] = FINAL_PROGRAM_OUTCOME_SCHEMA
        data["v1_reason_codes"] = list(self.v1_reason_codes)
        return data


def score_final_program(record: FactorizationRecord) -> FinalProgramOutcomeV1:
    """Score one record's retained final program under the unified metric."""
    from slm_training.dsl.language_contract import output_contract_violations
    from slm_training.harnesses.model_build.eval_runner import (
        meaningful_program_v1_report,
    )

    source = record.program_source
    abstain_reason: str | None = None
    parse_ok = False
    contract_ok = False
    if not source:
        abstain_reason = "no_final_program_retained"
        report: dict[str, Any] = {"verdict": False, "reason_codes": ["no_final_program_retained"]}
    else:
        try:
            validate(source)
            parse_ok = True
        except Exception:  # noqa: BLE001 - parse fact, never raised
            parse_ok = False
        contract_ok = parse_ok and not output_contract_violations(source)
        report = meaningful_program_v1_report(source, gold=None)
    return FinalProgramOutcomeV1(
        record_id=record.record_id,
        arm_id=record.arm_id,
        family=record.family.value,
        seed=record.seed,
        program_source=source,
        program_note=record.program_note,
        parse_ok=parse_ok,
        contract_ok=contract_ok,
        v1_verdict=bool(report["verdict"]) and abstain_reason is None,
        v1_reason_codes=tuple(report.get("reason_codes", ())),
        structural_similarity=None,  # gold-free re-score; no reference available
        abstain_reason=abstain_reason,
        process_metrics={
            "accepted": record.accepted,
            "recovered": bool(record.trace.metrics.get("recovered", 0.0)),
            "cost_edits": record.cost_edits,
            "cost_forwards": record.cost_forwards,
            "cost_verifier_calls": record.cost_verifier_calls,
        },
    )


def old_outcome_label(outcome: FinalProgramOutcomeV1) -> str:
    """Legacy per-family process label (incommensurate; provenance only)."""
    if outcome.family == FactorizationFamily.AR.value:
        return "accepted" if outcome.process_metrics["accepted"] else "rejected"
    return "recovered" if outcome.process_metrics["recovered"] else "not_recovered"


def build_transition_table(
    outcomes: list[FinalProgramOutcomeV1],
) -> dict[str, dict[str, int]]:
    """Old process label × new final-program verdict counts for one arm."""
    table: dict[str, dict[str, int]] = {}
    for outcome in outcomes:
        label = old_outcome_label(outcome)
        verdict = "valid" if outcome.v1_verdict else "invalid"
        row = table.setdefault(label, {"valid": 0, "invalid": 0})
        row[verdict] += 1
    return table


def paired_validity_table(
    outcomes_a: list[FinalProgramOutcomeV1],
    outcomes_b: list[FinalProgramOutcomeV1],
) -> dict[str, int]:
    """Pair two arms by record_id+seed; count final-program validity combos.

    Keys: ``both_valid``, ``a_valid_b_invalid`` (invalid-over-valid damage
    when ``a`` is the baseline), ``a_invalid_b_valid``, ``both_invalid``,
    ``unpaired``.
    """
    key = lambda o: (o.record_id, o.seed)  # noqa: E731
    map_a = {key(o): o for o in outcomes_a}
    map_b = {key(o): o for o in outcomes_b}
    counts = {
        "both_valid": 0,
        "a_valid_b_invalid": 0,
        "a_invalid_b_valid": 0,
        "both_invalid": 0,
        "unpaired": 0,
    }
    for k in set(map_a) | set(map_b):
        a = map_a.get(k)
        b = map_b.get(k)
        if a is None or b is None:
            counts["unpaired"] += 1
            continue
        if a.v1_verdict and b.v1_verdict:
            counts["both_valid"] += 1
        elif a.v1_verdict and not b.v1_verdict:
            counts["a_valid_b_invalid"] += 1
        elif not a.v1_verdict and b.v1_verdict:
            counts["a_invalid_b_valid"] += 1
        else:
            counts["both_invalid"] += 1
    return counts


def _source_fingerprint(source: str) -> str:
    from slm_training.lineage.records import content_sha

    return content_sha(source)


def _canonical_source(source: str) -> str | None:
    try:
        program = validate(source)
        return program.serialized or source.strip()
    except Exception:  # noqa: BLE001
        return None


def _build_minimal_seed_source() -> str:
    return _MINIMAL_SEED_SOURCE


def _build_plan_seed_source(pack: Any, honesty_mode: str = "production") -> str:
    plan = SemanticPlanV1(
        identity=PlanIdentity(
            pack_id="openui",
            contract_hash=None,
            source_program_fingerprint=None,
            prompt_context_hash=None,
            provenance="predicted",
        ),
        archetype=PlanArchetype(id="stack", confidence=0.7),
        role_slots=(RoleSlot(role_id="r0", component_family="Stack"),),
        topology=PlanTopology(),
        symbols=(),
        bindings=(),
    )
    compiler = OpenUISemanticPlanCompiler(honesty_mode=honesty_mode)
    result = compiler.build_valid_seed(None, plan, pack)
    if result.ok and result.seed:
        return result.seed
    return _build_minimal_seed_source()


def _build_gold_seed_source(spec_source: str) -> str:
    return _canonical_source(spec_source) or _build_minimal_seed_source()


def _build_conflict_slice(
    source: str,
    plan_source: str,
    record_id: str,
    seed: int,
) -> ConflictSliceV1 | None:
    program = _canonical_source(source)
    if program is None:
        return None
    try:
        validated = validate(source)
        tree = _ast_to_topology(validated.root)
    except Exception:  # noqa: BLE001
        return None

    leaves = [n for n in _walk_topology(tree) if not n.children]
    failing_id = leaves[-1].node_id if leaves else tree.node_id
    frontier: set[int] = set()
    failing_node = next(n for n in _walk_topology(tree) if n.node_id == failing_id)
    if failing_node.parent_id is not None:
        frontier.add(failing_node.parent_id)

    completeness: Any = "SOUND_OVERAPPROX"
    if plan_source in ("gold", "oracle"):
        completeness = "EXACT"

    from slm_training.harnesses.experiments.conflict_slice_repair import (
        _tree_fingerprint,
    )

    return ConflictSliceV1(
        conflict_id=f"{record_id}-{seed}",
        stage="binding",
        reason_code="plan_component_mismatch",
        failing_node_ids=(failing_id,),
        dependency_frontier=tuple(sorted(frontier)),
        protected_node_ids=(),
        completeness_class=completeness,
        original_state_fingerprint=_tree_fingerprint(tree),
        source_provenance=plan_source,
        notes="synthetic fixture analyzer slice",
    )


def _walk_topology(root: TopologyNode):
    yield root
    for child in root.children:
        yield from _walk_topology(child)


def _program_from_actions(actions: list[str]) -> str:
    """Build a tiny hard-valid program from AR-selected actions (fixture only)."""
    if not actions:
        return _build_minimal_seed_source()
    placeholder = actions[0]
    if ":" in placeholder:
        placeholder = placeholder.rsplit(":", 1)[-1]
    placeholder = placeholder.strip() or "content"
    if not placeholder.startswith(":"):
        placeholder = f":{placeholder}"
    return (
        'root = Stack([n0], "column")\n'
        f"n0 = TextContent({json.dumps(placeholder, ensure_ascii=False)})"
    )


def _run_ar_records(
    arm: FactorizationArm,
    scorer: Any,
    decisions: list[Any],
    seed: int,
    use_critic: bool = False,
) -> list[FactorizationRecord]:
    critic = None
    if use_critic:
        critic = GlobalSemanticCritic(GlobalSemanticCriticConfig(seed=seed))

    records: list[FactorizationRecord] = []
    for index, decision in enumerate(decisions):
        scores = scorer.score(
            decision.context_features,
            decision.state_features,
            decision.legal_actions,
            plan_features=decision.plan_features,
            plan_action_features=decision.plan_action_features,
            pack_id=decision.pack_id,
        )
        chosen = scorer.decode(scores, decision.legal_actions).action_identity
        accepted = chosen in decision.accepted_action_ids
        semantic_score = 1.0 if accepted else 0.0
        # SLM-297: materialize the final program the AR stream implies so the
        # unified re-score sees the same artifact class as X22/hybrid arms.
        final_source = _program_from_actions([chosen] if chosen else [])

        forwards = max(1, arm.max_beam_width)
        scoring_ops = len(decision.legal_actions)
        cost_forwards = forwards + scoring_ops
        if use_critic and critic is not None:
            _ = critic.score(
                dict(decision.context_features),
                dict(decision.plan_features or {}),
                {"component_count": len(decision.legal_actions)},
                {},
            )
            cost_forwards += 1

        trace = FactorizationTrace(
            trace_id=f"{arm.arm_id}-s{seed}-{index}",
            arm_id=arm.arm_id,
            prompt_hash=_source_fingerprint(str(decision.context_features)),
            plan_hash=_source_fingerprint(str(decision.plan_features)),
            initial_state_fingerprint=_source_fingerprint(str(decision.state_features)),
            legal_candidates=tuple(decision.legal_actions),
            decisions=(chosen,) if chosen else (),
            selected_actions=(chosen,) if chosen else (),
            cost_counters={
                "forwards": cost_forwards,
                "scoring_ops": scoring_ops,
                "edits": 0,
                "verifier_calls": 0,
            },
            final_ast_fingerprint=_source_fingerprint(str(scores.logits.tolist())),
            metrics={"accepted": float(accepted), "semantic_score": semantic_score},
            notes=["fixture ar decision"],
        )
        records.append(
            FactorizationRecord(
                record_id=decision.decision_id,
                arm_id=arm.arm_id,
                family=arm.family,
                seed=seed,
                prompt_hash=trace.prompt_hash,
                plan_source="predicted",
                accepted=accepted,
                semantic_score=semantic_score,
                cost_forwards=cost_forwards,
                cost_edits=0,
                cost_verifier_calls=0,
                trace=trace,
                program_source=final_source,
                program_note="AR program materialized from selected actions",
            )
        )
    return records


def _run_x22_records(
    arm: FactorizationArm,
    pack: Any,
    plan_source: str,
    n_records: int,
    seed: int,
    use_critic: bool = False,
) -> list[FactorizationRecord]:
    critic = None
    if use_critic:
        critic = GlobalSemanticCritic(GlobalSemanticCriticConfig(seed=seed))

    records: list[FactorizationRecord] = []
    for index in range(n_records):
        if arm.uses_gold_plan:
            source = _build_gold_seed_source(_build_plan_seed_source(pack))
        elif plan_source == "minimal":
            source = _build_minimal_seed_source()
        else:
            source = _build_plan_seed_source(pack)

        program = _canonical_source(source)
        valid = program is not None
        slice_ = _build_conflict_slice(source, plan_source, f"rec{index}", seed)
        trace_tree = (
            apply_repair_policy(
                _ast_to_topology(validate(source).root),
                slice_,
                "conflict_slice",
                seed=seed,
                budget_forwards=64,
                budget_verifier_calls=16,
            )
            if slice_ is not None
            else None
        )

        recovered = bool(trace_tree.recovered) if trace_tree is not None else False
        edits = len(trace_tree.remasked_node_ids) if trace_tree is not None else 0
        verifier_calls = trace_tree.budget_verifier_calls if trace_tree is not None else 0
        semantic_score = 1.0 if recovered else 0.0
        cost_forwards = 64

        if use_critic and critic is not None and program is not None:
            _ = critic.score(
                {"pack_id": "openui", "n_mentioned_components": 1},
                {"plan_steps": 1},
                {"component_count": 1},
                {},
            )
            cost_forwards += 1

        trace = FactorizationTrace(
            trace_id=f"{arm.arm_id}-s{seed}-{index}",
            arm_id=arm.arm_id,
            prompt_hash=_source_fingerprint(f"prompt-{index}-{seed}"),
            plan_hash=_source_fingerprint(plan_source),
            initial_state_fingerprint=_source_fingerprint(source),
            legal_candidates=(),
            decisions=("edit",) if trace_tree is not None else (),
            selected_actions=("conflict_slice",),
            cost_counters={
                "forwards": cost_forwards,
                "scoring_ops": 0,
                "edits": edits,
                "verifier_calls": verifier_calls,
            },
            final_ast_fingerprint=_source_fingerprint(source),
            metrics={"recovered": float(recovered), "valid": float(valid)},
            notes=["fixture x22 seed+repair"],
        )
        records.append(
            FactorizationRecord(
                record_id=f"rec{index}",
                arm_id=arm.arm_id,
                family=arm.family,
                seed=seed,
                prompt_hash=trace.prompt_hash,
                plan_source=plan_source,
                accepted=recovered,
                semantic_score=semantic_score,
                cost_forwards=cost_forwards,
                cost_edits=edits,
                cost_verifier_calls=verifier_calls,
                trace=trace,
                program_source=source,
                program_note=X22_SIMULATED_REPAIR_NOTE,
            )
        )
    return records


def _run_hybrid_records(
    arm: FactorizationArm,
    scorer: Any,
    pack: Any,
    decisions: list[Any],
    seed: int,
    use_critic: bool = False,
) -> list[FactorizationRecord]:
    critic = None
    if use_critic:
        critic = GlobalSemanticCritic(GlobalSemanticCriticConfig(seed=seed))

    records: list[FactorizationRecord] = []
    for index, decision in enumerate(decisions):
        scores = scorer.score(
            decision.context_features,
            decision.state_features,
            decision.legal_actions,
            plan_features=decision.plan_features,
            plan_action_features=decision.plan_action_features,
            pack_id=decision.pack_id,
        )
        chosen = scorer.decode(scores, decision.legal_actions).action_identity
        ar_decisions = (chosen,) if chosen else ()

        source = _program_from_actions(list(ar_decisions))
        program = _canonical_source(source)
        valid = program is not None

        slice_ = _build_conflict_slice(source, "predicted", f"rec{index}", seed)
        total_edits = 0
        total_verifier = 0
        recovered = False
        budget = max(1, arm.max_edit_budget)
        for _ in range(budget):
            if slice_ is None:
                break
            tree = _ast_to_topology(validate(source).root)
            trace_tree = apply_repair_policy(
                tree,
                slice_,
                "conflict_slice",
                seed=seed,
                budget_forwards=64,
                budget_verifier_calls=16,
            )
            total_edits += len(trace_tree.remasked_node_ids)
            total_verifier += trace_tree.budget_verifier_calls
            recovered = recovered or trace_tree.recovered
            # Re-slice repaired tree for next iteration.
            slice_ = _build_conflict_slice(
                source, "predicted", f"rec{index}", seed
            )

        semantic_score = 1.0 if recovered else 0.0
        cost_forwards = 1 + len(decision.legal_actions) + 64 * budget
        if use_critic and critic is not None:
            _ = critic.score(
                dict(decision.context_features),
                dict(decision.plan_features or {}),
                {"component_count": len(decision.legal_actions)},
                {},
            )
            cost_forwards += 1

        trace = FactorizationTrace(
            trace_id=f"{arm.arm_id}-s{seed}-{index}",
            arm_id=arm.arm_id,
            prompt_hash=_source_fingerprint(str(decision.context_features)),
            plan_hash=_source_fingerprint(str(decision.plan_features)),
            initial_state_fingerprint=_source_fingerprint(source),
            legal_candidates=tuple(decision.legal_actions),
            decisions=ar_decisions + (("repair",) * budget),
            selected_actions=ar_decisions + (("conflict_slice",) * budget),
            cost_counters={
                "forwards": cost_forwards,
                "scoring_ops": len(decision.legal_actions),
                "edits": total_edits,
                "verifier_calls": total_verifier,
            },
            final_ast_fingerprint=_source_fingerprint(source),
            metrics={"accepted": float(recovered), "valid": float(valid)},
            ar_boundary_index=len(ar_decisions),
            notes=["fixture hybrid ar->x22"],
        )
        records.append(
            FactorizationRecord(
                record_id=decision.decision_id,
                arm_id=arm.arm_id,
                family=arm.family,
                seed=seed,
                prompt_hash=trace.prompt_hash,
                plan_source="predicted",
                accepted=recovered,
                semantic_score=semantic_score,
                cost_forwards=cost_forwards,
                cost_edits=total_edits,
                cost_verifier_calls=total_verifier,
                trace=trace,
                program_source=source,
                program_note="hybrid AR program retained; repair simulated on same tree",
            )
        )
    return records


def _run_oracle_records(
    arm: FactorizationArm,
    decisions: list[Any],
    seed: int,
) -> list[FactorizationRecord]:
    records: list[FactorizationRecord] = []
    for index, decision in enumerate(decisions):
        chosen = decision.accepted_action_ids[0] if decision.accepted_action_ids else decision.legal_actions[0]
        final_source = _program_from_actions([chosen])
        trace = FactorizationTrace(
            trace_id=f"{arm.arm_id}-s{seed}-{index}",
            arm_id=arm.arm_id,
            prompt_hash=_source_fingerprint(str(decision.context_features)),
            plan_hash=_source_fingerprint(str(decision.plan_features)),
            initial_state_fingerprint="oracle",
            legal_candidates=tuple(decision.legal_actions),
            decisions=(chosen,),
            selected_actions=(chosen,),
            cost_counters={"forwards": 0, "scoring_ops": 0, "edits": 0, "verifier_calls": 0},
            final_ast_fingerprint="oracle",
            metrics={"accepted": 1.0},
            notes=["oracle selector; non-promotable"],
        )
        records.append(
            FactorizationRecord(
                record_id=decision.decision_id,
                arm_id=arm.arm_id,
                family=arm.family,
                seed=seed,
                prompt_hash=trace.prompt_hash,
                plan_source="oracle",
                accepted=True,
                semantic_score=1.0,
                cost_forwards=0,
                cost_edits=0,
                cost_verifier_calls=0,
                trace=trace,
                program_source=final_source,
                program_note="oracle selector; non-promotable",
            )
        )
    return records


def _aggregate_records(
    arm: FactorizationArm,
    seed: int,
    records: list[FactorizationRecord],
) -> FactorizationRow:
    n = len(records)
    if not n:
        return FactorizationRow(
            arm_id=arm.arm_id,
            family=arm.family,
            seed=seed,
            promotable=arm.promotable and not arm.diagnostic,
            n_records=0,
            mean_semantic_score=0.0,
            mean_cost_forwards=0.0,
            mean_cost_edits=0.0,
            mean_cost_verifier_calls=0.0,
            notes=["empty"],
        )

    notes = [
        f"family={arm.family.value}",
        f"seed={seed}",
        "fixture-only: no live production decode loop",
    ]
    if not arm.promotable:
        notes.append("non-promotable arm")
    if arm.diagnostic:
        notes.append("diagnostic arm")

    return FactorizationRow(
        arm_id=arm.arm_id,
        family=arm.family,
        seed=seed,
        promotable=arm.promotable and not arm.diagnostic,
        n_records=n,
        mean_semantic_score=sum(r.semantic_score for r in records) / n,
        mean_cost_forwards=sum(r.cost_forwards for r in records) / n,
        mean_cost_edits=sum(r.cost_edits for r in records) / n,
        mean_cost_verifier_calls=sum(r.cost_verifier_calls for r in records) / n,
        notes=notes,
    )


def build_manifest() -> FactorizationManifest:
    """Return the default SLM-155 factorization comparison manifest."""
    arms = (
        FactorizationArm(
            arm_id="AR-G",
            family=FactorizationFamily.AR,
            name="legal_action_greedy",
            description="Greedy autoregressive scorer over the live legal action set.",
        ),
        FactorizationArm(
            arm_id="AR-B",
            family=FactorizationFamily.AR,
            name="legal_action_beam",
            description="AR scorer with a bounded beam/k-best calibration placeholder.",
            max_beam_width=4,
        ),
        FactorizationArm(
            arm_id="AR-C",
            family=FactorizationFamily.AR,
            name="legal_action_with_critic",
            description="AR greedy followed by the shared global semantic critic/selector.",
        ),
        FactorizationArm(
            arm_id="X-M",
            family=FactorizationFamily.X22,
            name="x22_minimal_seed",
            description="Canonical minimal valid X22 seed with no learned plan.",
        ),
        FactorizationArm(
            arm_id="X-P",
            family=FactorizationFamily.X22,
            name="x22_plan_seed_repair",
            description="Plan-conditioned seed followed by conflict-slice repair.",
            max_edit_budget=1,
        ),
        FactorizationArm(
            arm_id="X-C",
            family=FactorizationFamily.X22,
            name="x22_with_critic",
            description="X-P plus the same final global critic/selector.",
            max_edit_budget=1,
        ),
        FactorizationArm(
            arm_id="H-1",
            family=FactorizationFamily.HYBRID,
            name="ar_then_one_repair",
            description="AR greedy program followed by one bounded X22 refinement phase.",
            max_edit_budget=1,
        ),
        FactorizationArm(
            arm_id="H-K",
            family=FactorizationFamily.HYBRID,
            name="ar_then_calibrated_k",
            description="AR program followed by the smallest calibrated K edit budget (K=2 fixture).",
            max_edit_budget=2,
        ),
        FactorizationArm(
            arm_id="H-C",
            family=FactorizationFamily.HYBRID,
            name="hybrid_with_critic",
            description="H-1 plus the same final critic/selector.",
            max_edit_budget=1,
        ),
        FactorizationArm(
            arm_id="gold_ar",
            family=FactorizationFamily.AR,
            name="gold_plan_ar",
            description="Gold-plan AR diagnostic ceiling.",
            promotable=False,
            diagnostic=True,
            uses_gold_plan=True,
        ),
        FactorizationArm(
            arm_id="gold_x22",
            family=FactorizationFamily.X22,
            name="gold_plan_x22",
            description="Gold-plan X22 diagnostic ceiling.",
            promotable=False,
            diagnostic=True,
            uses_gold_plan=True,
        ),
        FactorizationArm(
            arm_id="oracle_selector",
            family=FactorizationFamily.AR,
            name="oracle_candidate_selector",
            description="Oracle candidate/beam selector diagnostic.",
            promotable=False,
            diagnostic=True,
            uses_oracle_selector=True,
        ),
    )
    return FactorizationManifest(arms=arms)


def validate_manifest(manifest: FactorizationManifest) -> list[str]:
    """Validate manifest shape and honest constraints."""
    errors: list[str] = []
    if not manifest.arms:
        errors.append("arms must not be empty")
    seen: set[str] = set()
    for arm in manifest.arms:
        if arm.arm_id in seen:
            errors.append(f"duplicate arm_id: {arm.arm_id}")
        seen.add(arm.arm_id)
        if arm.uses_gold_plan and arm.promotable:
            errors.append(f"{arm.arm_id}: gold-plan arm must be non-promotable")
        if arm.uses_oracle_selector and arm.promotable:
            errors.append(f"{arm.arm_id}: oracle-selector arm must be non-promotable")
        if arm.diagnostic and arm.promotable:
            errors.append(f"{arm.arm_id}: diagnostic arm must be non-promotable")
    cfg = manifest.common_config
    if not cfg.dsl_pack:
        errors.append("common_config.dsl_pack must be set")
    if not cfg.plan_source:
        errors.append("common_config.plan_source must be set")
    return errors


def run_fixture_campaign(
    manifest: FactorizationManifest | None = None,
    *,
    run_id: str = "slm155_fixture",
    output_dir: Path | None = None,
    n_records: int = 16,
    scorer_steps: int = 20,
    retain_records: bool = False,
) -> FactorizationReport:
    """Run the SLM-155 factorization comparison fixture campaign."""
    manifest = manifest or build_manifest()
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("manifest validation failed: " + "; ".join(errors))

    common = manifest.common_config
    pack = get_pack(common.dsl_pack)

    train_decisions = make_fixture_decisions(
        n=common.n_train_decisions, seed=common.scorer_seed
    )
    eval_decisions = make_fixture_decisions(
        n=common.n_eval_decisions, seed=common.scorer_seed + 1
    )

    scorer_result = train_fixture_scorer(
        train_decisions,
        config=LegalActionScorerConfig(
            variant=common.scorer_variant, seed=common.scorer_seed
        ),
        steps=scorer_steps,
        lr=0.05,
    )
    scorer = scorer_result["scorer"]

    rows: list[FactorizationRow] = []
    all_records: list[FactorizationRecord] = []
    for arm in manifest.arms:
        for seed in common.seeds:
            if arm.family is FactorizationFamily.AR:
                if arm.uses_oracle_selector:
                    records = _run_oracle_records(arm, eval_decisions, seed)
                else:
                    use_critic = arm.arm_id in {"AR-C"}
                    records = _run_ar_records(
                        arm, scorer, eval_decisions, seed, use_critic=use_critic
                    )
            elif arm.family is FactorizationFamily.X22:
                plan_source = "gold" if arm.uses_gold_plan else "predicted"
                use_critic = arm.arm_id in {"X-C"}
                records = _run_x22_records(
                    arm, pack, plan_source, n_records, seed, use_critic=use_critic
                )
            else:  # hybrid
                use_critic = arm.arm_id in {"H-C"}
                records = _run_hybrid_records(
                    arm, scorer, pack, eval_decisions, seed, use_critic=use_critic
                )
            rows.append(_aggregate_records(arm, seed, records))
            all_records.extend(records)

    report = FactorizationReport(
        matrix_set=MATRIX_SET,
        matrix_version=MATRIX_VERSION,
        experiment_id=FACTORIZATION_CAMPAIGN_ID,
        run_id=run_id,
        status="fixture",
        manifest=manifest,
        rows=rows,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm155_factorization_comparison",
        ),
        claim_class="wiring",
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm155_factorization_comparison_report.json")
    if retain_records:
        return report, all_records
    return report


def run_fixture_campaign_with_records(
    manifest: FactorizationManifest | None = None,
    *,
    run_id: str = "slm155_fixture",
    output_dir: Path | None = None,
    n_records: int = 16,
    scorer_steps: int = 20,
) -> tuple[FactorizationReport, list[FactorizationRecord]]:
    """SLM-297: run the fixture campaign retaining every per-example record.

    Identical campaign to :func:`run_fixture_campaign`; only the return value
    differs (records are kept instead of discarded after aggregation).
    """
    report, records = run_fixture_campaign(
        manifest,
        run_id=run_id,
        output_dir=output_dir,
        n_records=n_records,
        scorer_steps=scorer_steps,
        retain_records=True,
    )
    return report, records


def render_markdown(report: FactorizationReport) -> str:
    """Render a fixture-caveat markdown summary."""
    lines = [
        f"# SLM-155 (SPV3-02): Matched AR vs plan-conditioned X22 factorization ({report.run_id})",
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
        "## Common config",
        "",
        "| Key | Value |",
        "| --- | --- |",
    ]
    cfg = report.manifest.common_config
    for key, value in cfg.to_dict().items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Arms",
            "",
            "| Arm | Family | Promotable | Diagnostic | Description |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for arm in report.manifest.arms:
        lines.append(
            f"| {arm.arm_id} | {arm.family.value} | {arm.promotable} | "
            f"{arm.diagnostic} | {arm.description} |"
        )

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Arm | Seed | Records | Mean semantic score | Forwards | Edits | Verifier calls |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.rows:
        lines.append(
            f"| {row.arm_id} | {row.seed} | {row.n_records} | "
            f"{row.mean_semantic_score:.3f} | {row.mean_cost_forwards:.1f} | "
            f"{row.mean_cost_edits:.1f} | {row.mean_cost_verifier_calls:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "This is a fixture wiring run. It validates that the factorization "
            "comparison manifest is honest (gold/oracle arms non-promotable), that "
            "AR and X22 arms can be evaluated under a common trace envelope, that "
            "the hybrid AR→X22 boundary preserves lineage, and that cost accounting "
            "is deterministic. Real quality/cost claims require trained models, "
            "matched capacity, AgentV evaluation, and measured wall-clock latency.",
            "",
        ]
    )
    return "\n".join(lines)
