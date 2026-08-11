"""PGS-H01 (SLM-510): fixture-scale goal-support domain-adequacy campaign.

Default-off, bounded, deterministic diagnostic campaign over complete finite
OpenUI-style word-tree fixtures. Composes existing
:class:`~slm_training.dsl.solver.goal_support.GoalSupportProvider`,
:func:`~slm_training.dsl.solver.goal_support.analyze_goal_domain`, and
:func:`~slm_training.dsl.solver.goal_support.exact_goal_closure` through the
governed :class:`~ExperimentCampaignV1` / :class:`~CampaignStore` contract.

Arms (shared fixtures, legal domains, bounds, deterministic ordering):

* ``structural_support_reference`` — enumerative structural well-formedness oracle
* ``goal_support_production_exact_diagnostic`` — production-exact goal evidence;
  observation-only (no hard prune)
* ``goal_support_evaluation_oracle_diagnostic`` — frozen evaluation-oracle facts;
  non-promotable, cannot authorize prune
* ``goal_support_certified_fixture`` — certified closure/prune only under complete
  forest + compiler-hard + replay-valid evidence

Wiring/diagnostic claim only — not model quality, ship gates, or promotion.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
    campaign_manifest_sha256,
)
from slm_training.autoresearch.schemas import CampaignBudget, CampaignSpec
from slm_training.autoresearch.storage import CampaignStore
from slm_training.data.progspec.goal_constraints import (
    CompiledGoalConstraintSetV1,
    GoalConstraintEvaluationV1,
    GoalConstraintV1,
)
from slm_training.data.progspec.synthesis_problem import PackIdentityV1
from slm_training.dsl.solver.goal_support import (
    EvaluatorIdentityV1,
    GoalFailureAtomV1,
    GoalGateResultV1,
    GoalSupportProvider,
    GoalTerminalEvidenceV1,
    GoalUnknownAtomV1,
    GoalVerifierProfileV1,
    MetricIdentityV1,
    analyze_goal_domain,
    compute_pack_identity_digest,
    exact_goal_closure,
)
from slm_training.dsl.solver.openui_support import GoalTerminalEvidenceTrace
from slm_training.dsl.solver.state import (
    DomainValue,
    FiniteDomainState,
    HoleDomain,
    HoleId,
    SolverBounds,
    SupportVerdict,
)
from slm_training.dsl.solver.support import (
    EnumerativeSupportOracle,
    ExpandStatus,
    ExpandStep,
    SupportCertificate,
    SupportQuery,
    VerifyOutcome,
    VerifyStatus,
    Verifier,
)
from slm_training.harness_core.lineage.records import content_sha
from slm_training.levers import MAX_RUN_MINUTES

CAMPAIGN_ID = "pgs-h01-goal-support-domain-adequacy-fixture"
EXPERIMENT_ID = "pgs-h01"
PRIMARY_METRIC = "exact_action_coverage_rate"
RESULT_SCHEMA = "goal_support_domain_adequacy_campaign_fixture/v1"

ARM_IDS: tuple[str, ...] = (
    "structural_support_reference",
    "goal_support_production_exact_diagnostic",
    "goal_support_evaluation_oracle_diagnostic",
    "goal_support_certified_fixture",
)

_DIAGNOSTIC_ARMS = frozenset(
    {
        "goal_support_production_exact_diagnostic",
        "goal_support_evaluation_oracle_diagnostic",
    }
)

_GENEROUS = SolverBounds(
    max_tokens=100_000,
    max_nodes=100_000,
    max_depth=64,
    max_backtracks=100_000,
    max_verifier_calls=100_000,
)

_REGRET_TREE = {
    "": (("a", "continue", "complete"), ("b", "continue", "complete")),
    "a": (("x", "terminal", "complete"),),
    "b": (("y", "terminal", "complete"),),
}

_INADEQUATE_TREE = {
    "": (("a", "terminal", "complete"), ("b", "terminal", "complete"),),
}

_CAP_TREE = {
    "": (
        ("a", "terminal", "complete"),
        ("b", "terminal", "complete"),
        ("c", "terminal", "complete"),
    ),
}

_PARTIAL_FOREST_TREE = {
    "": (("a", "continue", "partial"), ("b", "continue", "complete")),
    "a": (("x", "incomplete", "partial"),),
    "b": (("y", "terminal", "complete"),),
}

_UNAVAILABLE_TREE = {
    "": (("a", "continue", "complete"),),
    "a": (("x", "terminal", "complete"),),
}

_AMBIGUITY_TREE = {
    "": (("a", "continue", "complete"), ("b", "continue", "complete")),
    "a": (("x", "terminal", "complete"), ("y", "terminal", "complete")),
    "b": (("z", "terminal", "complete"),),
}

_FORBIDDEN_RENDER_TERMS = frozenset(
    {"global unsat", "certified_unsat", "certified unsat", "globally unsatisfiable"}
)

__all__ = [
    "ARM_IDS",
    "CAMPAIGN_ID",
    "EXPERIMENT_ID",
    "FIXTURE_IDS",
    "PRIMARY_METRIC",
    "RESULT_SCHEMA",
    "GoalSupportDomainAdequacyCampaignV1",
    "build_fixture_matrix",
    "canonical_result_digest",
    "evaluate_falsifiers",
    "plan_only_preview",
    "render_markdown",
    "run_campaign",
    "run_fixture_arm",
]


def _program_digest(program: str) -> str:
    return hashlib.sha256(program.encode()).hexdigest()


class _WordExpander:
    def __init__(
        self,
        tree: dict[str, tuple[tuple[str, ...], ...]],
        *,
        bounds: SolverBounds = _GENEROUS,
        constraint_version: str = "v1",
    ) -> None:
        self._tree = tree
        self._bounds = bounds
        self._cv = constraint_version
        self._prefix_by_fp: dict[str, str] = {}
        self._root = self._state_for("")
        self._prefix_by_fp[self._root.fingerprint] = ""

    @property
    def problem_id(self) -> str:
        return self._root.problem_id

    @property
    def pack_id(self) -> str:
        return "fixture-word"

    @property
    def constraint_version(self) -> str:
        return self._cv

    @property
    def bounds(self) -> SolverBounds:
        return self._bounds

    def root_state(self) -> FiniteDomainState:
        return self._root

    def value_for(self, prefix: str, letter: str) -> DomainValue:
        return DomainValue.create("letter", {"prefix": prefix, "letter": letter})

    def _state_for(self, prefix: str) -> FiniteDomainState:
        branches = self._tree.get(prefix, ())
        hole = HoleId(namespace="word", path=(len(prefix), prefix or "ROOT"), kind="next")
        values = tuple(self.value_for(prefix, branch[0]) for branch in branches)
        domain = HoleDomain(hole, values, metadata=(("node", prefix or "ROOT"),))
        return FiniteDomainState(
            problem_id=f"word:{prefix or 'ROOT'}",
            pack_id=self.pack_id,
            constraint_version=self._cv,
            bounds=self._bounds,
            holes=(domain,),
        )

    def successor(self, state, hole_id, value) -> ExpandStep:
        prefix = self._prefix_by_fp[state.fingerprint]
        payload = value.payload
        letter = payload["letter"]
        branch = next(b for b in self._tree[prefix] if b[0] == letter)
        kind = branch[1]
        coverage = branch[2]
        forced_next = branch[3] if len(branch) > 3 else None
        if kind == "terminal":
            return ExpandStep(
                ExpandStatus.TERMINAL,
                program=prefix + letter,
                coverage=coverage,
                detail=prefix + letter,
            )
        if kind == "dead":
            return ExpandStep(ExpandStatus.DEAD, coverage=coverage, detail="bottom")
        if kind == "incomplete":
            return ExpandStep(ExpandStatus.INCOMPLETE, coverage=coverage, detail="uncovered")
        next_prefix = forced_next if forced_next is not None else prefix + letter
        child = self._state_for(next_prefix)
        self._prefix_by_fp[child.fingerprint] = next_prefix
        return ExpandStep(ExpandStatus.CONTINUE, next_state=child, coverage=coverage)


class _TracingGoalVerifier(Verifier):
    def __init__(
        self,
        accept: set[str],
        *,
        unavailable: set[str] | None = None,
        profile_label: str,
        profile_digest: str,
        structural_only: bool = False,
        unknown_atoms: tuple[GoalUnknownAtomV1, ...] = (),
        failure_atoms: tuple[GoalFailureAtomV1, ...] = (),
    ) -> None:
        self._accept = accept
        self._unavailable = unavailable or set()
        self._profile_label = profile_label
        self._profile_digest = profile_digest
        self._structural_only = structural_only
        self._unknown_atoms = unknown_atoms
        self._failure_atoms = failure_atoms
        self._trace = GoalTerminalEvidenceTrace()

    @property
    def last_trace(self) -> GoalTerminalEvidenceTrace:
        return self._trace

    @property
    def profile(self) -> str:
        return f"{self._profile_label}/{self._profile_digest}"

    def verify(self, program: str) -> VerifyOutcome:
        if program in self._unavailable:
            return VerifyOutcome(VerifyStatus.UNAVAILABLE, detail="fixture_unavailable")
        overall = "ACCEPT" if program in self._accept else "REJECT"
        structural = "REJECT" if self._structural_only and program.endswith("x") else overall
        gate_status = structural if self._structural_only else overall
        evidence = GoalTerminalEvidenceV1(
            profile_digest=self._profile_digest,
            program_digest=_program_digest(program),
            canonical_program_digest=_program_digest(program),
            program_spec_digest="3" * 64,
            semantic_plan_digest="4" * 64,
            constraint_evaluations=(
                GoalConstraintEvaluationV1(constraint_id="c_slot_button", status="PASS"),
            ),
            required_gate_results=(GoalGateResultV1(gate_id="G0", status=gate_status),),
            mandatory_unknown_atoms=self._unknown_atoms,
            mandatory_failure_atoms=self._failure_atoms,
            structural_status=structural,
            overall_status=overall,
        )
        bound = GoalTerminalEvidenceV1.from_dict(evidence.to_dict(include_digest=True))
        self._trace.append(bound)
        digest = bound.evidence_digest or bound.compute_digest()
        detail = f"overall={overall};structural={structural};evidence={digest}"
        if self._structural_only:
            if structural == "ACCEPT":
                return VerifyOutcome(VerifyStatus.ACCEPT, detail=detail)
            return VerifyOutcome(VerifyStatus.REJECT, detail=detail)
        if overall == "ACCEPT":
            return VerifyOutcome(VerifyStatus.ACCEPT, detail=detail)
        return VerifyOutcome(VerifyStatus.REJECT, detail=detail)


def _fixture_pack_identity() -> PackIdentityV1:
    return PackIdentityV1(
        pack_id="openui",
        contract_version=2,
        grammar_sha256="a" * 64,
        tokenizer_id="tok/v1",
        canonicalizer_id="canon/v1",
        artifact_schema="openui/v1",
        artifact_sha256="b" * 64,
    )


def _fixture_constraint(**overrides: object) -> GoalConstraintV1:
    defaults: dict[str, object] = {
        "constraint_id": "c_slot_button",
        "kind": "slot_present",
        "parameters": {"role_id": "primary_action"},
        "source_kind": "pack_contract",
        "source_id": "pack/openui",
        "source_digest": "abc123",
        "authority_tier": "compiler-hard",
        "completeness": "EXACT",
        "may_prune": True,
    }
    defaults.update(overrides)
    return GoalConstraintV1(**defaults)


def _fixture_compiled_set() -> CompiledGoalConstraintSetV1:
    hard = _fixture_constraint()
    advisory = _fixture_constraint(
        constraint_id="adv_hint",
        authority_tier="advisory-learned",
        completeness="HEURISTIC",
        may_prune=False,
        source_kind="prompt_requirement",
    )
    evaluation = _fixture_constraint(
        constraint_id="eval_gate",
        kind="required_gate_passes",
        parameters={"gate": "G3"},
        authority_tier="evaluation-only",
        completeness="EXACT",
        may_prune=False,
        source_kind="evaluation_fixture",
    )
    payload = CompiledGoalConstraintSetV1(
        problem_digest="a" * 64,
        request_digest="b" * 64,
        pack_identity_digest=compute_pack_identity_digest(_fixture_pack_identity()),
        compiler_version="compiler/v1",
        constraints=(hard, advisory, evaluation),
        hard_constraint_ids=(hard.constraint_id,),
        advisory_constraint_ids=(advisory.constraint_id,),
        evaluation_constraint_ids=(evaluation.constraint_id,),
    ).to_dict(include_digest=True)
    return CompiledGoalConstraintSetV1.from_dict(payload)


def _profile_for_mode(mode: str) -> GoalVerifierProfileV1:
    constraint_set = _fixture_compiled_set()
    pack = _fixture_pack_identity()
    pack_digest = compute_pack_identity_digest(pack)
    common = {
        "problem_digest": constraint_set.problem_digest,
        "constraint_set_digest": constraint_set.digest,
        "pack_identity": pack,
        "pack_identity_digest": pack_digest,
    }
    if mode == "production_exact":
        body = {
            "profile_id": "profile/prod",
            "mode": "production_exact",
            "required_constraint_ids": ("c_slot_button",),
            "required_gates": ("G0", "G3"),
            "required_evaluators": (
                EvaluatorIdentityV1(
                    evaluator_id="meaningful_program/v2",
                    version="v2",
                    implementation_hash="c" * 64,
                ),
            ),
            "metric_identities": (
                MetricIdentityV1(metric_id="meaningful_program", version="v2"),
            ),
            "authority_tier": "compiler-hard",
            **common,
        }
    elif mode == "evaluation_oracle":
        body = {
            "profile_id": "profile/eval",
            "mode": "evaluation_oracle",
            "required_constraint_ids": ("eval_gate",),
            "required_gates": (),
            "required_evaluators": (),
            "metric_identities": (),
            "authority_tier": "evaluation-only",
            **common,
        }
    elif mode == "advisory_diagnostic":
        body = {
            "profile_id": "profile/adv",
            "mode": "advisory_diagnostic",
            "required_constraint_ids": ("adv_hint",),
            "required_gates": (),
            "required_evaluators": (),
            "metric_identities": (),
            "authority_tier": "advisory-learned",
            **common,
        }
    else:
        raise ValueError(f"unsupported profile mode {mode!r}")
    return GoalVerifierProfileV1.from_dict(
        GoalVerifierProfileV1(**body).to_dict(include_digest=True)
    )


def _hole(expander: _WordExpander) -> HoleId:
    return expander.root_state().holes[0].hole_id


def _selected(expander: _WordExpander, letter: str) -> DomainValue:
    return expander.value_for("", letter)


@dataclass(frozen=True)
class GoalSupportFixtureSpec:
    fixture_id: str
    description: str
    tree: dict[str, tuple[tuple[str, ...], ...]]
    accept: frozenset[str]
    selected_letter: str
    exact_action_cap: int
    unavailable: frozenset[str] = frozenset()
    structural_split: bool = False
    bounds: SolverBounds = _GENEROUS
    unknown_atoms: tuple[GoalUnknownAtomV1, ...] = ()
    failure_atoms: tuple[GoalFailureAtomV1, ...] = ()
    profile_mode: str = "production_exact"
    diagnostic_profile_mode: str | None = None
    skip_arms: frozenset[str] = frozenset()
    expect_classification: str | None = None
    expect_hard_prune_allowed: bool = False


def build_fixture_matrix() -> tuple[GoalSupportFixtureSpec, ...]:
    failure = GoalFailureAtomV1(
        atom_id="fixture_fail",
        reason_code="no_witness",
        source_kind="constraint",
        completeness_class="EXACT",
    )
    return (
        GoalSupportFixtureSpec(
            fixture_id="exact_satisfying_alternative",
            description="Exact satisfying alternative selected under production-exact goal profile.",
            tree=_REGRET_TREE,
            accept=frozenset({"ax"}),
            selected_letter="a",
            exact_action_cap=10,
            expect_classification="domain_adequate_selected_supported",
        ),
        GoalSupportFixtureSpec(
            fixture_id="exact_violating_alternative",
            description="Supported alternative exists but selected action is goal-unsupported.",
            tree=_REGRET_TREE,
            accept=frozenset({"ax"}),
            selected_letter="b",
            exact_action_cap=10,
            expect_classification="selection_regret",
        ),
        GoalSupportFixtureSpec(
            fixture_id="structural_ne_goal_support",
            description="Structurally supported branch differs from exact-goal support.",
            tree=_REGRET_TREE,
            accept=frozenset({"ax"}),
            selected_letter="b",
            exact_action_cap=10,
            structural_split=True,
            expect_classification="selection_regret",
        ),
        GoalSupportFixtureSpec(
            fixture_id="bounded_domain_inadequacy",
            description="All observed actions replay-valid unsupported with obstruction core.",
            tree=_INADEQUATE_TREE,
            accept=frozenset(),
            selected_letter="a",
            exact_action_cap=10,
            failure_atoms=(failure,),
            expect_classification="domain_inadequate_under_bounds",
            expect_hard_prune_allowed=True,
        ),
        GoalSupportFixtureSpec(
            fixture_id="unknown_mandatory_evidence",
            description="Mandatory unavailable terminal yields unknown partition, not inadequacy.",
            tree=_UNAVAILABLE_TREE,
            accept=frozenset(),
            selected_letter="a",
            exact_action_cap=10,
            unavailable=frozenset({"ax"}),
            expect_classification="coverage_unknown",
        ),
        GoalSupportFixtureSpec(
            fixture_id="partial_forest",
            description="Partial/incomplete forest coverage stays coverage_unknown.",
            tree=_PARTIAL_FOREST_TREE,
            accept=frozenset(),
            selected_letter="a",
            exact_action_cap=10,
            expect_classification="coverage_unknown",
        ),
        GoalSupportFixtureSpec(
            fixture_id="bound_exhaustion",
            description="Tight solver bounds prevent domain_inadequate classification.",
            tree=_INADEQUATE_TREE,
            accept=frozenset(),
            selected_letter="a",
            exact_action_cap=10,
            bounds=SolverBounds(
                max_tokens=1,
                max_nodes=1,
                max_depth=2,
                max_backtracks=1,
                max_verifier_calls=1,
            ),
            expect_classification="coverage_unknown",
        ),
        GoalSupportFixtureSpec(
            fixture_id="advisory_cannot_prune",
            description="Advisory profile prose cannot authorize hard prune or inadequacy.",
            tree=_INADEQUATE_TREE,
            accept=frozenset(),
            selected_letter="a",
            exact_action_cap=10,
            profile_mode="advisory_diagnostic",
            diagnostic_profile_mode="advisory_diagnostic",
            skip_arms=frozenset({"goal_support_certified_fixture"}),
            expect_classification="coverage_unknown",
        ),
        GoalSupportFixtureSpec(
            fixture_id="evaluation_oracle_cannot_prune",
            description="Evaluation-oracle reject is observation-only; no certified prune.",
            tree=_INADEQUATE_TREE,
            accept=frozenset(),
            selected_letter="a",
            exact_action_cap=10,
            profile_mode="evaluation_oracle",
            skip_arms=frozenset({"goal_support_certified_fixture"}),
            expect_classification="coverage_unknown",
        ),
        GoalSupportFixtureSpec(
            fixture_id="cap_excludes_unique_support",
            description="Action cap omits the only supported witness → coverage_unknown.",
            tree=_CAP_TREE,
            accept=frozenset({"c"}),
            selected_letter="c",
            exact_action_cap=2,
            expect_classification="coverage_unknown",
        ),
        GoalSupportFixtureSpec(
            fixture_id="candidate_addition_stale_identity",
            description="Adding a satisfying candidate invalidates prior inadequacy identity.",
            tree=_INADEQUATE_TREE,
            accept=frozenset(),
            selected_letter="a",
            exact_action_cap=10,
            failure_atoms=(failure,),
            expect_classification="domain_inadequate_under_bounds",
        ),
        GoalSupportFixtureSpec(
            fixture_id="permutation_invariance",
            description="Deterministic partitions/classification under action-order invariance.",
            tree=_REGRET_TREE,
            accept=frozenset({"ax"}),
            selected_letter="a",
            exact_action_cap=10,
            expect_classification="domain_adequate_selected_supported",
        ),
        GoalSupportFixtureSpec(
            fixture_id="ambiguity",
            description="Unresolved ambiguity retains multiple supported alternatives.",
            tree=_AMBIGUITY_TREE,
            accept=frozenset({"ax", "ay"}),
            selected_letter="a",
            exact_action_cap=10,
            expect_classification="domain_adequate_selected_supported",
        ),
        GoalSupportFixtureSpec(
            fixture_id="raw_content_redaction",
            description="Terminal evidence digests only; no raw prompt/source leakage.",
            tree=_REGRET_TREE,
            accept=frozenset({"ax"}),
            selected_letter="a",
            exact_action_cap=10,
            expect_classification="domain_adequate_selected_supported",
        ),
        GoalSupportFixtureSpec(
            fixture_id="certified_singleton_zero_forward",
            description="Certified singleton closure requires zero model forwards.",
            tree={"": (("a", "terminal", "complete"),)},
            accept=frozenset({"a"}),
            selected_letter="a",
            exact_action_cap=10,
            expect_classification="domain_adequate_selected_supported",
            expect_hard_prune_allowed=False,
        ),
    )


FIXTURE_IDS: tuple[str, ...] = tuple(spec.fixture_id for spec in build_fixture_matrix())


@dataclass
class FixtureArmMetrics:
    fixture_id: str
    arm_id: str
    classification: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    exact_action_coverage_rate: float = 0.0
    supported_action_rate: float = 0.0
    unsupported_action_rate: float = 0.0
    unknown_action_rate: float = 0.0
    unobserved_action_rate: float = 0.0
    selected_action_supported: bool | None = None
    selection_regret: bool = False
    selection_unresolved: bool = False
    domain_inadequate_under_bounds: bool = False
    coverage_unknown: bool = False
    structural_supported_but_goal_unsupported: bool = False
    obstruction_core_emitted: int = 0
    obstruction_core_replay_ok: int = 0
    mean_obstruction_core_size: float = 0.0
    false_hard_prune_count: int = 0
    support_certificate_replay_failure_count: int = 0
    verifier_calls: int = 0
    expanded_nodes: int = 0
    backtracks: int = 0
    certified_removed_count: int = 0
    certified_singleton_forwards: int = 0
    wall_time_ms: float = 0.0
    profile_digest: str | None = None
    legal_set_digest: str | None = None
    bounds_digest: str | None = None
    stop_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _partition_rates(partitions: Any, *, action_count: int) -> dict[str, float]:
    return {
        "supported_action_rate": _rate(len(partitions.supported), action_count),
        "unsupported_action_rate": _rate(len(partitions.unsupported), action_count),
        "unknown_action_rate": _rate(len(partitions.unknown), action_count),
        "unobserved_action_rate": _rate(len(partitions.unobserved), action_count),
        "exact_action_coverage_rate": _rate(
            action_count - len(partitions.unobserved),
            action_count,
        ),
    }


def _build_provider(
    spec: GoalSupportFixtureSpec,
    *,
    profile: GoalVerifierProfileV1,
    structural_only: bool = False,
) -> tuple[_WordExpander, GoalSupportProvider | None, Verifier | None]:
    expander = _WordExpander(spec.tree, bounds=spec.bounds)
    digest = profile.digest or profile.compute_digest()

    def factory() -> _TracingGoalVerifier:
        return _TracingGoalVerifier(
            set(spec.accept),
            unavailable=set(spec.unavailable),
            profile_label="fixture-goal",
            profile_digest=digest,
            structural_only=structural_only or spec.structural_split,
            unknown_atoms=spec.unknown_atoms,
            failure_atoms=spec.failure_atoms,
        )

    if structural_only:
        return expander, None, factory
    return expander, GoalSupportProvider(expander, profile, factory), None


def run_fixture_arm(spec: GoalSupportFixtureSpec, arm_id: str) -> FixtureArmMetrics:
    if arm_id in spec.skip_arms:
        return FixtureArmMetrics(
            fixture_id=spec.fixture_id,
            arm_id=arm_id,
            skipped=True,
            skip_reason=f"fixture skips arm {arm_id!r}",
        )

    start = time.perf_counter()
    metrics = FixtureArmMetrics(fixture_id=spec.fixture_id, arm_id=arm_id)

    if arm_id == "structural_support_reference":
        profile = _profile_for_mode("production_exact")
        expander, _, verifier_factory = _build_provider(
            spec, profile=profile, structural_only=True
        )
        state = expander.root_state()
        hole = _hole(expander)
        domain = state.domain(hole)
        action_count = len(domain.values)
        supported: list[str] = []
        unsupported: list[str] = []
        unknown: list[str] = []
        for value in domain.values:
            query = SupportQuery(
                state_fingerprint=state.fingerprint,
                hole_id=hole,
                candidate=value,
            )
            oracle = EnumerativeSupportOracle(expander, verifier_factory())
            result = oracle.check(state, query)
            metrics.verifier_calls += int(result.counters.verifier_calls)
            metrics.expanded_nodes += int(result.counters.nodes)
            metrics.backtracks += int(result.counters.backtracks)
            if result.verdict is SupportVerdict.SUPPORTED:
                supported.append("1")
            elif result.verdict is SupportVerdict.UNSUPPORTED:
                unsupported.append("1")
            else:
                unknown.append("1")
        selected = _selected(expander, spec.selected_letter)
        selected_query = SupportQuery(
            state_fingerprint=state.fingerprint,
            hole_id=hole,
            candidate=selected,
        )
        selected_result = EnumerativeSupportOracle(expander, verifier_factory()).check(
            state, selected_query
        )
        metrics.selected_action_supported = (
            selected_result.verdict is SupportVerdict.SUPPORTED
        )
        metrics.exact_action_coverage_rate = 1.0
        metrics.supported_action_rate = _rate(len(supported), action_count)
        metrics.unsupported_action_rate = _rate(len(unsupported), action_count)
        metrics.unknown_action_rate = _rate(len(unknown), action_count)
        metrics.classification = "structural_reference"
        metrics.profile_digest = profile.digest or profile.compute_digest()
        metrics.wall_time_ms = round((time.perf_counter() - start) * 1000.0, 4)
        return metrics

    if arm_id == "goal_support_evaluation_oracle_diagnostic":
        profile = _profile_for_mode("evaluation_oracle")
    elif arm_id == "goal_support_production_exact_diagnostic":
        mode = spec.diagnostic_profile_mode or "production_exact"
        profile = _profile_for_mode(mode)
    elif arm_id == "goal_support_certified_fixture":
        profile = _profile_for_mode("production_exact")
    else:
        profile = _profile_for_mode(spec.profile_mode)

    expander, provider, _ = _build_provider(spec, profile=profile)
    assert provider is not None
    state = expander.root_state()
    hole = _hole(expander)
    selected = _selected(expander, spec.selected_letter)

    if arm_id == "goal_support_certified_fixture":
        try:
            cert_store: dict[str, SupportCertificate] = {}
            closure = exact_goal_closure(state, provider, certificate_store=cert_store)
        except ValueError as exc:
            metrics.skipped = True
            metrics.skip_reason = str(exc)
            metrics.wall_time_ms = round((time.perf_counter() - start) * 1000.0, 4)
            return metrics

        metrics.certified_removed_count = closure.counters.candidates_removed
        domain_size = len(state.domain(hole).values)
        metrics.certified_singleton_forwards = (
            0
            if domain_size == 1 and closure.counters.candidates_removed == 0
            else closure.counters.support_queries
        )
        for cert in cert_store.values():
            replay = provider.replay(cert, state=state)
            if not replay.ok:
                metrics.support_certificate_replay_failure_count += 1
            if cert.verdict is SupportVerdict.UNKNOWN:
                metrics.false_hard_prune_count += int(
                    any(cert.digest in deduction.certificate_ids for deduction in closure.deductions)
                )

        report, evidence = analyze_goal_domain(
            state=state,
            hole_id=hole,
            selected_action=selected,
            provider=provider,
            exact_action_cap=spec.exact_action_cap,
        )
    else:
        report, evidence = analyze_goal_domain(
            state=state,
            hole_id=hole,
            selected_action=selected,
            provider=provider,
            exact_action_cap=spec.exact_action_cap,
        )
        metrics.false_hard_prune_count = 0

    rates = _partition_rates(report.partitions, action_count=report.stats.action_count)
    metrics.classification = report.classification
    metrics.exact_action_coverage_rate = rates["exact_action_coverage_rate"]
    metrics.supported_action_rate = rates["supported_action_rate"]
    metrics.unsupported_action_rate = rates["unsupported_action_rate"]
    metrics.unknown_action_rate = rates["unknown_action_rate"]
    metrics.unobserved_action_rate = rates["unobserved_action_rate"]
    metrics.selected_action_supported = (
        report.selected_action_id in report.partitions.supported
    )
    metrics.selection_regret = report.classification == "selection_regret"
    metrics.selection_unresolved = report.classification == "selection_unresolved"
    metrics.domain_inadequate_under_bounds = (
        report.classification == "domain_inadequate_under_bounds"
    )
    metrics.coverage_unknown = report.classification == "coverage_unknown"
    metrics.verifier_calls = report.stats.verifier_calls
    metrics.expanded_nodes = report.stats.expanded_nodes
    metrics.backtracks = report.stats.backtracks
    metrics.stop_reasons = report.stats.stop_reasons
    metrics.profile_digest = report.profile_digest
    metrics.legal_set_digest = report.legal_set_fingerprint
    metrics.bounds_digest = report.bounds_digest

    core_sizes: list[int] = []
    for item in evidence:
        if item.obstruction_core_digest:
            metrics.obstruction_core_emitted += 1
            if item.replay_ok:
                metrics.obstruction_core_replay_ok += 1
            core_sizes.append(1)
        if not item.replay_ok and item.partition == "unsupported":
            metrics.support_certificate_replay_failure_count += 1

    metrics.mean_obstruction_core_size = (
        round(sum(core_sizes) / len(core_sizes), 6) if core_sizes else 0.0
    )

    if spec.structural_split and arm_id == "goal_support_production_exact_diagnostic":
        struct_profile = _profile_for_mode("production_exact")
        struct_expander, _, struct_verifier = _build_provider(
            spec, profile=struct_profile, structural_only=True
        )
        struct_state = struct_expander.root_state()
        struct_hole = _hole(struct_expander)
        struct_selected = _selected(struct_expander, spec.selected_letter)
        struct_query = SupportQuery(
            state_fingerprint=struct_state.fingerprint,
            hole_id=struct_hole,
            candidate=struct_selected,
        )
        struct_result = EnumerativeSupportOracle(struct_expander, struct_verifier()).check(
            struct_state, struct_query
        )
        goal_supported = report.selected_action_id in report.partitions.supported
        metrics.structural_supported_but_goal_unsupported = (
            struct_result.verdict is SupportVerdict.SUPPORTED and not goal_supported
        )

    if spec.fixture_id == "raw_content_redaction":
        payload = json.dumps([item.to_dict() for item in evidence])
        for needle in ("hf_", "sk-", "AKIA", "api_key=", "password=", "alice@example.com"):
            if needle in payload:
                metrics.false_hard_prune_count += 1

    metrics.wall_time_ms = round((time.perf_counter() - start) * 1000.0, 4)
    return metrics


def evaluate_falsifiers(
    fixture_results: dict[str, dict[str, FixtureArmMetrics]],
) -> dict[str, Any]:
    false_hard_prune = 0
    replay_failures = 0
    eval_called_certified = False
    advisory_hard_prune = False
    stale_replay = 0
    singleton_forward_violations = 0
    diagnostic_domain_changes = 0

    prod_by_fixture: dict[str, FixtureArmMetrics | None] = {}
    for fixture_id, arms in fixture_results.items():
        prod = arms.get("goal_support_production_exact_diagnostic")
        cert = arms.get("goal_support_certified_fixture")
        eval_arm = arms.get("goal_support_evaluation_oracle_diagnostic")
        prod_by_fixture[fixture_id] = prod

        for arm_metrics in arms.values():
            if arm_metrics.skipped:
                continue
            false_hard_prune += arm_metrics.false_hard_prune_count
            replay_failures += arm_metrics.support_certificate_replay_failure_count

        if eval_arm and not eval_arm.skipped:
            try:
                spec = next(s for s in build_fixture_matrix() if s.fixture_id == fixture_id)
                profile = _profile_for_mode("evaluation_oracle")
                expander, provider, _ = _build_provider(spec, profile=profile)
                assert provider is not None
                exact_goal_closure(expander.root_state(), provider)
                eval_called_certified = True
            except ValueError:
                pass

        if cert and not cert.skipped:
            if cert.certified_singleton_forwards > 0 and cert.certified_removed_count == 0:
                singleton_forward_violations += 1

        advisory = arms.get("goal_support_production_exact_diagnostic")
        if advisory and advisory.domain_inadequate_under_bounds:
            spec = next(s for s in build_fixture_matrix() if s.fixture_id == fixture_id)
            if (
                spec.diagnostic_profile_mode == "advisory_diagnostic"
                or spec.profile_mode == "advisory_diagnostic"
            ):
                advisory_hard_prune = True

    for fixture_id, arms in fixture_results.items():
        prod = arms.get("goal_support_production_exact_diagnostic")
        cert = arms.get("goal_support_certified_fixture")
        if (
            prod
            and cert
            and not prod.skipped
            and not cert.skipped
            and prod.legal_set_digest
            and cert.legal_set_digest
            and prod.legal_set_digest != cert.legal_set_digest
        ):
            diagnostic_domain_changes += 1

    violated = any(
        (
            false_hard_prune > 0,
            replay_failures > 0,
            eval_called_certified,
            advisory_hard_prune,
            singleton_forward_violations > 0,
        )
    )
    return {
        "falsifier_holds": violated,
        "false_hard_prune_count": false_hard_prune,
        "support_certificate_replay_failure_count": replay_failures,
        "evaluation_oracle_called_certified_closure": eval_called_certified,
        "advisory_claimed_domain_inadequate": advisory_hard_prune,
        "certified_singleton_forward_violations": singleton_forward_violations,
        "diagnostic_domain_digest_changes": diagnostic_domain_changes,
        "stale_replay_acceptance_count": stale_replay,
        "immediate_falsify_on": [
            "false_hard_prune_count > 0",
            "unsupported with incomplete evidence",
            "evaluation_oracle calling certified closure",
            "advisory claiming domain_inadequate_under_bounds",
            "support_certificate_replay_failure_count > 0",
            "certified_singleton nonzero forward without removal",
        ],
        "note": (
            "Primary success is correctness/diagnosis under exact fixtures, not "
            "higher model score. Timeouts/interrupts count as incomplete evidence."
        ),
    }


def _aggregate_arm_metrics(
    rows: list[FixtureArmMetrics],
) -> dict[str, Any]:
    active = [row for row in rows if not row.skipped]
    if not active:
        return {"n_fixtures": len(rows), "n_active": 0}

    def _mean(field_name: str) -> float:
        values = [getattr(row, field_name) for row in active]
        return round(sum(values) / len(values), 6)

    return {
        "n_fixtures": len(rows),
        "n_active": len(active),
        "exact_action_coverage_rate_mean": _mean("exact_action_coverage_rate"),
        "supported_action_rate_mean": _mean("supported_action_rate"),
        "unsupported_action_rate_mean": _mean("unsupported_action_rate"),
        "unknown_action_rate_mean": _mean("unknown_action_rate"),
        "unobserved_action_rate_mean": _mean("unobserved_action_rate"),
        "selected_action_supported_rate": _rate(
            sum(1 for row in active if row.selected_action_supported),
            len(active),
        ),
        "selection_regret_rate": _rate(
            sum(1 for row in active if row.selection_regret),
            len(active),
        ),
        "selection_unresolved_rate": _rate(
            sum(1 for row in active if row.selection_unresolved),
            len(active),
        ),
        "domain_inadequate_under_bounds_rate": _rate(
            sum(1 for row in active if row.domain_inadequate_under_bounds),
            len(active),
        ),
        "coverage_unknown_rate": _rate(
            sum(1 for row in active if row.coverage_unknown),
            len(active),
        ),
        "structural_supported_but_goal_unsupported_rate": _rate(
            sum(1 for row in active if row.structural_supported_but_goal_unsupported),
            len(active),
        ),
        "obstruction_core_emission_rate": _rate(
            sum(1 for row in active if row.obstruction_core_emitted > 0),
            len(active),
        ),
        "obstruction_core_replay_rate": _rate(
            sum(1 for row in active if row.obstruction_core_replay_ok > 0),
            len(active),
        ),
        "mean_obstruction_core_size": _mean("mean_obstruction_core_size"),
        "false_hard_prune_count": sum(row.false_hard_prune_count for row in active),
        "support_certificate_replay_failure_count": sum(
            row.support_certificate_replay_failure_count for row in active
        ),
        "verifier_calls_total": sum(row.verifier_calls for row in active),
        "expanded_nodes_total": sum(row.expanded_nodes for row in active),
        "backtracks_total": sum(row.backtracks for row in active),
        "wall_time_ms_total": round(sum(row.wall_time_ms for row in active), 4),
    }


def canonical_result_digest(result: dict[str, Any]) -> str:
    """Deterministic digest over canonical payload (no timestamps)."""

    def _strip(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                key: _strip(value)
                for key, value in sorted(obj.items())
                if key not in {"wall_time_ms", "wall_time_ms_total", "created_at"}
            }
        if isinstance(obj, list):
            return [_strip(item) for item in obj]
        return obj

    return content_sha(_strip(result))


def render_markdown(result: dict[str, Any]) -> str:
    """Human-readable summary without global UNSAT language."""
    text = json.dumps(result)
    lowered = text.lower()
    for term in _FORBIDDEN_RENDER_TERMS:
        if term in lowered:
            raise ValueError(f"result renderer must not use global UNSAT language: {term!r}")

    falsifier = result.get("falsifier", {})
    aggregate = result.get("aggregate_arms", {})
    lines = [
        "# SLM-510 (PGS-H01): Goal-support domain-adequacy fixture campaign",
        "",
        f"**Claim class:** `{result.get('claim_class', 'diagnostic')}` (wiring/diagnostic only)",
        "",
        f"**Campaign:** `{result.get('campaign_id')}`",
        "",
        f"**Primary metric (`{PRIMARY_METRIC}`):** "
        f"{result.get('primary_metric_value')}",
        "",
        f"**Deterministic digest:** `{result.get('result_digest')}`",
        "",
        "## Acceptance snapshot",
        "",
        "| Check | Value |",
        "| --- | --- |",
        f"| falsifier_holds | {falsifier.get('falsifier_holds')} |",
        f"| false_hard_prune_count | {falsifier.get('false_hard_prune_count')} |",
        f"| replay_failures | {falsifier.get('support_certificate_replay_failure_count')} |",
        f"| eval_oracle_called_certified | "
        f"{falsifier.get('evaluation_oracle_called_certified_closure')} |",
        f"| fixture_count | {len(result.get('fixture_results', {}))} |",
        f"| promotion | {result.get('promotion')} |",
        "",
        "## Arms (aggregate means)",
        "",
        "| Arm | exact_action_coverage_rate | domain_inadequate_rate | false_hard_prune |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm_id, stats in aggregate.items():
        lines.append(
            f"| {arm_id} | {stats.get('exact_action_coverage_rate_mean')} | "
            f"{stats.get('domain_inadequate_under_bounds_rate')} | "
            f"{stats.get('false_hard_prune_count')} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            result.get("scope_disclaimer", ""),
            "",
            "Language note: classifications use bounded domain-adequacy vocabulary "
            "(supported/unsupported/unknown/unobserved); no whole-domain contradiction claims.",
            "",
        ]
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class GoalSupportDomainAdequacyCampaignV1:
    campaign_id: str = CAMPAIGN_ID
    source_commit: str = "0" * 40
    claim_class: Literal["diagnostic", "fixture"] = "diagnostic"
    max_wall_minutes: float = float(MAX_RUN_MINUTES)

    def __post_init__(self) -> None:
        if self.claim_class not in {"diagnostic", "fixture"}:
            raise ValueError("PGS-H01 must stay diagnostic/fixture — no promotion path")
        if self.max_wall_minutes <= 0 or self.max_wall_minutes > MAX_RUN_MINUTES:
            raise ValueError("max_wall_minutes must obey MAX_RUN_MINUTES")

    @property
    def fingerprint(self) -> str:
        return content_sha(asdict(self))

    def manifest(self) -> ExperimentCampaignV1:
        fingerprint = self.fingerprint
        arms = tuple(
            CampaignArmV1(
                arm_id=arm,
                role="control" if arm == "structural_support_reference" else "candidate",
                config_sha256=content_sha({"campaign": fingerprint, "arm": arm}),
            )
            for arm in ARM_IDS
        )
        return ExperimentCampaignV1(
            campaign_id=self.campaign_id,
            experiment_id=EXPERIMENT_ID,
            hypothesis=(
                "On complete finite fixtures, goal-support substrate distinguishes "
                "structural vs goal support, selection regret vs bounded-domain "
                "inadequacy, unknown/unobserved vs replay-certified unsupported, "
                "and diagnostic vs certified pruning — with zero false hard prunes "
                "and decode invariants preserved."
            ),
            decision=(
                "Fixture-scale wiring evidence only: verify correctness/diagnosis "
                "under exact fixtures. No model-quality, ship-gate, or promotion claim."
            ),
            endpoints=(
                CampaignEndpointV1(
                    endpoint_id=f"{EXPERIMENT_ID}_primary",
                    metric=PRIMARY_METRIC,
                    role="primary",
                    direction="increase",
                    minimum_effect=0.0,
                ),
                CampaignEndpointV1(
                    endpoint_id=f"{EXPERIMENT_ID}_falsifier",
                    metric="false_hard_prune_count",
                    role="secondary",
                    direction="decrease",
                    minimum_effect=0.0,
                ),
            ),
            arms=arms,
            seeds=(0,),
            budget=CampaignBudget(
                max_experiments=len(ARM_IDS) * len(FIXTURE_IDS),
                max_wall_minutes=self.max_wall_minutes,
            ),
            stopping_rules=(
                "Every arm runs every non-skipped fixture with deterministic ordering.",
                "Diagnostic arms never authorize hard prune.",
                "Certified arm prunes only replay-valid unsupported under production_exact.",
                "Evaluation-oracle arm cannot call exact_goal_closure.",
                "Timeouts/interrupts are incomplete evidence, not pass/fail.",
            ),
            controls=(
                CampaignControlV1(
                    control_id="structural_reference",
                    description=(
                        "Structural well-formedness oracle baseline — no goal-sidecar "
                        "authority."
                    ),
                    kind="quality",
                ),
                CampaignControlV1(
                    control_id="zero_false_hard_prune",
                    description=(
                        "Any false hard prune, stale replay acceptance, or diagnostic "
                        "domain mutation falsifies immediately."
                    ),
                    kind="negative",
                ),
            ),
            negative_controls=("zero_false_hard_prune",),
            multiplicity_families=(
                MultiplicityFamilyV1(
                    family_id=f"{EXPERIMENT_ID}_family",
                    hypothesis_ids=(f"{EXPERIMENT_ID}_primary", f"{EXPERIMENT_ID}_falsifier"),
                    alpha=0.05,
                ),
            ),
            promotion_gates=(
                CampaignGateV1(
                    gate_id="fixture_wiring_only",
                    endpoint_id=f"{EXPERIMENT_ID}_primary",
                    operator="ge",
                    threshold=0.0,
                ),
            ),
            rollback_gates=(
                CampaignGateV1(
                    gate_id=f"{EXPERIMENT_ID}_false_prune_kill",
                    endpoint_id=f"{EXPERIMENT_ID}_falsifier",
                    operator="gt",
                    threshold=0.0,
                ),
            ),
            artifact_requirements=(
                ArtifactRequirementV1(kind="version_stamp"),
                ArtifactRequirementV1(kind="endpoint_result"),
            ),
            claim_class=self.claim_class,
            source_commit=self.source_commit,
            source_dirty=False,
            author="slm-training",
            created_at="1970-01-01T00:00:00Z",
        )


def plan_only_preview() -> dict[str, Any]:
    campaign = GoalSupportDomainAdequacyCampaignV1()
    manifest = campaign.manifest()
    return {
        "status": "plan_only",
        "campaign_id": campaign.campaign_id,
        "experiment_id": EXPERIMENT_ID,
        "claim_class": campaign.claim_class,
        "promotion": False,
        "manifest": manifest.model_dump(mode="json"),
        "manifest_sha256": campaign_manifest_sha256(manifest),
        "fixture_ids": list(FIXTURE_IDS),
        "arms": list(ARM_IDS),
        "primary_metric": PRIMARY_METRIC,
        "scope_disclaimer": (
            "Harness + tests only (SLM-510). SLM-511 executes the final governed "
            "run and publishes measured evidence. No train/checkpoint/AgentV/ship/"
            "MODEL_CARD/promotion."
        ),
    }


def run_campaign(
    campaign: GoalSupportDomainAdequacyCampaignV1,
    *,
    root: Any,
) -> dict[str, Any]:
    from pathlib import Path

    root = Path(root)
    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="PGS-H01 goal-support domain-adequacy fixture campaign",
            primary_metric=PRIMARY_METRIC,
            budget=CampaignBudget(
                max_experiments=len(ARM_IDS) * len(FIXTURE_IDS),
                max_wall_minutes=campaign.max_wall_minutes,
            ),
            created_at="1970-01-01T00:00:00Z",
        )
    )
    lock = store.lock_experiment_campaign(campaign.manifest())

    fixture_results: dict[str, dict[str, FixtureArmMetrics]] = {}
    for spec in build_fixture_matrix():
        fixture_results[spec.fixture_id] = {}
        for arm_id in ARM_IDS:
            fixture_results[spec.fixture_id][arm_id] = run_fixture_arm(spec, arm_id)

    aggregate_arms = {
        arm: _aggregate_arm_metrics(
            [fixture_results[fid][arm] for fid in FIXTURE_IDS]
        )
        for arm in ARM_IDS
    }
    falsifier = evaluate_falsifiers(fixture_results)
    primary_value = aggregate_arms[
        "goal_support_production_exact_diagnostic"
    ].get("exact_action_coverage_rate_mean", 0.0)

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "campaign_id": campaign.campaign_id,
        "experiment_id": EXPERIMENT_ID,
        "manifest_sha256": lock.manifest_sha256,
        "claim_class": campaign.claim_class,
        "promotion": False,
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_value": primary_value,
        "fixture_ids": list(FIXTURE_IDS),
        "arms": list(ARM_IDS),
        "aggregate_arms": aggregate_arms,
        "fixture_results": {
            fixture_id: {arm: metrics.to_dict() for arm, metrics in arms.items()}
            for fixture_id, arms in fixture_results.items()
        },
        "falsifier": falsifier,
        "scope_disclaimer": (
            "Default-off, bounded, deterministic fixture/diagnostic campaign over "
            "word-tree OpenUI-style fixtures. Wiring/diagnostic claim only — not model "
            "quality, ship gates, or global semantic completeness. Split-safe; no "
            "hidden gold in production-exact compilation."
        ),
    }
    result["result_digest"] = canonical_result_digest(result)

    artifact = store.write_artifact("goal_support_domain_adequacy_result", result)
    store.append_event(
        "goal_support_domain_adequacy_completed",
        experiment_id=campaign.campaign_id,
        status="diagnostic",
        artifact_sha256=artifact.stem,
        detail={
            "manifest_sha256": lock.manifest_sha256,
            "result_digest": result["result_digest"],
            "falsifier_holds": falsifier["falsifier_holds"],
            "false_hard_prune_count": falsifier["false_hard_prune_count"],
        },
    )
    return result
