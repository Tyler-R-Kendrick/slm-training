"""Governed fixture evidence for proof-carrying goal support.

The campaign deliberately contains no model.  It exercises the production
request -> compiled constraints -> OpenUI goal verifier -> existing VSS oracle
-> replay sidecar -> domain diagnosis -> guarded exact closure path on small,
complete finite domains.  Persisted rows contain only digests and bounded
counters; terminal source strings remain request-local.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
)
from slm_training.autoresearch.schemas import CampaignBudget, CampaignSpec
from slm_training.autoresearch.storage import CampaignStore
from slm_training.data.contract import GenerationRequest
from slm_training.data.progspec.prompt_requirements import (
    PromptSemanticRequirementsV1,
    RequirementFact,
)
from slm_training.data.progspec.synthesis_problem import (
    VerifiedSynthesisProblemV1,
)
from slm_training.data.semantic_plan.requirements_compile import (
    compile_goal_constraints,
)
from slm_training.dsl.pack import get_pack
from slm_training.dsl.solver.goal_support import (
    GoalDomainAdequacyReportV1,
    GoalSupportProvider,
    GoalVerifierProfileV1,
    analyze_goal_domain,
    exact_goal_closure,
    goal_action_digest,
)
from slm_training.dsl.solver.openui_support import (
    OpenUIGoalVerifier,
    OpenUIWellFormedVerifier,
    current_openui_pack_identity,
)
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
    SupportQuery,
    replay_support_certificate,
)
from slm_training.levers import MAX_RUN_MINUTES
from slm_training.lineage.records import canonical_json

ARM_IDS = (
    "structural_support_reference",
    "goal_support_production_exact_diagnostic",
    "goal_support_evaluation_oracle_diagnostic",
    "goal_support_certified_fixture",
)
METRIC_IDS = (
    "exact_action_coverage_rate",
    "supported_action_rate",
    "unsupported_action_rate",
    "unknown_action_rate",
    "unobserved_action_rate",
    "selected_action_supported_rate",
    "selection_regret_rate",
    "selection_unresolved_rate",
    "domain_inadequate_under_bounds_rate",
    "coverage_unknown_rate",
    "structural_supported_but_goal_unsupported_rate",
    "obstruction_core_emission_rate",
    "obstruction_core_replay_rate",
    "mean_obstruction_core_size",
    "false_hard_prune_count",
    "support_certificate_replay_failure_count",
    "verifier_calls",
    "expanded_nodes",
    "backtracks",
    "wall_time",
)
BOUNDS = SolverBounds(
    max_tokens=4096,
    max_nodes=64,
    max_depth=8,
    max_backtracks=64,
    max_verifier_calls=64,
)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Candidate:
    action_id: str
    program: str
    incomplete: bool = False


@dataclass(frozen=True)
class _Case:
    case_id: str
    candidates: tuple[_Candidate, ...]
    selected_action_id: str
    coverage: str = "complete"


_CASES = (
    _Case(
        "exact_satisfying_alternative",
        (
            _Candidate("a_supported", 'root = Button(":slot_0")'),
            _Candidate("b_unsupported", 'root = Button(":slot_1")'),
        ),
        "b_unsupported",
    ),
    _Case(
        "domain_relative_inadequacy",
        (
            _Candidate("a_unsupported", 'root = Button(":slot_1")'),
            _Candidate("b_unsupported", 'root = Input(":slot_1")'),
        ),
        "a_unsupported",
    ),
    _Case(
        "partial_completion_coverage",
        (
            _Candidate("a_unknown", 'root = Button(":slot_0")', incomplete=True),
            _Candidate("b_unsupported", 'root = Input(":slot_1")'),
        ),
        "a_unknown",
        coverage="partial",
    ),
    _Case(
        "candidate_cap_excludes_unique_support",
        (
            _Candidate("a_unsupported", 'root = Button(":slot_1")'),
            _Candidate("b_unsupported", 'root = Input(":slot_1")'),
            _Candidate("z_supported", 'root = Button(":slot_0")'),
        ),
        "a_unsupported",
    ),
)


class _TerminalFixtureExpander:
    """A finite fixture domain; search remains owned by the VSS oracle."""

    def __init__(self, state: FiniteDomainState) -> None:
        self._state = state

    @property
    def problem_id(self) -> str:
        return self._state.problem_id

    @property
    def pack_id(self) -> str:
        return self._state.pack_id

    @property
    def constraint_version(self) -> str:
        return self._state.constraint_version

    @property
    def bounds(self) -> SolverBounds:
        return self._state.bounds

    def successor(
        self, state: FiniteDomainState, hole_id: HoleId, value: DomainValue
    ) -> ExpandStep:
        del state, hole_id
        payload = value.payload
        if payload.get("incomplete"):
            return ExpandStep(
                ExpandStatus.INCOMPLETE,
                coverage="partial",
                detail="fixture_completion_unavailable",
            )
        return ExpandStep(
            ExpandStatus.TERMINAL,
            program=str(payload["program"]),
            coverage="complete",
        )


@dataclass(frozen=True)
class GoalSupportDomainAdequacyCampaignV1:
    campaign_id: str = "proof-carrying-goal-support-fixture"
    exact_action_cap: int = 2
    goal_support_query_cap: int = 32
    seed: int = 0
    source_commit: str = "0" * 40
    source_dirty: bool = False
    max_wall_minutes: float = float(MAX_RUN_MINUTES)

    def __post_init__(self) -> None:
        if self.exact_action_cap < 1:
            raise ValueError("exact_action_cap must be positive")
        if self.goal_support_query_cap < 0:
            raise ValueError("goal_support_query_cap must be non-negative")
        if self.max_wall_minutes <= 0 or self.max_wall_minutes > MAX_RUN_MINUTES:
            raise ValueError("max_wall_minutes must obey MAX_RUN_MINUTES")

    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                "config": asdict(self),
                "solver_bounds": BOUNDS.to_dict(),
                "fixture_suite_digest": _digest([asdict(case) for case in _CASES]),
            }
        )

    def manifest(self) -> ExperimentCampaignV1:
        return ExperimentCampaignV1(
            campaign_id=self.campaign_id,
            experiment_id=self.campaign_id,
            hypothesis=(
                "Pinned goal verification separates supported, unsupported, "
                "unknown, and unobserved legal actions while certified closure "
                "removes only replay-valid compiler-hard exact obstructions."
            ),
            decision=(
                "Retain proof-carrying goal support as a default-off deterministic "
                "diagnostic/certified seam only if replay and authority controls hold."
            ),
            endpoints=(
                CampaignEndpointV1(
                    endpoint_id="false_hard_prune_count",
                    metric="replay_invalid_or_nonhard_candidate_removals",
                    role="primary",
                    direction="decrease",
                    minimum_effect=0.0,
                ),
            ),
            arms=tuple(
                CampaignArmV1(
                    arm_id=arm_id,
                    role="control" if arm_id == ARM_IDS[0] else "candidate",
                    config_sha256=_digest(
                        {"campaign": self.fingerprint, "arm_id": arm_id}
                    ),
                )
                for arm_id in ARM_IDS
            ),
            seeds=(self.seed,),
            budget=CampaignBudget(
                max_experiments=len(ARM_IDS) * len(_CASES),
                max_wall_minutes=self.max_wall_minutes,
            ),
            stopping_rules=(
                "Stop goal queries at goal_support_query_cap and classify omitted actions unobserved.",
                "Any incomplete expansion, verifier unavailability, or VSS bound exhaustion is UNKNOWN.",
                "Abort the fixture claim if any replay-invalid or nonhard action is removed.",
            ),
            controls=(
                CampaignControlV1(
                    control_id="structural_reference",
                    description="The same finite actions are checked by the structural verifier only.",
                    kind="quality",
                ),
                CampaignControlV1(
                    control_id="evaluation_authority_isolation",
                    description="Evaluation-only evidence remains observation-only and never reaches exact closure.",
                    kind="negative",
                ),
            ),
            negative_controls=("evaluation_authority_isolation",),
            multiplicity_families=(
                MultiplicityFamilyV1(
                    family_id="proof_carrying_goal_support_fixture",
                    hypothesis_ids=("false_hard_prune_count",),
                    alpha=0.05,
                ),
            ),
            promotion_gates=(
                CampaignGateV1(
                    gate_id="fixture_only_no_false_prune",
                    endpoint_id="false_hard_prune_count",
                    operator="eq",
                    threshold=0.0,
                ),
            ),
            rollback_gates=(
                CampaignGateV1(
                    gate_id="false_hard_prune_detected",
                    endpoint_id="false_hard_prune_count",
                    operator="gt",
                    threshold=0.0,
                ),
            ),
            artifact_requirements=(
                ArtifactRequirementV1(kind="version_stamp"),
                ArtifactRequirementV1(kind="endpoint_result"),
            ),
            claim_class="fixture",
            source_commit=self.source_commit,
            source_dirty=self.source_dirty,
            author="slm-training",
            created_at="1970-01-01T00:00:00Z",
        )


@dataclass(frozen=True)
class _Context:
    request: GenerationRequest
    problem: VerifiedSynthesisProblemV1
    constraints: Any
    profile: GoalVerifierProfileV1


def _contexts() -> tuple[_Context, _Context]:
    pack = get_pack("openui")
    identity = current_openui_pack_identity()
    request = GenerationRequest(
        prompt="Fill the exact opaque slot contract.", slot_contract=(":slot_0",)
    )
    problem = VerifiedSynthesisProblemV1(
        problem_id="goal-support-production-fixture", pack_identity=identity
    )
    constraints = compile_goal_constraints(problem, request, pack)
    required = tuple(
        constraint_id
        for constraint_id in constraints.hard_constraint_ids
        if constraint_id in {"hard.output_kind_equals", "hard.slot_inventory_exact"}
    )
    profile = GoalVerifierProfileV1.from_dict(
        GoalVerifierProfileV1(
            profile_id="goal-support/production-exact-fixture",
            mode="production_exact",
            problem_digest=constraints.problem_digest,
            constraint_set_digest=constraints.digest,
            pack_identity=identity,
            pack_identity_digest=constraints.pack_identity_digest,
            required_constraint_ids=required,
            authority_tier="compiler-hard",
        ).to_dict()
    )

    evaluation_request = GenerationRequest(prompt="Frozen evaluation fixture.")
    evaluation_problem = VerifiedSynthesisProblemV1(
        problem_id="goal-support-evaluation-fixture",
        pack_identity=identity,
        requirements=PromptSemanticRequirementsV1(
            facts=(
                RequirementFact(
                    fact_id="evaluation.button.required",
                    factor_family="Button",
                    disposition="required",
                    statement="frozen fixture expects Button",
                    confidence=1.0,
                    provenance="retrieved",
                    authority="evaluation-only",
                ),
            )
        ),
    )
    evaluation_constraints = compile_goal_constraints(
        evaluation_problem, evaluation_request, pack
    )
    evaluation_profile = GoalVerifierProfileV1.from_dict(
        GoalVerifierProfileV1(
            profile_id="goal-support/evaluation-oracle-fixture",
            mode="evaluation_oracle",
            problem_digest=evaluation_constraints.problem_digest,
            constraint_set_digest=evaluation_constraints.digest,
            pack_identity=identity,
            pack_identity_digest=evaluation_constraints.pack_identity_digest,
            required_constraint_ids=evaluation_constraints.evaluation_constraint_ids,
            required_gates=("G11",),
            authority_tier="evaluation-only",
        ).to_dict()
    )
    return (
        _Context(request, problem, constraints, profile),
        _Context(
            evaluation_request,
            evaluation_problem,
            evaluation_constraints,
            evaluation_profile,
        ),
    )


def _state(
    case: _Case, constraint_version: str
) -> tuple[FiniteDomainState, DomainValue]:
    values = {
        candidate.action_id: DomainValue.create(
            "action",
            {
                "action_id": candidate.action_id,
                "incomplete": candidate.incomplete,
                "program": candidate.program,
            },
        )
        for candidate in case.candidates
    }
    state = FiniteDomainState(
        problem_id=f"proof-carrying-goal-support:{case.case_id}",
        pack_id="openui",
        constraint_version=constraint_version,
        bounds=BOUNDS,
        holes=(
            HoleDomain(
                HoleId("goal-support-fixture", (case.case_id,), "action"),
                tuple(values.values()),
                (("coverage", case.coverage),),
            ),
        ),
    )
    return state, values[case.selected_action_id]


def _provider(state: FiniteDomainState, context: _Context) -> GoalSupportProvider:
    pack = get_pack("openui")
    expander = _TerminalFixtureExpander(state)

    def verifier() -> OpenUIGoalVerifier:
        return OpenUIGoalVerifier(
            profile=context.profile,
            constraint_set=context.constraints,
            problem=context.problem,
            request=context.request,
            pack_identity=context.problem.pack_identity,
            pack=pack,
        )

    return GoalSupportProvider(
        expander,
        verifier_factory=verifier,
        profile=context.profile,
        constraint_set=context.constraints,
    )


def _empty_metrics() -> dict[str, float | int]:
    return {metric: 0 for metric in METRIC_IDS}


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _summarize_goal_reports(
    reports: list[GoalDomainAdequacyReportV1],
    providers: list[GoalSupportProvider],
    *,
    elapsed: float,
) -> dict[str, float | int]:
    metrics = _empty_metrics()
    legal_n = sum(len(report.legal_action_digests) for report in reports)
    observed_n = sum(report.goal_support_queries for report in reports)
    supported_n = sum(len(report.supported_action_digests) for report in reports)
    unsupported_n = sum(len(report.unsupported_action_digests) for report in reports)
    unknown_n = sum(len(report.unknown_action_digests) for report in reports)
    unobserved_n = sum(len(report.unobserved_action_digests) for report in reports)
    classifications = [report.classification for report in reports]
    cores = []
    for report, provider in zip(reports, providers, strict=True):
        for action in report.action_evidence:
            if action.base_certificate_digest is None:
                continue
            sidecar = provider.goal_result(action.base_certificate_digest)
            if sidecar.obstruction_core is not None:
                cores.append(sidecar.obstruction_core)
    metrics.update(
        exact_action_coverage_rate=_rate(observed_n, legal_n),
        supported_action_rate=_rate(supported_n, legal_n),
        unsupported_action_rate=_rate(unsupported_n, legal_n),
        unknown_action_rate=_rate(unknown_n, legal_n),
        unobserved_action_rate=_rate(unobserved_n, legal_n),
        selected_action_supported_rate=_rate(
            classifications.count("domain_adequate_selected_supported"), len(reports)
        ),
        selection_regret_rate=_rate(
            classifications.count("selection_regret"), len(reports)
        ),
        selection_unresolved_rate=_rate(
            classifications.count("selection_unresolved"), len(reports)
        ),
        domain_inadequate_under_bounds_rate=_rate(
            classifications.count("domain_inadequate_under_bounds"), len(reports)
        ),
        coverage_unknown_rate=_rate(
            classifications.count("coverage_unknown"), len(reports)
        ),
        obstruction_core_emission_rate=_rate(len(cores), unsupported_n),
        obstruction_core_replay_rate=(
            1.0 if cores and not sum(r.replay_failures for r in reports) else 0.0
        ),
        mean_obstruction_core_size=(
            round(sum(len(core.atom_ids) for core in cores) / len(cores), 6)
            if cores
            else 0.0
        ),
        support_certificate_replay_failure_count=sum(
            r.replay_failures for r in reports
        ),
        verifier_calls=sum(r.verifier_calls for r in reports),
        expanded_nodes=sum(r.expanded_nodes for r in reports),
        backtracks=sum(r.backtracks for r in reports),
        wall_time=round(elapsed, 6),
    )
    return metrics


def _run_goal_arm(
    context: _Context,
    campaign: GoalSupportDomainAdequacyCampaignV1,
    *,
    deadline: float,
) -> tuple[dict[str, Any], list[GoalDomainAdequacyReportV1], list[GoalSupportProvider]]:
    started = time.perf_counter()
    remaining = campaign.goal_support_query_cap
    reports: list[GoalDomainAdequacyReportV1] = []
    providers: list[GoalSupportProvider] = []
    for case in _CASES:
        if time.monotonic() >= deadline:
            raise TimeoutError("goal-support fixture campaign exceeded max_wall_minutes")
        state, selected = _state(case, context.constraints.digest)
        provider = _provider(state, context)
        cap = min(campaign.exact_action_cap, len(state.holes[0].values), remaining)
        report = analyze_goal_domain(
            state, provider, selected_action=selected, exact_action_cap=cap
        )
        remaining -= report.goal_support_queries
        reports.append(report)
        providers.append(provider)
    elapsed = time.perf_counter() - started
    return (
        {
            "metrics": _summarize_goal_reports(reports, providers, elapsed=elapsed),
            "case_reports": [report.to_dict() for report in reports],
            "query_cap": campaign.goal_support_query_cap,
            "query_cap_remaining": remaining,
        },
        reports,
        providers,
    )


def _run_structural_arm(
    campaign: GoalSupportDomainAdequacyCampaignV1,
    constraint_version: str,
    *,
    deadline: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    supported_n = unknown_n = unobserved_n = replay_failures = 0
    verifier_calls = expanded_nodes = backtracks = 0
    selected_supported = 0
    legal_n = 0
    remaining = campaign.goal_support_query_cap
    for case in _CASES:
        if time.monotonic() >= deadline:
            raise TimeoutError("goal-support fixture campaign exceeded max_wall_minutes")
        state, selected = _state(case, constraint_version)
        expander = _TerminalFixtureExpander(state)
        values = tuple(sorted(state.holes[0].values, key=goal_action_digest))
        cap = min(campaign.exact_action_cap, len(values), remaining)
        observed = values[:cap]
        unobserved = values[cap:]
        remaining -= len(observed)
        action_rows: list[dict[str, Any]] = []
        selected_verdict = None
        for value in observed:
            verifier = OpenUIWellFormedVerifier()
            query = SupportQuery(state.fingerprint, state.holes[0].hole_id, value)
            result = EnumerativeSupportOracle(expander, verifier).check(state, query)
            replay = replay_support_certificate(
                result.certificate, state=state, expander=expander, verifier=verifier
            )
            verdict = result.verdict if replay.ok else SupportVerdict.UNKNOWN
            replay_failures += int(not replay.ok)
            supported_n += int(verdict is SupportVerdict.SUPPORTED)
            unknown_n += int(verdict is SupportVerdict.UNKNOWN)
            verifier_calls += result.counters.verifier_calls + replay.counters.verifier_calls
            expanded_nodes += result.counters.nodes + replay.counters.nodes
            backtracks += result.counters.backtracks + replay.counters.backtracks
            action_digest = _digest(value.to_dict())
            if value == selected:
                selected_verdict = verdict
            action_rows.append(
                {
                    "action_digest": action_digest,
                    "observation_status": "observed",
                    "support_verdict": verdict.name,
                    "base_certificate_digest": result.certificate.digest,
                    "replay_valid": replay.ok,
                }
            )
        for value in unobserved:
            action_rows.append(
                {
                    "action_digest": _digest(value.to_dict()),
                    "observation_status": "unobserved",
                }
            )
        legal_n += len(values)
        unobserved_n += len(unobserved)
        selected_supported += int(selected_verdict is SupportVerdict.SUPPORTED)
        rows.append(
            {
                "case_id": case.case_id,
                "state_fingerprint": state.fingerprint,
                "selected_action_digest": _digest(selected.to_dict()),
                "action_evidence": sorted(
                    action_rows, key=lambda row: row["action_digest"]
                ),
            }
        )
    observed_n = legal_n - unobserved_n
    metrics = _empty_metrics()
    metrics.update(
        exact_action_coverage_rate=_rate(observed_n, legal_n),
        supported_action_rate=_rate(supported_n, legal_n),
        unknown_action_rate=_rate(unknown_n, legal_n),
        unobserved_action_rate=_rate(unobserved_n, legal_n),
        selected_action_supported_rate=_rate(selected_supported, len(_CASES)),
        coverage_unknown_rate=_rate(
            sum(
                any(
                    action.get("support_verdict") == "UNKNOWN"
                    or action["observation_status"] == "unobserved"
                    for action in row["action_evidence"]
                )
                for row in rows
            ),
            len(rows),
        ),
        support_certificate_replay_failure_count=replay_failures,
        verifier_calls=verifier_calls,
        expanded_nodes=expanded_nodes,
        backtracks=backtracks,
        wall_time=round(time.perf_counter() - started, 6),
    )
    return {"metrics": metrics, "case_reports": rows}


def _run_certified_arm(
    context: _Context,
    campaign: GoalSupportDomainAdequacyCampaignV1,
    *,
    deadline: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    result, reports, _ = _run_goal_arm(context, campaign, deadline=deadline)
    remaining = int(result["query_cap_remaining"])
    false_prunes = pruned_n = empty_domains = skipped_n = 0
    certified_rows: list[dict[str, Any]] = []
    closure_verifier_calls = closure_nodes = 0
    for case, report in zip(_CASES, reports, strict=True):
        if time.monotonic() >= deadline:
            raise TimeoutError("goal-support fixture campaign exceeded max_wall_minutes")
        state, _ = _state(case, context.constraints.digest)
        legal = set(report.legal_action_digests)
        unsupported = set(report.unsupported_action_digests)
        can_certify = (
            case.coverage == "complete"
            and not report.unknown_action_digests
            and not report.unobserved_action_digests
            and len(legal) <= remaining
        )
        if not can_certify:
            skipped_n += 1
            certified_rows.append(
                {
                    "case_id": case.case_id,
                    "status": "unchanged",
                    "reason_code": (
                        "incomplete_or_unresolved"
                        if case.coverage != "complete" or report.unknown_action_digests
                        else "query_cap_or_unobserved"
                    ),
                    "live_action_digests": sorted(legal),
                    "pruned_action_digests": [],
                }
            )
            continue
        closure = exact_goal_closure(state, _provider(state, context))
        remaining -= closure.counters.support_queries
        closure_verifier_calls += closure.counters.verifier_calls
        closure_nodes += closure.counters.expanded_nodes
        live = {_digest(value.to_dict()) for value in closure.state.holes[0].values}
        removed = legal - live
        false_prunes += len(removed - unsupported)
        pruned_n += len(removed)
        empty_domains += int(not live)
        certified_rows.append(
            {
                "case_id": case.case_id,
                "status": "certified",
                "reason_code": "exact_goal_closure",
                "live_action_digests": sorted(live),
                "pruned_action_digests": sorted(removed),
            }
        )
    result["metrics"]["false_hard_prune_count"] = false_prunes
    result["metrics"]["verifier_calls"] += closure_verifier_calls
    result["metrics"]["expanded_nodes"] += closure_nodes
    result["metrics"]["wall_time"] = round(time.perf_counter() - started, 6)
    result.update(
        certified_case_reports=certified_rows,
        goal_support_pruned=pruned_n,
        certified_empty_domains=empty_domains,
        certified_cases_skipped=skipped_n,
        query_cap_remaining=remaining,
    )
    return result


def _semantic_result_digest(result: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in json.loads(canonical_json(result)).items()
        if key not in {"canonical_result_digest", "event_chain_verified"}
    }
    arms = payload.get("arm_results", {})
    for arm in arms.values():
        if isinstance(arm, dict) and isinstance(arm.get("metrics"), dict):
            arm["metrics"] = {
                key: value
                for key, value in arm["metrics"].items()
                if key != "wall_time"
            }
    return _digest(payload)


def run_campaign(
    campaign: GoalSupportDomainAdequacyCampaignV1, *, root: Path
) -> dict[str, Any]:
    """Run and persist one bounded, fixture-only governed campaign."""
    deadline = time.monotonic() + campaign.max_wall_minutes * 60.0
    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="Fixture diagnosis of grammar-legal goal support and inadequacy",
            primary_metric="false_hard_prune_count",
            budget=campaign.manifest().budget,
            created_at="1970-01-01T00:00:00Z",
        )
    )
    lock = store.lock_experiment_campaign(campaign.manifest())
    store.append_event(
        "experiment_started",
        experiment_id=campaign.campaign_id,
        status="fixture",
        detail={"campaign_manifest_sha256": lock.manifest_sha256},
    )
    production, evaluation = _contexts()
    structural = _run_structural_arm(
        campaign, production.constraints.digest, deadline=deadline
    )
    production_result, production_reports, _ = _run_goal_arm(
        production, campaign, deadline=deadline
    )
    evaluation_result, evaluation_reports, _ = _run_goal_arm(
        evaluation, campaign, deadline=deadline
    )
    certified = _run_certified_arm(production, campaign, deadline=deadline)

    for arm, reports in (
        (production_result, production_reports),
        (evaluation_result, evaluation_reports),
        (certified, production_reports),
    ):
        structural_goal_mismatches = sum(
            len(
                {
                    action["action_digest"]
                    for action in structural_case["action_evidence"]
                    if action.get("support_verdict") == "SUPPORTED"
                }
                & set(report.unsupported_action_digests)
            )
            for structural_case, report in zip(
                structural["case_reports"], reports, strict=True
            )
        )
        observed = sum(report.goal_support_queries for report in reports)
        arm["metrics"]["structural_supported_but_goal_unsupported_rate"] = _rate(
            structural_goal_mismatches, observed
        )

    result = {
        "schema": "proof_carrying_goal_support_fixture_result/v1",
        "campaign_id": campaign.campaign_id,
        "manifest_sha256": lock.manifest_sha256,
        "claim_class": "fixture",
        "promotion": False,
        "suite_n": len(_CASES),
        "seed": campaign.seed,
        "problem_digests": {
            "production_exact": production.problem.digest,
            "evaluation_oracle": evaluation.problem.digest,
        },
        "request_digests": {
            "production_exact": production.constraints.request_digest,
            "evaluation_oracle": evaluation.constraints.request_digest,
        },
        "constraint_set_digests": {
            "production_exact": production.constraints.digest,
            "evaluation_oracle": evaluation.constraints.digest,
        },
        "profile_digests": {
            "production_exact": production.profile.digest,
            "evaluation_oracle": evaluation.profile.digest,
        },
        "pack_identity_digests": {
            "production_exact": production.constraints.pack_identity_digest,
            "evaluation_oracle": evaluation.constraints.pack_identity_digest,
        },
        "bounds": {
            "solver": BOUNDS.to_dict(),
            "exact_action_cap": campaign.exact_action_cap,
            "goal_support_query_cap": campaign.goal_support_query_cap,
            "max_wall_minutes": campaign.max_wall_minutes,
        },
        "arm_results": {
            ARM_IDS[0]: structural,
            ARM_IDS[1]: production_result,
            ARM_IDS[2]: evaluation_result,
            ARM_IDS[3]: certified,
        },
        "metric_definitions": {
            "action_rates": "counts divided by all grammar-legal actions, including unobserved",
            "classification_rates": "case count divided by suite_n",
            "obstruction_core_emission_rate": "replayable sidecars with a core divided by unsupported observed actions",
            "obstruction_core_replay_rate": "one iff every emitted core rederived during sidecar replay",
            "wall_time": "complete local in-process arm seconds; excluded from canonical semantic digest",
        },
        "event_chain_verified": True,
        "scope_disclaimer": (
            "Deterministic local fixture/diagnostic evidence only. It proves bounded-domain "
            "support classifications under the pinned identities above; it makes no model-quality, "
            "ship-readiness, checkpoint, global unrealizability, or prompt-meaning claim."
        ),
        "checkpoint_created": False,
        "model_card_updated": False,
    }
    result["canonical_result_digest"] = _semantic_result_digest(result)
    store.verify_event_chain()
    artifact = store.write_artifact("goal_support_domain_adequacy_result", result)
    store.append_event(
        "experiment_finished",
        experiment_id=campaign.campaign_id,
        status="fixture",
        artifact_sha256=artifact.stem,
        detail={
            "campaign_manifest_sha256": lock.manifest_sha256,
            "canonical_result_digest": result["canonical_result_digest"],
            "false_hard_prune_count": certified["metrics"]["false_hard_prune_count"],
        },
    )
    store.verify_event_chain()
    return result
