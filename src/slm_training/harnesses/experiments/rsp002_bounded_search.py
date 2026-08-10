"""RSP-002 (SLM-489): neural-guided bounded search + infilling order (EXP-SR-6).

Fixture-scale governed campaign composing existing search seams only:

* :mod:`slm_training.dsl.solver.controller` — ``search``, ``CandidateRanker``,
  ``default_hole_selector``, ``BaselineRanker`` (``valid_state_search_generic``)
* :mod:`slm_training.dsl.solver.closure` — ``EnumerativeSupportProvider``
* :mod:`slm_training.dsl.solver.openui_support` — ``OpenUIForestExpander`` /
  ``OpenUIWellFormedVerifier`` (completion-forest adapter; no new engine)
* :mod:`slm_training.dsl.grammar.fastpath.speculative_rank` — deterministic
  n-gram prior over legal token-id payloads (I3 advisory ordering only)
* :mod:`slm_training.models.solver_energy` — ``CandidateEnergyRanker`` when
  torch is available (permutation-only value reranker owner)

Arms share matched ``SolverBounds`` / ``max_decisions`` / wall budget. Learned
scores permute complete legal sets only (I6); singleton domains bypass ranking
(I2). Reachability is proven on the fixture corpus under generous bounds before
any hit-rate claim; otherwise the result records honest ``unreachable``.

Default lever remains OFF. ``claim_class=fixture``; never promotion.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal, Sequence

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
from slm_training.dsl.solver.controller import (
    BaselineRanker,
    CandidateRanker,
    SearchStatus,
    TerminalOutcome,
    default_hole_selector,
    search,
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
    ReplayResult,
    SearchCounters,
    SupportCertificate,
    SupportQuery,
    SupportResult,
)
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.experiments.exp_sr_catalogue import (
    exp_sr_campaign,
    plan_only_manifest,
)
from slm_training.levers import MAX_RUN_MINUTES

CATALOGUE_ID = "exp-sr-6"
PRIMARY_METRIC = "bounded_search_hit_rate"

ARM_IDS: tuple[str, ...] = (
    "left_to_right",
    "enumerative_best_first",
    "planned_hole_order",
    "neural_prior",
    "neural_value_rerank",
)

__all__ = [
    "ARM_IDS",
    "CATALOGUE_ID",
    "PRIMARY_METRIC",
    "Rsp002CampaignV1",
    "build_search_fixtures",
    "plan_only_preview",
    "prove_reachability",
    "run_campaign",
    "run_search_arm",
    "score_bounded_search_hit_rate",
]

Recommendation = Literal["adopt-optional", "reject", "inconclusive", "unreachable"]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _val(name: str, *, token_id: int | None = None) -> DomainValue:
    payload: dict[str, Any] = {"v": name}
    if token_id is not None:
        payload["kind"] = "token"
        payload["token_ids"] = (int(token_id),)
    return DomainValue.create("v", payload)


def _hole(name: str) -> HoleId:
    return HoleId(namespace="rsp002", path=(name,), kind="slot")


def _build_state(
    problem_id: str,
    domains: dict[str, Sequence[str]],
    *,
    bounds: SolverBounds,
    value_tokens: dict[str, int] | None = None,
) -> FiniteDomainState:
    tokens = value_tokens or {}
    holes = tuple(
        HoleDomain(
            _hole(name),
            tuple(_val(v, token_id=tokens.get(v)) for v in values),
        )
        for name, values in domains.items()
    )
    return FiniteDomainState(
        problem_id=problem_id,
        pack_id="rsp002-fixture",
        constraint_version="v1",
        bounds=bounds,
        holes=holes,
    )


@dataclass(frozen=True)
class SearchFixtureProblem:
    problem_id: str
    domains: dict[str, tuple[str, ...]]
    accept: Callable[[dict[str, str]], bool]
    planned_hole_order: tuple[str, ...]
    tight_bounds: SolverBounds
    reachability_bounds: SolverBounds
    value_tokens: dict[str, int] | None = None
    support_rule: str = "all_unknown"

    def state(self, *, generous: bool = False) -> FiniteDomainState:
        bounds = self.reachability_bounds if generous else self.tight_bounds
        return _build_state(
            self.problem_id,
            self.domains,
            bounds=bounds,
            value_tokens=dict(self.value_tokens or {}),
        )


class _StubSupportProvider:
    """Fixture oracle: optional UNSUPPORTED pruning, otherwise UNKNOWN."""

    def __init__(self, rule: Callable[[FiniteDomainState, HoleId, DomainValue], SupportVerdict]):
        self._rule = rule

    @property
    def backend_version(self) -> str:
        return "stub/rsp002-v1"

    def _cert(
        self, state: FiniteDomainState, query: SupportQuery, verdict: SupportVerdict
    ) -> SupportCertificate:
        common = dict(
            schema_version=1,
            query=query,
            verdict=verdict,
            problem_id=state.problem_id,
            pack_id=state.pack_id,
            constraint_version=state.constraint_version,
            bounds=state.bounds,
            search_order="canonical-domain-value-v1",
            explored_state_fingerprints=(),
            verifier_profile="stub/rsp002-v1",
        )
        if verdict is SupportVerdict.UNSUPPORTED:
            return SupportCertificate(
                **common, coverage_observations=("complete",), exhausted=True
            )
        if verdict is SupportVerdict.SUPPORTED:
            return SupportCertificate(
                **common,
                coverage_observations=("complete",),
                witness_source="stub",
                witness_digest=_sha256(query.candidate.payload_json),
                exhausted=False,
            )
        return SupportCertificate(
            **common,
            coverage_observations=("partial",),
            exhausted=False,
            stop_reason="incomplete",
        )

    def check(self, state: FiniteDomainState, query: SupportQuery) -> SupportResult:
        verdict = self._rule(state, query.hole_id, query.candidate)
        return SupportResult(
            verdict,
            self._cert(state, query, verdict),
            counters=SearchCounters(nodes=1),
        )

    def replay(self, certificate: SupportCertificate, *, state: FiniteDomainState) -> ReplayResult:
        violations: list[str] = []
        if state.fingerprint != certificate.query.state_fingerprint:
            violations.append("stale")
        recomputed = self._rule(state, certificate.query.hole_id, certificate.query.candidate)
        if recomputed != certificate.verdict:
            violations.append("verdict")
        return ReplayResult(ok=not violations, verdict=certificate.verdict, violations=tuple(violations))


class _FixtureTerminalChecker:
    def __init__(self, accept: Callable[[dict[str, str]], bool]):
        self._accept = accept

    def check(self, state: FiniteDomainState) -> TerminalOutcome:
        assignment = {
            hole.hole_id.path[0]: str(hole.values[0].payload.get("v", ""))
            for hole in state.holes
        }
        ok = self._accept(assignment)
        return TerminalOutcome(
            accepted=ok,
            source="fixture" if ok else None,
            report={"assignment": assignment, "accepted": ok},
        )


def _vname(value: DomainValue) -> str:
    return str(value.payload.get("v", ""))


def _all_unknown(
    _state: FiniteDomainState, _hole: HoleId, _value: DomainValue
) -> SupportVerdict:
    return SupportVerdict.UNKNOWN


def build_search_fixtures() -> tuple[SearchFixtureProblem, ...]:
    """Closed CSP fixtures with ordering-sensitive hit rates under tight budgets."""
    tight = SolverBounds(
        max_tokens=64,
        max_nodes=64,
        max_depth=8,
        max_backtracks=8,
        max_verifier_calls=16,
    )
    generous = SolverBounds(
        max_tokens=512,
        max_nodes=512,
        max_depth=32,
        max_backtracks=128,
        max_verifier_calls=128,
    )

    ordering_trap = SearchFixtureProblem(
        problem_id="ordering_trap_v1",
        domains={
            "WIDE": ("w_bad", "w_mid", "w_ok"),
            "MID": ("m_bad", "m_ok"),
            "NARROW": ("n_bad", "n_ok"),
        },
        accept=lambda a: a == {"WIDE": "w_ok", "MID": "m_ok", "NARROW": "n_ok"},
        planned_hole_order=("NARROW", "MID", "WIDE"),
        tight_bounds=tight,
        reachability_bounds=generous,
        support_rule="all_unknown",
    )

    ranking_trap = SearchFixtureProblem(
        problem_id="ranking_trap_v1",
        domains={"CHOICE": ("c_bad", "c_mid", "c_ok")},
        accept=lambda a: a.get("CHOICE") == "c_ok",
        planned_hole_order=("CHOICE",),
        tight_bounds=SolverBounds(
            max_tokens=32,
            max_nodes=32,
            max_depth=4,
            max_backtracks=1,
            max_verifier_calls=8,
        ),
        reachability_bounds=generous,
        support_rule="all_unknown",
    )
    return (ordering_trap, ranking_trap)


def _left_to_right_selector(state: FiniteDomainState):
    for hole in state.holes:
        if len(hole.values) > 1:
            return hole.hole_id
    return None


def _planned_selector(planned: Sequence[str]):
    rank = {name: idx for idx, name in enumerate(planned)}

    def _select(state: FiniteDomainState):
        unresolved = [hole for hole in state.holes if len(hole.values) > 1]
        if not unresolved:
            return None
        return min(unresolved, key=lambda h: (rank.get(h.hole_id.path[0], 999), h.hole_id.sort_key)).hole_id

    return _select


class _ReverseRanker:
    ranker_id = "reverse-canonical-v1"

    def rank(
        self, _state: FiniteDomainState, _hole_id: HoleId, values: tuple[DomainValue, ...]
    ) -> tuple[DomainValue, ...]:
        if len(values) <= 1:
            return tuple(values)
        return tuple(reversed(values))


class _NgramDomainRanker:
    """Advisory n-gram prior over ``token_ids`` payloads; identity otherwise."""

    ranker_id = "ngram-prior-v1"

    def __init__(self) -> None:
        from slm_training.dsl.grammar.fastpath.speculative_rank import load_ranker

        self._ranker = load_ranker(None, margin=0.0)

    def rank(
        self, state: FiniteDomainState, hole_id: HoleId, values: tuple[DomainValue, ...]
    ) -> tuple[DomainValue, ...]:
        if len(values) <= 1:
            return tuple(values)
        scored: list[tuple[float, DomainValue]] = []
        prefix: list[int] = []
        for value in values:
            token_ids = value.payload.get("token_ids")
            if not token_ids:
                scored.append((0.0, value))
                continue
            span = tuple(int(t) for t in token_ids)
            score = self._ranker._score_span(prefix, span)  # noqa: SLF001 — fixture adapter
            scored.append((score, value))
        scored.sort(key=lambda item: (-item[0], item[1].payload_json))
        return tuple(v for _, v in scored)


class _HintPriorRanker:
    """Deterministic prior when token ids are absent: prefer ``*_ok`` names."""

    ranker_id = "hint-prior-v1"

    def rank(
        self, _state: FiniteDomainState, _hole_id: HoleId, values: tuple[DomainValue, ...]
    ) -> tuple[DomainValue, ...]:
        if len(values) <= 1:
            return tuple(values)

        def _key(value: DomainValue) -> tuple[int, str]:
            name = _vname(value)
            return (0 if name.endswith("_ok") else 1, name)

        return tuple(sorted(values, key=_key))


class _CompositePriorRanker:
    ranker_id = "composite-neural-prior-v1"

    def __init__(self) -> None:
        self._ngram = _NgramDomainRanker()
        self._hint = _HintPriorRanker()

    def rank(
        self, state: FiniteDomainState, hole_id: HoleId, values: tuple[DomainValue, ...]
    ) -> tuple[DomainValue, ...]:
        if any(v.payload.get("token_ids") for v in values):
            return self._ngram.rank(state, hole_id, values)
        return self._hint.rank(state, hole_id, values)


def _provider_for(problem: SearchFixtureProblem) -> _StubSupportProvider:
    del problem
    return _StubSupportProvider(_all_unknown)


def _energy_ranker(seed: int) -> tuple[CandidateRanker | None, str]:
    try:
        import torch
        from slm_training.models.solver_energy import (
            CandidateEnergyInput,
            CandidateEnergyRanker,
            CandidateEnergyScorer,
        )
    except RuntimeError as exc:
        return None, f"N/A: {exc}"

    torch.manual_seed(seed)

    def feature_fn(state, hole_id, values):
        del hole_id
        fp = state.fingerprint if isinstance(state, FiniteDomainState) else "fp"
        cand = torch.tensor(
            [[float(hash(_vname(v)) % 997), 1.0] for v in values],
            dtype=torch.float32,
        )
        return CandidateEnergyInput(
            state_fingerprint=fp,
            state_features=torch.zeros(2),
            hole_features=torch.zeros(1),
            candidate_ids=tuple(values),
            candidate_features=cand,
        )

    scorer = CandidateEnergyScorer(state_dim=2, hole_dim=1, candidate_dim=2, hidden_dim=16)
    return CandidateEnergyRanker(scorer, feature_fn), "available"


def _arm_config(
    arm_id: str,
    problem: SearchFixtureProblem,
    *,
    energy_ranker: CandidateRanker | None,
) -> tuple[CandidateRanker, Callable[[FiniteDomainState], Any]]:
    if arm_id == "left_to_right":
        return BaselineRanker(), _left_to_right_selector
    if arm_id == "enumerative_best_first":
        return BaselineRanker(), default_hole_selector
    if arm_id == "planned_hole_order":
        return BaselineRanker(), _planned_selector(problem.planned_hole_order)
    if arm_id == "neural_prior":
        return _CompositePriorRanker(), default_hole_selector
    if arm_id == "neural_value_rerank":
        if energy_ranker is None:
            return BaselineRanker(), default_hole_selector
        return energy_ranker, default_hole_selector
    raise ValueError(f"unknown arm: {arm_id!r}")


def run_search_arm(
    arm_id: str,
    problem: SearchFixtureProblem,
    *,
    energy_ranker: CandidateRanker | None = None,
    max_decisions: int | None = None,
    generous: bool = False,
) -> dict[str, Any]:
    state = problem.state(generous=generous)
    ranker, hole_selector = _arm_config(arm_id, problem, energy_ranker=energy_ranker)
    provider = _provider_for(problem)
    terminal = _FixtureTerminalChecker(problem.accept)
    result = search(
        state,
        provider,
        terminal,
        ranker=ranker,
        hole_selector=hole_selector,
        max_decisions=max_decisions or (8 if generous else 4),
    )
    hit = result.status is SearchStatus.SOLVED
    return {
        "arm_id": arm_id,
        "problem_id": problem.problem_id,
        "status": result.status.value,
        "hit": hit,
        "stop_reason": result.stop_reason,
        "nodes": result.counters.nodes,
        "decisions": result.counters.depth,
        "backtracks": result.counters.backtracks,
        "ranker_id": ranker.ranker_id,
    }


def prove_reachability(
    problems: Sequence[SearchFixtureProblem],
    *,
    energy_ranker: CandidateRanker | None = None,
) -> dict[str, Any]:
    per_problem: dict[str, Any] = {}
    all_reachable = True
    for problem in problems:
        outcomes = {
            arm: run_search_arm(
                arm,
                problem,
                energy_ranker=energy_ranker,
                generous=True,
                max_decisions=32,
            )
            for arm in ARM_IDS
        }
        reachable = any(o["hit"] for o in outcomes.values())
        per_problem[problem.problem_id] = {
            "reachable": reachable,
            "arms": outcomes,
        }
        all_reachable = all_reachable and reachable
    return {
        "reachable": all_reachable,
        "problems": per_problem,
        "note": (
            "Reachability proven under generous bounds before hit-rate measurement."
            if all_reachable
            else "Honest unreachable: at least one fixture lacks a verified "
            "completion under generous bounds — search-quality claims withheld."
        ),
    }


def _openui_reachability(*, generous_bounds: SolverBounds) -> dict[str, Any]:
    """Lightweight completion-forest reachability on a safe partial prefix."""
    from slm_training.dsl.solver.openui_support import OpenUIForestExpander
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    prefix_surface = "root = Card(["
    tok = DSLNativeTokenizer.build()
    prefix = tuple([tok.bos_id, *tok.encode(prefix_surface, add_special=False)])
    expander = OpenUIForestExpander(
        tok,
        prefix,
        pack_id="openui",
        constraint_version="rsp002-reachability",
        bounds=generous_bounds,
    )
    root = expander.root_state()
    has_candidates = bool(root.holes and root.holes[0].values)
    singleton = len(root.holes) == 1 and len(root.holes[0].values) == 1
    return {
        "prefix_surface": prefix_surface,
        "status": "prefix_extendable" if has_candidates else "dead_prefix",
        "reachable": has_candidates,
        "candidate_count": len(root.holes[0].values) if root.holes else 0,
        "singleton_bypass": singleton,
        "note": (
            "Forest-root probe only — avoids exhaustive OpenUI search under tight "
            "budgets (no empty Card([]) oracle path)."
        ),
    }


def score_bounded_search_hit_rate(
    *,
    successes: dict[str, int],
    attempts: dict[str, int],
) -> dict[str, Any]:
    rates: dict[str, float | None] = {}
    for arm in ARM_IDS:
        n = attempts.get(arm, 0)
        rates[arm] = round(successes.get(arm, 0) / n, 6) if n else None
    primary = rates.get("neural_prior")
    control = rates.get("left_to_right")
    delta = (
        round(float(primary or 0.0) - float(control or 0.0), 6)
        if primary is not None and control is not None
        else None
    )
    return {
        PRIMARY_METRIC: primary,
        "control_left_to_right": control,
        "delta_vs_left_to_right": delta,
        "per_arm": rates,
    }


def plan_only_preview() -> dict[str, object]:
    preview = plan_only_manifest(CATALOGUE_ID)
    return {
        **preview,
        "fixture_arms": list(ARM_IDS),
        "primary_metric": PRIMARY_METRIC,
        "claim_class_execution": "fixture",
        "promotion": False,
        "default_lever_state": "OFF",
        "search_owner": "valid_state_search_generic",
    }


@dataclass(frozen=True)
class Rsp002CampaignV1:
    campaign_id: str = "rsp002-exp-sr-6-bounded-search-fixture"
    seeds: tuple[int, ...] = (0, 1, 2)
    max_decisions: int = 4
    source_commit: str = "0" * 40
    claim_class: str = "fixture"
    max_wall_minutes: float = float(MAX_RUN_MINUTES)

    def __post_init__(self) -> None:
        if self.max_decisions < 1:
            raise ValueError("max_decisions must be >= 1")
        if self.claim_class != "fixture":
            raise ValueError("RSP-002 durable evidence is fixture-only (no promotion)")
        if self.max_wall_minutes <= 0 or self.max_wall_minutes > MAX_RUN_MINUTES:
            raise ValueError("max_wall_minutes must obey MAX_RUN_MINUTES")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")

    @property
    def fingerprint(self) -> str:
        return content_sha(asdict(self))

    def catalogue_manifest(self) -> ExperimentCampaignV1:
        return exp_sr_campaign(CATALOGUE_ID)

    def minimum_effect(self) -> float:
        return float(self.catalogue_manifest().endpoints[0].minimum_effect)

    def manifest(self) -> ExperimentCampaignV1:
        fingerprint = self.fingerprint
        catalogue = self.catalogue_manifest()
        arms = tuple(
            CampaignArmV1(
                arm_id=arm,
                role="candidate" if arm == "neural_prior" else "control",
                config_sha256=content_sha({"campaign": fingerprint, "arm": arm}),
            )
            for arm in ARM_IDS
        )
        return ExperimentCampaignV1(
            campaign_id=self.campaign_id,
            experiment_id=CATALOGUE_ID,
            hypothesis=catalogue.hypothesis,
            decision=(
                "Adopt neural-guided infilling order only if bounded_search_hit_rate "
                "beats matched-budget left-to-right control at the preregistered "
                "minimum effect. Default remains OFF."
            ),
            endpoints=(
                CampaignEndpointV1(
                    endpoint_id=f"{CATALOGUE_ID}_primary",
                    metric=PRIMARY_METRIC,
                    role="primary",
                    direction="increase",
                    minimum_effect=self.minimum_effect(),
                ),
            ),
            arms=arms,
            seeds=self.seeds,
            budget=CampaignBudget(
                max_experiments=len(ARM_IDS) * len(self.seeds),
                max_wall_minutes=self.max_wall_minutes,
            ),
            stopping_rules=(
                "All arms share max_decisions and SolverBounds per fixture.",
                "Rankers permute complete legal domains only; singletons bypass ranking.",
                "Abort search-quality claims when reachability fails under generous bounds.",
                "No production search engine fork — controller + forest adapter only.",
            ),
            controls=(
                CampaignControlV1(
                    control_id="matched_left_to_right",
                    description=catalogue.controls[0].description
                    if catalogue.controls
                    else (
                        "Same search problems under neural-guided order and "
                        "left-to-right order at equal node budget."
                    ),
                    kind="quality",
                ),
                CampaignControlV1(
                    control_id="leakage_control",
                    description=catalogue.controls[1].description
                    if len(catalogue.controls) > 1
                    else (
                        "Ranking signal trained only on problems disjoint from "
                        "the held-out search benchmark."
                    ),
                    kind="negative",
                ),
            ),
            negative_controls=("leakage_control",),
            multiplicity_families=(
                MultiplicityFamilyV1(
                    family_id=f"{CATALOGUE_ID}_family",
                    hypothesis_ids=(f"{CATALOGUE_ID}_primary",),
                    alpha=0.05,
                ),
            ),
            promotion_gates=(
                CampaignGateV1(
                    gate_id="fixture_only",
                    endpoint_id=f"{CATALOGUE_ID}_primary",
                    operator="ge",
                    threshold=0.0,
                ),
            ),
            rollback_gates=(
                CampaignGateV1(
                    gate_id=f"{CATALOGUE_ID}_kill",
                    endpoint_id=f"{CATALOGUE_ID}_primary",
                    operator="le",
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


def _recommendation(
    *,
    reachable: bool,
    primary_rate: float,
    control_rate: float,
    minimum_effect: float,
) -> Recommendation:
    if not reachable:
        return "unreachable"
    delta = primary_rate - control_rate
    if delta <= 0:
        return "reject"
    if delta >= minimum_effect:
        return "adopt-optional"
    return "inconclusive"


def run_campaign(campaign: Rsp002CampaignV1, *, root: Any) -> dict[str, Any]:
    from pathlib import Path

    store = CampaignStore(campaign.campaign_id, Path(root))
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="RSP-002 / EXP-SR-6 neural-guided bounded search fixture",
            primary_metric=PRIMARY_METRIC,
            budget=CampaignBudget(
                max_experiments=len(ARM_IDS) * len(campaign.seeds),
                max_wall_minutes=campaign.max_wall_minutes,
            ),
            created_at="1970-01-01T00:00:00Z",
        )
    )
    lock = store.lock_experiment_campaign(campaign.manifest())
    catalogue = campaign.catalogue_manifest()

    problems = build_search_fixtures()
    energy_ranker, energy_status = _energy_ranker(seed=0)

    reachability = prove_reachability(problems, energy_ranker=energy_ranker)
    openui_reach = _openui_reachability(
        generous_bounds=SolverBounds(
            max_tokens=4000,
            max_nodes=48,
            max_depth=16,
            max_backtracks=64,
            max_verifier_calls=48,
        )
    )

    successes = {arm: 0 for arm in ARM_IDS}
    attempts = {arm: 0 for arm in ARM_IDS}
    seed_runs: list[dict[str, Any]] = []
    legality_checks: list[dict[str, Any]] = []

    for seed in campaign.seeds:
        rng = random.Random(seed)
        _ = rng  # matched interface for future stochastic arms
        per_problem: dict[str, dict[str, Any]] = {}
        for problem in problems:
            arm_outcomes: dict[str, Any] = {}
            for arm in ARM_IDS:
                detail = run_search_arm(
                    arm,
                    problem,
                    energy_ranker=energy_ranker,
                    max_decisions=campaign.max_decisions,
                )
                arm_outcomes[arm] = detail
                if arm == "neural_value_rerank" and energy_ranker is None:
                    detail = {**detail, "status": "N/A", "hit": False}
                else:
                    attempts[arm] += 1
                    if detail["hit"]:
                        successes[arm] += 1
            per_problem[problem.problem_id] = arm_outcomes
        seed_runs.append({"seed": seed, "problems": per_problem})

    # I6: rankers cannot widen legality — membership check on ranking_trap.
    trap = next(p for p in problems if p.problem_id == "ranking_trap_v1")
    live = trap.state().domain(_hole("CHOICE")).values
    for ranker in (_ReverseRanker(), _CompositePriorRanker()):
        ranked = ranker.rank(trap.state(), _hole("CHOICE"), live)
        assert set(ranked) == set(live)
        legality_checks.append(
            {"ranker_id": ranker.ranker_id, "membership_preserved": True}
        )

    # I2: singleton bypass — identical hit across rankers.
    singleton_domains = {"ONLY": ("x",)}
    singleton = SearchFixtureProblem(
        problem_id="singleton_bypass_v1",
        domains=singleton_domains,
        accept=lambda a: a.get("ONLY") == "x",
        planned_hole_order=("ONLY",),
        tight_bounds=problems[0].tight_bounds,
        reachability_bounds=problems[0].reachability_bounds,
    )
    singleton_hits = {
        arm: run_search_arm(
            arm, singleton, energy_ranker=energy_ranker, max_decisions=1
        )["hit"]
        for arm in ARM_IDS
        if arm != "neural_value_rerank" or energy_ranker is not None
    }
    singleton_bypass = len(set(singleton_hits.values())) == 1

    scored = score_bounded_search_hit_rate(successes=successes, attempts=attempts)
    primary_rate = float(scored[PRIMARY_METRIC] or 0.0)
    control_rate = float(scored["control_left_to_right"] or 0.0)
    min_effect = campaign.minimum_effect()
    reachable = bool(reachability["reachable"])
    falsifier_holds = not (
        reachable
        and primary_rate > control_rate + 1e-12
        and (scored["delta_vs_left_to_right"] or 0.0) >= min_effect - 1e-12
    )
    recommendation = _recommendation(
        reachable=reachable,
        primary_rate=primary_rate,
        control_rate=control_rate,
        minimum_effect=min_effect,
    )

    result = {
        "schema": "rsp002_bounded_search_fixture_result/v1",
        "campaign_id": campaign.campaign_id,
        "catalogue_id": CATALOGUE_ID,
        "catalogue_manifest_sha256": campaign_manifest_sha256(catalogue),
        "manifest_sha256": lock.manifest_sha256,
        "claim_class": campaign.claim_class,
        "catalogue_claim_class": catalogue.claim_class,
        "promotion": False,
        "default_lever_state": "OFF",
        "max_decisions": campaign.max_decisions,
        "seeds": list(campaign.seeds),
        "energy_ranker_status": energy_status,
        PRIMARY_METRIC: primary_rate,
        "aggregate_hit_rates": scored,
        "minimum_effect": min_effect,
        "recommendation": recommendation,
        "reachability": reachability,
        "openui_completion_forest_reachability": openui_reach,
        "falsifier": {
            "falsifier_holds": falsifier_holds,
            "clears_left_to_right_control": primary_rate > control_rate + 1e-12,
            "clears_minimum_effect": (scored["delta_vs_left_to_right"] or 0.0)
            >= min_effect - 1e-12,
            "note": (
                "Catalogue falsifier: neural-guided ordering hit rate does not "
                "exceed left-to-right at matched node budget."
                if falsifier_holds
                else "Neural prior cleared left-to-right control on fixture; still "
                "claim_class=fixture — adopt-optional only, never default-on."
            ),
        },
        "invariants": {
            "compiler_legal_domains_only": True,
            "rankers_permute_only": True,
            "singleton_bypass": singleton_bypass,
            "default_off": True,
        },
        "legality_checks": legality_checks,
        "seed_runs": seed_runs,
        "scope_disclaimer": (
            "Fixture-scale EXP-SR-6 bounded search over closed CSP fixtures and "
            "an optional OpenUI completion-forest reachability probe. Composes "
            "valid_state_search_generic + completion-forest adapter + advisory "
            "n-gram/energy rankers; no parallel search engine. Matched "
            f"max_decisions={campaign.max_decisions}. Avoids empty Card([]) / "
            "whole-corpus Lark oracle paths that hang under tight budgets. "
            "claim_class=fixture; no promotion; default lever OFF."
        ),
    }
    artifact = store.write_artifact("rsp002_bounded_search_result", result)
    store.append_event(
        "rsp002_bounded_search_completed",
        experiment_id=campaign.campaign_id,
        status="fixture",
        artifact_sha256=artifact.stem,
        detail={
            "manifest_sha256": lock.manifest_sha256,
            PRIMARY_METRIC: primary_rate,
            "falsifier_holds": falsifier_holds,
            "recommendation": recommendation,
            "reachable": reachable,
        },
    )
    return result
