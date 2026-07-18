"""Capsule solve plans, SCC joint solving, and interface summaries (VSS2-02).

This module is the model-independent coordinator that solves verification
capsules in dependency order, treats SCCs as joint finite problems, propagates
explicit interface summaries, and always performs a final whole-program
verification. It keeps the hard/soft separation strict:

* **Certified deduction** — domain reductions produced by ``exact_closure``.
* **Reversible decision** — branch choices recorded by the search controller.
* **Local nogood** — request-local conflict memory; NOT a certified deduction.
* **Interface summary** — a conservative or exact capsule boundary description.
* **Local/global disagreement** — when a capsule-local oracle passes but the
  authoritative whole-program verifier fails.

The layer is Torch-free. It delegates hole-domain construction, summary
extraction, materialization, and final verification to pack-owned hooks so that
partial packs fail closed when capsule solving is requested.

Semantics are owned by ``docs/design/verified-scope-solver.md`` and
``docs/design/vss2-02-capsule-solver.md``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from slm_training.data.progspec.capsules import (
    CapsuleGraph,
    DependencyKind,
    VerificationCapsule,
)
from slm_training.dsl.solver.closure import SupportProvider
from slm_training.dsl.solver.controller import (
    CandidateRanker,
    SearchResult,
    TerminalChecker,
    TerminalOutcome,
    search,
)
from slm_training.dsl.solver.state import (
    DomainValue,
    FiniteDomainState,
    SolverBounds,
)
from slm_training.dsl.solver.support import SearchCounters

JsonValue = Any


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Interface summary contracts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BindingSummary:
    """One named binder crossing a capsule boundary.

    ``value`` carries the solved DomainValue when the summary is extracted from a
    concrete solved state. It is ``None`` for unresolved inputs or conservative
    placeholders.
    """

    name: str
    kind: str
    origin: str = ""
    value: DomainValue | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "origin": self.origin,
            "value": self.value.to_dict() if self.value is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BindingSummary:
        value = data.get("value")
        return cls(
            name=str(data["name"]),
            kind=str(data["kind"]),
            origin=str(data.get("origin", "")),
            value=DomainValue.from_dict(value) if value is not None else None,
        )


@dataclass(frozen=True)
class SlotSummary:
    """One external slot input required by a capsule."""

    name: str
    kind: str = "slot"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SlotSummary:
        return cls(name=str(data["name"]), kind=str(data.get("kind", "slot")))


@dataclass(frozen=True)
class CapsuleInterfaceSummary:
    """Conservative or exact boundary description for one solved capsule."""

    capsule_id: str
    input_bindings: tuple[BindingSummary, ...]
    output_bindings: tuple[BindingSummary, ...]
    slots: tuple[SlotSummary, ...]
    preconditions: tuple[str, ...]
    postconditions: tuple[str, ...]
    effects: tuple[str, ...]
    exceptions: tuple[str, ...]
    captures: tuple[str, ...]
    conservative: bool
    fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.capsule_id, str) or not self.capsule_id:
            raise ValueError("CapsuleInterfaceSummary requires a non-empty capsule_id")
        if not isinstance(self.fingerprint, str) or len(self.fingerprint) != 64:
            raise ValueError("CapsuleInterfaceSummary requires a SHA-256 fingerprint")

    def to_dict(self) -> dict[str, Any]:
        return {
            "capsule_id": self.capsule_id,
            "input_bindings": [b.to_dict() for b in self.input_bindings],
            "output_bindings": [b.to_dict() for b in self.output_bindings],
            "slots": [s.to_dict() for s in self.slots],
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "effects": list(self.effects),
            "exceptions": list(self.exceptions),
            "captures": list(self.captures),
            "conservative": self.conservative,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapsuleInterfaceSummary:
        return cls(
            capsule_id=str(data["capsule_id"]),
            input_bindings=tuple(
                BindingSummary.from_dict(d) for d in data.get("input_bindings", [])
            ),
            output_bindings=tuple(
                BindingSummary.from_dict(d) for d in data.get("output_bindings", [])
            ),
            slots=tuple(SlotSummary.from_dict(d) for d in data.get("slots", [])),
            preconditions=tuple(str(v) for v in data.get("preconditions", [])),
            postconditions=tuple(str(v) for v in data.get("postconditions", [])),
            effects=tuple(str(v) for v in data.get("effects", [])),
            exceptions=tuple(str(v) for v in data.get("exceptions", [])),
            captures=tuple(str(v) for v in data.get("captures", [])),
            conservative=bool(data.get("conservative", False)),
            fingerprint=str(data["fingerprint"]),
        )

    @property
    def is_exact(self) -> bool:
        """An empty interface is still conservative if the flag says so."""
        return not self.conservative


@dataclass(frozen=True)
class ExternalInput:
    """One typed input arriving from outside the capsule graph."""

    name: str
    kind: str = "external"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExternalInput:
        return cls(name=str(data["name"]), kind=str(data.get("kind", "external")))


# --------------------------------------------------------------------------- #
# Problem / plan / result contracts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CapsuleProblem:
    """One capsule plus its finite-domain state and boundary assumptions."""

    capsule: VerificationCapsule
    state: FiniteDomainState
    predecessor_summaries: tuple[CapsuleInterfaceSummary, ...]
    external_inputs: tuple[ExternalInput, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "capsule": self.capsule.to_dict(),
            "state": self.state.to_dict(),
            "predecessor_summaries": [s.to_dict() for s in self.predecessor_summaries],
            "external_inputs": [e.to_dict() for e in self.external_inputs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapsuleProblem:
        return cls(
            capsule=VerificationCapsule.from_dict(data["capsule"]),
            state=FiniteDomainState.from_dict(data["state"]),
            predecessor_summaries=tuple(
                CapsuleInterfaceSummary.from_dict(d)
                for d in data.get("predecessor_summaries", [])
            ),
            external_inputs=tuple(
                ExternalInput.from_dict(d) for d in data.get("external_inputs", [])
            ),
        )


@dataclass(frozen=True)
class PerCapsuleResult:
    """Outcome of solving one capsule, independent of assembly."""

    capsule_id: str
    status: str
    state: FiniteDomainState
    summary: CapsuleInterfaceSummary | None
    search_result: SearchResult | None
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "capsule_id": self.capsule_id,
            "status": self.status,
            "state": self.state.to_dict(),
            "summary": self.summary.to_dict() if self.summary is not None else None,
            "search_result": self.search_result.to_dict() if self.search_result is not None else None,
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerCapsuleResult:
        return cls(
            capsule_id=str(data["capsule_id"]),
            status=str(data["status"]),
            state=FiniteDomainState.from_dict(data["state"]),
            summary=CapsuleInterfaceSummary.from_dict(data["summary"])
            if data.get("summary") is not None
            else None,
            search_result=SearchResult.from_dict(data["search_result"])
            if data.get("search_result") is not None
            else None,
            stop_reason=data.get("stop_reason"),
        )


@dataclass(frozen=True)
class CapsuleSolvePlan:
    """Deterministic execution plan for a capsule dependency graph."""

    graph_fingerprint: str
    stages: tuple[tuple[str, ...], ...]
    joint_sccs: tuple[tuple[str, ...], ...]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_fingerprint": self.graph_fingerprint,
            "stages": [list(stage) for stage in self.stages],
            "joint_sccs": [list(scc) for scc in self.joint_sccs],
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapsuleSolvePlan:
        return cls(
            graph_fingerprint=str(data["graph_fingerprint"]),
            stages=tuple(tuple(str(v) for v in stage) for stage in data.get("stages", [])),
            joint_sccs=tuple(
                tuple(str(v) for v in scc) for scc in data.get("joint_sccs", [])
            ),
            fingerprint=str(data["fingerprint"]),
        )


@dataclass(frozen=True)
class Disagreement:
    """Local pass and global pass disagree; neither relabeled as solved."""

    kind: str
    capsule_id: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "capsule_id": self.capsule_id, "detail": self.detail}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Disagreement:
        return cls(
            kind=str(data["kind"]),
            capsule_id=str(data["capsule_id"]),
            detail=str(data["detail"]),
        )


@dataclass(frozen=True)
class CapsuleCounters:
    """Aggregated work counters across all capsules and the final verifier."""

    passes: int = 0
    support_queries: int = 0
    cache_hits: int = 0
    supported: int = 0
    unsupported: int = 0
    unknown: int = 0
    candidates_removed: int = 0
    decisions: int = 0
    backtracks: int = 0
    local_nogoods: int = 0
    expanded_solver_nodes: int = 0
    solver_verifier_calls: int = 0
    capsule_count: int = 0
    joint_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "passes": self.passes,
            "support_queries": self.support_queries,
            "cache_hits": self.cache_hits,
            "supported": self.supported,
            "unsupported": self.unsupported,
            "unknown": self.unknown,
            "candidates_removed": self.candidates_removed,
            "decisions": self.decisions,
            "backtracks": self.backtracks,
            "local_nogoods": self.local_nogoods,
            "expanded_solver_nodes": self.expanded_solver_nodes,
            "solver_verifier_calls": self.solver_verifier_calls,
            "capsule_count": self.capsule_count,
            "joint_count": self.joint_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapsuleCounters:
        return cls(**{key: int(data[key]) for key in cls().to_dict()})


@dataclass(frozen=True)
class CapsuleSolveResult:
    """Project-level outcome of a capsule solve campaign."""

    status: str
    capsule_results: tuple[PerCapsuleResult, ...]
    summaries: tuple[CapsuleInterfaceSummary, ...]
    assembled_source: str | None
    global_verifier_report: JsonValue | None
    local_global_disagreements: tuple[Disagreement, ...]
    counters: CapsuleCounters
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "capsule_results": [r.to_dict() for r in self.capsule_results],
            "summaries": [s.to_dict() for s in self.summaries],
            "assembled_source": self.assembled_source,
            "global_verifier_report": self.global_verifier_report,
            "local_global_disagreements": [d.to_dict() for d in self.local_global_disagreements],
            "counters": self.counters.to_dict(),
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapsuleSolveResult:
        return cls(
            status=str(data["status"]),
            capsule_results=tuple(
                PerCapsuleResult.from_dict(d) for d in data.get("capsule_results", [])
            ),
            summaries=tuple(
                CapsuleInterfaceSummary.from_dict(d) for d in data.get("summaries", [])
            ),
            assembled_source=data.get("assembled_source"),
            global_verifier_report=data.get("global_verifier_report"),
            local_global_disagreements=tuple(
                Disagreement.from_dict(d)
                for d in data.get("local_global_disagreements", [])
            ),
            counters=CapsuleCounters.from_dict(data["counters"]),
            stop_reason=data.get("stop_reason"),
        )


# --------------------------------------------------------------------------- #
# Pack seam protocols
# --------------------------------------------------------------------------- #


class CapsuleProblemBuilder(Protocol):
    """Pack-owned construction of a finite-domain problem for one capsule."""

    def build_problem(
        self,
        capsule: VerificationCapsule,
        predecessor_summaries: tuple[CapsuleInterfaceSummary, ...],
        external_inputs: tuple[ExternalInput, ...],
        bounds: SolverBounds,
    ) -> CapsuleProblem: ...


class CapsuleSummaryExtractor(Protocol):
    """Pack-owned extraction of an interface summary from a solved capsule."""

    def extract_summary(
        self, capsule: VerificationCapsule, state: FiniteDomainState
    ) -> CapsuleInterfaceSummary: ...


CapsuleMaterializer = Callable[[tuple[PerCapsuleResult, ...]], str | None]


class CapsuleGlobalVerifier(Protocol):
    """Authoritative whole-program verifier; the local oracle is not final."""

    def verify(self, source: str | None) -> TerminalOutcome: ...


# --------------------------------------------------------------------------- #
# Plan construction
# --------------------------------------------------------------------------- #


def _capsule_predecessors(
    graph: CapsuleGraph,
) -> dict[str, set[str]]:
    """Map each capsule to the set of predecessor capsules from REFERENCE edges."""
    node_to_capsule: dict[str, str] = {}
    for capsule in graph.capsules:
        for node_id in capsule.node_ids:
            node_to_capsule[node_id] = capsule.capsule_id

    predecessors: dict[str, set[str]] = {
        capsule.capsule_id: set() for capsule in graph.capsules
    }
    for edge in graph.edges:
        if edge.kind is not DependencyKind.REFERENCE:
            continue
        source_capsule = node_to_capsule.get(edge.source)
        target_capsule = node_to_capsule.get(edge.target)
        if source_capsule is None or target_capsule is None:
            continue
        if source_capsule != target_capsule:
            predecessors[target_capsule].add(source_capsule)
    return predecessors


def build_capsule_solve_plan(graph: CapsuleGraph) -> CapsuleSolvePlan:
    """Build a deterministic topological stage plan from a capsule graph.

    Each stage contains capsules whose predecessor capsules are all in earlier
    stages. SCCs with more than one member are recorded as joint problems.
    """
    predecessors = _capsule_predecessors(graph)
    remaining = set(predecessors)
    stages: list[tuple[str, ...]] = []
    placed: set[str] = set()

    while remaining:
        ready = sorted(
            capsule_id
            for capsule_id in remaining
            if predecessors[capsule_id] <= placed
        )
        if not ready:
            # The graph is expected to be acyclic; if not, fall back to one
            # stage per remaining capsule to avoid infinite looping.
            ready = sorted(remaining)
        stage = tuple(ready)
        stages.append(stage)
        placed.update(ready)
        remaining.difference_update(ready)

    joint_sccs = tuple(
        tuple(sorted(capsule.node_ids))
        for capsule in graph.capsules
        if len(capsule.node_ids) > 1
    )

    plan_payload = _canonical_json(
        {
            "graph_fingerprint": graph.fingerprint if hasattr(graph, "fingerprint") else graph.spec_id,
            "stages": [list(stage) for stage in stages],
            "joint_sccs": [list(scc) for scc in joint_sccs],
        }
    )
    fingerprint = _sha256(plan_payload)
    graph_fingerprint = getattr(graph, "fingerprint", graph.spec_id)

    return CapsuleSolvePlan(
        graph_fingerprint=graph_fingerprint,
        stages=tuple(stages),
        joint_sccs=joint_sccs,
        fingerprint=fingerprint,
    )


# --------------------------------------------------------------------------- #
# Core solve loop
# --------------------------------------------------------------------------- #


def _empty_summary(
    capsule: VerificationCapsule, *, conservative: bool, reason: str
) -> CapsuleInterfaceSummary:
    """A conservative empty-boundary summary for a skipped/failed capsule."""
    payload = _canonical_json(
        {
            "capsule_id": capsule.capsule_id,
            "reason": reason,
            "conservative": conservative,
        }
    )
    return CapsuleInterfaceSummary(
        capsule_id=capsule.capsule_id,
        input_bindings=(),
        output_bindings=(),
        slots=tuple(SlotSummary(name=name) for name in capsule.external_dependencies),
        preconditions=(),
        postconditions=(),
        effects=(),
        exceptions=(),
        captures=(),
        conservative=conservative,
        fingerprint=_sha256(payload),
    )


def solve_capsule_graph(
    graph: CapsuleGraph,
    *,
    builder: CapsuleProblemBuilder,
    provider: SupportProvider,
    terminal_checker: TerminalChecker,
    summary_extractor: CapsuleSummaryExtractor,
    materializer: CapsuleMaterializer,
    global_verifier: Callable[[str | None], TerminalOutcome],
    ranker: CandidateRanker | None = None,
    bounds: SolverBounds | None = None,
    cache: dict[str, Any] | None = None,
    certificate_store: dict[str, Any] | None = None,
) -> CapsuleSolveResult:
    """Solve a capsule dependency graph and run the whole-program verifier."""
    plan = build_capsule_solve_plan(graph)
    bounds = bounds or SolverBounds(
        max_tokens=10_000,
        max_nodes=10_000,
        max_depth=256,
        max_backtracks=1_000,
        max_verifier_calls=1_000,
    )
    cache = cache or {}

    results_by_id: dict[str, PerCapsuleResult] = {}
    summaries_by_id: dict[str, CapsuleInterfaceSummary] = {}
    capsule_results: list[PerCapsuleResult] = []
    counters = CapsuleCounters(capsule_count=len(graph.capsules), joint_count=len(plan.joint_sccs))

    status = "solved"
    stop_reason: str | None = None

    for stage_index, stage in enumerate(plan.stages):
        for capsule_id in sorted(stage):
            capsule = next(c for c in graph.capsules if c.capsule_id == capsule_id)
            preds = _capsule_predecessors(graph)[capsule_id]
            pred_summaries = tuple(summaries_by_id[p] for p in sorted(preds) if p in summaries_by_id)

            # Unknown/conservative predecessor boundaries cannot license an
            # exact successor solve.
            if any(s.conservative for s in pred_summaries):
                summary = _empty_summary(
                    capsule, conservative=True, reason="conservative_predecessor"
                )
                result = PerCapsuleResult(
                    capsule_id=capsule_id,
                    status="unknown",
                    state=_empty_state(capsule, bounds),
                    summary=summary,
                    search_result=None,
                    stop_reason="conservative_predecessor",
                )
                results_by_id[capsule_id] = result
                summaries_by_id[capsule_id] = summary
                capsule_results.append(result)
                status = "unknown"
                if stop_reason is None:
                    stop_reason = "conservative_predecessor"
                continue

            external_inputs = tuple(
                ExternalInput(name=name)
                for name in sorted(capsule.external_dependencies)
            )
            problem = builder.build_problem(
                capsule, pred_summaries, external_inputs, bounds
            )

            search_result = search(
                problem.state,
                provider,
                terminal_checker,
                ranker=ranker,
                cache=cache,
                certificate_store=certificate_store,
            )

            # Aggregate controller counters.
            counters = _add_search_counters(counters, search_result.counters)

            if search_result.status.value != "solved":
                summary = _empty_summary(
                    capsule,
                    conservative=True,
                    reason=f"solver:{search_result.status.value}",
                )
                result = PerCapsuleResult(
                    capsule_id=capsule_id,
                    status=search_result.status.value,
                    state=search_result.state,
                    summary=summary,
                    search_result=search_result,
                    stop_reason=search_result.stop_reason,
                )
                results_by_id[capsule_id] = result
                summaries_by_id[capsule_id] = summary
                capsule_results.append(result)
                status = "unknown" if status == "solved" else status
                if stop_reason is None:
                    stop_reason = f"capsule_{capsule_id}:{search_result.status.value}"
                continue

            summary = summary_extractor.extract_summary(capsule, search_result.state)
            result = PerCapsuleResult(
                capsule_id=capsule_id,
                status="solved",
                state=search_result.state,
                summary=summary,
                search_result=search_result,
                stop_reason=None,
            )
            results_by_id[capsule_id] = result
            summaries_by_id[capsule_id] = summary
            capsule_results.append(result)

    assembled = materializer(tuple(capsule_results))
    final = global_verifier(assembled)

    disagreements: list[Disagreement] = []
    if final.accepted:
        project_status = "solved" if status == "solved" else status
    else:
        project_status = "unknown"
        stop_reason = stop_reason or "global_verifier_rejected"
        disagreements.append(
            Disagreement(
                kind="local_pass_global_fail",
                capsule_id="assembly",
                detail=final.detail or "global verifier rejected assembled program",
            )
        )

    counters = CapsuleCounters(
        passes=counters.passes,
        support_queries=counters.support_queries,
        cache_hits=counters.cache_hits,
        supported=counters.supported,
        unsupported=counters.unsupported,
        unknown=counters.unknown,
        candidates_removed=counters.candidates_removed,
        decisions=counters.decisions,
        backtracks=counters.backtracks,
        local_nogoods=counters.local_nogoods,
        expanded_solver_nodes=counters.expanded_solver_nodes,
        solver_verifier_calls=counters.solver_verifier_calls + 1,
        capsule_count=counters.capsule_count,
        joint_count=counters.joint_count,
    )

    return CapsuleSolveResult(
        status=project_status,
        capsule_results=tuple(capsule_results),
        summaries=tuple(summaries_by_id[c] for c in sorted(summaries_by_id)),
        assembled_source=assembled,
        global_verifier_report=final.report,
        local_global_disagreements=tuple(disagreements),
        counters=counters,
        stop_reason=stop_reason,
    )


def _empty_state(capsule: VerificationCapsule, bounds: SolverBounds) -> FiniteDomainState:
    """A well-formed but empty state for skipped capsules."""
    return FiniteDomainState(
        problem_id=f"{capsule.capsule_id}:skipped",
        pack_id="capsule-solver",
        constraint_version="vss2-02-v1",
        bounds=bounds,
        holes=(),
    )


def _add_search_counters(counters: CapsuleCounters, sc: SearchCounters) -> CapsuleCounters:
    return CapsuleCounters(
        passes=counters.passes,
        support_queries=counters.support_queries + sc.tokens,
        cache_hits=counters.cache_hits,
        supported=counters.supported,
        unsupported=counters.unsupported,
        unknown=counters.unknown,
        candidates_removed=counters.candidates_removed,
        decisions=counters.decisions + sc.depth,
        backtracks=counters.backtracks + sc.backtracks,
        local_nogoods=counters.local_nogoods,
        expanded_solver_nodes=counters.expanded_solver_nodes + sc.nodes,
        solver_verifier_calls=counters.solver_verifier_calls + sc.verifier_calls,
        capsule_count=counters.capsule_count,
        joint_count=counters.joint_count,
    )


