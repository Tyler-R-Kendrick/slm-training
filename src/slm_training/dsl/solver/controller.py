"""Bounded proof-carrying search controller (VSS1-02).

Alternates exact closure (irreversible, certificate-backed candidate deletion)
with reversible branching (rankers choose only order, never membership). It keeps
the repository's hard/soft separation strict:

* **Certified deduction** — a domain reduction produced by `exact_closure`, each
  removal citing a replay-valid `UNSUPPORTED` certificate. Irreversible.
* **Reversible decision** — a chosen live value recorded on a decision stack and
  undone on backtracking. A `CandidateRanker` only permutes the exact live values;
  it can neither add nor drop a candidate.
* **Local nogood** — a request-local record that a decision branch conflicted. It
  is **not** a certified deduction and is never serialized as one.
* **Certificate-backed global contradiction** — `CERTIFIED_UNSAT`, returned only
  when the whole finite tree closes by certified deductions with no UNKNOWN,
  verifier-rejection, or budget truncation anywhere.
* **Timeout / unknown** — `UNKNOWN` / `BUDGET_EXHAUSTED`, never relabeled unsat.

Ownership: this is the new generic controller. The existing compiler-forest search
`LatticeSearchState` (`fastpath/lattice_search.py`) is retained unchanged as the
forest-specific adapter; its callers and tests are untouched. Semantics are owned
by ``docs/design/verified-scope-solver.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from slm_training.dsl.solver.closure import (
    CertifiedDeduction,
    SupportProvider,
    exact_closure,
)
from slm_training.dsl.solver.state import (
    DomainValue,
    FiniteDomainState,
    HoleId,
    SolverBounds,
)
from slm_training.dsl.solver.support import SearchCounters, SupportCertificate

JsonValue = Any


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


class SearchStatus(str, Enum):
    SOLVED = "solved"
    CERTIFIED_UNSAT = "certified_unsat"
    UNKNOWN = "unknown"
    BUDGET_EXHAUSTED = "budget_exhausted"


# --------------------------------------------------------------------------- #
# Injected seams: ranker (soft) and terminal checker (final verifier)
# --------------------------------------------------------------------------- #


class CandidateRanker(Protocol):
    """Orders the exact live values for a hole; never adds or drops a value."""

    @property
    def ranker_id(self) -> str: ...

    def rank(
        self, state: FiniteDomainState, hole_id: HoleId, values: tuple[DomainValue, ...]
    ) -> tuple[DomainValue, ...]: ...


class BaselineRanker:
    """Deterministic identity ordering (values arrive canonically ordered)."""

    @property
    def ranker_id(self) -> str:
        return "baseline-canonical-v1"

    def rank(
        self, state: FiniteDomainState, hole_id: HoleId, values: tuple[DomainValue, ...]
    ) -> tuple[DomainValue, ...]:
        return tuple(values)


@dataclass(frozen=True)
class TerminalOutcome:
    """Result of materializing + verifying a structurally-solved state."""

    accepted: bool
    source: str | None = None
    report: JsonValue | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "source": self.source,
            "report": self.report,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TerminalOutcome:
        return cls(
            accepted=bool(data["accepted"]),
            source=data.get("source"),
            report=data.get("report"),
            detail=str(data.get("detail", "")),
        )


class TerminalChecker(Protocol):
    """Materializes a structurally-solved state and runs the final verifier."""

    def check(self, state: FiniteDomainState) -> TerminalOutcome: ...


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SearchDecision:
    decision_id: str
    before_fingerprint: str
    after_fingerprint: str
    level: int
    hole_id: HoleId
    chosen: DomainValue
    alternatives: tuple[DomainValue, ...]
    ranker_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "before_fingerprint": self.before_fingerprint,
            "after_fingerprint": self.after_fingerprint,
            "level": self.level,
            "hole_id": self.hole_id.to_dict(),
            "chosen": self.chosen.to_dict(),
            "alternatives": [value.to_dict() for value in self.alternatives],
            "ranker_id": self.ranker_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchDecision:
        return cls(
            decision_id=str(data["decision_id"]),
            before_fingerprint=str(data["before_fingerprint"]),
            after_fingerprint=str(data["after_fingerprint"]),
            level=int(data["level"]),
            hole_id=HoleId.from_dict(data["hole_id"]),
            chosen=DomainValue.from_dict(data["chosen"]),
            alternatives=tuple(
                DomainValue.from_dict(d) for d in data.get("alternatives", [])
            ),
            ranker_id=str(data["ranker_id"]),
        )


@dataclass(frozen=True)
class Nogood:
    """A request-local conflict record; NOT a certified deduction."""

    problem_id: str
    constraint_version: str
    bounds: SolverBounds
    assignment: tuple[tuple[HoleId, DomainValue], ...]
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "constraint_version": self.constraint_version,
            "bounds": self.bounds.to_dict(),
            "assignment": [
                [hole.to_dict(), value.to_dict()] for hole, value in self.assignment
            ],
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Nogood:
        assignment: list[tuple[HoleId, DomainValue]] = []
        for pair in data.get("assignment", []):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Nogood assignment entries must be [hole, value] pairs")
            assignment.append((HoleId.from_dict(pair[0]), DomainValue.from_dict(pair[1])))
        return cls(
            problem_id=str(data["problem_id"]),
            constraint_version=str(data["constraint_version"]),
            bounds=SolverBounds.from_dict(data["bounds"]),
            assignment=tuple(assignment),
            provenance=str(data.get("provenance", "")),
        )

    @property
    def key(self) -> str:
        return _canonical_json(
            {
                "problem_id": self.problem_id,
                "constraint_version": self.constraint_version,
                "bounds": self.bounds.to_dict(),
                "assignment": [
                    [hole.to_dict(), value.to_dict()] for hole, value in self.assignment
                ],
            }
        )


@dataclass(frozen=True)
class SearchResult:
    status: SearchStatus
    state: FiniteDomainState
    source: str | None
    verifier_report: JsonValue | None
    deductions: tuple[CertifiedDeduction, ...]
    decisions: tuple[SearchDecision, ...]
    nogoods: tuple[Nogood, ...]
    counters: SearchCounters
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "state": self.state.to_dict(),
            "source": self.source,
            "verifier_report": self.verifier_report,
            "deductions": [d.to_dict() for d in self.deductions],
            "decisions": [d.to_dict() for d in self.decisions],
            "nogoods": [n.to_dict() for n in self.nogoods],
            "counters": self.counters.to_dict(),
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchResult:
        return cls(
            status=SearchStatus(data["status"]),
            state=FiniteDomainState.from_dict(data["state"]),
            source=data.get("source"),
            verifier_report=data.get("verifier_report"),
            deductions=tuple(
                CertifiedDeduction.from_dict(d) for d in data.get("deductions", [])
            ),
            decisions=tuple(
                SearchDecision.from_dict(d) for d in data.get("decisions", [])
            ),
            nogoods=tuple(Nogood.from_dict(d) for d in data.get("nogoods", [])),
            counters=SearchCounters.from_dict(data["counters"]),
            stop_reason=data.get("stop_reason"),
        )


def default_hole_selector(state: FiniteDomainState) -> HoleId | None:
    """Smallest live domain first, then canonical ``HoleId`` (deterministic)."""
    unresolved = [hole for hole in state.holes if len(hole.values) > 1]
    if not unresolved:
        return None
    best = min(unresolved, key=lambda hole: (len(hole.values), hole.hole_id.sort_key))
    return best.hole_id


def _validate_permutation(
    permuted: tuple[DomainValue, ...], values: tuple[DomainValue, ...]
) -> None:
    live = set(values)
    seen = set(permuted)
    if len(permuted) != len(values):
        raise ValueError("ranker returned a different candidate count")
    if len(seen) != len(permuted):
        raise ValueError("ranker returned a duplicate candidate")
    if seen != live:
        missing = live - seen
        extra = seen - live
        raise ValueError(
            f"ranker altered live candidates (missing={len(missing)}, extra={len(extra)})"
        )


@dataclass
class _Frame:
    pre_state: FiniteDomainState
    hole_id: HoleId
    chosen: DomainValue
    remaining: list[DomainValue]
    level: int


@dataclass
class _Mut:
    nodes: int = 0
    backtracks: int = 0
    verifier_calls: int = 0
    decisions: int = 0
    support_queries: int = 0

    def counters(self) -> SearchCounters:
        return SearchCounters(
            nodes=self.nodes,
            tokens=self.support_queries,
            depth=self.decisions,
            backtracks=self.backtracks,
            verifier_calls=self.verifier_calls,
        )


def search(
    state: FiniteDomainState,
    provider: SupportProvider,
    terminal_checker: TerminalChecker,
    *,
    ranker: CandidateRanker | None = None,
    hole_selector=default_hole_selector,
    cache: dict[str, Any] | None = None,
    certificate_store: dict[str, SupportCertificate] | None = None,
    max_decisions: int = 10_000,
    max_backtracks: int | None = None,
) -> SearchResult:
    """Bounded closure + reversible branching; sound tri-state termination."""
    if not isinstance(state, FiniteDomainState):
        raise ValueError("search requires a FiniteDomainState")
    ranker = ranker or BaselineRanker()
    if max_backtracks is None:
        max_backtracks = state.bounds.max_backtracks
    counters = _Mut()
    deductions: list[CertifiedDeduction] = []
    decisions: list[SearchDecision] = []
    nogoods: list[Nogood] = []
    stack: list[_Frame] = []
    current = state
    level = 0
    # Any non-certified reason a branch closed (UNKNOWN closure, verifier rejection,
    # budget) means the tree is not fully proof-closed -> UNSAT cannot be certified.
    saw_uncertified = False

    def record_nogood(hole_id: HoleId, value: DomainValue, provenance: str) -> None:
        nogood = Nogood(
            problem_id=current.problem_id,
            constraint_version=current.constraint_version,
            bounds=current.bounds,
            assignment=((hole_id, value),),
            provenance=provenance,
        )
        if all(existing.key != nogood.key for existing in nogoods):
            nogoods.append(nogood)

    def backtrack(provenance: str) -> bool:
        """Restore the latest pre-decision state and take the next alternative."""
        nonlocal current, level
        while stack:
            frame = stack[-1]
            counters.backtracks += 1
            record_nogood(frame.hole_id, frame.chosen, provenance)
            if counters.backtracks > max_backtracks:
                return False
            if frame.remaining:
                frame.chosen = frame.remaining.pop(0)
                current = frame.pre_state.with_decision(frame.hole_id, frame.chosen)
                level = frame.level + 1
                # The alternative is a fresh decision at the same point; log it so
                # the whole trajectory (including backtracked attempts) replays.
                counters.decisions += 1
                decisions.append(
                    SearchDecision(
                        decision_id=f"d{counters.decisions}",
                        before_fingerprint=frame.pre_state.fingerprint,
                        after_fingerprint=current.fingerprint,
                        level=level,
                        hole_id=frame.hole_id,
                        chosen=frame.chosen,
                        alternatives=tuple(frame.remaining),
                        ranker_id=ranker.ranker_id,
                    )
                )
                return True
            stack.pop()
        return False

    while True:
        closure = exact_closure(
            current, provider, cache=cache, certificate_store=certificate_store
        )
        deductions.extend(closure.deductions)
        counters.nodes += 1
        counters.verifier_calls += closure.counters.verifier_calls
        counters.support_queries += closure.counters.support_queries
        current = closure.state
        # A closure that left unknowns or was budget-truncated cannot license unsat.
        if closure.unknown_queries or not closure.reached_fixed_point:
            saw_uncertified = True

        if current.is_bottom:
            # Certified bottom: closure removes only certified-unsupported values.
            if not backtrack("certified_bottom"):
                status = (
                    SearchStatus.CERTIFIED_UNSAT
                    if not saw_uncertified
                    else SearchStatus.UNKNOWN
                )
                return _result(status, current, None, None, deductions, decisions,
                               nogoods, counters,
                               None if status is SearchStatus.CERTIFIED_UNSAT
                               else "exhausted_with_unknown")
            continue

        if current.is_structurally_solved:
            outcome = terminal_checker.check(current)
            counters.verifier_calls += 1
            if outcome.accepted:
                return _result(SearchStatus.SOLVED, current, outcome.source,
                               outcome.report, deductions, decisions, nogoods, counters,
                               None)
            # Verifier rejection is a conflict, not a certified deduction.
            saw_uncertified = True
            if not backtrack("terminal_verifier_failure"):
                return _result(SearchStatus.UNKNOWN, current, None, outcome.report,
                               deductions, decisions, nogoods, counters,
                               "terminal_rejected_exhausted")
            continue

        # Budgets before branching.
        if counters.decisions >= max_decisions:
            return _result(SearchStatus.BUDGET_EXHAUSTED, current, None, None,
                           deductions, decisions, nogoods, counters, "budget:max_decisions")
        if counters.nodes > current.bounds.max_nodes:
            return _result(SearchStatus.BUDGET_EXHAUSTED, current, None, None,
                           deductions, decisions, nogoods, counters, "budget:max_nodes")

        hole_id = hole_selector(current)
        if hole_id is None:  # defensive: no unresolved hole though not solved
            saw_uncertified = True
            if not backtrack("no_branching_hole"):
                return _result(SearchStatus.UNKNOWN, current, None, None, deductions,
                               decisions, nogoods, counters, "no_branching_hole")
            continue
        live = current.domain(hole_id).values
        permuted = ranker.rank(current, hole_id, live)
        _validate_permutation(permuted, live)  # ranker cannot alter membership
        chosen, alternatives = permuted[0], permuted[1:]
        pre = current
        after = current.with_decision(hole_id, chosen)
        counters.decisions += 1
        decisions.append(
            SearchDecision(
                decision_id=f"d{counters.decisions}",
                before_fingerprint=pre.fingerprint,
                after_fingerprint=after.fingerprint,
                level=level + 1,
                hole_id=hole_id,
                chosen=chosen,
                alternatives=alternatives,
                ranker_id=ranker.ranker_id,
            )
        )
        stack.append(_Frame(pre, hole_id, chosen, list(alternatives), level))
        current = after
        level += 1


def _result(status, state, source, report, deductions, decisions, nogoods, counters,
            stop_reason) -> SearchResult:
    return SearchResult(
        status=status,
        state=state,
        source=source,
        verifier_report=report,
        deductions=tuple(deductions),
        decisions=tuple(decisions),
        nogoods=tuple(nogoods),
        counters=counters.counters(),
        stop_reason=stop_reason,
    )
