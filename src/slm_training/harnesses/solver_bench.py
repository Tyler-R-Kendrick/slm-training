"""VSS4-01 exhaustive finite solver benchmark with independent ground truth (SLM-74).

A small, deterministic benchmark whose full bounded solution space is exhaustively
enumerable, providing ground truth for the verified-scope solver before any
frontier/model-quality claim. The benchmark's primary invariant is:

    Every candidate labeled ``unsupported`` by the evaluated solver is absent from
    the benchmark's independently enumerated verifier-accepted solution set for the
    same pack, constraint version, and finite bounds.

Two deterministic paths decide each case and must agree on every closed fixture:

1. the reference ``EnumerativeSupportOracle`` (VSS0-04), the runtime under test; and
2. a benchmark-only brute-force transition enumerator written here directly against
   the ``ProblemExpander``/``Verifier`` protocols and the finite case schema.

Torch-free; no learned model, no network, no large committed corpus. This validates
finite-fixture implementation invariants, not frontier generalization or ship quality.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

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
    ProblemExpander,
    SupportQuery,
    Verifier,
    VerifyOutcome,
    VerifyStatus,
    replay_support_certificate,
)

__all__ = [
    "SolverBenchmarkCase",
    "GroundTruth",
    "CaseResult",
    "SuiteReport",
    "ReferenceFixture",
    "enumerate_ground_truth",
    "ground_truth_verdict",
    "run_case",
    "run_suite",
    "build_reference_fixture",
    "run_reference_suite",
]

_SUPPORTED = SupportVerdict.SUPPORTED.value
_UNSUPPORTED = SupportVerdict.UNSUPPORTED.value
_UNKNOWN = SupportVerdict.UNKNOWN.value


@dataclass(frozen=True)
class SolverBenchmarkCase:
    """One closed benchmark candidate with its expected exact support verdict."""

    case_id: str
    family: str
    candidate: DomainValue
    expected_verdict: str  # supported | unsupported | unknown
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.expected_verdict not in (_SUPPORTED, _UNSUPPORTED, _UNKNOWN):
            raise ValueError(f"bad expected_verdict {self.expected_verdict!r}")
        if not isinstance(self.candidate, DomainValue):
            raise ValueError("case candidate must be a DomainValue")


@dataclass(frozen=True)
class GroundTruth:
    """Independent brute-force result for one candidate subtree."""

    verdict: str  # supported | unsupported | unknown
    accepted_terminals: tuple[str, ...]  # sorted digests of accepted programs
    exhausted: bool  # True iff the subtree was fully covered within budget


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    family: str
    oracle_verdict: str
    ground_truth_verdict: str
    expected_verdict: str
    certificate_replays: bool
    false_unsupported: bool  # oracle=unsupported but ground truth has accepted terminals
    unknown_preservation_violation: bool  # oracle pruned a candidate ground truth keeps live
    agrees: bool  # oracle == ground truth == expected

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "oracle_verdict": self.oracle_verdict,
            "ground_truth_verdict": self.ground_truth_verdict,
            "expected_verdict": self.expected_verdict,
            "certificate_replays": self.certificate_replays,
            "false_unsupported": self.false_unsupported,
            "unknown_preservation_violation": self.unknown_preservation_violation,
            "agrees": self.agrees,
        }


@dataclass(frozen=True)
class SuiteReport:
    results: tuple[CaseResult, ...]
    manifest_digest: str

    @property
    def hard_failures(self) -> tuple[CaseResult, ...]:
        return tuple(
            r
            for r in self.results
            if r.false_unsupported
            or r.unknown_preservation_violation
            or not r.certificate_replays
            or not r.agrees
        )

    @property
    def passed(self) -> bool:
        return not self.hard_failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_digest": self.manifest_digest,
            "case_count": len(self.results),
            "false_unsupported_count": sum(r.false_unsupported for r in self.results),
            "unknown_preservation_violations": sum(
                r.unknown_preservation_violation for r in self.results
            ),
            "certificate_replay_failures": sum(
                not r.certificate_replays for r in self.results
            ),
            "passed": self.passed,
            "cases": [r.to_dict() for r in self.results],
        }


def _program_digest(program: str) -> str:
    return hashlib.sha256(program.encode("utf-8")).hexdigest()[:16]


def enumerate_ground_truth(
    expander: ProblemExpander,
    verifier: Verifier,
    start_state: FiniteDomainState,
    *,
    max_nodes: int = 4096,
) -> GroundTruth:
    """Brute-force transition enumeration over the finite subtree rooted at
    ``start_state``. Independent of the oracle's search/certificate internals: it
    simply walks every (hole, value) transition, applies the pack verifier at each
    terminal, and records the accepted set and whether coverage was complete.
    """
    accepted: set[str] = set()
    has_incomplete = False
    exhausted = True
    seen: set[str] = set()
    stack: list[FiniteDomainState] = [start_state]
    nodes = 0
    while stack:
        state = stack.pop()
        if state.fingerprint in seen:
            continue
        seen.add(state.fingerprint)
        nodes += 1
        if nodes > max_nodes:
            exhausted = False
            has_incomplete = True
            break
        for domain in state.holes:
            for value in domain.values:
                step = expander.successor(state, domain.hole_id, value)
                if step.status is ExpandStatus.TERMINAL:
                    program = step.program or ""
                    if verifier.verify(program).status is VerifyStatus.ACCEPT:
                        accepted.add(_program_digest(program))
                elif step.status is ExpandStatus.DEAD:
                    continue
                elif step.status is ExpandStatus.INCOMPLETE:
                    has_incomplete = True
                elif step.status is ExpandStatus.CONTINUE and step.next_state is not None:
                    stack.append(step.next_state)
    if accepted:
        verdict = _SUPPORTED
    elif has_incomplete:
        verdict = _UNKNOWN
    else:
        verdict = _UNSUPPORTED
    return GroundTruth(
        verdict=verdict,
        accepted_terminals=tuple(sorted(accepted)),
        exhausted=exhausted and not has_incomplete,
    )


def ground_truth_verdict(
    expander: ProblemExpander,
    verifier: Verifier,
    state: FiniteDomainState,
    hole_id: HoleId,
    candidate: DomainValue,
    *,
    max_nodes: int = 4096,
) -> GroundTruth:
    """Ground-truth verdict for applying ``candidate`` at ``hole_id`` in ``state``."""
    step = expander.successor(state, hole_id, candidate)
    if step.status is ExpandStatus.TERMINAL:
        program = step.program or ""
        accepted = verifier.verify(program).status is VerifyStatus.ACCEPT
        digest = (_program_digest(program),) if accepted else ()
        return GroundTruth(
            verdict=_SUPPORTED if accepted else _UNSUPPORTED,
            accepted_terminals=digest,
            exhausted=True,
        )
    if step.status is ExpandStatus.DEAD:
        return GroundTruth(verdict=_UNSUPPORTED, accepted_terminals=(), exhausted=True)
    if step.status is ExpandStatus.INCOMPLETE:
        return GroundTruth(verdict=_UNKNOWN, accepted_terminals=(), exhausted=False)
    if step.next_state is None:
        return GroundTruth(verdict=_UNKNOWN, accepted_terminals=(), exhausted=False)
    return enumerate_ground_truth(expander, verifier, step.next_state, max_nodes=max_nodes)


def run_case(
    oracle: EnumerativeSupportOracle,
    expander: ProblemExpander,
    verifier: Verifier,
    state: FiniteDomainState,
    hole_id: HoleId,
    case: SolverBenchmarkCase,
    *,
    max_nodes: int = 4096,
) -> CaseResult:
    """Decide one case via both paths and cross-check the hard invariants."""
    query = SupportQuery(
        state_fingerprint=state.fingerprint, hole_id=hole_id, candidate=case.candidate
    )
    result = oracle.check(state, query)
    oracle_verdict = result.verdict.value
    truth = ground_truth_verdict(
        expander, verifier, state, hole_id, case.candidate, max_nodes=max_nodes
    )
    replays = replay_support_certificate(
        result.certificate, state=state, expander=expander, verifier=verifier
    ).ok
    # A false certified prune: the solver said unsupported, yet ground truth found
    # an accepted terminal reachable from the candidate.
    false_unsupported = (
        oracle_verdict == _UNSUPPORTED and bool(truth.accepted_terminals)
    )
    # Unknown preservation: the solver must not certify unsupported where ground
    # truth cannot rule out an accepted terminal (incomplete coverage).
    unknown_violation = (
        oracle_verdict == _UNSUPPORTED and truth.verdict == _UNKNOWN
    )
    agrees = (
        oracle_verdict == truth.verdict == case.expected_verdict
    )
    return CaseResult(
        case_id=case.case_id,
        family=case.family,
        oracle_verdict=oracle_verdict,
        ground_truth_verdict=truth.verdict,
        expected_verdict=case.expected_verdict,
        certificate_replays=replays,
        false_unsupported=false_unsupported,
        unknown_preservation_violation=unknown_violation,
        agrees=agrees,
    )


def run_suite(
    oracle: EnumerativeSupportOracle,
    expander: ProblemExpander,
    verifier: Verifier,
    state: FiniteDomainState,
    hole_id: HoleId,
    cases: Mapping[str, SolverBenchmarkCase] | tuple[SolverBenchmarkCase, ...],
    *,
    max_nodes: int = 4096,
) -> SuiteReport:
    ordered = (
        tuple(cases.values()) if isinstance(cases, Mapping) else tuple(cases)
    )
    results = tuple(
        run_case(oracle, expander, verifier, state, hole_id, case, max_nodes=max_nodes)
        for case in sorted(ordered, key=lambda c: c.case_id)
    )
    manifest = "|".join(
        f"{r.case_id}:{r.oracle_verdict}:{r.ground_truth_verdict}" for r in results
    )
    digest = hashlib.sha256(manifest.encode("utf-8")).hexdigest()[:16]
    return SuiteReport(results=results, manifest_digest=digest)


# --------------------------------------------------------------------------- #
# Committed v1 reference fixture (family A: finite-domain / certificate)
# --------------------------------------------------------------------------- #

_V1_TREE = {
    "": (("a", "continue"), ("b", "continue"), ("c", "terminal"), ("d", "incomplete")),
    "a": (("a", "terminal"),),  # "aa" accepted
    "b": (("b", "terminal"),),  # "bb" rejected
}
_V1_ACCEPTED = frozenset({"aa"})
_V1_BOUNDS = SolverBounds(
    max_tokens=10_000, max_nodes=10_000, max_depth=32,
    max_backtracks=10_000, max_verifier_calls=10_000,
)


class _V1Verifier:
    profile = "vss4-fixture-verifier-v1"

    def verify(self, program: str) -> VerifyOutcome:
        ok = program in _V1_ACCEPTED
        return VerifyOutcome(
            status=VerifyStatus.ACCEPT if ok else VerifyStatus.REJECT, detail=program
        )


class _V1Expander:
    problem_id = "word:ROOT"
    pack_id = "vss4-fixture-word"
    constraint_version = "v1"
    bounds = _V1_BOUNDS

    def __init__(self) -> None:
        self._prefix_by_fp: dict[str, str] = {}
        self.root = self._state_for("")

    def _state_for(self, prefix: str) -> FiniteDomainState:
        branches = _V1_TREE.get(prefix, ())
        hole = HoleId(namespace="word", path=(len(prefix), prefix or "ROOT"), kind="next")
        values = tuple(
            DomainValue.create("letter", {"prefix": prefix, "letter": b[0]})
            for b in branches
        )
        state = FiniteDomainState(
            problem_id="word:ROOT", pack_id=self.pack_id,
            constraint_version=self.constraint_version, bounds=self.bounds,
            holes=(HoleDomain(hole, values, metadata=(("node", prefix or "ROOT"),)),),
        )
        self._prefix_by_fp[state.fingerprint] = prefix
        return state

    def successor(self, state, hole_id, value) -> ExpandStep:
        prefix = self._prefix_by_fp[state.fingerprint]
        letter = value.payload["letter"]
        _l, kind = next(b for b in _V1_TREE[prefix] if b[0] == letter)
        if kind == "terminal":
            return ExpandStep(ExpandStatus.TERMINAL, program=prefix + letter)
        if kind == "incomplete":
            return ExpandStep(ExpandStatus.INCOMPLETE, coverage="partial")
        return ExpandStep(
            ExpandStatus.CONTINUE, next_state=self._state_for(prefix + letter)
        )


@dataclass(frozen=True)
class ReferenceFixture:
    expander: Any
    verifier: Any
    oracle: EnumerativeSupportOracle
    state: FiniteDomainState
    hole_id: HoleId
    cases: tuple[SolverBenchmarkCase, ...]


def build_reference_fixture() -> ReferenceFixture:
    """Committed v1 closed fixture: candidate 'a' supported, 'b'/'c' unsupported,
    'd' unknown. The whole bounded space is enumerable, so ground truth is exact."""
    exp = _V1Expander()
    ver = _V1Verifier()

    def cand(letter: str) -> DomainValue:
        return DomainValue.create("letter", {"prefix": "", "letter": letter})

    cases = (
        SolverBenchmarkCase("A-supported", "finite-domain", cand("a"), _SUPPORTED),
        SolverBenchmarkCase("A-unsupported-subtree", "finite-domain", cand("b"), _UNSUPPORTED),
        SolverBenchmarkCase("A-unsupported-terminal", "finite-domain", cand("c"), _UNSUPPORTED),
        SolverBenchmarkCase("A-unknown-incomplete", "finite-domain", cand("d"), _UNKNOWN),
    )
    return ReferenceFixture(
        expander=exp, verifier=ver, oracle=EnumerativeSupportOracle(exp, ver),
        state=exp.root, hole_id=exp.root.holes[0].hole_id, cases=cases,
    )


def run_reference_suite() -> SuiteReport:
    fx = build_reference_fixture()
    return run_suite(fx.oracle, fx.expander, fx.verifier, fx.state, fx.hole_id, fx.cases)
