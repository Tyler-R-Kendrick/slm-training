"""VSS4-01 solver benchmark: independent ground truth vs the reference oracle (SLM-74).

A tiny closed word-tree fixture whose whole bounded space is enumerable. Every
oracle verdict is cross-checked against a brute-force enumerator, a deliberately
faulty prune is caught, unknown coverage is preserved, and the suite manifest is
deterministic. Torch-free; no model.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

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
    VerifyOutcome,
    VerifyStatus,
)
from slm_training.harnesses.solver_bench import (
    SolverBenchmarkCase,
    ground_truth_verdict,
    run_case,
    run_suite,
)

_BOUNDS = SolverBounds(
    max_tokens=10_000, max_nodes=10_000, max_depth=32,
    max_backtracks=10_000, max_verifier_calls=10_000,
)

# Closed tree. Branch = (letter, kind, coverage). Verifier accepts only "aa".
_TREE = {
    "": (("a", "continue", "complete"), ("b", "continue", "complete"),
         ("c", "terminal", "complete"), ("d", "incomplete", "partial")),
    "a": (("a", "terminal", "complete"),),  # "aa" -> accepted
    "b": (("b", "terminal", "complete"),),  # "bb" -> rejected
}
_ACCEPTED = {"aa"}


class _Verifier:
    profile = "fixture-verifier-v1"

    def verify(self, program: str) -> VerifyOutcome:
        status = VerifyStatus.ACCEPT if program in _ACCEPTED else VerifyStatus.REJECT
        return VerifyOutcome(status=status, detail=program)


class _WordExpander:
    problem_id = "word:ROOT"
    pack_id = "fixture-word"
    constraint_version = "v1"
    bounds = _BOUNDS

    def __init__(self) -> None:
        self._prefix_by_fp: dict[str, str] = {}
        self._root = self._state_for("")

    def root_state(self) -> FiniteDomainState:
        return self._root

    def _state_for(self, prefix: str) -> FiniteDomainState:
        branches = _TREE.get(prefix, ())
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
        _l, kind, coverage = next(b for b in _TREE[prefix] if b[0] == letter)
        if kind == "terminal":
            return ExpandStep(ExpandStatus.TERMINAL, program=prefix + letter, coverage=coverage)
        if kind == "dead":
            return ExpandStep(ExpandStatus.DEAD, coverage=coverage)
        if kind == "incomplete":
            return ExpandStep(ExpandStatus.INCOMPLETE, coverage=coverage)
        return ExpandStep(ExpandStatus.CONTINUE, next_state=self._state_for(prefix + letter), coverage=coverage)


def _cand(letter: str) -> DomainValue:
    return DomainValue.create("letter", {"prefix": "", "letter": letter})


_CASES = (
    SolverBenchmarkCase("A-supported", "finite-domain", _cand("a"), "supported"),
    SolverBenchmarkCase("A-unsupported-subtree", "finite-domain", _cand("b"), "unsupported"),
    SolverBenchmarkCase("A-unsupported-terminal", "finite-domain", _cand("c"), "unsupported"),
    SolverBenchmarkCase("A-unknown-incomplete", "finite-domain", _cand("d"), "unknown"),
)


@pytest.fixture()
def harness():
    exp = _WordExpander()
    ver = _Verifier()
    oracle = EnumerativeSupportOracle(exp, ver)
    root = exp.root_state()
    hole = root.holes[0].hole_id
    return exp, ver, oracle, root, hole


def test_ground_truth_matches_expected(harness):
    exp, ver, _oracle, root, hole = harness
    for case in _CASES:
        gt = ground_truth_verdict(exp, ver, root, hole, case.candidate)
        assert gt.verdict == case.expected_verdict, case.case_id
    # supported case has a witness terminal; unsupported/unknown do not
    gt_a = ground_truth_verdict(exp, ver, root, hole, _cand("a"))
    assert gt_a.accepted_terminals  # "aa" digest present


def test_oracle_agrees_with_ground_truth_and_replays(harness):
    exp, ver, oracle, root, hole = harness
    report = run_suite(oracle, exp, ver, root, hole, _CASES)
    assert report.passed, [r.to_dict() for r in report.hard_failures]
    for r in report.results:
        assert r.agrees, r.case_id
        assert r.certificate_replays, r.case_id
        assert not r.false_unsupported
        assert not r.unknown_preservation_violation


def test_faulty_prune_is_caught(harness):
    """A solver that wrongly certifies a supported candidate as unsupported must be
    flagged as a false certified prune by the benchmark cross-check."""
    exp, ver, oracle, root, hole = harness

    class _ForceUnsupported:
        def check(self, state, query):
            return replace(oracle.check(state, query), verdict=SupportVerdict.UNSUPPORTED)

    case = _CASES[0]  # candidate "a" is genuinely supported
    result = run_case(_ForceUnsupported(), exp, ver, root, hole, case)
    assert result.oracle_verdict == "unsupported"
    assert result.false_unsupported  # ground truth found "aa" -> caught
    assert not result.agrees


def test_unknown_is_preserved(harness):
    exp, ver, oracle, root, hole = harness
    result = run_case(oracle, exp, ver, root, hole, _CASES[3])  # candidate "d"
    assert result.oracle_verdict == "unknown"
    assert result.ground_truth_verdict == "unknown"
    assert not result.unknown_preservation_violation


def test_manifest_is_deterministic(harness):
    exp, ver, oracle, root, hole = harness
    a = run_suite(oracle, exp, ver, root, hole, _CASES)
    b = run_suite(oracle, exp, ver, root, hole, _CASES)
    assert a.manifest_digest == b.manifest_digest
    # order independence: same cases in a different order -> same digest
    c = run_suite(oracle, exp, ver, root, hole, tuple(reversed(_CASES)))
    assert a.manifest_digest == c.manifest_digest


def test_bad_expected_verdict_rejected():
    with pytest.raises(ValueError):
        SolverBenchmarkCase("bad", "finite-domain", _cand("a"), "maybe")
