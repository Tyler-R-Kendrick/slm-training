"""VSS1-02 (SLM-61): bounded proof-carrying search controller.

Tiny finite problems with known branches pin the separation of certified
deductions (irreversible, closure-owned) from reversible decisions (ranker-ordered)
and local nogoods, and the sound tri-state termination: CERTIFIED_UNSAT only when
the whole tree closes by certified deductions with no UNKNOWN / verifier-rejection /
budget truncation. Torch-free.
"""

from __future__ import annotations

import hashlib
import inspect

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
    ReplayResult,
    SearchCounters,
    SupportCertificate,
    SupportResult,
)
from slm_training.dsl.solver.controller import (
    BaselineRanker,
    SearchStatus,
    TerminalOutcome,
    search,
)

_BOUNDS = SolverBounds(
    max_tokens=1000, max_nodes=1000, max_depth=32, max_backtracks=1000,
    max_verifier_calls=1000,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _val(name: str) -> DomainValue:
    return DomainValue.create("v", {"v": name})


def _hole(name: str) -> HoleId:
    return HoleId(namespace="ctl-test", path=(name,), kind="slot")


def _state(domains: dict[str, list[str]]) -> FiniteDomainState:
    holes = tuple(
        HoleDomain(_hole(name), tuple(_val(v) for v in values))
        for name, values in domains.items()
    )
    return FiniteDomainState(
        problem_id="ctl-test", pack_id="fixture", constraint_version="v1",
        bounds=_BOUNDS, holes=holes,
    )


def _vname(value: DomainValue) -> str:
    return value.payload["v"]


class _RuleProvider:
    """State-dependent stub oracle with canonical, replayable certificates."""

    def __init__(self, rule, *, profile="stub-v1"):
        self._rule = rule
        self._profile = profile

    @property
    def backend_version(self) -> str:
        return f"stub/{self._profile}"

    def _cert(self, state, query, verdict) -> SupportCertificate:
        common = dict(
            schema_version=1, query=query, verdict=verdict, problem_id=state.problem_id,
            pack_id=state.pack_id, constraint_version=state.constraint_version,
            bounds=state.bounds, search_order="canonical-domain-value-v1",
            explored_state_fingerprints=(), verifier_profile=self._profile,
        )
        if verdict is SupportVerdict.UNSUPPORTED:
            return SupportCertificate(**common, coverage_observations=("complete",), exhausted=True)
        if verdict is SupportVerdict.SUPPORTED:
            return SupportCertificate(
                **common, coverage_observations=("complete",), witness_source="stub",
                witness_digest=_sha(query.candidate.payload_json), exhausted=False,
            )
        return SupportCertificate(
            **common, coverage_observations=("partial",), exhausted=False, stop_reason="incomplete",
        )

    def check(self, state, query) -> SupportResult:
        verdict = self._rule(state, query.hole_id, query.candidate)
        return SupportResult(verdict, self._cert(state, query, verdict), counters=SearchCounters(nodes=1))

    def replay(self, certificate, *, state) -> ReplayResult:
        violations = []
        if state.fingerprint != certificate.query.state_fingerprint:
            violations.append("stale")
        recomputed = self._rule(state, certificate.query.hole_id, certificate.query.candidate)
        if recomputed != certificate.verdict:
            violations.append("verdict")
        if certificate.verdict is SupportVerdict.UNSUPPORTED and not certificate.exhausted:
            violations.append("not exhausted")
        if self._cert(state, certificate.query, recomputed).digest != certificate.digest:
            violations.append("digest")
        return ReplayResult(ok=not violations, verdict=certificate.verdict, violations=tuple(violations))


class _Terminal:
    """Accepts a structurally-solved assignment iff `accept(assignment)` is true."""

    def __init__(self, accept):
        self._accept = accept

    def check(self, state) -> TerminalOutcome:
        assignment = {h.hole_id.path[0]: _vname(h.values[0]) for h in state.holes}
        ok = self._accept(assignment)
        return TerminalOutcome(accepted=ok, source="fixture" if ok else None,
                               report={"assignment": assignment, "accepted": ok})


_ALL_UNKNOWN = lambda state, hole, value: SupportVerdict.UNKNOWN  # noqa: E731
_ACCEPT_ALL = lambda assignment: True  # noqa: E731


def _run(domains, rule, accept, **kw):
    return search(_state(domains), _RuleProvider(rule), _Terminal(accept), **kw)


# --------------------------------------------------------------------------- #
# Certified deductions before decisions; solving
# --------------------------------------------------------------------------- #


def test_exact_deductions_before_first_decision():
    def rule(state, hole, value):
        return SupportVerdict.UNSUPPORTED if _vname(value) == "x_bad" else SupportVerdict.UNKNOWN

    result = _run({"X": ["x_ok", "x_bad"], "Y": ["y1", "y2"]}, rule,
                  lambda a: a == {"X": "x_ok", "Y": "y1"})
    assert result.status is SearchStatus.SOLVED
    # x_bad was removed by closure before any decision; decisions only concern Y.
    removed = {_vname(v) for d in result.deductions for v in d.removed}
    assert "x_bad" in removed
    assert all(d.hole_id.path[0] == "Y" for d in result.decisions)


def test_solve_after_one_and_multiple_decisions():
    one = _run({"X": ["x1", "x2"]}, _ALL_UNKNOWN, lambda a: a["X"] == "x1")
    assert one.status is SearchStatus.SOLVED
    assert one.source is not None and one.verifier_report is not None
    assert len(one.decisions) == 1

    multi = _run({"X": ["x1", "x2"], "Y": ["y1", "y2"]}, _ALL_UNKNOWN,
                 lambda a: a == {"X": "x1", "Y": "y1"})
    assert multi.status is SearchStatus.SOLVED
    assert len(multi.decisions) == 2


def test_failed_first_branch_rolls_back_to_alternate():
    # Only x2 yields an acceptable terminal; x1 is rejected -> backtrack -> x2.
    result = _run({"X": ["x1", "x2"]}, _ALL_UNKNOWN, lambda a: a["X"] == "x2")
    assert result.status is SearchStatus.SOLVED
    assert [_vname(d.chosen) for d in result.decisions] == ["x1", "x2"]
    # A local nogood was recorded for the failed x1 branch...
    assert result.nogoods
    assert any(_vname(v) == "x1" for ng in result.nogoods for _h, v in ng.assignment)
    # ...but it is NOT a certified deduction.
    removed = {_vname(v) for d in result.deductions for v in d.removed}
    assert "x1" not in removed


# --------------------------------------------------------------------------- #
# Sound termination
# --------------------------------------------------------------------------- #


def test_root_certified_unsat_only_when_all_branches_proof_closed():
    result = _run({"X": ["a", "b"]},
                  lambda s, h, v: SupportVerdict.UNSUPPORTED, _ACCEPT_ALL)
    assert result.status is SearchStatus.CERTIFIED_UNSAT
    assert result.state.is_bottom
    assert result.deductions
    assert not result.decisions


def test_unknown_branch_prevents_certified_unsat():
    # 'a' is certified-unsupported, 'b' is UNKNOWN and its terminal is rejected.
    def rule(state, hole, value):
        return SupportVerdict.UNSUPPORTED if _vname(value) == "a" else SupportVerdict.UNKNOWN

    result = _run({"X": ["a", "b"]}, rule, lambda a: False)  # terminal rejects everything
    assert result.status is SearchStatus.UNKNOWN
    assert result.status is not SearchStatus.CERTIFIED_UNSAT


def test_terminal_verifier_failure_causes_conflict_backtrack():
    # Single value, terminal rejects it -> no alternatives -> honest UNKNOWN.
    result = _run({"X": ["x1", "x2"]}, _ALL_UNKNOWN, lambda a: False)
    assert result.status is SearchStatus.UNKNOWN
    assert result.nogoods  # verifier failures recorded as nogoods
    assert all("verifier" in ng.provenance or "certified" in ng.provenance
               for ng in result.nogoods)


# --------------------------------------------------------------------------- #
# Ranker safety
# --------------------------------------------------------------------------- #


class _ReverseRanker:
    ranker_id = "reverse-vX"

    def rank(self, state, hole_id, values):
        return tuple(reversed(values))


class _DroppingRanker:
    ranker_id = "drop-vX"

    def rank(self, state, hole_id, values):
        return values[:-1]  # drops a candidate


class _AddingRanker:
    ranker_id = "add-vX"

    def rank(self, state, hole_id, values):
        return values + (DomainValue.create("v", {"v": "phantom"}),)


class _DupRanker:
    ranker_id = "dup-vX"

    def rank(self, state, hole_id, values):
        return (values[0],) + values


@pytest.mark.parametrize("ranker", [_DroppingRanker(), _AddingRanker(), _DupRanker()])
def test_ranker_that_alters_membership_is_rejected(ranker):
    with pytest.raises(ValueError):
        search(_state({"X": ["x1", "x2"]}), _RuleProvider(_ALL_UNKNOWN), _Terminal(_ACCEPT_ALL),
               ranker=ranker)


def test_adversarial_but_valid_ranker_cannot_change_hard_membership():
    domains = {"X": ["x1", "x2"], "Y": ["y1", "y2"]}
    accept = lambda a: a == {"X": "x2", "Y": "y2"}  # noqa: E731
    baseline = search(_state(domains), _RuleProvider(_ALL_UNKNOWN), _Terminal(accept),
                      ranker=BaselineRanker())
    reversed_run = search(_state(domains), _RuleProvider(_ALL_UNKNOWN), _Terminal(accept),
                          ranker=_ReverseRanker())
    # Both reach the same verified solution; only the trajectory order differs.
    assert baseline.status is reversed_run.status is SearchStatus.SOLVED
    assert baseline.state == reversed_run.state


# --------------------------------------------------------------------------- #
# Determinism + budgets
# --------------------------------------------------------------------------- #


def test_deterministic_baseline_trajectory():
    domains = {"X": ["x1", "x2"], "Y": ["y1", "y2"]}
    accept = lambda a: a == {"X": "x2", "Y": "y1"}  # noqa: E731
    a = _run(domains, _ALL_UNKNOWN, accept)
    b = _run(domains, _ALL_UNKNOWN, accept)
    assert [d.to_dict() for d in a.decisions] == [d.to_dict() for d in b.decisions]
    assert a.counters.to_dict() == b.counters.to_dict()
    assert a.state == b.state


def test_decision_budget_stops_honestly():
    result = _run({"X": ["x1", "x2"], "Y": ["y1", "y2"]}, _ALL_UNKNOWN, lambda a: False,
                  max_decisions=1)
    assert result.status is SearchStatus.BUDGET_EXHAUSTED
    assert result.stop_reason == "budget:max_decisions"


def test_solver_controller_is_torch_free():
    import slm_training.dsl.solver.controller as controller_mod

    assert "torch" not in inspect.getsource(controller_mod)
