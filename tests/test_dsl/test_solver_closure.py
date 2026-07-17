"""VSS1-01 (SLM-61): certificate-checked exact closure over finite domains.

Closure is a pure orchestration layer over the VSS0-04 oracle, so it is tested
against a stub :class:`SupportProvider` whose verdicts are an exact function of the
current state — this pins multi-pass propagation, stale/tampered-certificate
rejection, certified bottom, partial-budget stops, determinism, and cache identity
without depending on the oracle's search. Torch-free.
"""

from __future__ import annotations

import hashlib
import inspect

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
    SupportCertificate,
    SupportQuery,
    SupportResult,
    SearchCounters,
)
from slm_training.dsl.solver.closure import (
    CertifiedDeduction,
    ClosureResult,
    WitnessRef,
    default_query_order,
    exact_closure,
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
    return HoleId(namespace="closure-test", path=(name,), kind="slot")


def _state(domains: dict[str, list[str]], *, constraint_version: str = "v1") -> FiniteDomainState:
    holes = tuple(
        HoleDomain(_hole(name), tuple(_val(v) for v in values))
        for name, values in domains.items()
    )
    return FiniteDomainState(
        problem_id="closure-test",
        pack_id="fixture",
        constraint_version=constraint_version,
        bounds=_BOUNDS,
        holes=holes,
    )


class _RuleProvider:
    """Stub oracle whose verdict is an exact function of the current state.

    Certificates are canonical for ``(state, query, verdict)``, so an honest
    replay reproduces them and any tamper/stale mismatch is caught.
    """

    def __init__(self, rule, *, profile: str = "stub-v1"):
        self._rule = rule
        self._profile = profile

    @property
    def backend_version(self) -> str:
        return f"stub/{self._profile}"

    def _cert(self, state, query, verdict) -> SupportCertificate:
        common = dict(
            schema_version=1, query=query, verdict=verdict,
            problem_id=state.problem_id, pack_id=state.pack_id,
            constraint_version=state.constraint_version, bounds=state.bounds,
            search_order="canonical-domain-value-v1",
            explored_state_fingerprints=(), verifier_profile=self._profile,
        )
        if verdict is SupportVerdict.UNSUPPORTED:
            return SupportCertificate(
                **common, coverage_observations=("complete",), exhausted=True,
            )
        if verdict is SupportVerdict.SUPPORTED:
            witness = f"witness:{query.candidate.payload_json}"
            return SupportCertificate(
                **common, coverage_observations=("complete",),
                witness_source="stub", witness_digest=_sha(witness), exhausted=False,
            )
        return SupportCertificate(
            **common, coverage_observations=("partial",), exhausted=False,
            stop_reason="incomplete",
        )

    def check(self, state, query) -> SupportResult:
        verdict = self._rule(state, query.hole_id, query.candidate)
        cert = self._cert(state, query, verdict)
        witness = (
            f"witness:{query.candidate.payload_json}"
            if verdict is SupportVerdict.SUPPORTED
            else None
        )
        return SupportResult(verdict, cert, witness=witness, counters=SearchCounters(nodes=1))

    def replay(self, certificate, *, state) -> ReplayResult:
        violations: list[str] = []
        if state.fingerprint != certificate.query.state_fingerprint:
            violations.append("stale fingerprint")
        recomputed = self._rule(state, certificate.query.hole_id, certificate.query.candidate)
        if recomputed != certificate.verdict:
            violations.append("verdict changed")
        if certificate.verdict is SupportVerdict.UNSUPPORTED:
            if not certificate.exhausted:
                violations.append("not exhausted")
            if any(c in {"partial", "none"} for c in certificate.coverage_observations):
                violations.append("incomplete coverage")
        expected = self._cert(state, certificate.query, recomputed)
        if expected.digest != certificate.digest:
            violations.append("digest mismatch")
        return ReplayResult(ok=not violations, verdict=certificate.verdict, violations=tuple(violations))


def _v(value: DomainValue) -> str:
    return value.payload["v"]


# --------------------------------------------------------------------------- #
# Multi-pass propagation
# --------------------------------------------------------------------------- #


def _propagation_rule(state, hole_id, value):
    name = _v(value)
    if name in {"a1", "b1"}:
        return SupportVerdict.UNSUPPORTED
    if name in {"a3", "b2"}:
        return SupportVerdict.SUPPORTED
    if name == "a2":
        b = next(h for h in state.holes if h.hole_id == _hole("B"))
        b1_present = any(_v(dv) == "b1" for dv in b.values)
        return SupportVerdict.UNKNOWN if b1_present else SupportVerdict.UNSUPPORTED
    return SupportVerdict.UNKNOWN


def test_multipass_propagation_removes_only_after_earlier_reduction():
    state = _state({"A": ["a1", "a2", "a3"], "B": ["b1", "b2"]})
    result = exact_closure(state, _RuleProvider(_propagation_rule))
    assert result.reached_fixed_point
    live = {h.hole_id.path[0]: sorted(_v(v) for v in h.values) for h in result.state.holes}
    assert live == {"A": ["a3"], "B": ["b2"]}
    # a2 was removed in a *later* pass than a1/b1 (needs >= 2 passes).
    assert result.counters.passes >= 2
    # a2 was UNKNOWN in pass 1 and UNSUPPORTED in pass 2 -> it appears in a later
    # deduction, and was recorded unknown at least once.
    removed_names = {_v(v) for d in result.deductions for v in d.removed}
    assert removed_names == {"a1", "b1", "a2"}
    assert any(q.candidate.payload["v"] == "a2" for q in result.unknown_queries)
    # SUPPORTED witnesses are kept and recorded.
    assert {w.value.payload["v"] for w in result.witnesses} >= {"a3", "b2"}


def test_every_removed_value_has_a_certificate_and_state_is_subset():
    state = _state({"A": ["a1", "a2", "a3"], "B": ["b1", "b2"]})
    result = exact_closure(state, _RuleProvider(_propagation_rule))
    orig = {_v(v) for h in state.holes for v in h.values}
    kept = {_v(v) for h in result.state.holes for v in h.values}
    assert kept <= orig  # monotone subset
    for deduction in result.deductions:
        assert len(deduction.certificate_ids) == len(deduction.removed)
        assert all(cid for cid in deduction.certificate_ids)


# --------------------------------------------------------------------------- #
# Honesty: UNKNOWN never removes; tampered / stale proofs never mutate state
# --------------------------------------------------------------------------- #


def test_unknown_and_supported_are_kept():
    # Everything is UNKNOWN or SUPPORTED -> no removals, immediate fixed point.
    def rule(state, hole_id, value):
        return SupportVerdict.SUPPORTED if _v(value) == "keep" else SupportVerdict.UNKNOWN

    state = _state({"A": ["keep", "maybe", "dunno"]})
    result = exact_closure(state, _RuleProvider(rule))
    assert result.reached_fixed_point
    assert result.state == state  # unchanged
    assert not result.deductions
    assert result.counters.candidates_removed == 0


def test_tampered_unsupported_certificate_cannot_change_state():
    # Verdict says UNSUPPORTED but the certificate is forged (exhausted=False),
    # so replay fails and nothing is removed.
    class _Dishonest(_RuleProvider):
        def check(self, state, query):
            result = super().check(state, query)
            if result.verdict is SupportVerdict.UNSUPPORTED:
                from dataclasses import replace
                forged = replace(result.certificate, exhausted=False)
                return SupportResult(result.verdict, forged, counters=result.counters)
            return result

    state = _state({"A": ["a1", "a3"]})  # a1 unsupported, a3 supported
    result = exact_closure(state, _Dishonest(_propagation_rule))
    assert result.state == state  # identity unchanged
    assert not result.deductions
    # The value was seen as unsupported but demoted to unknown because replay failed.
    assert result.counters.candidates_removed == 0
    assert any(q.candidate.payload["v"] == "a1" for q in result.unknown_queries)


def test_stale_certificate_is_rejected_by_replay():
    provider = _RuleProvider(_propagation_rule)
    s0 = _state({"A": ["a1", "a3"], "B": ["b1", "b2"]})
    query = SupportQuery(s0.fingerprint, _hole("A"), _val("a1"))
    result = provider.check(s0, query)
    # Replay the s0 certificate against a *different* state -> stale.
    s1 = _state({"A": ["a1", "a3"], "B": ["b2"]})
    replay = provider.replay(result.certificate, state=s1)
    assert not replay.ok
    assert any("stale" in v for v in replay.violations)


# --------------------------------------------------------------------------- #
# Certified bottom + partial budget
# --------------------------------------------------------------------------- #


def test_all_unsupported_yields_certified_bottom():
    def rule(state, hole_id, value):
        return SupportVerdict.UNSUPPORTED

    state = _state({"A": ["a1", "a2"]})
    result = exact_closure(state, _RuleProvider(rule))
    assert result.state.is_bottom
    assert result.reached_fixed_point
    assert result.counters.candidates_removed == 2


def test_partial_budget_stops_with_live_domains():
    def rule(state, hole_id, value):
        return SupportVerdict.UNSUPPORTED if _v(value) == "a1" else SupportVerdict.UNKNOWN

    state = _state({"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b3"]})
    result = exact_closure(state, _RuleProvider(rule), max_queries=1)
    assert not result.reached_fixed_point
    assert result.stop_reason == "budget:max_queries"
    assert result.counters.support_queries <= 1
    # Domains remain live.
    assert any(len(h.values) > 1 for h in result.state.holes)


# --------------------------------------------------------------------------- #
# Determinism + cache identity
# --------------------------------------------------------------------------- #


def test_closure_is_deterministic():
    state = _state({"A": ["a1", "a2", "a3"], "B": ["b1", "b2"]})
    a = exact_closure(state, _RuleProvider(_propagation_rule))
    b = exact_closure(state, _RuleProvider(_propagation_rule))
    assert a.state == b.state
    assert [d.to_dict() for d in a.deductions] == [d.to_dict() for d in b.deductions]
    assert a.counters.to_dict() == b.counters.to_dict()


def test_cache_reuses_results_by_full_identity():
    state = _state({"A": ["a1", "a2", "a3"], "B": ["b1", "b2"]})
    cache: dict = {}
    first = exact_closure(state, _RuleProvider(_propagation_rule), cache=cache)
    assert first.counters.cache_hits == 0
    # A second closure over the same state + a shared cache reuses stored results.
    second = exact_closure(state, _RuleProvider(_propagation_rule), cache=cache)
    assert second.counters.cache_hits > 0
    assert second.state == first.state
    # A different backend version must miss the cache (identity includes it).
    third = exact_closure(
        state, _RuleProvider(_propagation_rule, profile="other-vX"), cache=cache
    )
    assert third.counters.support_queries > 0


def test_default_query_order_is_smallest_domain_first():
    state = _state({"A": ["a1", "a2", "a3"], "B": ["b1", "b2"], "C": ["c1"]})
    order = default_query_order(state)
    names = [h.hole_id.path[0] for h in order]
    assert names == ["B", "A"]  # C is singleton (resolved), B(2) before A(3)


def test_certificate_store_collects_full_certificates():
    state = _state({"A": ["a1", "a3"]})
    store: dict = {}
    result = exact_closure(state, _RuleProvider(_propagation_rule), certificate_store=store)
    # Every deduction certificate id resolves to a stored full certificate.
    for deduction in result.deductions:
        for cid in deduction.certificate_ids:
            assert cid in store
            assert store[cid].digest == cid


def test_records_round_trip_json():
    state = _state({"A": ["a1", "a3"], "B": ["b1", "b2"]})
    result = exact_closure(state, _RuleProvider(_propagation_rule))
    import json

    for deduction in result.deductions:
        blob = json.dumps(deduction.to_dict(), sort_keys=True)
        assert "before_fingerprint" in json.loads(blob)
    for witness in result.witnesses:
        assert "certificate_id" in witness.to_dict()


def test_solver_closure_is_torch_free():
    import slm_training.dsl.solver.closure as closure_mod

    assert "torch" not in inspect.getsource(closure_mod)
