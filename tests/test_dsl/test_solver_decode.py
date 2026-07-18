"""VSS1-03 (SLM-63): decode-time forest pruning via exact closure — core logic.

Torch-free unit tests for `solver_prune`: it removes a candidate only when its
UNSUPPORTED certificate replays, keeps UNKNOWN candidates, returns a subset (so a
later ranker can never reintroduce a removed candidate), leaves non-`complete`
coverage untouched, and yields an empty forest on certified bottom. The full
decode wiring/parity is covered under tests/test_models/.
"""

from __future__ import annotations

import hashlib

import pytest

from slm_training.dsl.grammar.fastpath.compiler_draft import CompletionForest, CompletionPath
from slm_training.dsl.solver.decode import solver_prune
from slm_training.dsl.solver.state import SolverBounds, SupportVerdict
from slm_training.dsl.solver.support import (
    ReplayResult,
    SearchCounters,
    SupportCertificate,
    SupportResult,
)

_BOUNDS = SolverBounds(
    max_tokens=1000, max_nodes=1000, max_depth=32, max_backtracks=1000,
    max_verifier_calls=1000,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class _RuleProvider:
    """Verdict is a function of the candidate's token_ids; certificates replay."""

    def __init__(self, rule):
        self._rule = rule

    @property
    def backend_version(self) -> str:
        return "stub/decode-v1"

    def _cert(self, state, query, verdict) -> SupportCertificate:
        common = dict(
            schema_version=1, query=query, verdict=verdict, problem_id=state.problem_id,
            pack_id=state.pack_id, constraint_version=state.constraint_version,
            bounds=state.bounds, search_order="canonical-domain-value-v1",
            explored_state_fingerprints=(), verifier_profile="stub",
        )
        if verdict is SupportVerdict.UNSUPPORTED:
            return SupportCertificate(**common, coverage_observations=("complete",), exhausted=True)
        if verdict is SupportVerdict.SUPPORTED:
            return SupportCertificate(
                **common, coverage_observations=("complete",), witness_source="stub",
                witness_digest=_sha(query.candidate.payload_json), exhausted=False,
            )
        return SupportCertificate(**common, coverage_observations=("partial",), exhausted=False)

    def check(self, state, query) -> SupportResult:
        verdict = self._rule(tuple(query.candidate.payload["token_ids"]))
        return SupportResult(verdict, self._cert(state, query, verdict), counters=SearchCounters(nodes=1))

    def replay(self, certificate, *, state) -> ReplayResult:
        violations = []
        if state.fingerprint != certificate.query.state_fingerprint:
            violations.append("stale")
        recomputed = self._rule(tuple(certificate.query.candidate.payload["token_ids"]))
        if recomputed != certificate.verdict:
            violations.append("verdict")
        if certificate.verdict is SupportVerdict.UNSUPPORTED and not certificate.exhausted:
            violations.append("not exhausted")
        if self._cert(state, certificate.query, recomputed).digest != certificate.digest:
            violations.append("digest")
        return ReplayResult(ok=not violations, verdict=certificate.verdict, violations=tuple(violations))


def _forest(coverage="complete"):
    return CompletionForest(
        (CompletionPath((10,), "a"), CompletionPath((20,), "b"), CompletionPath((30,), "c")),
        coverage,
        ("NAME", "COMPONENT"),
    )


def _kinds(forest):
    return [p.kind for p in forest.paths]


def test_removes_only_certified_unsupported_and_keeps_unknown():
    rule = lambda tids: SupportVerdict.UNSUPPORTED if tids == (10,) else SupportVerdict.UNKNOWN  # noqa: E731
    forest = _forest()
    pruned, result = solver_prune(
        forest, [1, 2, 3], _RuleProvider(rule),
        pack_id="openui", constraint_version="cv", bounds=_BOUNDS,
    )
    # Only the certified-unsupported candidate (10,) is dropped; UNKNOWNs kept.
    assert _kinds(pruned) == ["b", "c"]
    # Subset of the original -> a ranker cannot reintroduce the removed candidate.
    assert set(p.token_ids for p in pruned.paths) < set(p.token_ids for p in forest.paths)
    assert pruned.coverage == "complete" and pruned.terminals == forest.terminals
    assert result is not None and result.counters.candidates_removed == 1


def test_all_unsupported_yields_empty_forest_certified_bottom():
    forest = _forest()
    pruned, result = solver_prune(
        forest, [1], _RuleProvider(lambda tids: SupportVerdict.UNSUPPORTED),
        pack_id="openui", constraint_version="cv", bounds=_BOUNDS,
    )
    assert pruned.paths == ()  # certified bottom -> decode dead-end/rollback handles it
    assert result is not None and result.state.is_bottom


def test_all_unknown_keeps_forest_identity():
    forest = _forest()
    pruned, result = solver_prune(
        forest, [1], _RuleProvider(lambda tids: SupportVerdict.UNKNOWN),
        pack_id="openui", constraint_version="cv", bounds=_BOUNDS,
    )
    assert pruned is forest  # nothing removed -> identity preserved
    assert result is not None


def test_non_complete_coverage_is_never_pruned():
    for coverage in ("partial", "none"):
        forest = _forest(coverage)
        pruned, result = solver_prune(
            forest, [1], _RuleProvider(lambda tids: SupportVerdict.UNSUPPORTED),
            pack_id="openui", constraint_version="cv", bounds=_BOUNDS,
        )
        # Closure is authoritative only over an exhaustive candidate set.
        assert pruned is forest
        assert result is None


def test_unsupported_policy_rejected():
    with pytest.raises(ValueError, match="solver_unknown_policy"):
        solver_prune(
            _forest(), [1], _RuleProvider(lambda tids: SupportVerdict.UNKNOWN),
            pack_id="openui", constraint_version="cv", bounds=_BOUNDS,
            unknown_policy="drop_unknown",
        )


def test_decode_module_is_torch_free():
    import inspect
    import slm_training.dsl.solver.decode as decode_mod

    assert "torch" not in inspect.getsource(decode_mod)
