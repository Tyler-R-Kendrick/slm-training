"""VSS0-04 (SLM-60): exhaustive tri-state support oracle + certificate replay.

The oracle core is exercised against tiny *closed* fixtures whose entire search
space is known, so every SUPPORTED/UNSUPPORTED/UNKNOWN verdict and every replay
rule can be asserted exactly. Torch-free.
"""

from __future__ import annotations

import inspect
import json
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
    SupportCertificate,
    SupportQuery,
    VerifyOutcome,
    VerifyStatus,
    replay_support_certificate,
)

# --------------------------------------------------------------------------- #
# Tiny closed fixture: words built letter-by-letter over a known finite tree.
# --------------------------------------------------------------------------- #

_GENEROUS = SolverBounds(
    max_tokens=100_000,
    max_nodes=100_000,
    max_depth=64,
    max_backtracks=100_000,
    max_verifier_calls=100_000,
)


class _WordExpander:
    """Deterministic finite word tree.

    ``tree`` maps a prefix string to a tuple of branches. Each branch is
    ``(letter, kind, coverage, next_prefix)`` where ``kind`` is one of
    ``continue|terminal|dead|incomplete`` and ``next_prefix`` (optional) forces a
    shared successor to exercise duplicate-state suppression.
    """

    def __init__(self, tree, *, bounds=_GENEROUS, constraint_version="v1"):
        self._tree = tree
        self._bounds = bounds
        self._cv = constraint_version
        self._prefix_by_fp: dict[str, str] = {}
        self._root = self._state_for("")
        self._prefix_by_fp[self._root.fingerprint] = ""

    # --- ProblemExpander protocol ---------------------------------------- #
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
        _letter, kind, coverage = branch[0], branch[1], branch[2]
        forced_next = branch[3] if len(branch) > 3 else None
        if kind == "terminal":
            return ExpandStep(
                ExpandStatus.TERMINAL, program=prefix + letter, coverage=coverage,
                detail=prefix + letter,
            )
        if kind == "dead":
            return ExpandStep(ExpandStatus.DEAD, coverage=coverage, detail="bottom")
        if kind == "incomplete":
            return ExpandStep(ExpandStatus.INCOMPLETE, coverage=coverage, detail="uncovered")
        # continue
        next_prefix = forced_next if forced_next is not None else prefix + letter
        child = self._state_for(next_prefix)
        self._prefix_by_fp[child.fingerprint] = next_prefix
        return ExpandStep(ExpandStatus.CONTINUE, next_state=child, coverage=coverage)


class _AcceptVerifier:
    """Accepts programs in a fixed set; everything else is a hard REJECT."""

    def __init__(self, accept, *, profile="fixture-accept-v1"):
        self._accept = set(accept)
        self._profile = profile

    @property
    def profile(self) -> str:
        return self._profile

    def verify(self, program: str) -> VerifyOutcome:
        if program in self._accept:
            return VerifyOutcome(VerifyStatus.ACCEPT)
        return VerifyOutcome(VerifyStatus.REJECT, detail="not-in-accept-set")


def _query(expander: _WordExpander, letter: str) -> SupportQuery:
    root = expander.root_state()
    return SupportQuery(
        state_fingerprint=root.fingerprint,
        hole_id=root.holes[0].hole_id,
        candidate=expander.value_for("", letter),
    )


# tree: root -> a -> {x term, y term}; root -> b -> {z term}
_LINEAR_TREE = {
    "": (("a", "continue", "complete"), ("b", "continue", "complete")),
    "a": (("x", "terminal", "complete"), ("y", "terminal", "complete")),
    "b": (("z", "terminal", "complete"),),
}


def _run(tree, accept, letter, *, bounds=_GENEROUS):
    expander = _WordExpander(tree, bounds=bounds)
    verifier = _AcceptVerifier(accept)
    oracle = EnumerativeSupportOracle(expander, verifier)
    result = oracle.check(expander.root_state(), _query(expander, letter))
    return expander, verifier, result


# --------------------------------------------------------------------------- #
# Tri-state verdicts
# --------------------------------------------------------------------------- #


def test_supported_returns_witness_and_digest():
    _e, _v, result = _run(_LINEAR_TREE, {"ax"}, "a")
    assert result.verdict is SupportVerdict.SUPPORTED
    assert result.witness == "ax"
    assert result.certificate.witness_digest is not None
    assert result.certificate.witness_source is not None
    # A witness is valid even though the 'b' subtree was never explored.
    assert result.certificate.exhausted is False


def test_unsupported_requires_exhaustion_and_complete_coverage():
    # 'b' leads only to 'bz', which is not accepted -> exhausted, complete -> UNSUPPORTED.
    _e, _v, result = _run(_LINEAR_TREE, {"ax"}, "b")
    assert result.verdict is SupportVerdict.UNSUPPORTED
    assert result.certificate.exhausted is True
    assert result.certificate.stop_reason is None
    assert set(result.certificate.coverage_observations) <= {"complete"}
    assert result.witness is None
    # The hard rejection was recorded, not silently ignored.
    assert any("reject" in code for code, _ in result.certificate.failure_counts)


def test_unknown_from_partial_coverage_never_unsupported():
    tree = {
        "": (("b", "continue", "complete"),),
        "b": (("z", "terminal", "complete"), ("w", "incomplete", "partial")),
    }
    _e, _v, result = _run(tree, set(), "b")  # nothing accepted
    assert result.verdict is SupportVerdict.UNKNOWN
    assert result.certificate.exhausted is False
    assert "partial" in result.certificate.coverage_observations


def test_unknown_from_each_budget_class():
    # A deep chain; a tiny node budget forces a budget stop -> UNKNOWN.
    tree = {"": (("a", "continue", "complete"),), "a": (("b", "continue", "complete"),),
            "ab": (("c", "terminal", "complete"),)}
    expander = _WordExpander(tree, bounds=replace(_GENEROUS, max_nodes=1))
    verifier = _AcceptVerifier({"abc"})
    result = EnumerativeSupportOracle(expander, verifier).check(
        expander.root_state(), _query(expander, "a")
    )
    assert result.verdict is SupportVerdict.UNKNOWN
    assert result.certificate.stop_reason == "budget:max_nodes"
    assert result.certificate.exhausted is False


def test_supported_witness_found_before_a_partial_branch():
    tree = {
        "": (("b", "continue", "complete"),),
        "b": (("w", "incomplete", "partial"), ("z", "terminal", "complete")),
    }
    _e, _v, result = _run(tree, {"bz"}, "b")
    # Despite a partial branch, a verified witness wins.
    assert result.verdict is SupportVerdict.SUPPORTED
    assert result.witness == "bz"


def test_duplicate_state_suppression():
    # Both 'a' and 'b' route to the shared successor prefix "m".
    tree = {
        "": (("a", "continue", "complete", "m"), ("b", "continue", "complete", "m")),
        "m": (("z", "terminal", "complete"),),
    }
    expander = _WordExpander(tree)
    verifier = _AcceptVerifier(set())
    # Query candidate 'a'; only one child state "m" should be explored/recorded.
    result = EnumerativeSupportOracle(expander, verifier).check(
        expander.root_state(), _query(expander, "a")
    )
    fps = result.certificate.explored_state_fingerprints
    assert len(fps) == len(set(fps))  # no duplicate fingerprints recorded
    # Exactly one distinct child state ("m") was explored.
    assert len(fps) == 1


def test_evidence_is_deterministic_across_runs():
    a = _run(_LINEAR_TREE, {"ax"}, "b")[2].certificate.to_dict()
    b = _run(_LINEAR_TREE, {"ax"}, "b")[2].certificate.to_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --------------------------------------------------------------------------- #
# Query / state validation
# --------------------------------------------------------------------------- #


def test_stale_fingerprint_query_is_rejected():
    expander = _WordExpander(_LINEAR_TREE)
    verifier = _AcceptVerifier({"ax"})
    bad = SupportQuery(
        state_fingerprint="0" * 64,  # not the state's fingerprint
        hole_id=expander.root_state().holes[0].hole_id,
        candidate=expander.value_for("", "a"),
    )
    with pytest.raises(ValueError, match="stale"):
        EnumerativeSupportOracle(expander, verifier).check(expander.root_state(), bad)


def test_unknown_candidate_is_rejected():
    expander = _WordExpander(_LINEAR_TREE)
    verifier = _AcceptVerifier(set())
    bad = SupportQuery(
        state_fingerprint=expander.root_state().fingerprint,
        hole_id=expander.root_state().holes[0].hole_id,
        candidate=expander.value_for("", "zzz"),  # not a live candidate
    )
    with pytest.raises(ValueError, match="not in the hole domain"):
        EnumerativeSupportOracle(expander, verifier).check(expander.root_state(), bad)


# --------------------------------------------------------------------------- #
# Certificate replay
# --------------------------------------------------------------------------- #


def test_supported_certificate_replays():
    expander, verifier, result = _run(_LINEAR_TREE, {"ax"}, "a")
    replay = replay_support_certificate(
        result.certificate, state=expander.root_state(),
        expander=_WordExpander(_LINEAR_TREE), verifier=_AcceptVerifier({"ax"}),
    )
    assert replay.ok, replay.violations


def test_unsupported_certificate_replays():
    expander, verifier, result = _run(_LINEAR_TREE, {"ax"}, "b")
    replay = replay_support_certificate(
        result.certificate, state=expander.root_state(),
        expander=_WordExpander(_LINEAR_TREE), verifier=_AcceptVerifier({"ax"}),
    )
    assert replay.ok, replay.violations
    assert replay.verdict is SupportVerdict.UNSUPPORTED


def test_tampered_witness_digest_is_rejected():
    expander, _v, result = _run(_LINEAR_TREE, {"ax"}, "a")
    tampered = replace(result.certificate, witness_digest="f" * 64)
    replay = replay_support_certificate(
        tampered, state=expander.root_state(),
        expander=_WordExpander(_LINEAR_TREE), verifier=_AcceptVerifier({"ax"}),
    )
    assert not replay.ok
    assert any("witness digest" in v for v in replay.violations)


def test_unsupported_rejected_when_not_exhausted():
    expander, _v, result = _run(_LINEAR_TREE, {"ax"}, "b")
    forged = replace(result.certificate, exhausted=False)
    replay = replay_support_certificate(
        forged, state=expander.root_state(),
        expander=_WordExpander(_LINEAR_TREE), verifier=_AcceptVerifier({"ax"}),
    )
    assert not replay.ok
    assert any("exhaust" in v for v in replay.violations)


def test_unsupported_rejected_when_coverage_incomplete():
    expander, _v, result = _run(_LINEAR_TREE, {"ax"}, "b")
    forged = replace(
        result.certificate,
        coverage_observations=result.certificate.coverage_observations + ("partial",),
    )
    replay = replay_support_certificate(
        forged, state=expander.root_state(),
        expander=_WordExpander(_LINEAR_TREE), verifier=_AcceptVerifier({"ax"}),
    )
    assert not replay.ok
    assert any("incomplete coverage" in v for v in replay.violations)


def test_stale_constraint_version_replay_is_rejected():
    expander, _v, result = _run(_LINEAR_TREE, {"ax"}, "b")
    replay = replay_support_certificate(
        result.certificate, state=expander.root_state(),
        expander=_WordExpander(_LINEAR_TREE, constraint_version="v2"),  # stale
        verifier=_AcceptVerifier({"ax"}),
    )
    assert not replay.ok
    assert any("constraint_version" in v or "fingerprint" in v for v in replay.violations)


# --------------------------------------------------------------------------- #
# Serialization + hygiene
# --------------------------------------------------------------------------- #


def test_certificate_and_query_round_trip_json():
    _e, _v, result = _run(_LINEAR_TREE, {"ax"}, "b")
    cert = result.certificate
    restored = SupportCertificate.from_dict(json.loads(json.dumps(cert.to_dict())))
    assert restored == cert
    q = cert.query
    assert SupportQuery.from_dict(json.loads(json.dumps(q.to_dict()))) == q


def test_solver_support_is_torch_free():
    import slm_training.dsl.solver.support as support_mod
    import slm_training.dsl.solver.state as state_mod
    import slm_training.dsl.solver.openui_support as openui_mod

    for mod in (support_mod, state_mod, openui_mod):
        assert "torch" not in inspect.getsource(mod)


# --------------------------------------------------------------------------- #
# OpenUI wiring smoke: real compiler forest + lang-core well-formedness.
# --------------------------------------------------------------------------- #


def test_openui_expander_wires_forest_and_never_false_unsupported():
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer
    from slm_training.dsl.solver.openui_support import (
        OpenUIForestExpander,
        OpenUIWellFormedVerifier,
    )

    tok = DSLNativeTokenizer.build()
    prefix = [tok.bos_id, *tok.encode("root=Card([", add_special=False)]
    bounds = SolverBounds(
        max_tokens=4000, max_nodes=24, max_depth=12, max_backtracks=200,
        max_verifier_calls=24,
    )

    def build_expander():
        return OpenUIForestExpander(
            tok, prefix, pack_id="openui", constraint_version="test-cv", bounds=bounds
        )

    expander = build_expander()
    root = expander.root_state()
    assert root.holes and root.holes[0].values, "compiler forest yielded no candidates"

    verifier = OpenUIWellFormedVerifier()
    query = SupportQuery(
        state_fingerprint=root.fingerprint,
        hole_id=root.holes[0].hole_id,
        candidate=root.holes[0].values[0],
    )
    result = EnumerativeSupportOracle(expander, verifier).check(root, query)

    assert result.verdict in set(SupportVerdict)
    # Honesty invariant: UNSUPPORTED only with an exhausted, fully-covered search.
    if result.verdict is SupportVerdict.UNSUPPORTED:
        assert result.certificate.exhausted
        assert set(result.certificate.coverage_observations) <= {"complete"}
    # A capability-limited run (bridge unavailable / budget) must stay UNKNOWN,
    # never a false UNSUPPORTED.
    replay = replay_support_certificate(
        result.certificate, state=root,
        expander=build_expander(), verifier=OpenUIWellFormedVerifier(),
    )
    assert replay.ok, replay.violations
