"""OpenUIForestExpander on the packed kernel: payload parity + honesty.

The session-backed expander must produce exactly the ``FiniteDomainState``
payloads a fresh-forest reference produces (interned state ids never escape
into certificates), certificate replay must still succeed, and the honesty
invariants hold: timeout/unavailable stays UNKNOWN (never UNSUPPORTED), and
certified bottom (dead/illegal prefix) stays distinct from an
incomplete-authority state.
"""

from __future__ import annotations

import pytest

from slm_training.dsl.grammar.fastpath.compiler_draft import build_completion_forest
from slm_training.dsl.solver.adapters import completion_forest_state
from slm_training.dsl.solver.openui_support import (
    OpenUIForestExpander,
    OpenUIWellFormedVerifier,
)
from slm_training.dsl.solver.state import SolverBounds
from slm_training.dsl.solver.support import (
    EnumerativeSupportOracle,
    ExpandStatus,
    SupportQuery,
    SupportVerdict,
    VerifyOutcome,
    VerifyStatus,
    replay_support_certificate,
)
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer
from slm_training.models.decode_stats import collect_decode_stats


@pytest.fixture(scope="module")
def tok() -> DSLNativeTokenizer:
    return DSLNativeTokenizer.build()


@pytest.fixture
def bounds() -> SolverBounds:
    return SolverBounds(
        max_tokens=4000,
        max_nodes=24,
        max_depth=12,
        max_backtracks=200,
        max_verifier_calls=24,
    )


def _expander(tok, prefix, bounds) -> OpenUIForestExpander:
    return OpenUIForestExpander(
        tok, prefix, pack_id="openui", constraint_version="test-cv", bounds=bounds
    )


def _reference_state(tok, prefix, forest, bounds):
    return completion_forest_state(
        prefix_ids=prefix,
        forest=forest,
        pack_id="openui",
        constraint_version="test-cv",
        bounds=bounds,
    )


_PREFIXES = (
    "root = Card([",
    "root = Card([b1",
    "root = Card([b1,",
    'root = TextContent(":slot_0")',
)


def _state_payload(state) -> tuple:
    return (
        state.fingerprint,
        tuple(
            (hole.hole_id, tuple(value.payload for value in hole.values), hole.metadata)
            for hole in state.holes
        ),
    )


@pytest.mark.parametrize("text", _PREFIXES, ids=lambda p: repr(p[-16:]))
def test_expander_root_payload_matches_fresh_forest(
    tok: DSLNativeTokenizer, bounds: SolverBounds, text: str
) -> None:
    prefix = tuple([tok.bos_id, *tok.encode(text, add_special=False)])
    expander = _expander(tok, prefix, bounds)
    reference = _reference_state(
        tok,
        prefix,
        build_completion_forest(tok, list(prefix), max_path_tokens=8),
        bounds,
    )
    assert _state_payload(expander.root_state()) == _state_payload(reference)


@pytest.mark.parametrize("text", _PREFIXES, ids=lambda p: repr(p[-16:]))
def test_expander_successor_payloads_match_fresh_forest(
    tok: DSLNativeTokenizer, bounds: SolverBounds, text: str
) -> None:
    prefix = tuple([tok.bos_id, *tok.encode(text, add_special=False)])
    expander = _expander(tok, prefix, bounds)
    root = expander.root_state()
    hole = root.holes[0]
    for value in hole.values:
        step = expander.successor(root, hole.hole_id, value)
        token_ids = tuple(int(t) for t in value.payload.get("token_ids", ()))
        kind = str(value.payload.get("kind", ""))
        if kind == "eos" or token_ids == (int(tok.eos_id),):
            assert step.status is ExpandStatus.TERMINAL
            continue
        new_prefix = prefix + token_ids
        reference_forest = build_completion_forest(
            tok, list(new_prefix), max_path_tokens=8
        )
        if reference_forest.coverage == "none":
            assert step.status is ExpandStatus.DEAD
            assert step.coverage == "none"
            continue
        if not reference_forest.paths:
            assert step.status is ExpandStatus.INCOMPLETE
            continue
        assert step.status is ExpandStatus.CONTINUE
        reference = _reference_state(tok, new_prefix, reference_forest, bounds)
        assert _state_payload(step.next_state) == _state_payload(reference)


def test_certificate_replay_still_succeeds(
    tok: DSLNativeTokenizer, bounds: SolverBounds
) -> None:
    prefix = [tok.bos_id, *tok.encode("root=Card([", add_special=False)]
    expander = _expander(tok, prefix, bounds)
    root = expander.root_state()
    assert root.holes and root.holes[0].values
    query = SupportQuery(
        state_fingerprint=root.fingerprint,
        hole_id=root.holes[0].hole_id,
        candidate=root.holes[0].values[0],
    )
    result = EnumerativeSupportOracle(expander, OpenUIWellFormedVerifier()).check(
        root, query
    )
    assert result.verdict in set(SupportVerdict)
    if result.verdict is SupportVerdict.UNSUPPORTED:
        assert result.certificate.exhausted
        assert set(result.certificate.coverage_observations) <= {"complete"}
    replay = replay_support_certificate(
        result.certificate,
        state=root,
        expander=_expander(tok, prefix, bounds),
        verifier=OpenUIWellFormedVerifier(),
    )
    assert replay.ok, replay.violations


def test_unavailable_verifier_stays_unknown(
    tok: DSLNativeTokenizer, bounds: SolverBounds
) -> None:
    class _UnavailableVerifier:
        profile = "test/unavailable"

        def verify(self, program: str) -> VerifyOutcome:
            return VerifyOutcome(VerifyStatus.UNAVAILABLE, detail="timeout")

    prefix = [tok.bos_id, *tok.encode("root=Card([", add_special=False)]
    expander = _expander(tok, prefix, bounds)
    root = expander.root_state()
    query = SupportQuery(
        state_fingerprint=root.fingerprint,
        hole_id=root.holes[0].hole_id,
        candidate=root.holes[0].values[0],
    )
    result = EnumerativeSupportOracle(expander, _UnavailableVerifier()).check(
        root, query
    )
    # A capability gap (timeout / unavailable) is UNKNOWN, never UNSUPPORTED.
    assert result.verdict is SupportVerdict.UNKNOWN


def test_certified_bottom_stays_distinct_from_empty_authority(
    tok: DSLNativeTokenizer, bounds: SolverBounds
) -> None:
    prefix = tuple([tok.bos_id, *tok.encode("root = Card([", add_special=False)])
    expander = _expander(tok, prefix, bounds)
    root = expander.root_state()
    hole = root.holes[0]

    # Certified bottom: a payload whose tokens the grammar rejects is DEAD
    # with complete coverage authority (coverage="none").
    from slm_training.dsl.solver.state import DomainValue

    rpar = int(tok.token_to_id[")"])
    dead = DomainValue.create("completion_path", {"kind": "component", "token_ids": [rpar]})
    step = expander.successor(root, hole.hole_id, dead)
    assert step.status is ExpandStatus.DEAD
    assert step.coverage == "none"

    # Empty authority: a state this expander never created is INCOMPLETE
    # (UNKNOWN), never confused with a proven dead end.
    from slm_training.dsl.solver.adapters import completion_forest_state
    from slm_training.dsl.grammar.fastpath.compiler_draft import CompletionForest

    foreign = completion_forest_state(
        prefix_ids=(int(tok.bos_id),),
        forest=CompletionForest((), "none"),
        pack_id="openui",
        constraint_version="test-cv",
        bounds=bounds,
    )
    step = expander.successor(foreign, hole.hole_id, dead)
    assert step.status is ExpandStatus.INCOMPLETE
    assert step.detail == "unknown_state"


def test_successor_reuses_session_without_new_engine_allocations(
    tok: DSLNativeTokenizer, bounds: SolverBounds
) -> None:
    prefix = tuple([tok.bos_id, *tok.encode("root = Card([", add_special=False)])
    expander = _expander(tok, prefix, bounds)
    root = expander.root_state()
    allocations_after_root = expander._session.stats()["candidate_engine_allocations"]
    hole = root.holes[0]
    for value in hole.values:
        expander.successor(root, hole.hole_id, value)
    stats = expander._session.stats()
    # Warm successor queries fork interned engines; they never construct
    # fresh OpenUIIncrementalEngine instances beyond the root seed.
    assert stats["candidate_engine_allocations"] == allocations_after_root
    assert stats["transition_cache_misses"] > 0
    assert stats["unique_states"] >= 2


def test_identical_successor_expands_once(
    tok: DSLNativeTokenizer, bounds: SolverBounds
) -> None:
    prefix = tuple([tok.bos_id, *tok.encode("root = Card([", add_special=False)])
    expander = _expander(tok, prefix, bounds)
    root = expander.root_state()
    value = root.holes[0].values[0]
    with collect_decode_stats() as decode_stats:
        first = expander.successor(root, root.holes[0].hole_id, value)
        session_stats = expander._session.stats()
        second = expander.successor(root, root.holes[0].hole_id, value)

    assert second is first
    assert expander._session.stats() == session_stats
    assert decode_stats.solver_successor_cache_misses == 1
    assert decode_stats.solver_successor_cache_hits == 1
    assert decode_stats.solver_successor_expansions == 1
    # Hot cache identity is session-local integers. Canonical DomainValue
    # payloads remain only in the edge table/certificate-facing state.
    assert all(
        isinstance(state_id, int) and isinstance(edge_id, int)
        for state_id, edge_id in expander._successors
    )
