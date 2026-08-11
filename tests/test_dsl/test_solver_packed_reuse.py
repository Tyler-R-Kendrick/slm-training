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

from slm_training.data.contract import RuntimeSymbol
from slm_training.dsl.grammar.fastpath.compiler_draft import build_completion_forest
from slm_training.dsl.solver.adapters import completion_forest_state
from slm_training.dsl.solver.closure import EnumerativeSupportProvider
from slm_training.dsl.solver.openui_support import (
    OpenUIForestExpander,
    OpenUIWellFormedVerifier,
)
from slm_training.dsl.solver.state import (
    DomainValue,
    FiniteDomainState,
    HoleDomain,
    HoleId,
    SolverBounds,
)
from slm_training.dsl.solver.support import (
    EnumerativeSupportOracle,
    ExpandStatus,
    ExpandStep,
    SupportQuery,
    SupportVerdict,
    VerifyOutcome,
    VerifyStatus,
    replay_support_certificate,
)
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer
from slm_training.models.grammar import make_grammar_state


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


def _reference_state(tok, prefix, forest, bounds, *, authority_digest=None):
    return completion_forest_state(
        prefix_ids=prefix,
        forest=forest,
        pack_id="openui",
        constraint_version="test-cv",
        bounds=bounds,
        authority_digest=authority_digest,
    )


def _request_bound_expander(
    tok,
    prefix,
    bounds,
    *,
    slot_contract=(),
    min_content=0,
    runtime_symbols=(),
    remaining_tokens=32,
):
    decode_state = make_grammar_state()
    forest = build_completion_forest(
        tok,
        list(prefix),
        state=decode_state,
        slot_contract=list(slot_contract),
        max_path_tokens=8,
        min_content=min_content,
        runtime_symbols=runtime_symbols,
        remaining_tokens=remaining_tokens,
    )
    expander = OpenUIForestExpander(
        tok,
        prefix,
        pack_id="openui",
        constraint_version="test-cv",
        bounds=bounds,
        max_path_tokens=8,
        slot_contract=slot_contract,
        min_content=min_content,
        runtime_symbols=runtime_symbols,
        remaining_tokens=remaining_tokens,
        root_forest=forest,
        completion_session=decode_state.completion_session,
        completion_state_id=decode_state.completion_state_id,
    )
    return forest, decode_state, expander


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
        authority_digest=expander.authority_digest,
    )
    assert _state_payload(expander.root_state()) == _state_payload(reference)


def test_request_bound_root_reuses_exact_session_and_forest(
    tok: DSLNativeTokenizer, bounds: SolverBounds
) -> None:
    prefix = tuple([tok.bos_id, *tok.encode("root = Card([", add_special=False)])
    runtime_symbols = (RuntimeSymbol(":slot_0", "external_entity"),)
    forest, decode_state, expander = _request_bound_expander(
        tok,
        prefix,
        bounds,
        slot_contract=(":slot_0",),
        min_content=1,
        runtime_symbols=runtime_symbols,
        remaining_tokens=32,
    )

    assert expander._session is decode_state.completion_session
    assert _state_payload(expander.root_state()) == _state_payload(
        _reference_state(
            tok,
            prefix,
            forest,
            expander.bounds,
            authority_digest=expander.authority_digest,
        )
    )


@pytest.mark.parametrize(
    ("text", "forest_kwargs", "expander_kwargs"),
    (
        (
            "root = TextContent(",
            {"slot_contract": (":slot_0",), "remaining_tokens": 32},
            {"slot_contract": (), "remaining_tokens": 32},
        ),
        (
            "root = Card([])",
            {"min_content": 0, "remaining_tokens": 32},
            {"min_content": 2, "remaining_tokens": 32},
        ),
        (
            "root = Card([",
            {"remaining_tokens": 32},
            {"remaining_tokens": 8},
        ),
    ),
    ids=("slot-contract", "min-content", "remaining-token-horizon"),
)
def test_request_authority_mismatch_rejects_root_forest(
    tok: DSLNativeTokenizer,
    bounds: SolverBounds,
    text: str,
    forest_kwargs: dict,
    expander_kwargs: dict,
) -> None:
    prefix = tuple([tok.bos_id, *tok.encode(text, add_special=False)])
    forest = build_completion_forest(
        tok, list(prefix), max_path_tokens=8, **forest_kwargs
    )

    with pytest.raises(ValueError, match="root forest differs"):
        OpenUIForestExpander(
            tok,
            prefix,
            pack_id="openui",
            constraint_version="test-cv",
            bounds=bounds,
            max_path_tokens=8,
            root_forest=forest,
            **expander_kwargs,
        )


def test_request_horizon_is_independent_of_solver_work_bound(
    tok: DSLNativeTokenizer, bounds: SolverBounds
) -> None:
    prefix = tuple([tok.bos_id, *tok.encode("root = Card([", add_special=False)])
    forest = build_completion_forest(
        tok, list(prefix), max_path_tokens=8, remaining_tokens=32
    )

    expander = OpenUIForestExpander(
        tok,
        prefix,
        pack_id="openui",
        constraint_version="test-cv",
        bounds=bounds,
        max_path_tokens=8,
        remaining_tokens=32,
        root_forest=forest,
    )

    assert expander.bounds.max_tokens == 4000
    assert expander._remaining_tokens == 32


def test_horizon_changes_state_backend_and_certificate_identity(
    tok: DSLNativeTokenizer, bounds: SolverBounds
) -> None:
    class _AcceptingVerifier:
        profile = "test/accepting"

        def verify(self, _program: str) -> VerifyOutcome:
            return VerifyOutcome(VerifyStatus.ACCEPT)

    prefix = tuple([tok.bos_id, *tok.encode("root = Card([])", add_special=False)])

    def _for_horizon(horizon: int):
        forest = build_completion_forest(
            tok, list(prefix), max_path_tokens=8, remaining_tokens=horizon
        )
        expander = OpenUIForestExpander(
            tok,
            prefix,
            pack_id="openui",
            constraint_version="test-cv",
            bounds=bounds,
            max_path_tokens=8,
            remaining_tokens=horizon,
            root_forest=forest,
        )
        root = expander.root_state()
        query = SupportQuery(
            root.fingerprint, root.holes[0].hole_id, root.holes[0].values[0]
        )
        provider = EnumerativeSupportProvider(expander, _AcceptingVerifier())
        return expander, root, provider, provider.check(root, query)

    expander_32, root_32, provider_32, result_32 = _for_horizon(32)
    expander_33, root_33, provider_33, result_33 = _for_horizon(33)

    assert expander_32.authority_digest != expander_33.authority_digest
    assert expander_32.problem_id != expander_33.problem_id
    assert root_32.fingerprint != root_33.fingerprint
    assert provider_32.backend_version != provider_33.backend_version
    assert result_32.certificate.digest != result_33.certificate.digest
    assert not provider_33.replay(result_32.certificate, state=root_33).ok


def test_horizon_32_does_not_become_payload_json_token_budget() -> None:
    """A >32-byte action still reaches a terminal under an independent budget."""

    value = DomainValue.create(
        "completion_path",
        {"kind": "terminal", "token_ids": [7], "padding": "x" * 32},
    )
    bounds = SolverBounds(256, 8, 4, 8, 4)
    state = FiniteDomainState(
        problem_id="horizon-32-fixture",
        pack_id="openui",
        constraint_version="test-cv",
        bounds=bounds,
        holes=(
            HoleDomain(
                HoleId("fixture", (0,), "action"),
                (value,),
                (("coverage", "complete"), ("remaining_tokens", 32)),
            ),
        ),
    )

    class _TerminalExpander:
        problem_id = state.problem_id
        pack_id = state.pack_id
        constraint_version = state.constraint_version
        remaining_tokens = 32

        @property
        def bounds(self):
            return bounds

        def successor(self, _state, _hole_id, _value):
            return ExpandStep(
                ExpandStatus.TERMINAL,
                program="root = Card([])",
                coverage="complete",
            )

    class _AcceptingVerifier:
        profile = "test/accepting"

        def verify(self, _program: str) -> VerifyOutcome:
            return VerifyOutcome(VerifyStatus.ACCEPT)

    query = SupportQuery(state.fingerprint, state.holes[0].hole_id, value)
    result = EnumerativeSupportOracle(
        _TerminalExpander(), _AcceptingVerifier()
    ).check(state, query)

    assert len(value.payload_json) > 32
    assert result.counters.tokens > 32
    assert result.verdict is SupportVerdict.SUPPORTED
    assert result.certificate.stop_reason is None


def test_successor_forwards_runtime_authority_and_reduces_horizon(
    tok: DSLNativeTokenizer,
    bounds: SolverBounds,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import slm_training.dsl.solver.openui_support as openui_support

    prefix = tuple([tok.bos_id, *tok.encode("root = Card([", add_special=False)])
    runtime_symbols = (RuntimeSymbol(":slot_0", "external_entity"),)
    _forest, _decode_state, expander = _request_bound_expander(
        tok,
        prefix,
        bounds,
        slot_contract=(":slot_0",),
        min_content=1,
        runtime_symbols=runtime_symbols,
        remaining_tokens=32,
    )
    root = expander.root_state()
    value = next(
        item
        for item in root.holes[0].values
        if item.payload.get("kind") != "eos"
    )
    token_count = len(value.payload["token_ids"])
    calls = []
    real_build = openui_support.build_completion_forest

    def _recording_build(*args, **kwargs):
        calls.append(kwargs)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(openui_support, "build_completion_forest", _recording_build)
    expander.successor(root, root.holes[0].hole_id, value)

    assert len(calls) == 1
    assert tuple(calls[0]["slot_contract"]) == (":slot_0",)
    assert calls[0]["min_content"] == 1
    assert calls[0]["runtime_symbols"] == runtime_symbols
    assert calls[0]["remaining_tokens"] == 32 - token_count


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
        reference = _reference_state(
            tok,
            new_prefix,
            reference_forest,
            bounds,
            authority_digest=expander.authority_digest,
        )
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
    assert result.verdict is SupportVerdict.UNKNOWN
    assert result.certificate.stop_reason == "budget:max_depth"
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


def test_advertised_edge_replay_failure_is_unknown_not_dead(
    tok: DSLNativeTokenizer, bounds: SolverBounds, monkeypatch
) -> None:
    prefix = tuple([tok.bos_id, *tok.encode("root = Card([", add_special=False)])
    expander = _expander(tok, prefix, bounds)
    root = expander.root_state()
    hole = root.holes[0]
    value = next(
        candidate
        for candidate in hole.values
        if str(candidate.payload.get("kind", "")) != "eos"
    )
    monkeypatch.setattr(expander._session, "advance_path", lambda *_args: None)

    step = expander.successor(root, hole.hole_id, value)

    assert step.status is ExpandStatus.INCOMPLETE
    assert step.detail == "advertised_edge_transition_failed"
