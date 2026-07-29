"""Packed completion kernel: fixture proofs, closure exactness, V1 parity.

Two layers of evidence:

* Synthetic packed-graph fixtures prove the shared witness DP expands
  convergent suffix states once, matches a V1-style brute-force recursion on
  every start/room (including a positive-cost cycle and deterministic
  chains), respects in/out-of-budget terminals, and keeps an
  incomplete-authority branch UNKNOWN without erasing proven sibling
  witnesses.  Forced-closure compression is proven exact on the same
  fixtures.
* Differential parity: for every prefix of the witness corpus (including the
  hard ``root = Card([b1`` / ``root = Card([b1,`` cases — the latter yields
  exactly 12 complete paths) and fixed-seed generated programs, the packed
  kernel's ``_openui_completion_domain`` equals
  ``_openui_completion_domain_reference`` byte-for-byte, and every emitted
  terminal witness replays through a fresh engine to ``$END``.
"""

from __future__ import annotations

from functools import lru_cache

import pytest

from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionForest,
    CompletionPath,
)
from slm_training.dsl.grammar.fastpath.completion_kernel import (
    CompletionSession,
    WitnessVerdict,
    _StateRecord,
)
from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
from slm_training.dsl.grammar.fastpath.token_map import token_surface_piece
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer
from slm_training.models.grammar import make_grammar_state


@pytest.fixture(scope="module")
def tok() -> DSLNativeTokenizer:
    return DSLNativeTokenizer.build()


# ---------------------------------------------------------------------------
# Synthetic packed-graph fixtures (no tokenizer/engine in the loop)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    id_to_token: dict[int, str] = {}


class _GraphSession(CompletionSession):
    """CompletionSession with a synthetic graph behind ``outgoing``/``advance_path``.

    ``graph`` maps node name -> (coverage, ((token_ids, kind, child), ...)).
    ``child=None`` marks an edge the grammar rejects.  ``kind="eos"``
    terminates.  Forest builds are counted (``builds``) and memoized exactly
    like the real session.
    """

    def __init__(self, graph, *, node_budget: int = 16) -> None:
        super().__init__(_FakeTokenizer(), node_budget=node_budget)
        self._graph = graph
        self._node_ids: dict[str, int] = {}
        for node in graph:
            sid = len(self._states)
            self._states.append(
                _StateRecord(engine=None, semantic=None, prefix_ids=(node,))
            )
            self._counters["unique_states"] += 1
            self._node_ids[node] = sid
        self.builds = 0

    def sid(self, node: str) -> int:
        return self._node_ids[node]

    def outgoing(self, state_id: int, *, state=None) -> CompletionForest:
        cached = self._outgoing.get(int(state_id))
        if cached is not None:
            return cached
        self.builds += 1
        coverage, edges = self._graph[self._states[int(state_id)].prefix_ids[0]]
        forest = CompletionForest(
            tuple(CompletionPath(tuple(tokens), kind) for tokens, kind, _ in edges),
            coverage,
        )
        self._outgoing[int(state_id)] = forest
        return forest

    def advance_path(self, state_id: int, token_ids):
        node = self._states[int(state_id)].prefix_ids[0]
        for tokens, _kind, child in self._graph[node][1]:
            if tuple(tokens) == tuple(token_ids):
                return None if child is None else self._node_ids[child]
        return None


def _bruteforce(graph, start: str, room: int, node_budget: int = 16):
    """V1-style reference: fresh per-query budget + per-query lru(64)."""
    nodes_left = [node_budget]

    @lru_cache(maxsize=64)
    def _tail(node: str, rm: int):
        if rm <= 0 or nodes_left[0] <= 0:
            return None
        nodes_left[0] -= 1
        coverage, edges = graph[node]
        if coverage != "complete":
            return None
        for tokens, kind, child in edges:
            tokens = tuple(tokens)
            if not tokens or len(tokens) > rm or child is None and kind != "eos":
                continue
            if kind == "eos":
                return tokens
            suffix = _tail(child, rm - len(tokens))
            if suffix is not None:
                return tokens + suffix
        return None

    return _tail(start, room)


@pytest.fixture
def packed_graph() -> dict:
    # Convergent suffix D reached from both B and C; a positive-cost cycle
    # F <-> G; a deterministic chain H -> I -> eos; an incomplete-authority
    # branch J (partial coverage, must stay UNKNOWN); a rejected edge K.
    return {
        "A": ("complete", (((1,), "bind", "B"), ((2,), "bind", "C"))),
        "B": ("complete", (((3,), "bind", "D"),)),
        "C": ("complete", (((4,), "bind", "D"),)),
        "D": (
            "complete",
            (((5,), "bind", "E"), ((6,), "bind", "F"), ((7,), "bind", "J")),
        ),
        "E": ("complete", (((8,), "bind", "H"),)),
        "F": ("complete", (((9,), "bind", "G"),)),
        "G": ("complete", (((10,), "bind", "F"), ((11,), "bind", "I"))),
        "H": ("complete", (((12,), "bind", "I"),)),
        "I": ("complete", (((13,), "eos", None),)),
        "J": ("partial", (((14,), "bind", "I"),)),
        "K": ("complete", (((15,), "bind", None),)),
    }


def test_convergent_suffix_states_expand_once(packed_graph: dict) -> None:
    # D is reachable via both B and C; its subtree carries no witness, so the
    # DP evaluates (D, room) under several rooms across one query.  The
    # outgoing forest for D is still built exactly once (shared domain
    # memo), and a second query replays D's proven-UNSUPPORTED verdict from
    # the session memo instead of re-expanding it.
    session = _GraphSession(packed_graph)
    assert session.terminal_witness(session.sid("A"), 5) is None
    evaluated = {"A", "B", "C", "D", "E", "F", "G", "H", "J"}
    assert session.builds == len(evaluated)
    builds_after_first_query = session.builds
    hits_before = session.stats()["reachability_cache_hits"]
    assert session.terminal_witness(session.sid("A"), 5) is None
    assert session.builds == builds_after_first_query
    assert session.stats()["reachability_cache_hits"] > hits_before


def test_witness_matches_bruteforce_everywhere(packed_graph: dict) -> None:
    session = _GraphSession(packed_graph)
    for start in packed_graph:
        for room in range(0, 10):
            got = session.terminal_witness(session.sid(start), room)
            want = _bruteforce(packed_graph, start, room)
            assert got == want, f"{start=} {room=}: {got=} != {want=}"


def test_positive_cost_cycle_terminates(packed_graph: dict) -> None:
    session = _GraphSession(packed_graph)
    # F <-> G costs one token per hop: room strictly decreases, so the cycle
    # cannot spin; the only exit is G -> I -> eos.
    assert session.terminal_witness(session.sid("F"), 4) == (9, 11, 13)
    assert session.terminal_witness(session.sid("F"), 3) == (9, 11, 13)
    assert session.terminal_witness(session.sid("F"), 2) is None
    assert session.terminal_witness(session.sid("G"), 2) == (11, 13)
    assert session.terminal_witness(session.sid("G"), 1) is None


def test_positive_witness_cache_replays_node_budget(packed_graph: dict) -> None:
    session = _GraphSession(packed_graph, node_budget=8)
    assert session.terminal_witness(session.sid("F"), 4) == (9, 11, 13)
    hits = session.stats()["witness_cache_hits"]
    assert session.terminal_witness(session.sid("F"), 4) == (9, 11, 13)
    assert session.stats()["witness_cache_hits"] > hits
    # A cached proof never widens a smaller fresh node budget.
    session._node_budget = 1
    assert session.terminal_witness(session.sid("F"), 4) is None


def test_budget_bounds_terminal_paths(packed_graph: dict) -> None:
    session = _GraphSession(packed_graph)
    # Chain H -> I -> eos costs exactly two tokens.
    assert session.terminal_witness(session.sid("H"), 2) == (12, 13)
    assert session.terminal_witness(session.sid("H"), 1) is None
    assert session.terminal_witness(session.sid("H"), 0) is None


def test_incomplete_authority_stays_unknown_and_keeps_siblings(
    packed_graph: dict,
) -> None:
    session = _GraphSession(packed_graph)
    # J's partial coverage is an authority gap, never a witness — and it must
    # not erase D's proven witnesses on earlier paths.
    assert session.terminal_witness(session.sid("J"), 8) is None
    assert (
        session.terminal_witness_verdict(session.sid("J"), 8).verdict
        is WitnessVerdict.UNKNOWN
    )
    assert session.terminal_witness(session.sid("D"), 5) == (5, 8, 12, 13)
    assert (
        session.terminal_witness_verdict(session.sid("D"), 5).verdict
        is WitnessVerdict.SUPPORTED
    )


def test_node_budget_matches_reference(packed_graph: dict) -> None:
    # With a tiny budget both sides exhaust identically.
    session = _GraphSession(packed_graph, node_budget=2)
    for start in packed_graph:
        for room in range(0, 6):
            got = session.terminal_witness(session.sid(start), room)
            want = _bruteforce(packed_graph, start, room, node_budget=2)
            assert got == want, f"{start=} {room=}: {got=} != {want=}"


def test_budget_unknown_replays_exact_fresh_charge(packed_graph: dict) -> None:
    session = _GraphSession(packed_graph, node_budget=2)
    assert session.terminal_witness(session.sid("A"), 8) is None
    expanded = session.stats()["witness_states_expanded"]
    assert session._budget_unknown
    assert session.terminal_witness(session.sid("A"), 8) is None
    assert session.stats()["witness_states_expanded"] == expanded
    assert session.stats()["budget_unknown_cache_hits"] == 1
    # The node budget is proof authority and therefore part of the cache key.
    session._node_budget = 16
    assert session.terminal_witness(session.sid("A"), 8) == _bruteforce(
        packed_graph, "A", 8
    )


def test_rejected_edge_is_skipped(packed_graph: dict) -> None:
    session = _GraphSession(packed_graph)
    assert session.terminal_witness(session.sid("K"), 8) is None


def test_zero_cost_graph_edge_is_rejected() -> None:
    session = _GraphSession(
        {
            "n0": ("complete", (((), "bind", "n1"),)),
            "n1": ("complete", (((1,), "eos", None),)),
        }
    )
    with pytest.raises(ValueError, match="must consume"):
        session.terminal_witness(session.sid("n0"), 8)


def test_forced_closure_exactness() -> None:
    graph = {
        "n0": ("complete", (((1, 2), "bind", "n1"),)),
        "n1": ("complete", (((3,), "bind", "n2"),)),
        "n2": ("complete", (((4,), "bind", "n3"), ((5,), "bind", "n3"))),
        "n3": ("complete", (((6,), "eos", None),)),
        "lit": ("complete", (((7,), "lit", "n3"),)),
        "inc": ("partial", (((8,), "bind", "n3"),)),
    }
    session = _GraphSession(graph)
    # Follows the complete-coverage singleton chain and stops at the branch.
    tokens, terminal, coverage = session.forced_closure(session.sid("n0"), 8)
    assert tokens == (1, 2, 3)
    assert terminal == session.sid("n2")
    assert coverage == "complete"
    # Budget stop.
    tokens, terminal, _ = session.forced_closure(session.sid("n0"), 2)
    assert tokens == ()
    assert terminal == session.sid("n0")
    # EOS-only path stops immediately.
    tokens, _, _ = session.forced_closure(session.sid("n3"), 8)
    assert tokens == ()
    # Incomplete authority stops immediately.
    tokens, _, coverage = session.forced_closure(session.sid("inc"), 8)
    assert tokens == ()
    assert coverage == "partial"
    # Memoized second call.
    hits_before = session.stats()["forced_closure_hits"]
    assert session.forced_closure(session.sid("n0"), 8)[0] == (1, 2, 3)
    assert session.stats()["forced_closure_hits"] == hits_before + 1


def test_forced_closure_stops_at_literal_boundary() -> None:
    tokenizer = _FakeTokenizer()
    tokenizer.id_to_token = {7: "LIT_STR", 8: "B:61", 9: "LIT_END"}
    session = _GraphSession(
        {
            "n0": ("complete", (((7, 8, 9), "lit", "n1"),)),
            "n1": ("complete", (((1,), "eos", None),)),
        }
    )
    session._tokenizer = tokenizer
    tokens, terminal, _ = session.forced_closure(session.sid("n0"), 8)
    assert tokens == ()
    assert terminal == session.sid("n0")


# ---------------------------------------------------------------------------
# Real-session mechanics: interning, transitions, counters
# ---------------------------------------------------------------------------


def test_session_counters_and_warm_allocation(tok: DSLNativeTokenizer) -> None:
    session = CompletionSession(tok, slot_contract=(":slot_0",))
    seed = session.seed(list(tok.encode("root =", add_special=False)))
    allocations_after_seed = session.stats()["candidate_engine_allocations"]
    first = session.outgoing(seed)
    assert session.stats()["scope_reference_scans_avoided"] > 0
    assert session.outgoing(seed) is first
    assert session.stats()["domain_cache_hits"] == 1
    child = session.advance_path(seed, first.paths[0].token_ids)
    assert child is not None
    assert session._rows[seed].targets[first.paths[0].token_ids] == child
    again = session.advance_path(seed, first.paths[0].token_ids)
    assert again == child
    sequential = seed
    for token_id in first.paths[0].token_ids:
        sequential = session.advance(sequential, token_id)
        assert sequential is not None
    assert session.prefix_ids_of(sequential) == (
        *session.prefix_ids_of(seed),
        *first.paths[0].token_ids,
    )
    stats = session.stats()
    assert stats["transition_cache_hits"] >= 1
    assert stats["transition_rows_built"] == 1
    assert stats["transition_rows_hits"] == 1
    assert stats["semantic_masks_applied"] == 1
    assert stats["edge_replays"] == 0
    assert stats["ast_bridge_calls"] == 0
    assert stats["direct_terminal_feeds"] > 0
    # Warm kernel queries fork interned engines; they never construct fresh
    # OpenUIIncrementalEngine instances beyond the seed.
    assert stats["candidate_engine_allocations"] == allocations_after_seed
    assert stats["unique_states"] >= 2


def test_packed_row_never_calls_ast_bridge(
    tok: DSLNativeTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grammar acceptance plus the semantic cursor owns EOS completeness."""
    from slm_training.dsl.grammar.fastpath import compiler_draft as cd

    def _explode(_prefix: str) -> bool:
        raise AssertionError("packed row reparsed the generated AST")

    monkeypatch.setattr(cd, "_generated_ast_is_complete", _explode)
    session = CompletionSession(tok, slot_contract=(":slot_0",))
    ids = list(tok.encode('root = TextContent(":slot_0")', add_special=False))
    forest = session.outgoing(session.seed(ids))
    assert any(path.kind == "eos" for path in forest.paths)


def test_restore_decode_state_uses_interned_engine_without_replay(
    tok: DSLNativeTokenizer,
) -> None:
    prefix = [tok.bos_id, *tok.encode("root =", add_special=False)]
    session = CompletionSession(tok)
    state_id = session.seed(prefix)
    state = make_grammar_state()
    pool = {("ctx", tuple(prefix)): (session, state_id)}

    session.restore_decode_state(state_id, state, context="ctx", pool=pool)

    assert state.prefix_ids == prefix
    assert state.completion_session is session
    assert state.completion_state_id == state_id
    assert state.engine_ids_len == len(prefix)
    assert state.engine is not session._states[state_id].engine
    assert (
        state.engine.parser_state_key()
        == session._states[state_id].engine.parser_state_key()
    )


def test_cache_isolation_across_authority_contexts(tok: DSLNativeTokenizer) -> None:
    ids = list(tok.encode('root = TextContent(":slot_0")', add_special=False))
    contexts = [
        {},
        {"min_content": 2},
        {"max_path_tokens": 4},
        {"explain": True},
        {"slot_contract": (":slot_0",)},
        {"runtime_symbols": ("sym",)},
        {"enforce_schema_component_types": True},
    ]
    sessions = [CompletionSession(tok, **ctx) for ctx in contexts]
    seeds = [session.seed(ids) for session in sessions]
    forests = [session.outgoing(sid) for session, sid in zip(sessions, seeds)]
    # min_content=2 withholds EOS (only one component emitted), the rest
    # admit it: contexts never share state across sessions.
    assert "eos" not in {p.kind for p in forests[1].paths}
    assert any(p.kind == "eos" for p in forests[0].paths)
    for forest in forests[2:]:
        assert any(p.kind == "eos" for p in forest.paths)
    # Stats are request-local.
    for session in sessions[1:]:
        assert session.stats()["domain_cache_hits"] == 0
    sessions[0].outgoing(seeds[0])
    assert sessions[0].stats()["domain_cache_hits"] == 1
    assert sessions[1].stats()["domain_cache_hits"] == 0


# ---------------------------------------------------------------------------
# Differential parity: packed kernel vs the reference implementation
# ---------------------------------------------------------------------------

_CORPUS = (
    'root = TextContent(":slot_0")\n',
    "root = Card([])\n",
    'root = Card([b1, b2])\nb1 = TextContent(":slot_0")\nb2 = TextContent(":slot_1")\n',
    'root = Card([b1])\nb1 = Card([b2])\nb2 = TextContent(":slot_0")\n',
    'root = Callout("info", ":slot_0", ":slot_1", true)\n',
    'root = Card([TextContent(":slot_0"), b1])\nb1 = TextContent(":slot_1")\n',
    "root = Card([b1",
    "root = Card([b1,",
    "root =",
    "root",
    "",
)


def _domain_pair(
    tok, prefix_ids, *, budget, slot_contract=(":slot_0", ":slot_1"), **kw
):
    from slm_training.dsl.grammar_capabilities import CompletionDomainRequestV1
    from slm_training.dsl.pack import (
        _openui_completion_domain,
        _openui_completion_domain_reference,
    )

    request = CompletionDomainRequestV1(
        prefix_ids=tuple(prefix_ids),
        tokenizer=tok,
        slot_contract=tuple(slot_contract),
        remaining_tokens=budget,
        **kw,
    )
    kernel = _openui_completion_domain(request)
    reference = _openui_completion_domain_reference(request)
    return kernel, reference


def _assert_domain_equal(kernel, reference, label: str) -> None:
    assert (kernel.status, kernel.reason, kernel.terminals) == (
        reference.status,
        reference.reason,
        reference.terminals,
    ), label
    got = [(c.token_ids, c.kind, c.terminal_witness) for c in kernel.candidates]
    want = [(c.token_ids, c.kind, c.terminal_witness) for c in reference.candidates]
    assert got == want, label


def _replays_to_end(tok, prefix_ids, witness) -> bool:
    """Replay prefix + witness token-by-token through a fresh engine."""
    engine = OpenUIIncrementalEngine()
    for token_id in (*prefix_ids, *witness):
        outcome = engine.feed_token_id(tok, int(token_id))
        if outcome is None:
            outcome = engine.advance_checked(token_surface_piece(tok, int(token_id)))
        if outcome is False:
            return False
    return "$END" in engine.next_terminals()


def _check_prefix_parity(tok, prefix_ids, label: str, budgets=(8, 32)) -> None:
    for budget in budgets:
        kernel, reference = _domain_pair(tok, prefix_ids, budget=budget)
        _assert_domain_equal(kernel, reference, f"{label} budget={budget}")
        for candidate in kernel.candidates:
            assert candidate.terminal_witness[-1] == int(tok.eos_id), label
            assert _replays_to_end(tok, prefix_ids, candidate.terminal_witness), (
                f"{label} budget={budget} witness={candidate.terminal_witness}"
            )


@pytest.mark.parametrize("program", _CORPUS, ids=lambda p: repr(p[-24:]))
def test_domain_parity_every_prefix(tok: DSLNativeTokenizer, program: str) -> None:
    ids = [int(t) for t in tok.encode(program, add_special=False)]
    for end in range(0, len(ids) + 1):
        _check_prefix_parity(tok, [tok.bos_id, *ids[:end]], f"{program!r}[:{end}]")


def test_domain_parity_card_open_item_yields_twelve_paths(
    tok: DSLNativeTokenizer,
) -> None:
    ids = [tok.bos_id, *tok.encode("root = Card([b1,", add_special=False)]
    kernel, reference = _domain_pair(tok, ids, budget=32)
    _assert_domain_equal(kernel, reference, "root = Card([b1,")
    assert len(reference.candidates) == 12
    assert len(kernel.candidates) == 12


def test_domain_parity_seeded_generated_programs(tok: DSLNativeTokenizer) -> None:
    from slm_training.data.progspec.generate import GeneratorConfig, ProgramGenerator

    programs = ProgramGenerator(GeneratorConfig(), seed=7).generate(2).programs
    checked = 0
    for program in programs:
        try:
            ids = [
                int(t) for t in tok.encode(program.canonical_openui, add_special=False)
            ]
        except ValueError:
            continue  # template content outside the symbolic surface
        checked += 1
        for end in range(0, len(ids) + 1, 2):
            _check_prefix_parity(
                tok,
                [tok.bos_id, *ids[:end]],
                f"{program.id}[:{end}]",
                budgets=(32,),
            )
    if not checked:
        pytest.skip("seeded programs are not encodable on the symbolic surface")


def test_domain_parity_with_grammar_decode_state(tok: DSLNativeTokenizer) -> None:
    from slm_training.dsl.grammar_capabilities import CompletionDomainRequestV1
    from slm_training.dsl.pack import (
        _openui_completion_domain,
        _openui_completion_domain_reference,
    )
    from slm_training.models.grammar import make_grammar_state

    ids = tuple(
        [tok.bos_id] + [int(t) for t in tok.encode("root = Card([", add_special=False)]
    )
    kernel = _openui_completion_domain(
        CompletionDomainRequestV1(
            prefix_ids=ids,
            tokenizer=tok,
            slot_contract=(":slot_0",),
            remaining_tokens=32,
            state=make_grammar_state(),
        )
    )
    reference = _openui_completion_domain_reference(
        CompletionDomainRequestV1(
            prefix_ids=ids,
            tokenizer=tok,
            slot_contract=(":slot_0",),
            remaining_tokens=32,
            state=make_grammar_state(),
        )
    )
    _assert_domain_equal(kernel, reference, "decode-state")


def test_decode_state_reuses_one_session_across_positions(
    tok: DSLNativeTokenizer,
) -> None:
    from slm_training.dsl.grammar_capabilities import CompletionDomainRequestV1
    from slm_training.dsl.pack import _openui_completion_domain
    from slm_training.models.decode_stats import collect_decode_stats
    from slm_training.models.grammar import make_grammar_state

    state = make_grammar_state()
    prefix = [tok.bos_id]
    with collect_decode_stats() as stats:
        first = _openui_completion_domain(
            CompletionDomainRequestV1(
                prefix_ids=tuple(prefix),
                tokenizer=tok,
                slot_contract=(":slot_0",),
                remaining_tokens=32,
                state=state,
            )
        )
        assert first.status == "complete"
        session = state.completion_session
        assert session is not None
        initial = session.stats()
        token_id = int(first.candidates[0].token_ids[0])
        prefix.append(token_id)
        state.advance_token(tok, token_id)
        second = _openui_completion_domain(
            CompletionDomainRequestV1(
                prefix_ids=tuple(prefix),
                tokenizer=tok,
                slot_contract=(":slot_0",),
                remaining_tokens=31,
                state=state,
            )
        )

    assert second.status == "complete"
    assert state.completion_session is session
    assert session.stats()["session_starts"] == 1
    assert (
        session.stats()["candidate_engine_allocations"]
        == initial["candidate_engine_allocations"]
    )
    assert session.stats()["full_prefix_lex_bytes"] == initial["full_prefix_lex_bytes"]
    assert stats.completion_session_starts == 1
    assert stats.completion_transition_cache_hits > 0


def test_warm_direct_query_replays_no_general_work(
    tok: DSLNativeTokenizer,
) -> None:
    from slm_training.dsl.grammar_capabilities import CompletionDomainRequestV1
    from slm_training.dsl.pack import _openui_completion_domain

    ids = tuple(
        [tok.bos_id, *tok.encode("root = Card([b1,", add_special=False)]
    )
    request = CompletionDomainRequestV1(
        prefix_ids=ids,
        tokenizer=tok,
        slot_contract=(":slot_0", ":slot_1"),
        remaining_tokens=32,
        state=make_grammar_state(),
    )
    first = _openui_completion_domain(request)
    assert len(first.candidates) == 12
    session = request.state.completion_session
    # First graph-only replay records complete-authority budget UNKNOWN plans.
    session._results.clear()
    assert _openui_completion_domain(request) == first
    session._results.clear()
    before = session.stats()
    assert _openui_completion_domain(request) == first
    after = session.stats()
    assert after["budget_unknown_cache_hits"] - before["budget_unknown_cache_hits"] == 2
    for counter in ("general_forest_builds", "value_tree_clones", "edge_replays"):
        assert after[counter] == before[counter]


def test_domain_parity_context_variants(tok: DSLNativeTokenizer) -> None:
    ids = [tok.bos_id, *tok.encode('root = Callout("info", ', add_special=False)]
    for kw in ({"min_content": 2}, {"max_path_tokens": 4}, {"explain": True}):
        kernel, reference = _domain_pair(tok, ids, budget=32, **kw)
        _assert_domain_equal(kernel, reference, f"ctx {kw}")


# ---------------------------------------------------------------------------
# Formal-review blocking findings (T-WIT-3, T-WIT-4, T-MERGE-2, T-CTX-1)
# ---------------------------------------------------------------------------


def test_twit3_incomplete_authority_never_cached_as_unsupported(
    packed_graph: dict,
) -> None:
    """UNKNOWN is not UNSUPPORTED: a coverage-incomplete verdict stays
    query-local, and repairing the authority re-expands the node."""
    session = _GraphSession(packed_graph)
    # J has partial coverage: no witness provable, nothing disproven.
    assert session.terminal_witness(session.sid("J"), 8) is None
    # The session memo must NOT mark (J, room) proven-UNSUPPORTED.
    assert session._reach == {}
    # Repair the authority (a fresh query rebuilds the forest): the same
    # (state, room) key is re-expanded and now finds the terminal route.
    session._graph["J"] = ("complete", (((14,), "bind", "I"),))
    session._outgoing.pop(session.sid("J"), None)
    assert session.terminal_witness(session.sid("J"), 8) == (14, 13)


def test_twit4_timeout_mid_exploration_propagates_and_poisons_nothing(
    packed_graph: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deadline raised mid-exploration propagates out of terminal_witness;
    no verdict is written for the interrupted keys; a later query recomputes."""
    from slm_training.dsl.grammar.fastpath import completion_kernel as kernel_mod

    session = _GraphSession(packed_graph)
    calls = {"n": 0}

    def _boom() -> None:
        calls["n"] += 1
        if calls["n"] >= 3:
            raise TimeoutError("decode deadline")

    monkeypatch.setattr(kernel_mod, "check_decode_deadline", _boom)
    with pytest.raises(TimeoutError):
        session.terminal_witness(session.sid("A"), 8)
    assert session._reach == {}
    assert session._witness == {}
    assert session._budget_unknown == {}
    assert session._forced == {}
    monkeypatch.setattr(kernel_mod, "check_decode_deadline", lambda: None)
    # A fresh query recomputes cleanly and agrees with the reference.
    assert session.terminal_witness(session.sid("A"), 8) == _bruteforce(
        packed_graph, "A", 8
    )


def test_twit4_timeout_inside_string_role_try_propagates(
    tok: DSLNativeTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The string-role restriction's broad ``except Exception`` must not
    convert a decode deadline into a silently unrestricted candidate set."""
    import slm_training.dsl.language_contract as language_contract
    from slm_training.dsl.grammar.fastpath import compiler_draft as cd

    class _ExplodingAtoms:
        def __iter__(self):
            raise TimeoutError("decode deadline")

    # During a build only the lazy import inside the guarded try re-reads this
    # attribute (every other user bound it at module import), so the deadline
    # fires exactly inside the try block.
    # The stateless forest cache is process-global: never serve a cached
    # forest for this injection (the guarded block must actually run).
    cd._STATELESS_FOREST_CACHE.clear()
    monkeypatch.setattr(language_contract, "STRUCTURAL_ID_ATOMS", _ExplodingAtoms())
    ids = [tok.bos_id, *tok.encode('root = Callout("info", ', add_special=False)]
    with pytest.raises(TimeoutError):
        cd._build_openui_completion_forest(tok, ids, slot_contract=[":slot_0"])


def test_twit4_timeout_inside_inventory_try_propagates(
    tok: DSLNativeTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The binding-inventory block's broad ``except Exception`` must not
    convert a decode deadline into silent ``inventory_complete = False``
    (which would then feed the UNKNOWN-vs-UNSUPPORTED path)."""
    from slm_training.dsl.grammar.fastpath import compiler_draft as cd
    from slm_training.models import dsl_tokenizer as dt

    armed = {"on": False}
    real_token_kind = dt.TokenKind

    class _ExplodingTokenKindMeta(type):
        def __call__(cls, *args: object, **kwargs: object):
            return real_token_kind(*args, **kwargs)

        def __getattr__(cls, name: str):
            if armed["on"]:
                raise TimeoutError("decode deadline")
            return getattr(real_token_kind, name)

    # A real (meta)class so ``isinstance(x, TokenKind)`` keeps working for the
    # tokenizer's own internals; only attribute access (``TokenKind.BIND`` …)
    # is armed, and it is reached lazily inside the guarded try.
    _ExplodingTokenKind = _ExplodingTokenKindMeta("_ExplodingTokenKind", (), {})

    real_apply_literal_frame = cd.apply_literal_frame

    def _arming_apply_literal_frame(*args: object, **kwargs: object):
        # Called immediately before the guarded inventory try; arm only after
        # the real call (it lazy-imports TokenKind itself).
        result = real_apply_literal_frame(*args, **kwargs)
        armed["on"] = True
        return result

    monkeypatch.setattr(dt, "TokenKind", _ExplodingTokenKind)
    monkeypatch.setattr(cd, "apply_literal_frame", _arming_apply_literal_frame)
    cd._STATELESS_FOREST_CACHE.clear()  # the guarded block must actually run
    ids = [tok.bos_id, *tok.encode("root = Card([b1", add_special=False)]
    with pytest.raises(TimeoutError):
        cd._build_openui_completion_forest(tok, ids, slot_contract=[":slot_0"])


def test_twit4_timeout_in_semantic_kind_of_propagates(
    tok: DSLNativeTokenizer,
) -> None:
    """semantic_state._kind_of must not swallow the decode deadline."""
    from slm_training.dsl.grammar.fastpath import semantic_state as ss

    class _ExplodingTokenizer:
        def __init__(self, inner: DSLNativeTokenizer) -> None:
            self._inner = inner

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

        def kind_of(self, token_id: int):
            raise TimeoutError("decode deadline")

    proxy = _ExplodingTokenizer(tok)
    state = ss.initial_state(proxy)
    with pytest.raises(TimeoutError):
        ss.advance(state, int(tok.token_to_id["="]), proxy)


def test_twit4_timeout_in_advance_token_fallback_propagates(
    tok: DSLNativeTokenizer,
) -> None:
    """GrammarDecodeState.advance_token's nested fallback swallows must let
    the decode deadline through at every layer."""
    from slm_training.models.grammar import GrammarDecodeState

    class _AdvanceBoom:
        _ip = None
        _prefix = ""
        full_sync_fallbacks = 0

        def feed_token_id(self, tokenizer, token_id):
            return None  # force the text fallback

        def advance(self, chunk: str) -> bool:
            raise TimeoutError("decode deadline")

    state = GrammarDecodeState(engine=_AdvanceBoom())
    with pytest.raises(TimeoutError):
        state.advance_token(tok, int(tok.token_to_id["="]))

    class _ResyncBoom(_AdvanceBoom):
        def advance(self, chunk: str) -> bool:
            raise ValueError("broken junction")

        def set_prefix(self, prefix: str) -> bool:
            raise TimeoutError("decode deadline")

    state = GrammarDecodeState(engine=_ResyncBoom())
    with pytest.raises(TimeoutError):
        state.advance_token(tok, int(tok.token_to_id["="]))


def test_tmerge2_no_false_merge_stack_vs_card(tok: DSLNativeTokenizer) -> None:
    """Equal parser state stacks (``root = Stack(`` vs ``root = Card(``) must
    intern differently — the semantic state's CallFrame.component separates
    them — and their schema-filtered candidate sets must differ."""
    session = CompletionSession(tok, slot_contract=(":slot_0",))
    stack_sid = session.seed(list(tok.encode("root = Stack([b1], ", add_special=False)))
    card_sid = session.seed(list(tok.encode("root = Card([b1], ", add_special=False)))
    assert stack_sid != card_sid
    stack_forest = session.outgoing(stack_sid)
    card_forest = session.outgoing(card_sid)
    stack_tokens = {(tuple(p.token_ids), p.kind) for p in stack_forest.paths}
    card_tokens = {(tuple(p.token_ids), p.kind) for p in card_forest.paths}
    assert stack_tokens != card_tokens


def test_tctx1_session_context_separation(tok: DSLNativeTokenizer) -> None:
    """Sessions differing only in min_content / explain must expose different
    forests (EOS admission / evidence payload) and share no cache."""
    ids = list(tok.encode('root = TextContent(":slot_0")', add_special=False))
    base = CompletionSession(tok)
    explain = CompletionSession(tok, explain=True)
    floor = CompletionSession(tok, min_content=2)
    f_base = base.outgoing(base.seed(ids))
    f_explain = explain.outgoing(explain.seed(ids))
    f_floor = floor.outgoing(floor.seed(ids))
    # min_content=2 withholds EOS: only one component has been emitted.
    assert any(p.kind == "eos" for p in f_base.paths)
    assert "eos" not in {p.kind for p in f_floor.paths}
    # explain=True carries an evidence payload the plain build lacks.
    assert f_explain.evidence and not f_base.evidence
    # Request-local caches: neither session served the other's build.
    for session in (base, explain, floor):
        assert session.stats()["domain_cache_hits"] == 0
