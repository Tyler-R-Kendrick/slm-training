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
  kernel matches the executable reference byte-for-byte, and every emitted
  terminal witness replays through a fresh engine to ``$END``.
"""

from __future__ import annotations

import gc
import weakref
from dataclasses import replace
from functools import lru_cache

import pytest

from slm_training.dsl.grammar.fastpath import semantic_state as ss
from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionForest,
    CompletionPath,
)
from slm_training.dsl.grammar.fastpath.completion_kernel import (
    CompletionSession,
    WitnessResult,
    WitnessStatus,
    _StateRecord,
)
from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
from slm_training.dsl.grammar.fastpath.token_map import token_surface_piece
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer


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


def _supported(tokens: tuple[int, ...]) -> WitnessResult:
    return WitnessResult(WitnessStatus.SUPPORTED, tokens)


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
    # outgoing forest for D is still built exactly once (shared domain memo);
    # a second identical top-level query reuses its completed exact verdict.
    # Nested subproblems remain query-local, preserving the per-candidate
    # expansion budget and first-path order.
    session = _GraphSession(packed_graph)
    assert session.terminal_witness(session.sid("A"), 5) == _supported(
        (1, 3, 7, 14, 13)
    )
    assert session.builds > 0
    builds_after_first_query = session.builds
    expanded_after_first_query = session.stats()["witness_states_expanded"]
    assert session.terminal_witness(session.sid("A"), 5) == _supported(
        (1, 3, 7, 14, 13)
    )
    assert session.builds == builds_after_first_query
    assert session.stats()["witness_states_expanded"] == expanded_after_first_query
    assert session.stats()["reachability_cache_hits"] > 0


def test_child_prefixes_are_persistent_until_authority_scan(
    tok: DSLNativeTokenizer,
) -> None:
    """Transition-only states do not copy the full token prefix eagerly."""
    prefix = tuple(tok.encode("root = Card()", add_special=False))
    session = CompletionSession(tok)
    seed = session.seed(prefix[:1])
    current = seed
    for token_id in prefix[1:]:
        child = session.advance(current, int(token_id))
        assert child is not None
        current = child

    assert session._states[seed].prefix_ids == prefix[:1]
    assert session._states[current].prefix_ids is None
    assert session.prefix_ids_of(current) == prefix
    assert session._states[current].prefix_ids == prefix


def test_witness_matches_bruteforce_everywhere(packed_graph: dict) -> None:
    complete_graph = dict(packed_graph)
    complete_graph["D"] = ("complete", packed_graph["D"][1][:2])
    complete_graph["J"] = ("complete", (((14,), "bind", "I"),))
    session = _GraphSession(complete_graph)
    for start in complete_graph:
        for room in range(0, 10):
            got = session.terminal_witness(session.sid(start), room)
            want = _bruteforce(complete_graph, start, room)
            if want is not None:
                assert got.status is WitnessStatus.SUPPORTED
                assert got.witness == want
            else:
                assert got.status is WitnessStatus.UNSUPPORTED


def test_positive_cost_cycle_terminates(packed_graph: dict) -> None:
    session = _GraphSession(packed_graph)
    # F <-> G costs one token per hop: room strictly decreases, so the cycle
    # cannot spin; the only exit is G -> I -> eos.
    assert session.terminal_witness(session.sid("F"), 4) == _supported((9, 11, 13))
    assert session.terminal_witness(session.sid("F"), 3) == _supported((9, 11, 13))
    assert (
        session.terminal_witness(session.sid("F"), 2).status
        is WitnessStatus.UNSUPPORTED
    )
    assert session.terminal_witness(session.sid("G"), 2) == _supported((11, 13))
    assert (
        session.terminal_witness(session.sid("G"), 1).status
        is WitnessStatus.UNSUPPORTED
    )


def test_budget_bounds_terminal_paths(packed_graph: dict) -> None:
    session = _GraphSession(packed_graph)
    # Chain H -> I -> eos costs exactly two tokens.
    assert session.terminal_witness(session.sid("H"), 2) == _supported((12, 13))
    assert (
        session.terminal_witness(session.sid("H"), 1).status
        is WitnessStatus.UNSUPPORTED
    )
    assert (
        session.terminal_witness(session.sid("H"), 0).status
        is WitnessStatus.UNSUPPORTED
    )


def test_incomplete_authority_stays_unknown_and_keeps_siblings(
    packed_graph: dict,
) -> None:
    session = _GraphSession(packed_graph)
    # J's partial coverage is an authority gap, never a witness — and it must
    # not become a false negative for D.
    assert session.terminal_witness(session.sid("J"), 8) == _supported((14, 13))
    assert session.terminal_witness(session.sid("D"), 5) == _supported(
        (5, 8, 12, 13)
    )


def test_node_budget_exhaustion_is_typed_unknown(packed_graph: dict) -> None:
    # A tiny expansion budget cannot prove the graph; it is never reported
    # as unsupported.
    session = _GraphSession(packed_graph, node_budget=2)
    assert (
        session.terminal_witness(session.sid("A"), 8).status
        is WitnessStatus.UNKNOWN
    )


def test_rejected_edge_is_skipped(packed_graph: dict) -> None:
    session = _GraphSession(packed_graph)
    assert (
        session.terminal_witness(session.sid("K"), 8).status
        is WitnessStatus.UNSUPPORTED
    )


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
    assert tokens == (1, 2)
    assert terminal == session.sid("n1")
    # A singleton edge may itself be wider than the remaining room.
    tokens, terminal, _ = session.forced_closure(session.sid("n0"), 1)
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
    again = session.advance_path(seed, first.paths[0].token_ids)
    assert again == child
    stats = session.stats()
    assert stats["transition_cache_hits"] >= 1
    assert stats["direct_terminal_feeds"] > 0
    # Warm kernel queries fork interned engines; they never construct fresh
    # OpenUIIncrementalEngine instances beyond the seed.
    assert stats["candidate_engine_allocations"] == allocations_after_seed
    assert stats["unique_states"] >= 2


def test_direct_outgoing_uses_control_state_and_semantic_views(
    tok: DSLNativeTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production session builds never clone trees or read Lark values."""
    from slm_training.dsl.grammar.fastpath import compiler_draft as cd

    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy value-stack/reference path used")

    monkeypatch.setattr(OpenUIIncrementalEngine, "_ip_with_cloned_values", _explode)
    monkeypatch.setattr(cd, "_build_openui_completion_forest_impl", _explode)
    for name in (
        "_active_call",
        "_schema_slot_components",
        "_schema_slot_type",
        "_schema_slot_name",
        "_schema_call_arity",
        "_schema_array_item_components",
        "_active_array_position",
        "_active_array_is_empty",
        "_active_list_occupancy",
        "_completed_call_string_values",
    ):
        monkeypatch.setattr(cd, name, _explode)

    ids = [tok.bos_id, *tok.encode("root = Card([b1,", add_special=False)]
    session = CompletionSession(tok, slot_contract=(":slot_0", ":slot_1"))
    seed = session.seed(ids)
    assert session._states[seed].engine._control_only
    forest = session.outgoing(seed)
    assert forest.coverage == "complete"
    assert len(forest.paths) == 14
    supported = 0
    for path in forest.paths:
        child = session.advance_path(seed, path.token_ids)
        assert child is not None
        proof = session.terminal_witness(child, 32 - len(path.token_ids))
        supported += int(proof.status is WitnessStatus.SUPPORTED)
    assert supported == 12
    assert session.stats()["candidate_engine_allocations"] == 1


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


def test_special_tokens_rejected_and_eos_is_terminal_only(
    tok: DSLNativeTokenizer,
) -> None:
    session = CompletionSession(tok)
    seed = session.seed(
        [tok.bos_id, *tok.encode('root = TextContent(":slot_0")', add_special=False)]
    )
    assert session.accepts_eos(seed)
    for token_id in (tok.pad_id, tok.mask_id, tok.bos_id, tok.eos_id, tok.unk_id):
        assert session.advance(seed, token_id) is None
    with pytest.raises(ValueError, match="BOS"):
        session.seed([*tok.encode("root =", add_special=False), tok.bos_id])
    for token_id in (tok.pad_id, tok.mask_id, tok.eos_id, tok.unk_id):
        with pytest.raises(ValueError, match="special token"):
            session.seed([token_id])


def test_tokenizer_authority_includes_special_token_ids(
    tok: DSLNativeTokenizer,
) -> None:
    from slm_training.dsl.pack import _openui_tokenizer_authority_fingerprint

    baseline = _openui_tokenizer_authority_fingerprint(tok)
    for attribute in ("pad_id", "bos_id", "eos_id", "mask_id", "unk_id"):
        token_id = int(getattr(tok, attribute))
        token = tok.id_to_token[token_id]
        changed = replace(tok, token_to_id={**tok.token_to_id, token: tok.vocab_size})
        assert _openui_tokenizer_authority_fingerprint(changed) != baseline


def test_macro_transition_updates_parser_and_semantics_identically() -> None:
    tok = DSLNativeTokenizer.build()
    tok.set_macro_expansions([["Card", "(", ")"]])
    prefix = list(tok.encode("root =", add_special=False))
    session = CompletionSession(tok)
    seed = session.seed(prefix)
    macro = tok.macro_id(0)
    child = session.advance(seed, macro)
    assert child is not None

    schema = session._schema
    arena = type(session._semantic_arena)()
    expected = ss.initial_state(tok, arena=arena)
    for token_id in [*prefix, *tok.expand_macros([macro])]:
        expected = ss.advance(
            expected, token_id, tok, schema=schema, arena=arena
        )
    assert session.semantic_of(child) == expected
    assert ss.emitted_component_count(session.semantic_of(child)) == 1
    # An orphaned reserved macro has no authority and is rejected.
    assert session.advance(seed, tok.macro_id(1)) is None


def test_session_release_drops_request_local_semantic_arena(
    tok: DSLNativeTokenizer,
) -> None:
    session = CompletionSession(tok)
    sid = session.seed(list(tok.encode("root = Card([b1,", add_special=False)))
    semantic = session.semantic_of(sid)
    semantic_ref = weakref.ref(semantic)
    arena_ref = weakref.ref(session._semantic_arena)
    del semantic, session
    gc.collect()
    assert arena_ref() is None
    assert semantic_ref() is None


# ---------------------------------------------------------------------------
# Differential parity: packed kernel vs the reference implementation
# ---------------------------------------------------------------------------

_CORPUS = (
    'root = TextContent(":slot_0")\n',
    'root = Card([])\n',
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


def _domain_pair(tok, prefix_ids, *, budget, slot_contract=(":slot_0", ":slot_1"), **kw):
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
    assert kernel.status == reference.status, label
    assert kernel.reason == reference.reason, label
    assert kernel.terminals == reference.terminals, label
    assert kernel.scope_fingerprint == reference.scope_fingerprint, label
    assert kernel.candidates == reference.candidates, label


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
            assert _replays_to_end(
                tok, prefix_ids, candidate.terminal_witness
            ), f"{label} budget={budget} witness={candidate.terminal_witness}"


def _sampled_prefix_ends(length: int, *, max_samples: int = 12) -> tuple[int, ...]:
    """Keep generated-program parity broad without making CI scale by token count."""
    candidates = list(range(0, length + 1, 2))
    if candidates[-1] != length:
        candidates.append(length)
    if len(candidates) <= max_samples:
        return tuple(candidates)
    last = len(candidates) - 1
    return tuple(candidates[index * last // (max_samples - 1)] for index in range(max_samples))


@pytest.mark.parametrize("program", _CORPUS, ids=lambda p: repr(p[-24:]))
def test_domain_parity_every_prefix(tok: DSLNativeTokenizer, program: str) -> None:
    ids = [int(t) for t in tok.encode(program, add_special=False)]
    for end in range(0, len(ids) + 1):
        _check_prefix_parity(
            tok, [tok.bos_id, *ids[:end]], f"{program!r}[:{end}]"
        )


def test_domain_card_open_item_preserves_twelve_reference_paths(
    tok: DSLNativeTokenizer,
) -> None:
    ids = [tok.bos_id, *tok.encode("root = Card([b1,", add_special=False)]
    kernel, reference = _domain_pair(tok, ids, budget=32)
    _assert_domain_equal(kernel, reference, "root = Card([b1,")
    assert len(reference.candidates) == 12
    assert len(kernel.candidates) == 12
    session = CompletionSession(tok, slot_contract=(":slot_0", ":slot_1"))
    seed = session.seed(ids)
    forest = session.outgoing(seed)
    assert forest.coverage == "complete"
    proofs = []
    for path in forest.paths:
        child = session.advance_path(seed, path.token_ids)
        assert child is not None
        proofs.append(
            session.terminal_witness(child, 32 - len(path.token_ids))
        )
    assert sum(proof.status is WitnessStatus.SUPPORTED for proof in proofs) == 12


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
        for end in _sampled_prefix_ends(len(ids)):
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
    session._graph["J"] = ("partial", ())
    session._outgoing.pop(session.sid("J"), None)
    # J has partial coverage: no witness provable, nothing disproven.
    assert (
        session.terminal_witness(session.sid("J"), 8).status
        is WitnessStatus.UNKNOWN
    )
    # Repair the authority (a fresh query rebuilds the forest): the same
    # (state, room) key is re-expanded and now finds the terminal route.
    session._graph["J"] = ("complete", (((14,), "bind", "I"),))
    session._outgoing.pop(session.sid("J"), None)
    assert session.terminal_witness(session.sid("J"), 8) == _supported((14, 13))


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
    assert session._forced == {}
    monkeypatch.setattr(kernel_mod, "check_decode_deadline", lambda: None)
    # A fresh query recomputes cleanly and agrees with the reference.
    assert session.terminal_witness(session.sid("A"), 8) == _supported(
        (1, 3, 5, 8, 12, 13)
    )


def test_deadline_is_checked_before_cached_closure(
    packed_graph: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slm_training.dsl.grammar.fastpath import completion_kernel as kernel_mod

    session = _GraphSession(packed_graph)
    session.forced_closure(session.sid("H"), 2)

    def _expired() -> None:
        raise TimeoutError("decode deadline")

    monkeypatch.setattr(kernel_mod, "check_decode_deadline", _expired)
    with pytest.raises(TimeoutError):
        session.forced_closure(session.sid("H"), 2)


def test_deadline_is_checked_before_cached_native_forest(
    tok: DSLNativeTokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slm_training.dsl.grammar.fastpath import completion_kernel as kernel_mod

    session = CompletionSession(tok)
    seed = session.seed((tok.bos_id,))
    session.outgoing(seed)

    def _expired() -> None:
        raise TimeoutError("decode deadline")

    monkeypatch.setattr(kernel_mod, "check_decode_deadline", _expired)
    with pytest.raises(TimeoutError):
        session.outgoing(seed)


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
    stack_sid = session.seed(
        list(tok.encode("root = Stack([b1], ", add_special=False))
    )
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
