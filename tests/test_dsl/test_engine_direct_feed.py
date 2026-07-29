"""Differential tests: direct DSL-native token-id feeds == canonical text route.

``OpenUIIncrementalEngine.feed_token_id`` maps verified token ids straight to
Lark terminals (zero lexing); anything ambiguous (byte-channel identifiers,
dirty tails, str-frame escapes) falls back to ``advance_checked`` on the
token's surface piece. For every prefix of every witness program, the
direct-fed engine must expose exactly the same grammar state as a fresh
``set_prefix(decode_prefix(ids))`` engine.
"""

from __future__ import annotations

from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
from slm_training.dsl.grammar.fastpath.token_map import (
    decode_prefix,
    token_surface_piece,
)
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer, TokenKind

# Byte-free sources: every token maps to a verified terminal or frame, so
# direct feeding performs zero full-prefix lexes after the initial sync.
_BYTE_FREE_SOURCES = (
    "root = Card([b1,",
    'root = Card([b1, b2])\nb1 = TextContent(":slot_0")\nb2 = Button(":slot_1")',
    "$count = 0\nroot = TextContent($count)",
    "root = Slider(0.5)",
    "flag = true\nnone = null\nother = false",
    # Forward binder reference: b2 is used before its definition.
    'root = Card([b1])\nb1 = TextContent(b2)\nb2 = Label(":slot_0")',
)

# Sources whose byte-channel identifiers (object keys, member names) force
# the counted text-route fallback — correctness over coverage.
_BYTE_FALLBACK_SOURCES = (
    "cfg = {count: 12, ratio: 0.5, flag: true, missing: null}",
    "root = Text(q.text)",
)


def _tok() -> DSLNativeTokenizer:
    return DSLNativeTokenizer.build()


def _ids(tok: DSLNativeTokenizer, src: str, *, trailing_nl: bool = False) -> list[int]:
    ids = list(tok.encode(src, add_special=False))
    if trailing_nl:
        ids.append(tok.token_to_id["NL"])
    return ids


def _assert_same_state(
    text_engine: OpenUIIncrementalEngine, direct_engine: OpenUIIncrementalEngine
) -> None:
    assert direct_engine.next_terminals() == text_engine.next_terminals()
    assert direct_engine.is_deterministic_next() == text_engine.is_deterministic_next()
    assert direct_engine.terminals_are_exact() == text_engine.terminals_are_exact()


def _differential_run(tok: DSLNativeTokenizer, ids: list[int]) -> OpenUIIncrementalEngine:
    text_engine = OpenUIIncrementalEngine()
    direct_engine = OpenUIIncrementalEngine()
    direct_engine.reset()  # initial sync: empty prefix, zero lexing
    frame_accepts: frozenset[str] | None = None
    for k in range(len(ids) + 1):
        assert text_engine.set_prefix(decode_prefix(tok, ids[:k]))
        if k:
            result = direct_engine.feed_token_id(tok, ids[k - 1])
            assert result is True, (k, tok.id_to_token.get(ids[k - 1]), result)
        if direct_engine._frame is not None:
            # Mid-frame: the direct engine defers the literal terminal to the
            # frame close while the text baseline lexes the partial body as a
            # complete lexeme — the states diverge by design. The frame must
            # leave the parser untouched (accepts frozen at the opener).
            if frame_accepts is None:
                frame_accepts = direct_engine.next_terminals()
            assert direct_engine.next_terminals() == frame_accepts
            continue
        frame_accepts = None
        _assert_same_state(text_engine, direct_engine)
    return direct_engine


def _probe_sample(tok: DSLNativeTokenizer) -> list[int]:
    names = [
        "=", "(", ")", "[", "]", "{", "}", ",", ".", ":", "?",
        "+", "-", "==", "&&", "NL",
        "Card", "TextContent", "@Run", "true", "false", "null",
        "<BIND_0>", "<BIND_1>", "<STATE_0>", "<SYM_0>", "<SYM_1>",
        "LIT_NUM", "LIT_END", "B:78", "B:31", "B:22", "STR::slot_0",
        "<pad>", "<bos>", "<eos>", "<mask>", "<unk>", "<MACRO_0>",
    ]
    return [tok.token_to_id[n] for n in names if n in tok.token_to_id]


def test_differential_byte_free_witnesses() -> None:
    tok = _tok()
    for src in _BYTE_FREE_SOURCES:
        _differential_run(tok, _ids(tok, src))
    # Trailing newline is part of the prefix contract.
    _differential_run(tok, _ids(tok, _BYTE_FREE_SOURCES[1], trailing_nl=True))


def test_differential_byte_fallback_witnesses() -> None:
    tok = _tok()
    for src in _BYTE_FALLBACK_SOURCES:
        _differential_run(tok, _ids(tok, src))


def test_zero_full_prefix_lexes_when_append_only() -> None:
    tok = _tok()
    for src in _BYTE_FREE_SOURCES:
        engine = OpenUIIncrementalEngine()
        engine.reset()
        for tid in _ids(tok, src):
            assert engine.feed_token_id(tok, tid) is True
        stats = engine.stats
        assert stats["full_prefix_lex_bytes"] == 0, src
        assert stats["full_sync_fallbacks"] == 0, src
        assert stats["full_syncs"] == 0, src
        assert stats["direct_terminal_feeds"] > 0, src


def test_fallbacks_are_counted() -> None:
    tok = _tok()
    engine = OpenUIIncrementalEngine()
    engine.reset()
    for tid in _ids(tok, _BYTE_FALLBACK_SOURCES[0]):
        assert engine.feed_token_id(tok, tid) is True
    stats = engine.stats
    assert stats["full_sync_fallbacks"] > 0
    assert stats["full_prefix_lex_bytes"] > 0


def test_single_token_probes_match_text_route() -> None:
    tok = _tok()
    frame_ids = {
        tok.token_to_id[n] for n in ("LIT_NUM", "LIT_STR", "LIT_END") if n in tok.token_to_id
    }
    for src in ("root = Card([b1,", _BYTE_FREE_SOURCES[1]):
        ids = _ids(tok, src)
        text_engine = OpenUIIncrementalEngine()
        direct_engine = OpenUIIncrementalEngine()
        direct_engine.reset()
        for k in range(len(ids) + 1):
            assert text_engine.set_prefix(decode_prefix(tok, ids[:k]))
            if k:
                assert direct_engine.feed_token_id(tok, ids[k - 1]) is True
            if direct_engine._frame is not None:
                continue  # mid-frame states diverge by design (see above)
            for tid in _probe_sample(tok):
                probe_direct = direct_engine.copy()
                accepted_direct = probe_direct.feed_token_id(tok, tid)
                if accepted_direct is None:
                    continue  # unsupported kind: caller uses the text path
                if tid in frame_ids:
                    continue  # frame protocol is asserted separately
                if tok.kind_of(tid) is TokenKind.BYTE:
                    # Bytes take the canonical text route inside feed_token_id;
                    # compare against that same route.
                    probe_text = text_engine.copy()
                    accepted_text = probe_text.advance_checked(
                        token_surface_piece(tok, tid)
                    )
                    if accepted_direct:
                        _assert_same_state(probe_text, probe_direct)
                else:
                    # Canonical baseline: a fresh sync of the decoded stream.
                    probe_text = OpenUIIncrementalEngine()
                    accepted_text = probe_text.set_prefix(
                        decode_prefix(tok, ids[:k] + [tid])
                    )
                    if accepted_direct and accepted_text:
                        _assert_same_state(probe_text, probe_direct)
                assert accepted_direct == accepted_text, (k, tid)


def test_literal_frame_feeds_terminal_only_at_close() -> None:
    tok = _tok()
    ids = _ids(tok, "root = Slider(0.5)")
    opener = tok.token_to_id["LIT_NUM"]
    closer = tok.token_to_id["LIT_END"]
    engine = OpenUIIncrementalEngine()
    engine.reset()
    for tid in ids:
        before = engine.next_terminals()
        assert engine.feed_token_id(tok, tid) is True
        if tid == opener or engine._frame is not None and tid != closer:
            # Opener and frame bytes advance frame state without feeding.
            if engine._frame is not None:
                assert engine.next_terminals() == before
    # After the closer, exactly one NUMBER terminal was fed for the frame.
    text_engine = OpenUIIncrementalEngine()
    assert text_engine.set_prefix(decode_prefix(tok, ids))
    _assert_same_state(text_engine, engine)
    num_feeds = sum(1 for t, _ in engine._fed_tokens if t == "NUMBER")
    assert num_feeds == 1


def test_malformed_frames_fail_closed() -> None:
    tok = _tok()
    lit_num = tok.token_to_id["LIT_NUM"]
    lit_end = tok.token_to_id["LIT_END"]
    dot = tok.token_to_id["B:2e"]
    one = tok.token_to_id["B:31"]
    card = tok.token_to_id["Card"]

    # Closer without an opener.
    engine = OpenUIIncrementalEngine()
    engine.reset()
    key = engine.parser_state_key()
    assert engine.feed_token_id(tok, lit_end) is False
    assert engine.parser_state_key() == key

    # Empty number frame.
    assert engine.feed_token_id(tok, lit_num) is True
    assert engine.feed_token_id(tok, lit_end) is False
    assert engine._frame == ("num", "")

    # Invalid number byte.
    engine = OpenUIIncrementalEngine()
    engine.reset()
    assert engine.feed_token_id(tok, lit_num) is True
    assert engine.feed_token_id(tok, one) is True
    assert engine.feed_token_id(tok, dot) is True
    assert engine.feed_token_id(tok, dot) is False  # "1.." is not a NUMBER
    assert engine._frame == ("num", "1.")

    # Non-byte token inside a frame.
    engine = OpenUIIncrementalEngine()
    engine.reset()
    assert engine.feed_token_id(tok, lit_num) is True
    assert engine.feed_token_id(tok, card) is False
    assert engine._frame == ("num", "")


def test_special_and_unknown_kinds() -> None:
    tok = _tok()
    engine = OpenUIIncrementalEngine()
    engine.reset()
    key = engine.parser_state_key()
    for tid in (tok.bos_id, tok.eos_id, tok.pad_id, tok.mask_id):
        assert engine.feed_token_id(tok, tid) is True  # consumed, never fed
    assert engine.parser_state_key() == key
    assert engine.feed_token_id(tok, tok.unk_id) is None
    # Compositional / unverifiable tokenizers stay on the text path.
    assert engine.feed_token_id(object(), 0) is None
    # Reserved abstract-plan rows are unsupported.
    tok_abs = DSLNativeTokenizer.build(abstract_plan_slots=4)
    begin = tok_abs.abstract_begin_id
    assert begin is not None
    assert engine.feed_token_id(tok_abs, begin) is None


def test_macro_expansion_feeds_iteratively() -> None:
    tok = _tok()
    tok.set_macro_expansions([["Card", "(", ")"]])
    ids = _ids(tok, "root = Card()")
    assert any(tok.is_macro_id(tid) for tid in ids)
    _differential_run(tok, ids)
    # Orphaned macro rows (no table entry) have no verified expansion —
    # the caller uses the text path.
    engine = OpenUIIncrementalEngine()
    engine.reset()
    key = engine.parser_state_key()
    assert engine.feed_token_id(tok, tok.macro_id(1)) is None
    assert engine.parser_state_key() == key


def test_parser_state_key_identity() -> None:
    tok = _tok()
    ids = _ids(tok, _BYTE_FREE_SOURCES[1], trailing_nl=True)
    # Same prefix reached via text route, direct feed, and fork: equal keys.
    text_engine = OpenUIIncrementalEngine()
    assert text_engine.set_prefix(decode_prefix(tok, ids))
    direct_engine = OpenUIIncrementalEngine()
    direct_engine.reset()
    for tid in ids:
        assert direct_engine.feed_token_id(tok, tid) is True
    fork = direct_engine.copy()
    assert text_engine.parser_state_key() == direct_engine.parser_state_key()
    assert fork.parser_state_key() == direct_engine.parser_state_key()
    # Fork counters are tracked.
    assert direct_engine.stats["parser_forks"] == 1
    # Advancing changes the key; both routes change it identically.
    nxt = tok.token_to_id["<BIND_0>"]
    before = direct_engine.parser_state_key()
    if direct_engine.feed_token_id(tok, nxt) is True:
        assert direct_engine.parser_state_key() != before
        assert text_engine.advance_checked(token_surface_piece(tok, nxt))
        assert text_engine.parser_state_key() == direct_engine.parser_state_key()


# --- Phase 1: production append-path wiring ---------------------------------


def _witness_sources() -> list[str]:
    from slm_training.dsl.pack import _openui_witness_candidates

    return [item.source for item in _openui_witness_candidates()]


def test_differential_all_witness_candidate_prefixes() -> None:
    """accepts(direct_feed(w)) == accepts(lex_and_feed(decode_prefix(w))) for
    every prefix of every pack witness candidate."""
    tok = _tok()
    for src in _witness_sources():
        _differential_run(tok, _ids(tok, src))


def test_differential_fixed_seed_generated_sample() -> None:
    """Same differential proof over a deterministic generated sample."""
    import random

    rng = random.Random(20260729)
    components = ["Card", "Stack", "TextContent", "Button", "Label", "Separator"]
    directions = ['"column"', '"row"']
    binders = ["b1", "b2", "b3"]
    sources: list[str] = []
    for _ in range(12):
        n_defs = rng.randint(1, 3)
        # Statement order is shuffled so forward references occur.
        defs = [
            f'{name} = {rng.choice(components)}(":slot_{i}")'
            for i, name in enumerate(binders[:n_defs])
        ]
        rng.shuffle(defs)
        refs = ", ".join(binders[:n_defs])
        root = f"root = Stack([{refs}], {rng.choice(directions)})"
        prefix = "$flag = true\n" if rng.random() < 0.5 else ""
        sources.append(prefix + "\n".join([root, *defs]))
    tok = _tok()
    for src in sources:
        _differential_run(tok, _ids(tok, src))


def test_append_only_advance_never_full_prefix_lexes() -> None:
    """Append-only direct feeds must never call the full-prefix lexer."""
    tok = _tok()
    for src in _BYTE_FREE_SOURCES:
        engine = OpenUIIncrementalEngine()
        engine.reset()
        calls: list[str] = []
        orig_report = engine._lex_tokens_report

        def _spy(prefix: str, _orig=orig_report, _calls=calls):
            _calls.append(prefix)
            return _orig(prefix)

        engine._lex_tokens_report = _spy  # type: ignore[method-assign]
        for tid in _ids(tok, src):
            assert engine.feed_token_id(tok, tid) is True
        assert calls == [], (src, calls)
        assert engine.stats["full_prefix_lex_bytes"] == 0


def test_forks_do_not_rebuild_or_alex() -> None:
    """``copy()`` is a structural fork: no lexing, no rebuild, independent state."""
    tok = _tok()
    ids = _ids(tok, _BYTE_FREE_SOURCES[1], trailing_nl=True)
    engine = OpenUIIncrementalEngine()
    engine.reset()
    for tid in ids:
        assert engine.feed_token_id(tok, tid) is True
    lexed_before = engine.full_prefix_lex_bytes
    fork = engine.copy()
    assert engine.full_prefix_lex_bytes == lexed_before
    assert fork.full_prefix_lex_bytes == 0
    assert engine.stats["parser_forks"] == 1
    assert fork.parser_state_key() == engine.parser_state_key()
    assert fork.next_terminals() == engine.next_terminals()
    # Feeding the fork leaves the source engine untouched.
    nxt = tok.token_to_id["<BIND_0>"]
    key_before = engine.parser_state_key()
    result = fork.feed_token_id(tok, nxt)
    assert engine.parser_state_key() == key_before
    if result is True:
        assert fork.parser_state_key() != key_before


def test_decode_state_append_path_uses_direct_feed() -> None:
    """GrammarDecodeState.advance_token commits via direct feed: zero full
    syncs after ``reset`` and grammar state identical to the text baseline."""
    from slm_training.models.grammar import (
        dfa_admits_token,
        exact_forced_token_id,
        force_emit_token_id,
        make_grammar_state,
    )

    tok = _tok()
    # A trailing NL is legal only after a complete statement; use plain
    # sources for the sweep and exercise the trailing-NL contract on the
    # complete program only (mirrors _differential_run).
    sources = [_ids(tok, src) for src in _BYTE_FREE_SOURCES]
    sources.append(_ids(tok, _BYTE_FREE_SOURCES[1], trailing_nl=True))
    for ids in sources:
        state = make_grammar_state()
        engine = state.engine
        assert engine is not None
        baseline = OpenUIIncrementalEngine()
        for k in range(len(ids) + 1):
            prefix = ids[:k]
            assert baseline.set_prefix(decode_prefix(tok, prefix))
            if engine._frame is None:
                assert engine.next_terminals() == baseline.next_terminals(), (
                    ids,
                    k,
                )
            # The production query paths must not trigger a full re-sync of a
            # direct-fed engine.
            syncs_before = engine.stats["full_syncs"]
            force_emit_token_id(tok, prefix, state=state)
            exact_forced_token_id(tok, prefix, state=state, runtime_symbols=())
            for cand in prefix[-3:] or [tok.token_to_id["NL"]]:
                dfa_admits_token(tok, prefix, int(cand), state=state)
            assert engine.stats["full_syncs"] == syncs_before, (ids, k)
            if k < len(ids):
                state.advance_token(tok, ids[k])
                assert state.engine_ids_len == k + 1
        assert engine.stats["full_syncs"] == 0
        assert engine.stats["direct_terminal_feeds"] > 0


def test_decode_state_fallbacks_are_counted_and_correct() -> None:
    """Byte-channel identifiers route through the counted text fallback while
    the decode state stays grammar-equivalent to the text baseline."""
    from slm_training.models.grammar import make_grammar_state

    tok = _tok()
    ids = _ids(tok, _BYTE_FALLBACK_SOURCES[0])
    state = make_grammar_state()
    engine = state.engine
    baseline = OpenUIIncrementalEngine()
    for k, tid in enumerate(ids):
        state.advance_token(tok, tid)
        assert state.engine_ids_len == k + 1
        assert baseline.set_prefix(decode_prefix(tok, ids[: k + 1]))
        if engine._frame is None:
            assert engine.next_terminals() == baseline.next_terminals()
    stats = engine.stats
    assert stats["full_sync_fallbacks"] > 0
    assert stats["full_prefix_lex_bytes"] > 0


def test_decode_state_compositional_tokenizer_text_route() -> None:
    """Legacy compositional tokenizers keep the text/lexer route unchanged."""
    from slm_training.models.grammar import make_grammar_state
    from slm_training.models.tokenizer import OpenUITokenizer

    src = 'root = TextContent(":slot_0")'
    tok = OpenUITokenizer.build([src])
    ids = list(tok.encode(src, add_special=False))
    state = make_grammar_state()
    engine = state.engine
    assert engine is not None
    for k, tid in enumerate(ids):
        state.advance_token(tok, tid)
        # No verified map: every token takes the text route, engine stays
        # textually synced, and no direct feed is recorded.
        assert state.engine_ids_len == k + 1
    assert engine.stats["direct_terminal_feeds"] == 0
    assert engine._prefix == state.prefix_text
    baseline = OpenUIIncrementalEngine()
    assert baseline.set_prefix(state.prefix_text)
    assert engine.next_terminals() == baseline.next_terminals()


def test_decode_state_replacement_clears_direct_sync_marker() -> None:
    """A non-append ``sync_ids`` replacement must not trust the stale marker."""
    from slm_training.models.grammar import make_grammar_state

    tok = _tok()
    ids = _ids(tok, _BYTE_FREE_SOURCES[1])
    state = make_grammar_state()
    for tid in ids[:5]:
        state.advance_token(tok, tid)
    assert state.engine_ids_len == 5
    # Replace (not extend) the prefix with different ids of the same length:
    # the marker is cleared, so the next query takes the canonical full sync.
    state.sync_ids(tok, ids[5:10])
    assert state.engine_ids_len is None
    assert not state.engine_in_sync(ids[5:10], state.prefix_text)
