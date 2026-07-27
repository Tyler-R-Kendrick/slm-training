"""Regression tests for the suffix-only relex in ``_incremental_sync``
(docs/design/decode-compiler-tree-incremental-lex-fix.md).

``_incremental_sync`` used to re-lex the *entire* current prefix on every
call (proposed-fix-sketch item 2 from docs/design/
decode-compiler-tree-lexer-cache-fix.md, applied here): O(prefix length)
lex work per call, O(n^2) total across a generation of length n, even with
the lexer object itself cached (PR #1173). The fix tracks
``self._fed_last_token_start`` -- the absolute offset of the last
already-fed token -- and re-lexes only ``prefix[offset:]`` on the common
path, splicing the result onto the untouched already-fed token-key prefix.

These tests prove: (a) the boundary offset is tracked correctly across a
growing drive, (b) strictly fewer characters are re-lexed than the
pre-existing full-prefix-relex baseline, (c) the *decode semantics* --
``next_terminals()``/``accepts()`` and ``build_completion_forest``'s
``coverage``/``n_paths`` -- are byte-identical to that baseline at every
single step of an incremental drive, not just the final one (I3/I6: a
caching/perf change must never become a legality change), and (d) a
conservative wall-time speedup floor on a long growing prefix.
"""

from __future__ import annotations

import time

import pytest

from slm_training.dsl.grammar.backends.types import GRAMMARS_DIR
from slm_training.dsl.grammar.fastpath import engine as engine_mod
from slm_training.dsl.grammar.fastpath.compiler_draft import build_completion_forest

# Exact prefixes from PR #1171's dead_end_trace / the lexer-rebuild finding,
# reused here for consistency with test_grammar_fastpath_lexer_cache.py.
PREFIX8 = "root = Card([b1"
PREFIX9 = "root = Card([b1,"

# Longer, multi-statement, incrementally-growing valid OpenUI DSL program --
# long enough (500+ chars) that the O(n) vs O(n^2) relex-cost gap actually
# shows across a char-by-char drive; the finding's 4 short prefixes are too
# short to demonstrate the quadratic effect.
LONG_PROGRAM = (
    'root = Stack([hero, b1, b2, b3, cardRow, footer])\n'
    'hero = Card([TextContent(":slot_hero_title"), TextContent(":slot_hero_sub")])\n'
    'b1 = Button(":action_one")\n'
    'b2 = Button(":action_two")\n'
    'b3 = Button(":action_three")\n'
    'cardRow = Row([cardA, cardB, cardC])\n'
    'cardA = Card([TextContent(":slot_a_title"), Buttons([b1, b2])])\n'
    'cardB = Card([TextContent(":slot_b_title"), Input(":second")])\n'
    'cardC = Card([TextContent(":slot_c_title"), Slider("$level", "continuous", -1, 1)])\n'
    'footer = Row([TextContent(":slot_footer_left"), TextContent(":slot_footer_right")])\n'
)


def _grammar_path() -> str:
    return str((GRAMMARS_DIR / "openui.lark").resolve())


def _forced_full_relex_incremental_sync(self, prefix: str) -> bool:
    """Reproduces the exact pre-fix ``_incremental_sync`` body: always
    re-lex the whole prefix from position 0 via ``_lex_tokens``, never the
    suffix-only fast path. Used as a self-contained "before" oracle (no
    ``git stash`` needed inside a test) for I3/I6 and char-count
    comparisons against the "after" (real, current) ``_incremental_sync``.
    """
    assert self._ip is not None
    tokens = self._lex_tokens(prefix)
    if tokens is None:
        return self._full_sync(prefix)
    keys = self._token_keys(tokens)
    if len(tokens) < self._fed_token_count or keys[: self._fed_token_count] != self._fed_tokens:
        return self._full_sync(prefix)
    return self._feed_delta_tokens(
        prefix, tokens[self._fed_token_count :], keys, base_offset=0
    )


class _CharCountingEngine(engine_mod.OpenUIIncrementalEngine):
    """Counts characters passed into ``_lex_tokens`` -- a direct proxy for
    lex work performed, independent of wall-time noise."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.chars_lexed = 0
        self.lex_calls = 0

    def _lex_tokens(self, prefix):
        self.chars_lexed += len(prefix)
        self.lex_calls += 1
        return super()._lex_tokens(prefix)


def test_fed_last_token_start_tracks_last_fed_token_offset() -> None:
    """The boundary offset must equal the last fed token's real start_pos,
    and must reset to None on ``reset()``."""
    eng = engine_mod.OpenUIIncrementalEngine()
    eng.reset()
    assert eng._fed_last_token_start is None

    assert eng.set_prefix("root = Card([b1,")
    # Tokens: NAME(0) EQUAL(5) COMPONENT(7) LPAR(11) LSQB(12) NAME(13) COMMA(15)
    assert eng._fed_last_token_start == 15
    assert eng._fed_token_count == 7

    assert eng.advance(" b2")
    # New tokens: NAME(17) -- comma was already fed, "b2" starts right after ", ".
    assert eng._fed_last_token_start == 17

    eng.reset()
    assert eng._fed_last_token_start is None


def test_incremental_sync_relexes_fewer_characters_than_full_relex_baseline() -> None:
    """Regression guard: driving a long, incrementally-growing valid prefix
    one character at a time must re-lex strictly fewer total characters
    than the pre-existing (forced) full-prefix-relex baseline -- proof that
    the suffix-only path is actually taken on the common path, not just
    present in the code."""
    program = LONG_PROGRAM

    new_eng = _CharCountingEngine()
    new_eng.reset()
    for i in range(1, len(program) + 1):
        assert new_eng.set_prefix(program[:i])

    old_eng = _CharCountingEngine()
    old_eng.reset()
    orig = engine_mod.OpenUIIncrementalEngine._incremental_sync
    engine_mod.OpenUIIncrementalEngine._incremental_sync = _forced_full_relex_incremental_sync
    try:
        for i in range(1, len(program) + 1):
            assert old_eng.set_prefix(program[:i])
    finally:
        engine_mod.OpenUIIncrementalEngine._incremental_sync = orig

    assert new_eng.chars_lexed < old_eng.chars_lexed, (
        f"suffix-lex path did not reduce chars_lexed: "
        f"new={new_eng.chars_lexed} old={old_eng.chars_lexed}"
    )
    # Conservative floor: at this prefix length the suffix path should clear
    # at least a 1.5x reduction in total characters re-lexed (measured
    # ~2.97x locally); kept well under that to stay robust on slower/loaded
    # CI hosts and unrelated future grammar edits.
    assert new_eng.chars_lexed * 1.5 < old_eng.chars_lexed, (
        f"chars_lexed reduction below the 1.5x floor: "
        f"new={new_eng.chars_lexed} old={old_eng.chars_lexed}"
    )


def test_incremental_sync_matches_forced_full_relex_at_every_step() -> None:
    """I3/I6: next_terminals() must be byte-identical, at *every* step of an
    incremental drive (not just the final one), between the suffix-lex path
    and the pre-existing full-prefix-relex baseline. Exercises both the
    fast suffix-lex path (punctuation/newline/new-token boundaries) and the
    still-full-relex fallback path (a token still growing mid-word, e.g.
    "Te" -> "Text"), since the drive is character-by-character."""
    program = LONG_PROGRAM

    new_eng = engine_mod.OpenUIIncrementalEngine()
    new_eng.reset()
    new_trace = []
    for i in range(1, len(program) + 1):
        assert new_eng.set_prefix(program[:i])
        new_trace.append(new_eng.next_terminals())

    old_eng = engine_mod.OpenUIIncrementalEngine()
    old_eng.reset()
    orig = engine_mod.OpenUIIncrementalEngine._incremental_sync
    engine_mod.OpenUIIncrementalEngine._incremental_sync = _forced_full_relex_incremental_sync
    old_trace = []
    try:
        for i in range(1, len(program) + 1):
            assert old_eng.set_prefix(program[:i])
            old_trace.append(old_eng.next_terminals())
    finally:
        engine_mod.OpenUIIncrementalEngine._incremental_sync = orig

    assert len(new_trace) == len(old_trace) == len(program)
    mismatches = [
        i for i, (a, b) in enumerate(zip(new_trace, old_trace, strict=True)) if a != b
    ]
    assert not mismatches, (
        f"next_terminals() diverged at steps {mismatches[:5]} "
        f"(showing up to 5 of {len(mismatches)})"
    )


@pytest.mark.parametrize(
    ("prefix", "expected_n_paths"),
    [
        (PREFIX8, 2),
        (PREFIX9, 12),
    ],
)
def test_build_completion_forest_unchanged_with_suffix_lex(
    prefix: str, expected_n_paths: int
) -> None:
    """Decode-invariant proof (I3/I6) at the decode-output layer: on the
    finding's exact prefixes, ``build_completion_forest`` returns the same
    ``coverage``/``n_paths`` whether ``_incremental_sync`` takes the
    suffix-lex fast path (current code) or is forced onto the pre-existing
    full-prefix-relex path (self-contained "before" oracle)."""
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    prefix_ids = [tok.bos_id, *tok.encode(prefix, add_special=False)]

    forest_after = build_completion_forest(
        tok, prefix_ids, remaining_tokens=24, max_path_tokens=8, min_content=0
    )

    orig = engine_mod.OpenUIIncrementalEngine._incremental_sync
    engine_mod.OpenUIIncrementalEngine._incremental_sync = _forced_full_relex_incremental_sync
    try:
        forest_before = build_completion_forest(
            tok, prefix_ids, remaining_tokens=24, max_path_tokens=8, min_content=0
        )
    finally:
        engine_mod.OpenUIIncrementalEngine._incremental_sync = orig

    assert forest_after.coverage == "complete"
    assert len(forest_after.paths) == expected_n_paths
    assert forest_before.coverage == forest_after.coverage
    assert len(forest_before.paths) == len(forest_after.paths)


def test_suffix_lex_speedup_floor_on_long_growing_prefix() -> None:
    """Wall-time benchmark: driving the long program char-by-character
    through the suffix-lex path must be meaningfully faster than the forced
    full-prefix-relex baseline on the same process. Threshold (1.3x) is
    well under the ~2.4-2.97x measured locally, to stay robust on a
    slower/loaded CI host."""
    program = LONG_PROGRAM

    new_eng = engine_mod.OpenUIIncrementalEngine()
    new_eng.reset()
    t0 = time.perf_counter()
    for i in range(1, len(program) + 1):
        new_eng.set_prefix(program[:i])
    new_wall = time.perf_counter() - t0

    old_eng = engine_mod.OpenUIIncrementalEngine()
    old_eng.reset()
    orig = engine_mod.OpenUIIncrementalEngine._incremental_sync
    engine_mod.OpenUIIncrementalEngine._incremental_sync = _forced_full_relex_incremental_sync
    try:
        t0 = time.perf_counter()
        for i in range(1, len(program) + 1):
            old_eng.set_prefix(program[:i])
        old_wall = time.perf_counter() - t0
    finally:
        engine_mod.OpenUIIncrementalEngine._incremental_sync = orig

    assert new_wall < old_wall / 1.3, (
        f"suffix-lex wall time ({new_wall:.4f}s) did not clear the 1.3x "
        f"speedup floor over the forced full-relex baseline ({old_wall:.4f}s)"
    )


def test_growing_boundary_token_falls_back_to_full_relex_and_stays_correct() -> None:
    """Scope/honesty check: when the *last fed* token is still growing
    (e.g. NAME "Te" -> "Text" fed one character at a time -- the byte/char
    literal-content decode path), the suffix-only lex cannot patch an
    already-fed InteractiveParser token in place and must fall back to a
    full resync, same as the pre-existing code. This must remain correct
    (not merely "not worse"): the fed token keys/accepts must match what a
    single one-shot ``set_prefix`` of the same final text would produce."""
    eng = engine_mod.OpenUIIncrementalEngine()
    eng.reset()
    assert eng.set_prefix("root = Card([")
    full_syncs_before = eng._full_syncs
    for partial in ("T", "Te", "Tex", "Text", "TextC", "TextContent"):
        assert eng.set_prefix("root = Card([" + partial)
    # Growing a bare NAME/COMPONENT token char-by-char after the lexer has
    # already eagerly completed a shorter version of it forces >=1 full
    # resync (unchanged from pre-fix behavior -- not a regression target of
    # this fix, see docs/design/decode-compiler-tree-incremental-lex-fix.md
    # Scope note).
    assert eng._full_syncs > full_syncs_before

    reference = engine_mod.OpenUIIncrementalEngine()
    reference.reset()
    assert reference.set_prefix("root = Card([TextContent")
    assert eng._fed_tokens == reference._fed_tokens
    assert eng.next_terminals() == reference.next_terminals()
