"""Regression tests for the suffix-only relex in ``probe_chunk``
(docs/design/decode-compiler-tree-probe-chunk-suffix-lex-fix.md).

``probe_chunk`` used to re-lex the *entire* ``self._prefix + chunk`` from
character 0 on every call -- the same O(prefix length) per call / O(n^2)
total cost across a generation of length n that PR #1174 fixed for
``_incremental_sync``, left explicitly out of scope there ("``probe_chunk``
(``engine.py:207``) still fully re-lexes on every call -- out of scope for
this PR"). This fix reuses the exact same ``self._fed_last_token_start``
anchor and maximal-munch safety argument via ``_probe_delta_from_anchor``,
without mutating engine state (``probe_chunk`` must stay read-only).

These tests prove: (a) strictly fewer characters are re-lexed than the
pre-existing full-prefix-relex baseline on a realistic speculative-probe
drive, (b) probe results -- the actual decode-facing return value -- are
identical to that baseline at every single probe of an incremental drive
(I3/I6: a caching/perf change must never become a legality change), (c) a
conservative wall-time speedup floor, and (d) the growing-boundary-token /
gluing fallback still returns the exact pre-fix result.
"""

from __future__ import annotations

import time

from slm_training.dsl.grammar.fastpath import engine as engine_mod

# Longer, multi-statement, incrementally-growing valid OpenUI DSL program --
# reused verbatim from test_grammar_fastpath_incremental_lex.py's LONG_PROGRAM
# so the two suites drive the exact same realistic text.
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


def _disabled_anchor(self, chunk: str) -> None:
    """Reproduces the exact pre-fix ``probe_chunk`` body: always re-lex the
    whole ``self._prefix + chunk`` via ``_lex_tokens``, never the
    suffix-only anchor path. Used as a self-contained "before" oracle (no
    ``git stash`` needed inside a test) for I3/I6 and char-count comparisons
    against the "after" (real, current) ``probe_chunk``."""
    return None


class _CharCountingEngine(engine_mod.OpenUIIncrementalEngine):
    """Counts characters passed into ``_lex_tokens`` -- a direct proxy for
    lex work performed, independent of wall-time noise."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.chars_lexed = 0

    def _lex_tokens(self, prefix):
        self.chars_lexed += len(prefix)
        return super()._lex_tokens(prefix)


def _drive_probe_and_commit(eng: engine_mod.OpenUIIncrementalEngine, program: str) -> list:
    """Realistic speculative-probe drive matching ``compiler_draft.py``'s
    usage: at each step, probe the next character against the currently
    synced prefix (without committing), record the probe result, then
    commit it via ``set_prefix`` before probing the next one -- exactly the
    probe-then-advance pattern ``build_completion_forest`` uses."""
    eng.reset()
    results = []
    for i in range(len(program)):
        results.append(eng.probe_chunk(program[i]))
        assert eng.set_prefix(program[: i + 1])
    return results


def test_probe_chunk_relexes_fewer_characters_than_full_relex_baseline() -> None:
    """Regression guard: probing a long, incrementally-growing valid prefix
    one character at a time must re-lex strictly fewer total characters
    than the pre-existing (forced) full-prefix-relex baseline -- proof that
    the suffix-anchor path is actually taken on the common path, not just
    present in the code."""
    program = LONG_PROGRAM

    new_eng = _CharCountingEngine()
    _drive_probe_and_commit(new_eng, program)

    old_eng = _CharCountingEngine()
    orig = engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor
    engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor = _disabled_anchor
    try:
        _drive_probe_and_commit(old_eng, program)
    finally:
        engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor = orig

    assert new_eng.chars_lexed < old_eng.chars_lexed, (
        f"suffix-anchor path did not reduce chars_lexed: "
        f"new={new_eng.chars_lexed} old={old_eng.chars_lexed}"
    )
    # Conservative floor, kept well under the reduction typically measured
    # for the analogous _incremental_sync fix (~2.4-3x), to stay robust on
    # slower/loaded CI hosts and unrelated future grammar edits.
    assert new_eng.chars_lexed * 1.5 < old_eng.chars_lexed, (
        f"chars_lexed reduction below the 1.5x floor: "
        f"new={new_eng.chars_lexed} old={old_eng.chars_lexed}"
    )


def test_probe_chunk_matches_forced_full_relex_at_every_step() -> None:
    """I3/I6: probe_chunk's return value must be byte-identical, at *every*
    step of an incremental probe-then-advance drive (not just the final
    one), between the suffix-anchor path and the pre-existing
    full-prefix-relex baseline. Exercises both the fast anchor path
    (punctuation/newline/new-token boundaries) and the still-full-relex
    fallback path (a token still growing mid-word, e.g. "Te" -> "Text"),
    since the drive is character-by-character."""
    program = LONG_PROGRAM

    new_eng = engine_mod.OpenUIIncrementalEngine()
    new_trace = _drive_probe_and_commit(new_eng, program)

    old_eng = engine_mod.OpenUIIncrementalEngine()
    orig = engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor
    engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor = _disabled_anchor
    try:
        old_trace = _drive_probe_and_commit(old_eng, program)
    finally:
        engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor = orig

    assert len(new_trace) == len(old_trace) == len(program)
    mismatches = [
        i for i, (a, b) in enumerate(zip(new_trace, old_trace, strict=True)) if a != b
    ]
    assert not mismatches, (
        f"probe_chunk diverged at steps {mismatches[:5]} "
        f"(showing up to 5 of {len(mismatches)})"
    )


def test_probe_chunk_speedup_floor_on_long_growing_prefix() -> None:
    """Wall-time benchmark: probing the long program char-by-character
    through the suffix-anchor path must be meaningfully faster than the
    forced full-prefix-relex baseline on the same process. Threshold (1.2x)
    is kept well under what the analogous _incremental_sync fix measured
    (~2.4-2.97x), to stay robust on a slower/loaded CI host -- probe_chunk's
    per-call overhead (InteractiveParser.copy()) is larger relative to the
    lex-work savings than _incremental_sync's, so the margin is tighter."""
    program = LONG_PROGRAM

    new_eng = engine_mod.OpenUIIncrementalEngine()
    t0 = time.perf_counter()
    _drive_probe_and_commit(new_eng, program)
    new_wall = time.perf_counter() - t0

    old_eng = engine_mod.OpenUIIncrementalEngine()
    orig = engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor
    engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor = _disabled_anchor
    try:
        t0 = time.perf_counter()
        _drive_probe_and_commit(old_eng, program)
        old_wall = time.perf_counter() - t0
    finally:
        engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor = orig

    assert new_wall < old_wall / 1.2, (
        f"suffix-anchor wall time ({new_wall:.4f}s) did not clear the 1.2x "
        f"speedup floor over the forced full-relex baseline ({old_wall:.4f}s)"
    )


def test_growing_boundary_token_probe_falls_back_and_stays_correct() -> None:
    """Scope/honesty check: when the *last fed* token is still growing
    (e.g. NAME "Te" -> "Text" fed one character at a time), the suffix-only
    anchor cannot verify a still-live boundary token and must fall back to
    the pre-existing full-prefix lex, same as before this fix. Probing must
    stay correct (not merely "not worse") through that fallback, matching a
    reference engine driven with the anchor disabled entirely."""
    eng = engine_mod.OpenUIIncrementalEngine()
    eng.reset()
    assert eng.set_prefix("root = Card([")

    reference = engine_mod.OpenUIIncrementalEngine()
    reference.reset()
    orig = engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor
    engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor = _disabled_anchor
    try:
        assert reference.set_prefix("root = Card([")
        for partial, next_char in (
            ("", "T"),
            ("T", "e"),
            ("Te", "x"),
            ("Tex", "t"),
            ("Text", "C"),
            ("TextC", "ontent"),
        ):
            got = eng.probe_chunk(next_char)
            want = reference.probe_chunk(next_char)
            assert got == want, (
                f"probe mismatch after {partial!r} + {next_char!r}: "
                f"anchor-enabled={got} anchor-disabled(reference)={want}"
            )
            assert eng.set_prefix("root = Card([" + partial + next_char)
            assert reference.set_prefix("root = Card([" + partial + next_char)
    finally:
        engine_mod.OpenUIIncrementalEngine._probe_delta_from_anchor = orig

    assert eng._fed_tokens == reference._fed_tokens
    assert eng.next_terminals() == reference.next_terminals()


def test_probe_chunk_anchor_unused_before_anything_fed() -> None:
    """Before any token has been fed (fresh ``reset()``), there's no anchor
    to lex a safe suffix from -- ``_probe_delta_from_anchor`` must return
    ``None`` so ``probe_chunk`` takes the pre-existing full-prefix path,
    same as it always did."""
    eng = engine_mod.OpenUIIncrementalEngine()
    eng.reset()
    assert eng._probe_delta_from_anchor("root") is None
    assert eng.probe_chunk("root") is True
